"""
Face Detection Service
----------------------
Dedicated, independent face detection service for detecting faces in video frames.

Capabilities:
  - Detects faces independently of the YOLO11 person detector.
  - Handles frames containing zero, one, or multiple faces.
  - Supports BGR (standard OpenCV format) or RGB input frames, with automatic conversion.
  - Returns clamped bounding boxes [x1, y1, x2, y2], confidence scores, and clean BGR crops.
  - Handles invalid or empty frames gracefully without exceptions.
  - Engine:
      1. Primary: MediaPipe Tasks FaceDetector using official 'weights/face/blaze_face_short_range.tflite'.
      2. Fallback: MediaPipe Pose Landmarker facial keypoints using 'weights/pose/pose_landmarker_lite.task'.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class FaceDetectionService:
    """
    Independent face detector providing bounding boxes, confidences, and face crops.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        pose_model_path: Optional[str | Path] = None,
        min_confidence: float = 0.45,
    ):
        """
        Initializes the face detection service.

        Parameters:
            model_path: Path to dedicated face detection model asset ('blaze_face_short_range.tflite' or '.task').
            pose_model_path: Optional path to pose landmarker task asset for fallback landmark-guided face detection.
            min_confidence: Minimum detection confidence threshold (default 0.45).
        """
        root_dir = Path(__file__).resolve().parent.parent.parent.parent

        self.min_confidence = min_confidence

        # Candidates for dedicated face model asset
        if model_path:
            face_candidates = [Path(model_path)]
        else:
            face_candidates = [
                root_dir / "weights" / "face" / "blaze_face_short_range.tflite",
                root_dir / "weights" / "face" / "blaze_face_short_range.task",
            ]

        self.face_model_path = None
        for cand in face_candidates:
            if cand.exists() and cand.stat().st_size > 0:
                self.face_model_path = cand
                break

        self.pose_model_path = Path(pose_model_path or root_dir / "weights" / "pose" / "pose_landmarker_lite.task")

        self.engine: Optional[str] = None
        self.detector: Any = None

        # 1. Initialize dedicated MediaPipe FaceDetector if model file exists
        if self.face_model_path is not None:
            try:
                base_options = python.BaseOptions(model_asset_path=str(self.face_model_path))
                options = vision.FaceDetectorOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    min_detection_confidence=self.min_confidence,
                )
                self.detector = vision.FaceDetector.create_from_options(options)
                self.engine = "mediapipe_face"
                print(f"[FaceService] Initialized dedicated MediaPipe FaceDetector ({self.face_model_path.name})")
            except Exception as e:
                print(f"[FaceService] Warning: Could not initialize dedicated FaceDetector: {e}")

        # 2. Fallback to MediaPipe Pose Landmarker head/face keypoint extractor
        if self.engine is None and self.pose_model_path.exists() and self.pose_model_path.stat().st_size > 0:
            try:
                base_options = python.BaseOptions(model_asset_path=str(self.pose_model_path))
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_poses=4,
                    min_pose_detection_confidence=self.min_confidence,
                    min_pose_presence_confidence=self.min_confidence,
                )
                self.detector = vision.PoseLandmarker.create_from_options(options)
                self.engine = "pose_landmarks"
                print(f"[FaceService] Initialized MediaPipe Pose Facial Keypoint Extractor ({self.pose_model_path.name})")
            except Exception as e:
                print(f"[FaceService] Warning: Could not initialize Pose fallback: {e}")

        if self.engine is None:
            raise FileNotFoundError(
                f"[FaceService] No compatible face detection asset available.\n"
                f"Please ensure 'blaze_face_short_range.tflite' exists under 'weights/face/'."
            )

    def detect_faces(
        self, frame: np.ndarray, input_format: str = "BGR"
    ) -> List[Dict[str, Any]]:
        """
        Detects faces in an input video frame.

        Parameters:
            frame: Input numpy array (uint8). Typically BGR as returned by cv2.imread() or cv2.VideoCapture().
            input_format: Format of input frame, either 'BGR' or 'RGB' (default: 'BGR').

        Returns:
            List of detected face dictionaries:
              [
                {
                  "bbox": [x1, y1, x2, y2],       # Integer coordinates clamped to frame boundaries
                  "confidence": float,            # Detection confidence score (0.0 - 1.0)
                  "crop": np.ndarray,             # Clean BGR image crop of the face
                  "width": int,                   # Bounding box width
                  "height": int,                  # Bounding box height
                },
                ...
              ]
        """
        # 1. Guard against empty, None, or degenerate frames
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or len(frame.shape) < 2:
            return []

        h, w = frame.shape[:2]
        if h < 5 or w < 5:
            return []

        # 2. Convert color space for MediaPipe (MediaPipe expects RGB)
        if input_format.upper() == "BGR":
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bgr_frame = frame
            else:
                return []
        elif input_format.upper() == "RGB":
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = frame
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                return []
        else:
            raise ValueError(f"Unsupported input_format '{input_format}'. Use 'BGR' or 'RGB'.")

        try:
            mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        except Exception:
            return []

        detections: List[Dict[str, Any]] = []

        # 3. Detection via dedicated FaceDetector
        if self.engine == "mediapipe_face":
            try:
                res = self.detector.detect(mp_image)
                if res and res.detections:
                    for det in res.detections:
                        conf = float(det.categories[0].score) if det.categories else 1.0
                        box = det.bounding_box
                        x1 = max(0, min(w - 1, int(box.origin_x)))
                        y1 = max(0, min(h - 1, int(box.origin_y)))
                        x2 = max(0, min(w - 1, int(box.origin_x + box.width)))
                        y2 = max(0, min(h - 1, int(box.origin_y + box.height)))

                        if (x2 - x1) >= 10 and (y2 - y1) >= 10:
                            face_crop = bgr_frame[y1:y2, x1:x2].copy()
                            detections.append({
                                "bbox": [x1, y1, x2, y2],
                                "confidence": round(conf, 4),
                                "crop": face_crop,
                                "width": x2 - x1,
                                "height": y2 - y1,
                            })
            except Exception as e:
                print(f"[FaceService] Detection error in mediapipe_face: {e}")
                return []

        # 4. Fallback Detection via Pose Landmarker head/face keypoints (landmarks 0-10)
        elif self.engine == "pose_landmarks":
            try:
                res = self.detector.detect(mp_image)
                if res and res.pose_landmarks:
                    for pose in res.pose_landmarks:
                        pts = [[lm.x * w, lm.y * h] for lm in pose[:11]]
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]

                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)

                        box_w = max_x - min_x
                        box_h = max_y - min_y
                        margin_x = box_w * 0.35
                        margin_y_top = box_h * 0.65
                        margin_y_bot = box_h * 0.35

                        x1 = max(0, min(w - 1, int(min_x - margin_x)))
                        y1 = max(0, min(h - 1, int(min_y - margin_y_top)))
                        x2 = max(0, min(w - 1, int(max_x + margin_x)))
                        y2 = max(0, min(h - 1, int(max_y + margin_y_bot)))

                        if (x2 - x1) >= 10 and (y2 - y1) >= 10:
                            face_crop = bgr_frame[y1:y2, x1:x2].copy()
                            conf = float(np.mean([lm.visibility for lm in pose[:11] if hasattr(lm, "visibility")] or [0.90]))
                            detections.append({
                                "bbox": [x1, y1, x2, y2],
                                "confidence": round(conf, 4),
                                "crop": face_crop,
                                "width": x2 - x1,
                                "height": y2 - y1,
                            })
            except Exception as e:
                print(f"[FaceService] Detection error in pose_landmarks: {e}")
                return []

        return detections

    def draw_faces(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Draws bounding boxes and confidence scores on a copy of the frame.
        """
        if frame is None or frame.size == 0:
            return frame

        canvas = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det.get("confidence", 0.0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"Face {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                canvas, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )
        return canvas
