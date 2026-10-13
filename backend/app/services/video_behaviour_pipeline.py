"""
Unified Video Behaviour Pipeline Service
-----------------------------------------
Integrates all multi-branch backend machine learning and computer vision services
into a unified, frame-by-frame video processing engine.

Branches executed per frame:
  1. YOLO11: Person detection & persistent tracking (classes=[0])
  2. MediaPipe Pose + BiLSTM: Body behaviour inference per track ID (165 features, 30-frame sequence)
  3. MediaPipe FaceDetector: Independent facial detection
  4. EfficientNetB0: Facial emotion prediction per face crop (7 classes)
  5. Geometric Association: Spatial 1-to-1 matching linking faces to person tracks
  6. Visual Multi-Branch Annotation: Person boxes, tracking badges, independent telemetry, HUD

Preserves independent model confidences and predictions (no composite risk scoring).
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import importlib.util

import cv2
import numpy as np
from ultralytics import YOLO

# Ensure UTF-8 output encoding on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.app.services.face_service import FaceDetectionService
from backend.app.services.emotion_service import EmotionInferenceService
from backend.app.services.association_service import FacePersonAssociationService


def get_behaviour_tracker_class():
    """Dynamically loads BehaviourTracker from 'behaviour_tracker (1).py'."""
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    tracker_file = root_dir / "behaviour_tracker (1).py"
    if not tracker_file.exists():
        tracker_file = root_dir.parent / "behaviour_tracker (1).py"
    if not tracker_file.exists():
        raise FileNotFoundError(f"Cannot find 'behaviour_tracker (1).py' at: {tracker_file}")

    spec = importlib.util.spec_from_file_location("bt_original_module", str(tracker_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for {tracker_file}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BehaviourTracker


class WebVideoWriter:
    """
    Web-compatible H.264 video writer utilizing PyAV (libx264, yuv420p).
    Guarantees that generated MP4 files are natively playable in modern web browsers (<video> element).
    Gracefully falls back to OpenCV VideoWriter if PyAV cannot initialize.
    """

    def __init__(self, output_path: Union[str, Path], fps: float, width: int, height: int):
        self.output_path = str(output_path)
        self.fps = float(fps) if (fps and fps > 0) else 25.0
        # Ensure dimensions are even numbers (requirement for libx264 yuv420p)
        self.width = int(width) - (int(width) % 2)
        self.height = int(height) - (int(height) % 2)
        self.use_av = False
        self.container = None
        self.stream = None
        self.cv2_writer = None

        try:
            import av
            self.container = av.open(self.output_path, mode="w")
            rate = int(round(self.fps)) if self.fps > 0 else 25
            self.stream = self.container.add_stream("libx264", rate=rate)
            self.stream.width = self.width
            self.stream.height = self.height
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"crf": "23", "preset": "fast"}
            self.av_module = av
            self.use_av = True
        except Exception as e:
            print(f"[WebVideoWriter] PyAV initialization failed ({e}), falling back to OpenCV VideoWriter.")
            self.use_av = False
            for codec in ["avc1", "H264", "mp4v"]:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
                if writer.isOpened():
                    self.cv2_writer = writer
                    break

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None or frame_bgr.size == 0:
            return

        h, w = frame_bgr.shape[:2]
        if w != self.width or h != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))

        if self.use_av and self.stream and self.container:
            try:
                av_frame = self.av_module.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
                for packet in self.stream.encode(av_frame):
                    self.container.mux(packet)
            except Exception as e:
                print(f"[WebVideoWriter] Frame encoding error: {e}")
        elif self.cv2_writer is not None:
            self.cv2_writer.write(frame_bgr)

    def release(self) -> None:
        if self.use_av and self.stream and self.container:
            try:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            except Exception:
                pass
            try:
                self.container.close()
            except Exception:
                pass
            self.container = None
            self.stream = None
        elif self.cv2_writer is not None:
            self.cv2_writer.release()
            self.cv2_writer = None


class VideoBehaviourPipeline:
    """
    Coordinates YOLO11, BehaviourTracker, FaceDetectionService, EmotionInferenceService,
    and FacePersonAssociationService into a complete video processing engine.
    """

    COLOR_MAP = {
        "collecting": (0, 200, 255),    # Amber / Yellow (Gathering 30 sequence observations)
        "lost": (0, 180, 240),          # Amber / Yellow (Unavailable / Search state)
        "untracked": (0, 180, 240),     # Amber / Yellow (Unassigned detection)
        0: (0, 220, 0),                 # Green (Low Risk: NonFight / Normal)
        1: (0, 0, 255),                 # Red (High Risk: Fight alert)
        2: (0, 0, 255),                 # Red (High Risk: Fall alert)
    }

    FACE_COLOR = (255, 200, 0)          # Aqua/Cyan for detected faces

    def __init__(
        self,
        behaviour_model_path: Optional[Union[str, Path]] = None,
        pose_model_path: Optional[Union[str, Path]] = None,
        yolo_model_path: Optional[Union[str, Path]] = None,
        face_model_path: Optional[Union[str, Path]] = None,
        emotion_model_path: Optional[Union[str, Path]] = None,
        sequence_length: int = 30,
        behaviour_conf_thresh: float = 0.55,
        smooth_window: int = 7,
        yolo_conf: float = 0.35,
        yolo_iou: float = 0.50,
        face_conf: float = 0.45,
        upper_body_ratio: float = 0.40,
        min_containment: float = 0.35,
        ambiguity_margin: float = 0.15,
        max_missed_frames: int = 30,
    ):
        root_dir = Path(__file__).resolve().parent.parent.parent.parent

        self.behaviour_model_path = Path(
            behaviour_model_path or root_dir / "weights" / "behaviour" / "behaviour_model_rich_165_v2_best.keras"
        )
        self.pose_model_path = Path(
            pose_model_path or root_dir / "weights" / "pose" / "pose_landmarker_lite.task"
        )
        self.yolo_model_path = Path(
            yolo_model_path or root_dir / "weights" / "yolo" / "yolo11n.pt"
        )
        self.face_model_path = Path(
            face_model_path or root_dir / "weights" / "face" / "blaze_face_short_range.tflite"
        )
        self.emotion_model_path = Path(
            emotion_model_path or root_dir / "weights" / "emotion" / "rafdb_emotion_best.keras"
        )

        self.sequence_length = sequence_length
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou
        self.max_missed_frames = max_missed_frames

        # Track management
        self.last_seen_frame: Dict[int, int] = {}

        # 1. Initialize YOLO11 Person Tracker
        print(f"[VideoBehaviourPipeline] Loading YOLO11 from {self.yolo_model_path.name}...")
        self.yolo = YOLO(str(self.yolo_model_path))

        # 2. Initialize BehaviourTracker (MediaPipe Pose + BiLSTM)
        print(f"[VideoBehaviourPipeline] Initializing BehaviourTracker...")
        BTClass = get_behaviour_tracker_class()
        self.behaviour_tracker = BTClass(
            model_path=str(self.behaviour_model_path),
            pose_model_path=str(self.pose_model_path),
            sequence_length=sequence_length,
            confidence_threshold=behaviour_conf_thresh,
            smooth_window=smooth_window,
            max_fail_frames=20,
        )

        # 3. Initialize Face Detection Service
        print(f"[VideoBehaviourPipeline] Initializing FaceDetectionService...")
        self.face_service = FaceDetectionService(
            model_path=self.face_model_path,
            min_confidence=face_conf,
        )

        # 4. Initialize Emotion Inference Service
        print(f"[VideoBehaviourPipeline] Initializing EmotionInferenceService...")
        self.emotion_service = EmotionInferenceService(
            model_path=self.emotion_model_path
        )

        # 5. Initialize Face-Person Association Service
        self.association_service = FacePersonAssociationService(
            upper_body_ratio=upper_body_ratio,
            min_containment=min_containment,
            min_match_score=0.40,
            ambiguity_margin=ambiguity_margin,
        )

        print("[VideoBehaviourPipeline] All 5 multi-branch pipeline engines initialized successfully.")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        total_frames: int = 0,
        fps: float = 25.0,
        max_frames: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Executes all branches on a single BGR video frame:
        1. YOLO11 person detection and tracking.
        2. MediaPipe Pose feature extraction and BiLSTM behaviour inference.
        3. Independent Face detection.
        4. Emotion inference on detected faces.
        5. Geometric association between face and person tracks.
        6. Visual multi-branch frame annotation.

        Returns:
            Tuple of (annotated_frame, integrated_person_records, raw_face_detections)
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or len(frame.shape) < 2:
            return frame, [], []

        annotated = frame.copy()
        h, w = frame.shape[:2]

        # -------------------------------------------------------------
        # Branch 1: YOLO11 Person Tracking & Body Behaviour Inference
        # -------------------------------------------------------------
        person_tracks: List[Dict[str, Any]] = []

        try:
            results = self.yolo.track(
                frame,
                persist=True,
                classes=[0],            # Person filter only
                conf=self.yolo_conf,
                iou=self.yolo_iou,
                verbose=False,
            )
        except Exception as e:
            print(f"[VideoBehaviourPipeline] YOLO tracking warning on frame {frame_idx}: {e}")
            results = []

        active_tids_in_frame = set()

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes_obj = results[0].boxes
            boxes = boxes_obj.xyxy.cpu().numpy() if hasattr(boxes_obj, "xyxy") else []
            confs = boxes_obj.conf.cpu().numpy() if hasattr(boxes_obj, "conf") else []
            if boxes_obj.id is not None:
                track_ids = boxes_obj.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(boxes)

            for i, (box, det_conf) in enumerate(zip(boxes, confs)):
                raw_tid = track_ids[i] if i < len(track_ids) else None
                # If YOLO did not assign a track ID, assign temporary pseudo ID
                is_temporary_id = raw_tid is None
                track_id = int(raw_tid) if not is_temporary_id else -(i + 1)

                # Clamp bounding box coordinates safely
                x1 = max(0, min(w - 1, int(box[0])))
                y1 = max(0, min(h - 1, int(box[1])))
                x2 = max(0, min(w - 1, int(box[2])))
                y2 = max(0, min(h - 1, int(box[3])))

                # Validate bounding box dimensions
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue

                if not is_temporary_id:
                    active_tids_in_frame.add(track_id)
                    self.last_seen_frame[track_id] = frame_idx

                # Safe person crop extraction with 8% contextual padding for MediaPipe Pose
                pad_w = int((x2 - x1) * 0.08)
                pad_h = int((y2 - y1) * 0.08)
                crop_x1 = max(0, x1 - pad_w)
                crop_y1 = max(0, y1 - pad_h)
                crop_x2 = min(w, x2 + pad_w)
                crop_y2 = min(h, y2 + pad_h)
                person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                if not is_temporary_id:
                    try:
                        b_result = self.behaviour_tracker.update(track_id, person_crop)
                    except Exception as e:
                        print(f"[VideoBehaviourPipeline] BehaviourTracker warning (track {track_id}): {e}")
                        b_result = None
                else:
                    # Ephemeral unassigned detection: do NOT accumulate sequential buffer across unrelated frames
                    b_result = {
                        "status": "untracked",
                        "class_id": -1,
                        "class_name": "Untracked",
                        "risk_level": "—",
                        "confidence": 0.0,
                        "probabilities": {"NonFight": 0.0, "Fight": 0.0, "Fall": 0.0},
                        "collected": 0,
                    }

                # Normalize prediction status
                if b_result is not None:
                    if b_result.get("class_id", -1) >= 0 and "status" not in b_result:
                        b_result["status"] = "predicted"
                else:
                    b_result = {
                        "status": "lost" if not is_temporary_id else "untracked",
                        "class_id": -1,
                        "class_name": "Lost" if not is_temporary_id else "Untracked",
                        "risk_level": "—",
                        "confidence": 0.0,
                        "probabilities": {"NonFight": 0.0, "Fight": 0.0, "Fall": 0.0},
                        "collected": 0,
                    }

                person_tracks.append({
                    "track_id": track_id,
                    "is_temporary_id": is_temporary_id,
                    "bbox": [x1, y1, x2, y2],
                    "det_confidence": float(det_conf),
                    "behaviour": b_result,
                })

        # Track cleanup: reset state for tracks missing for > max_missed_frames
        if frame_idx > 0:
            expired_tids = [
                tid for tid, last_seen in self.last_seen_frame.items()
                if (frame_idx - last_seen) > self.max_missed_frames
            ]
            for tid in expired_tids:
                self.behaviour_tracker.reset_track(tid)
                del self.last_seen_frame[tid]

        # -------------------------------------------------------------
        # Branch 2: Face Detection & Emotion Inference
        # -------------------------------------------------------------
        try:
            face_detections = self.face_service.detect_faces(frame, input_format="BGR")
        except Exception as e:
            print(f"[VideoBehaviourPipeline] Face detection warning on frame {frame_idx}: {e}")
            face_detections = []

        for face in face_detections:
            face_crop = face.get("crop")
            try:
                face["emotion"] = self.emotion_service.predict_emotion(face_crop, input_format="BGR")
            except Exception as e:
                print(f"[VideoBehaviourPipeline] Emotion inference warning on frame {frame_idx}: {e}")
                face["emotion"] = {
                    "label": "Unavailable",
                    "class_id": -1,
                    "confidence": 0.0,
                    "probabilities": {},
                    "status": "unavailable",
                    "error": str(e),
                }

        # -------------------------------------------------------------
        # Branch 3: Face-to-Person ID Association
        # -------------------------------------------------------------
        try:
            integrated_records = self.association_service.associate_frame(
                person_tracks=person_tracks,
                face_detections=face_detections,
            )
        except Exception as e:
            print(f"[VideoBehaviourPipeline] Association warning on frame {frame_idx}: {e}")
            integrated_records = []
            for p in person_tracks:
                integrated_records.append({
                    "track_id": p["track_id"],
                    "person_bbox": p["bbox"],
                    "det_confidence": p["det_confidence"],
                    "behaviour": p["behaviour"],
                    "emotion": {
                        "status": "unavailable",
                        "label": "Unavailable",
                        "confidence": 0.0,
                        "class_id": -1,
                        "probabilities": {},
                        "face_bbox": None,
                        "match_score": 0.0,
                        "reason": f"association_error: {e}",
                    },
                })

        # -------------------------------------------------------------
        # Visual Annotation: Person Bounding Boxes & Independent Telemetry
        # -------------------------------------------------------------

        font = cv2.FONT_HERSHEY_SIMPLEX
        min_dim = min(w, h)

        # Adaptive typography and styling based on frame dimensions (e.g. 240p vs 720p vs 1080p)
        if min_dim < 360:
            hud_scale = 0.32
            hud_thick = 1
            box_font_scale = 0.29
            box_font_thick = 1
            box_thickness = 2
            face_font_scale = 0.27
            face_font_thick = 1
        elif min_dim < 600:
            hud_scale = 0.44
            hud_thick = 1
            box_font_scale = 0.38
            box_font_thick = 1
            box_thickness = 2
            face_font_scale = 0.34
            face_font_thick = 1
        else:
            hud_scale = 0.56
            hud_thick = 2
            box_font_scale = 0.48
            box_font_thick = 2
            box_thickness = 3
            face_font_scale = 0.40
            face_font_thick = 1

        # 1. Render HUD Overview Banner at top-left
        # Distinguish current frame from total frames selected for analysis
        target_frames = max_frames if (max_frames and max_frames < total_frames) else total_frames
        if max_frames and max_frames < total_frames:
            hud_text = f"Frame: {frame_idx}/{max_frames} (Sample of {total_frames}) | Students: {len(integrated_records)} | Faces: {len(face_detections)}"
        else:
            hud_text = f"Frame: {frame_idx}/{total_frames} (Full Video) | Students: {len(integrated_records)} | Faces: {len(face_detections)}"

        (htw, hth), hbase = cv2.getTextSize(hud_text, font, hud_scale, hud_thick)
        # Dynamically scale HUD text if it would exceed available frame width
        if htw > w - 16:
            hud_scale = max(0.24, hud_scale * ((w - 16) / float(htw)))
            (htw, hth), hbase = cv2.getTextSize(hud_text, font, hud_scale, hud_thick)

        hud_x1, hud_y1 = 6, 6
        hud_x2 = min(w - 2, hud_x1 + htw + 8)
        hud_y2 = min(h - 2, hud_y1 + hth + hbase + 6)
        cv2.rectangle(annotated, (hud_x1, hud_y1), (hud_x2, hud_y2), (0, 0, 0), -1)
        cv2.putText(
            annotated, hud_text, (hud_x1 + 4, hud_y1 + hth + 2),
            font, hud_scale, (255, 255, 255), hud_thick, cv2.LINE_AA
        )

        # Collision tracker to prevent overlapping labels and HUD collisions
        placed_rects: List[Tuple[int, int, int, int]] = [(hud_x1, hud_y1, hud_x2, hud_y2)]

        def rects_overlap(r1: Tuple[int, int, int, int], r2: Tuple[int, int, int, int]) -> bool:
            return not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3])

        def find_non_overlapping_slot(
            primary_candidates: List[Tuple[int, int]],
            pill_w: int,
            pill_h: int,
            pref_x: int,
            pref_y: int,
        ) -> Tuple[int, int, int, int]:
            # 1. Try ordered primary candidates
            for cx, cy in primary_candidates:
                r = (cx, cy, cx + pill_w, cy + pill_h)
                if 2 <= r[0] and 2 <= r[1] and r[2] <= w - 2 and r[3] <= h - 2:
                    if not any(rects_overlap(r, p) for p in placed_rects):
                        return r
            # 2. Spiral outward search around preferred anchor (dx, dy)
            for radius in range(4, max(w, h), 6):
                for dx in (-radius, 0, radius):
                    for dy in (-radius, 0, radius):
                        cx = max(2, min(w - pill_w - 2, pref_x + dx))
                        cy = max(2, min(h - pill_h - 2, pref_y + dy))
                        r = (cx, cy, cx + pill_w, cy + pill_h)
                        if not any(rects_overlap(r, p) for p in placed_rects):
                            return r
            # 3. Fallback within bounds
            fb_x = max(2, min(w - pill_w - 2, pref_x))
            fb_y = max(2, min(h - pill_h - 2, pref_y))
            return (fb_x, fb_y, fb_x + pill_w, fb_y + pill_h)

        # Sort render order by bounding box area (descending) so prominent foreground people
        # claim their primary label slots first without background slivers displacing them.
        draw_records = sorted(
            integrated_records,
            key=lambda r: (r["person_bbox"][2] - r["person_bbox"][0]) * (r["person_bbox"][3] - r["person_bbox"][1]),
            reverse=True,
        )

        # 2. Render each detected person's bounding box and independent telemetry
        for record in draw_records:
            tid = record["track_id"]
            raw_px1, raw_py1, raw_px2, raw_py2 = record["person_bbox"]
            behaviour = record["behaviour"]
            emotion = record["emotion"]

            # Clamp bounding box strictly inside frame boundaries
            px1 = max(0, min(w - 1, int(round(raw_px1))))
            py1 = max(0, min(h - 1, int(round(raw_py1))))
            px2 = max(px1 + 1, min(w - 1, int(round(raw_px2))))
            py2 = max(py1 + 1, min(h - 1, int(round(raw_py2))))

            b_status = behaviour.get("status", "lost")
            risk_level = behaviour.get("risk_level", "—")
            cid = behaviour.get("class_id", -1)
            cname = behaviour.get("class_name", "")
            b_conf = float(behaviour.get("confidence", 0.0))

            # Select bounding box color & label according to requirements:
            # - RED for High risk (Fight / Fall)
            # - GREEN for Low risk (Normal / NonFight)
            # - AMBER/YELLOW for Collecting / Unavailable / Gathering data
            if b_status == "predicted":
                if risk_level == "High" or cid in [1, 2]:
                    # High Risk: Fight or Fall -> RED
                    color = (0, 0, 255)
                    text_color = (255, 255, 255)
                    beh_display = "FIGHT" if (cid == 1 or ("fight" in cname.lower() and "non" not in cname.lower())) else "FALL" if (cid == 2 or "fall" in cname.lower()) else cname.upper()
                    b_label = f"ID {tid} | {beh_display} | Conf: {b_conf:.2f}"
                else:
                    # Low Risk: NonFight / Normal -> GREEN
                    color = (0, 220, 0)
                    text_color = (255, 255, 255)
                    beh_display = "NORMAL" if ("non" in cname.lower() or "norm" in cname.lower()) else cname.upper()
                    b_label = f"ID {tid} | {beh_display} | Conf: {b_conf:.2f}"
            elif b_status == "collecting":
                collected = behaviour.get("collected", 0)
                if collected > 0:
                    color = (0, 200, 255)  # Amber / Yellow
                    text_color = (0, 0, 0)
                    b_label = f"ID {tid} | COLLECTING {collected}/{self.sequence_length}"
                else:
                    color = (0, 190, 240)  # Amber
                    text_color = (0, 0, 0)
                    b_label = f"ID {tid} | POSE UNAVAILABLE"
            elif b_status == "untracked":
                color = (0, 190, 240)  # Amber
                text_color = (0, 0, 0)
                det_conf = float(record.get("det_confidence", 0.0))
                b_label = f"ID {tid} | DETECTED | Conf: {det_conf:.2f}"
            else:
                color = (0, 190, 240)  # Amber
                text_color = (0, 0, 0)
                b_label = f"ID {tid} | POSE UNAVAILABLE"

            # Draw actual YOLO11 person bounding box
            cv2.rectangle(annotated, (px1, py1), (px2, py2), color, box_thickness)

            # Compute label size
            (tw, th), baseline = cv2.getTextSize(b_label, font, box_font_scale, box_font_thick)
            pill_w = tw + 8
            pill_h = th + baseline + 6

            # Determine best label position without colliding with HUD or other labels
            candidate_positions = [
                # 1. Above bounding box
                (max(2, min(px1, w - pill_w - 2)), py1 - pill_h - 2),
                # 2. Inside top of bounding box (left-aligned)
                (max(2, min(px1 + 2, w - pill_w - 2)), py1 + 2),
                # 3. Inside top of bounding box (right-aligned)
                (max(2, min(px2 - pill_w - 2, w - pill_w - 2)), py1 + 2),
                # 4. Inside bounding box (stacked lower)
                (max(2, min(px1 + 2, w - pill_w - 2)), min(py2 - pill_h - 2, py1 + pill_h + 4)),
                # 5. Below bounding box (if room)
                (max(2, min(px1, w - pill_w - 2)), py2 + 2),
            ]

            chosen_pill = find_non_overlapping_slot(
                primary_candidates=candidate_positions,
                pill_w=pill_w,
                pill_h=pill_h,
                pref_x=px1,
                pref_y=max(2, py1 - pill_h - 2),
            )

            pill_x1, pill_y1, pill_x2, pill_y2 = chosen_pill
            placed_rects.append(chosen_pill)

            # Draw background pill with 1px border for high contrast
            cv2.rectangle(annotated, (pill_x1, pill_y1), (pill_x2, pill_y2), color, -1)
            cv2.rectangle(annotated, (pill_x1, pill_y1), (pill_x2, pill_y2), (0, 0, 0), 1)
            cv2.putText(
                annotated, b_label, (pill_x1 + 4, pill_y1 + th + 2),
                font, box_font_scale, text_color, box_font_thick, cv2.LINE_AA
            )

            # 3. Draw facial emotion telemetry (strictly separated, never affects body box color)
            if emotion.get("status") == "available" and emotion.get("face_bbox"):
                raw_fx1, raw_fy1, raw_fx2, raw_fy2 = emotion["face_bbox"]
                fx1 = max(0, min(w - 1, int(round(raw_fx1))))
                fy1 = max(0, min(h - 1, int(round(raw_fy1))))
                fx2 = max(fx1 + 1, min(w - 1, int(round(raw_fx2))))
                fy2 = max(fy1 + 1, min(h - 1, int(round(raw_fy2))))

                em_conf = float(emotion.get("confidence", 0.0))
                em_label = f"Face {tid} | {emotion.get('label', '').upper()} | Conf: {em_conf:.2f}"
                face_color = self.FACE_COLOR

                # Draw face rectangle
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), face_color, 2)

                (ftw, fth), fbase = cv2.getTextSize(em_label, font, face_font_scale, face_font_thick)
                f_pill_w = ftw + 6
                f_pill_h = fth + fbase + 4

                face_candidates = [
                    (max(2, min(fx1, w - f_pill_w - 2)), fy1 - f_pill_h - 2),
                    (max(2, min(fx1 + 2, w - f_pill_w - 2)), fy1 + 2),
                    (max(2, min(fx1, w - f_pill_w - 2)), fy2 + 2),
                ]
                chosen_f_pill = find_non_overlapping_slot(
                    primary_candidates=face_candidates,
                    pill_w=f_pill_w,
                    pill_h=f_pill_h,
                    pref_x=fx1,
                    pref_y=fy1,
                )

                f_x1, f_y1, f_x2, f_y2 = chosen_f_pill
                placed_rects.append(chosen_f_pill)

                cv2.rectangle(annotated, (f_x1, f_y1), (f_x2, f_y2), face_color, -1)
                cv2.rectangle(annotated, (f_x1, f_y1), (f_x2, f_y2), (0, 0, 0), 1)
                cv2.putText(
                    annotated, em_label, (f_x1 + 3, f_y1 + fth + 1),
                    font, face_font_scale, (0, 0, 0), face_font_thick, cv2.LINE_AA
                )
            else:
                # Emotion Unavailable: show clean dark pill near bottom of person box
                unavail_label = "Emotion: Unavailable"
                (utw, uth), ubase = cv2.getTextSize(unavail_label, font, face_font_scale, face_font_thick)
                u_pill_w = utw + 6
                u_pill_h = uth + ubase + 4

                u_candidates = [
                    (max(2, min(px1 + 2, w - u_pill_w - 2)), py2 - u_pill_h - 2),
                    (max(2, min(px1 + 2, w - u_pill_w - 2)), py2 + 2),
                    (max(2, min(px2 - u_pill_w - 2, w - u_pill_w - 2)), py2 - u_pill_h - 2),
                    (max(2, min(px1 + 2, w - u_pill_w - 2)), pill_y2 + 2),
                ]
                chosen_u_pill = find_non_overlapping_slot(
                    primary_candidates=u_candidates,
                    pill_w=u_pill_w,
                    pill_h=u_pill_h,
                    pref_x=px1,
                    pref_y=max(2, py2 - u_pill_h - 2),
                )

                u_x1, u_y1, u_x2, u_y2 = chosen_u_pill
                placed_rects.append(chosen_u_pill)

                cv2.rectangle(annotated, (u_x1, u_y1), (u_x2, u_y2), (35, 35, 35), -1)
                cv2.rectangle(annotated, (u_x1, u_y1), (u_x2, u_y2), (70, 70, 70), 1)
                cv2.putText(
                    annotated, unavail_label, (u_x1 + 3, u_y1 + uth + 1),
                    font, face_font_scale, (220, 220, 220), face_font_thick, cv2.LINE_AA
                )

        return annotated, integrated_records, face_detections

    def reset_all(self):
        """Resets all tracker buffers and clears YOLO tracking state."""
        self.behaviour_tracker.reset_all()
        self.last_seen_frame.clear()
        if hasattr(self.yolo, "predictor") and self.yolo.predictor is not None:
            if hasattr(self.yolo.predictor, "trackers"):
                try:
                    delattr(self.yolo.predictor, "trackers")
                except Exception:
                    pass

    def process_video(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
        progress_callback = None,
        incident_detector = None,
        video_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Processes an input video file through the multi-branch pipeline,
        producing annotated frames and returning complete telemetry history.
        Optionally streams frames to incident_detector for automated incident capture.
        """
        self.reset_all()

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(raw_fps) if (raw_fps is not None and raw_fps > 0 and not np.isnan(raw_fps)) else 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError(f"Invalid video dimensions: {width}x{height} in {input_path}")

        writer = None
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = WebVideoWriter(str(output_path), fps, width, height)

        print(f"[VideoBehaviourPipeline] Processing video: {input_path.name}")
        print(f"  Dimensions: {width}x{height} | FPS: {fps:.2f} | Frames: {total_frames}")
        if output_path:
            print(f"  Target annotated output: {output_path}")

        frame_idx = 0
        all_track_ids = set()
        person_telemetry_history: Dict[int, List[Dict[str, Any]]] = {}
        track_predictions: Dict[int, List[Dict[str, Any]]] = {}
        start_time = time.time()

        target_frames = min(max_frames, total_frames) if max_frames else total_frames

        try:
            while cap.isOpened():
                if max_frames and frame_idx >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                annotated_frame, records, faces = self.process_frame(
                    frame, frame_idx=frame_idx, total_frames=total_frames, fps=fps, max_frames=max_frames
                )

                for r in records:
                    tid = r["track_id"]
                    if tid > 0:
                        all_track_ids.add(tid)
                    if tid not in person_telemetry_history:
                        person_telemetry_history[tid] = []
                        track_predictions[tid] = []

                    timestamp_sec = round((frame_idx - 1) / fps, 3)

                    # Unified rich record
                    person_telemetry_history[tid].append({
                        "frame": frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "bbox": r["person_bbox"],
                        "det_confidence": r["det_confidence"],
                        "behaviour": r["behaviour"],
                        "emotion": r["emotion"],
                    })

                    # Backwards-compatible track_predictions entry
                    beh = r["behaviour"]
                    track_predictions[tid].append({
                        "frame": frame_idx,
                        "status": beh.get("status", "lost"),
                        "class_id": beh.get("class_id", -1),
                        "class_name": beh.get("class_name", "Unknown"),
                        "confidence": beh.get("confidence", 0.0),
                        "probabilities": beh.get("probabilities", {}),
                        "risk_level": beh.get("risk_level", "—"),
                    })

                if incident_detector is not None:
                    try:
                        incident_detector.process_frame(
                            frame_idx=frame_idx,
                            timestamp_sec=round((frame_idx - 1) / fps, 3),
                            annotated_frame=annotated_frame,
                            integrated_records=records,
                            video_id=video_id or input_path.stem,
                            source_filename=input_path.name,
                        )
                    except Exception as inc_err:
                        print(f"[VideoBehaviourPipeline] Incident detector warning on frame {frame_idx}: {inc_err}")

                if writer is not None:
                    writer.write(annotated_frame)

                if progress_callback:
                    progress_callback(frame_idx, target_frames)

                if frame_idx % 10 == 0 or frame_idx == target_frames:
                    elapsed = time.time() - start_time
                    fps_proc = frame_idx / elapsed if elapsed > 0 else 0
                    print(f"  Processed {frame_idx}/{target_frames} frames ({fps_proc:.1f} FPS)...")

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            self.reset_all()

        total_elapsed = time.time() - start_time
        fps_overall = frame_idx / total_elapsed if total_elapsed > 0 else 0
        print(f"[VideoBehaviourPipeline] Completed in {total_elapsed:.2f}s ({frame_idx} frames, {fps_overall:.1f} FPS).")

        return {
            "input_file": str(input_path),
            "output_file": str(output_path) if output_path else None,
            "total_frames_processed": frame_idx,
            "unique_track_ids": sorted(list(all_track_ids)),
            "person_telemetry_history": person_telemetry_history,
            "track_predictions": track_predictions,
            "recorded_incidents": incident_detector.session_incidents if incident_detector is not None else [],
            "elapsed_seconds": round(total_elapsed, 2),
            "fps": round(fps_overall, 2),
        }


# Alias for unified naming convention
UnifiedVideoPipeline = VideoBehaviourPipeline


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified Multi-Branch Video Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated output video")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    args = parser.parse_args()

    pipeline = VideoBehaviourPipeline()
    res = pipeline.process_video(args.input, args.output, max_frames=args.max_frames)
    print("\nProcessing Summary:")
    print(f"  Frames processed: {res['total_frames_processed']}")
    print(f"  Unique track IDs: {res['unique_track_ids']}")
    for tid, hist in res["person_telemetry_history"].items():
        ready_b = [h for h in hist if h["behaviour"].get("status") == "predicted"]
        ready_e = [h for h in hist if h["emotion"].get("status") == "available"]
        print(f"  Track ID {tid}: {len(hist)} frames ({len(ready_b)} behaviour preds, {len(ready_e)} face emotion matches)")


if __name__ == "__main__":
    main()
