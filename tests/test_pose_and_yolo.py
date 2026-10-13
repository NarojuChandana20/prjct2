"""
Test script for independently verifying MediaPipe Pose Landmarker and Ultralytics YOLO11.
Assets tested:
  - weights/pose/pose_landmarker_lite.task
  - weights/yolo/yolo11n.pt
Sample assets used:
  - tests/assets/person_sample.jpg
  - tests/assets/test_sample.mp4
"""

import sys
from pathlib import Path
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
POSE_MODEL_PATH = ROOT_DIR / "weights" / "pose" / "pose_landmarker_lite.task"
YOLO_MODEL_PATH = ROOT_DIR / "weights" / "yolo" / "yolo11n.pt"
SAMPLE_IMAGE_PATH = ROOT_DIR / "tests" / "assets" / "person_sample.jpg"
SAMPLE_VIDEO_PATH = ROOT_DIR / "tests" / "assets" / "test_sample.mp4"

def test_mediapipe_pose():
    print("=" * 60)
    print("[TEST 1/3] MediaPipe Pose Landmarker Independent Test")
    print("=" * 60)
    print(f"Target model path: {POSE_MODEL_PATH}")

    if not POSE_MODEL_PATH.exists():
        print(f"[FAIL] Pose model file not found: {POSE_MODEL_PATH}")
        return False

    size = POSE_MODEL_PATH.stat().st_size
    print(f"Pose model file size: {size:,} bytes")
    if size == 0:
        print("[FAIL] Pose model file is empty!")
        return False

    # 1. Initialize Pose Landmarker
    try:
        base_options = python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=2,
            min_pose_detection_confidence=0.45,
            min_pose_presence_confidence=0.45,
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)
        print("  --> [PASS] MediaPipe Pose Landmarker initialized successfully.")
    except Exception as e:
        print(f"  --> [FAIL] MediaPipe Pose initialization error: {e}")
        return False

    # 2. Check for sample image
    if not SAMPLE_IMAGE_PATH.exists():
        print(f"  --> [REQUIREMENT MISSING] No sample person image found at: {SAMPLE_IMAGE_PATH}")
        return False

    # 3. Run pose detection on sample image
    img_bgr = cv2.imread(str(SAMPLE_IMAGE_PATH))
    if img_bgr is None:
        print(f"  --> [FAIL] Could not read sample image: {SAMPLE_IMAGE_PATH}")
        return False

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = MPImage(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_img)

    num_poses = len(result.pose_landmarks) if result.pose_landmarks else 0
    print(f"  --> Poses detected on full image: {num_poses}")

    if num_poses > 0:
        landmark_count = len(result.pose_landmarks[0])
        print(f"  --> Landmark count in first pose: {landmark_count} (Expected: 33)")
        if landmark_count == 33:
            print("  --> [PASS] Verified exactly 33 pose landmarks extracted.")
            return True
        else:
            print(f"  --> [FAIL] Unexpected landmark count: {landmark_count}")
            return False
    else:
        print("  --> [WARNING] No full-body pose detected on uncropped wide frame.")
        return False

def test_yolo_detection():
    print("\n" + "=" * 60)
    print("[TEST 2/3] Ultralytics YOLO11 Person Detection Test")
    print("=" * 60)
    print(f"Target model path: {YOLO_MODEL_PATH}")

    if not YOLO_MODEL_PATH.exists():
        print(f"[FAIL] YOLO model file not found: {YOLO_MODEL_PATH}")
        return False

    size = YOLO_MODEL_PATH.stat().st_size
    print(f"YOLO model file size: {size:,} bytes")
    if size == 0:
        print("[FAIL] YOLO model file is empty!")
        return False

    # 1. Load model
    try:
        model = YOLO(str(YOLO_MODEL_PATH))
        print("  --> [PASS] YOLO11 model weights loaded successfully.")
    except Exception as e:
        print(f"  --> [FAIL] Error loading YOLO11 weights: {e}")
        return False

    # 2. Check sample image
    if not SAMPLE_IMAGE_PATH.exists():
        print(f"  --> [REQUIREMENT MISSING] No sample person image at: {SAMPLE_IMAGE_PATH}")
        return False

    # 3. Run person detection (classes=[0])
    try:
        results = model(str(SAMPLE_IMAGE_PATH), classes=[0], conf=0.35, verbose=False)
        boxes = results[0].boxes
        num_persons = len(boxes)
        print(f"  --> [PASS] Person detection completed. Detected persons: {num_persons}")

        for i, box in enumerate(boxes):
            coords = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            print(f"      Person {i+1}: Class={cls_id} (person), Conf={conf:.4f}, Box={coords.astype(int).tolist()}")

        return num_persons > 0
    except Exception as e:
        print(f"  --> [FAIL] Person detection error: {e}")
        return False

def test_yolo_tracking():
    print("\n" + "=" * 60)
    print("[TEST 3/3] Ultralytics YOLO11 Video Tracking Test")
    print("=" * 60)

    if not SAMPLE_VIDEO_PATH.exists():
        print(f"  --> [SKIP] No sample video found at: {SAMPLE_VIDEO_PATH}")
        return False

    try:
        model = YOLO(str(YOLO_MODEL_PATH))
        cap = cv2.VideoCapture(str(SAMPLE_VIDEO_PATH))
        if not cap.isOpened():
            print(f"  --> [FAIL] Could not open test video: {SAMPLE_VIDEO_PATH}")
            return False

        frame_count = 0
        all_track_ids = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            results = model.track(frame, persist=True, classes=[0], conf=0.35, verbose=False)
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                all_track_ids.update(track_ids)

        cap.release()
        print(f"  --> Frames processed: {frame_count}")
        print(f"  --> Unique Track IDs assigned across video: {sorted(list(all_track_ids))}")

        if all_track_ids:
            print("  --> [PASS] Tracking test succeeded with persistent tracking IDs assigned.")
            return True
        else:
            print("  --> [WARNING] Tracking ran, but no track IDs were confirmed.")
            return False
    except Exception as e:
        print(f"  --> [FAIL] Video tracking error: {e}")
        return False

def main():
    pose_ok = test_mediapipe_pose()
    yolo_det_ok = test_yolo_detection()
    yolo_track_ok = test_yolo_tracking()

    print("\n" + "=" * 60)
    print("STAGE 3 SUMMARY RESULTS")
    print("=" * 60)
    print(f"1. MediaPipe Pose Initialization & 33 Landmarks: {'PASSED' if pose_ok else 'FAILED'}")
    print(f"2. YOLO11 Person Detection:                       {'PASSED' if yolo_det_ok else 'FAILED'}")
    print(f"3. YOLO11 Video Tracking:                         {'PASSED' if yolo_track_ok else 'FAILED'}")

    if pose_ok and yolo_det_ok and yolo_track_ok:
        print("\nAll Stage 3 tests passed successfully!")
        sys.exit(0)
    else:
        print("\nSome tests did not pass.")
        sys.exit(1)

if __name__ == "__main__":
    main()
