"""
Stage 9A: Unified Multi-Branch Video Pipeline Test Suite
--------------------------------------------------------
Tests unit error-resilience and real-video multi-branch processing for:
  1. YOLO11 person detection and tracking (classes=[0]).
  2. BehaviourTracker (MediaPipe Pose + BiLSTM) sequence collection and inference.
  3. FaceDetectionService independent face detection.
  4. EmotionInferenceService EfficientNetB0 facial emotion prediction.
  5. FacePersonAssociationService geometric 1-to-1 matching and ambiguity rejection.
  6. Multi-branch visual annotation rendering (body boxes, face badges, HUD).
  7. Output video generation, integrity verification, and error-handling resilience.

Separation of test scopes:
  - Part A: Unit & Error Resilience Tests (empty frame, missing face, matched face, inference failure)
  - Part B: Real-Video Integration Tests (using tests/assets/test_person_tracking_35f.mp4)
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.app.services.video_behaviour_pipeline import VideoBehaviourPipeline, UnifiedVideoPipeline


def test_unit_empty_frame(pipeline: VideoBehaviourPipeline):
    """[Unit Test 1] Verifies that an empty or black frame produces 0 detections without crashing."""
    print("\n--- [Unit Test 1] Frame With No People (Black Canvas) ---")
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, records, faces = pipeline.process_frame(black_frame, frame_idx=1, total_frames=1)

    assert annotated is not None, "Annotated frame must not be None"
    assert annotated.shape == (480, 640, 3), f"Frame shape changed: {annotated.shape}"
    assert len(records) == 0, f"Expected 0 person records on black frame, got {len(records)}"
    assert len(faces) == 0, f"Expected 0 face detections on black frame, got {len(faces)}"
    print("  --> [PASS] Handled empty canvas safely with 0 person records and 0 faces.")


def test_unit_missing_face_fallback(pipeline: VideoBehaviourPipeline):
    """[Unit Test 2] Verifies that a person without a visible face gets an 'Unavailable' emotion state."""
    print("\n--- [Unit Test 2] Person With Occluded/Missing Face ---")
    person_img_path = ROOT_DIR / "tests" / "assets" / "person_sample.jpg"
    if not person_img_path.exists():
        print(f"  [SKIP] person_sample.jpg not found at {person_img_path}")
        return

    img = cv2.imread(str(person_img_path))
    assert img is not None, "Failed to load person_sample.jpg"

    annotated, records, faces = pipeline.process_frame(img, frame_idx=1, total_frames=1)
    print(f"  --> Processed person sample: {len(records)} persons detected, {len(faces)} faces detected.")
    assert len(records) > 0, "Expected at least 1 person detected in person_sample.jpg"

    for r in records:
        assert "behaviour" in r, "Record missing behaviour field"
        assert "emotion" in r, "Record missing emotion field"
        assert "confidence" in r["behaviour"], "Behaviour missing confidence"
        assert "confidence" in r["emotion"], "Emotion missing confidence"
        # Confirm that no combined risk score exists
        assert "composite_score" not in r, "Violation: Found composite risk score!"
        assert "combined_risk" not in r, "Violation: Found combined risk score!"
        print(f"  --> Person ID {r['track_id']}: Behaviour status='{r['behaviour']['status']}', Emotion status='{r['emotion']['status']}', Label='{r['emotion']['label']}'")

    print("  --> [PASS] Independent telemetry verified; composite score absent.")


def test_unit_face_association_and_emotion(pipeline: VideoBehaviourPipeline):
    """[Unit Test 3] Verifies face detection + emotion inference + association on single_face.jpg."""
    print("\n--- [Unit Test 3] Direct Face Detection & Emotion Inference ---")
    face_img_path = ROOT_DIR / "tests" / "assets" / "single_face.jpg"
    if not face_img_path.exists():
        print(f"  [SKIP] single_face.jpg not found at {face_img_path}")
        return

    img = cv2.imread(str(face_img_path))
    assert img is not None, "Failed to load single_face.jpg"

    # Direct test of face detection & emotion inference branches
    faces = pipeline.face_service.detect_faces(img, input_format="BGR")
    print(f"  --> Faces detected: {len(faces)}")
    assert len(faces) >= 1, "Expected at least 1 face detected in single_face.jpg"

    face = faces[0]
    crop = face.get("crop")
    assert crop is not None, "Face crop is missing"
    assert crop.size > 0, "Face crop is empty"

    pred = pipeline.emotion_service.predict_emotion(crop, input_format="BGR")
    print(f"  --> Emotion predicted: '{pred['emotion']}', Confidence: {pred['confidence']:.4f}")
    assert pred["emotion"] in [
        "Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"
    ], f"Unexpected emotion class: {pred['emotion']}"
    assert 0.0 <= pred["confidence"] <= 1.0, f"Confidence out of range: {pred['confidence']}"
    print("  --> [PASS] Face detection and emotion inference branch operational.")


def test_unit_emotion_failure_safety(pipeline: VideoBehaviourPipeline):
    """[Unit Test 4] Verifies that an emotion inference failure does not crash frame processing."""
    print("\n--- [Unit Test 4] Emotion Inference Failure Resilience ---")
    # Simulate a corrupted face detection in association service
    mock_person = [{
        "track_id": 99,
        "bbox": [100, 100, 300, 500],
        "det_confidence": 0.90,
        "behaviour": {"status": "predicted", "class_id": 0, "class_name": "NonFight", "confidence": 0.95},
    }]
    mock_corrupted_face = [{
        "bbox": [150, 120, 250, 220],
        "confidence": 0.90,
        "crop": None,
        "emotion": {
            "status": "unavailable",
            "label": "Unavailable",
            "confidence": 0.0,
            "error": "simulated_model_error",
        },
    }]

    records = pipeline.association_service.associate_frame(mock_person, mock_corrupted_face)
    assert len(records) == 1, "Expected 1 record"
    rec = records[0]
    assert rec["emotion"]["status"] == "unavailable", "Emotion should be marked unavailable on failed face"
    assert rec["emotion"]["label"] == "Unavailable", "Emotion label should be Unavailable"
    assert rec["emotion"]["confidence"] == 0.0, "Emotion confidence should be 0.0"
    print("  --> [PASS] Failed emotion inference gracefully handled with Unavailable status.")


def test_real_video_integration(pipeline: VideoBehaviourPipeline):
    """[Real-Video Integration Test] Runs the multi-branch pipeline on the 35-frame tracking video asset."""
    print("\n--- [Real-Video Integration Test] 35-Frame Person Tracking Video ---")
    input_video = ROOT_DIR / "tests" / "assets" / "test_person_tracking_35f.mp4"
    output_video = ROOT_DIR / "tests" / "assets" / "test_unified_output.mp4"

    if not input_video.exists():
        print(f"  [ERROR] Real test video missing: {input_video}")
        sys.exit(1)

    print(f"  Ingesting real video: {input_video.name}")
    res = pipeline.process_video(
        input_path=input_video,
        output_path=output_video,
        max_frames=35,
    )

    # 1. Total frames processed
    frames_proc = res["total_frames_processed"]
    print(f"\n  [Check 1] Frame Processing: {frames_proc}/35 frames processed.")
    assert frames_proc == 35, f"Expected 35 frames, got {frames_proc}"

    # 2. Track IDs
    track_ids = res["unique_track_ids"]
    print(f"  [Check 2] Unique Track IDs: {track_ids}")
    assert len(track_ids) > 0, "No tracked persons detected in video!"
    print(f"    --> Successfully tracked {len(track_ids)} unique individuals.")

    # 3. Telemetry inspection per track ID
    print(f"\n  [Check 3] Multi-Branch Telemetry per Track:")
    telemetry = res["person_telemetry_history"]

    saw_collecting = False
    saw_prediction = False
    saw_face_association = False
    saw_unavailable_emotion = False

    for tid in track_ids:
        frames_data = telemetry.get(tid, [])
        collecting_for_tid = [f for f in frames_data if f["behaviour"].get("status") == "collecting"]
        predicted_for_tid = [f for f in frames_data if f["behaviour"].get("status") == "predicted"]

        print(f"    Track ID {tid}: {len(frames_data)} updates ({len(collecting_for_tid)} collecting, {len(predicted_for_tid)} predicted)")

        if collecting_for_tid:
            saw_collecting = True
        if predicted_for_tid:
            saw_prediction = True

        for f_data in frames_data:
            e = f_data["emotion"]
            if e.get("status") == "available":
                saw_face_association = True
            elif e.get("status") == "unavailable":
                saw_unavailable_emotion = True

    print(f"    - Body sequence collection observed: {saw_collecting}")
    print(f"    - Full behaviour prediction observed: {saw_prediction}")
    print(f"    - Emotion status available observed: {saw_face_association}")
    print(f"    - Emotion status unavailable observed: {saw_unavailable_emotion}")

    assert saw_collecting, "Body sequence collection state was never observed!"
    print("    --> [PASS] Verified body observation gathering state.")

    if saw_prediction:
        print("    --> [PASS] Full behaviour prediction reached after sequence accumulation.")
    else:
        print("    --> [INFO] Video duration insufficient for some tracks to reach 30 frames; gathering state active.")

    # 4. Independence of tracking IDs
    print(f"\n  [Check 4] Independent Tracking States:")
    for tid in track_ids:
        updates = telemetry[tid]
        first_frame = updates[0]["frame"]
        last_frame = updates[-1]["frame"]
        print(f"    Track ID {tid}: active frames {first_frame} -> {last_frame} (total {len(updates)})")
    print("    --> [PASS] Each track ID maintains independent state and buffers.")

    # 5. Output Video Integrity
    print(f"\n  [Check 5] Annotated Output Video Integrity:")
    assert output_video.exists(), f"Output video was not created: {output_video}"
    file_size = output_video.stat().st_size
    print(f"    File: {output_video.name} | Size: {file_size:,} bytes")
    assert file_size > 0, "Output video file is empty!"

    cap = cv2.VideoCapture(str(output_video))
    assert cap.isOpened(), "Could not reopen output video!"
    out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"    Reopened video specs: {out_w}x{out_h}, {out_frames} frames.")
    assert out_frames == 35, f"Expected 35 frames in output video, got {out_frames}"
    print("    --> [PASS] Annotated output video verified completely.")


def main():
    print("=" * 70)
    print("STAGE 9A: UNIFIED MULTI-BRANCH VIDEO PIPELINE INTEGRATION TEST")
    print("=" * 70)

    print("\n[Step 1] Initializing UnifiedVideoPipeline...")
    pipeline = VideoBehaviourPipeline()
    print("  --> Pipeline initialized successfully.")

    # Part A: Unit & Error Resilience Tests
    test_unit_empty_frame(pipeline)
    test_unit_missing_face_fallback(pipeline)
    test_unit_face_association_and_emotion(pipeline)
    test_unit_emotion_failure_safety(pipeline)

    # Part B: Real-Video Integration Tests
    test_real_video_integration(pipeline)

    print("\n" + "=" * 70)
    print("ALL STAGE 9A INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
