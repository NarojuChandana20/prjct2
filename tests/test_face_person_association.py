"""
Unit and integration test suite for FacePersonAssociationService.
Verifies:
  1. One person with one detected face (clear reliable match).
  2. Multiple people with multiple disjoint faces (clean 1-to-1 matching, no crosstalk).
  3. No detected faces in frame (person gets Unavailable emotion).
  4. Ambiguous face match between overlapping candidates (match safely rejected).
  5. Overlapping person bounding boxes with one distinct face.
  6. Missing or invalid tracking ID handling.
  7. Unavailable emotion prediction (face present but emotion inference failed).
  8. Real pipeline end-to-end integration using FaceDetectionService + EmotionInferenceService.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.association_service import FacePersonAssociationService
from backend.app.services.face_service import FaceDetectionService
from backend.app.services.emotion_service import EmotionInferenceService

def test_association_service():
    print("=" * 70)
    print("TEST: FacePersonAssociationService Independent Verification")
    print("=" * 70)

    service = FacePersonAssociationService(
        upper_body_ratio=0.40,
        min_containment=0.35,
        min_match_score=0.40,
        ambiguity_margin=0.15,
    )

    # -------------------------------------------------------------
    # 1. One Person with One Detected Face (Clear Match)
    # -------------------------------------------------------------
    print("\n[Test 1/8] One Person with One Clear Face Match...")
    person = {
        "track_id": 1,
        "bbox": [100, 100, 250, 500],  # Width 150, Height 400. Head zone: Y in [100, 260]
        "det_confidence": 0.89,
        "behaviour": {"status": "predicted", "class_id": 0, "class_name": "NonFight", "confidence": 0.98},
    }
    face = {
        "bbox": [135, 110, 215, 200],  # Centered in head zone
        "confidence": 0.95,
        "emotion": {"emotion": "Happy", "class_id": 3, "confidence": 0.92, "probabilities": {"Happy": 0.92}},
    }

    results = service.associate_frame([person], [face])
    assert len(results) == 1, "Expected exactly 1 result"
    rec = results[0]

    assert rec["track_id"] == 1, "Track ID mismatch"
    assert rec["behaviour"]["class_name"] == "NonFight", "Behaviour data was altered"
    assert rec["emotion"]["status"] == "available", "Emotion should be available"
    assert rec["emotion"]["label"] == "Happy", "Emotion label mismatch"
    assert rec["emotion"]["confidence"] == 0.92, "Emotion confidence mismatch"
    assert rec["emotion"]["face_bbox"] == [135, 110, 215, 200], "Face bbox not attached"
    print("  --> [PASS] One person correctly matched with detected face and emotion.")

    # -------------------------------------------------------------
    # 2. Multiple People with Multiple Faces (1-to-1 Disjoint Matching)
    # -------------------------------------------------------------
    print("\n[Test 2/8] Multiple People with Multiple Faces (Clean 1-to-1 Matching)...")
    person_a = {
        "track_id": 10,
        "bbox": [50, 100, 200, 500],
        "det_confidence": 0.88,
        "behaviour": {"status": "predicted", "class_id": 0, "class_name": "NonFight", "confidence": 0.95},
    }
    person_b = {
        "track_id": 20,
        "bbox": [350, 100, 500, 500],
        "det_confidence": 0.85,
        "behaviour": {"status": "predicted", "class_id": 1, "class_name": "Fight", "confidence": 0.75},
    }
    face_a = {
        "bbox": [80, 110, 170, 200],
        "confidence": 0.91,
        "emotion": {"emotion": "Surprise", "class_id": 0, "confidence": 0.87},
    }
    face_b = {
        "bbox": [380, 110, 470, 200],
        "confidence": 0.93,
        "emotion": {"emotion": "Neutral", "class_id": 6, "confidence": 0.82},
    }

    results = service.associate_frame([person_a, person_b], [face_a, face_b])
    assert len(results) == 2, "Expected 2 person results"

    p10 = next(r for r in results if r["track_id"] == 10)
    p20 = next(r for r in results if r["track_id"] == 20)

    assert p10["emotion"]["label"] == "Surprise", f"Expected Surprise for ID 10, got {p10['emotion']['label']}"
    assert p20["emotion"]["label"] == "Neutral", f"Expected Neutral for ID 20, got {p20['emotion']['label']}"
    assert p10["behaviour"]["class_name"] == "NonFight"
    assert p20["behaviour"]["class_name"] == "Fight"
    print("  --> [PASS] Clean 1-to-1 matching verified with zero identity/emotion crosstalk.")

    # -------------------------------------------------------------
    # 3. No Detected Faces
    # -------------------------------------------------------------
    print("\n[Test 3/8] Person with No Detected Faces...")
    person_solo = {
        "track_id": 5,
        "bbox": [100, 100, 250, 500],
        "det_confidence": 0.90,
        "behaviour": {"status": "predicted", "class_id": 0, "class_name": "NonFight", "confidence": 0.99},
    }

    results = service.associate_frame([person_solo], [])
    assert len(results) == 1
    rec = results[0]
    assert rec["emotion"]["status"] == "unavailable", "Status must be unavailable"
    assert rec["emotion"]["label"] == "Unavailable", "Label must be 'Unavailable'"
    assert rec["emotion"]["confidence"] == 0.0, "Confidence must be 0.0"
    print("  --> [PASS] Absence of faces correctly yielded 'Unavailable' emotion with 0.0 confidence.")

    # -------------------------------------------------------------
    # 4. Ambiguous Face Match Between Two Overlapping People
    # -------------------------------------------------------------
    print("\n[Test 4/8] Ambiguous Face Match Between Competing Persons...")
    person_left = {
        "track_id": 101,
        "bbox": [100, 100, 260, 500],  # [100, 260]
        "det_confidence": 0.85,
    }
    person_right = {
        "track_id": 102,
        "bbox": [190, 100, 350, 500],  # [190, 350]
        "det_confidence": 0.85,
    }
    # Face placed squarely in the overlapping zone between both people
    ambiguous_face = {
        "bbox": [185, 120, 265, 200],
        "confidence": 0.90,
        "emotion": {"emotion": "Angry", "class_id": 5, "confidence": 0.80},
    }

    results = service.associate_frame([person_left, person_right], [ambiguous_face])
    assert len(results) == 2

    # Both persons must NOT be assigned the ambiguous face
    for r in results:
        assert r["emotion"]["label"] == "Unavailable", (
            f"Ambiguous face was incorrectly assigned to ID {r['track_id']}!"
        )
        assert r["emotion"]["status"] in ("ambiguous", "unavailable")
    print("  --> [PASS] Ambiguous face match safely rejected; neither person falsely assigned.")

    # -------------------------------------------------------------
    # 5. Overlapping Persons with One Clear Face
    # -------------------------------------------------------------
    print("\n[Test 5/8] Overlapping Persons with One Distinct Head Match...")
    p1 = {
        "track_id": 1,
        "bbox": [100, 100, 250, 500],
        "det_confidence": 0.88,
    }
    p2 = {
        "track_id": 2,
        "bbox": [180, 120, 330, 520],
        "det_confidence": 0.84,
    }
    # Face clearly on Person 1's left side (outside Person 2's head zone)
    clear_face = {
        "bbox": [120, 110, 175, 175],
        "confidence": 0.92,
        "emotion": {"emotion": "Happy", "class_id": 3, "confidence": 0.89},
    }

    results = service.associate_frame([p1, p2], [clear_face])
    r1 = next(r for r in results if r["track_id"] == 1)
    r2 = next(r for r in results if r["track_id"] == 2)

    assert r1["emotion"]["label"] == "Happy", "Face should match Person 1"
    assert r2["emotion"]["label"] == "Unavailable", "Person 2 should remain Unavailable"
    print("  --> [PASS] Clear face correctly assigned to proper person despite bounding box overlap.")

    # -------------------------------------------------------------
    # 6. Missing or Invalid Tracking ID
    # -------------------------------------------------------------
    print("\n[Test 6/8] Missing or Invalid Tracking ID Handling...")
    p_invalid = {
        "track_id": None,  # Dropped or unassigned track ID
        "bbox": [100, 100, 200, 400],
    }
    p_valid = {
        "track_id": 42,
        "bbox": [300, 100, 450, 500],
    }
    f_sample = {
        "bbox": [340, 110, 410, 180],
        "confidence": 0.90,
        "emotion": {"emotion": "Neutral", "class_id": 6, "confidence": 0.77},
    }

    results = service.associate_frame([p_invalid, p_valid], [f_sample])
    assert len(results) == 1, "Invalid track ID should be safely ignored"
    assert results[0]["track_id"] == 42, "Valid person must be preserved"
    assert results[0]["emotion"]["label"] == "Neutral"
    print("  --> [PASS] Invalid track IDs safely handled without exceptions.")

    # -------------------------------------------------------------
    # 7. Face Detected but Emotion Prediction Failed/None
    # -------------------------------------------------------------
    print("\n[Test 7/8] Face Detected but Emotion Prediction Unavailable...")
    p_target = {
        "track_id": 88,
        "bbox": [100, 100, 250, 500],
        "det_confidence": 0.91,
    }
    f_no_emotion = {
        "bbox": [135, 110, 215, 200],
        "confidence": 0.88,
        "emotion": None,  # Emotion inference was not run or failed
    }

    results = service.associate_frame([p_target], [f_no_emotion])
    assert len(results) == 1
    r88 = results[0]
    assert r88["emotion"]["status"] == "unavailable"
    assert r88["emotion"]["label"] == "Unavailable"
    assert r88["emotion"]["confidence"] == 0.0
    assert r88["emotion"]["reason"] == "emotion_inference_failed"
    assert r88["emotion"]["face_bbox"] == [135, 110, 215, 200]
    print("  --> [PASS] Failed emotion inference safely yields 'Unavailable' while retaining face box.")

    # -------------------------------------------------------------
    # 8. Real Pipeline Integration with Live Detection & Emotion Services
    # -------------------------------------------------------------
    print("\n[Test 8/8] Live Pipeline Integration with Real Image...")
    img_path = ROOT_DIR / "tests" / "assets" / "single_face.jpg"
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    face_svc = FaceDetectionService(min_confidence=0.45)
    emotion_svc = EmotionInferenceService()

    # 1. Real face detection on sample image
    face_dets = face_svc.detect_faces(img, input_format="BGR")
    assert len(face_dets) >= 1, "Real face detection failed on sample image"

    # 2. Real emotion prediction on detected face
    for f in face_dets:
        f["emotion"] = emotion_svc.predict_emotion(f["crop"], input_format="BGR")

    # 3. Simulate person bbox enclosing the single person in image [0, 0, w, h]
    real_person = {
        "track_id": 1,
        "bbox": [0, 0, w, h],
        "det_confidence": 0.95,
        "behaviour": {"status": "predicted", "class_id": 0, "class_name": "NonFight / Normal", "confidence": 0.99},
    }

    real_results = service.associate_frame([real_person], face_dets)
    assert len(real_results) == 1
    real_rec = real_results[0]

    assert real_rec["track_id"] == 1
    assert real_rec["behaviour"]["class_name"] == "NonFight / Normal"
    assert real_rec["emotion"]["status"] == "available"
    assert real_rec["emotion"]["confidence"] > 0.0
    print(f"  --> Real Integration Result:")
    print(f"      Track ID:           {real_rec['track_id']}")
    print(f"      Body Behaviour:     {real_rec['behaviour']['class_name']} ({real_rec['behaviour']['confidence']:.2f})")
    print(f"      Facial Emotion:     {real_rec['emotion']['label']} ({real_rec['emotion']['confidence']:.2f})")
    print(f"      Matched Face BBox:  {real_rec['emotion']['face_bbox']}")
    print("  --> [PASS] Full real-service pipeline integration succeeded.")

    print("\n" + "=" * 70)
    print("ALL 8 FACE-PERSON ASSOCIATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_association_service()
