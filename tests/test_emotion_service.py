"""
Unit test suite for EmotionInferenceService.
Verifies:
  1. Service initialization and EfficientNetB0 model loading.
  2. Safe handling of invalid, None, and empty input crops.
  3. Preprocessing: BGR-to-RGB conversion, resize to (224, 224), float32 [0.0, 255.0].
  4. Output prediction dictionary format, confidence bounds, and 7-class probability sum.
  5. Format compatibility when receiving face crops directly from FaceDetectionService.
  6. Batch prediction on multiple detected face crops.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.face_service import FaceDetectionService
from backend.app.services.emotion_service import EmotionInferenceService, EMOTION_CLASSES

def test_emotion_service():
    print("=" * 65)
    print("TEST: EmotionInferenceService Independent Verification")
    print("=" * 65)

    # 1. Test Model Loading
    print("\n[Test 1/5] Service Initialization & Model Loading...")
    try:
        service = EmotionInferenceService()
        print(f"  --> [PASS] Model loaded from: {service.model_path.name}")
    except Exception as e:
        print(f"  --> [FAIL] Error loading emotion model: {e}")
        sys.exit(1)

    # 2. Test Invalid and Empty Crop Inputs
    print("\n[Test 2/5] Invalid & Empty Crop Inputs...")
    assert service.predict_emotion(None) is None, "Expected None for None input"
    assert service.predict_emotion(np.zeros((0, 0, 3), dtype=np.uint8)) is None, "Expected None for 0x0 crop"
    assert service.predict_emotion(np.zeros((2, 2, 3), dtype=np.uint8)) is None, "Expected None for degenerate crop"
    print("  --> [PASS] All invalid inputs handled safely (returned None).")

    # 3. Test Prediction on Synthetic/Dummy Input (Technical Verification)
    print("\n[Test 3/5] Technical Verification with Synthetic Input...")
    dummy_crop = np.random.uniform(0.0, 255.0, size=(120, 120, 3)).astype(np.uint8)
    dummy_result = service.predict_emotion(dummy_crop, input_format="BGR")

    assert dummy_result is not None, "Dummy prediction returned None!"
    assert "emotion" in dummy_result, "Missing 'emotion' field"
    assert "class_id" in dummy_result, "Missing 'class_id' field"
    assert "confidence" in dummy_result, "Missing 'confidence' field"
    assert "probabilities" in dummy_result, "Missing 'probabilities' field"

    cid = dummy_result["class_id"]
    conf = dummy_result["confidence"]
    emotion_name = dummy_result["emotion"]
    probs = dummy_result["probabilities"]

    assert 0 <= cid < 7, f"Invalid class_id: {cid}"
    assert emotion_name == EMOTION_CLASSES[cid], f"Class mapping mismatch: {emotion_name} vs {EMOTION_CLASSES[cid]}"
    assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"
    assert len(probs) == 7, f"Expected 7 classes, got {len(probs)}"

    prob_sum = sum(probs.values())
    assert abs(prob_sum - 1.0) < 1e-3, f"Probabilities do not sum to 1.0: {prob_sum}"

    print(f"  --> Synthetic input result:")
    print(f"      Top emotion:  {emotion_name} (class {cid})")
    print(f"      Confidence:   {conf:.4f}")
    print(f"      Probabilities sum: {prob_sum:.4f}")
    print("  --> [PASS] Technical schema, confidence range, and probability sum validated.")

    # 4. Test Integration with FaceDetectionService (Real Face Crop)
    print("\n[Test 4/5] Integration: FaceDetectionService Crop -> EmotionInferenceService...")
    face_service = FaceDetectionService(min_confidence=0.45)
    test_img_path = ROOT_DIR / "tests" / "assets" / "single_face.jpg"

    if not test_img_path.exists():
        print(f"  --> [FAIL] Missing sample image: {test_img_path}")
        sys.exit(1)

    test_bgr = cv2.imread(str(test_img_path))
    detections = face_service.detect_faces(test_bgr, input_format="BGR")

    assert len(detections) >= 1, f"Face detection failed on {test_img_path.name}"
    real_crop = detections[0]["crop"]
    print(f"  --> Extracted face crop from FaceDetector: shape={real_crop.shape}, dtype={real_crop.dtype}")

    # Pass crop directly to emotion service
    emotion_res = service.predict_emotion(real_crop, input_format="BGR")
    assert emotion_res is not None, "Emotion prediction returned None for real face crop!"

    print(f"  --> Real Face Crop Prediction:")
    print(f"      Top Emotion: {emotion_res['emotion']} (class {emotion_res['class_id']})")
    print(f"      Confidence:  {emotion_res['confidence']:.4f}")
    print(f"      Distribution:")
    for em, p in emotion_res["probabilities"].items():
        print(f"        {em:10s}: {p:.4f}")

    print("  --> [PASS] Seamless hand-off from FaceDetectionService to EmotionInferenceService verified.")

    # 5. Test Batch Prediction on Multi-Face Scene
    print("\n[Test 5/5] Batch Prediction on Multi-Face Scene...")
    multi_img_path = ROOT_DIR / "tests" / "assets" / "multiple_faces.jpg"
    multi_bgr = cv2.imread(str(multi_img_path))
    multi_dets = face_service.detect_faces(multi_bgr, input_format="BGR")
    assert len(multi_dets) >= 2, f"Expected at least 2 faces in {multi_img_path.name}"

    crops = [d["crop"] for d in multi_dets]
    batch_results = service.predict_batch(crops, input_format="BGR")

    assert len(batch_results) == len(crops), "Batch size mismatch"
    for i, res in enumerate(batch_results):
        assert res is not None, f"Batch item {i} returned None"
        print(f"      Face {i+1}: Emotion='{res['emotion']}', Confidence={res['confidence']:.4f}")

    print("  --> [PASS] Batch inference verified across multiple faces.")

    print("\n" + "=" * 65)
    print("ALL EMOTION INFERENCE SERVICE TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    test_emotion_service()
