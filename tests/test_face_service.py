"""
Unit test suite for FaceDetectionService with dedicated MediaPipe Face Detector.
Verifies:
  1. Service initialization with dedicated 'blaze_face_short_range.tflite' model.
  2. Safe handling of invalid, None, and empty input frames.
  3. Processing an image with zero visible faces (no_face.jpg).
  4. Detecting exactly one face in an image with one person/face (single_face.jpg).
  5. Detecting multiple faces in an image with multiple visible faces (multiple_faces.jpg).
  6. Correct coordinate bounds, confidence values, and face crop extraction.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.face_service import FaceDetectionService

def test_face_service():
    print("=" * 65)
    print("TEST: Dedicated FaceDetectionService Verification")
    print("=" * 65)

    # 1. Test Model Initialization
    print("\n[Test 1/6] Dedicated Service Initialization...")
    try:
        service = FaceDetectionService(min_confidence=0.45)
        print(f"  --> [PASS] Service initialized. Active Engine: '{service.engine}'")
        assert service.engine == "mediapipe_face", f"Expected 'mediapipe_face', got '{service.engine}'"
    except Exception as e:
        print(f"  --> [FAIL] Initialization error: {e}")
        sys.exit(1)

    # 2. Test Invalid and Empty Frame Inputs
    print("\n[Test 2/6] Invalid & Empty Frame Inputs...")
    res_none = service.detect_faces(None)
    assert res_none == [], f"Expected [], got {res_none} for None frame"
    print("  --> None frame handled safely (returned empty list).")

    res_empty = service.detect_faces(np.zeros((0, 0, 3), dtype=np.uint8))
    assert res_empty == [], f"Expected [], got {res_empty} for empty frame"
    print("  --> Empty (0x0) frame handled safely (returned empty list).")

    res_degenerate = service.detect_faces(np.zeros((2, 2, 3), dtype=np.uint8))
    assert res_degenerate == [], f"Expected [], got {res_degenerate} for degenerate frame"
    print("  --> Degenerate (2x2) frame handled safely (returned empty list).")
    print("  --> [PASS] All invalid inputs handled safely without exceptions.")

    # 3. Test Image with No Visible Faces
    no_face_path = ROOT_DIR / "tests" / "assets" / "no_face.jpg"
    print(f"\n[Test 3/6] Image with No Visible Faces: {no_face_path.name}...")
    if not no_face_path.exists():
        print(f"  --> [REQUIREMENT MISSING] Sample file missing: {no_face_path}")
        sys.exit(1)

    no_face_img = cv2.imread(str(no_face_path))
    res_no_face = service.detect_faces(no_face_img, input_format="BGR")
    print(f"  --> Faces detected on background image: {len(res_no_face)} (Expected: 0)")
    assert len(res_no_face) == 0, f"Expected 0 faces, but detected {len(res_no_face)}"
    print("  --> [PASS] Zero faces confirmed on non-face background image.")

    # 4. Test Image with Single Face
    single_face_path = ROOT_DIR / "tests" / "assets" / "single_face.jpg"
    print(f"\n[Test 4/6] Image with Single Face: {single_face_path.name}...")
    if not single_face_path.exists():
        print(f"  --> [REQUIREMENT MISSING] Sample file missing: {single_face_path}")
        sys.exit(1)

    single_face_img = cv2.imread(str(single_face_path))
    h, w = single_face_img.shape[:2]
    res_single = service.detect_faces(single_face_img, input_format="BGR")
    print(f"  --> Faces detected: {len(res_single)}")

    assert len(res_single) == 1, f"Expected exactly 1 face detected, got {len(res_single)}"
    face = res_single[0]
    bbox = face["bbox"]
    conf = face["confidence"]
    crop = face["crop"]

    print(f"  --> Bounding Box: {bbox} (Image dimensions: {w}x{h})")
    print(f"  --> Confidence:   {conf:.4f}")
    print(f"  --> Crop shape:   {crop.shape}")

    # Validate coordinate boundaries
    x1, y1, x2, y2 = bbox
    assert 0 <= x1 < x2 <= w, f"Invalid x bounds: {x1}, {x2} (width {w})"
    assert 0 <= y1 < y2 <= h, f"Invalid y bounds: {y1}, {y2} (height {h})"
    assert 0.0 <= conf <= 1.0, f"Invalid confidence: {conf}"
    assert crop.shape[0] == (y2 - y1) and crop.shape[1] == (x2 - x1), "Crop shape mismatch"
    print("  --> [PASS] Single face bounding box, confidence, and crop fully validated.")

    # 5. Test Image with Multiple Faces
    multi_path = ROOT_DIR / "tests" / "assets" / "multiple_faces.jpg"
    print(f"\n[Test 5/6] Image with Multiple Faces: {multi_path.name}...")
    if not multi_path.exists():
        print(f"  --> [REQUIREMENT MISSING] Sample file missing: {multi_path}")
        sys.exit(1)

    multi_img = cv2.imread(str(multi_path))
    res_multi = service.detect_faces(multi_img, input_format="BGR")
    print(f"  --> Faces detected on multi-face image: {len(res_multi)}")
    for i, f in enumerate(res_multi):
        print(f"      Face {i+1}: BBox={f['bbox']}, Conf={f['confidence']:.4f}, CropSize={f['width']}x{f['height']}")
    assert len(res_multi) >= 2, f"Expected at least 2 faces detected, got {len(res_multi)}"
    print("  --> [PASS] Multiple faces detected and extracted successfully.")

    # 6. Test Input Color Conversion (RGB vs BGR)
    print("\n[Test 6/6] Color Format Conversion (RGB input)...")
    single_rgb = cv2.cvtColor(single_face_img, cv2.COLOR_BGR2RGB)
    res_rgb = service.detect_faces(single_rgb, input_format="RGB")
    print(f"  --> Faces detected with input_format='RGB': {len(res_rgb)}")
    assert len(res_rgb) == len(res_single), "Color format conversion caused detection count mismatch"
    print("  --> [PASS] RGB color format handled correctly.")

    print("\n" + "=" * 65)
    print("ALL DEDICATED FACE DETECTION TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    test_face_service()
