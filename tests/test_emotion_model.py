"""
Test script for verifying the Facial Emotion Model independently.
Model: weights/emotion/rafdb_emotion_best.keras
Input: (batch_size, 224, 224, 3), float32 RGB in range [0.0, 255.0]
Output: (batch_size, 7) -> ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras

EMOTION_CLASSES = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happy",
    "Sad",
    "Angry",
    "Neutral",
]

def main():
    model_path = Path(__file__).resolve().parent.parent / "weights" / "emotion" / "rafdb_emotion_best.keras"
    print(f"[TEST - EMOTION] Target model path: {model_path}")

    if not model_path.exists():
        print(f"[ERROR] Model file not found at: {model_path}")
        sys.exit(1)

    # 1. Test model loading
    print("[TEST - EMOTION] Loading emotion model...")
    try:
        model = keras.models.load_model(str(model_path), compile=False)
        print("  --> Model loaded successfully with compile=False!")
    except Exception as e:
        print(f"[FAILED] Error loading emotion model: {e}")
        sys.exit(1)

    # 2. Verify model architecture specifications
    input_shape = model.input_shape
    output_shape = model.output_shape
    print(f"[TEST - EMOTION] Architecture verified:")
    print(f"  - Input shape:  {input_shape} (Expected: (None, 224, 224, 3))")
    print(f"  - Output shape: {output_shape} (Expected: (None, 7))")
    print(f"  - Model type:   {type(model).__name__}")

    assert input_shape == (None, 224, 224, 3), f"Unexpected input shape: {input_shape}"
    assert output_shape == (None, 7), f"Unexpected output shape: {output_shape}"

    # 3. Dummy input inference test (Range [0.0, 255.0] RGB as expected by EfficientNetB0)
    print("[TEST - EMOTION] Running dummy inference test (224x224 RGB, float32, range [0, 255])...")
    dummy_input = np.random.uniform(0.0, 255.0, size=(1, 224, 224, 3)).astype(np.float32)
    predictions = model.predict(dummy_input, verbose=0)

    print(f"  - Raw prediction output shape: {predictions.shape}")
    probs = predictions[0]
    prob_sum = float(np.sum(probs))
    print(f"  - Probability sum: {prob_sum:.6f}")

    print("  - Per-class probabilities:")
    for idx, (label, prob) in enumerate(zip(EMOTION_CLASSES, probs)):
        print(f"      [{idx}] {label:10s}: {prob:.6f}")

    assert predictions.shape == (1, 7), f"Unexpected prediction shape: {predictions.shape}"
    assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {prob_sum}"
    assert all(p >= 0.0 for p in probs), "Found negative probability"

    pred_idx = int(np.argmax(probs))
    print(f"  - Top predicted emotion: {EMOTION_CLASSES[pred_idx]} (confidence: {probs[pred_idx]:.4f})")
    print("[PASS] Facial Emotion Model test passed completely!\n")

if __name__ == "__main__":
    main()
