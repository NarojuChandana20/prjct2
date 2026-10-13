"""
Test script for verifying the Body Behaviour Model independently.
Model: weights/behaviour/behaviour_model_rich_165_v2_best.keras
Input: (batch_size, 30, 165)
Output: (batch_size, 3) -> ['NonFight / Normal', 'Fight', 'Fall']
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras

CLASS_NAMES = {
    0: "NonFight / Normal",
    1: "Fight",
    2: "Fall",
}

def focal_loss(gamma=2.0, alpha=0.25):
    """Custom focal loss matching training configuration in Untitled5 (2).ipynb."""
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)
    return loss

def main():
    model_path = Path(__file__).resolve().parent.parent / "weights" / "behaviour" / "behaviour_model_rich_165_v2_best.keras"
    print(f"[TEST - BEHAVIOUR] Target model path: {model_path}")

    if not model_path.exists():
        print(f"[ERROR] Model file not found at: {model_path}")
        sys.exit(1)

    # 1. Test model loading with custom focal loss
    print("[TEST - BEHAVIOUR] Loading model with custom focal_loss...")
    try:
        model = keras.models.load_model(
            str(model_path),
            custom_objects={"loss": focal_loss(gamma=2.0, alpha=0.25)}
        )
        print("  --> Model loaded successfully with custom_objects!")
    except Exception as e:
        print(f"[FAILED] Error loading model with custom_objects: {e}")
        sys.exit(1)

    # 2. Test model loading with compile=False (inference mode)
    print("[TEST - BEHAVIOUR] Testing loading with compile=False...")
    try:
        model_inf = keras.models.load_model(str(model_path), compile=False)
        print("  --> Model loaded successfully with compile=False!")
    except Exception as e:
        print(f"[WARNING] compile=False loading failed: {e}")

    # 3. Verify model architecture specifications
    input_shape = model.input_shape
    output_shape = model.output_shape
    print(f"[TEST - BEHAVIOUR] Architecture verified:")
    print(f"  - Input shape:  {input_shape} (Expected: (None, 30, 165))")
    print(f"  - Output shape: {output_shape} (Expected: (None, 3))")
    print(f"  - Total layers: {len(model.layers)}")

    assert input_shape == (None, 30, 165), f"Unexpected input shape: {input_shape}"
    assert output_shape == (None, 3), f"Unexpected output shape: {output_shape}"

    # 4. Dummy input inference test
    print("[TEST - BEHAVIOUR] Running dummy inference test...")
    dummy_input = np.zeros((1, 30, 165), dtype=np.float32)
    predictions = model.predict(dummy_input, verbose=0)

    print(f"  - Raw prediction output shape: {predictions.shape}")
    probs = predictions[0]
    prob_sum = float(np.sum(probs))
    print(f"  - Probabilities: NonFight={probs[0]:.6f}, Fight={probs[1]:.6f}, Fall={probs[2]:.6f}")
    print(f"  - Probability sum: {prob_sum:.6f}")

    assert predictions.shape == (1, 3), f"Unexpected prediction shape: {predictions.shape}"
    assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {prob_sum}"
    assert all(p >= 0.0 for p in probs), "Found negative probability"

    pred_idx = int(np.argmax(probs))
    print(f"  - Predicted class index: {pred_idx} ({CLASS_NAMES[pred_idx]})")
    print(f"  - Confidence: {probs[pred_idx]:.4f}")
    print("[PASS] Body Behaviour Model test passed completely!\n")

if __name__ == "__main__":
    main()
