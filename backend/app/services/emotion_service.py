"""
Facial Emotion Inference Service
--------------------------------
Wraps the trained EfficientNetB0 facial emotion classification model
(trained on RAF-DB in Untitled15.ipynb).

Pipeline:
  Face Crop (BGR from FaceDetectionService) -> Color conversion (BGR -> RGB)
  -> Resize to (224, 224) -> Cast to float32 [0.0, 255.0]
  -> EfficientNetB0 Model -> Softmax Emotion Probabilities (7 classes)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import cv2
import numpy as np
import tensorflow as tf
import keras

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Exact 7 emotion classes matching Untitled15.ipynb (RAF-DB 0-indexed targets)
EMOTION_CLASSES = [
    "Surprise",  # 0
    "Fear",      # 1
    "Disgust",   # 2
    "Happy",     # 3
    "Sad",       # 4
    "Angry",     # 5
    "Neutral",   # 6
]


class EmotionInferenceService:
    """
    Service for running inference with the trained EfficientNetB0 emotion model.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        root_dir = Path(__file__).resolve().parent.parent.parent.parent

        self.model_path = Path(
            model_path or root_dir / "weights" / "emotion" / "rafdb_emotion_best.keras"
        )

        if not self.model_path.exists():
            raise FileNotFoundError(f"Emotion model not found at: {self.model_path}")

        print(f"[EmotionService] Loading model from: {self.model_path.name}...")
        self.model = keras.models.load_model(str(self.model_path), compile=False)
        print("[EmotionService] Model loaded successfully.")

        # Warm-up pass to initialize graph and prevent first-call latency
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = self.model.predict(dummy, verbose=0)
        print("[EmotionService] Model warm-up complete.")

    def preprocess_crop(
        self, face_crop: Optional[np.ndarray], input_format: str = "BGR"
    ) -> Optional[np.ndarray]:
        """
        Preprocesses a raw face crop image to the exact format expected by EfficientNetB0:
          - Validates dimensions (minimum 5x5 pixels)
          - Converts color format to RGB
          - Resizes to (224, 224) pixels
          - Casts to float32 in range [0.0, 255.0] (no division by 255)
          - Expands batch dimension to (1, 224, 224, 3)
        """
        if face_crop is None or not isinstance(face_crop, np.ndarray) or face_crop.size == 0:
            return None

        if len(face_crop.shape) != 3 or face_crop.shape[2] != 3:
            return None

        h, w = face_crop.shape[:2]
        if h < 5 or w < 5:
            return None

        # 1. Color space conversion
        if input_format.upper() == "BGR":
            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        elif input_format.upper() == "RGB":
            rgb_crop = face_crop
        else:
            raise ValueError(f"Unsupported input_format '{input_format}'. Use 'BGR' or 'RGB'.")

        # 2. Resize to (224, 224) matching model input
        if (w, h) != (224, 224):
            interp = cv2.INTER_AREA if (w > 224 or h > 224) else cv2.INTER_LINEAR
            resized = cv2.resize(rgb_crop, (224, 224), interpolation=interp)
        else:
            resized = rgb_crop

        # 3. Cast to float32 range [0.0, 255.0] matching Untitled15.ipynb
        tensor = resized.astype(np.float32)

        # 4. Expand batch dimension: (1, 224, 224, 3)
        batch = np.expand_dims(tensor, axis=0)
        return batch

    def predict_emotion(
        self, face_crop: Optional[np.ndarray], input_format: str = "BGR"
    ) -> Optional[Dict[str, Any]]:
        """
        Runs emotion prediction on a single face crop.

        Returns:
            Dict with emotion label, class_id, confidence, and all class probabilities,
            or None if the crop is invalid or empty.
        """
        batch = self.preprocess_crop(face_crop, input_format=input_format)
        if batch is None:
            return None

        probs = self.model.predict(batch, verbose=0)[0]

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        prob_dict = {
            label: round(float(probs[i]), 4)
            for i, label in enumerate(EMOTION_CLASSES)
        }

        return {
            "emotion": EMOTION_CLASSES[class_id],
            "class_id": class_id,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
        }

    def predict_batch(
        self, face_crops: List[Optional[np.ndarray]], input_format: str = "BGR"
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Runs emotion prediction on a list of face crops.
        """
        return [self.predict_emotion(crop, input_format=input_format) for crop in face_crops]
