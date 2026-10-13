"""
Behaviour Tracker - Phase 1 (Rich Features Version) - Improved
-------------------------------------------------------------
Features per frame: 165
- 99  : Normalized landmarks (x, y, z)
- 33  : Joint distances
- 33  : Joint velocities

Input shape for model: (30, 165)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from collections import deque
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage
import tensorflow as tf


CLASS_NAMES = {
    0: "NonFight / Normal",
    1: "Fight",
    2: "Fall",
}

RISK_LEVEL = {
    0: "Low",
    1: "High",
    2: "High",
}


def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)
    return loss


class BehaviourTracker:
    def __init__(
        self,
        model_path: str | Path,
        pose_model_path: str | Path,
        sequence_length: int = 30,
        confidence_threshold: float = 0.55,
        smooth_window: int = 7,
        max_fail_frames: int = 20,          # how many consecutive bad frames before clearing buffer
    ):
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.smooth_window = smooth_window
        self.max_fail_frames = max_fail_frames

        # Load model
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Behaviour model not found: {model_path}")

        self.model = tf.keras.models.load_model(
            str(model_path),
            custom_objects={"loss": focal_loss(gamma=2.0, alpha=0.25)}
        )
        print(f"[BehaviourTracker] Model loaded → {model_path.name}")

        # MediaPipe Pose
        base_options = python.BaseOptions(model_asset_path=str(pose_model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.45,   # slightly lower for better recall during fights
            min_pose_presence_confidence=0.45,
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        print("[BehaviourTracker] MediaPipe Pose Landmarker ready")

        # Buffers
        self.buffers: Dict[int, deque] = {}
        self.prev_coords: Dict[int, np.ndarray] = {}
        self.prediction_history: Dict[int, deque] = {}
        self.fail_count: Dict[int, int] = {}
        self.last_good_pred: Dict[int, Dict] = {}   # keep last good result for a few frames

    def _extract_features(self, frame: np.ndarray, track_id: int) -> Optional[np.ndarray]:
        """
        Returns 165 features or None
        """
        if frame is None or frame.size == 0:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.pose_landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)  # (33, 3)

        # ---------- Quality check (much more tolerant) ----------
        if np.mean(np.abs(coords[:, :2])) < 0.005:          # was 0.02 → now 0.005
            return None

        # ---------- Normalization ----------
        left_hip = coords[23]
        right_hip = coords[24]
        hip_center = (left_hip + right_hip) / 2.0
        coords = coords - hip_center

        left_shoulder = coords[11]
        right_shoulder = coords[12]
        shoulder_dist = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
        if shoulder_dist < 1e-5:
            shoulder_dist = 1.0
        coords = coords / shoulder_dist

        # 1. Landmark positions (99)
        pos_features = coords.flatten()

        # 2. Joint distances (33)
        distances = []
        for i in range(32):
            dist = np.linalg.norm(coords[i] - coords[i + 1])
            distances.append(dist)
        distances.append(0.0)  # pad to 33
        dist_features = np.array(distances, dtype=np.float32)

        # 3. Joint velocities (33)
        if track_id in self.prev_coords:
            velocity = np.linalg.norm(coords - self.prev_coords[track_id], axis=1)
        else:
            velocity = np.zeros(33, dtype=np.float32)

        self.prev_coords[track_id] = coords.copy()

        # Combine → 165 features
        features = np.concatenate([pos_features, dist_features, velocity])
        return features

    def update(self, track_id: int, person_crop: np.ndarray) -> Optional[Dict]:
        if person_crop is None or person_crop.size == 0:
            return self._handle_failed_frame(track_id)

        features = self._extract_features(person_crop, track_id)

        # Initialize buffers if new track
        if track_id not in self.buffers:
            self.buffers[track_id] = deque(maxlen=self.sequence_length)
            self.prediction_history[track_id] = deque(maxlen=self.smooth_window)
            self.fail_count[track_id] = 0

        if features is None:
            return self._handle_failed_frame(track_id)

        # Good frame → reset fail counter and add features
        self.fail_count[track_id] = 0
        self.buffers[track_id].append(features)

        # Still collecting
        if len(self.buffers[track_id]) < self.sequence_length:
            collected = len(self.buffers[track_id])
            return {
                "status": "collecting",
                "class_id": -1,
                "class_name": f"Collecting {collected}/{self.sequence_length}",
                "risk_level": "—",
                "confidence": 0.0,
                "probabilities": {"NonFight": 0.0, "Fight": 0.0, "Fall": 0.0},
                "collected": collected,
            }

        # Ready for prediction
        sequence = np.stack(list(self.buffers[track_id]), axis=0)  # (30, 165)
        raw_pred = self._predict(sequence)

        self.prediction_history[track_id].append(raw_pred["class_id"])
        smoothed = self._get_smoothed_prediction(track_id, raw_pred)

        # Save as last good prediction
        self.last_good_pred[track_id] = smoothed
        return smoothed

    def _handle_failed_frame(self, track_id: int) -> Optional[Dict]:
        """Called when pose detection fails."""
        if track_id not in self.buffers:
            return None

        self.fail_count[track_id] = self.fail_count.get(track_id, 0) + 1
        collected = len(self.buffers[track_id])

        # Too many consecutive failures → clear buffer
        if self.fail_count[track_id] > self.max_fail_frames:
            self.buffers[track_id].clear()
            self.fail_count[track_id] = 0
            self.last_good_pred.pop(track_id, None)
            return {
                "status": "unavailable",
                "class_id": -1,
                "class_name": "Pose Unavailable",
                "risk_level": "—",
                "confidence": 0.0,
                "probabilities": {"NonFight": 0.0, "Fight": 0.0, "Fall": 0.0},
                "collected": 0,
            }

        # If buffer is empty (pose has never succeeded or buffer was cleared), state is unavailable
        if collected == 0:
            return {
                "status": "unavailable",
                "class_id": -1,
                "class_name": "Pose Unavailable",
                "risk_level": "—",
                "confidence": 0.0,
                "probabilities": {"NonFight": 0.0, "Fight": 0.0, "Fall": 0.0},
                "collected": 0,
            }

        # Genuine collection in progress
        if collected < self.sequence_length:
            return {
                "status": "collecting",
                "class_id": -1,
                "class_name": f"Collecting {collected}/{self.sequence_length}",
                "risk_level": "—",
                "confidence": 0.0,
                "probabilities": {"NonFight": 0.0, "Fight": 0.0, "Fall": 0.0},
                "collected": collected,
            }

        # We already have a full sequence → return last good prediction during temporary pose dropouts
        if track_id in self.last_good_pred:
            return self.last_good_pred[track_id]

        return {
            "status": "unavailable",
            "class_id": -1,
            "class_name": "Pose Unavailable",
            "risk_level": "—",
            "confidence": 0.0,
            "probabilities": {"NonFight": 0.0, "Fight": 0.0, "Fall": 0.0},
            "collected": 0,
        }

    def _predict(self, sequence: np.ndarray) -> Dict:
        x = np.expand_dims(sequence, axis=0).astype(np.float32)
        probs = self.model.predict(x, verbose=0)[0]

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        # Force NonFight if confidence is low (except for Fall which is usually high conf)
        if confidence < self.confidence_threshold and class_id != 2:
            class_id = 0
            confidence = float(probs[0])

        return {
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "risk_level": RISK_LEVEL[class_id],
            "confidence": round(confidence, 4),
            "probabilities": {
                "NonFight": round(float(probs[0]), 4),
                "Fight": round(float(probs[1]), 4),
                "Fall": round(float(probs[2]), 4),
            },
        }

    def _get_smoothed_prediction(self, track_id: int, raw_pred: Optional[Dict] = None) -> Dict:
        history = list(self.prediction_history.get(track_id, []))
        if not history:
            return raw_pred

        counts = np.bincount(history, minlength=3)
        smoothed_id = int(np.argmax(counts))

        if raw_pred is not None:
            result = raw_pred.copy()
            result["class_id"] = smoothed_id
            result["class_name"] = CLASS_NAMES[smoothed_id]
            result["risk_level"] = RISK_LEVEL[smoothed_id]
            return result

        return {
            "class_id": smoothed_id,
            "class_name": CLASS_NAMES[smoothed_id],
            "risk_level": RISK_LEVEL[smoothed_id],
            "confidence": 0.0,
            "probabilities": {"NonFight": 0.0, "Fight": 0.0, "Fall": 0.0},
        }

    def reset_track(self, track_id: int):
        for d in [self.buffers, self.prediction_history, self.prev_coords, self.fail_count, self.last_good_pred]:
            if track_id in d:
                del d[track_id]

    def reset_all(self):
        self.buffers.clear()
        self.prediction_history.clear()
        self.prev_coords.clear()
        self.fail_count.clear()
        self.last_good_pred.clear()