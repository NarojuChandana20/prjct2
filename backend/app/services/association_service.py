"""
Face-to-Person ID Association Service
-------------------------------------
Associates detected face bounding boxes and facial emotion predictions with
the corresponding YOLO11 tracked person IDs using geometric constraints.

Key Principles:
  1. Spatial Matching: Uses face containment in the upper portion (e.g. top 40%) of the person bounding box.
  2. Conflict Resolution: Ensures a face is assigned to AT MOST ONE person, and a person receives AT MOST ONE face.
  3. Ambiguity Handling: If multiple people compete for a face with near-equal score, the match is rejected as ambiguous.
  4. Uncertainty Fallback: If no face is visible, occluded, or ambiguous, emotion is marked 'Unavailable' with 0.0 confidence.
  5. Separation of Outputs: Maintains independent body behaviour and facial emotion outputs (no composite risk score).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class FacePersonAssociationService:
    """
    Associates face detections with YOLO person tracks geometrically.
    """

    def __init__(
        self,
        upper_body_ratio: float = 0.40,
        min_containment: float = 0.35,
        min_match_score: float = 0.40,
        ambiguity_margin: float = 0.15,
    ):
        """
        Parameters:
            upper_body_ratio: Fraction of person height defining the head/face zone (default 0.40).
            min_containment: Minimum fraction of face bbox that must intersect the head zone (default 0.35).
            min_match_score: Minimum geometric score to accept an association (default 0.40).
            ambiguity_margin: Score delta between top-2 candidates below which match is rejected as ambiguous (default 0.15).
        """
        self.upper_body_ratio = upper_body_ratio
        self.min_containment = min_containment
        self.min_match_score = min_match_score
        self.ambiguity_margin = ambiguity_margin

    def compute_match_score(
        self, face_bbox: List[int], person_bbox: List[int]
    ) -> float:
        """
        Calculates a geometric compatibility score in [0.0, 1.0] between a face box and a person box.
        """
        fx1, fy1, fx2, fy2 = face_bbox
        px1, py1, px2, py2 = person_bbox

        fw = max(0, fx2 - fx1)
        fh = max(0, fy2 - fy1)
        pw = max(0, px2 - px1)
        ph = max(0, py2 - py1)

        if fw <= 0 or fh <= 0 or pw <= 0 or ph <= 0:
            return 0.0

        face_area = fw * fh

        # Define person head/upper zone (top upper_body_ratio of person box)
        head_zone_y2 = py1 + int(ph * self.upper_body_ratio)

        # Intersection between face and person head zone
        ix1 = max(fx1, px1)
        iy1 = max(fy1, py1)
        ix2 = min(fx2, px2)
        iy2 = min(fy2, head_zone_y2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter_area = iw * ih

        containment = inter_area / float(face_area)

        # Face center
        fc_x = (fx1 + fx2) / 2.0
        fc_y = (fy1 + fy2) / 2.0

        center_in_head_zone = (px1 <= fc_x <= px2) and (py1 <= fc_y <= head_zone_y2)

        # Reject if face is not in the head zone and containment is negligible
        if containment < self.min_containment and not center_in_head_zone:
            return 0.0

        # Horizontal alignment: penalty for face being offset from person center
        person_center_x = (px1 + px2) / 2.0
        norm_h_dist = abs(fc_x - person_center_x) / float(pw)
        alignment_score = max(0.0, 1.0 - (norm_h_dist * 2.0))

        # Vertical score: prefer faces near the top of the person bbox
        norm_v_pos = max(0.0, min(1.0, (fc_y - py1) / float(ph * self.upper_body_ratio)))
        v_score = 1.0 - (norm_v_pos * 0.5)

        # Composite geometric score
        score = (containment * 0.55) + (alignment_score * 0.25) + (v_score * 0.20)
        return float(np.clip(score, 0.0, 1.0))

    def associate_frame(
        self,
        person_tracks: List[Dict[str, Any]],
        face_detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Performs geometric matching between detected faces and tracked persons for a single frame.

        Parameters:
            person_tracks: List of tracked persons, each containing:
                - 'track_id': int (or valid ID)
                - 'bbox': [px1, py1, px2, py2]
                - 'det_confidence': float
                - (optional) 'behaviour': dict
            face_detections: List of detected faces, each containing:
                - 'bbox': [fx1, fy1, fx2, fy2]
                - 'confidence': float
                - (optional) 'emotion': dict

        Returns:
            List of combined person records preserving separate body and emotion outputs:
              [
                {
                  "track_id": int,
                  "person_bbox": [x1, y1, x2, y2],
                  "det_confidence": float,
                  "behaviour": { ... },
                  "emotion": {
                      "status": "available" | "unavailable" | "ambiguous",
                      "label": str ("Surprise", "Happy", ... or "Unavailable"),
                      "confidence": float,
                      "class_id": Optional[int],
                      "probabilities": Optional[dict],
                      "face_bbox": Optional[List[int]],
                  }
                },
                ...
              ]
        """
        # Default unavailable emotion structure
        def make_unavailable_emotion(status: str = "unavailable", reason: str = "no_face") -> Dict[str, Any]:
            return {
                "status": status,
                "label": "Unavailable",
                "confidence": 0.0,
                "class_id": None,
                "probabilities": None,
                "face_bbox": None,
                "reason": reason,
            }

        # 1. Validate inputs
        if not person_tracks:
            return []

        # Filter out person tracks with missing or invalid track_ids or degenerate boxes
        valid_persons = []
        for p in person_tracks:
            tid = p.get("track_id")
            bbox = p.get("bbox")
            if tid is not None and bbox and len(bbox) == 4:
                px1, py1, px2, py2 = bbox
                if px2 > px1 and py2 > py1:
                    valid_persons.append(p)

        if not valid_persons:
            return []

        # If no face detections present, return all persons with Unavailable emotion
        if not face_detections:
            results = []
            for p in valid_persons:
                rec = {
                    "track_id": p["track_id"],
                    "person_bbox": p["bbox"],
                    "det_confidence": p.get("det_confidence", 0.0),
                    "behaviour": p.get("behaviour", {
                        "status": "unknown", "class_id": -1, "class_name": "Unknown", "confidence": 0.0
                    }),
                    "emotion": make_unavailable_emotion(status="unavailable", reason="no_face_detected"),
                }
                results.append(rec)
            return results

        # 2. Build Score Matrix [Num_Faces x Num_Persons]
        num_faces = len(face_detections)
        num_persons = len(valid_persons)
        score_matrix = np.zeros((num_faces, num_persons), dtype=np.float32)

        for f_idx, face in enumerate(face_detections):
            f_box = face.get("bbox")
            if not f_box or len(f_box) != 4:
                continue
            for p_idx, person in enumerate(valid_persons):
                score_matrix[f_idx, p_idx] = self.compute_match_score(f_box, person["bbox"])

        # 3. Detect Ambiguous Face Matches
        # If a face has multiple person candidates with high scores and delta < ambiguity_margin, reject match
        face_is_ambiguous = [False] * num_faces
        for f_idx in range(num_faces):
            scores = score_matrix[f_idx]
            qualifying = np.where(scores >= self.min_match_score)[0]
            if len(qualifying) > 1:
                sorted_scores = np.sort(scores[qualifying])[::-1]
                if (sorted_scores[0] - sorted_scores[1]) < self.ambiguity_margin:
                    face_is_ambiguous[f_idx] = True

        # 4. Greedy 1-to-1 Association
        # Pair highest scoring (face, person) matches first
        assigned_faces = set()
        person_to_face: Dict[int, int] = {}  # person_idx -> face_idx

        # Sort all possible (face, person) pairs by score descending
        pair_candidates = []
        for f_idx in range(num_faces):
            if face_is_ambiguous[f_idx]:
                continue
            for p_idx in range(num_persons):
                sc = score_matrix[f_idx, p_idx]
                if sc >= self.min_match_score:
                    pair_candidates.append((sc, f_idx, p_idx))

        pair_candidates.sort(key=lambda x: x[0], reverse=True)

        for sc, f_idx, p_idx in pair_candidates:
            if f_idx not in assigned_faces and p_idx not in person_to_face:
                person_to_face[p_idx] = f_idx
                assigned_faces.add(f_idx)

        # 5. Assemble Integrated Records
        output_records = []
        for p_idx, person in enumerate(valid_persons):
            tid = person["track_id"]
            p_bbox = person["bbox"]
            p_conf = person.get("det_confidence", 0.0)
            behaviour_data = person.get("behaviour", {
                "status": "unknown", "class_id": -1, "class_name": "Unknown", "confidence": 0.0
            })

            if p_idx in person_to_face:
                f_idx = person_to_face[p_idx]
                face = face_detections[f_idx]
                face_bbox = face.get("bbox")

                # Extract emotion data from face if available
                emotion_data = face.get("emotion")
                if emotion_data and emotion_data.get("emotion"):
                    emotion_record = {
                        "status": "available",
                        "label": emotion_data.get("emotion"),
                        "confidence": emotion_data.get("confidence", 0.0),
                        "class_id": emotion_data.get("class_id"),
                        "probabilities": emotion_data.get("probabilities"),
                        "face_bbox": face_bbox,
                        "reason": "matched",
                    }
                else:
                    # Face matched, but emotion inference was not run or failed
                    emotion_record = make_unavailable_emotion(
                        status="unavailable", reason="emotion_inference_failed"
                    )
                    emotion_record["face_bbox"] = face_bbox
            else:
                # Check if unassigned due to ambiguity in nearby faces
                # Check if any face near this person was marked ambiguous
                nearby_ambiguous = any(
                    face_is_ambiguous[f] and score_matrix[f, p_idx] >= self.min_match_score
                    for f in range(num_faces)
                )
                if nearby_ambiguous:
                    emotion_record = make_unavailable_emotion(status="ambiguous", reason="ambiguous_face_candidates")
                else:
                    emotion_record = make_unavailable_emotion(status="unavailable", reason="no_face_matched")

            output_records.append({
                "track_id": tid,
                "person_bbox": p_bbox,
                "det_confidence": p_conf,
                "behaviour": behaviour_data,
                "emotion": emotion_record,
            })

        return output_records
