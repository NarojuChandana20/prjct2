import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage

video_p = "storage/uploads/a94dc34536e4_WhatsApp Video 2026-10-06 at 3.03.12 PM.mp4"
pose_model_p = "weights/pose/pose_landmarker_lite.task"
yolo_p = "weights/yolo/yolo11n.pt"

yolo = YOLO(yolo_p)
base_opts = python.BaseOptions(model_asset_path=pose_model_p)
opts = vision.PoseLandmarkerOptions(
    base_options=base_opts,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.45,
    min_pose_presence_confidence=0.45,
)
landmarker = vision.PoseLandmarker.create_from_options(opts)

cap = cv2.VideoCapture(video_p)
track_stats = {}
fidx = 0

while cap.isOpened() and fidx < 200:
    ret, frame = cap.read()
    if not ret:
        break
    fidx += 1
    h, w = frame.shape[:2]
    res = yolo.track(frame, persist=True, classes=[0], conf=0.35, iou=0.50, verbose=False)
    if res and len(res) > 0 and res[0].boxes is not None and res[0].boxes.id is not None:
        tids = res[0].boxes.id.int().cpu().tolist()
        boxes = res[0].boxes.xyxy.cpu().numpy()
        for tid, box in zip(tids, boxes):
            x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, int(box[3]))
            bw, bh = x2 - x1, y2 - y1
            if bw < 10 or bh < 10:
                continue

            # Crop with 8% padding
            pad_w, pad_h = int(bw * 0.08), int(bh * 0.08)
            crop = frame[max(0, y1 - pad_h):min(h, y2 + pad_h), max(0, x1 - pad_w):min(w, x2 + pad_w)]

            # Pose detection
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_img = MPImage(image_format=mp.ImageFormat.SRGB, data=rgb)
            pose_res = landmarker.detect(mp_img)
            pose_found = bool(pose_res.pose_landmarks)

            if tid not in track_stats:
                track_stats[tid] = {"total_frames": 0, "pose_ok": 0, "pose_fail": 0, "sizes": []}
            track_stats[tid]["total_frames"] += 1
            track_stats[tid]["sizes"].append((bw, bh))
            if pose_found:
                track_stats[tid]["pose_ok"] += 1
            else:
                track_stats[tid]["pose_fail"] += 1

cap.release()
print(f"Track Statistics on WhatsApp Video ({fidx} frames processed):")
for tid, s in sorted(track_stats.items()):
    avg_w = np.mean([sz[0] for sz in s["sizes"]])
    avg_h = np.mean([sz[1] for sz in s["sizes"]])
    print(f"Track ID {tid:2d}: Frames: {s['total_frames']:3d} | Pose OK: {s['pose_ok']:3d} | Pose FAIL: {s['pose_fail']:3d} | Avg Crop: {avg_w:.1f}x{avg_h:.1f}")
