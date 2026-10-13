"""
Probe tracks and pose detection on WhatsApp video
"""

from pathlib import Path
import cv2
from ultralytics import YOLO

p = Path("storage/uploads/a94dc34536e4_WhatsApp Video 2026-10-06 at 3.03.12 PM.mp4")
yolo = YOLO("weights/yolo/yolo11n.pt")
cap = cv2.VideoCapture(str(p))

frame_count = 0
track_counts = {}
while cap.isOpened() and frame_count < 250:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    res = yolo.track(frame, persist=True, classes=[0], conf=0.35, iou=0.50, verbose=False)
    if res and len(res) > 0 and res[0].boxes is not None and res[0].boxes.id is not None:
        tids = res[0].boxes.id.int().cpu().tolist()
        boxes = res[0].boxes.xyxy.cpu().numpy()
        for tid, box in zip(tids, boxes):
            wb = box[2] - box[0]
            hb = box[3] - box[1]
            if tid not in track_counts:
                track_counts[tid] = {"frames": 0, "min_w": wb, "max_w": wb, "min_h": hb, "max_h": hb}
            track_counts[tid]["frames"] += 1
            track_counts[tid]["min_w"] = min(track_counts[tid]["min_w"], wb)
            track_counts[tid]["max_w"] = max(track_counts[tid]["max_w"], wb)
            track_counts[tid]["min_h"] = min(track_counts[tid]["min_h"], hb)
            track_counts[tid]["max_h"] = max(track_counts[tid]["max_h"], hb)

cap.release()
print(f"Processed {frame_count} frames. Track statistics:")
for tid, info in sorted(track_counts.items(), key=lambda x: -x[1]["frames"]):
    fc = info["frames"]
    w_range = f"{info['min_w']:.0f}-{info['max_w']:.0f}"
    h_range = f"{info['min_h']:.0f}-{info['max_h']:.0f}"
    print(f"  Track ID {tid:2d}: {fc:3d} frames | Width: {w_range:7s} | Height: {h_range:7s}")
