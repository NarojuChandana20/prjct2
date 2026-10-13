"""
Detailed Pipeline & Frame Content Verification Script
"""

from __future__ import annotations

import sys
from pathlib import Path
import cv2
import av
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.video_behaviour_pipeline import VideoBehaviourPipeline

video_path = ROOT_DIR / "tests" / "assets" / "test_person_tracking_35f.mp4"
out_video = ROOT_DIR / "storage" / "outputs" / "deep_verification_35f.mp4"
out_video.parent.mkdir(parents=True, exist_ok=True)

pipeline = VideoBehaviourPipeline()
res = pipeline.process_video(video_path, out_video, max_frames=35)

print("\n=== 1. Pipeline Processing Verification ===")
print("Frames processed:", res["total_frames_processed"])
print("Output video path:", res["output_file"])
file_size = Path(res["output_file"]).stat().st_size
print(f"Output file size: {file_size:,} bytes")
assert file_size > 0, "Output file is empty"

# 2. Check decodability with PyAV
c = av.open(str(out_video))
v_streams = [s for s in c.streams if s.type == "video"]
vs = v_streams[0]
print("\n=== 2. PyAV Codec & Stream Verification ===")
print("Stream Codec:", vs.codec_context.name)
print("Pixel format:", vs.codec_context.pix_fmt)
print("Dimensions:", f"{vs.width}x{vs.height}")
print("FPS:", float(vs.average_rate))
c.close()
assert vs.codec_context.name == "h264", f"Expected h264, got {vs.codec_context.name}"
assert vs.codec_context.pix_fmt == "yuv420p", f"Expected yuv420p, got {vs.codec_context.pix_fmt}"

# 3. Read frames and inspect visual bounding boxes
cap = cv2.VideoCapture(str(out_video))
frames = []
while cap.isOpened():
    ret, f = cap.read()
    if not ret:
        break
    frames.append(f)
cap.release()
print("\n=== 3. OpenCV Frame Read Verification ===")
print("Total frames read by cv2:", len(frames))
assert len(frames) == 35, f"Expected 35 frames, got {len(frames)}"

# Save sample frames for inspection
out_dir = ROOT_DIR / "storage" / "verification_frames"
out_dir.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_dir / "frame_05_collecting.png"), frames[4])
cv2.imwrite(str(out_dir / "frame_20_progress.png"), frames[19])
cv2.imwrite(str(out_dir / "frame_32_predicted.png"), frames[31])
print("Saved sample frames to storage/verification_frames/:")
print("  - frame_05_collecting.png")
print("  - frame_20_progress.png")
print("  - frame_32_predicted.png")

# Inspect frame 32 colors: Track 2 and 3 should have GREEN boxes [0, 220, 0]
f32 = frames[31]
h32 = res["person_telemetry_history"]
# Check bounding box pixels for Track 2 on frame 32
t2_rec = [u for u in h32[2] if u["frame"] == 32][0]
t2_bbox = t2_rec["bbox"] # [x1, y1, x2, y2]
t2_color_sample = f32[t2_bbox[1] + 10, t2_bbox[0]] # along left vertical edge
print(f"Track 2 (Frame 32, Predicted) bounding box edge color: BGR={t2_color_sample.tolist()}")
assert np.allclose(t2_color_sample, [0, 220, 0], atol=15), f"Expected GREEN [0, 220, 0], got {t2_color_sample}"

# Check bounding box pixels for Track 1 on frame 5 (Collecting)
f5 = frames[4]
t1_rec_f5 = [u for u in h32[1] if u["frame"] == 5][0]
t1_bbox_f5 = t1_rec_f5["bbox"]
t1_color_sample = f5[t1_bbox_f5[1] + 10, t1_bbox_f5[0]]
print(f"Track 1 (Frame 5, Collecting) bounding box edge color: BGR={t1_color_sample.tolist()}")
assert np.allclose(t1_color_sample, [0, 200, 255], atol=25), f"Expected AMBER [0, 200, 255], got {t1_color_sample}"

# 4. Detailed Track Buffer Analysis
print("\n=== 4. Detailed Track Buffer Progression (Part D Analysis) ===")
history = res["person_telemetry_history"]
for tid in sorted(res["unique_track_ids"]):
    updates = history[tid]
    print(f"\nTrack ID {tid} (Total frames: {len(updates)}):")
    sample_indices = [0, 9, 19, 29, len(updates) - 1]
    for idx in sample_indices:
        u = updates[idx]
        f_idx = u["frame"]
        beh = u["behaviour"]
        st = beh.get("status")
        cname = beh.get("class_name")
        coll = beh.get("collected", "N/A")
        conf = beh.get("confidence", 0.0)
        risk = beh.get("risk_level", "—")
        print(f"  Frame {f_idx:2d}: status={st:<10} class={cname:<22} coll={coll:<5} conf={conf:.4f} risk={risk}")

print("\n=== VERIFICATION FINISHED SUCCESSFULLY ===")
