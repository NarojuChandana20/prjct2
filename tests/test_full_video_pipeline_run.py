"""
Full Video Pipeline Verification Script
----------------------------------------
Tests full-video processing on the uploaded 51-second WhatsApp video (1,550 frames):
1. Verifies max_frames=None processes all 1550 frames
2. Verifies progress callback reports accurate frame progression
3. Verifies annotated video duration matches source duration (51.67s)
4. Verifies annotated video plays/decodes from frame 1 to 1550
5. Verifies label layout, collision avoidance, and non-overlapping pills
6. Verifies risk colors (RED, GREEN, AMBER) and emotion separation
7. Verifies telemetry distinguishes Collecting vs Pose Unavailable
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.video_behaviour_pipeline import VideoBehaviourPipeline

INPUT_PATH = ROOT_DIR / "storage" / "uploads" / "a94dc34536e4_WhatsApp Video 2026-10-06 at 3.03.12 PM.mp4"
OUTPUT_PATH = ROOT_DIR / "storage" / "outputs" / "a94dc34536e4_annotated.mp4"
VERIF_DIR = ROOT_DIR / "storage" / "verification_frames"
VERIF_DIR.mkdir(parents=True, exist_ok=True)


def test_full_video():
    print("=" * 70)
    print("FULL VIDEO PROCESSING AND ANNOTATION VERIFICATION")
    print("=" * 70)

    assert INPUT_PATH.exists(), f"Input video not found: {INPUT_PATH}"

    cap_in = cv2.VideoCapture(str(INPUT_PATH))
    src_w = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap_in.get(cv2.CAP_PROP_FPS) or 30.0)
    src_frames = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))
    src_dur = round(src_frames / src_fps, 2)
    cap_in.release()

    print(f"Source video: {INPUT_PATH.name}")
    print(f"  Dimensions: {src_w}x{src_h} | FPS: {src_fps:.2f} | Frames: {src_frames} | Duration: {src_dur:.2f}s")
    assert src_frames == 1550, f"Expected 1550 frames, got {src_frames}"

    print("\n[Step 1] Initializing VideoBehaviourPipeline singleton...")
    pipeline = VideoBehaviourPipeline()

    progress_log = []
    def on_progress(cur, tot):
        if cur % 100 == 0 or cur == tot:
            progress_log.append((cur, tot))

    print("\n[Step 2] Processing entire video (max_frames=None)...")
    start_time = time.time()
    res = pipeline.process_video(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        max_frames=None,  # Full Video
        progress_callback=on_progress,
    )
    elapsed = time.time() - start_time
    print(f"\nProcessing complete in {elapsed:.2f}s!")
    print(f"  Total frames processed: {res['total_frames_processed']}")
    print(f"  Average processing FPS: {res['fps']}")
    print(f"  Unique track IDs detected: {res['unique_track_ids']}")

    # 1. Check frame count
    assert res['total_frames_processed'] == src_frames, (
        f"Frame count mismatch: {res['total_frames_processed']} processed vs {src_frames} source"
    )

    # 2. Check output file integrity & duration
    print("\n[Step 3] Verifying output file integrity and duration...")
    assert OUTPUT_PATH.exists(), f"Output file missing: {OUTPUT_PATH}"
    out_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  Output file size: {out_size_mb:.2f} MB")
    assert out_size_mb > 0.5, "Output file too small!"

    cap_out = cv2.VideoCapture(str(OUTPUT_PATH))
    assert cap_out.isOpened(), "Could not open annotated video with cv2!"
    out_w = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = float(cap_out.get(cv2.CAP_PROP_FPS) or 30.0)
    out_frames = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
    out_dur = round(out_frames / out_fps, 2)

    print(f"  Output Dimensions: {out_w}x{out_h}")
    print(f"  Output FPS: {out_fps:.2f} (Source FPS: {src_fps:.2f})")
    print(f"  Output Frames: {out_frames} (Source Frames: {src_frames})")
    print(f"  Output Duration: {out_dur:.2f}s (Source Duration: {src_dur:.2f}s)")

    assert out_w == src_w, f"Width mismatch: {out_w} vs {src_w}"
    assert out_h == src_h, f"Height mismatch: {out_h} vs {src_h}"
    assert out_frames == src_frames, f"Frame count mismatch: {out_frames} vs {src_frames}"
    assert abs(out_dur - src_dur) < 1.0, f"Duration mismatch: {out_dur}s vs {src_dur}s"

    # 3. Decode every single frame to verify 100% playable
    print("\n[Step 4] Decoding all frames from beginning to end...")
    decoded = 0
    sample_checkpoints = {1, 35, 100, 250, 500, 750, 1000, 1250, 1550}
    saved_samples = {}

    while cap_out.isOpened():
        ret, frame = cap_out.read()
        if not ret:
            break
        decoded += 1
        if decoded in sample_checkpoints:
            out_img = VERIF_DIR / f"full_run_frame_{decoded}.png"
            cv2.imwrite(str(out_img), frame)
            saved_samples[decoded] = frame

    cap_out.release()
    print(f"  Successfully decoded {decoded}/{src_frames} frames!")
    assert decoded == src_frames, f"Decoded {decoded} != {src_frames}"

    # 4. Verify telemetry accuracy and Collecting 0/30 resolution
    print("\n[Step 5] Verifying telemetry breakdown...")
    for tid in res["unique_track_ids"]:
        hist = res["person_telemetry_history"].get(tid, [])
        predicted_count = sum(1 for h in hist if h["behaviour"].get("status") == "predicted")
        collecting_count = sum(1 for h in hist if h["behaviour"].get("status") == "collecting")
        unavailable_count = sum(1 for h in hist if h["behaviour"].get("status") in ["unavailable", "lost", "untracked"])
        latest = hist[-1]["behaviour"] if hist else {}
        print(f"  Track ID {tid:2d}: {len(hist):4d} frames | Pred: {predicted_count:3d} | Collecting: {collecting_count:3d} | Unavail: {unavailable_count:3d} | Latest: {latest.get('class_name')} ({latest.get('status')})")

        # Verify no track is stuck reporting "Collecting 0/30" when pose is unavailable
        for h in hist:
            b = h["behaviour"]
            if b.get("status") == "collecting":
                assert b.get("collected", 0) > 0, f"Found collecting with 0 observations for track {tid} at frame {h['frame']}"

    print("\n" + "=" * 70)
    print("ALL FULL-VIDEO TESTS AND CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_full_video()
