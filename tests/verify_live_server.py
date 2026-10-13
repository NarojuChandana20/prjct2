"""
Live Server HTTP & Video Streaming Verification Script
"""

import urllib.request
import json
import time

base_url = "http://127.0.0.1:8000"

# 1. Use sample video
print("=== 1. Testing Live Server Sample Video Ingestion ===")
req1 = urllib.request.Request(f"{base_url}/api/use-sample/test_person_tracking_35f.mp4", method="POST")
with urllib.request.urlopen(req1) as resp:
    sample_data = json.loads(resp.read().decode("utf-8"))
video_id = sample_data["video_id"]
print("Ingested Video ID:", video_id)
print("Preview URL:", sample_data["preview_url"])

# 2. Run analysis
print("\n=== 2. Testing Live Server Pipeline Execution ===")
start = time.time()
req2 = urllib.request.Request(f"{base_url}/api/analyze/{video_id}?max_frames=35", method="POST")
with urllib.request.urlopen(req2) as resp:
    analyze_data = json.loads(resp.read().decode("utf-8"))
elapsed = time.time() - start
print(f"Analysis completed in {elapsed:.2f}s!")
annotated_url = analyze_data["video_urls"]["annotated"]
print("Annotated Video URL:", annotated_url)
print("Total Frames Processed:", analyze_data["total_frames_processed"])
print("Statistics:", json.dumps(analyze_data["statistics"], indent=2))

# 3. Stream annotated video from live server
print("\n=== 3. Testing Live Video Streaming Endpoint ===")
req3 = urllib.request.Request(f"{base_url}{annotated_url}")
with urllib.request.urlopen(req3) as resp:
    video_bytes = resp.read()
    headers = dict(resp.headers)
print(f"Received {len(video_bytes):,} bytes from server.")
print("Content-Type:", headers.get("content-type") or headers.get("Content-Type"))
assert len(video_bytes) > 0, "Video content is empty"

# 4. Tracked people summary
print("\n=== 4. Tracked Individuals Telemetry ===")
for p in analyze_data["tracked_people"]:
    tid = p["track_id"]
    frames = p["total_frames_tracked"]
    b = p["latest_behaviour"]
    e = p["latest_emotion"]
    b_class = b.get("class_name")
    b_stat = b.get("status")
    b_risk = b.get("risk_level")
    e_label = e.get("label")
    e_stat = e.get("status")
    print(f"Track #{tid}: Frames={frames}, Behaviour={b_class} (status={b_stat}, risk={b_risk}), Emotion={e_label} (status={e_stat})")

print("\n=== ALL LIVE SERVER CHECKS PASSED ===")
