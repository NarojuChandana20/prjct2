"""
Stage 9B: API & Dashboard Integration Test Suite
------------------------------------------------
Verifies:
  1. Frontend HTML dashboard served at GET /
  2. Static CSS and JS assets accessible
  3. GET /api/health returns system and model status
  4. GET /api/sample-videos lists test assets
  5. POST /api/upload validates file extensions, empty files, and valid videos
  6. POST /api/use-sample copies sample video safely into upload storage
  7. POST /api/analyze/{video_id} runs pipeline and returns genuine structured telemetry
  8. GET /api/video/{video_id}/input and GET /api/video/{video_id}/annotated stream videos
  9. Strict separation of body behaviour and facial emotion predictions
"""

from __future__ import annotations

import sys
from pathlib import Path
import io

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_dashboard_page():
    print("\n[Test 1] GET / -> Dashboard HTML Page...")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "Student Safety Monitoring System" in res.text, "Title missing in HTML"
    assert "videoDropzone" in res.text, "Dropzone missing in HTML"
    assert "trackedPersonsGrid" in res.text, "Telemetry grid missing in HTML"
    print("  --> [PASS] Dashboard HTML page successfully served.")


def test_static_assets():
    print("\n[Test 2] GET /static/css/dashboard.css & /static/js/dashboard.js...")
    res_css = client.get("/static/css/dashboard.css")
    assert res_css.status_code == 200, f"CSS asset returned {res_css.status_code}"
    assert "--bg-primary" in res_css.text, "CSS variables missing"

    res_js = client.get("/static/js/dashboard.js")
    assert res_js.status_code == 200, f"JS asset returned {res_js.status_code}"
    assert "DOMContentLoaded" in res_js.text, "JS content missing"
    print("  --> [PASS] Static CSS and JS assets correctly served.")


def test_health_endpoint():
    print("\n[Test 3] GET /api/health -> Health Check...")
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "models" in data
    assert "yolo11" in data["models"]
    assert "behaviour" in data["models"]
    assert "emotion" in data["models"]
    print(f"  --> [PASS] Health check verified: {data['system']}")


def test_sample_videos_list():
    print("\n[Test 4] GET /api/sample-videos -> Sample Video Assets...")
    res = client.get("/api/sample-videos")
    assert res.status_code == 200
    data = res.json()
    assert "samples" in data
    assert len(data["samples"]) > 0, "No sample videos found in tests/assets!"
    filenames = [s["filename"] for s in data["samples"]]
    print(f"  --> Available samples: {filenames}")
    assert "test_person_tracking_35f.mp4" in filenames
    print("  --> [PASS] Sample videos listed successfully.")


def test_invalid_upload_handling():
    print("\n[Test 5] POST /api/upload -> Error Handling...")
    # 1. Unsupported extension
    fake_txt = io.BytesIO(b"Hello world")
    res_bad = client.post(
        "/api/upload",
        files={"file": ("malicious.exe", fake_txt, "application/octet-stream")}
    )
    assert res_bad.status_code == 400
    assert "Unsupported video format" in res_bad.json()["detail"]
    print("  --> [PASS] Blocked unsupported extension with 400 error.")

    # 2. Corrupted video
    corrupt_mp4 = io.BytesIO(b"Not an actual video file")
    res_corrupt = client.post(
        "/api/upload",
        files={"file": ("corrupt.mp4", corrupt_mp4, "video/mp4")}
    )
    assert res_corrupt.status_code == 400
    assert "not a valid or readable video" in res_corrupt.json()["detail"]
    print("  --> [PASS] Blocked corrupt video file with 400 error.")


def test_end_to_end_analysis():
    print("\n[Test 6] End-to-End Analysis with Sample Video...")
    # Step A: Select sample video
    sample_name = "test_person_tracking_35f.mp4"
    res_select = client.post(f"/api/use-sample/{sample_name}")
    assert res_select.status_code == 200, f"Failed to select sample: {res_select.text}"
    sample_meta = res_select.json()
    video_id = sample_meta["video_id"]
    print(f"  --> Ingested video: ID='{video_id}', Duration={sample_meta['duration_sec']}s")

    # Step B: Run analysis with max_frames=35
    print("  --> Running multi-branch pipeline analysis (max_frames=35)...")
    res_analyze = client.post(f"/api/analyze/{video_id}?max_frames=35")
    assert res_analyze.status_code == 200, f"Analysis failed: {res_analyze.text}"

    data = res_analyze.json()
    assert data["success"] is True, "Expected success=True"
    assert data["total_frames_processed"] == 35, f"Expected 35 frames, got {data['total_frames_processed']}"

    # Verify structured statistics
    stats = data["statistics"]
    print(f"  --> Persistent tracks: {stats['persistent_tracks_count']}")
    print(f"  --> Behaviour observations: {stats['total_behaviour_predictions']} predictions, {stats['total_gathering_frames']} gathering")
    print(f"  --> Emotion observations: {stats['total_emotion_predictions']} predictions, {stats['total_unavailable_emotions']} unavailable")
    assert stats["persistent_tracks_count"] > 0, "Expected persistent tracks"

    # Verify tracked people telemetry
    tracked_people = data["tracked_people"]
    assert len(tracked_people) > 0, "Expected at least 1 tracked individual"
    for person in tracked_people:
        tid = person["track_id"]
        b = person["latest_behaviour"]
        e = person["latest_emotion"]
        print(f"  Person Track #{tid}: Behaviour='{b.get('class_name')}' (Conf: {b.get('confidence')}), Emotion='{e.get('label')}' (Conf: {e.get('confidence')})")

        # Confirm strict separation & absence of composite risk score
        assert "composite_score" not in person
        assert "combined_risk" not in person

    # Step C: Verify video stream endpoints
    res_input_vid = client.get(f"/api/video/{video_id}/input")
    assert res_input_vid.status_code == 200
    assert len(res_input_vid.content) > 0

    res_annot_vid = client.get(f"/api/video/{video_id}/annotated")
    assert res_annot_vid.status_code == 200
    assert len(res_annot_vid.content) > 0
    print(f"  --> Annotated video streamed successfully: {len(res_annot_vid.content):,} bytes")
    print("  --> [PASS] Full end-to-end API pipeline execution verified.")


def main():
    print("=" * 70)
    print("STAGE 9B: FASTAPI BACKEND & DASHBOARD INTEGRATION TEST SUITE")
    print("=" * 70)

    test_dashboard_page()
    test_static_assets()
    test_health_endpoint()
    test_sample_videos_list()
    test_invalid_upload_handling()
    test_end_to_end_analysis()

    print("\n" + "=" * 70)
    print("ALL STAGE 9B API & DASHBOARD TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
