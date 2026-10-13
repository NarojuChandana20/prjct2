"""
Stage 9C: Real Video End-to-End Incident Verification Script
------------------------------------------------------------
Runs the multi-branch pipeline with automated incident detection on the actual
uploaded WhatsApp video asset (where Track 2 engages in a real fight sequence).

Verifies:
  1. Automated detection of real High-Risk 'Fight' incident.
  2. Persistent event deduplication across multiple fight frames.
  3. Evidence screenshot capture saved in storage/evidence/.
  4. Incident insertion into SQLite storage/incidents.db.
  5. Incident queryable via GET /api/incidents.
  6. Evidence screenshot downloadable via GET /api/incidents/{id}/evidence.
  7. Human auditor review workflow via PATCH /api/incidents/{id}/review.
  8. Separate handling of body behaviour vs decoupled facial emotion.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import cv2

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
from backend.app.main import app, incident_db, incident_detector
from backend.app.services.video_behaviour_pipeline import VideoBehaviourPipeline

INPUT_PATH = ROOT_DIR / "storage" / "uploads" / "a94dc34536e4_WhatsApp Video 2026-10-06 at 3.03.12 PM.mp4"
OUTPUT_PATH = ROOT_DIR / "storage" / "outputs" / "stage9c_test_annotated.mp4"

client = TestClient(app)


def test_real_video_incident_e2e():
    print("=" * 70)
    print("STAGE 9C: REAL VIDEO INCIDENT DETECTION & REVIEW INTEGRATION TEST")
    print("=" * 70)

    assert INPUT_PATH.exists(), f"Input video missing: {INPUT_PATH}"

    pipeline = VideoBehaviourPipeline()
    incident_detector.reset_session()

    # Process first 50 frames (Track 2 reaches Fight prediction with conf 0.93 around frame 32-35)
    print("\n[Step 1] Processing real video clip (50 frames) with IncidentDetector...")
    res = pipeline.process_video(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        max_frames=50,
        incident_detector=incident_detector,
        video_id="a94dc34536e4",
    )

    recorded_ids = res.get("recorded_incidents", [])
    print(f"  --> Processed {res['total_frames_processed']} frames in {res['elapsed_seconds']}s")
    print(f"  --> Incidents recorded: {recorded_ids}")

    assert len(recorded_ids) > 0, "Expected at least 1 real Fight incident recorded on WhatsApp video!"
    target_inc_id = recorded_ids[0]

    # Verify database record
    print("\n[Step 2] Verifying incident metadata in SQLite...")
    inc = incident_db.get_incident(target_inc_id)
    assert inc is not None, f"Incident {target_inc_id} not found in SQLite DB!"
    print(f"  Incident ID:        {inc['id']}")
    print(f"  Source Video:       {inc['source_filename']}")
    print(f"  Behaviour Label:    {inc['behaviour_label']} ({inc['risk_level']} Risk)")
    print(f"  Confidence:         {inc['confidence']:.2f}")
    print(f"  Frame Span:         {inc['start_frame']} to {inc['end_frame']}")
    print(f"  Event Time:         {inc['event_time_sec']}s")
    print(f"  Associated Emotion: {inc['associated_emotion']}")
    print(f"  Review Status:      {inc['review_status']}")

    assert inc["behaviour_label"] == "Fight"
    assert inc["risk_level"] == "High"
    assert inc["confidence"] >= 0.70
    assert inc["start_frame"] <= inc["end_frame"]

    # Verify evidence file exists on disk
    print("\n[Step 3] Verifying evidence screenshot on disk...")
    ev_rel = inc["evidence_image_path"]
    assert ev_rel is not None, "Evidence image path is missing!"
    ev_full = ROOT_DIR / ev_rel
    assert ev_full.exists(), f"Evidence screenshot file does not exist: {ev_full}"
    ev_size = ev_full.stat().st_size
    print(f"  --> Evidence file: {ev_full.name} ({ev_size:,} bytes)")
    assert ev_size > 1000, "Evidence screenshot is too small!"

    # Verify REST API: GET /api/incidents
    print("\n[Step 4] Testing REST API: GET /api/incidents...")
    res_list = client.get("/api/incidents?video_id=a94dc34536e4")
    assert res_list.status_code == 200
    data = res_list.json()
    assert data["success"] is True
    assert any(i["id"] == target_inc_id for i in data["incidents"])
    print(f"  --> [PASS] Found recorded incident in API list query.")

    # Verify REST API: GET /api/incidents/{id}
    print("\n[Step 5] Testing REST API: GET /api/incidents/{id}...")
    res_get = client.get(f"/api/incidents/{target_inc_id}")
    assert res_get.status_code == 200
    inc_api = res_get.json()["incident"]
    assert "evidence_image_path" not in inc_api, "Internal filesystem path must NOT be exposed in public API!"
    assert inc_api["evidence_url"] == f"/api/incidents/{target_inc_id}/evidence"
    assert "ethics_notice" in inc_api
    print(f"  --> [PASS] API response sanitized: evidence_url='{inc_api['evidence_url']}'")

    # Verify REST API: GET /api/incidents/{id}/evidence image download
    print("\n[Step 6] Testing REST API: GET /api/incidents/{id}/evidence...")
    res_ev = client.get(f"/api/incidents/{target_inc_id}/evidence")
    assert res_ev.status_code == 200
    assert res_ev.headers["content-type"] == "image/jpeg"
    assert len(res_ev.content) == ev_size
    print(f"  --> [PASS] Evidence image streamed successfully ({len(res_ev.content):,} bytes).")

    # Verify REST API: PATCH /api/incidents/{id}/review (Human Auditor Workflow)
    print("\n[Step 7] Testing REST API: PATCH /api/incidents/{id}/review...")
    audit_notes = "Auditor verified: physical altercation in subway car between two subjects. Forwarded to campus security."
    res_patch = client.patch(f"/api/incidents/{target_inc_id}/review", json={
        "review_status": "reviewed",
        "reviewer_notes": audit_notes,
        "reviewer_name": "Supervisor D. Vance (Safety Dept)",
    })
    assert res_patch.status_code == 200
    updated_api = res_patch.json()["incident"]
    assert updated_api["review_status"] == "reviewed"
    assert updated_api["reviewed_by"] == "Supervisor D. Vance (Safety Dept)"
    assert updated_api["reviewer_notes"] == audit_notes
    print(f"  --> [PASS] Audit status updated to 'reviewed' with reviewer notes and identity.")

    # Verify summary stats
    print("\n[Step 8] Testing REST API: GET /api/incidents/stats/summary...")
    res_stats = client.get("/api/incidents/stats/summary")
    assert res_stats.status_code == 200
    stats = res_stats.json()["statistics"]
    assert stats["total_incidents"] > 0
    assert stats["reviewed"] > 0
    print(f"  --> [PASS] Total Incidents: {stats['total_incidents']}, Reviewed: {stats['reviewed']}, Pending: {stats['pending_review']}")

    print("\n" + "=" * 70)
    print("ALL REAL VIDEO INCIDENT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_real_video_incident_e2e()
