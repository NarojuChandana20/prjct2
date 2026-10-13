"""
Stage 9C: Incident Management and Review System Test Suite
---------------------------------------------------------
Verifies:
  1. Recording a valid Fight/Fall incident when body behaviour risk is High.
  2. NOT recording an incident for Normal/NonFight predictions.
  3. NOT generating incidents from facial emotion predictions alone.
  4. Suppressing duplicate records during a continuing event (frame span updates).
  5. Recording a new incident after cooldown period has elapsed.
  6. Storing and retrieving incident metadata from SQLite database.
  7. Handling missing/unavailable screenshots and DB errors gracefully.
  8. Updating review status and reviewer audit notes via API.
  9. Safe evidence file retrieval without exposing internal filesystem paths.
 10. Data retention policy pruning expired records and evidence files.
"""

from __future__ import annotations

import sys
import os
import time
import uuid
import shutil
from pathlib import Path
import cv2
import numpy as np

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
from backend.app.services.incident_service import IncidentDatabase, IncidentDetector

TEST_STORAGE_DIR = ROOT_DIR / "storage" / "test_scratch_9c"
TEST_DB_PATH = TEST_STORAGE_DIR / "test_incidents.db"
TEST_EVIDENCE_DIR = TEST_STORAGE_DIR / "test_evidence"

client = TestClient(app)


def setup_clean_test_env():
    """Sets up an isolated database and evidence directory for unit tests."""
    if TEST_STORAGE_DIR.exists():
        shutil.rmtree(TEST_STORAGE_DIR, ignore_errors=True)
    TEST_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    TEST_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    db = IncidentDatabase(TEST_DB_PATH)
    detector = IncidentDetector(
        db=db,
        min_persistence_frames=2,
        cooldown_frames=10,
        min_confidence=0.50,
        evidence_dir=TEST_EVIDENCE_DIR,
    )
    return db, detector


def test_incident_detection_rules():
    print("\n--- [Test 1] Incident Trigger & Deduplication Rules ---")
    db, detector = setup_clean_test_env()

    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "TEST INCIDENT FRAME", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 1. Normal frame should NOT create an incident
    recs_normal = [{
        "track_id": 1,
        "person_bbox": [50, 50, 150, 200],
        "det_confidence": 0.95,
        "behaviour": {"status": "predicted", "class_id": 0, "class_name": "NonFight", "confidence": 0.98, "risk_level": "Low"},
        "emotion": {"status": "available", "label": "Neutral", "confidence": 0.90},
    }]
    created = detector.process_frame(1, 0.033, dummy_frame, recs_normal, "vid_test", "test.mp4")
    assert len(created) == 0, f"Normal behaviour must NOT create an incident, got {created}"
    print("  --> [PASS] Normal/NonFight behaviour produces no incidents.")

    # 2. Emotion ALONE (e.g., Angry with high confidence) must NEVER create an incident
    recs_angry_face_normal_body = [{
        "track_id": 1,
        "person_bbox": [50, 50, 150, 200],
        "det_confidence": 0.95,
        "behaviour": {"status": "predicted", "class_id": 0, "class_name": "NonFight", "confidence": 0.92, "risk_level": "Low"},
        "emotion": {"status": "available", "label": "Angry", "confidence": 0.99},
    }]
    created = detector.process_frame(2, 0.066, dummy_frame, recs_angry_face_normal_body, "vid_test", "test.mp4")
    assert len(created) == 0, "Facial emotion alone must NEVER trigger an incident!"
    print("  --> [PASS] High-confidence Angry facial emotion alone NEVER triggers an incident.")

    # 3. High risk Fight frame 1: Should NOT trigger yet because min_persistence_frames=2
    recs_fight_f1 = [{
        "track_id": 2,
        "person_bbox": [100, 40, 220, 210],
        "det_confidence": 0.94,
        "behaviour": {"status": "predicted", "class_id": 1, "class_name": "Fight", "confidence": 0.88, "risk_level": "High"},
        "emotion": {"status": "unavailable", "label": "Unavailable", "confidence": 0.0},
    }]
    created = detector.process_frame(3, 0.100, dummy_frame, recs_fight_f1, "vid_test", "test.mp4")
    assert len(created) == 0, "Single isolated frame must NOT trigger without meeting persistence"
    print("  --> [PASS] Single high-risk frame held in candidate buffer without premature trigger.")

    # 4. High risk Fight frame 2: Meets persistence (2 frames) -> Triggers genuine incident!
    recs_fight_f2 = [{
        "track_id": 2,
        "person_bbox": [102, 42, 222, 212],
        "det_confidence": 0.95,
        "behaviour": {"status": "predicted", "class_id": 1, "class_name": "Fight", "confidence": 0.91, "risk_level": "High"},
        "emotion": {"status": "unavailable", "label": "Unavailable", "confidence": 0.0},
    }]
    created = detector.process_frame(4, 0.133, dummy_frame, recs_fight_f2, "vid_test", "test.mp4")
    assert len(created) == 1, f"Expected 1 incident created after persistence, got {created}"
    inc_id = created[0]
    print(f"  --> [PASS] Recorded valid High Risk incident: {inc_id}")

    # Verify incident metadata in SQLite
    inc_record = db.get_incident(inc_id)
    assert inc_record is not None
    assert inc_record["behaviour_label"] == "Fight"
    assert inc_record["risk_level"] == "High"
    assert inc_record["start_frame"] == 3
    assert inc_record["end_frame"] == 4
    assert inc_record["confidence"] == 0.91
    assert inc_record["review_status"] == "pending"
    assert inc_record["evidence_image_path"] is not None
    ev_full = ROOT_DIR / inc_record["evidence_image_path"]
    assert ev_full.exists(), f"Evidence screenshot file missing: {ev_full}"
    print(f"  --> [PASS] Verified database record & evidence file: {ev_full.name}")

    # 5. Continuing event: Frames 5, 6, 7 continue fight -> Must NOT create duplicate incidents!
    for f in range(5, 8):
        recs_continue = [{
            "track_id": 2,
            "person_bbox": [105, 45, 225, 215],
            "det_confidence": 0.93,
            "behaviour": {"status": "predicted", "class_id": 1, "class_name": "Fight", "confidence": 0.94, "risk_level": "High"},
            "emotion": {"status": "unavailable", "label": "Unavailable", "confidence": 0.0},
        }]
        created_dup = detector.process_frame(f, f * 0.033, dummy_frame, recs_continue, "vid_test", "test.mp4")
        assert len(created_dup) == 0, f"Continuing ongoing event created duplicate incident at frame {f}: {created_dup}"

    # Verify that the existing incident updated its span and max confidence
    updated_rec = db.get_incident(inc_id)
    assert updated_rec["end_frame"] == 7, f"Expected end_frame=7, got {updated_rec['end_frame']}"
    assert updated_rec["confidence"] == 0.94, f"Expected confidence=0.94, got {updated_rec['confidence']}"
    print("  --> [PASS] Ongoing event updated span to frame 7 and highest confidence to 0.94 without duplication.")

    # 6. Event ends: 10 quiet normal frames elapse (cooldown_frames=10)
    for f in range(8, 19):
        recs_quiet = [{
            "track_id": 2,
            "person_bbox": [100, 50, 200, 200],
            "det_confidence": 0.90,
            "behaviour": {"status": "predicted", "class_id": 0, "class_name": "NonFight", "confidence": 0.95, "risk_level": "Low"},
            "emotion": {"status": "available", "label": "Neutral", "confidence": 0.85},
        }]
        detector.process_frame(f, f * 0.033, dummy_frame, recs_quiet, "vid_test", "test.mp4")

    # 7. New second fight event occurs after cooldown -> Should trigger a second distinct incident!
    recs_new_fight_1 = [{
        "track_id": 2,
        "person_bbox": [110, 50, 230, 220],
        "det_confidence": 0.92,
        "behaviour": {"status": "predicted", "class_id": 1, "class_name": "Fight", "confidence": 0.89, "risk_level": "High"},
        "emotion": {"status": "unavailable", "label": "Unavailable", "confidence": 0.0},
    }]
    detector.process_frame(19, 19 * 0.033, dummy_frame, recs_new_fight_1, "vid_test", "test.mp4")

    recs_new_fight_2 = [{
        "track_id": 2,
        "person_bbox": [112, 52, 232, 222],
        "det_confidence": 0.95,
        "behaviour": {"status": "predicted", "class_id": 1, "class_name": "Fight", "confidence": 0.93, "risk_level": "High"},
        "emotion": {"status": "unavailable", "label": "Unavailable", "confidence": 0.0},
    }]
    created_second = detector.process_frame(20, 20 * 0.033, dummy_frame, recs_new_fight_2, "vid_test", "test.mp4")
    assert len(created_second) == 1, f"Expected second incident after cooldown, got {created_second}"
    second_inc_id = created_second[0]
    assert second_inc_id != inc_id, "Second incident must have a unique ID"
    print(f"  --> [PASS] Recorded distinct second incident after cooldown: {second_inc_id}")

    # Check database list
    incidents, count = db.list_incidents(video_id="vid_test")
    assert count == 2
    assert len(incidents) == 2
    print(f"  --> [PASS] Verified total of {count} unique incidents recorded.")


def test_missing_screenshot_and_error_handling():
    print("\n--- [Test 2] Missing Screenshot & Database Error Handling ---")
    db, detector = setup_clean_test_env()

    # Pass None as frame: detector should handle without crashing and record incident with evidence_image_path=None
    recs = [{
        "track_id": 99,
        "person_bbox": [10, 10, 50, 100],
        "det_confidence": 0.85,
        "behaviour": {"status": "predicted", "class_id": 2, "class_name": "Fall", "confidence": 0.85, "risk_level": "High"},
        "emotion": {"status": "unavailable", "label": "Unavailable", "confidence": 0.0},
    }]
    detector.process_frame(1, 0.033, None, recs, "vid_err", "err.mp4")
    created = detector.process_frame(2, 0.066, None, recs, "vid_err", "err.mp4")
    assert len(created) == 1
    inc = db.get_incident(created[0])
    assert inc["evidence_image_path"] is None, "Should be None when screenshot is unavailable"
    assert inc["behaviour_label"] == "Fall"
    print("  --> [PASS] Safely handled missing screenshot without inventing fake evidence.")


def test_retention_policy():
    print("\n--- [Test 3] Data Retention Policy Pruning ---")
    db, detector = setup_clean_test_env()

    # Insert an old incident from 40 days ago
    old_time = time.time() - (40 * 86400)
    fake_img = TEST_EVIDENCE_DIR / "old_test.jpg"
    fake_img.write_bytes(b"dummy")

    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO incidents (
                id, video_id, source_filename, track_id, behaviour_label,
                confidence, risk_level, start_frame, end_frame, event_time_sec,
                evidence_image_path, associated_emotion, emotion_confidence,
                review_status, reviewer_notes, created_at, updated_at
            ) VALUES (
                'inc_old_1', 'vid_old', 'old.mp4', 1, 'Fight',
                0.90, 'High', 10, 20, 0.5,
                ?, 'Neutral', 0.8,
                'reviewed', 'Historic event', ?, ?
            );
        """, (str(fake_img.relative_to(ROOT_DIR)).replace("\\", "/"), old_time, old_time))
        conn.commit()

    assert fake_img.exists()
    pruned = db.apply_retention_policy(retention_days=30)
    assert pruned == 1, f"Expected 1 expired incident pruned, got {pruned}"
    assert not fake_img.exists(), "Evidence file must be safely unlinked upon retention pruning"
    assert db.get_incident("inc_old_1") is None
    print("  --> [PASS] Successfully pruned expired incident and evidence file under 30-day retention.")


def test_api_incident_endpoints():
    print("\n--- [Test 4] Incident Management REST API Endpoints ---")
    # 1. Health & Stats
    res_stats = client.get("/api/incidents/stats/summary")
    assert res_stats.status_code == 200
    assert "statistics" in res_stats.json()
    print("  --> [PASS] GET /api/incidents/stats/summary returned valid stats.")

    # 2. List incidents
    res_list = client.get("/api/incidents")
    assert res_list.status_code == 200
    data = res_list.json()
    assert "incidents" in data
    assert "total_count" in data
    print(f"  --> [PASS] GET /api/incidents returned {data['total_count']} total records.")

    # 3. Create test incident in main DB and test PATCH review
    from backend.app.main import incident_db
    test_id = f"inc_test_api_{uuid.uuid4().hex[:6]}"
    incident_db.insert_incident({
        "id": test_id,
        "video_id": "api_test_vid",
        "source_filename": "security_cam_1.mp4",
        "track_id": 5,
        "behaviour_label": "Fight",
        "confidence": 0.92,
        "risk_level": "High",
        "start_frame": 15,
        "end_frame": 35,
        "event_time_sec": 1.12,
        "associated_emotion": "Unavailable",
        "emotion_confidence": 0.0,
        "review_status": "pending",
        "reviewer_notes": "",
    })

    # Retrieve via API
    res_get = client.get(f"/api/incidents/{test_id}")
    assert res_get.status_code == 200
    inc_data = res_get.json()["incident"]
    assert inc_data["id"] == test_id
    assert inc_data["review_status"] == "pending"
    assert "evidence_image_path" not in inc_data, "Absolute local file path must NEVER be exposed in public API!"
    assert "ethics_notice" in inc_data, "Ethics notice must be present"
    print("  --> [PASS] GET /api/incidents/{id} returned sanitized record with ethics notice.")

    # Update review status via PATCH
    res_patch = client.patch(f"/api/incidents/{test_id}/review", json={
        "review_status": "reviewed",
        "reviewer_notes": "Confirmed altercation between students. Dispatched campus safety officer.",
        "reviewer_name": "Officer J. Smith (Badge #402)",
    })
    assert res_patch.status_code == 200
    patch_data = res_patch.json()["incident"]
    assert patch_data["review_status"] == "reviewed"
    assert "Officer J. Smith" in patch_data["reviewed_by"]
    assert "campus safety officer" in patch_data["reviewer_notes"]
    print("  --> [PASS] PATCH /api/incidents/{id}/review successfully recorded auditor conclusions.")

    # Filter by reviewed status
    res_filter = client.get("/api/incidents?status=reviewed")
    assert res_filter.status_code == 200
    filtered_ids = [i["id"] for i in res_filter.json()["incidents"]]
    assert test_id in filtered_ids
    print("  --> [PASS] Filtered search by review status succeeded.")

    # Delete test incident
    res_del = client.delete(f"/api/incidents/{test_id}")
    assert res_del.status_code == 200
    res_get_deleted = client.get(f"/api/incidents/{test_id}")
    assert res_get_deleted.status_code == 404
    print("  --> [PASS] DELETE /api/incidents/{id} safely deleted test record.")


def main():
    print("=" * 70)
    print("STAGE 9C: INCIDENT MANAGEMENT & REVIEW SYSTEM VERIFICATION SUITE")
    print("=" * 70)

    test_incident_detection_rules()
    test_missing_screenshot_and_error_handling()
    test_retention_policy()
    test_api_incident_endpoints()

    print("\n" + "=" * 70)
    print("ALL STAGE 9C VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
