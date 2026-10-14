"""
Student Safety Monitoring System - Backend API Server
------------------------------------------------------
FastAPI server exposing endpoints to run the Unified Multi-Branch Video Pipeline
and serve the Frontend Dashboard.

Endpoints:
  GET  /                          -> Serves frontend dashboard SPA
  GET  /api/health                -> System health & loaded model statuses
  GET  /api/sample-videos         -> Lists available test video assets
  POST /api/upload                -> Validates & stores input video file
  POST /api/analyze/{video_id}    -> Runs unified pipeline & returns structured results
  GET  /api/video/{video_id}/input -> Streams original input video
  GET  /api/video/{video_id}/annotated -> Streams annotated output video
"""

from __future__ import annotations

import sys
import os
import shutil
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import threading
import base64

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure UTF-8 terminal encoding on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pydantic import BaseModel, Field

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.video_behaviour_pipeline import VideoBehaviourPipeline
from backend.app.services.incident_service import IncidentDatabase, IncidentDetector, EVIDENCE_DIR, DB_PATH

# Directories
STORAGE_DIR = ROOT_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
OUTPUT_DIR = STORAGE_DIR / "outputs"
FRONTEND_DIR = ROOT_DIR / "frontend"
ASSETS_DIR = ROOT_DIR / "tests" / "assets"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# Incident Database & Detector singletons (Stage 9C)
incident_db = IncidentDatabase(DB_PATH)
incident_detector = IncidentDetector(db=incident_db, evidence_dir=EVIDENCE_DIR)

# Global singleton pipeline instance
_pipeline_instance: Optional[VideoBehaviourPipeline] = None


def get_pipeline() -> VideoBehaviourPipeline:
    """Returns the cached singleton pipeline instance to avoid reloading models."""
    global _pipeline_instance
    if _pipeline_instance is None:
        print("[API] Initializing Unified Multi-Branch Video Pipeline singleton...")
        _pipeline_instance = VideoBehaviourPipeline()
        print("[API] Pipeline singleton ready.")
    return _pipeline_instance


# ==============================================================================
# Live Webcam Session State & Dedicated Pipeline (Isolated from video analysis)
# ==============================================================================
_webcam_pipeline_instance: Optional[VideoBehaviourPipeline] = None
_webcam_pipeline_lock = threading.Lock()


def get_webcam_pipeline() -> VideoBehaviourPipeline:
    """Returns a dedicated webcam pipeline instance, separate from the video analysis pipeline."""
    global _webcam_pipeline_instance
    if _webcam_pipeline_instance is None:
        print("[API] Initializing dedicated Webcam Pipeline instance...")
        _webcam_pipeline_instance = VideoBehaviourPipeline()
        print("[API] Webcam Pipeline ready.")
    return _webcam_pipeline_instance


# Active webcam session state (only one session allowed at a time)
_webcam_session: Dict[str, Any] = {
    "active": False,
    "session_id": None,
    "video_id": None,
    "frame_idx": 0,
    "start_time": 0.0,
    "fps": 25.0,
    "incident_detector": None,
    "total_detections": 0,
    "total_behaviour_preds": 0,
    "total_emotion_preds": 0,
    "all_track_ids": set(),
    "person_telemetry_history": {},
    "latest_frame_bytes": b"",
}


# FastAPI Application
app = FastAPI(
    title="Student Safety Monitoring System",
    description="Multi-Branch AI System for Student Safety & Behaviour Analysis",
    version="1.0.0",
)

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory registry for uploaded videos metadata
video_registry: Dict[str, Dict[str, Any]] = {}

# In-memory registry for live video analysis progress
analysis_progress: Dict[str, Dict[str, Any]] = {}

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@app.get("/api/progress/{video_id}")
def get_analysis_progress(video_id: str):
    """Returns real-time progress for an ongoing video analysis task."""
    if video_id in analysis_progress:
        return analysis_progress[video_id]
    return {
        "status": "idle",
        "current_frame": 0,
        "total_frames": 0,
        "percent": 0.0,
        "fps": 0.0,
    }


@app.get("/api/health")
def health_check():
    """Returns health status, active models, and pipeline state."""
    pipeline_ready = _pipeline_instance is not None
    return {
        "status": "healthy",
        "system": "Student Safety Monitoring System",
        "stage": "Stage 9C - Complete Application",
        "pipeline_initialized": pipeline_ready,
        "models": {
            "yolo11": "yolo11n.pt (Person Class 0)",
            "pose": "pose_landmarker_lite.task (33 3D Keypoints)",
            "behaviour": "behaviour_model_rich_165_v2_best.keras (BiLSTM, 165 Features)",
            "face": "blaze_face_short_range.tflite (MediaPipe FaceDetector)",
            "emotion": "rafdb_emotion_best.keras (EfficientNetB0, 7 Classes)",
        },
        "supported_formats": list(SUPPORTED_EXTENSIONS),
    }


@app.get("/api/sample-videos")
def list_sample_videos():
    """Lists pre-existing sample videos in tests/assets for easy 1-click testing."""
    samples = []
    if ASSETS_DIR.exists():
        for p in ASSETS_DIR.glob("*.mp4"):
            # Exclude output videos from sample list
            if "output" in p.name.lower():
                continue
            cap = cv2.VideoCapture(str(p))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            cap.release()
            duration = round(fc / fps, 2) if fps > 0 else 0.0

            samples.append({
                "filename": p.name,
                "file_path": str(p),
                "file_size": p.stat().st_size,
                "width": w,
                "height": h,
                "frames": fc,
                "fps": round(fps, 1),
                "duration_sec": duration,
            })
    return {"samples": samples}


@app.post("/api/use-sample/{sample_name}")
def use_sample_video(sample_name: str):
    """Copies a sample video from tests/assets into upload storage for analysis."""
    sample_path = ASSETS_DIR / sample_name
    if not sample_path.exists() or not sample_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample video '{sample_name}' not found in test assets."
        )

    video_id = uuid.uuid4().hex[:12]
    dest_path = UPLOAD_DIR / f"{video_id}_{sample_path.name}"
    shutil.copyfile(str(sample_path), str(dest_path))

    cap = cv2.VideoCapture(str(dest_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()

    meta = {
        "video_id": video_id,
        "filename": sample_path.name,
        "file_path": str(dest_path),
        "file_size": dest_path.stat().st_size,
        "width": w,
        "height": h,
        "frames": fc,
        "fps": round(fps, 2),
        "duration_sec": round(fc / fps, 2) if fps > 0 else 0.0,
        "uploaded_at": time.time(),
        "preview_url": f"/api/video/{video_id}/input",
    }
    video_registry[video_id] = meta
    return meta


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Accepts an uploaded video file, validates extension and dimensions,
    and stores it for pipeline processing.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in upload."
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    video_id = uuid.uuid4().hex[:12]
    safe_name = f"{video_id}_{file.filename}"
    dest_path = UPLOAD_DIR / safe_name

    # Save uploaded file safely
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {e}"
        )

    # Verify that the file is a readable video
    cap = cv2.VideoCapture(str(dest_path))
    if not cap.isOpened():
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid or readable video file."
        )

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()

    if w <= 0 or h <= 0 or fc <= 0:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded video contains no frames or invalid dimensions."
        )

    meta = {
        "video_id": video_id,
        "filename": file.filename,
        "file_path": str(dest_path),
        "file_size": dest_path.stat().st_size,
        "width": w,
        "height": h,
        "frames": fc,
        "fps": round(fps, 2),
        "duration_sec": round(fc / fps, 2) if fps > 0 else 0.0,
        "uploaded_at": time.time(),
        "preview_url": f"/api/video/{video_id}/input",
    }
    video_registry[video_id] = meta

    return meta


@app.post("/api/analyze/{video_id}")
def analyze_video(
    video_id: str,
    max_frames: Optional[int] = Query(None, description="Maximum frames to analyze (optional, None for full video)"),
):
    """
    Executes the unified video behaviour & emotion pipeline on the uploaded video.
    Returns structured results, summary statistics, and URLs for output playback.
    """
    if video_id not in video_registry:
        # Check if file exists in uploads folder
        matches = list(UPLOAD_DIR.glob(f"{video_id}_*"))
        if not matches:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video ID '{video_id}' not found."
            )
        input_file = matches[0]
        meta = {
            "video_id": video_id,
            "filename": input_file.name.replace(f"{video_id}_", "", 1),
            "file_path": str(input_file),
        }
        video_registry[video_id] = meta
    else:
        meta = video_registry[video_id]
        input_file = Path(meta["file_path"])

    if not input_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file missing on disk: {input_file.name}"
        )

    output_file = OUTPUT_DIR / f"{video_id}_annotated.mp4"

    # Treat non-positive max_frames as full video analysis
    if max_frames is not None and max_frames <= 0:
        max_frames = None

    # Initialize live progress tracking
    analysis_progress[video_id] = {
        "status": "processing",
        "current_frame": 0,
        "total_frames": 0,
        "percent": 0.0,
        "fps": 0.0,
        "start_time": time.time(),
    }

    def on_pipeline_progress(cur: int, tot: int):
        st = analysis_progress[video_id]["start_time"]
        el = time.time() - st
        proc_fps = round(cur / el, 1) if el > 0 else 0.0
        pct = round((cur / tot * 100), 1) if tot > 0 else 0.0
        analysis_progress[video_id].update({
            "status": "processing",
            "current_frame": cur,
            "total_frames": tot,
            "percent": pct,
            "fps": proc_fps,
        })

    # Initialize session in database
    session_id = f"ses_{uuid.uuid4().hex[:10]}"
    try:
        incident_db.create_session({
            "id": session_id,
            "video_id": video_id,
            "source_filename": meta.get("filename", "video.mp4"),
            "total_frames": meta.get("frames", 0),
            "fps": meta.get("fps", 25.0),
            "status": "processing",
            "created_at": time.time(),
        })
    except Exception as e:
        print(f"[API] Warning initializing session {session_id}: {e}")

    # Run unified pipeline with automated incident detection (Stage 9C + Multi-Session)
    pipeline = get_pipeline()
    incident_detector.reset_session(session_id=session_id)
    try:
        raw_res = pipeline.process_video(
            input_path=input_file,
            output_path=output_file,
            max_frames=max_frames,
            progress_callback=on_pipeline_progress,
            incident_detector=incident_detector,
            video_id=video_id,
        )
        analysis_progress[video_id].update({
            "status": "completed",
            "current_frame": raw_res["total_frames_processed"],
            "total_frames": raw_res["total_frames_processed"],
            "percent": 100.0,
            "fps": raw_res["fps"],
        })
    except Exception as e:
        analysis_progress[video_id].update({
            "status": "failed",
            "error": str(e),
        })
        print(f"[API ERROR] Pipeline processing failed: {e}")
        try:
            incident_db.update_session(session_id=session_id, status="failed")
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing failed: {str(e)}"
        )

    # -------------------------------------------------------------
    # Compute genuine structured statistics from real telemetry
    # -------------------------------------------------------------
    person_history = raw_res.get("person_telemetry_history", {})
    all_tids = raw_res.get("unique_track_ids", [])

    persistent_tracks = [tid for tid in all_tids if tid > 0]
    temporary_detections = [tid for tid in all_tids if tid < 0]

    total_behaviour_predictions = 0
    total_gathering_frames = 0
    total_unavailable_behaviours = 0
    total_emotion_predictions = 0
    total_unavailable_emotions = 0

    behaviour_counts = {"NonFight": 0, "Fight": 0, "Fall": 0, "Collecting": 0, "Unavailable": 0}
    emotion_counts = {
        "Surprise": 0, "Fear": 0, "Disgust": 0, "Happy": 0,
        "Sad": 0, "Angry": 0, "Neutral": 0, "Unavailable": 0
    }

    per_person_summary = []
    telemetry_timeline: Dict[int, List[Dict[str, Any]]] = {}

    for tid in persistent_tracks:
        updates = person_history.get(tid, [])
        if not updates:
            continue

        p_beh_counts = {"NonFight": 0, "Fight": 0, "Fall": 0, "Collecting": 0, "Unavailable": 0}
        p_em_counts = {k: 0 for k in emotion_counts.keys()}

        latest_rec = updates[-1]
        first_rec = updates[0]

        for u in updates:
            f_idx = u["frame"]
            if f_idx not in telemetry_timeline:
                telemetry_timeline[f_idx] = []

            # Populate timeline item
            telemetry_timeline[f_idx].append({
                "track_id": tid,
                "bbox": u["bbox"],
                "det_confidence": round(u["det_confidence"], 3),
                "behaviour": {
                    "status": u["behaviour"].get("status"),
                    "class_name": u["behaviour"].get("class_name"),
                    "confidence": round(u["behaviour"].get("confidence", 0.0), 3),
                    "risk_level": u["behaviour"].get("risk_level", "—"),
                },
                "emotion": {
                    "status": u["emotion"].get("status"),
                    "label": u["emotion"].get("label"),
                    "confidence": round(u["emotion"].get("confidence", 0.0), 3),
                    "reason": u["emotion"].get("reason"),
                },
            })

            # Aggregate behaviour counts: distinguish predicted vs collecting vs unavailable
            b_status = u["behaviour"].get("status")
            if b_status == "predicted":
                total_behaviour_predictions += 1
                cname = u["behaviour"].get("class_name", "")
                if "Fight" in cname and "Non" not in cname:
                    behaviour_counts["Fight"] += 1
                    p_beh_counts["Fight"] += 1
                elif "Fall" in cname:
                    behaviour_counts["Fall"] += 1
                    p_beh_counts["Fall"] += 1
                else:
                    behaviour_counts["NonFight"] += 1
                    p_beh_counts["NonFight"] += 1
            elif b_status == "collecting":
                total_gathering_frames += 1
                behaviour_counts["Collecting"] += 1
                p_beh_counts["Collecting"] += 1
            else:
                total_unavailable_behaviours += 1
                behaviour_counts["Unavailable"] += 1
                p_beh_counts["Unavailable"] += 1

            # Aggregate emotion counts
            e_status = u["emotion"].get("status")
            if e_status == "available":
                total_emotion_predictions += 1
                elabel = u["emotion"].get("label", "Unavailable")
                if elabel in emotion_counts:
                    emotion_counts[elabel] += 1
                    p_em_counts[elabel] += 1
            else:
                total_unavailable_emotions += 1
                emotion_counts["Unavailable"] += 1
                p_em_counts["Unavailable"] += 1

        # Build person summary card
        latest_b = latest_rec["behaviour"]
        latest_e = latest_rec["emotion"]

        person_obj = {
            "track_id": tid,
            "is_temporary_id": False,
            "total_frames_tracked": len(updates),
            "first_frame": first_rec["frame"],
            "last_frame": latest_rec["frame"],
            "first_timestamp_sec": first_rec["timestamp_sec"],
            "last_timestamp_sec": latest_rec["timestamp_sec"],
            "latest_behaviour": {
                "status": latest_b.get("status"),
                "class_name": latest_b.get("class_name"),
                "confidence": round(latest_b.get("confidence", 0.0), 3),
                "risk_level": latest_b.get("risk_level", "—"),
            },
            "latest_emotion": {
                "status": latest_e.get("status"),
                "label": latest_e.get("label"),
                "confidence": round(latest_e.get("confidence", 0.0), 3),
                "reason": latest_e.get("reason"),
            },
            "behaviour_breakdown": p_beh_counts,
            "emotion_breakdown": p_em_counts,
        }
        per_person_summary.append(person_obj)

        # Store track in persistent tracks table
        try:
            incident_db.insert_track({
                "id": f"trk_{session_id}_{tid}",
                "session_id": session_id,
                "track_id": tid,
                "is_persistent": True,
                "first_frame": first_rec["frame"],
                "last_frame": latest_rec["frame"],
                "total_frames": len(updates),
                "primary_behaviour": latest_b.get("class_name", "Unavailable"),
                "primary_emotion": latest_e.get("label", "Unavailable"),
                "highest_conf": latest_b.get("confidence", 0.0),
                "highest_risk": latest_b.get("risk_level", "—"),
                "metadata_json": {
                    "behaviour_breakdown": p_beh_counts,
                    "emotion_breakdown": p_em_counts,
                }
            })
        except Exception as trk_err:
            print(f"[API] Warning recording track {tid} into database: {trk_err}")

    # Sort persons by track ID
    per_person_summary.sort(key=lambda x: x["track_id"])

    # Compute risk contribution ratios based on behaviour confidence scores
    total_conf = sum(
        p["latest_behaviour"]["confidence"]
        for p in per_person_summary
        if p["latest_behaviour"]["confidence"] >= 0
    )
    for p in per_person_summary:
        if total_conf > 0:
            p["risk_contribution_ratio"] = round(p["latest_behaviour"]["confidence"] / total_conf * 100, 1)
        else:
            p["risk_contribution_ratio"] = "Insufficient evidence"

    recorded_inc_ids = raw_res.get("recorded_incidents", [])

    # Update session completion in database
    try:
        incident_db.update_session(
            session_id=session_id,
            status="completed",
            completed_at=time.time(),
            metrics={
                "total_frames_processed": raw_res["total_frames_processed"],
                "persistent_tracks_count": len(persistent_tracks),
                "recorded_incidents_count": len(recorded_inc_ids),
                "processing_fps": raw_res["fps"],
            }
        )
    except Exception as sess_err:
        print(f"[API] Warning finalizing session {session_id}: {sess_err}")

    # Calculate people count in final observed frame
    final_frame_idx = raw_res["total_frames_processed"]
    current_in_final = len(telemetry_timeline.get(final_frame_idx, []))
    if current_in_final == 0 and telemetry_timeline:
        latest_f = max(telemetry_timeline.keys())
        current_in_final = len(telemetry_timeline[latest_f])

    response_data = {
        "success": True,
        "session_id": session_id,
        "video_id": video_id,
        "filename": meta.get("filename"),
        "total_frames_processed": raw_res["total_frames_processed"],
        "elapsed_seconds": raw_res["elapsed_seconds"],
        "processing_fps": raw_res["fps"],
        "recorded_incidents_count": len(recorded_inc_ids),
        "recorded_incident_ids": recorded_inc_ids,
        "video_urls": {
            "input": f"/api/video/{video_id}/input",
            "annotated": f"/api/video/{video_id}/annotated",
        },
        "statistics": {
            "current_visible_detections": current_in_final,
            "persistent_tracks_count": len(persistent_tracks),
            "total_observed_tracks_count": len(persistent_tracks),
            "temporary_detections_count": len(temporary_detections),
            "total_behaviour_predictions": total_behaviour_predictions,
            "total_gathering_frames": total_gathering_frames,
            "total_unavailable_behaviours": total_unavailable_behaviours,
            "total_emotion_predictions": total_emotion_predictions,
            "total_unavailable_emotions": total_unavailable_emotions,
            "behaviour_distribution": behaviour_counts,
            "emotion_distribution": emotion_counts,
        },
        "tracked_people": per_person_summary,
        "timeline_sample_count": len(telemetry_timeline),
    }

    return response_data


@app.get("/api/video/{video_id}/input")
def stream_input_video(video_id: str):
    """Streams original input video file."""
    matches = list(UPLOAD_DIR.glob(f"{video_id}_*"))
    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Input video not found."
        )
    return FileResponse(str(matches[0]), media_type="video/mp4")


@app.get("/api/video/{video_id}/annotated")
def stream_annotated_video(video_id: str):
    """Streams annotated output video file."""
    output_path = OUTPUT_DIR / f"{video_id}_annotated.mp4"
    if not output_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotated video not found or not yet processed."
        )
    return FileResponse(str(output_path), media_type="video/mp4")


# ==============================================================================
# Stage 9C: Incident Management & Review Endpoints
# ==============================================================================

class ReviewUpdateRequest(BaseModel):
    review_status: str = Field(..., description="'pending', 'reviewed', or 'dismissed'")
    reviewer_notes: Optional[str] = Field("", description="Human reviewer audit notes")
    reviewer_name: Optional[str] = Field("Authorized Reviewer", description="Identity or handle of reviewer")


class VerifiedPersonCreate(BaseModel):
    official_id: str = Field(..., description="Official Student or Staff Identifier")
    full_name: str = Field(..., description="Full Name of Individual")
    role: str = Field("student", description="Role: student, staff, or visitor")
    profile_photo_path: Optional[str] = Field(None, description="Optional profile photo file path")
    notes: Optional[str] = Field("", description="Authorized administrative notes")


class LinkTrackRequest(BaseModel):
    person_id: str = Field(..., description="Internal person ID or official identifier")
    linked_by: str = Field("Authorized Reviewer", description="Name/badge of auditor performing linkage")
    notes: Optional[str] = Field("", description="Review rationale and notes")


def _sanitize_incident_response(inc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes incident dictionary for public API consumption:
    - Never exposes raw local filesystem paths.
    - Exposes a secure relative stream URL: /api/incidents/{id}/evidence if image exists.
    - Adds standard safety & ethics notice.
    """
    res = dict(inc)
    img_path = res.pop("evidence_image_path", None)
    if img_path:
        full_p = ROOT_DIR / img_path if not Path(img_path).is_absolute() else Path(img_path)
        has_file = full_p.exists()
        res["has_evidence"] = has_file
        res["evidence_url"] = f"/api/incidents/{res['id']}/evidence" if has_file else None
    else:
        res["has_evidence"] = False
        res["evidence_url"] = None

    # Clarify that tracking IDs are temporary track tokens
    tid = res.get("track_id")
    res["track_display"] = f"Track #{tid}" if tid is not None else "Untracked"

    res["ethics_notice"] = (
        "Model predictions are assistive statistical observations and do not establish guilt, intent, or real-world identity. "
        "A qualified human reviewer must assess the incident in context."
    )
    return res


@app.get("/api/incidents")
def list_incidents(
    video_id: Optional[str] = Query(None, description="Filter by source video ID"),
    session_id: Optional[str] = Query(None, description="Filter by analysis session ID"),
    status: Optional[str] = Query(None, description="Filter by review status (pending, reviewed, dismissed)"),
    behaviour: Optional[str] = Query(None, description="Filter by behaviour label (Fight, Fall)"),
    limit: int = Query(50, ge=1, le=200, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
):
    """Lists recorded incidents with optional session/video filters and pagination."""
    rows, total_count = incident_db.list_incidents(
        video_id=video_id,
        session_id=session_id,
        review_status=status,
        behaviour_label=behaviour,
        limit=limit,
        offset=offset,
    )
    sanitized = [_sanitize_incident_response(r) for r in rows]
    return {
        "success": True,
        "total_count": total_count,
        "count": len(sanitized),
        "limit": limit,
        "offset": offset,
        "incidents": sanitized,
    }


@app.get("/api/incidents/stats/summary")
def get_incident_stats(
    video_id: Optional[str] = Query(None, description="Filter stats by video ID"),
    session_id: Optional[str] = Query(None, description="Filter stats by session ID"),
):
    """Returns incident summary metrics, optionally scoped to video or session."""
    stats = incident_db.get_statistics(video_id=video_id, session_id=session_id)
    return {
        "success": True,
        "statistics": stats,
    }


@app.get("/api/sessions")
def list_analysis_sessions(
    video_id: Optional[str] = Query(None, description="Filter sessions by video ID"),
    limit: int = Query(50, ge=1, le=100),
):
    """Lists past and current video analysis sessions."""
    sessions = incident_db.list_sessions(video_id=video_id, limit=limit)
    return {
        "success": True,
        "count": len(sessions),
        "sessions": sessions,
    }


@app.get("/api/sessions/{session_id}")
def get_session_details(session_id: str):
    """Retrieves session details, aggregated tracks, and associated incidents."""
    sess = incident_db.get_session(session_id)
    if not sess:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found."
        )
    tracks = incident_db.list_tracks(session_id)
    incidents, inc_count = incident_db.list_incidents(session_id=session_id)
    return {
        "success": True,
        "session": sess,
        "tracks": tracks,
        "incident_count": inc_count,
        "incidents": [_sanitize_incident_response(r) for r in incidents],
    }


@app.get("/api/verified-persons")
def list_verified_persons():
    """Lists registered individuals in the authorized person registry."""
    persons = incident_db.list_verified_persons()
    return {
        "success": True,
        "count": len(persons),
        "persons": persons,
    }


@app.post("/api/verified-persons")
def create_verified_person(body: VerifiedPersonCreate):
    """Registers an authorized individual in the person registry (manual authorized review)."""
    p_id = incident_db.create_verified_person(body.dict())
    return {
        "success": True,
        "message": f"Person '{body.full_name}' ({body.official_id}) registered successfully.",
        "person_id": p_id,
    }


@app.post("/api/sessions/{session_id}/tracks/{track_id}/link")
def link_track_to_person(session_id: str, track_id: int, body: LinkTrackRequest):
    """Links an ephemeral tracking ID to an authorized person upon explicit human audit."""
    assoc_id = incident_db.link_track_to_person(
        session_id=session_id,
        track_id=track_id,
        person_id=body.person_id,
        linked_by=body.linked_by,
        notes=body.notes or "",
    )
    return {
        "success": True,
        "message": f"Track #{track_id} in session '{session_id}' successfully linked to person '{body.person_id}'.",
        "association_id": assoc_id,
    }


@app.get("/api/incidents/{incident_id}")
def get_incident(incident_id: str):
    """Retrieves a single incident record by ID."""
    inc = incident_db.get_incident(incident_id)
    if not inc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident '{incident_id}' not found."
        )
    return {
        "success": True,
        "incident": _sanitize_incident_response(inc),
    }


@app.get("/api/incidents/{incident_id}/evidence")
def get_incident_evidence(incident_id: str):
    """Streams the evidence frame screenshot safely without exposing local filesystem paths."""
    inc = incident_db.get_incident(incident_id)
    if not inc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident '{incident_id}' not found."
        )
    img_rel = inc.get("evidence_image_path")
    if not img_rel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No evidence screenshot was captured for this incident."
        )
    full_path = ROOT_DIR / img_rel if not Path(img_rel).is_absolute() else Path(img_rel)
    if not full_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evidence image file not found on disk."
        )
    return FileResponse(str(full_path), media_type="image/jpeg")


@app.patch("/api/incidents/{incident_id}/review")
def update_incident_review(incident_id: str, body: ReviewUpdateRequest):
    """Updates human reviewer audit conclusion, reviewer identity, and notes."""
    inc = incident_db.get_incident(incident_id)
    if not inc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident '{incident_id}' not found."
        )
    try:
        updated = incident_db.update_review(
            incident_id=incident_id,
            review_status=body.review_status,
            reviewer_notes=body.reviewer_notes or "",
            reviewer_name=body.reviewer_name,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    updated_inc = incident_db.get_incident(incident_id)
    return {
        "success": True,
        "message": f"Incident '{incident_id}' review status updated to '{body.review_status}'.",
        "incident": _sanitize_incident_response(updated_inc),
    }


@app.delete("/api/incidents/{incident_id}")
def delete_incident(incident_id: str):
    """Deletes an incident and safely removes its associated evidence screenshot."""
    success = incident_db.delete_incident(incident_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident '{incident_id}' not found."
        )
    return {
        "success": True,
        "message": f"Incident '{incident_id}' and evidence safely deleted.",
    }


@app.post("/api/incidents/retention/apply")
def apply_retention(retention_days: int = Query(30, ge=1, le=365)):
    """Applies data retention policy, removing incidents and evidence older than retention_days."""
    deleted_count = incident_db.apply_retention_policy(retention_days=retention_days)
    return {
        "success": True,
        "retention_days": retention_days,
        "deleted_count": deleted_count,
    }


# ==============================================================================
# Live Webcam Analysis Endpoints
# ==============================================================================

class WebcamFrameRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded JPEG/PNG frame from browser camera")


@app.post("/api/webcam/start")
def webcam_start():
    """
    Starts a live webcam analysis session.
    Resets the dedicated webcam pipeline and incident detector state.
    Only one webcam session can be active at a time.
    """
    global _webcam_session
    if _webcam_session["active"]:
        return {
            "success": True,
            "message": "Webcam session already active.",
            "session_id": _webcam_session["session_id"],
            "video_id": _webcam_session["video_id"],
        }

    video_id = f"webcam_{uuid.uuid4().hex[:8]}"
    session_id = f"ses_cam_{uuid.uuid4().hex[:8]}"

    # Initialize dedicated webcam pipeline (thread-safe)
    with _webcam_pipeline_lock:
        pipeline = get_webcam_pipeline()
        pipeline.reset_all()

    # Create a dedicated incident detector for this webcam session
    cam_incident_detector = IncidentDetector(db=incident_db, evidence_dir=EVIDENCE_DIR)
    cam_incident_detector.reset_session(session_id=session_id)

    _webcam_session.update({
        "active": True,
        "session_id": session_id,
        "video_id": video_id,
        "frame_idx": 0,
        "start_time": time.time(),
        "fps": 25.0,
        "incident_detector": cam_incident_detector,
        "total_detections": 0,
        "total_behaviour_preds": 0,
        "total_emotion_preds": 0,
        "all_track_ids": set(),
        "person_telemetry_history": {},
    })

    # Register session in database
    try:
        incident_db.create_session({
            "id": session_id,
            "video_id": video_id,
            "source_filename": "Live Webcam",
            "total_frames": 0,
            "fps": 25.0,
            "status": "processing",
            "created_at": time.time(),
        })
    except Exception as e:
        print(f"[API] Warning initializing webcam session {session_id}: {e}")

    print(f"[API] Webcam session started: {session_id} (video_id={video_id})")
    return {
        "success": True,
        "message": "Webcam session started.",
        "session_id": session_id,
        "video_id": video_id,
    }


@app.post("/api/webcam/frame")
def webcam_process_frame(body: WebcamFrameRequest):
    """
    Processes a single webcam frame through the dedicated pipeline.
    Expects a base64-encoded JPEG/PNG image.
    Returns annotated frame (base64 JPEG) and detection/behaviour telemetry.
    Maintains pipeline/tracking state across frames (no reset between frames).
    """
    global _webcam_session

    if not _webcam_session["active"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active webcam session. Call /api/webcam/start first.",
        )

    # Decode base64 image
    try:
        img_data = body.image
        # Strip data URL prefix if present (e.g., "data:image/jpeg;base64,...")
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode frame image: {e}",
        )

    if frame is None or frame.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Decoded frame is empty or invalid.",
        )

    _webcam_session["frame_idx"] += 1
    frame_idx = _webcam_session["frame_idx"]
    session_id = _webcam_session["session_id"]
    video_id = _webcam_session["video_id"]
    elapsed = time.time() - _webcam_session["start_time"]
    cam_fps = round(frame_idx / elapsed, 1) if elapsed > 0 else 0.0

    # Process frame through dedicated webcam pipeline (with lock for thread safety)
    with _webcam_pipeline_lock:
        pipeline = get_webcam_pipeline()
        try:
            annotated_frame, records, faces = pipeline.process_frame(
                frame, frame_idx=frame_idx, total_frames=0, fps=cam_fps,
            )
        except Exception as e:
            print(f"[API] Webcam pipeline error on frame {frame_idx}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Pipeline processing error: {e}",
            )

    # Feed records to incident detector
    cam_det = _webcam_session["incident_detector"]
    new_incidents = []
    if cam_det is not None:
        try:
            new_incidents = cam_det.process_frame(
                frame_idx=frame_idx,
                timestamp_sec=round(elapsed, 3),
                annotated_frame=annotated_frame,
                integrated_records=records,
                video_id=video_id,
                source_filename="Live Webcam",
            )
        except Exception as e:
            print(f"[API] Webcam incident detector warning on frame {frame_idx}: {e}")

    # Accumulate telemetry
    for r in records:
        tid = r["track_id"]
        if tid > 0:
            _webcam_session["all_track_ids"].add(tid)
        beh = r.get("behaviour", {})
        emo = r.get("emotion", {})
        if beh.get("status") == "predicted":
            _webcam_session["total_behaviour_preds"] += 1
        if emo.get("status") == "available":
            _webcam_session["total_emotion_preds"] += 1

    _webcam_session["total_detections"] += len(records)

    # Encode annotated frame to base64 JPEG for frontend display and store raw bytes for MJPEG stream
    try:
        _, jpeg_buf = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        annotated_b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")
        # Save raw JPEG bytes for MJPEG streaming endpoint
        _webcam_session["latest_frame_bytes"] = jpeg_buf.tobytes()
    except Exception:
        annotated_b64 = ""
        _webcam_session["latest_frame_bytes"] = b""
    # Build per-person summary for this frame
    people_in_frame = []
    for r in records:
        beh = r.get("behaviour", {})
        emo = r.get("emotion", {})
        people_in_frame.append({
            "track_id": r["track_id"],
            "bbox": r.get("person_bbox", []),
            "det_confidence": round(r.get("det_confidence", 0.0), 3),
            "behaviour": {
                "status": beh.get("status", "unknown"),
                "class_name": beh.get("class_name", "Unknown"),
                "confidence": round(beh.get("confidence", 0.0), 3),
                "risk_level": beh.get("risk_level", "—"),
                "collected": beh.get("collected", 0),
            },
            "emotion": {
                "status": emo.get("status", "unavailable"),
                "label": emo.get("label", "Unavailable"),
                "confidence": round(emo.get("confidence", 0.0), 3),
            },
        })

    return {
        "success": True,
        "frame_idx": frame_idx,
        "elapsed_sec": round(elapsed, 2),
        "processing_fps": cam_fps,
        "annotated_frame": annotated_b64,
        "people_count": len(records),
        "people": people_in_frame,
        "persistent_tracks": len(_webcam_session["all_track_ids"]),
        "total_behaviour_preds": _webcam_session["total_behaviour_preds"],
        "total_emotion_preds": _webcam_session["total_emotion_preds"],
        "new_incidents": new_incidents,
        "total_incidents": len(cam_det.session_incidents) if cam_det else 0,
    }


@app.post("/api/webcam/stop")
def webcam_stop():
    """
    Stops the active webcam session, finalizes the session in the database,
    and resets the dedicated webcam pipeline state.
    """
    global _webcam_session

    if not _webcam_session["active"]:
        return {
            "success": True,
            "message": "No active webcam session to stop.",
        }

    session_id = _webcam_session["session_id"]
    video_id = _webcam_session["video_id"]
    total_frames = _webcam_session["frame_idx"]
    elapsed = time.time() - _webcam_session["start_time"]
    cam_det = _webcam_session["incident_detector"]
    incident_count = len(cam_det.session_incidents) if cam_det else 0

    # Finalize session in database
    try:
        incident_db.update_session(
            session_id=session_id,
            status="completed",
            completed_at=time.time(),
            metrics={
                "total_frames_processed": total_frames,
                "persistent_tracks_count": len(_webcam_session["all_track_ids"]),
                "recorded_incidents_count": incident_count,
                "processing_fps": round(total_frames / elapsed, 2) if elapsed > 0 else 0.0,
                "source": "webcam",
            },
        )
    except Exception as e:
        print(f"[API] Warning finalizing webcam session {session_id}: {e}")

    # Reset webcam pipeline state
    with _webcam_pipeline_lock:
        pipeline = get_webcam_pipeline()
        pipeline.reset_all()

    # Clear session state
    _webcam_session.update({
        "active": False,
        "session_id": None,
        "video_id": None,
        "frame_idx": 0,
        "start_time": 0.0,
        "fps": 25.0,
        "incident_detector": None,
        "total_detections": 0,
        "total_behaviour_preds": 0,
        "total_emotion_preds": 0,
        "all_track_ids": set(),
        "person_telemetry_history": {},
    })

    print(f"[API] Webcam session stopped: {session_id} ({total_frames} frames, {incident_count} incidents)")
    return {
        "success": True,
        "message": "Webcam session stopped and finalized.",
        "session_id": session_id,
        "video_id": video_id,
        "total_frames_processed": total_frames,
        "elapsed_seconds": round(elapsed, 2),
        "recorded_incidents_count": incident_count,
    }


@app.get("/api/webcam/status")
def webcam_status():
    """Returns the current webcam session status."""
    if not _webcam_session["active"]:
        return {"active": False}

    elapsed = time.time() - _webcam_session["start_time"]
    cam_det = _webcam_session["incident_detector"]
    return {
        "active": True,
        "session_id": _webcam_session["session_id"],
        "video_id": _webcam_session["video_id"],
        "frame_idx": _webcam_session["frame_idx"],
        "elapsed_sec": round(elapsed, 2),
        "persistent_tracks": len(_webcam_session["all_track_ids"]),
        "total_behaviour_preds": _webcam_session["total_behaviour_preds"],
        "total_emotion_preds": _webcam_session["total_emotion_preds"],
        "total_incidents": len(cam_det.session_incidents) if cam_det else 0,
    }


# Mount static assets (CSS, JS)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def get_dashboard():
    """Serves the main single-page frontend application."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({
            "message": "Student Safety Monitoring System API",
            "dashboard": "Frontend index.html not yet initialized.",
            "docs_url": "/docs",
        })
    return FileResponse(str(index_file), media_type="text/html")
