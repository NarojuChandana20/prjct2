"""
Incident Management & Review Service (Stage 9C + Multi-Session Refactoring)
---------------------------------------------------------------------------
Provides SQLite-backed persistence, automated incident detection with per-track
persistence and cooldown deduplication, session management, evidence screenshot capture,
and reviewer audit logging.

Design Constraints & Ethics:
- Triggers ONLY from actual body behaviour predictions reporting HIGH risk (Fight / Fall).
- Never triggers from facial emotion alone.
- Facial emotion is kept strictly separate as contextual telemetry.
- Temporary tracking IDs are explicitly flagged as ephemeral tracker tokens,
  never associated with real-world identity.
- No automated facial recognition or face embedding matching. Person-track linking
  requires explicit authorized human audit review.
- No fake predictions, fabricated incidents, or placeholder screenshots.
- Zero external database dependencies (uses standard library sqlite3).
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
STORAGE_DIR = ROOT_DIR / "storage"
EVIDENCE_DIR = STORAGE_DIR / "evidence"
DB_PATH = STORAGE_DIR / "incidents.db"

# Ensure storage directories exist
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


class IncidentDatabase:
    """Manages SQLite storage for sessions, tracks, incidents, evidence, and audit logs."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes database schema with safe migrations for existing tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Analysis Sessions Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    source_filename TEXT NOT NULL,
                    total_frames INTEGER NOT NULL,
                    fps REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    completed_at REAL,
                    metrics_json TEXT
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_video_id ON analysis_sessions(video_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON analysis_sessions(created_at);")

            # 2. Tracks Table (Aggregated track metrics per session)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tracks (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    is_persistent INTEGER NOT NULL DEFAULT 1,
                    first_frame INTEGER NOT NULL,
                    last_frame INTEGER NOT NULL,
                    total_frames INTEGER NOT NULL,
                    primary_behaviour TEXT,
                    primary_emotion TEXT,
                    highest_conf REAL,
                    highest_risk TEXT,
                    metadata_json TEXT,
                    FOREIGN KEY(session_id) REFERENCES analysis_sessions(id)
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracks_session_id ON tracks(session_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracks_track_id ON tracks(track_id);")

            # 3. Incidents Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    source_filename TEXT NOT NULL,
                    track_id INTEGER,
                    behaviour_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    start_frame INTEGER NOT NULL,
                    end_frame INTEGER NOT NULL,
                    event_time_sec REAL NOT NULL,
                    evidence_image_path TEXT,
                    associated_emotion TEXT,
                    emotion_confidence REAL,
                    review_status TEXT NOT NULL DEFAULT 'pending',
                    reviewer_notes TEXT DEFAULT '',
                    reviewed_at REAL,
                    reviewed_by TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    dedup_signature TEXT,
                    session_id TEXT
                );
            """)

            # Safe Schema Migration: Ensure session_id column exists on existing incidents table
            cursor.execute("PRAGMA table_info(incidents);")
            columns = [row["name"] for row in cursor.fetchall()]
            if "session_id" not in columns:
                cursor.execute("ALTER TABLE incidents ADD COLUMN session_id TEXT;")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_video_id ON incidents(video_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_session_id ON incidents(session_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_review_status ON incidents(review_status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_behaviour ON incidents(behaviour_label);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_created_at ON incidents(created_at);")

            # 4. Evidence Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evidence (
                    id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    session_id TEXT,
                    frame_idx INTEGER NOT NULL,
                    timestamp_sec REAL NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(incident_id) REFERENCES incidents(id)
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evidence_incident_id ON evidence(incident_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evidence_session_id ON evidence(session_id);")

            # 5. Verified Person Registry Table (Manual authorized records; NO automated facial recognition)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS verified_persons (
                    id TEXT PRIMARY KEY,
                    official_id TEXT UNIQUE NOT NULL,
                    full_name TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'student',
                    profile_photo_path TEXT,
                    notes TEXT DEFAULT '',
                    created_at REAL NOT NULL
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_verified_persons_official_id ON verified_persons(official_id);")

            # 6. Person Track Associations Table (Explicit manual linkage by authorized staff)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS person_track_associations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    person_id TEXT NOT NULL,
                    linked_by TEXT NOT NULL,
                    linked_at REAL NOT NULL,
                    notes TEXT DEFAULT '',
                    FOREIGN KEY(person_id) REFERENCES verified_persons(id)
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_associations_session_track ON person_track_associations(session_id, track_id);")

            # 7. Safe Migration of Legacy Incidents: Link rows with NULL session_id to synthetic legacy sessions
            cursor.execute("SELECT DISTINCT video_id, source_filename FROM incidents WHERE session_id IS NULL OR session_id = '';")
            legacy_videos = cursor.fetchall()
            for leg in legacy_videos:
                vid = leg["video_id"]
                s_name = leg["source_filename"]
                leg_session_id = f"ses_legacy_{vid[:8]}"

                cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM incidents WHERE video_id = ?;", (vid,))
                times = cursor.fetchone()
                t_created = times[0] if times and times[0] else time.time()
                t_completed = times[1] if times and times[1] else t_created

                cursor.execute("""
                    INSERT OR IGNORE INTO analysis_sessions (
                        id, video_id, source_filename, total_frames, fps, status, created_at, completed_at, metrics_json
                    ) VALUES (?, ?, ?, ?, ?, 'completed', ?, ?, '{}');
                """, (leg_session_id, vid, s_name, 0, 25.0, t_created, t_completed))

                cursor.execute("""
                    UPDATE incidents SET session_id = ? WHERE video_id = ? AND (session_id IS NULL OR session_id = '');
                """, (leg_session_id, vid))

            # 8. Safe Migration of Legacy Evidence: Populate evidence table for existing incidents with images
            cursor.execute("""
                SELECT id, session_id, start_frame, event_time_sec, evidence_image_path, created_at
                FROM incidents
                WHERE evidence_image_path IS NOT NULL AND evidence_image_path != '';
            """)
            for row in cursor.fetchall():
                ev_id = f"ev_{row['id']}"
                cursor.execute("""
                    INSERT OR IGNORE INTO evidence (
                        id, incident_id, session_id, frame_idx, timestamp_sec, file_path, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """, (
                    ev_id,
                    row["id"],
                    row["session_id"] or "",
                    row["start_frame"],
                    row["event_time_sec"],
                    row["evidence_image_path"],
                    row["created_at"],
                ))

            conn.commit()

    # -------------------------------------------------------------------------
    # Analysis Sessions
    # -------------------------------------------------------------------------

    def create_session(self, session_data: Dict[str, Any]) -> str:
        """Creates a new analysis session record."""
        session_id = session_data.get("id") or f"ses_{uuid.uuid4().hex[:10]}"
        now = time.time()
        metrics = session_data.get("metrics_json")
        if isinstance(metrics, dict):
            metrics_str = json.dumps(metrics)
        else:
            metrics_str = metrics or "{}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO analysis_sessions (
                    id, video_id, source_filename, total_frames, fps, status, created_at, completed_at, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                session_id,
                session_data["video_id"],
                session_data["source_filename"],
                int(session_data.get("total_frames", 0)),
                float(session_data.get("fps", 25.0)),
                session_data.get("status", "processing"),
                float(session_data.get("created_at", now)),
                session_data.get("completed_at"),
                metrics_str,
            ))
            conn.commit()
        return session_id

    def update_session(
        self,
        session_id: str,
        status: str,
        completed_at: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Updates session status, completion timestamp, and summary metrics."""
        metrics_str = json.dumps(metrics) if isinstance(metrics, dict) else metrics
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE analysis_sessions
                SET status = ?,
                    completed_at = COALESCE(?, completed_at),
                    metrics_json = COALESCE(?, metrics_json)
                WHERE id = ?;
            """, (status, completed_at, metrics_str, session_id))
            conn.commit()
            return cursor.rowcount > 0

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single session with parsed metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analysis_sessions WHERE id = ?;", (session_id,))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            try:
                res["metrics"] = json.loads(res.get("metrics_json") or "{}")
            except Exception:
                res["metrics"] = {}
            return res

    def list_sessions(self, video_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists analysis sessions, optionally filtered by video_id."""
        query = "SELECT * FROM analysis_sessions"
        params: List[Any] = []
        if video_id:
            query += " WHERE video_id = ?"
            params.append(video_id)
        query += " ORDER BY created_at DESC LIMIT ?;"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = []
            for r in cursor.fetchall():
                d = dict(r)
                try:
                    d["metrics"] = json.loads(d.get("metrics_json") or "{}")
                except Exception:
                    d["metrics"] = {}
                rows.append(d)
            return rows

    # -------------------------------------------------------------------------
    # Tracks (Session-Scoped Telemetry Aggregation)
    # -------------------------------------------------------------------------

    def insert_track(self, track_data: Dict[str, Any]) -> str:
        """Inserts or replaces an aggregated track summary for a session."""
        track_pk = track_data.get("id") or f"trk_{track_data['session_id']}_{track_data['track_id']}"
        meta = track_data.get("metadata_json")
        if isinstance(meta, dict):
            meta_str = json.dumps(meta)
        else:
            meta_str = meta or "{}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO tracks (
                    id, session_id, track_id, is_persistent, first_frame, last_frame,
                    total_frames, primary_behaviour, primary_emotion, highest_conf,
                    highest_risk, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                track_pk,
                track_data["session_id"],
                int(track_data["track_id"]),
                1 if track_data.get("is_persistent", True) else 0,
                int(track_data["first_frame"]),
                int(track_data["last_frame"]),
                int(track_data["total_frames"]),
                track_data.get("primary_behaviour", "Unavailable"),
                track_data.get("primary_emotion", "Unavailable"),
                float(track_data.get("highest_conf", 0.0)),
                track_data.get("highest_risk", "—"),
                meta_str,
            ))
            conn.commit()
        return track_pk

    def list_tracks(self, session_id: str) -> List[Dict[str, Any]]:
        """Lists aggregated tracks belonging to an analysis session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks WHERE session_id = ? ORDER BY track_id ASC;", (session_id,))
            rows = []
            for r in cursor.fetchall():
                d = dict(r)
                try:
                    d["metadata"] = json.loads(d.get("metadata_json") or "{}")
                except Exception:
                    d["metadata"] = {}
                rows.append(d)
            return rows

    # -------------------------------------------------------------------------
    # Evidence
    # -------------------------------------------------------------------------

    def insert_evidence(self, ev_data: Dict[str, Any]) -> str:
        """Inserts a screenshot evidence record linked to an incident."""
        ev_id = ev_data.get("id") or f"ev_{uuid.uuid4().hex[:10]}"
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO evidence (
                    id, incident_id, session_id, frame_idx, timestamp_sec, file_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (
                ev_id,
                ev_data["incident_id"],
                ev_data.get("session_id"),
                int(ev_data["frame_idx"]),
                float(ev_data["timestamp_sec"]),
                ev_data["file_path"],
                float(ev_data.get("created_at", now)),
            ))
            conn.commit()
        return ev_id

    def list_evidence(
        self,
        incident_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves evidence files for a specific incident or session."""
        query = "SELECT * FROM evidence WHERE 1=1"
        params: List[Any] = []
        if incident_id:
            query += " AND incident_id = ?"
            params.append(incident_id)
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY frame_idx ASC;"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Incidents
    # -------------------------------------------------------------------------

    def insert_incident(self, incident: Dict[str, Any]) -> str:
        """Inserts or replaces an incident record and synchronizes evidence."""
        now = time.time()
        inc_id = incident.get("id") or f"inc_{uuid.uuid4().hex[:10]}"
        sess_id = incident.get("session_id")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO incidents (
                    id, video_id, source_filename, track_id, behaviour_label,
                    confidence, risk_level, start_frame, end_frame, event_time_sec,
                    evidence_image_path, associated_emotion, emotion_confidence,
                    review_status, reviewer_notes, reviewed_at, reviewed_by,
                    created_at, updated_at, dedup_signature, session_id
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?
                );
            """, (
                inc_id,
                incident["video_id"],
                incident["source_filename"],
                incident.get("track_id"),
                incident["behaviour_label"],
                float(incident["confidence"]),
                incident.get("risk_level", "High"),
                int(incident["start_frame"]),
                int(incident["end_frame"]),
                float(incident["event_time_sec"]),
                incident.get("evidence_image_path"),
                incident.get("associated_emotion"),
                float(incident.get("emotion_confidence", 0.0)) if incident.get("emotion_confidence") is not None else 0.0,
                incident.get("review_status", "pending"),
                incident.get("reviewer_notes", ""),
                incident.get("reviewed_at"),
                incident.get("reviewed_by"),
                now,
                now,
                incident.get("dedup_signature", ""),
                sess_id,
            ))

            # Auto-sync evidence entry if image path provided
            img_path = incident.get("evidence_image_path")
            if img_path:
                cursor.execute("""
                    INSERT OR REPLACE INTO evidence (
                        id, incident_id, session_id, frame_idx, timestamp_sec, file_path, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """, (
                    f"ev_{inc_id}",
                    inc_id,
                    sess_id or "",
                    int(incident["start_frame"]),
                    float(incident["event_time_sec"]),
                    img_path,
                    now,
                ))

            conn.commit()
        return inc_id

    def update_incident_event_span(self, inc_id: str, end_frame: int, confidence: float) -> None:
        """Updates end frame and highest confidence for an ongoing event."""
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE incidents
                SET end_frame = ?,
                    confidence = MAX(confidence, ?),
                    updated_at = ?
                WHERE id = ?;
            """, (int(end_frame), float(confidence), now, inc_id))
            conn.commit()

    def update_review(
        self,
        incident_id: str,
        review_status: str,
        reviewer_notes: str,
        reviewer_name: Optional[str] = None
    ) -> bool:
        """Updates reviewer status and auditor notes for an incident."""
        now = time.time()
        valid_statuses = {"pending", "reviewed", "dismissed"}
        if review_status.lower() not in valid_statuses:
            raise ValueError(f"Invalid review status '{review_status}'. Must be one of {valid_statuses}")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE incidents
                SET review_status = ?,
                    reviewer_notes = ?,
                    reviewed_at = ?,
                    reviewed_by = ?,
                    updated_at = ?
                WHERE id = ?;
            """, (
                review_status.lower(),
                reviewer_notes.strip(),
                now,
                reviewer_name.strip() if reviewer_name else "Authorized Reviewer",
                now,
                incident_id
            ))
            conn.commit()
            return cursor.rowcount > 0

    def get_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single incident by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM incidents WHERE id = ?;", (incident_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return dict(row)

    def list_incidents(
        self,
        video_id: Optional[str] = None,
        session_id: Optional[str] = None,
        review_status: Optional[str] = None,
        behaviour_label: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Lists incidents with optional filtering by video, session, review status, or behaviour."""
        query = "SELECT * FROM incidents WHERE 1=1"
        count_query = "SELECT COUNT(*) FROM incidents WHERE 1=1"
        params: List[Any] = []

        if video_id:
            query += " AND video_id = ?"
            count_query += " AND video_id = ?"
            params.append(video_id)

        if session_id:
            query += " AND session_id = ?"
            count_query += " AND session_id = ?"
            params.append(session_id)

        if review_status:
            query += " AND LOWER(review_status) = LOWER(?)"
            count_query += " AND LOWER(review_status) = LOWER(?)"
            params.append(review_status)

        if behaviour_label:
            query += " AND LOWER(behaviour_label) = LOWER(?)"
            count_query += " AND LOWER(behaviour_label) = LOWER(?)"
            params.append(behaviour_label)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(count_query, params)
            total_count = cursor.fetchone()[0]

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?;"
            paginated_params = params + [limit, offset]
            cursor.execute(query, paginated_params)
            rows = [dict(r) for r in cursor.fetchall()]

            return rows, total_count

    def get_statistics(
        self,
        video_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Returns overall or scoped incident statistics."""
        filter_sql = ""
        params: List[Any] = []
        if video_id:
            filter_sql += " AND video_id = ?"
            params.append(video_id)
        if session_id:
            filter_sql += " AND session_id = ?"
            params.append(session_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(f"SELECT COUNT(*) FROM incidents WHERE 1=1{filter_sql};", params)
            total = cursor.fetchone()[0]

            cursor.execute(f"SELECT COUNT(*) FROM incidents WHERE LOWER(review_status) = 'pending'{filter_sql};", params)
            pending = cursor.fetchone()[0]

            cursor.execute(f"SELECT COUNT(*) FROM incidents WHERE LOWER(review_status) = 'reviewed'{filter_sql};", params)
            reviewed = cursor.fetchone()[0]

            cursor.execute(f"SELECT COUNT(*) FROM incidents WHERE LOWER(review_status) = 'dismissed'{filter_sql};", params)
            dismissed = cursor.fetchone()[0]

            cursor.execute(f"SELECT behaviour_label, COUNT(*) FROM incidents WHERE 1=1{filter_sql} GROUP BY behaviour_label;", params)
            by_behaviour = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                "total_incidents": total,
                "pending_review": pending,
                "reviewed": reviewed,
                "dismissed": dismissed,
                "by_behaviour": by_behaviour,
            }

    def delete_incident(self, incident_id: str) -> bool:
        """Deletes an incident and removes any associated evidence image file."""
        incident = self.get_incident(incident_id)
        if not incident:
            return False

        # Safely remove evidence image
        img_path = incident.get("evidence_image_path")
        if img_path:
            full_path = ROOT_DIR / img_path if not Path(img_path).is_absolute() else Path(img_path)
            try:
                if full_path.exists():
                    full_path.unlink()
            except Exception as e:
                print(f"[IncidentDB] Warning deleting evidence file {img_path}: {e}")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM evidence WHERE incident_id = ?;", (incident_id,))
            cursor.execute("DELETE FROM incidents WHERE id = ?;", (incident_id,))
            conn.commit()
            return cursor.rowcount > 0

    def apply_retention_policy(self, retention_days: int = 30) -> int:
        """Deletes incidents and associated evidence files older than retention_days."""
        if retention_days <= 0:
            return 0

        cutoff = time.time() - (retention_days * 86400)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM incidents WHERE created_at < ?;", (cutoff,))
            expired_ids = [row[0] for row in cursor.fetchall()]

        deleted_count = 0
        for inc_id in expired_ids:
            if self.delete_incident(inc_id):
                deleted_count += 1

        return deleted_count

    # -------------------------------------------------------------------------
    # Optional Verified Person Registry (Authorized Reviewer Identity Linking)
    # Strict Note: NO automated facial recognition or embeddings.
    # -------------------------------------------------------------------------

    def create_verified_person(self, person_data: Dict[str, Any]) -> str:
        """Registers an authorized official person record in the registry."""
        p_id = person_data.get("id") or f"per_{uuid.uuid4().hex[:8]}"
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO verified_persons (
                    id, official_id, full_name, role, profile_photo_path, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (
                p_id,
                person_data["official_id"],
                person_data["full_name"],
                person_data.get("role", "student"),
                person_data.get("profile_photo_path"),
                person_data.get("notes", ""),
                now,
            ))
            conn.commit()
        return p_id

    def get_verified_person(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a verified person by ID or official_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM verified_persons WHERE id = ? OR official_id = ?;", (person_id, person_id))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_verified_persons(self) -> List[Dict[str, Any]]:
        """Lists registered persons in the registry."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM verified_persons ORDER BY full_name ASC;")
            return [dict(r) for r in cursor.fetchall()]

    def link_track_to_person(
        self,
        session_id: str,
        track_id: int,
        person_id: str,
        linked_by: str,
        notes: str = ""
    ) -> str:
        """Links a temporary tracking ID to an authorized person record upon human review."""
        assoc_id = f"asc_{session_id}_{track_id}"
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO person_track_associations (
                    id, session_id, track_id, person_id, linked_by, linked_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (assoc_id, session_id, int(track_id), person_id, linked_by, now, notes))
            conn.commit()
        return assoc_id

    def get_track_person_association(self, session_id: str, track_id: int) -> Optional[Dict[str, Any]]:
        """Gets authorized person linkage for a temporary track, if one exists."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT a.*, p.official_id, p.full_name, p.role
                FROM person_track_associations a
                JOIN verified_persons p ON a.person_id = p.id
                WHERE a.session_id = ? AND a.track_id = ?;
            """, (session_id, int(track_id)))
            row = cursor.fetchone()
            return dict(row) if row else None


class IncidentDetector:
    """
    Manages online and offline incident detection on video streams.
    Implements configurable persistence filtering and per-track cooldown deduplication.
    Strictly filters out negative ephemeral tracker IDs (tid < 0).
    """

    def __init__(
        self,
        db: Optional[IncidentDatabase] = None,
        min_persistence_frames: int = 2,
        cooldown_frames: int = 45,
        min_confidence: float = 0.50,
        evidence_dir: Path = EVIDENCE_DIR,
        session_id: Optional[str] = None,
    ):
        self.db = db or IncidentDatabase()
        self.min_persistence_frames = max(1, min_persistence_frames)
        self.cooldown_frames = max(5, cooldown_frames)
        self.min_confidence = min_confidence
        self.evidence_dir = evidence_dir
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.session_id: Optional[str] = session_id

        # Per-track event state machine
        # Maps track_id -> dict with tracking state
        self.track_states: Dict[int, Dict[str, Any]] = {}
        # Records created during current video session
        self.session_incidents: List[str] = []

    def reset_session(self, session_id: Optional[str] = None) -> None:
        """Resets per-track detection state for a new video analysis run."""
        self.track_states.clear()
        self.session_incidents.clear()
        self.session_id = session_id

    def process_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        annotated_frame: Optional[np.ndarray],
        integrated_records: List[Dict[str, Any]],
        video_id: str,
        source_filename: str,
    ) -> List[str]:
        """
        Processes a single frame's integrated records, updating track states and recording
        incidents when body behaviour reports a verified High Risk prediction on a persistent track.

        Strict Rules:
        - Only persistent positive tracks (tid > 0) can trigger incidents.
        - Negative pseudo-tracks (tid < 0) are ephemeral unassigned detections and are ignored.
        - Facial emotion ALONE must NEVER trigger an incident.
        - Ongoing events update frame span instead of creating duplicate records.
        """
        new_incidents_in_frame: List[str] = []
        active_tids_in_frame = set()

        for record in integrated_records:
            tid = record.get("track_id")

            # STRICT RULE: Ignore unassigned or temporary negative IDs.
            # Temporary tracking IDs (tid <= 0) must never trigger incidents!
            if tid is None or tid <= 0:
                continue

            active_tids_in_frame.add(tid)
            behaviour = record.get("behaviour") or {}
            emotion = record.get("emotion") or {}

            b_status = behaviour.get("status")
            risk_level = behaviour.get("risk_level", "—")
            cname = behaviour.get("class_name", "")
            cid = behaviour.get("class_id", -1)
            b_conf = float(behaviour.get("confidence", 0.0))

            # Strictly evaluate HIGH risk from body behaviour ONLY
            is_high_risk_body = (
                b_status == "predicted"
                and (
                    risk_level == "High"
                    or cid in [1, 2]
                    or ("fight" in cname.lower() and "non" not in cname.lower())
                    or ("fall" in cname.lower())
                )
                and b_conf >= self.min_confidence
            )

            # Standardize behaviour label
            if is_high_risk_body:
                if "fall" in cname.lower() or cid == 2:
                    beh_label = "Fall"
                else:
                    beh_label = "Fight"
            else:
                beh_label = "Normal"

            state = self.track_states.setdefault(tid, {
                "active_incident_id": None,
                "consecutive_high_frames": 0,
                "cooldown_counter": 0,
                "highest_conf": 0.0,
                "start_frame": frame_idx,
                "last_seen_frame": frame_idx,
                "beh_label": beh_label,
            })

            state["last_seen_frame"] = frame_idx

            if is_high_risk_body:
                state["highest_conf"] = max(state["highest_conf"], b_conf)
                state["beh_label"] = beh_label
                state["cooldown_counter"] = 0  # Reset cooldown since high-risk continues

                if state["active_incident_id"] is not None:
                    # Ongoing event continuation: update end frame and confidence in DB
                    self.db.update_incident_event_span(
                        inc_id=state["active_incident_id"],
                        end_frame=frame_idx,
                        confidence=state["highest_conf"]
                    )
                else:
                    # Increment persistence counter
                    state["consecutive_high_frames"] += 1
                    if state["consecutive_high_frames"] == 1:
                        state["event_start_frame"] = frame_idx
                        state["event_start_time"] = round(timestamp_sec, 2)

                    if state["consecutive_high_frames"] >= self.min_persistence_frames:
                        # Persistence threshold reached: Record genuine incident!
                        inc_id = f"inc_{video_id[:8]}_{tid}_{frame_idx}"
                        evidence_rel_path = self._save_evidence_screenshot(inc_id, annotated_frame)

                        # Decouple emotion telemetry cleanly
                        em_label = emotion.get("label", "Unavailable") if emotion.get("status") == "available" else "Unavailable"
                        em_conf = float(emotion.get("confidence", 0.0)) if emotion.get("status") == "available" else 0.0

                        event_start_f = state.get("event_start_frame", frame_idx)
                        event_start_t = state.get("event_start_time", round(timestamp_sec, 2))
                        sess_id = self.session_id or f"ses_{video_id[:8]}"

                        incident_record = {
                            "id": inc_id,
                            "video_id": video_id,
                            "session_id": sess_id,
                            "source_filename": source_filename,
                            "track_id": tid,
                            "behaviour_label": beh_label,
                            "confidence": state["highest_conf"],
                            "risk_level": "High",
                            "start_frame": event_start_f,
                            "end_frame": frame_idx,
                            "event_time_sec": event_start_t,
                            "evidence_image_path": evidence_rel_path,
                            "associated_emotion": em_label,
                            "emotion_confidence": em_conf,
                            "review_status": "pending",
                            "reviewer_notes": "",
                            "dedup_signature": f"{sess_id}_{tid}_{event_start_f}",
                        }

                        try:
                            self.db.insert_incident(incident_record)
                            state["active_incident_id"] = inc_id
                            self.session_incidents.append(inc_id)
                            new_incidents_in_frame.append(inc_id)
                            print(f"[IncidentDetector] Recorded High Risk Incident: {inc_id} ({beh_label}, Conf: {b_conf:.2f}, Track: {tid})")
                        except Exception as e:
                            print(f"[IncidentDetector] Database error recording incident {inc_id}: {e}")
            else:
                # Normal or unavailable body behaviour for this track
                state["consecutive_high_frames"] = 0
                if state["active_incident_id"] is not None:
                    # Event is subsiding: increment cooldown
                    state["cooldown_counter"] += 1
                    if state["cooldown_counter"] >= self.cooldown_frames:
                        # Cooldown complete: finalize event so future events on this track can trigger a new incident
                        state["active_incident_id"] = None
                        state["highest_conf"] = 0.0
                        state["consecutive_high_frames"] = 0
                        state["cooldown_counter"] = 0
                        state["start_frame"] = frame_idx + 1

        # Check tracks that were missing this frame
        for tid, state in list(self.track_states.items()):
            if tid not in active_tids_in_frame and state["active_incident_id"] is not None:
                state["cooldown_counter"] += 1
                if state["cooldown_counter"] >= self.cooldown_frames:
                    state["active_incident_id"] = None
                    state["highest_conf"] = 0.0
                    state["consecutive_high_frames"] = 0
                    state["cooldown_counter"] = 0

        return new_incidents_in_frame

    def _save_evidence_screenshot(self, inc_id: str, frame: Optional[np.ndarray]) -> Optional[str]:
        """Safely saves an annotated frame screenshot as evidence."""
        if frame is None or frame.size == 0:
            return None
        try:
            filename = f"{inc_id}.jpg"
            full_path = self.evidence_dir / filename
            cv2.imwrite(str(full_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            try:
                rel_path = str(full_path.relative_to(ROOT_DIR)).replace("\\", "/")
            except ValueError:
                rel_path = str(full_path).replace("\\", "/")
            return rel_path
        except Exception as e:
            print(f"[IncidentDetector] Warning: failed to save evidence screenshot {inc_id}: {e}")
            return None
