"""
Student Safety Monitoring System — Unified Application Launcher
================================================================
Single-command launcher for the complete end-to-end application.
- Sets up system paths and working directories.
- Verifies model weights and assets.
- Detects whether server is already active on port 8000.
- Automatically launches FastAPI and opens browser dashboard.
"""

from __future__ import annotations

import sys
import os
import time
import socket
import webbrowser
import threading
from pathlib import Path
import urllib.request
import uvicorn

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure UTF-8 console output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def check_models_exist() -> bool:
    """Verifies that all 5 trained model files exist."""
    required_models = [
        ("YOLO11 Person Detector", ROOT_DIR / "weights" / "yolo" / "yolo11n.pt"),
        ("MediaPipe Pose Landmarker", ROOT_DIR / "weights" / "pose" / "pose_landmarker_lite.task"),
        ("BiLSTM Behaviour Model", ROOT_DIR / "weights" / "behaviour" / "behaviour_model_rich_165_v2_best.keras"),
        ("MediaPipe BlazeFace Detector", ROOT_DIR / "weights" / "face" / "blaze_face_short_range.tflite"),
        ("EfficientNetB0 Emotion Model", ROOT_DIR / "weights" / "emotion" / "rafdb_emotion_best.keras"),
    ]

    print("\n" + "=" * 65)
    print("STUDENT SAFETY MONITORING SYSTEM — MODEL INTEGRITY CHECK")
    print("=" * 65)
    all_ok = True
    for name, path in required_models:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {name:<30}: {path.name} ({size_mb:.2f} MB)")
        else:
            print(f"  [MISSING] {name:<26}: {path}")
            all_ok = False
    print("=" * 65 + "\n")
    return all_ok


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Checks if a local TCP port is already open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def is_app_responding(port: int = 8000, host: str = "127.0.0.1") -> bool:
    """Checks if our FastAPI application is healthy on the given port."""
    try:
        url = f"http://{host}:{port}/api/health"
        req = urllib.request.Request(url, headers={"User-Agent": "AntigravityLauncher"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def open_browser_delayed(url: str, delay_sec: float = 1.5):
    """Opens the web browser to the dashboard URL after server initialization."""
    def _open():
        time.sleep(delay_sec)
        print(f"\n[Browser] Opening dashboard: {url}")
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"[Browser] Note: Please open {url} manually in your browser ({e}).")

    t = threading.Thread(target=_open, daemon=True)
    t.start()


def main():
    print("=" * 65)
    print("STUDENT SAFETY MONITORING SYSTEM — APPLICATION STARTUP")
    print("=" * 65)
    print(f"Root Directory: {ROOT_DIR}")

    # 1. Verify models
    if not check_models_exist():
        print("[WARNING] Some model weights appear to be missing. Check weights/ folder.")

    target_host = "127.0.0.1"
    target_port = 8000
    dashboard_url = f"http://{target_host}:{target_port}/"

    # 2. Check if server is already running
    if is_port_in_use(target_port, target_host):
        if is_app_responding(target_port, target_host):
            print(f"[INFO] Student Safety Monitoring System is ALREADY RUNNING on {dashboard_url}!")
            print(f"[INFO] Opening dashboard in your default browser...")
            open_browser_delayed(dashboard_url, delay_sec=0.5)
            print("\nPress Ctrl+C to exit launcher, or keep this window open.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nLauncher closed.")
            return
        else:
            print(f"[WARNING] Port {target_port} is occupied by another process.")
            # Try port 8001 fallback
            target_port = 8001
            dashboard_url = f"http://{target_host}:{target_port}/"
            print(f"[INFO] Binding to fallback port {target_port}...")

    # 3. Launch Uvicorn
    print(f"\n[Server] Starting FastAPI server on http://{target_host}:{target_port}...")
    print(f"[Server] Dashboard URL: {dashboard_url}")
    print("[Server] Press Ctrl+C in this terminal to stop the server.")

    open_browser_delayed(dashboard_url, delay_sec=1.5)

    try:
        uvicorn.run(
            "backend.app.main:app",
            host=target_host,
            port=target_port,
            log_level="info",
            access_log=True,
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
