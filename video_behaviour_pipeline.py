"""
Video Behaviour Pipeline (Root Facade)
--------------------------------------
Re-exports the unified multi-branch video pipeline from
backend.app.services.video_behaviour_pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.video_behaviour_pipeline import (
    VideoBehaviourPipeline,
    UnifiedVideoPipeline,
    get_behaviour_tracker_class,
    main,
)

__all__ = [
    "VideoBehaviourPipeline",
    "UnifiedVideoPipeline",
    "get_behaviour_tracker_class",
    "main",
]

if __name__ == "__main__":
    main()
