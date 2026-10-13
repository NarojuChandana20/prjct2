"""
Unified Pipeline Module
-----------------------
Re-exports UnifiedVideoPipeline and VideoBehaviourPipeline from
backend.app.services.video_behaviour_pipeline for backwards compatibility
and clean modular imports.
"""

from backend.app.services.video_behaviour_pipeline import (
    VideoBehaviourPipeline,
    UnifiedVideoPipeline,
    get_behaviour_tracker_class,
)

__all__ = [
    "VideoBehaviourPipeline",
    "UnifiedVideoPipeline",
    "get_behaviour_tracker_class",
]
