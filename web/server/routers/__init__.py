"""
Web Server API and WebSocket Routers.
"""

from .video_stream import router as video_router
from .telemetry import router as telemetry_router
from .system import router as system_router
from .dataset import router as dataset_router

__all__ = ["video_router", "telemetry_router", "system_router", "dataset_router"]

