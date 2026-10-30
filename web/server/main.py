"""
Main FastAPI Server for Dual-Arm Robotic Pick-and-Handover Platform.

Integrates REST control endpoints, high-frequency WebSocket telemetry,
and real-time MJPEG multi-camera video streaming.
"""

import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure workspace root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from web.server.simulation_service import SimulationService
from web.server.routers import video_router, telemetry_router, system_router, dataset_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: initializes simulation engine on start, cleanly frees resources on shutdown."""
    print("=" * 70)
    print("  INITIALIZING DUAL-ARM ROBOTIC PLATFORM (PHASE 1 BACKEND)")
    print("=" * 70)
    # Warm up SimulationService singleton
    service = SimulationService.get_instance()
    print(f"  MuJoCo Model Loaded: {service.model.nq} DOF, {service.model.nu} Actuators")
    print(f"  Active Cameras     : {service.get_camera_list()}")
    print("=" * 70)
    yield
    print("[FastAPI Lifespan] Shutting down simulation service...")
    service.shutdown()


app = FastAPI(
    title="Dual-Arm Robotic Pick-and-Handover Platform",
    description="Full-Stack Control and Telemetry Engine for Bimanual UR10e Manipulation",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API & WebSocket routers
app.include_router(system_router)
app.include_router(video_router)
app.include_router(telemetry_router)
app.include_router(dataset_router)

# Mount recordings directory for demonstration video playback
RECORDINGS_DIR = os.path.abspath(os.path.join(ROOT_DIR, "recordings"))
os.makedirs(RECORDINGS_DIR, exist_ok=True)
app.mount("/recordings", StaticFiles(directory=RECORDINGS_DIR), name="recordings")

# Mount frontend static directory
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server.main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
