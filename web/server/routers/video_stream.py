"""
Multi-Camera Video Streaming Router.

Provides MJPEG continuous video streaming and snapshot endpoints for all
MuJoCo offscreen camera feeds.
"""

import asyncio
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import StreamingResponse

from ..simulation_service import SimulationService

router = APIRouter(prefix="/api/stream", tags=["Video Streaming"])


@router.get("/cameras", summary="List available camera feeds")
def list_cameras() -> Dict[str, Any]:
    """Returns the registered camera names and viewing capabilities."""
    service = SimulationService.get_instance()
    cameras = service.get_camera_list()
    return {
        "cameras": cameras,
        "count": len(cameras),
        "resolution": {
            "width": service.render_width,
            "height": service.render_height,
        },
    }


@router.get("/snapshot/{camera_name}", summary="Capture single JPEG snapshot")
async def get_snapshot(camera_name: str, quality: int = Query(85, ge=20, le=100)):
    """Returns a single JPEG image snapshot from the specified camera."""
    service = SimulationService.get_instance()
    if camera_name not in service.camera_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{camera_name}' not found. Available: {service.get_camera_list()}",
        )

    loop = asyncio.get_event_loop()
    frame_bytes = await loop.run_in_executor(None, service.get_jpeg_frame, camera_name, quality)
    return Response(content=frame_bytes, media_type="image/jpeg")


async def mjpeg_frame_generator(camera_name: str, fps: int = 20, quality: int = 75):
    """Asynchronous generator yielding multipart/x-mixed-replace JPEG frames."""
    service = SimulationService.get_instance()
    interval = 1.0 / max(1, min(fps, 30))
    loop = asyncio.get_event_loop()

    try:
        while True:
            t0 = asyncio.get_event_loop().time()
            frame_bytes = await loop.run_in_executor(None, service.get_jpeg_frame, camera_name, quality)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
                + frame_bytes + b"\r\n"
            )
            elapsed = asyncio.get_event_loop().time() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                await asyncio.sleep(sleep_t)
            else:
                await asyncio.sleep(0.005)  # Yield control to event loop
    except asyncio.CancelledError:
        # Browser client closed the stream
        pass


@router.get("/feed/{camera_name}", summary="Stream continuous MJPEG video")
async def stream_feed(
    camera_name: str,
    fps: int = Query(20, ge=1, le=30),
    quality: int = Query(75, ge=20, le=95),
):
    """
    Streams a live multipart/x-mixed-replace MJPEG video stream from the specified camera.
    Supported cameras: overhead_cam, shelf_cam, handover_cam, arm1_wrist_cam, arm2_wrist_cam.
    """
    service = SimulationService.get_instance()
    if camera_name not in service.camera_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{camera_name}' not found. Available: {service.get_camera_list()}",
        )

    return StreamingResponse(
        mjpeg_frame_generator(camera_name, fps=fps, quality=quality),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
