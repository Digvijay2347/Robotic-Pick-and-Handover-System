"""
High-Frequency Telemetry Router & WebSocket Broadcaster.

Publishes 14-DOF joint positions, end-effector poses, contact forces (F1, F2),
active FSM state, and safety alerts over WebSocket at 20 Hz.
"""

import asyncio
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..simulation_service import SimulationService

router = APIRouter(tags=["Telemetry"])


@router.get("/api/telemetry/current", summary="Get single telemetry snapshot")
def get_current_telemetry() -> Dict[str, Any]:
    """Returns the most recent simulation telemetry snapshot via REST."""
    service = SimulationService.get_instance()
    return service.get_telemetry()


@router.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint streaming live robotics telemetry at 20 Hz.
    Sends JSON packets containing 14-DOF joint positions, contact dynamics,
    and FSM state transitions.
    """
    await websocket.accept()
    service = SimulationService.get_instance()

    async def incoming_reader():
        """Reads client messages (e.g. heartbeat pings) so connection remains open."""
        try:
            while True:
                msg = await websocket.receive_text()
                # Respond to ping if requested
                if "ping" in msg.lower():
                    await websocket.send_text('{"event": "pong"}')
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass

    reader_task = asyncio.create_task(incoming_reader())

    try:
        while True:
            # Extract latest telemetry
            telemetry_data = service.get_telemetry()
            await websocket.send_json(telemetry_data)
            await asyncio.sleep(0.05)  # 20 Hz broadcast frequency
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass
