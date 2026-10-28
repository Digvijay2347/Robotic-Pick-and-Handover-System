"""
System Health, Hardware Telemetry, and State Machine Execution Router.
"""

import time
import platform
import psutil
from typing import Dict, Any
from fastapi import APIRouter

from ..simulation_service import SimulationService

router = APIRouter(prefix="/api", tags=["System & Control"])

_SERVER_START_TIME = time.time()


@router.get("/system/status", summary="Host and simulation platform health")
def get_system_status() -> Dict[str, Any]:
    """Returns host hardware utilization, server uptime, and simulation status."""
    service = SimulationService.get_instance()
    mem = psutil.virtual_memory()

    return {
        "status": "online",
        "uptime_sec": round(time.time() - _SERVER_START_TIME, 1),
        "host": {
            "platform": platform.platform(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": mem.percent,
            "memory_used_gb": round(mem.used / (1024**3), 2),
            "memory_total_gb": round(mem.total / (1024**3), 2),
        },
        "simulation": {
            "is_running_fsm": service.is_running_fsm,
            "fsm_state": service.fsm.current_state.name,
            "cycle_count": service.cycle_count,
            "sim_time": round(float(service.fsm.sim_time), 3),
            "cameras": service.get_camera_list(),
            "last_summary": service.last_cycle_summary,
        },
    }


@router.post("/fsm/start", summary="Start autonomous pick-and-handover execution")
def start_fsm() -> Dict[str, Any]:
    """Triggers autonomous execution of the 16-stage pick-and-handover cycle."""
    service = SimulationService.get_instance()
    return service.start_fsm_cycle()


@router.post("/fsm/stop", summary="Emergency stop execution")
def stop_fsm() -> Dict[str, Any]:
    """Halts active FSM trajectory and commands arms to a safe posture."""
    service = SimulationService.get_instance()
    return service.stop_fsm()


@router.post("/fsm/randomize", summary="Randomize target cube pose")
def randomize_cube() -> Dict[str, Any]:
    """Perturbs initial pose of the target cube on the shelf."""
    service = SimulationService.get_instance()
    return service.randomize_cube()


@router.post("/fsm/reset", summary="Reset simulation to initial keyframe")
def reset_simulation() -> Dict[str, Any]:
    """Resets physics and joint postures to home configuration."""
    service = SimulationService.get_instance()
    return service.reset_simulation()
