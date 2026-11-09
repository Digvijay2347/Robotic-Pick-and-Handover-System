"""
System Health, Hardware Telemetry, and State Machine Execution Router.
"""

import time
import platform
import psutil
from typing import Dict, Any, List
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..simulation_service import SimulationService

router = APIRouter(prefix="/api", tags=["System & Control"])

_SERVER_START_TIME = time.time()


class JogJointRequest(BaseModel):
    arm: int = Field(..., ge=1, le=2, description="1 for Arm 1 (Giver), 2 for Arm 2 (Receiver)")
    joint_idx: int = Field(..., ge=0, le=5, description="0 to 5 for base to wrist3")
    delta_rad: float = Field(..., description="Angle change in radians")


class JogCartesianRequest(BaseModel):
    arm: int = Field(..., ge=1, le=2, description="1 for Arm 1 (Giver), 2 for Arm 2 (Receiver)")
    delta_xyz: List[float] = Field(..., min_length=3, max_length=3, description="[dx, dy, dz] in meters")


class GripperRequest(BaseModel):
    arm: int = Field(..., ge=1, le=2, description="1 for Arm 1, 2 for Arm 2")
    position: float = Field(..., ge=0.0, le=1.0, description="0.0 for open, 1.0 for closed")


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
    """Triggers autonomous execution of the trained ACT Neural Policy (best_model_v3.pt)."""
    service = SimulationService.get_instance()
    return service.start_fsm_cycle()


@router.post("/policy/start", summary="Start autonomous ACT neural policy handover")
def start_policy() -> Dict[str, Any]:
    """Triggers autonomous execution of the trained ACT Neural Policy (best_model_v3.pt)."""
    service = SimulationService.get_instance()
    return service.start_policy_cycle()


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


@router.post("/control/jog_joint", summary="Teleoperate / jog single joint")
def jog_joint(req: JogJointRequest) -> Dict[str, Any]:
    """Jogs a single joint angle on the specified arm."""
    service = SimulationService.get_instance()
    return service.jog_joint(arm=req.arm, joint_idx=req.joint_idx, delta_rad=req.delta_rad)


@router.post("/control/jog_cartesian", summary="Teleoperate / jog end-effector in Cartesian space")
def jog_cartesian(req: JogCartesianRequest) -> Dict[str, Any]:
    """Jogs the end-effector of the specified arm by [dx, dy, dz] in meters."""
    service = SimulationService.get_instance()
    return service.jog_cartesian(arm=req.arm, delta_xyz=req.delta_xyz)


@router.post("/control/gripper", summary="Set gripper open/close state")
def set_gripper(req: GripperRequest) -> Dict[str, Any]:
    """Commands gripper target position (0.0=open, 1.0=closed)."""
    service = SimulationService.get_instance()
    return service.set_gripper_state(arm=req.arm, position=req.position)

