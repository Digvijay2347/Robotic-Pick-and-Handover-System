"""
Handover State Definitions and Telemetry Data Structures.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np


class HandoverState(Enum):
    """Finite State Machine states for coordinated dual-arm pick-handover-place."""
    IDLE = auto()               # Arms at home ready poses
    SCAN = auto()               # RGB-D perception & point cloud generation
    PLAN_GRASP = auto()         # 6-DOF grasp synthesis & IK reachability filtering
    APPROACH_1 = auto()         # Arm 1 moves from home to shelf pre-grasp standoff
    GRASP_1 = auto()            # Arm 1 advances linearly into compartment & clamps object
    LIFT_1 = auto()             # Arm 1 lifts object vertically to break shelf friction
    RETRACT_1 = auto()          # Arm 1 extracts object horizontally out of shelf
    TRANSFER_1 = auto()         # Arm 1 moves holding object to central handover zone
    PRE_HANDOVER_2 = auto()     # Arm 2 navigates to pre-handover standoff
    APPROACH_2 = auto()         # Arm 2 advances linearly, enclosing lower half of object
    DUAL_GRASP = auto()         # Both arms hold object simultaneously (overlapping custody)
    RELEASE_1 = auto()          # Arm 1 opens gripper, transferring custody to Arm 2
    RETRACT_CLEAR_1 = auto()    # Arm 1 retracts vertically & returns to home
    TRANSPORT_2 = auto()        # Arm 2 transports object along smooth arc to storage bin
    PLACE_2 = auto()            # Arm 2 opens gripper above bin; object drops into basin
    RETRACT_2 = auto()          # Arm 2 returns to home ready posture
    COMPLETED = auto()          # Task finished successfully
    ERROR_RECOVERY = auto()     # Safety abort triggered on slip, timeout, or collision


@dataclass
class HandoverTelemetry:
    """Real-time telemetry tracked during state machine execution."""
    state: HandoverState = HandoverState.IDLE
    sim_time: float = 0.0
    state_elapsed_time: float = 0.0
    arm1_ee_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    arm2_ee_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_object_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    arm1_contact_force: float = 0.0
    arm2_contact_force: float = 0.0
    arm1_holding: bool = False
    arm2_holding: bool = False
    dual_hold_confirmed: bool = False
    shelf_collision: bool = False
    inter_arm_collision: bool = False
    max_tracking_error_arm1: float = 0.0
    max_tracking_error_arm2: float = 0.0


@dataclass
class CycleSummary:
    """Evaluation summary for a single complete pick-handover-place cycle."""
    cycle_index: int
    success: bool
    total_duration_sec: float
    dual_hold_duration_sec: float
    arm1_grip_force_mean: float
    arm2_grip_force_mean: float
    final_object_pos: np.ndarray
    distance_to_bin_center_cm: float
    shelf_collision_detected: bool
    inter_arm_collision_detected: bool
    state_history: List[str] = field(default_factory=list)
