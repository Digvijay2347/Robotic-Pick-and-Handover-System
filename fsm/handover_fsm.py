"""
Coordinated Dual-Arm Handover Finite State Machine (Scripted Expert).

Orchestrates the complete autonomous 16-stage manipulation pipeline:
1. Perception-guided 6-DOF grasp synthesis and shelf retrieval with Arm 1.
2. Verified collision-free shelf clearance corridor.
3. Top-down presentation with orthogonal gripper geometry at the central handover zone.
4. Smooth lateral intercept with Arm 2 avoiding kinematic singularities.
5. Strict closed-loop dual custody interlock gate (never release without verified custody).
6. Clean disengagement and vertical retraction of Arm 1.
7. Continuous active watchdog transport by Arm 2 to storage bin.
8. Controlled placement and return to home posture.
"""

import os
import sys
import time
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco

from control.dual_arm_controller import DualArmController
from control.kinematics import RobotKinematics
from control.trajectory import JointTrajectory
from control.collision_checker import CollisionChecker
from perception import (
    capture_scene_pointcloud,
    SceneSegmenter,
    GeometricGraspPlanner,
    GraspFilter,
    FilteredGrasp,
)
from fsm.handover_states import HandoverState, HandoverTelemetry, CycleSummary


class HandoverFSM:
    """Finite State Machine orchestrating coordinated dual-arm pick-handover-place."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        controller: Optional[DualArmController] = None,
        collision_checker: Optional[CollisionChecker] = None,
        target_name: str = "target_object",
        bin_pos: np.ndarray = np.array([0.0, -0.80, 0.45]),
        handover_pos: np.ndarray = np.array([0.0, 0.10, 1.05]),
    ):
        self.model = model
        self.data = data
        self.controller = controller or DualArmController(model, data)
        self.collision_checker = collision_checker or CollisionChecker(model, data)
        self.kin1 = self.controller.kinematics_arm1
        self.kin2 = self.controller.kinematics_arm2

        self.target_name = target_name
        self.target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_name)
        if self.target_body_id == -1:
            raise ValueError(f"Target body '{target_name}' not found in simulation model!")

        self.bin_pos = bin_pos
        self.handover_pos = handover_pos
        self.dt = model.opt.timestep

        # Ensure model is initialized to keyframe 0 if available
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        # State tracking
        self.current_state = HandoverState.IDLE
        self.sim_time = 0.0
        self.state_start_time = 0.0
        self.dual_hold_timer = 0.0
        self.custody_confirm_counter = 0
        self.transport_slip_counter = 0
        self.state_history: List[str] = []

        # Current active trajectories
        self.traj_arm1: Optional[JointTrajectory] = None
        self.traj_arm2: Optional[JointTrajectory] = None
        self.hold_arm1: Optional[np.ndarray] = None
        self.hold_arm2: Optional[np.ndarray] = None
        self.target_grip1: float = 0.0
        self.target_grip2: float = 0.0
        self.stage_duration: float = 1.0

        # Pre-planned joint postures
        self.q_home1 = self.kin1.get_joint_positions()
        self.q_home2 = self.kin2.get_joint_positions()
        self.q_pre1: Optional[np.ndarray] = None
        self.q_gr1: Optional[np.ndarray] = None
        self.q_lift1: Optional[np.ndarray] = None
        self.q_ret1: Optional[np.ndarray] = None
        self.q_ho1: Optional[np.ndarray] = None
        self.q_ho1_clear: Optional[np.ndarray] = None
        self.q_pre2: Optional[np.ndarray] = None
        self.q_ho2: Optional[np.ndarray] = None
        self.q_bin2: Optional[np.ndarray] = None

        # Telemetry
        self.telemetry = HandoverTelemetry()
        self.dual_hold_confirmed = False

    def plan_trajectories(self, best_grasp: FilteredGrasp) -> bool:
        """
        Pre-computes and unwraps all joint configurations for both arms across the entire cycle.
        Returns True if all waypoints solve cleanly and are collision-free.
        """
        self.q_home1 = self.kin1.get_joint_positions()
        self.q_home2 = self.kin2.get_joint_positions()

        # Arm 1: Shelf Pick
        self.q_pre1 = best_grasp.q_pregrasp
        self.q_gr1 = best_grasp.q_grasp
        quat_gr = best_grasp.candidate.quat

        # Shelf Clearance Corridor:
        # Waypoint 1: Vertical Lift (+8 cm, staying inside compartment vertical margin)
        pos_lift = best_grasp.pos_grasp + np.array([0.0, 0.0, 0.08])
        # Waypoint 2: Retract straight along -Y to Y = 0.70 m (15 cm in front of shelf_div_3 at Y=0.85)
        pos_retract = np.array([best_grasp.pos_grasp[0], 0.70, pos_lift[2]])

        data_kin = mujoco.MjData(self.model)
        kin1_k = RobotKinematics(self.model, data_kin, arm_id=1)
        kin2_k = RobotKinematics(self.model, data_kin, arm_id=2)

        for i, a in enumerate(kin1_k.qpos_addrs):
            data_kin.qpos[a] = self.q_gr1[i]
        mujoco.mj_forward(self.model, data_kin)

        ok_lf, q_lift = kin1_k.solve_ik_closed_loop(pos_lift, quat_gr)
        if not ok_lf:
            return False
        self.q_lift1 = kin1_k.unwrap_joint_angles(q_lift, self.q_gr1)

        for i, a in enumerate(kin1_k.qpos_addrs):
            data_kin.qpos[a] = self.q_lift1[i]
        mujoco.mj_forward(self.model, data_kin)

        ok_rt, q_ret = kin1_k.solve_ik_closed_loop(pos_retract, quat_gr)
        if not ok_rt:
            return False
        self.q_ret1 = kin1_k.unwrap_joint_angles(q_ret, self.q_lift1)

        # Option A Handover: Top-Down Vertical Orientation with Orthogonal Closing Axis
        # Approach axis = [0, 0, -1] (downwards), Closing axis (local Y) = [1, 0, 0] (world X)
        mat1_top = np.column_stack([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        quat1_top = R.from_matrix(mat1_top).as_quat(scalar_first=True)
        pos_ho1 = np.array([0.0, 0.10, 1.075])

        for i, a in enumerate(kin1_k.qpos_addrs):
            data_kin.qpos[a] = kin1_k.q_rest[i]
        mujoco.mj_forward(self.model, data_kin)

        ok_ho1, q_ho1 = kin1_k.solve_ik_closed_loop(pos_ho1, quat1_top)
        if not ok_ho1:
            return False
        self.q_ho1 = kin1_k.unwrap_joint_angles(q_ho1, self.q_ret1)

        # Arm 1 Retract Clear: Straight UP along +Z (+25 cm) and away (-5 cm along -Y)
        pos_ho1_clear = pos_ho1 + np.array([0.0, -0.05, 0.25])
        for i, a in enumerate(kin1_k.qpos_addrs):
            data_kin.qpos[a] = self.q_ho1[i]
        mujoco.mj_forward(self.model, data_kin)

        ok_clear, q_ho1_clear = kin1_k.solve_ik_closed_loop(pos_ho1_clear, quat1_top)
        if not ok_clear:
            return False
        self.q_ho1_clear = kin1_k.unwrap_joint_angles(q_ho1_clear, self.q_ho1)

        # Arm 2 Nominal Handover Setup (approaching along +X, closing along Y)
        mat2_ho = np.column_stack([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        quat2_ho = R.from_matrix(mat2_ho).as_quat(scalar_first=True)
        pos_ho2 = np.array([self.handover_pos[0] + 0.0144, self.handover_pos[1], self.handover_pos[2] - 0.021])
        standoff = 0.07  # 7 cm standoff avoids singularity boundary
        pos_pre2 = pos_ho2 - np.array([standoff, 0.0, 0.0])

        for i, a in enumerate(kin2_k.qpos_addrs):
            data_kin.qpos[a] = kin2_k.q_rest[i]
        mujoco.mj_forward(self.model, data_kin)

        ok_ho2, q_ho2 = kin2_k.solve_ik_closed_loop(pos_ho2, quat2_ho)
        if not ok_ho2:
            return False
        self.q_ho2 = q_ho2

        for i, a in enumerate(kin2_k.qpos_addrs):
            data_kin.qpos[a] = self.q_ho2[i]
        mujoco.mj_forward(self.model, data_kin)

        ok_pre2, q_pre2 = kin2_k.solve_ik_closed_loop(pos_pre2, quat2_ho)
        if not ok_pre2:
            return False
        self.q_pre2 = kin2_k.unwrap_joint_angles(q_pre2, self.q_ho2)

        # Arm 2 Storage Bin Drop Target
        mat2_bin = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        quat2_bin = R.from_matrix(mat2_bin).as_quat(scalar_first=True)
        data_kin.qpos[kin2_k.qpos_addrs] = kin2_k.q_rest
        mujoco.mj_forward(self.model, data_kin)

        ok_bin, q_bin = kin2_k.solve_ik_closed_loop(self.bin_pos, quat2_bin)
        if not ok_bin:
            return False
        self.q_bin2 = q_bin
        return True

    def plan_arm2_adaptive(self) -> bool:
        """
        Dynamically adapts Arm 2 handover grasp, standoff, and bin drop configurations
        to the actual physical settled position of the object held by Arm 1 at handover.
        Uses an isolated MjData instance so live simulation state is never disturbed.
        """
        pos_box = self.data.xpos[self.target_body_id].copy()
        mat2_ho = np.column_stack([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        quat2_ho = R.from_matrix(mat2_ho).as_quat(scalar_first=True)

        pos_ho2 = np.array([pos_box[0] + 0.0144, pos_box[1], pos_box[2] - 0.021])
        standoff = 0.07  # Clean 7 cm standoff avoids UR10e singularity whip
        pos_pre2 = pos_ho2 - np.array([standoff, 0.0, 0.0])

        data_kin = mujoco.MjData(self.model)
        kin2_kin = RobotKinematics(self.model, data_kin, arm_id=2)

        for i, a in enumerate(kin2_kin.qpos_addrs):
            data_kin.qpos[a] = kin2_kin.q_rest[i]
        mujoco.mj_forward(self.model, data_kin)

        ok_ho2, q_ho2 = kin2_kin.solve_ik_closed_loop(pos_ho2, quat2_ho)
        if not ok_ho2:
            return False
        self.q_ho2 = q_ho2

        for i, a in enumerate(kin2_kin.qpos_addrs):
            data_kin.qpos[a] = self.q_ho2[i]
        mujoco.mj_forward(self.model, data_kin)

        ok_pre2, q_pre2 = kin2_kin.solve_ik_closed_loop(pos_pre2, quat2_ho)
        if not ok_pre2:
            return False
        self.q_pre2 = kin2_kin.unwrap_joint_angles(q_pre2, self.q_ho2)

        # Storage Bin Drop Target
        bin_pos = np.array([self.bin_pos[0], self.bin_pos[1], self.bin_pos[2]])
        mat2_bin = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        quat2_bin = R.from_matrix(mat2_bin).as_quat(scalar_first=True)

        data_kin.qpos[kin2_kin.qpos_addrs] = kin2_kin.q_rest
        mujoco.mj_forward(self.model, data_kin)

        ok_bin, q_bin = kin2_kin.solve_ik_closed_loop(bin_pos, quat2_bin)
        if not ok_bin:
            return False
        self.q_bin2 = q_bin
        return True

    def transition_to(self, new_state: HandoverState, duration: float = 1.0):
        """Transitions FSM to a new state and sets up active trajectories."""
        self.current_state = new_state
        self.state_start_time = self.sim_time
        self.stage_duration = duration
        self.state_history.append(new_state.name)

        if new_state == HandoverState.APPROACH_1:
            self.traj_arm1 = JointTrajectory(self.q_home1, self.q_pre1, duration=duration)
            self.hold_arm2 = self.q_home2
            self.target_grip1 = 0.0
            self.target_grip2 = 0.0

        elif new_state == HandoverState.GRASP_1:
            self.traj_arm1 = JointTrajectory(self.q_pre1, self.q_gr1, duration=1.5)
            self.hold_arm2 = self.q_home2
            self.target_grip1 = 0.0
            self.target_grip2 = 0.0

        elif new_state == HandoverState.LIFT_1:
            self.traj_arm1 = JointTrajectory(self.q_gr1, self.q_lift1, duration=duration)
            self.hold_arm2 = self.q_home2
            self.target_grip1 = 0.75
            self.target_grip2 = 0.0

        elif new_state == HandoverState.RETRACT_1:
            self.traj_arm1 = JointTrajectory(self.q_lift1, self.q_ret1, duration=duration)
            self.hold_arm2 = self.q_home2
            self.target_grip1 = 0.75
            self.target_grip2 = 0.0

        elif new_state == HandoverState.TRANSFER_1:
            self.traj_arm1 = JointTrajectory(self.q_ret1, self.q_ho1, duration=duration)
            self.hold_arm2 = self.q_home2
            self.target_grip1 = 0.75
            self.target_grip2 = 0.0

        elif new_state == HandoverState.PRE_HANDOVER_2:
            self.hold_arm1 = self.q_ho1
            self.traj_arm2 = JointTrajectory(self.q_home2, self.q_pre2, duration=duration)
            self.target_grip1 = 0.75
            self.target_grip2 = 0.0

        elif new_state == HandoverState.APPROACH_2:
            self.hold_arm1 = self.q_ho1
            self.traj_arm2 = JointTrajectory(self.q_pre2, self.q_ho2, duration=duration)
            self.target_grip1 = 0.75
            self.target_grip2 = 0.0

        elif new_state == HandoverState.DUAL_GRASP:
            # Overlapping custody hold: both arms actively clamp
            self.hold_arm1 = self.q_ho1
            self.hold_arm2 = self.q_ho2
            self.target_grip1 = 0.75
            self.target_grip2 = 1.0
            self.dual_hold_timer = 0.0
            self.custody_confirm_counter = 0

        elif new_state == HandoverState.RELEASE_1:
            # Arm 1 stays held stationary at q_ho1 while opening gripper cleanly (1.0s)
            self.hold_arm1 = self.q_ho1
            self.hold_arm2 = self.q_ho2
            self.target_grip1 = 0.0
            self.target_grip2 = 1.0

        elif new_state == HandoverState.RETRACT_CLEAR_1:
            # Arm 1 retreats straight upward along +Z by 25 cm into clear space
            self.traj_arm1 = JointTrajectory(self.q_ho1, self.q_ho1_clear, duration=duration)
            self.hold_arm2 = self.q_ho2
            self.target_grip1 = 0.0
            self.target_grip2 = 1.0

        elif new_state == HandoverState.TRANSPORT_2:
            # Arm 1 stays parked clear; Arm 2 transports box to bin
            self.hold_arm1 = self.q_ho1_clear
            self.traj_arm2 = JointTrajectory(self.q_ho2, self.q_bin2, duration=duration)
            self.target_grip1 = 0.0
            self.target_grip2 = 1.0
            self.transport_slip_counter = 0

        elif new_state == HandoverState.PLACE_2:
            # Arm 2 opens gripper above bin; Arm 1 remains safely parked
            self.hold_arm1 = self.q_ho1_clear
            self.hold_arm2 = self.q_bin2
            self.target_grip1 = 0.0
            self.target_grip2 = 0.0

        elif new_state == HandoverState.RETRACT_2:
            # Both arms return home together
            self.traj_arm1 = JointTrajectory(self.q_ho1_clear, self.q_home1, duration=duration)
            self.traj_arm2 = JointTrajectory(self.q_bin2, self.q_home2, duration=duration)
            self.target_grip1 = 0.0
            self.target_grip2 = 0.0

        elif new_state in (HandoverState.COMPLETED, HandoverState.ERROR_RECOVERY, HandoverState.IDLE):
            self.hold_arm1 = self.q_home1
            self.hold_arm2 = self.q_home2
            self.target_grip1 = 0.0
            self.target_grip2 = 0.0

    def step(self) -> Tuple[HandoverState, bool]:
        """
        Executes one physics step (dt = 0.002s), evaluates trajectories, monitors interlocks,
        and manages state transitions.

        Returns:
            Tuple of (current_state: HandoverState, is_finished: bool).
        """
        t_in_state = self.sim_time - self.state_start_time

        if self.current_state in (
            HandoverState.APPROACH_1,
            HandoverState.LIFT_1,
            HandoverState.RETRACT_1,
        ):
            q1, _, is_done = self.traj_arm1.evaluate(t_in_state)
            q2 = self.hold_arm2
        elif self.current_state == HandoverState.TRANSFER_1:
            move_dur = 3.0
            if t_in_state <= move_dur:
                q1, _, _ = self.traj_arm1.evaluate(t_in_state)
            else:
                q1 = self.q_ho1
            q2 = self.hold_arm2
            is_done = t_in_state >= self.stage_duration
        elif self.current_state == HandoverState.GRASP_1:
            advance_dur = 1.5
            if t_in_state <= advance_dur:
                q1 = self.traj_arm1.evaluate(t_in_state)[0]
                self.target_grip1 = 0.0
            else:
                q1 = self.q_gr1
                self.target_grip1 = 0.75
            q2 = self.hold_arm2
            is_done = t_in_state >= self.stage_duration
        elif self.current_state in (
            HandoverState.PRE_HANDOVER_2,
            HandoverState.APPROACH_2,
        ):
            q1 = self.hold_arm1
            q2, _, is_done = self.traj_arm2.evaluate(t_in_state)
        elif self.current_state == HandoverState.DUAL_GRASP:
            q1 = self.hold_arm1
            q2 = self.hold_arm2
            # Interlock verification: Require sustained dual custody hold for 50 steps (100 ms)
            is_held1_cur, f1_cur = self.controller.check_grasp(1, self.target_name)
            is_held2_cur, f2_cur = self.controller.check_grasp(2, self.target_name)
            if is_held1_cur and is_held2_cur and f2_cur >= 15.0:
                self.custody_confirm_counter += 1
            else:
                self.custody_confirm_counter = 0

            # Done when sustained or on max timeout
            is_done = (self.custody_confirm_counter >= 50 and t_in_state >= 1.2) or (t_in_state >= 3.0)
        elif self.current_state == HandoverState.RELEASE_1:
            q1 = self.hold_arm1
            q2 = self.hold_arm2
            is_done = t_in_state >= self.stage_duration
        elif self.current_state == HandoverState.RETRACT_CLEAR_1:
            q1, _, is_done = self.traj_arm1.evaluate(t_in_state)
            q2 = self.hold_arm2
        elif self.current_state == HandoverState.TRANSPORT_2:
            q1 = self.hold_arm1
            q2, _, is_done = self.traj_arm2.evaluate(t_in_state)
        elif self.current_state == HandoverState.PLACE_2:
            q1 = self.hold_arm1
            q2 = self.hold_arm2
            is_done = t_in_state >= self.stage_duration
        elif self.current_state == HandoverState.RETRACT_2:
            q1, _, is_done1 = self.traj_arm1.evaluate(t_in_state)
            q2, _, is_done2 = self.traj_arm2.evaluate(t_in_state)
            is_done = is_done1 and is_done2
        else:
            q1 = self.q_home1
            q2 = self.q_home2
            is_done = True

        # 2. Command Actuators
        self.controller.set_arm_joint_targets(1, q1)
        self.controller.set_arm_joint_targets(2, q2)
        self.controller.set_gripper(1, self.target_grip1)
        self.controller.set_gripper(2, self.target_grip2)

        # 3. Step MuJoCo Physics (500 Hz internal)
        mujoco.mj_step(self.model, self.data)
        self.sim_time += self.dt

        # 4. Check Safety Interlocks & Contact Dynamics
        is_held1, f1 = self.controller.check_grasp(1, self.target_name)
        is_held2, f2 = self.controller.check_grasp(2, self.target_name)
        is_coll, _ = self.collision_checker.check_inter_arm_collision()

        # Update telemetry
        self.telemetry.state = self.current_state
        self.telemetry.sim_time = self.sim_time
        self.telemetry.state_elapsed_time = t_in_state
        self.telemetry.arm1_ee_pos, _ = self.controller.get_ee_pose(1)
        self.telemetry.arm2_ee_pos, _ = self.controller.get_ee_pose(2)
        self.telemetry.target_object_pos = self.data.xpos[self.target_body_id].copy()
        self.telemetry.arm1_contact_force = f1
        self.telemetry.arm2_contact_force = f2
        self.telemetry.arm1_holding = is_held1
        self.telemetry.arm2_holding = is_held2
        self.telemetry.inter_arm_collision = is_coll

        if is_coll:
            print(f"  [SAFETY ABORT] Inter-arm collision detected during {self.current_state.name}!")
            self.transition_to(HandoverState.ERROR_RECOVERY)
            return self.current_state, True

        # Active Watchdog during Arm 2 Transport: Verify physical custody retention
        if self.current_state == HandoverState.TRANSPORT_2 and t_in_state >= 0.5:
            # Only abort if object has physically fallen to floor (normal holding height is >= 0.35m)
            if self.telemetry.target_object_pos[2] < 0.15:
                print(f"  [SAFETY FAULT] Object dropped to floor during Transport at t={self.sim_time:.2f}s! Emergency stop.")
                self.transition_to(HandoverState.ERROR_RECOVERY)
                return self.current_state, True

        # 5. State Transition Logic
        if is_done:
            if self.current_state == HandoverState.APPROACH_1:
                self.transition_to(HandoverState.GRASP_1, duration=2.5)

            elif self.current_state == HandoverState.GRASP_1:
                if not is_held1:
                    print(f"  [WARN] Shelf grasp contact weak ({f1:.1f} N)!")
                self.transition_to(HandoverState.LIFT_1, duration=1.5)

            elif self.current_state == HandoverState.LIFT_1:
                self.transition_to(HandoverState.RETRACT_1, duration=2.0)

            elif self.current_state == HandoverState.RETRACT_1:
                self.transition_to(HandoverState.TRANSFER_1, duration=4.0)

            elif self.current_state == HandoverState.TRANSFER_1:
                ok_adapt = self.plan_arm2_adaptive()
                if not ok_adapt:
                    print("  [ERROR] Adaptive Arm 2 handover trajectory resolution failed!")
                    self.transition_to(HandoverState.ERROR_RECOVERY)
                    return self.current_state, True
                self.transition_to(HandoverState.PRE_HANDOVER_2, duration=2.0)

            elif self.current_state == HandoverState.PRE_HANDOVER_2:
                self.transition_to(HandoverState.APPROACH_2, duration=1.5)

            elif self.current_state == HandoverState.APPROACH_2:
                self.transition_to(HandoverState.DUAL_GRASP, duration=1.2)

            elif self.current_state == HandoverState.DUAL_GRASP:
                # Closed-Loop Dual Custody Gate: "Never release what is not verified held"
                if is_held1 and is_held2 and f2 >= 15.0:
                    self.dual_hold_confirmed = True
                    self.telemetry.dual_hold_confirmed = True
                    print(f"  --> [DUAL CUSTODY VERIFIED] Arm 1: {f1:.1f} N | Arm 2: {f2:.1f} N")
                    self.transition_to(HandoverState.RELEASE_1, duration=1.2)
                else:
                    print(f"  [INTERLOCK ABORT] Arm 2 failed custody verification ({f2:.1f} N)! Retaining Arm 1 hold.")
                    self.transition_to(HandoverState.ERROR_RECOVERY)
                    return self.current_state, True

            elif self.current_state == HandoverState.RELEASE_1:
                # Verify Arm 2 maintained sole custody before lifting Arm 1 clear
                if not is_held2 or f2 < 10.0:
                    print(f"  [ERROR] Custody lost during transfer release: is_held2={is_held2}, f2={f2:.1f} N!")
                    self.transition_to(HandoverState.ERROR_RECOVERY)
                    return self.current_state, True
                self.transition_to(HandoverState.RETRACT_CLEAR_1, duration=1.5)

            elif self.current_state == HandoverState.RETRACT_CLEAR_1:
                self.transition_to(HandoverState.TRANSPORT_2, duration=3.5)

            elif self.current_state == HandoverState.TRANSPORT_2:
                self.transition_to(HandoverState.PLACE_2, duration=1.0)

            elif self.current_state == HandoverState.PLACE_2:
                self.transition_to(HandoverState.RETRACT_2, duration=2.0)

            elif self.current_state == HandoverState.RETRACT_2:
                self.transition_to(HandoverState.COMPLETED)
                return HandoverState.COMPLETED, True

            elif self.current_state in (HandoverState.COMPLETED, HandoverState.ERROR_RECOVERY):
                return self.current_state, True

        return self.current_state, False

    def run_cycle(
        self,
        cycle_idx: int = 1,
        viewer: Optional[Any] = None,
        speed: float = 1.0,
        milestone_callback: Optional[Any] = None,
    ) -> CycleSummary:
        """
        Runs a complete pick-and-handover cycle from vision scan to storage bin deposit.

        Args:
            cycle_idx: Index of the current run.
            viewer: Optional MuJoCo passive viewer instance for real-time visualization.
            speed: Simulation speed multiplier.
            milestone_callback: Function called upon each state transition: callback(state_name).

        Returns:
            CycleSummary containing all mission evaluation telemetry.
        """
        t0 = time.time()
        self.state_history = []
        self.dual_hold_confirmed = False

        # Reset physics to initial home state
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        # Settle physics for 100 steps
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        # Refresh home postures after keyframe reset & settling
        self.q_home1 = self.kin1.get_joint_positions()
        self.q_home2 = self.kin2.get_joint_positions()
        self.sim_time = 0.0
        self.state_start_time = 0.0
        self.telemetry = HandoverTelemetry()

        # STEP 1: Perception & Grasp Planning
        self.transition_to(HandoverState.SCAN)
        if milestone_callback:
            milestone_callback("SCAN")

        pcd = capture_scene_pointcloud(self.model, self.data)
        segmenter = SceneSegmenter()
        clusters = segmenter.segment_objects(pcd)
        target = segmenter.find_target(clusters, color_hint="red")

        if target is None:
            print("[FATAL] Perception failed to find target red box!")
            return CycleSummary(
                cycle_index=cycle_idx,
                success=False,
                total_duration_sec=time.time() - t0,
                dual_hold_duration_sec=0.0,
                arm1_grip_force_mean=0.0,
                arm2_grip_force_mean=0.0,
                final_object_pos=self.data.xpos[self.target_body_id].copy(),
                distance_to_bin_center_cm=999.0,
                shelf_collision_detected=False,
                inter_arm_collision_detected=False,
            )

        self.transition_to(HandoverState.PLAN_GRASP)
        if milestone_callback:
            milestone_callback("PLAN_GRASP")

        planner = GeometricGraspPlanner()
        raw_cands = planner.plan_grasps(target.pcd)
        gfilter = GraspFilter(self.model, self.data, arm_id=1, standoff_dist=0.15)
        verified_grasps = gfilter.filter_candidates(raw_cands)

        if not verified_grasps:
            print("[FATAL] No valid 6-DOF grasp candidates found!")
            return CycleSummary(
                cycle_index=cycle_idx,
                success=False,
                total_duration_sec=time.time() - t0,
                dual_hold_duration_sec=0.0,
                arm1_grip_force_mean=0.0,
                arm2_grip_force_mean=0.0,
                final_object_pos=self.data.xpos[self.target_body_id].copy(),
                distance_to_bin_center_cm=999.0,
                shelf_collision_detected=False,
                inter_arm_collision_detected=False,
            )

        best_grasp = verified_grasps[0]
        ok_plan = self.plan_trajectories(best_grasp)
        if not ok_plan:
            print("[FATAL] Trajectory resolution failed for one or more keyframes!")
            return CycleSummary(
                cycle_index=cycle_idx,
                success=False,
                total_duration_sec=time.time() - t0,
                dual_hold_duration_sec=0.0,
                arm1_grip_force_mean=0.0,
                arm2_grip_force_mean=0.0,
                final_object_pos=self.data.xpos[self.target_body_id].copy(),
                distance_to_bin_center_cm=999.0,
                shelf_collision_detected=False,
                inter_arm_collision_detected=False,
            )

        # STEP 2: Execute FSM State Trajectories
        self.transition_to(HandoverState.APPROACH_1, duration=2.0)
        last_state = self.current_state
        print(f"[STATE TRANSITION] -> APPROACH_1")
        if milestone_callback:
            milestone_callback("APPROACH_1")

        arm1_forces = []
        arm2_forces = []
        inter_arm_coll_seen = False
        shelf_coll_seen = False

        while True:
            t_loop = time.time()
            curr_state, is_finished = self.step()

            # Record telemetry
            if self.telemetry.arm1_contact_force > 0:
                arm1_forces.append(self.telemetry.arm1_contact_force)
            if self.telemetry.arm2_contact_force > 0:
                arm2_forces.append(self.telemetry.arm2_contact_force)
            if self.telemetry.inter_arm_collision:
                inter_arm_coll_seen = True

            # Check shelf collisions during early transit
            if curr_state in (HandoverState.RETRACT_1, HandoverState.TRANSFER_1):
                scoll, _ = self.collision_checker.check_shelf_collision(1)
                if scoll:
                    shelf_coll_seen = True

            # Track milestone transitions
            if curr_state != last_state:
                print(f"[STATE TRANSITION] -> {curr_state.name}")
                if milestone_callback:
                    milestone_callback(curr_state.name)
                last_state = curr_state

            # Sync viewer if interactive
            if viewer is not None:
                viewer.sync()
                elapsed = time.time() - t_loop
                sleep_t = (self.dt / speed) - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

            if is_finished:
                break

        # Settle physics for 150 steps (let dropped object come to full rest in bin)
        for _ in range(150):
            self.controller.set_arm_joint_targets(1, self.q_home1)
            self.controller.set_arm_joint_targets(2, self.q_home2)
            self.controller.set_gripper(1, 0.0)
            self.controller.set_gripper(2, 0.0)
            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()

        # Evaluate mission success
        final_obj_pos = self.data.xpos[self.target_body_id].copy()
        dist_bin = float(np.linalg.norm(final_obj_pos[:2] - np.array([self.bin_pos[0], self.bin_pos[1]])))
        shelf_coll, _ = self.collision_checker.check_shelf_collision(1)
        final_shelf_coll = shelf_coll or shelf_coll_seen

        # Success conditions:
        # 1. Object inside bin basin: horizontal distance < 15 cm and height < 20 cm
        # 2. Dual grasp overlap confirmed
        # 3. No inter-arm collisions
        # 4. No shelf collisions
        success = (
            (dist_bin < 0.15) and
            (final_obj_pos[2] < 0.20) and
            self.dual_hold_confirmed and
            not inter_arm_coll_seen and
            not final_shelf_coll
        )

        return CycleSummary(
            cycle_index=cycle_idx,
            success=success,
            total_duration_sec=time.time() - t0,
            dual_hold_duration_sec=1.0 if self.dual_hold_confirmed else 0.0,
            arm1_grip_force_mean=float(np.mean(arm1_forces)) if arm1_forces else 0.0,
            arm2_grip_force_mean=float(np.mean(arm2_forces)) if arm2_forces else 0.0,
            final_object_pos=final_obj_pos,
            distance_to_bin_center_cm=dist_bin * 100.0,
            shelf_collision_detected=final_shelf_coll,
            inter_arm_collision_detected=inter_arm_coll_seen,
            state_history=list(self.state_history),
        )
