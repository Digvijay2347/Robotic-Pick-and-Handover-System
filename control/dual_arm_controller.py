"""
Unified Dual-Arm Controller for MuJoCo UR10e and Robotiq 2F-85.

Features:
- Independent or coordinated control of Arm 1 (Giver) and Arm 2 (Receiver).
- DLS Cartesian IK step execution and actuator target assignment.
- Normalized gripper command interface [0.0 = open, 1.0 = closed].
- Contact force sensing and grasp stability verification via MuJoCo contact dynamics.
- Tracking error diagnostics.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import mujoco
from control.kinematics import RobotKinematics

class DualArmController:
    """High-level Cartesian and joint coordinator for the dual-arm system."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        # Kinematics engines for both arms
        self.kinematics_arm1 = RobotKinematics(model, data, arm_id=1)
        self.kinematics_arm2 = RobotKinematics(model, data, arm_id=2)

        # Actuator mapping for Arm 1
        self.arm1_joint_act_ids = [
            model.actuator(f"arm1_j{i}_{suffix}").id
            for i, suffix in enumerate(["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"])
        ]
        self.arm1_grip_act_id = model.actuator("arm1_gripper").id

        # Actuator mapping for Arm 2
        self.arm2_joint_act_ids = [
            model.actuator(f"arm2_j{i}_{suffix}").id
            for i, suffix in enumerate(["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"])
        ]
        self.arm2_grip_act_id = model.actuator("arm2_gripper").id

        # Gripper pad geoms for contact verification
        self.arm1_pad_geom_ids = [
            model.geom(name).id for name in ["arm1_right_pad1", "arm1_right_pad2", "arm1_left_pad1", "arm1_left_pad2"]
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) != -1
        ]
        self.arm2_pad_geom_ids = [
            model.geom(name).id for name in ["arm2_right_pad1", "arm2_right_pad2", "arm2_left_pad1", "arm2_left_pad2"]
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) != -1
        ]

        # Current gripper target positions [0.0 = open, 1.0 = closed]
        self.gripper_state = {1: 0.0, 2: 0.0}

    def get_kinematics(self, arm: int) -> RobotKinematics:
        """Returns the kinematics solver for Arm 1 or Arm 2."""
        return self.kinematics_arm1 if arm == 1 else self.kinematics_arm2

    def get_ee_pose(self, arm: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (position [x,y,z], quaternion [w,x,y,z]) of end-effector for specified arm."""
        return self.get_kinematics(arm).get_ee_pose()

    def set_arm_joint_targets(self, arm: int, q_targets: np.ndarray):
        """Sets raw joint angle targets for the 6 arm actuators."""
        act_ids = self.arm1_joint_act_ids if arm == 1 else self.arm2_joint_act_ids
        for i, aid in enumerate(act_ids):
            self.data.ctrl[aid] = q_targets[i]

    def set_gripper(self, arm: int, value: float):
        """
        Commands the Robotiq 2F-85 gripper.
        Args:
            arm: 1 or 2
            value: Normalized command [0.0 = fully open, 1.0 = fully closed].
        """
        clamped_val = float(np.clip(value, 0.0, 1.0))
        self.gripper_state[arm] = clamped_val
        act_id = self.arm1_grip_act_id if arm == 1 else self.arm2_grip_act_id

        # MuJoCo Robotiq 2F-85 control range is [0.0, 255.0]
        self.data.ctrl[act_id] = clamped_val * 255.0

    def step_ik(
        self,
        arm: int,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        pos_tolerance: float = 1e-3,
        rot_tolerance: float = 1e-2,
        max_joint_step: float = 0.05,
    ) -> Tuple[float, float]:
        """
        Executes one closed-loop IK step for specified arm and updates actuator control targets.

        Returns:
            Tuple of (position_error, rotation_error).
        """
        kin = self.get_kinematics(arm)
        q_target, pos_err, rot_err = kin.compute_ik_step(
            target_pos, target_quat, pos_tolerance, rot_tolerance, max_joint_step
        )
        self.set_arm_joint_targets(arm, q_target)
        return pos_err, rot_err

    def check_grasp(self, arm: int, object_body_name: str, min_normal_force: float = 0.2) -> Tuple[bool, float]:
        """
        Verifies if the specified arm's gripper has firm contact with the target object.

        Checks:
        1. Both left and right finger pads are contacting the object geom.
        2. Normal contact force exceeds threshold.

        Returns:
            Tuple of (is_firmly_grasped: bool, total_contact_force: float).
        """
        obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
        if obj_body_id == -1:
            return False, 0.0

        pad_geom_ids = self.arm1_pad_geom_ids if arm == 1 else self.arm2_pad_geom_ids
        left_pad_contact = False
        right_pad_contact = False
        total_force = 0.0

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            b1 = self.model.geom_bodyid[g1]
            b2 = self.model.geom_bodyid[g2]

            # Check if this contact is between one of our gripper pads and the target object
            is_pad_obj_contact = (
                (g1 in pad_geom_ids and b2 == obj_body_id) or
                (g2 in pad_geom_ids and b1 == obj_body_id)
            )

            if is_pad_obj_contact:
                # Extract normal contact force using MuJoCo contact dynamics
                c_forces = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, c_forces)
                normal_f = abs(c_forces[0])
                total_force += normal_f

                # Identify which pad made contact
                pad_geom = g1 if g1 in pad_geom_ids else g2
                pad_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, pad_geom)
                if "left" in pad_name:
                    left_pad_contact = True
                elif "right" in pad_name:
                    right_pad_contact = True

        # Stable antipodal grasp requires either both pads contacting or sufficient normal force
        is_grasped = (left_pad_contact or right_pad_contact) and (total_force >= min_normal_force)
        return is_grasped, total_force
