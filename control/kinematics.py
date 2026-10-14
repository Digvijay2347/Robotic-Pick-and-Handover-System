"""
Kinematics and Damped Least Squares (DLS) Inverse Kinematics for Dual UR10e.

Provides:
- Forward kinematics and end-effector pose extraction.
- 6-DOF Jacobian computation via MuJoCo site Jacobians.
- Damped Least Squares (DLS) numerical IK with singularity damping.
- Null-space postural optimization (biasing joints towards neutral resting poses).
- Joint limit clamping and velocity rate limiting.
"""

from typing import Tuple, List, Optional
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

class RobotKinematics:
    """Manages forward and inverse kinematics for a single UR10e arm in MuJoCo."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm_id: int,  # 1 for Arm 1 (Giver), 2 for Arm 2 (Receiver)
        ee_site_name: Optional[str] = None,
        damping: float = 1e-3,
        step_gain: float = 0.5,
        nullspace_gain: float = 0.1,
    ):
        self.model = model
        self.data = data
        self.arm_id = arm_id
        self.prefix = f"arm{arm_id}_"
        self.damping = damping
        self.step_gain = step_gain
        self.nullspace_gain = nullspace_gain

        # Identify arm joints (6 UR10e revolute joints)
        self.joint_names = [
            f"{self.prefix}shoulder_pan_joint",
            f"{self.prefix}shoulder_lift_joint",
            f"{self.prefix}elbow_joint",
            f"{self.prefix}wrist_1_joint",
            f"{self.prefix}wrist_2_joint",
            f"{self.prefix}wrist_3_joint",
        ]
        self.joint_ids = [model.joint(name).id for name in self.joint_names]
        self.qpos_addrs = [model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.dof_addrs = [model.joint(jid).dofadr[0] for jid in self.joint_ids]

        # Joint limits
        self.joint_limits_lower = np.array([model.jnt_range[jid][0] for jid in self.joint_ids])
        self.joint_limits_upper = np.array([model.jnt_range[jid][1] for jid in self.joint_ids])

        # Ensure kinematics forward pass has populated site poses
        mujoco.mj_forward(self.model, self.data)

        # Nominal resting / neutral posture for null-space biasing
        self.q_rest = np.array([0.0, -2.0, 2.0, -1.57, -1.57, 0.0])

        # End-effector tracking site
        if ee_site_name is None:
            ee_site_name = f"{self.prefix}ee_site"
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        if self.ee_site_id == -1:
            raise ValueError(f"Site '{ee_site_name}' not found in MuJoCo model!")

    @staticmethod
    def unwrap_joint_angles(q_target: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
        """Unwraps joint angles relative to a reference posture to avoid 2*pi branch jumps."""
        return q_ref + ((q_target - q_ref + np.pi) % (2.0 * np.pi) - np.pi)

    def get_joint_positions(self) -> np.ndarray:
        """Returns current 6 joint angles in radians."""
        return np.array([self.data.qpos[addr] for addr in self.qpos_addrs], dtype=np.float64)

    def get_joint_velocities(self) -> np.ndarray:
        """Returns current 6 joint velocities in rad/s."""
        return np.array([self.data.qvel[addr] for addr in self.dof_addrs], dtype=np.float64)

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns end-effector position and orientation quaternion.
        - position: [x, y, z] in world coordinates.
        - quaternion: [w, x, y, z] scalar-first format.
        """
        pos = self.data.site_xpos[self.ee_site_id].copy()
        rot_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        quat = R.from_matrix(rot_mat).as_quat(scalar_first=True)
        return pos, quat

    def compute_jacobian(self) -> np.ndarray:
        """
        Computes 6x6 geometric Jacobian J mapping joint velocities to EE spatial twist:
        [v_ee; omega_ee] = J * q_dot
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)

        # Slice the 6 DOF columns corresponding to this arm
        J = np.vstack([
            jacp[:, self.dof_addrs],
            jacr[:, self.dof_addrs]
        ])
        return J

    def compute_ik_step(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        pos_tolerance: float = 1e-3,
        rot_tolerance: float = 1e-2,
        max_joint_step: float = 0.05,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Computes one iteration of Damped Least Squares (DLS) IK with null-space biasing.

        Args:
            target_pos: Desired EE position [x, y, z] in world frame.
            target_quat: Desired EE quaternion [w, x, y, z] in world frame.
            pos_tolerance: Position error convergence threshold (meters).
            rot_tolerance: Rotation error convergence threshold (radians).
            max_joint_step: Maximum allowed delta angle per joint (radians).

        Returns:
            Tuple: (target_qpos, pos_error_norm, rot_error_norm)
        """
        curr_pos, curr_quat = self.get_ee_pose()

        # 1. Position error
        pos_err = target_pos - curr_pos
        pos_err_norm = float(np.linalg.norm(pos_err))

        # 2. Orientation error (using axis-angle representation)
        r_curr = R.from_quat(curr_quat, scalar_first=True)
        r_target = R.from_quat(target_quat, scalar_first=True)
        r_err = r_target * r_curr.inv()
        rot_err = r_err.as_rotvec()
        rot_err_norm = float(np.linalg.norm(rot_err))

        # Combined 6D spatial twist error
        error_6d = np.concatenate([pos_err, rot_err])

        # 3. Geometric Jacobian
        J = self.compute_jacobian()

        # 4. Adaptive Damped Least Squares Solver
        # J_dls = J^T * (J * J^T + lambda^2 * I)^(-1)
        damping = self.damping
        # Increase damping dynamically if manipulability drops near singularities
        manipulability = np.sqrt(max(0.0, float(np.linalg.det(J @ J.T))))
        if manipulability < 0.01:
            damping = self.damping * (1.0 + (0.01 - manipulability) / 0.01)

        JJT_damped = J @ J.T + (damping ** 2) * np.eye(6)
        inv_JJT_err = np.linalg.solve(JJT_damped, error_6d)
        delta_q_task = J.T @ inv_JJT_err

        # 5. Null-Space Postural Projection
        # N = I - J_pinv * J
        J_pinv = J.T @ np.linalg.inv(JJT_damped)
        N = np.eye(6) - J_pinv @ J
        curr_q = self.get_joint_positions()
        delta_q_null = N @ (self.nullspace_gain * (self.q_rest - curr_q))

        # Combined joint displacement
        delta_q = self.step_gain * delta_q_task + delta_q_null

        # 6. Joint rate limiting
        delta_q = np.clip(delta_q, -max_joint_step, max_joint_step)

        # 7. Compute target joint position and enforce physical limits
        q_target = curr_q + delta_q
        q_target = np.clip(q_target, self.joint_limits_lower, self.joint_limits_upper)

        return q_target, pos_err_norm, rot_err_norm

    def solve_ik_closed_loop(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        max_iterations: int = 150,
        pos_tolerance: float = 2e-3,
        rot_tolerance: float = 2e-2,
    ) -> Tuple[bool, np.ndarray]:
        """
        Solves IK iteratively to convergence without altering physical simulation data.

        Returns:
            Tuple (success: bool, joint_angles: np.ndarray)
        """
        # Save original state
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()

        converged = False
        final_q = self.get_joint_positions()

        for _ in range(max_iterations):
            q_next, pos_err, rot_err = self.compute_ik_step(
                target_pos, target_quat, pos_tolerance, rot_tolerance
            )
            # Apply to simulation state for forward kinematics update
            for i, addr in enumerate(self.qpos_addrs):
                self.data.qpos[addr] = q_next[i]
            mujoco.mj_forward(self.model, self.data)

            final_q = q_next
            if pos_err < pos_tolerance and rot_err < rot_tolerance:
                converged = True
                break

        # Restore original physical simulation state
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.model, self.data)
        start_q = np.array([saved_qpos[addr] for addr in self.qpos_addrs], dtype=np.float64)
        return converged, self.unwrap_joint_angles(final_q, start_q)
