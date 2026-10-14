"""
Cartesian SE(3) Trajectory Generation and Interpolation.

Implements:
- 5th-order minimum-jerk polynomial interpolation (zero velocity and acceleration at endpoints).
- Spherical Linear Interpolation (SLERP) for rotation quaternions.
- Multi-waypoint trajectory sequencing with automatic duration profiling.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

class Waypoint:
    """Represents a spatial SE(3) waypoint."""
    def __init__(self, pos: np.ndarray, quat: np.ndarray, duration: Optional[float] = None, name: str = ""):
        self.pos = np.array(pos, dtype=np.float64)
        self.quat = np.array(quat, dtype=np.float64)  # scalar-first [w, x, y, z]
        self.duration = duration
        self.name = name

class SE3Trajectory:
    """Generates continuous, minimum-jerk Cartesian trajectories through SE(3) waypoints."""

    def __init__(
        self,
        start_pos: np.ndarray,
        start_quat: np.ndarray,
        goal_pos: np.ndarray,
        goal_quat: np.ndarray,
        duration: Optional[float] = None,
        max_lin_vel: float = 0.25,  # m/s
        max_ang_vel: float = 0.50,  # rad/s
    ):
        self.start_pos = np.array(start_pos, dtype=np.float64)
        self.start_quat = np.array(start_quat, dtype=np.float64)
        self.goal_pos = np.array(goal_pos, dtype=np.float64)
        self.goal_quat = np.array(goal_quat, dtype=np.float64)

        # Automatic duration computation if not specified
        if duration is None:
            lin_dist = float(np.linalg.norm(self.goal_pos - self.start_pos))
            r_start = R.from_quat(self.start_quat, scalar_first=True)
            r_goal = R.from_quat(self.goal_quat, scalar_first=True)
            ang_dist = float(np.linalg.norm((r_goal * r_start.inv()).as_rotvec()))

            t_lin = lin_dist / max_lin_vel if max_lin_vel > 0 else 1.0
            t_ang = ang_dist / max_ang_vel if max_ang_vel > 0 else 1.0
            self.duration = max(0.5, max(t_lin, t_ang))
        else:
            self.duration = max(0.01, duration)

        # Setup SLERP for orientation
        r_start = R.from_quat(self.start_quat, scalar_first=True)
        r_goal = R.from_quat(self.goal_quat, scalar_first=True)
        self._slerp = Slerp([0.0, 1.0], R.from_quat([
            r_start.as_quat(),  # scipy Slerp uses scalar-last internally
            r_goal.as_quat()
        ]))

    @staticmethod
    def quintic_poly(tau: float) -> Tuple[float, float, float]:
        """
        Computes 5th-order minimum-jerk polynomial scaling:
        s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5 for tau in [0, 1].
        Returns (s, s_dot_factor, s_ddot_factor).
        """
        tau = np.clip(tau, 0.0, 1.0)
        s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
        s_dot = 30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)
        s_ddot = 60.0 * tau - 180.0 * (tau ** 2) + 120.0 * (tau ** 3)
        return s, s_dot, s_ddot

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Evaluates the trajectory at elapsed time t.

        Returns:
            pos: [x, y, z] target position.
            quat: [w, x, y, z] target orientation (scalar-first).
            lin_vel: [vx, vy, vz] feedforward linear velocity.
            is_done: True if t >= duration.
        """
        is_done = t >= self.duration
        tau = np.clip(t / self.duration, 0.0, 1.0)
        s, s_dot, _ = self.quintic_poly(tau)

        # Interpolate position
        pos = self.start_pos + s * (self.goal_pos - self.start_pos)
        lin_vel = (s_dot / self.duration) * (self.goal_pos - self.start_pos)

        # Interpolate orientation via SLERP
        interp_rot = self._slerp([s])[0]
        quat = interp_rot.as_quat(scalar_first=True)

        return pos, quat, lin_vel, is_done


class MultiWaypointTrajectory:
    """Sequences multiple SE(3) waypoints into a continuous trajectory."""

    def __init__(self, waypoints: List[Waypoint], max_lin_vel: float = 0.25, max_ang_vel: float = 0.5):
        if len(waypoints) < 2:
            raise ValueError("At least 2 waypoints required for trajectory sequencing.")

        self.segments: List[SE3Trajectory] = []
        self.segment_start_times: List[float] = []
        self.total_duration = 0.0

        current_time = 0.0
        for i in range(len(waypoints) - 1):
            w0 = waypoints[i]
            w1 = waypoints[i + 1]
            seg = SE3Trajectory(
                w0.pos, w0.quat, w1.pos, w1.quat,
                duration=w1.duration, max_lin_vel=max_lin_vel, max_ang_vel=max_ang_vel
            )
            self.segments.append(seg)
            self.segment_start_times.append(current_time)
            current_time += seg.duration

        self.total_duration = current_time

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Evaluates pose and velocity across multi-segment sequence at time t."""
        if t >= self.total_duration:
            last_seg = self.segments[-1]
            pos, quat, _, _ = last_seg.evaluate(last_seg.duration)
            return pos, quat, np.zeros(3), True

        # Find active segment
        seg_idx = 0
        for i, st in enumerate(self.segment_start_times):
            if t >= st:
                seg_idx = i

        seg = self.segments[seg_idx]
        local_t = t - self.segment_start_times[seg_idx]
        pos, quat, lin_vel, _ = seg.evaluate(local_t)
        return pos, quat, lin_vel, False


class JointTrajectory:
    """Minimum-jerk quintic polynomial interpolation between two joint configurations."""

    def __init__(self, q_start: np.ndarray, q_goal: np.ndarray, duration: float = 2.0):
        self.q_start = np.array(q_start, dtype=np.float64)
        self.q_goal = np.array(q_goal, dtype=np.float64)
        self.duration = max(0.01, duration)

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Evaluates joint positions and velocities at elapsed time t.
        Returns:
            (q: np.ndarray, q_dot: np.ndarray, is_done: bool)
        """
        is_done = t >= self.duration
        tau = np.clip(t / self.duration, 0.0, 1.0)
        s, s_dot, _ = SE3Trajectory.quintic_poly(tau)
        q = self.q_start + s * (self.q_goal - self.q_start)
        q_dot = (s_dot / self.duration) * (self.q_goal - self.q_start)
        return q, q_dot, is_done
