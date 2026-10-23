"""
Base Grasp Planner Interface & Data Structures for 6-DOF Robotic Grasping.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import open3d as o3d
except Exception:
    o3d = None


@dataclass
class GraspCandidate:
    """Represents a 6-DOF candidate grasp pose for a parallel-jaw gripper."""
    pos: np.ndarray             # EE pinch center [x, y, z] in World Frame
    rot_mat: np.ndarray         # 3x3 rotation matrix
    quat: np.ndarray            # 4D quaternion [w, x, y, z] (scalar-first)
    score: float                # Grasp confidence/quality score in [0.0, 1.0]
    width: float                # Required gripper aperture width (meters)
    approach_dir: np.ndarray    # Unit vector of tool approach direction (+Z_ee)
    cluster_id: int = -1

    @classmethod
    def from_pos_and_matrix(
        cls,
        pos: np.ndarray,
        rot_mat: np.ndarray,
        score: float,
        width: float,
        cluster_id: int = -1,
    ) -> "GraspCandidate":
        """Instantiates GraspCandidate deriving quaternion and approach direction."""
        quat = R.from_matrix(rot_mat).as_quat(scalar_first=True)
        approach_dir = rot_mat[:, 2].copy()  # Tool +Z is the forward approach axis
        return cls(
            pos=np.array(pos, dtype=np.float64),
            rot_mat=np.array(rot_mat, dtype=np.float64),
            quat=quat,
            score=float(np.clip(score, 0.0, 1.0)),
            width=float(width),
            approach_dir=approach_dir,
            cluster_id=cluster_id,
        )


class BaseGraspPlanner(ABC):
    """Abstract base class for 6-DOF grasp planning algorithms."""

    @abstractmethod
    def plan_grasps(
        self,
        pcd: Any,
        num_candidates: int = 10,
        cluster_id: int = -1,
    ) -> List[GraspCandidate]:
        """
        Generates ranked 6-DOF grasp candidates for an object point cloud.

        Args:
            pcd: Segmented object point cloud.
            num_candidates: Target number of candidates to generate.
            cluster_id: Identifier of the segmented object.

        Returns:
            List of GraspCandidate instances sorted by score in descending order.
        """
        pass
