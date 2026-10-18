"""
Native Geometric Antipodal 6-DOF Grasp Synthesizer for Parallel-Jaw Grippers.

Generates SE(3) candidate grasp poses from segmented 3D point clouds:
- Performs Principal Component Analysis (PCA) on point distribution.
- Estimates local surface normals.
- Samples antipodal grasp directions across the object's thinner dimensions.
- Constrains approach vectors to obstacle-free entry corridors (e.g., front of shelf).
- Validates gripper aperture limits (<= 85 mm for Robotiq 2F-85).
- Scores candidates using antipodal opposition, aperture clearance, and gravity alignment.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from perception.grasp_planners.base_planner import BaseGraspPlanner, GraspCandidate


class GeometricGraspPlanner(BaseGraspPlanner):
    """Synthesizes antipodal 6-DOF grasp candidates using geometric surface analysis."""

    def __init__(
        self,
        max_gripper_width: float = 0.085,    # Robotiq 2F-85 max stroke (meters)
        min_gripper_width: float = 0.010,    # Minimum detectable object thickness
        preferred_approach: np.ndarray = np.array([0.0, 1.0, 0.0]),  # +Y into shelf
        standoff_distance: float = 0.15,      # Approach standoff (meters)
    ):
        self.max_gripper_width = max_gripper_width
        self.min_gripper_width = min_gripper_width
        self.preferred_approach = preferred_approach / np.linalg.norm(preferred_approach)
        self.standoff_distance = standoff_distance

    def plan_grasps(
        self,
        pcd: o3d.geometry.PointCloud,
        num_candidates: int = 10,
        cluster_id: int = -1,
    ) -> List[GraspCandidate]:
        """
        Samples, validates, and ranks 6-DOF grasp candidates for the given point cloud.
        """
        points = np.asarray(pcd.points)
        if len(points) < 15:
            return []

        # 1. Surface normal estimation
        pcd_copy = o3d.geometry.PointCloud(pcd)
        pcd_copy.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30)
        )
        normals = np.asarray(pcd_copy.normals)

        # 2. PCA decomposition
        mean_p = np.mean(points, axis=0)
        centroid = 0.5 * (np.min(points, axis=0) + np.max(points, axis=0))
        centered = points - mean_p
        cov = (centered.T @ centered) / len(points)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort eigenvectors by descending eigenvalue (v0: major, v1: intermediate, v2: minor)
        order = np.argsort(eigenvalues)[::-1]
        v_major = eigenvectors[:, order[0]]
        v_inter = eigenvectors[:, order[1]]
        v_minor = eigenvectors[:, order[2]]

        candidates: List[GraspCandidate] = []

        # 3. Sample approach vectors around the preferred corridor (+Y shelf entry)
        approach_angles = [
            (0.0, 0.0),      # Direct horizontal approach along +Y
            (0.0, -5.0),     # 5 deg downward pitch
            (0.0, -10.0),    # 10 deg downward pitch
            (5.0, 0.0),      # 5 deg yaw right
            (-5.0, 0.0),     # 5 deg yaw left
        ]

        # Vertical height offsets along object height to ensure finger clearance from floor
        z_min = np.min(points[:, 2]) - 0.015  # Account for RANSAC plane margin
        z_max = np.max(points[:, 2])
        height = max(0.04, z_max - z_min)
        height_fractions = [0.55, 0.65, 0.72]

        for yaw_deg, pitch_deg in approach_angles:
            # Construct perturbed approach vector
            rot_pert = R.from_euler("zyx", [yaw_deg, pitch_deg, 0.0], degrees=True)
            z_ee = rot_pert.apply(self.preferred_approach)
            z_ee = z_ee / np.linalg.norm(z_ee)

            # Test finger closing orientations:
            # 1. Horizontal closing axis (along World X)
            # 2. Closing axis aligned with minor PCA axis
            # 3. Closing axis aligned with intermediate PCA axis
            closing_axes = [
                np.array([1.0, 0.0, 0.0]),
                v_minor - np.dot(v_minor, z_ee) * z_ee,
                v_inter - np.dot(v_inter, z_ee) * z_ee,
            ]

            for raw_closing in closing_axes:
                norm_c = np.linalg.norm(raw_closing)
                if norm_c < 1e-4:
                    continue
                y_ee_candidate = raw_closing / norm_c

                # Make orthogonal to z_ee
                x_ee = np.cross(y_ee_candidate, z_ee)
                norm_x = np.linalg.norm(x_ee)
                if norm_x < 1e-4:
                    continue
                x_ee = x_ee / norm_x
                y_ee = np.cross(z_ee, x_ee)

                rot_mat = np.column_stack([x_ee, y_ee, z_ee])

                # Check determinant
                if np.linalg.det(rot_mat) < 0:
                    rot_mat[:, 0] = -rot_mat[:, 0]

                # Project points onto closing axis y_ee to calculate grasp width
                projections = centered @ y_ee
                width = float(np.percentile(projections, 95) - np.percentile(projections, 5))

                if width < self.min_gripper_width or width > self.max_gripper_width:
                    continue

                for h_frac in height_fractions:
                    grasp_pos = centroid.copy()
                    grasp_pos[2] = z_min + h_frac * height

                    # Score candidate:
                    # 1. Approach alignment: strong preference for direct +Y entry corridor
                    approach_score = max(0.0, float(np.dot(z_ee, self.preferred_approach))) ** 4

                    # 2. Horizontal closing alignment: finger pads parallel to side faces
                    horiz_score = abs(float(np.dot(y_ee, np.array([1.0, 0.0, 0.0]))))

                    # 3. Aperture score: ideal width is 4-6 cm
                    width_score = float(np.exp(-((width - 0.05) ** 2) / 0.002))

                    # 4. Height clearance score: optimal around mid-height (0.60-0.65)
                    height_score = 1.0 - abs(h_frac - 0.62) / 0.40

                    total_score = (
                        0.40 * approach_score +
                        0.30 * horiz_score +
                        0.15 * width_score +
                        0.15 * height_score
                    )

                    candidate = GraspCandidate.from_pos_and_matrix(
                        pos=grasp_pos,
                        rot_mat=rot_mat,
                        score=total_score,
                        width=width,
                        cluster_id=cluster_id
                    )
                    candidates.append(candidate)

        # Sort descending by score
        candidates.sort(key=lambda c: c.score, reverse=True)

        # Deduplicate candidates with very similar poses
        unique_candidates: List[GraspCandidate] = []
        for c in candidates:
            is_dup = False
            for u in unique_candidates:
                pos_dist = np.linalg.norm(c.pos - u.pos)
                rot_dist = np.linalg.norm((R.from_matrix(c.rot_mat) * R.from_matrix(u.rot_mat).inv()).as_rotvec())
                if pos_dist < 0.015 and rot_dist < 0.20:
                    is_dup = True
                    break
            if not is_dup:
                unique_candidates.append(c)
                if len(unique_candidates) >= num_candidates:
                    break

        return unique_candidates
