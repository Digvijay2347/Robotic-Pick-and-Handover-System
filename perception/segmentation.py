"""
Point Cloud Scene Parsing & Object Segmentation for Wall Shelf Manipulation.

Provides:
- 3D Region of Interest (ROI) spatial cropping.
- RANSAC plane removal to strip horizontal shelf levels and vertical dividers.
- DBSCAN Euclidean clustering to isolate individual manipulable objects.
- ObjectCluster representation with centroids, extents, and color profiles.
- Color and spatial target matching to identify desired grasp objects.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False
    o3d = None

from perception.pointcloud import NumpyPointCloud


@dataclass
class ObjectCluster:
    """Represents a segmented 3D object cluster from the scene point cloud."""
    cluster_id: int
    pcd: Any
    points: np.ndarray
    colors: np.ndarray
    centroid: np.ndarray          # 3D spatial center [x, y, z]
    obb: Any
    extent: np.ndarray            # [length, width, height] along principal axes
    mean_color: np.ndarray        # Normalized RGB [r, g, b] in [0, 1]


class SceneSegmenter:
    """Segments objects inside a multi-compartment shelf from RGB-D point clouds."""

    def __init__(
        self,
        shelf_roi_min: np.ndarray = np.array([-0.62, 0.90, 0.45]),
        shelf_roi_max: np.ndarray = np.array([ 0.62, 1.25, 1.45]),
        plane_distance_threshold: float = 0.015,
        dbscan_eps: float = 0.035,
        dbscan_min_points: int = 30,
    ):
        self.shelf_roi_min = shelf_roi_min
        self.shelf_roi_max = shelf_roi_max
        self.plane_distance_threshold = plane_distance_threshold
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_points = dbscan_min_points

    def crop_shelf(self, pcd: Any) -> Any:
        """Crops point cloud to the shelf bounding volume."""
        if hasattr(pcd, "crop"):
            if HAS_O3D and o3d is not None:
                bbox = o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=self.shelf_roi_min,
                    max_bound=self.shelf_roi_max
                )
                return pcd.crop(bbox)
            else:
                return pcd.crop(self.shelf_roi_min, self.shelf_roi_max)
        return pcd

    def remove_shelf_planes(
        self,
        pcd: Any,
        max_planes: int = 5,
        min_inliers: int = 250,
    ) -> Any:
        """Iteratively strips large planar surfaces (shelf plates, dividers, walls)."""
        current_pcd = pcd
        for _ in range(max_planes):
            if len(current_pcd.points) < min_inliers:
                break
            _, inliers = current_pcd.segment_plane(
                distance_threshold=self.plane_distance_threshold,
                ransac_n=3,
                num_iterations=200
            )
            if len(inliers) < min_inliers:
                break
            current_pcd = current_pcd.select_by_index(inliers, invert=True)

        return current_pcd

    def segment_objects(self, pcd: Any) -> List[ObjectCluster]:
        """
        Segments and extracts distinct 3D object clusters from point cloud.
        """
        cropped = self.crop_shelf(pcd)
        non_planar = self.remove_shelf_planes(cropped)

        if len(non_planar.points) < self.dbscan_min_points:
            return []

        labels = np.array(non_planar.cluster_dbscan(
            eps=self.dbscan_eps,
            min_points=self.dbscan_min_points,
            print_progress=False
        ))

        all_points = np.asarray(non_planar.points)
        all_colors = np.asarray(non_planar.colors)

        clusters: List[ObjectCluster] = []
        max_label = labels.max()

        for cid in range(max_label + 1):
            mask = (labels == cid)
            c_points = all_points[mask]
            c_colors = all_colors[mask] if len(all_colors) == len(all_points) else np.zeros_like(c_points)

            # Filter out tiny noise clusters or oversized divider remnants
            if len(c_points) < self.dbscan_min_points or len(c_points) > 5000:
                continue

            if HAS_O3D and o3d is not None:
                c_pcd = o3d.geometry.PointCloud()
                c_pcd.points = o3d.utility.Vector3dVector(c_points)
                c_pcd.colors = o3d.utility.Vector3dVector(c_colors)
            else:
                c_pcd = NumpyPointCloud(c_points, c_colors)

            # Accurate 3D spatial center from bounding volume midpoint
            aabb_min = np.min(c_points, axis=0)
            aabb_max = np.max(c_points, axis=0)
            aabb_min_adj = aabb_min.copy()
            aabb_min_adj[2] = max(self.shelf_roi_min[2], aabb_min[2] - self.plane_distance_threshold)
            centroid = 0.5 * (aabb_min_adj + aabb_max)
            mean_color = np.mean(c_colors, axis=0) if len(c_colors) > 0 else np.array([0.5, 0.5, 0.5])

            try:
                obb = c_pcd.get_oriented_bounding_box()
                extent = np.asarray(obb.extent)
            except Exception:
                aabb = c_pcd.get_axis_aligned_bounding_box()
                extent = np.asarray(aabb.get_extent())
                if HAS_O3D and o3d is not None:
                    obb = o3d.geometry.OrientedBoundingBox(centroid, np.eye(3), extent)
                else:
                    obb = aabb

            # Skip clusters that are overly elongated (like thin divider edges)
            if np.max(extent) > 0.40:
                continue

            clusters.append(ObjectCluster(
                cluster_id=cid,
                pcd=c_pcd,
                points=c_points,
                colors=c_colors,
                centroid=centroid,
                obb=obb,
                extent=extent,
                mean_color=mean_color
            ))

        return clusters

    def find_target(
        self,
        clusters: List[ObjectCluster],
        expected_pos: Optional[np.ndarray] = None,
        color_hint: str = "red",
        max_pos_distance: float = 0.20,
    ) -> Optional[ObjectCluster]:
        """
        Selects the target object cluster matching expected color and/or position.

        Args:
            clusters: List of detected ObjectClusters.
            expected_pos: Approximate 3D position [x, y, z].
            color_hint: "red", "green", "blue", "yellow", or "any".
            max_pos_distance: Maximum distance from expected_pos in meters.
        """
        if not clusters:
            return None

        candidates = []
        for c in clusters:
            r, g, b = c.mean_color

            # Color signature matching
            color_match = False
            if color_hint == "red" and (r > 0.35 and r > g * 1.5 and r > b * 1.5):
                color_match = True
            elif color_hint == "green" and (g > 0.30 and g > r * 1.2 and g > b * 1.2):
                color_match = True
            elif color_hint == "yellow" and (r > 0.35 and g > 0.30 and b < 0.25):
                color_match = True
            elif color_hint == "blue" and (b > 0.30 and b > r * 1.2):
                color_match = True
            elif color_hint == "any":
                color_match = True

            dist = np.linalg.norm(c.centroid - expected_pos) if expected_pos is not None else 0.0

            if expected_pos is not None and dist > max_pos_distance:
                continue

            candidates.append((color_match, -dist, c))

        if not candidates:
            # Fallback to closest cluster if spatial prior was provided
            if expected_pos is not None and clusters:
                return min(clusters, key=lambda c: np.linalg.norm(c.centroid - expected_pos))
            return clusters[0] if clusters else None

        # Sort by color match first, then by closest distance
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]
