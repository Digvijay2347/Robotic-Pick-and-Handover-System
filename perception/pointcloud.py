"""
RGB-D Depth Capture, Camera Intrinsics & 3D Point Cloud Generator for MuJoCo.

Provides:
- Camera intrinsic extraction from MuJoCo field-of-view and sensor specifications.
- Synchronized metric depth and RGB rendering.
- Exact pinhole back-projection from 2D pixel space (u, v, d) to 3D World Frame (X, Y, Z).
- Multi-camera point cloud fusion and Open3D integration.
"""

from typing import Tuple, List, Optional
from dataclasses import dataclass
import numpy as np
import mujoco
import open3d as o3d


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""
    width: int
    height: int
    fovy: float  # Vertical FOV in degrees
    fx: float    # Focal length along X (pixels)
    fy: float    # Focal length along Y (pixels)
    cx: float    # Principal point X (pixels)
    cy: float    # Principal point Y (pixels)
    K: np.ndarray  # 3x3 Intrinsic matrix


def get_camera_intrinsics(model: mujoco.MjModel, camera_name: str, width: int = 640, height: int = 480) -> CameraIntrinsics:
    """Derives camera intrinsics from MuJoCo camera model."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id == -1:
        raise ValueError(f"Camera '{camera_name}' not found in model!")

    fovy = float(model.cam_fovy[cam_id])
    fy = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
    fx = fy  # Square pixels

    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return CameraIntrinsics(
        width=width,
        height=height,
        fovy=fovy,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        K=K
    )


def capture_rgbd(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    width: int = 640,
    height: int = 480
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Renders synchronized RGB and linear metric depth images from specified camera.

    Returns:
        rgb: (H, W, 3) uint8 image.
        depth: (H, W) float32 linear depth map in meters.
        cam_pos: (3,) Camera center position in World Frame.
        cam_mat: (3, 3) Camera rotation matrix in World Frame.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id == -1:
        raise ValueError(f"Camera '{camera_name}' not found in model!")

    renderer = mujoco.Renderer(model, height=height, width=width)

    # 1. Render RGB
    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render().copy()

    # 2. Render linear metric depth
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_id)
    depth = renderer.render().copy()
    renderer.close()

    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()

    return rgb, depth, cam_pos, cam_mat


def depth_to_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: CameraIntrinsics,
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
    depth_min: float = 0.20,
    depth_max: float = 3.50,
) -> Tuple[np.ndarray, np.ndarray, o3d.geometry.PointCloud]:
    """
    Back-projects 2D depth map into 3D Cartesian coordinates in World Frame.

    MuJoCo Camera Frame Convention:
    - X_c: Right
    - Y_c: Up
    - Z_c: Negative viewing direction (camera looks down -Z_c)

    Returns:
        points_world: (N, 3) float64 array of coordinates in World Frame.
        colors: (N, 3) float64 array of normalized RGB colors in [0, 1].
        pcd: open3d.geometry.PointCloud object.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Mask valid depth values
    valid = (depth > depth_min) & (depth < depth_max) & np.isfinite(depth)
    u_val = u[valid]
    v_val = v[valid]
    d_val = depth[valid]

    # Back-projection to camera frame (in meters)
    # x = (u - cx) * depth / fx
    # y = -(v - cy) * depth / fy (inverted because image row v increases downward)
    # z = -depth (MuJoCo camera optical axis looks along -Z)
    x_c = (u_val - intrinsics.cx) * d_val / intrinsics.fx
    y_c = -(v_val - intrinsics.cy) * d_val / intrinsics.fy
    z_c = -d_val

    points_camera = np.column_stack([x_c, y_c, z_c])

    # Transform to World Frame: P_w = R_cam * P_c + T_cam
    points_world = points_camera @ cam_mat.T + cam_pos

    # Normalized RGB colors
    colors = (rgb[valid].astype(np.float64) / 255.0)

    # Open3D PointCloud creation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return points_world, colors, pcd


def capture_scene_pointcloud(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str = "shelf_cam",
    voxel_size: float = 0.005,
) -> o3d.geometry.PointCloud:
    """Captures RGB-D and returns a filtered, voxel-downsampled World PointCloud."""
    intrinsics = get_camera_intrinsics(model, camera_name)
    rgb, depth, cam_pos, cam_mat = capture_rgbd(model, data, camera_name)
    _, _, pcd = depth_to_pointcloud(depth, rgb, intrinsics, cam_pos, cam_mat)

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    return pcd
