"""
Phase 3 Validation Script: Perception & 6-DOF Grasp Planning.

Autonomous Perception-Driven Pick Pipeline:
1. Capture synchronized RGB and metric linear depth from 'shelf_cam'.
2. Back-project depth into 3D World Frame point cloud using camera intrinsics.
3. Crop shelf Region of Interest (ROI) and strip planar surfaces (shelf plates/walls) with RANSAC.
4. Cluster isolated manipulable objects with DBSCAN Euclidean clustering.
5. Identify target object (red box) using color signature and spatial bounds.
6. Synthesize 6-DOF antipodal grasp candidates using GeometricGraspPlanner (or GraspGenAdapter).
7. Filter grasp candidates against UR10e DLS inverse kinematics and link-to-shelf clearances.
8. Execute closed-loop 6-stage physical pick using perception-derived grasp pose.
9. Measure and verify:
   - Perception localization error vs ground-truth (< 15 mm).
   - Firm contact normal force (> 0.5 N).
   - Collision-free lift (> 8 cm) and retraction out of shelf.

Usage:
    python scripts/03_test_perception.py [--viewer] [--speed SPEED] [--use-graspgen]
"""

import os
import sys
import argparse
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

# Ensure workspace root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import mujoco
import mujoco.viewer
from control.dual_arm_controller import DualArmController
from control.trajectory import JointTrajectory
from control.collision_checker import CollisionChecker
from perception.pointcloud import capture_rgbd, depth_to_pointcloud, get_camera_intrinsics
from perception.segmentation import SceneSegmenter, ObjectCluster
from perception.grasp_planners import GeometricGraspPlanner, GraspGenAdapter
from perception.grasp_filter import GraspFilter, FilteredGrasp


def save_camera_snapshot(model, data, cam_name: str, out_path: str):
    """Renders and saves an offscreen snapshot from the specified camera."""
    try:
        import imageio
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id != -1:
            renderer.update_scene(data, camera=cam_id)
            img = renderer.render()
            imageio.imwrite(out_path, img)
        renderer.close()
    except Exception as e:
        print(f"  [WARN] Failed to capture snapshot '{cam_name}': {e}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Perception & 6-DOF Grasp Planning Validation.")
    parser.add_argument("--viewer", action="store_true", help="Launch interactive passive viewer.")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation speed multiplier in viewer mode.")
    parser.add_argument("--use-graspgen", action="store_true", help="Use GraspGen ML adapter (falls back to geometric if unavailable).")
    args = parser.parse_args()

    scene_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "dual_ur10e_scene.xml"))
    rec_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "recordings"))
    os.makedirs(rec_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("  PHASE 3: PERCEPTION & 6-DOF GRASP PLANNING VALIDATION")
    print("=" * 65)
    print(f"Scene Path       : {scene_path}")
    print(f"Recordings Dir   : {rec_dir}\n")

    # 1. Load Simulation
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    controller = DualArmController(model, data)
    collision_checker = CollisionChecker(model, data)
    kin1 = controller.kinematics_arm1

    # Reset to home keyframe posture
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # Settle physics for 100 steps
    for _ in range(100):
        mujoco.mj_step(model, data)

    # Ground-truth object position for evaluation comparison
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    if target_body_id == -1:
        print("[FATAL] target_object body not found in scene!")
        sys.exit(1)
    gt_obj_pos = data.xpos[target_body_id].copy()

    # Save initial camera snapshot
    save_camera_snapshot(model, data, "shelf_cam", os.path.join(rec_dir, "phase3_initial_shelf_view.png"))

    # =========================================================================
    # STEP 1: RGB-D Perception & Point Cloud Generation
    # =========================================================================
    print("-----------------------------------------------------------------")
    print("[STEP 1/4] Capturing RGB-D and Generating 3D Scene Point Cloud...")
    t_perc_start = time.time()

    intrinsics = get_camera_intrinsics(model, "shelf_cam", width=640, height=480)
    rgb, depth, cam_pos, cam_mat = capture_rgbd(model, data, "shelf_cam", width=640, height=480)
    points_w, colors_w, scene_pcd = depth_to_pointcloud(depth, rgb, intrinsics, cam_pos, cam_mat)

    # Voxel downsampling for performance
    pcd_down = scene_pcd.voxel_down_sample(voxel_size=0.005)
    print(f"  Camera Intrinsics   : fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, cx={intrinsics.cx:.1f}, cy={intrinsics.cy:.1f}")
    print(f"  Raw Scene Points    : {len(points_w):,}")
    print(f"  Downsampled Points  : {len(pcd_down.points):,}")

    # =========================================================================
    # STEP 2: Scene Parsing & Object Segmentation
    # =========================================================================
    print("\n[STEP 2/4] Segmenting Objects via ROI Cropping, RANSAC & DBSCAN...")
    segmenter = SceneSegmenter(
        shelf_roi_min=np.array([-0.62, 0.90, 0.45]),
        shelf_roi_max=np.array([ 0.62, 1.25, 1.45]),
        plane_distance_threshold=0.015,
        dbscan_eps=0.035,
        dbscan_min_points=30,
    )
    clusters = segmenter.segment_objects(pcd_down)
    print(f"  Detected Clusters   : {len(clusters)}")

    for idx, c in enumerate(clusters):
        print(f"    Cluster {c.cluster_id}: Centroid={np.round(c.centroid, 3)}, "
              f"Extent={np.round(c.extent, 3)}, Mean RGB={np.round(c.mean_color, 2)}, Points={len(c.points)}")

    # Target matching: isolate the target red box
    target_cluster: ObjectCluster = segmenter.find_target(
        clusters,
        expected_pos=np.array([0.45, 1.05, 0.90]),
        color_hint="red"
    )

    if target_cluster is None:
        print("[FATAL] Could not identify target object cluster from point cloud!")
        sys.exit(1)

    perc_centroid = target_cluster.centroid
    loc_error_mm = np.linalg.norm(perc_centroid - gt_obj_pos) * 1000.0
    print(f"  Target Cluster ID   : {target_cluster.cluster_id}")
    print(f"  Perceived Centroid  : {np.round(perc_centroid, 4)}")
    print(f"  Ground-Truth Center : {np.round(gt_obj_pos, 4)}")
    print(f"  Perception 3D Error : {loc_error_mm:.2f} mm (< 15.0 mm acceptance gate)")

    # Save target cluster point cloud
    ply_path = os.path.join(rec_dir, "phase3_target_cluster.ply")
    o3d.io.write_point_cloud(ply_path, target_cluster.pcd)
    print(f"  Saved Target PCD    : {ply_path}")

    # =========================================================================
    # STEP 3: 6-DOF Grasp Planning & Kinematic/Collision Filtering
    # =========================================================================
    print("\n[STEP 3/4] Synthesizing 6-DOF Grasps & Evaluating Reachability...")
    if args.use_graspgen:
        print("  Using GraspGen ML Planner Adapter...")
        planner = GraspGenAdapter(preferred_approach=np.array([0.0, 1.0, 0.0]))
    else:
        print("  Using Native Geometric Antipodal Planner...")
        planner = GeometricGraspPlanner(
            max_gripper_width=0.085,
            min_gripper_width=0.010,
            preferred_approach=np.array([0.0, 1.0, 0.0]),
            standoff_distance=0.15,
        )

    raw_candidates = planner.plan_grasps(target_cluster.pcd, num_candidates=10, cluster_id=target_cluster.cluster_id)
    print(f"  Generated Candidates: {len(raw_candidates)}")

    grasp_filter = GraspFilter(model, data, arm_id=1, standoff_dist=0.15)
    verified_grasps = grasp_filter.filter_candidates(raw_candidates, max_candidates_to_eval=10)
    print(f"  Kinematically Valid : {len(verified_grasps)} / {len(raw_candidates)}")

    if not verified_grasps:
        print("[FATAL] No kinematically reachable, collision-free grasp candidates found!")
        sys.exit(1)

    best_grasp: FilteredGrasp = verified_grasps[0]
    print("\n  Top Ranked Grasp Candidate:")
    print(f"    Grasp Pos         : {np.round(best_grasp.pos_grasp, 4)}")
    print(f"    Pre-Grasp Pos     : {np.round(best_grasp.pos_pregrasp, 4)}")
    print(f"    Grasp Orientation : {np.round(best_grasp.candidate.quat, 4)} [w, x, y, z]")
    print(f"    Approach Vector   : {np.round(best_grasp.candidate.approach_dir, 3)}")
    print(f"    Aperture Width    : {best_grasp.candidate.width * 1000.0:.1f} mm")
    print(f"    Composite Score   : {best_grasp.filter_score:.3f}")
    t_perc_total = time.time() - t_perc_start
    print(f"  Perception Pipeline Duration: {t_perc_total:.3f} s")

    # =========================================================================
    # STEP 4: Closed-Loop Physical Pick Execution
    # =========================================================================
    print("\n-----------------------------------------------------------------")
    print("[STEP 4/4] Executing Perception-Driven Physical Pick on Arm 1...")

    pos_grasp = best_grasp.pos_grasp
    quat_grasp = best_grasp.candidate.quat
    pos_pregrasp = best_grasp.pos_pregrasp
    approach_dir = best_grasp.candidate.approach_dir / np.linalg.norm(best_grasp.candidate.approach_dir)

    # Vertical lift (+12 cm) to clear shelf floor
    pos_lift = pos_grasp + np.array([0.0, 0.0, 0.12])
    # Horizontal retraction (-25 cm) along backward approach corridor
    pos_retract = pos_lift - 0.25 * approach_dir

    # Solve IK for lift and retract trajectories
    # Temporarily set arm joints to grasp pose for accurate sequential unwrapping
    saved_qpos = data.qpos.copy()
    for i, addr in enumerate(kin1.qpos_addrs):
        data.qpos[addr] = best_grasp.q_grasp[i]
    mujoco.mj_forward(model, data)

    ok_lf, q_lift = kin1.solve_ik_closed_loop(pos_lift, quat_grasp)
    for i, addr in enumerate(kin1.qpos_addrs):
        data.qpos[addr] = q_lift[i]
    mujoco.mj_forward(model, data)

    ok_rt, q_retract = kin1.solve_ik_closed_loop(pos_retract, quat_grasp)

    # Restore simulation state
    data.qpos[:] = saved_qpos
    mujoco.mj_forward(model, data)

    if not (ok_lf and ok_rt):
        print(f"[ERROR] Lift/Retract IK solve failed: lift={ok_lf}, retract={ok_rt}")
        sys.exit(1)

    q_home = kin1.get_joint_positions()
    q_pregrasp = best_grasp.q_pregrasp
    q_grasp = best_grasp.q_grasp

    stages = [
        {"name": "1_PREGRASP", "type": "move", "q_from": q_home,     "q_to": q_pregrasp, "des_pos": pos_pregrasp, "gripper": 0.0,  "duration": 2.0},
        {"name": "2_APPROACH", "type": "move", "q_from": q_pregrasp, "q_to": q_grasp,    "des_pos": pos_grasp,    "gripper": 0.0,  "duration": 1.5},
        {"name": "3_GRASP",    "type": "grip", "q_hold": q_grasp,                         "des_pos": pos_grasp,    "gripper": 0.75, "duration": 1.0},
        {"name": "4_LIFT",     "type": "move", "q_from": q_grasp,    "q_to": q_lift,     "des_pos": pos_lift,     "gripper": 0.75, "duration": 1.5},
        {"name": "5_RETRACT",  "type": "move", "q_from": q_lift,     "q_to": q_retract,  "des_pos": pos_retract,  "gripper": 0.75, "duration": 2.0},
        {"name": "6_HOLD",     "type": "hold", "q_hold": q_retract,                       "des_pos": pos_retract,  "gripper": 0.75, "duration": 1.5},
    ]

    current_stage_idx = 0
    stage_start_time = 0.0
    sim_time = 0.0
    dt = model.opt.timestep

    current_traj = JointTrajectory(q_home, q_pregrasp, duration=stages[0]["duration"])
    max_tracking_err = 0.0
    grasp_confirmed = False
    pick_successful = False

    def step_simulation(viewer=None):
        nonlocal sim_time, current_stage_idx, stage_start_time, current_traj
        nonlocal max_tracking_err, grasp_confirmed, pick_successful

        if current_stage_idx >= len(stages):
            return True

        stage = stages[current_stage_idx]
        t_in_stage = sim_time - stage_start_time

        if stage["type"] == "move":
            q_target, _, is_done = current_traj.evaluate(t_in_stage)
        else:
            q_target = stage["q_hold"]
            is_done = t_in_stage >= stage["duration"]

        controller.set_arm_joint_targets(1, q_target)
        controller.set_gripper(1, stage["gripper"])
        controller.set_gripper(2, 0.0)

        curr_ee_pos, _ = controller.get_ee_pose(1)
        cart_err = float(np.linalg.norm(curr_ee_pos - stage["des_pos"]))
        if stage["type"] != "move":
            max_tracking_err = max(max_tracking_err, cart_err)

        mujoco.mj_step(model, data)
        sim_time += dt

        if is_done:
            # Capture milestone snapshot
            snap_path = os.path.join(rec_dir, f"phase3_stage_{stage['name']}.png")
            save_camera_snapshot(model, data, "shelf_cam", snap_path)

            if stage["name"] == "3_GRASP":
                is_grasped, f_norm = controller.check_grasp(1, "target_object")
                grasp_confirmed = is_grasped
                print(f"  --> [GRASP CHECK] Firm contact: {is_grasped} (Normal Force: {f_norm:.2f} N)")

            elif stage["name"] == "6_HOLD":
                curr_obj_pos = data.xpos[target_body_id]
                z_delta = curr_obj_pos[2] - gt_obj_pos[2]
                is_grasped, f_norm = controller.check_grasp(1, "target_object")
                print(f"  --> [FINAL HOLD] Height Delta: +{z_delta*100:.1f} cm | Grasped: {is_grasped} ({f_norm:.2f} N)")
                if z_delta >= 0.08 and is_grasped:
                    pick_successful = True

            current_stage_idx += 1
            if current_stage_idx < len(stages):
                next_stage = stages[current_stage_idx]
                stage_start_time = sim_time
                print(f"[STAGE] Entering {next_stage['name']} (duration={next_stage['duration']}s)...")

                if next_stage["type"] == "move":
                    current_traj = JointTrajectory(
                        next_stage["q_from"], next_stage["q_to"],
                        duration=next_stage["duration"]
                    )
                return False
            else:
                return True
        return False

    print(f"[STAGE] Entering {stages[0]['name']} (duration={stages[0]['duration']}s)...")

    if args.viewer:
        print("[INFO] Running in interactive viewer mode.\n")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                t0 = time.time()
                finished = step_simulation(viewer)
                viewer.sync()

                if finished:
                    print("\n[COMPLETE] All stages finished! Robot is holding the object steady in mid-air.")
                    print("[INFO] The 3D viewer remains open for inspection. Close the viewer window when you wish to exit.\n")
                    while viewer.is_running():
                        controller.set_arm_joint_targets(1, q_retract)
                        controller.set_gripper(1, 0.75)
                        controller.set_gripper(2, 0.0)
                        mujoco.mj_step(model, data)
                        viewer.sync()
                        time.sleep(dt)
                    break

                elapsed = time.time() - t0
                sleep_t = (dt / args.speed) - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
    else:
        # Headless benchmark
        t0 = time.time()
        step_count = 0
        while current_stage_idx < len(stages):
            finished = step_simulation()
            step_count += 1
            if finished:
                break
        real_duration = time.time() - t0
        print(f"\n[BENCHMARK] Executed {step_count} physics steps in {real_duration:.3f}s ({step_count/real_duration:.0f} Hz).")

    # =========================================================================
    # Final Evaluation & Verification Report
    # =========================================================================
    print("\n" + "=" * 65)
    print("               PHASE 3 EVALUATION REPORT")
    print("=" * 65)
    final_obj_pos = data.xpos[target_body_id]
    final_z_delta = final_obj_pos[2] - gt_obj_pos[2]
    is_firmly_held, final_force = controller.check_grasp(1, "target_object")
    has_shelf_collision, coll_pairs = collision_checker.check_shelf_collision(1)

    print(f"  Ground-Truth Initial Pos  : {np.round(gt_obj_pos, 4)}")
    print(f"  Perception Centroid       : {np.round(perc_centroid, 4)}")
    print(f"  Perception Localization Err: {loc_error_mm:.2f} mm  (Threshold < 15.0 mm)")
    print(f"  Grasp Candidate Pos       : {np.round(pos_grasp, 4)}")
    print(f"  Final Object Position     : {np.round(final_obj_pos, 4)}")
    print(f"  Vertical Lift Clearance   : +{final_z_delta*100:.1f} cm  (Threshold >= 8.0 cm)")
    print(f"  Cartesian Tracking Error  : {max_tracking_err*1000:.2f} mm")
    print(f"  Grasp Contact Normal Force: {final_force:.2f} N  (Threshold > 0.5 N)")
    print(f"  Grasp Confirmed           : {'YES [OK]' if is_firmly_held else 'NO [FAIL]'}")
    print(f"  Shelf Collision Status    : {'NONE [CLEAN]' if not has_shelf_collision else f'COLLIDED: {coll_pairs}'}")

    # Capture overhead snapshot
    save_camera_snapshot(model, data, "overhead_cam", os.path.join(rec_dir, "phase3_final_overhead.png"))

    gates_passed = (
        loc_error_mm < 15.0 and
        pick_successful and
        is_firmly_held and
        not has_shelf_collision and
        final_z_delta >= 0.08
    )

    if gates_passed:
        print("\n>>> [ACCEPTANCE GATE PASSED] Perception & 6-DOF Grasp Planning Verified! <<<\n")
        return 0
    else:
        print("\n>>> [ACCEPTANCE GATE FAILED] Pipeline did not meet all acceptance criteria. <<<\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
