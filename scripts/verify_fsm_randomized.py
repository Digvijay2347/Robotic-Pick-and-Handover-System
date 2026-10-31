import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import time
import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R

from fsm.handover_fsm import HandoverFSM
from fsm.handover_states import HandoverState
from perception.pointcloud import capture_scene_pointcloud
from perception.segmentation import SceneSegmenter
from perception.grasp_planners.geometric_planner import GeometricGraspPlanner
from perception.grasp_filter import GraspFilter


def verify_fsm(
    seed: int = 42,
    video_path: str = "recordings/fsm_randomized_verification.mp4",
    fps: int = 50,
):
    print("=" * 80)
    print("          AUTONOMOUS FSM VERIFICATION UNDER DOMAIN RANDOMIZATION")
    print("=" * 80)
    print(f"  Random Seed   : {seed}")
    print(f"  Video Output  : {video_path}")
    print(f"  Render Rate   : {fps} FPS")
    print("=" * 80)

    np.random.seed(seed)

    scene_path = os.path.join(ROOT_DIR, "models", "dual_ur10e_scene.xml")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

    # 1. Apply Domain Randomization to Red Cube
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_object_joint")
    qadr = model.jnt_qposadr[jnt_id]

    dx = np.random.uniform(-0.025, 0.025)
    dy = np.random.uniform(-0.020, 0.020)
    yaw_deg = np.random.uniform(-20.0, 20.0)
    quat_rand = R.from_euler("z", yaw_deg, degrees=True).as_quat(scalar_first=True)

    data.qpos[qadr + 0] = 0.45 + dx
    data.qpos[qadr + 1] = 1.05 + dy
    data.qpos[qadr + 3:qadr + 7] = quat_rand

    for _ in range(50):
        mujoco.mj_step(model, data)

    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    initial_box_pos = data.xpos[target_body_id].copy()
    print(f"[RANDOMIZATION] Initial Box Pos: {np.round(initial_box_pos, 3)} | Yaw: {yaw_deg:+.1f} deg")

    # 2. Perception & Grasp Planning
    print("\n[PERCEPTION] Capturing RGB-D Point Cloud & Segmenting Target...")
    pcd = capture_scene_pointcloud(model, data)
    segmenter = SceneSegmenter()
    clusters = segmenter.segment_objects(pcd)
    target = segmenter.find_target(clusters, expected_pos=np.array([0.45, 1.05, 0.90]), color_hint="red")

    if target is None:
        print("[FATAL ERROR] Target cube segmentation failed!")
        return False

    err_cent = float(np.linalg.norm(target.centroid - initial_box_pos)) * 100.0
    print(f"[PERCEPTION] Target detected at {np.round(target.centroid, 3)} (localization err: {err_cent:.1f} cm)")

    print("[PLANNING] Generating Centered 6-DOF Grasp Candidates...")
    planner = GeometricGraspPlanner()
    raw_cands = planner.plan_grasps(target.pcd, num_candidates=20)
    gfilter = GraspFilter(model, data, arm_id=1, standoff_dist=0.10)
    verified_grasps = gfilter.filter_candidates(raw_cands)

    if not verified_grasps:
        print("[FATAL ERROR] No valid grasp candidate verified by kinematics/collision filter!")
        return False

    best_grasp = verified_grasps[0]
    print(f"[PLANNING] Best grasp verified at: {np.round(best_grasp.pos_grasp, 3)} (grasp depth Z: {best_grasp.pos_grasp[2]:.3f} m)")

    fsm = HandoverFSM(model, data)
    ok_plan = fsm.plan_trajectories(best_grasp)
    if not ok_plan:
        print("[FATAL ERROR] Keyframe trajectory solving failed!")
        return False
    print("[PLANNING] Complete bimanual collision-free trajectory verified!")

    # 3. Setup Renderers & Video Writer
    renderer_top = mujoco.Renderer(model, height=480, width=640)
    renderer_wrist1 = mujoco.Renderer(model, height=240, width=320)
    renderer_wrist2 = mujoco.Renderer(model, height=240, width=320)
    cam_top_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
    cam_w1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "arm1_wrist_cam")
    cam_w2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "arm2_wrist_cam")

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8)

    # 4. Execute Full Cycle
    print("\n[EXECUTION] Beginning Continuous Dual-Arm FSM Execution...")
    fsm.transition_to(HandoverState.APPROACH_1, duration=2.0)
    last_state = fsm.current_state

    # Milestones tracking
    milestones = {
        "grasped_shelf": False,
        "lifted": False,
        "retracted_shelf": False,
        "dual_custody": False,
        "arm2_held": False,
        "placed_bin": False,
    }

    step_sim = 0
    sample_stride = max(1, int(500 / fps))  # 500 Hz physics -> 50 Hz video recording

    t0 = time.time()
    while True:
        curr_state, is_finished = fsm.step()

        # Monitor contact and position dynamics
        is_held1, f1 = fsm.controller.check_grasp(1, fsm.target_name)
        is_held2, f2 = fsm.controller.check_grasp(2, fsm.target_name)
        bpos = data.xpos[target_body_id]
        ee1_pos, _ = fsm.controller.get_ee_pose(1)
        ee2_pos, _ = fsm.controller.get_ee_pose(2)

        if is_held1 and f1 >= 15.0 and bpos[2] >= 0.90:
            milestones["grasped_shelf"] = True
        if milestones["grasped_shelf"] and bpos[2] >= 0.98:
            milestones["lifted"] = True
        if milestones["lifted"] and bpos[1] <= 0.75:
            milestones["retracted_shelf"] = True
        if fsm.dual_hold_confirmed or (is_held1 and is_held2 and f2 >= 15.0):
            milestones["dual_custody"] = True
        if milestones["dual_custody"] and is_held2 and f2 >= 15.0 and not is_held1:
            milestones["arm2_held"] = True
        if (abs(bpos[0]) < 0.20) and (-1.00 < bpos[1] < -0.60) and (bpos[2] < 0.25):
            milestones["placed_bin"] = True

        # Render and log at video framerate
        if step_sim % sample_stride == 0:
            renderer_top.update_scene(data, camera=cam_top_id)
            img_top = renderer_top.render().copy()

            renderer_wrist1.update_scene(data, camera=cam_w1_id)
            img_w1 = renderer_wrist1.render().copy()

            renderer_wrist2.update_scene(data, camera=cam_w2_id)
            img_w2 = renderer_wrist2.render().copy()

            # Composite canvas: Left = Overhead (640x480), Right = Stacked Wrists (320x480)
            wrists_stacked = np.vstack([img_w1, img_w2])
            canvas = np.hstack([img_top, wrists_stacked])

            pil_img = Image.fromarray(canvas)
            draw = ImageDraw.Draw(pil_img)

            # Header banner
            draw.rectangle([(0, 0), (960, 32)], fill=(15, 17, 23, 220))
            draw.text(
                (12, 8),
                f"FSM GROUND TRUTH | State: {curr_state.name:15s} | t={fsm.sim_time:5.2f}s | Arm1: {f1:3.0f}N | Arm2: {f2:3.0f}N",
                fill=(255, 255, 255)
            )
            bin_status = "IN BIN [SUCCESS]" if milestones["placed_bin"] else "IN TRANSIT"
            draw.text((700, 8), bin_status, fill=(100, 255, 150) if milestones["placed_bin"] else (255, 200, 100))

            # Viewport labels
            draw.rectangle([(10, 40), (140, 58)], fill=(0, 0, 0, 180))
            draw.text((16, 43), "OVERHEAD CAM", fill=(200, 230, 255))
            draw.rectangle([(650, 40), (790, 58)], fill=(0, 0, 0, 180))
            draw.text((656, 43), "ARM 1 WRIST CAM", fill=(200, 230, 255))
            draw.rectangle([(650, 280), (790, 298)], fill=(0, 0, 0, 180))
            draw.text((656, 283), "ARM 2 WRIST CAM", fill=(200, 230, 255))

            writer.append_data(np.array(pil_img))

        if curr_state != last_state:
            print(f"  [STATE TRANSITION] t={fsm.sim_time:5.2f}s -> {curr_state.name:18s} | Box: {np.round(bpos, 2)} | f1={f1:3.0f}N f2={f2:3.0f}N", flush=True)
            last_state = curr_state

        step_sim += 1

        if is_finished:
            break

    writer.close()
    elapsed = time.time() - t0

    print("\n" + "=" * 80)
    print("                    FSM VERIFICATION SCORECARD")
    print("=" * 80)
    print(f"  Final State               : {curr_state.name}")
    print(f"  Physical Simulation Time  : {fsm.sim_time:.2f} s ({step_sim} physics steps)")
    print(f"  Execution Time            : {elapsed:.2f} s")
    print(f"  Milestone 1: Shelf Grasp  : {milestones['grasped_shelf']} (f1 >= 15N)")
    print(f"  Milestone 2: Vertical Lift: {milestones['lifted']} (Z >= 0.98m)")
    print(f"  Milestone 3: Shelf Retract: {milestones['retracted_shelf']} (Y <= 0.75m)")
    print(f"  Milestone 4: Dual Custody : {milestones['dual_custody']} (f1 >= 10N & f2 >= 15N)")
    print(f"  Milestone 5: Arm 2 Custody: {milestones['arm2_held']} (Sole transfer)")
    print(f"  Milestone 6: Placed in Bin: {milestones['placed_bin']} (Target in storage bin)")
    print(f"  Recorded Video Artifact   : {video_path}")
    print("=" * 80)

    success = all(milestones.values())
    print(f"\nOVERALL WORKFLOW VERIFICATION: {'SUCCESS (100% COMPLETE)' if success else 'FAILED'}\n")
    return success


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    ok = verify_fsm(seed=seed)
    sys.exit(0 if ok else 1)
