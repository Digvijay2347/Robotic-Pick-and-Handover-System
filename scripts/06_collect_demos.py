"""
Phase 6: Automated Expert Demonstration Dataset Collector (HDF5 & LeRobot Format).

Orchestrates batch data collection for dual-arm pick, handover, and place manipulation:
1. Resets environment with optional domain randomization (object position and orientation).
2. Executes perception-guided 16-stage expert handover trajectory via HandoverFSM.
3. Samples observations and 14-DOF actions at 50 Hz control rate.
4. Records multi-camera RGB image streams ('top', 'wrist1', 'wrist2') with zero-allocation renderers.
5. Applies quality gate filtering: discards any failed or incomplete attempts automatically.
6. Exports datasets in both consolidated ALOHA HDF5 and LeRobot dataset formats with full metadata.

Usage:
    python scripts/06_collect_demos.py --num-episodes 50 --save-images --randomize
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, List, Optional
import numpy as np

# Ensure workspace root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import mujoco
from fsm.handover_states import HandoverState
from fsm.handover_fsm import HandoverFSM
from envs.dual_ur10e_env import DualUR10eHandoverEnv
from dataset.recorder import EpisodeRecorder
from dataset.dataset_utils import inspect_dataset
from perception import (
    capture_scene_pointcloud,
    SceneSegmenter,
    GeometricGraspPlanner,
    GraspFilter,
)


def collect_demonstrations(
    num_episodes: int = 50,
    output_dir: str = "data",
    dataset_name: str = "handover_demos.hdf5",
    save_images: bool = True,
    randomize: bool = True,
    compression: str = "lzf",
    fps: int = 50,
    max_attempts: int = 100,
):
    """
    Collects and logs expert demonstration episodes.
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("      PHASE 6: EXPERT DEMONSTRATION DATASET COLLECTION (ALOHA / LEROBOT)")
    print("=" * 80)
    print(f"  Target Episodes   : {num_episodes}")
    print(f"  Output Directory  : {output_dir}")
    print(f"  Dataset Name      : {dataset_name}")
    print(f"  Save Images (RGB) : {save_images}")
    print(f"  Compression       : {compression}")
    print(f"  Control Rate      : {fps} Hz (500 Hz internal physics)")
    print(f"  Randomization     : {randomize}")
    print("=" * 80 + "\n")

    scene_path = os.path.join(ROOT_DIR, "models", "dual_ur10e_scene.xml")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    env = DualUR10eHandoverEnv(
        model_path=scene_path,
        obs_type="pixels" if save_images else "state",
        action_mode="absolute",
        reward_type="dense",
        randomize_on_reset=randomize,
    )

    recorder = EpisodeRecorder(
        output_dir=output_dir,
        dataset_name=dataset_name,
        fps=fps,
        save_images=save_images,
        compression=compression,
    )

    # Persistent camera renderers if saving images
    renderer_top = None
    renderer_wrist = None
    cam_top_id = -1
    cam_wrist1_id = -1
    cam_wrist2_id = -1

    if save_images:
        renderer_top = mujoco.Renderer(model, height=480, width=640)
        renderer_wrist = mujoco.Renderer(model, height=240, width=320)
        cam_top_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
        cam_wrist1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "arm1_wrist_cam")
        cam_wrist2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "arm2_wrist_cam")

    # Joint limits for 14-DOF action normalization
    q_min1, q_max1 = env.q_min_arm1, env.q_max_arm1
    q_min2, q_max2 = env.q_min_arm2, env.q_max_arm2

    successful_count = 0
    attempt_count = 0
    t_start_total = time.time()

    while successful_count < num_episodes and attempt_count < max_attempts:
        attempt_count += 1
        ep_idx = successful_count
        print(f"\n>>> [EPISODE {successful_count + 1}/{num_episodes}] (Attempt {attempt_count}/{max_attempts}) <<<", flush=True)

        # 1. Reset simulation to initial keyframe
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        else:
            mujoco.mj_resetData(model, data)

        # Apply domain randomization on target object
        if randomize and env.target_joint_id != -1:
            qadr = model.jnt_qposadr[env.target_joint_id]
            dx = np.random.uniform(-0.03, 0.03)
            dy = np.random.uniform(-0.02, 0.02)
            data.qpos[qadr + 0] = 0.45 + dx
            data.qpos[qadr + 1] = 1.05 + dy
            yaw_deg = np.random.uniform(-25.0, 25.0)
            from scipy.spatial.transform import Rotation as R
            quat_rand = R.from_euler("z", yaw_deg, degrees=True).as_quat(scalar_first=True)
            data.qpos[qadr + 3:qadr + 7] = quat_rand

        # Settle physics
        for _ in range(50):
            mujoco.mj_step(model, data)

        initial_box_pos = data.xpos[env.target_body_id].copy()
        print(f"  Target initial pose: {np.round(initial_box_pos, 3)}", flush=True)

        # 2. Perception & 6-DOF Grasp Planning
        fsm = HandoverFSM(model, data)
        pcd = capture_scene_pointcloud(model, data)
        segmenter = SceneSegmenter()
        clusters = segmenter.segment_objects(pcd)
        target = segmenter.find_target(clusters, color_hint="red")

        if target is None:
            print("  [WARN] Perception failed to find target object! Skipping attempt.", flush=True)
            continue

        planner = GeometricGraspPlanner()
        raw_cands = planner.plan_grasps(target.pcd)
        gfilter = GraspFilter(model, data, arm_id=1, standoff_dist=0.15)
        verified_grasps = gfilter.filter_candidates(raw_cands)

        if not verified_grasps:
            print("  [WARN] No reachable grasp candidates found! Skipping attempt.", flush=True)
            continue

        best_grasp = verified_grasps[0]
        ok_plan = fsm.plan_trajectories(best_grasp)
        if not ok_plan:
            print("  [WARN] Trajectory planning failed! Skipping attempt.", flush=True)
            continue

        # 3. Execute Handover FSM and Record at 50 Hz
        fsm.transition_to(HandoverState.APPROACH_1, duration=2.0)
        step_physics = 0
        last_state = fsm.current_state
        t_ep_start = time.time()

        while True:
            # Sample at 50 Hz (every 10 physics steps = 0.020s)
            if step_physics % 10 == 0:
                # Observations: qpos (14), qvel (14)
                q1 = data.qpos[fsm.kin1.qpos_addrs].copy()
                g1 = float(data.qpos[env.grip1_qposadr] / 0.8)
                q2 = data.qpos[fsm.kin2.qpos_addrs].copy()
                g2 = float(data.qpos[env.grip2_qposadr] / 0.8)
                qpos_14d = np.concatenate([q1, [g1], q2, [g2]]).astype(np.float64)

                v1 = data.qvel[fsm.kin1.dof_addrs].copy()
                gv1 = float(data.qvel[env.grip1_dofadr])
                v2 = data.qvel[fsm.kin2.dof_addrs].copy()
                gv2 = float(data.qvel[env.grip2_dofadr])
                qvel_14d = np.concatenate([v1, [gv1], v2, [gv2]]).astype(np.float64)

                obs_dict: Dict[str, Any] = {
                    "qpos": qpos_14d,
                    "qvel": qvel_14d,
                }

                # Render cameras if requested
                if save_images:
                    renderer_top.update_scene(data, camera=cam_top_id)
                    renderer_wrist.update_scene(data, camera=cam_wrist1_id)
                    img_wrist1 = renderer_wrist.render().copy()
                    renderer_wrist.update_scene(data, camera=cam_wrist2_id)
                    img_wrist2 = renderer_wrist.render().copy()
                    img_top = renderer_top.render().copy()

                    obs_dict["images"] = {
                        "top": img_top,
                        "wrist1": img_wrist1,
                        "wrist2": img_wrist2,
                    }

                # 14-DOF Action Vector commanded by FSM
                act_arm1 = 2.0 * (data.ctrl[fsm.controller.arm1_joint_act_ids] - q_min1) / (q_max1 - q_min1) - 1.0
                act_grip1 = 2.0 * (fsm.target_grip1) - 1.0
                act_arm2 = 2.0 * (data.ctrl[fsm.controller.arm2_joint_act_ids] - q_min2) / (q_max2 - q_min2) - 1.0
                act_grip2 = 2.0 * (fsm.target_grip2) - 1.0
                action_14d = np.clip(
                    np.concatenate([act_arm1, [act_grip1], act_arm2, [act_grip2]]), -1.0, 1.0
                ).astype(np.float32)

                # Compute reward
                is_held1, _ = fsm.controller.check_grasp(1, fsm.target_name)
                is_held2, _ = fsm.controller.check_grasp(2, fsm.target_name)
                in_bin = env._check_in_bin()
                reward = env._compute_reward(is_held1, is_held2, in_bin)

                recorder.add_step(
                    obs=obs_dict,
                    action=action_14d,
                    reward=reward,
                    done=False,
                )

            curr_state, is_finished = fsm.step()
            step_physics += 1

            if curr_state != last_state:
                print(f"  Step {step_physics:5d} ({step_physics/500:.2f}s) | State -> {curr_state.name:15s}", flush=True)
                last_state = curr_state

            if is_finished:
                break

        # Physics settle (let object come to complete rest in bin)
        for _ in range(30):
            fsm.controller.set_arm_joint_targets(1, fsm.q_home1)
            fsm.controller.set_arm_joint_targets(2, fsm.q_home2)
            fsm.controller.set_gripper(1, 0.0)
            fsm.controller.set_gripper(2, 0.0)
            mujoco.mj_step(model, data)
            step_physics += 1

        # 4. Verify Episode Success
        final_box_pos = data.xpos[env.target_body_id].copy()
        dist_bin = float(np.linalg.norm(final_box_pos[:2] - np.array([0.0, -0.80])))
        is_success = bool(dist_bin < 0.15 and final_box_pos[2] < 0.20 and fsm.dual_hold_confirmed)
        ep_duration = time.time() - t_ep_start

        meta = {
            "initial_object_pos": initial_box_pos.tolist(),
            "final_object_pos": final_box_pos.tolist(),
            "distance_to_bin_m": dist_bin,
            "dual_hold_confirmed": fsm.dual_hold_confirmed,
            "real_duration_sec": ep_duration,
        }

        saved = recorder.save_episode(
            episode_idx=ep_idx,
            success=is_success,
            extra_metadata=meta,
        )

        if saved:
            successful_count += 1
            print(f"  Episode {successful_count}/{num_episodes} SUCCESS! (Dist to bin: {dist_bin*100:.1f} cm)", flush=True)
        else:
            print(f"  Episode attempt failed quality filter! (Dist to bin: {dist_bin*100:.1f} cm)", flush=True)

    # Export LeRobot metadata
    recorder.export_lerobot_metadata()

    if renderer_top is not None:
        renderer_top.close()
    if renderer_wrist is not None:
        renderer_wrist.close()
    env.close()

    total_time = time.time() - t_start_total
    print("\n" + "=" * 80)
    print("                    DATASET COLLECTION COMPLETE")
    print("=" * 80)
    print(f"  Total Valid Episodes Saved : {successful_count}/{num_episodes}")
    print(f"  Total Attempts Executed    : {attempt_count}")
    print(f"  Collection Success Rate    : {(successful_count/attempt_count)*100.0:.1f}%")
    print(f"  Total Wall Clock Time      : {total_time:.2f} s ({total_time/60.0:.1f} min)")
    print(f"  Output HDF5 File           : {recorder.dataset_path}")
    print(f"  LeRobot Metadata Dir       : {recorder.meta_dir}")
    print("=" * 80 + "\n")

    # Run inspection
    inspect_dataset(recorder.dataset_path)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: Automated Expert Demonstration Dataset Collector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of successful episodes to collect")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for HDF5 datasets")
    parser.add_argument("--dataset-name", type=str, default="handover_demos.hdf5", help="Consolidated HDF5 filename")
    parser.add_argument("--save-images", action="store_true", default=True, help="Record multi-camera RGB images")
    parser.add_argument("--no-images", dest="save_images", action="store_false", help="Record state-only transitions (fast)")
    parser.add_argument("--randomize", action="store_true", default=True, help="Randomize object pose on reset")
    parser.add_argument("--no-randomize", dest="randomize", action="store_false", help="Use nominal object pose")
    parser.add_argument("--compression", type=str, default="lzf", choices=["lzf", "gzip"], help="Image compression algorithm")
    parser.add_argument("--fps", type=int, default=50, help="Control sampling frequency")
    args = parser.parse_args()

    collect_demonstrations(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        save_images=args.save_images,
        randomize=args.randomize,
        compression=args.compression,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
