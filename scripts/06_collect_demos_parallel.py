"""
Phase 6: High-Performance Parallel Expert Demonstration Dataset Collector.

Leverages multi-core CPU architectures (e.g., Intel Core Ultra 7 8-core) to collect
demonstration episodes concurrently across isolated worker processes:
1. Distributes episode quotas across N parallel worker processes.
2. Each worker operates an independent MuJoCo physics engine and zero-allocation renderers.
3. Automatically applies domain randomization (cube position and yaw).
4. Enforces the perception-guided 16-stage HandoverFSM and observation/action logging.
5. Employs independent temporary HDF5 storage per worker to eliminate file locking contention.
6. Merges all worker datasets into the consolidated ALOHA HDF5 and exports LeRobot metadata.

Usage:
    python scripts/06_collect_demos_parallel.py --num-episodes 50 --num-workers 6 --fps 25 --randomize
"""

import os
import sys
import time
import argparse
import traceback
import multiprocessing as mp
from typing import Dict, Any, List, Optional
import numpy as np

# Ensure workspace root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import mujoco
import h5py
from scipy.spatial.transform import Rotation as R

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


def _worker_collection_task(
    worker_id: int,
    assigned_episodes: List[int],
    output_dir: str,
    fps: int,
    save_images: bool,
    randomize: bool,
    render_quality: str,
    compression: str,
    max_attempts_per_ep: int,
    seed: int,
    status_queue: mp.Queue,
):
    """
    Dedicated worker process routine for collecting assigned demonstration episodes.
    """
    try:
        worker_seed = seed + worker_id * 10007
        np.random.seed(worker_seed)

        scene_path = os.path.join(ROOT_DIR, "models", "dual_ur10e_scene.xml")
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)

        env = DualUR10eHandoverEnv(
            model_path=scene_path,
            obs_type="pixels" if save_images else "state",
            action_mode="absolute",
            reward_type="dense",
            randomize_on_reset=False,
        )

        worker_tmp_name = f".tmp_worker_{worker_id}.hdf5"
        recorder = EpisodeRecorder(
            output_dir=output_dir,
            dataset_name=worker_tmp_name,
            fps=fps,
            save_images=save_images,
            compression=compression,
        )

        renderer_top = None
        renderer_wrist = None
        cam_top_id = -1
        cam_wrist1_id = -1
        cam_wrist2_id = -1

        if save_images:
            if render_quality == "compact":
                h_top, w_top = 112, 112
                h_wrist, w_wrist = 112, 112
            elif render_quality == "fast":
                h_top, w_top = 240, 320
                h_wrist, w_wrist = 120, 160
            else:  # standard
                h_top, w_top = 480, 640
                h_wrist, w_wrist = 240, 320

            renderer_top = mujoco.Renderer(model, height=h_top, width=w_top)
            renderer_wrist = mujoco.Renderer(model, height=h_wrist, width=w_wrist)
            cam_top_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
            cam_wrist1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "arm1_wrist_cam")
            cam_wrist2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "arm2_wrist_cam")

        q_min1, q_max1 = env.q_min_arm1, env.q_max_arm1
        q_min2, q_max2 = env.q_min_arm2, env.q_max_arm2
        sample_stride = max(1, int(500 / fps))

        for ep_idx in assigned_episodes:
            collected = False
            attempt = 0

            while not collected and attempt < max_attempts_per_ep:
                attempt += 1

                # 1. Reset simulation
                if model.nkey > 0:
                    mujoco.mj_resetDataKeyframe(model, data, 0)
                else:
                    mujoco.mj_resetData(model, data)

                # Domain randomization
                if randomize and env.target_joint_id != -1:
                    qadr = model.jnt_qposadr[env.target_joint_id]
                    dx = np.random.uniform(-0.03, 0.03)
                    dy = np.random.uniform(-0.02, 0.02)
                    data.qpos[qadr + 0] = 0.45 + dx
                    data.qpos[qadr + 1] = 1.05 + dy
                    yaw_deg = np.random.uniform(-25.0, 25.0)
                    quat_rand = R.from_euler("z", yaw_deg, degrees=True).as_quat(scalar_first=True)
                    data.qpos[qadr + 3:qadr + 7] = quat_rand

                # Settle physics
                for _ in range(50):
                    mujoco.mj_step(model, data)

                initial_box_pos = data.xpos[env.target_body_id].copy()

                # 2. Perception & Grasp Planning
                fsm = HandoverFSM(model, data)
                pcd = capture_scene_pointcloud(model, data)
                segmenter = SceneSegmenter()
                clusters = segmenter.segment_objects(pcd)
                target = segmenter.find_target(clusters, color_hint="red")

                if target is None:
                    status_queue.put({
                        "type": "attempt_retry",
                        "worker_id": worker_id,
                        "ep_idx": ep_idx,
                        "attempt": attempt,
                        "reason": "Target segmentation failed",
                    })
                    continue

                planner = GeometricGraspPlanner()
                raw_cands = planner.plan_grasps(target.pcd)
                gfilter = GraspFilter(model, data, arm_id=1, standoff_dist=0.15)
                verified_grasps = gfilter.filter_candidates(raw_cands)

                if not verified_grasps:
                    status_queue.put({
                        "type": "attempt_retry",
                        "worker_id": worker_id,
                        "ep_idx": ep_idx,
                        "attempt": attempt,
                        "reason": "No valid grasp candidate found",
                    })
                    continue

                best_grasp = verified_grasps[0]
                ok_plan = fsm.plan_trajectories(best_grasp)
                if not ok_plan:
                    status_queue.put({
                        "type": "attempt_retry",
                        "worker_id": worker_id,
                        "ep_idx": ep_idx,
                        "attempt": attempt,
                        "reason": "Trajectory planning failed",
                    })
                    continue

                # 3. Execution & Recording
                fsm.transition_to(HandoverState.APPROACH_1, duration=2.5)
                step_physics = 0
                last_state = fsm.current_state
                t_ep_start = time.time()

                status_queue.put({
                    "type": "stage_update",
                    "worker_id": worker_id,
                    "ep_idx": ep_idx,
                    "state": fsm.current_state.name,
                    "sim_time": 0.0,
                })

                while True:
                    if step_physics % sample_stride == 0:
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

                        act_arm1 = 2.0 * (data.ctrl[fsm.controller.arm1_joint_act_ids] - q_min1) / (q_max1 - q_min1) - 1.0
                        act_grip1 = 2.0 * (fsm.target_grip1) - 1.0
                        act_arm2 = 2.0 * (data.ctrl[fsm.controller.arm2_joint_act_ids] - q_min2) / (q_max2 - q_min2) - 1.0
                        act_grip2 = 2.0 * (fsm.target_grip2) - 1.0
                        action_14d = np.clip(
                            np.concatenate([act_arm1, [act_grip1], act_arm2, [act_grip2]]), -1.0, 1.0
                        ).astype(np.float32)

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
                        last_state = curr_state
                        status_queue.put({
                            "type": "stage_update",
                            "worker_id": worker_id,
                            "ep_idx": ep_idx,
                            "state": curr_state.name,
                            "sim_time": fsm.sim_time,
                        })

                    if is_finished:
                        break

                # Settle in bin
                for _ in range(30):
                    fsm.controller.set_arm_joint_targets(1, fsm.q_home1)
                    fsm.controller.set_arm_joint_targets(2, fsm.q_home2)
                    fsm.controller.set_gripper(1, 0.0)
                    fsm.controller.set_gripper(2, 0.0)
                    mujoco.mj_step(model, data)

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
                    "worker_id": worker_id,
                }

                saved = recorder.save_episode(
                    episode_idx=ep_idx,
                    success=is_success,
                    extra_metadata=meta,
                )

                if saved:
                    collected = True
                    status_queue.put({
                        "type": "episode_success",
                        "worker_id": worker_id,
                        "ep_idx": ep_idx,
                        "steps": len(recorder._actions) if not saved else meta["real_duration_sec"],
                        "duration": ep_duration,
                        "dist_bin": dist_bin,
                    })
                else:
                    status_queue.put({
                        "type": "episode_failed",
                        "worker_id": worker_id,
                        "ep_idx": ep_idx,
                        "attempt": attempt,
                        "dist_bin": dist_bin,
                    })

        if renderer_top is not None:
            renderer_top.close()
        if renderer_wrist is not None:
            renderer_wrist.close()
        env.close()

        status_queue.put({
            "type": "worker_finished",
            "worker_id": worker_id,
            "saved_episodes": recorder.saved_episodes,
            "tmp_dataset": os.path.join(output_dir, worker_tmp_name),
        })

    except Exception as e:
        status_queue.put({
            "type": "worker_error",
            "worker_id": worker_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })


def collect_demonstrations_parallel(
    num_episodes: int = 50,
    num_workers: int = 6,
    output_dir: str = "data",
    dataset_name: str = "handover_demos.hdf5",
    save_images: bool = True,
    randomize: bool = True,
    compression: str = "lzf",
    fps: int = 25,
    render_quality: str = "compact",
    overwrite: bool = False,
    seed: int = 42,
):
    """
    Main orchestration routine for multi-worker parallel demonstration collection.
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    master_hdf5_path = os.path.join(output_dir, dataset_name)

    start_ep_idx = 0
    if os.path.exists(master_hdf5_path) and not overwrite:
        with h5py.File(master_hdf5_path, "r") as f:
            if "data" in f:
                start_ep_idx = len(f["data"].keys())

    total_target_indices = list(range(start_ep_idx, start_ep_idx + num_episodes))
    num_workers = min(num_workers, num_episodes)

    worker_partitions: List[List[int]] = [[] for _ in range(num_workers)]
    for i, ep_idx in enumerate(total_target_indices):
        worker_partitions[i % num_workers].append(ep_idx)

    print("=" * 80)
    print("      PHASE 6: HIGH-PERFORMANCE PARALLEL DEMONSTRATION COLLECTOR")
    print("=" * 80)
    print(f"  Target Episodes      : {num_episodes} (Global demo_{start_ep_idx} to demo_{start_ep_idx + num_episodes - 1})")
    print(f"  Active CPU Workers   : {num_workers} parallel processes")
    print(f"  Episodes per Worker  : {[len(p) for p in worker_partitions]}")
    print(f"  Output Directory     : {output_dir}")
    print(f"  Consolidated HDF5    : {master_hdf5_path}")
    print(f"  Save Images (RGB)    : {save_images}")
    print(f"  Render Quality       : {render_quality}")
    print(f"  Compression          : {compression}")
    print(f"  Control Rate         : {fps} Hz (500 Hz internal physics)")
    print(f"  Domain Randomization : {randomize}")
    print("=" * 80 + "\n", flush=True)

    status_queue = mp.Queue()
    workers = []
    t_start = time.time()

    for w_id in range(num_workers):
        p = mp.Process(
            target=_worker_collection_task,
            args=(
                w_id,
                worker_partitions[w_id],
                output_dir,
                fps,
                save_images,
                randomize,
                render_quality,
                compression,
                10,
                seed,
                status_queue,
            ),
        )
        p.start()
        workers.append(p)

    completed_episodes = 0
    active_workers = num_workers
    worker_results: Dict[int, Dict[str, Any]] = {}

    while active_workers > 0:
        msg = status_queue.get()
        m_type = msg.get("type")

        if m_type == "stage_update":
            w_id = msg["worker_id"]
            ep = msg["ep_idx"]
            st = msg["state"]
            sim_t = msg.get("sim_time", 0.0)
            print(f"  [Worker {w_id} | demo_{ep:02d}] {sim_t:4.1f}s -> {st:16s}", flush=True)

        elif m_type == "attempt_retry":
            print(
                f"  [RETRY W{msg['worker_id']}] demo_{msg['ep_idx']:02d} attempt {msg['attempt']}: {msg['reason']}. Retrying...",
                flush=True,
            )

        elif m_type == "episode_success":
            completed_episodes += 1
            elapsed = time.time() - t_start
            rate = completed_episodes / elapsed if elapsed > 0 else 0.0
            eta_sec = (num_episodes - completed_episodes) / rate if rate > 0 else 0.0
            print(
                f"\n  >>> [{completed_episodes:02d}/{num_episodes}] Worker {msg['worker_id']} -> "
                f"demo_{msg['ep_idx']:03d} SUCCESS (dist={msg['dist_bin']*100:.1f}cm, "
                f"{msg['duration']:.1f}s) | Progress: {completed_episodes/num_episodes*100.0:.1f}% | "
                f"Elapsed: {elapsed/60.0:.1f}m | ETA: {eta_sec/60.0:.1f}m <<<\n",
                flush=True,
            )

        elif m_type == "episode_failed":
            print(
                f"  [FAILED QUALITY W{msg['worker_id']}] demo_{msg['ep_idx']:03d} attempt {msg['attempt']} "
                f"(dist={msg['dist_bin']*100:.1f}cm). Retrying...",
                flush=True,
            )

        elif m_type == "worker_finished":
            active_workers -= 1
            w_id = msg["worker_id"]
            worker_results[w_id] = msg
            print(f"  >>> Worker {w_id} has completed all assigned episodes! ({active_workers} workers remaining)", flush=True)

        elif m_type == "worker_error":
            w_id = msg["worker_id"]
            print(f"  [FATAL WORKER {w_id} ERROR]: {msg['error']}\n{msg['traceback']}", flush=True)
            active_workers -= 1

    for p in workers:
        p.join()

    # 4. Merge Temporary Datasets into Consolidated HDF5
    print("\n" + "-" * 80)
    print("  Merging worker datasets into master HDF5...")
    print("-" * 80, flush=True)

    mode = "w" if overwrite or not os.path.exists(master_hdf5_path) else "a"
    all_saved_summaries = []

    with h5py.File(master_hdf5_path, mode) as master_f:
        if "data" not in master_f:
            master_f.create_group("data")

        for w_id in range(num_workers):
            res = worker_results.get(w_id)
            if not res:
                continue

            tmp_file = res["tmp_dataset"]
            all_saved_summaries.extend(res.get("saved_episodes", []))

            if os.path.exists(tmp_file):
                with h5py.File(tmp_file, "r") as worker_f:
                    if "data" in worker_f:
                        for demo_key in worker_f["data"].keys():
                            dest_key = f"data/{demo_key}"
                            if dest_key in master_f:
                                del master_f[dest_key]
                            master_f.copy(worker_f[f"data/{demo_key}"], dest_key)

                try:
                    os.remove(tmp_file)
                except OSError:
                    pass

        total_demos = len(master_f["data"].keys())
        master_f.attrs["num_episodes"] = total_demos
        master_f.attrs["fps"] = fps

    print(f"  Master HDF5 consolidated: {total_demos} total episodes in {master_hdf5_path}.")

    # 5. Export LeRobot Metadata
    recorder_meta = EpisodeRecorder(
        output_dir=output_dir,
        dataset_name=dataset_name,
        fps=fps,
        save_images=save_images,
        compression=compression,
    )
    recorder_meta.saved_episodes = all_saved_summaries
    recorder_meta.export_lerobot_metadata()

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print("             PARALLEL DATASET COLLECTION COMPLETE")
    print("=" * 80)
    print(f"  Total Valid Episodes Saved : {completed_episodes}/{num_episodes}")
    print(f"  Total Wall Clock Time      : {total_time:.2f} s ({total_time/60.0:.2f} min)")
    print(f"  Collection Throughput      : {completed_episodes / (total_time/60.0):.2f} episodes/min")
    print(f"  Output HDF5 File           : {master_hdf5_path}")
    print(f"  LeRobot Metadata Dir       : {recorder_meta.meta_dir}")
    print("=" * 80 + "\n", flush=True)

    inspect_dataset(master_hdf5_path)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: High-Performance Parallel Demonstration Dataset Collector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-episodes", type=int, default=50, help="Number of successful episodes to collect")
    parser.add_argument("--num-workers", type=int, default=6, help="Number of parallel worker processes")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for HDF5 datasets")
    parser.add_argument("--dataset-name", type=str, default="handover_demos.hdf5", help="Consolidated HDF5 filename")
    parser.add_argument("--save-images", action="store_true", default=True, help="Record multi-camera RGB images")
    parser.add_argument("--no-images", dest="save_images", action="store_false", help="Record state-only transitions")
    parser.add_argument("--randomize", action="store_true", default=True, help="Randomize object pose on reset")
    parser.add_argument("--no-randomize", dest="randomize", action="store_false", help="Use nominal object pose")
    parser.add_argument("--compression", type=str, default="lzf", choices=["lzf", "gzip"], help="Image compression algorithm")
    parser.add_argument("--fps", type=int, default=25, help="Control sampling frequency (25 Hz recommended for fast collection)")
    parser.add_argument("--render-quality", type=str, default="compact", choices=["compact", "fast", "standard"], help="Image render resolution ('compact'=112x112, 'fast'=240x320/120x160, 'standard'=480x640/240x320)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset rather than appending")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for workers")
    args = parser.parse_args()

    collect_demonstrations_parallel(
        num_episodes=args.num_episodes,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        save_images=args.save_images,
        randomize=args.randomize,
        compression=args.compression,
        fps=args.fps,
        render_quality=args.render_quality,
        overwrite=args.overwrite,
        seed=args.seed,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
