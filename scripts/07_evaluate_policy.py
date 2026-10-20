"""
Phase 8: Closed-Loop Policy Evaluation and Telemetry Suite.
Evaluates the trained ACT policy in simulation with temporal ensembling,
interactive 3D MuJoCo visualization, composite video recording, and benchmark reporting.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torchvision.transforms.functional as TF

# Ensure project root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import mujoco
try:
    import mujoco.viewer
except Exception:
    pass

from envs.dual_ur10e_env import DualUR10eHandoverEnv
from policies.act_model import ACTPolicy


class TemporalEnsembler:
    """
    Exponential Temporal Ensembling for Action Chunking with Transformers (ACT).
    Averages overlapping action chunk predictions across rolling temporal windows.
    Based on Tony Zhao et al. (Stanford / Google, 2023).
    """

    def __init__(self, chunk_size: int = 25, action_dim: int = 14, decay: float = 0.01):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.decay = decay
        self.predictions: Dict[int, List[Tuple[np.ndarray, float]]] = {}

    def reset(self):
        self.predictions.clear()

    def add_chunk(self, chunk: np.ndarray, current_t: int):
        """Registers a predicted action chunk of shape (K, 14) starting at current_t."""
        for i in range(self.chunk_size):
            t_target = current_t + i
            if t_target not in self.predictions:
                self.predictions[t_target] = []
            weight = np.exp(-self.decay * i)
            self.predictions[t_target].append((chunk[i], weight))

    def get_action(self, current_t: int, fallback_chunk: Optional[np.ndarray] = None) -> np.ndarray:
        """Returns the exponentially weighted blended action for timestep current_t."""
        if current_t in self.predictions and len(self.predictions[current_t]) > 0:
            acts, weights = zip(*self.predictions[current_t])
            acts = np.array(acts, dtype=np.float32)
            weights = np.array(weights, dtype=np.float32)
            weights = weights / np.sum(weights)
            blended = np.sum(acts * weights[:, None], axis=0)
            del self.predictions[current_t]
            return blended
        elif fallback_chunk is not None:
            return fallback_chunk[0]
        else:
            return np.zeros(self.action_dim, dtype=np.float32)


def load_normalizers(stats_path: str):
    """Loads action and qpos normalization parameters from stats.json."""
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        action_mean = np.array(stats["action"]["mean"], dtype=np.float32)
        action_std = np.array(stats["action"]["std"], dtype=np.float32)
        qpos_mean = np.array(stats["observation.state"]["mean"], dtype=np.float32)
        qpos_std = np.array(stats["observation.state"]["std"], dtype=np.float32)
    else:
        action_mean = np.zeros(14, dtype=np.float32)
        action_std = np.ones(14, dtype=np.float32)
        qpos_mean = np.zeros(14, dtype=np.float32)
        qpos_std = np.ones(14, dtype=np.float32)

    action_std = np.where(action_std < 1e-6, 1.0, action_std)
    qpos_std = np.where(qpos_std < 1e-6, 1.0, qpos_std)

    def normalize_qpos(qpos: np.ndarray) -> np.ndarray:
        return (qpos - qpos_mean) / qpos_std

    def unnormalize_action(norm_action: np.ndarray) -> np.ndarray:
        return norm_action * action_std + action_mean

    return normalize_qpos, unnormalize_action


def preprocess_images(imgs_dict: Dict[str, np.ndarray], target_size: Tuple[int, int] = (112, 112)) -> Dict[str, torch.Tensor]:
    """Preprocesses raw environment RGB frames (uint8) into ImageNet-normalized PyTorch tensors."""
    tensors = {}
    for cam_name, img in imgs_dict.items():
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        t = TF.resize(t, list(target_size), antialias=True)
        t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tensors[cam_name] = t.unsqueeze(0)  # (1, 3, H, W)
    return tensors


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained ACT Policy in Closed-Loop Simulation")
    parser.add_argument("--checkpoint", type=str, default="outputs/act_handover/best_model.pt", help="Path to .pt model checkpoint")
    parser.add_argument("--viewer", action="store_true", help="Launch interactive 3D passive MuJoCo viewer")
    parser.add_argument("--speed", type=float, default=1.0, help="Viewer simulation playback speed multiplier")
    parser.add_argument("--record", action="store_true", help="Record composite multi-view MP4 video preview")
    parser.add_argument("--num-episodes", type=int, default=None, help="Number of evaluation trials")
    parser.add_argument("--max-steps", type=int, default=1400, help="Maximum control steps per trial (50 Hz)")
    parser.add_argument("--randomize", action="store_true", help="Apply initial object pose domain randomization")
    parser.add_argument("--temporal-ensemble", action="store_true", default=True, help="Enable exponential temporal ensembling")
    parser.add_argument("--no-ensemble", action="store_true", help="Disable temporal ensembling (query every K steps)")
    parser.add_argument("--infer-stride", type=int, default=1, help="Query stride in steps (default 1 for 50Hz GPU ensembling; set 5-10 for fast CPU playback)")
    parser.add_argument("--ensemble-decay", type=float, default=0.20, help="Temporal ensembling exponential decay m (default 0.20)")
    parser.add_argument("--replan-freq", type=int, default=10, help="Receding horizon replan frequency when not ensembling (default 10)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cpu', 'cuda', 'auto')")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_ensemble:
        args.temporal_ensemble = False

    num_episodes = args.num_episodes or (1 if args.viewer else 5)

    # Robust path resolution for checkpoint
    if not os.path.isabs(args.checkpoint):
        candidate = os.path.join(ROOT_DIR, args.checkpoint)
        if os.path.exists(candidate):
            args.checkpoint = candidate

    print("=" * 80)
    print("      PHASE 8: CLOSED-LOOP VISUOMOTOR POLICY EVALUATION (ACT)")
    print("=" * 80)
    print(f"  Checkpoint        : {args.checkpoint}")
    print(f"  Interactive Viewer: {args.viewer} (speed={args.speed}x)")
    print(f"  Record Video      : {args.record}")
    print(f"  Evaluation Trials : {num_episodes}")
    print(f"  Max Steps / Trial : {args.max_steps} ({args.max_steps/50:.1f}s)")
    print(f"  Domain Randomize  : {args.randomize}")
    print(f"  Temporal Ensemble : {args.temporal_ensemble} (m={args.ensemble_decay}, stride={args.infer_stride})")
    print("=" * 80)

    if not os.path.exists(args.checkpoint):
        print(f"[FATAL] Checkpoint not found at: {args.checkpoint}")
        print("Please ensure training has completed or specify a valid checkpoint with --checkpoint.")
        sys.exit(1)

    # 1. Device configuration
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using Device      : {device}")

    # 2. Load Checkpoint and Rebuild ACT Model
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})

    camera_names = dataset_cfg.get("camera_names", ["top", "wrist1", "wrist2"])
    chunk_size = model_cfg.get("chunk_size", 25)
    image_size = tuple(dataset_cfg.get("input_image_size", [112, 112]))
    stats_path = dataset_cfg.get("stats_path", "data/meta/stats.json")
    if not os.path.isabs(stats_path):
        candidate_stats = os.path.join(ROOT_DIR, stats_path)
        if os.path.exists(candidate_stats):
            stats_path = candidate_stats

    model = ACTPolicy(
        camera_names=camera_names,
        chunk_size=chunk_size,
        state_dim=model_cfg.get("state_dim", 14),
        action_dim=model_cfg.get("action_dim", 14),
        d_model=model_cfg.get("d_model", 256),
        nhead=model_cfg.get("nheads", 8),
        enc_layers=model_cfg.get("enc_layers", 3),
        dec_layers=model_cfg.get("dec_layers", 3),
        dim_feedforward=model_cfg.get("dim_feedforward", 1024),
        latent_dim=model_cfg.get("latent_dim", 32),
        dropout=0.0,
        freeze_backbone=model_cfg.get("freeze_backbone", True),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Model loaded from Epoch {ckpt.get('epoch', '?')} (Val Loss: {ckpt.get('val_loss', 0.0):.4f})")

    # 3. Normalization helpers
    norm_qpos_fn, unnorm_action_fn = load_normalizers(stats_path)

    # 4. Initialize Environment
    env = DualUR10eHandoverEnv(
        obs_type="pixels",
        action_mode="absolute",
        randomize_on_reset=args.randomize,
        max_episode_steps=args.max_steps,
    )

    ensembler = TemporalEnsembler(chunk_size=chunk_size, action_dim=14, decay=args.ensemble_decay)

    # Optional video recording setup
    video_writer = None
    if args.record:
        import imageio
        rec_dir = os.path.join(ROOT_DIR, "recordings")
        os.makedirs(rec_dir, exist_ok=True)
        vid_path = os.path.join(rec_dir, "phase8_act_eval_rollout.mp4")
        video_writer = imageio.get_writer(vid_path, fps=50, codec="libx264", quality=8)
        print(f"  Recording rollout to: {vid_path}")

    # Track trial statistics
    trial_results = []

    def run_eval_loop(viewer_handle=None):
        nonlocal video_writer
        for trial_i in range(1, num_episodes + 1):
            print(f"\n>>> [EVALUATION TRIAL {trial_i}/{num_episodes}] <<<", flush=True)
            obs, info = env.reset()
            ensembler.reset()

            box_initial = env.data.xpos[env.target_body_id].copy()
            print(f"  Target initial pose: {np.round(box_initial, 3)}")

            arm1_grasped = False
            arm2_grasped = False
            in_bin_final = False
            inter_coll = False
            shelf_coll = False

            step = 0
            cached_chunk = None

            t_trial_start = time.time()
            ctrl_dt = 1.0 / 50.0

            while step < args.max_steps:
                t_step_start = time.time()

                # 1. Observations
                raw_qpos = obs["qpos"]
                norm_qpos = norm_qpos_fn(raw_qpos)
                qpos_t = torch.from_numpy(norm_qpos).float().unsqueeze(0).to(device)

                # Preprocess camera images
                images_t = preprocess_images(obs["images"], target_size=image_size)
                images_t = {k: v.to(device) for k, v in images_t.items()}

                # 2. Query Policy (every step / infer_stride for ensembling, or every replan_freq steps)
                should_query = (step % max(1, args.infer_stride) == 0) if args.temporal_ensemble else (step % args.replan_freq == 0)
                if should_query:
                    with torch.no_grad():
                        pred_chunk_t = model.predict_action_chunk(images_t, qpos_t)  # (1, K, 14)
                        norm_chunk = pred_chunk_t.squeeze(0).cpu().numpy()          # (K, 14)
                        cached_chunk = unnorm_action_fn(norm_chunk)                  # Unnormalized [-1, 1]

                    if args.temporal_ensemble:
                        ensembler.add_chunk(cached_chunk, step)

                # 3. Select action to apply
                if args.temporal_ensemble:
                    action_to_apply = ensembler.get_action(step, fallback_chunk=cached_chunk)
                    # Discrete gripper thresholding (ACT ALOHA standard):
                    # Temporal ensembling is ideal for continuous 6-DOF arm joints (0..5 and 7..12).
                    # For discrete grippers (6 and 13), use the latest chunk's immediate prediction
                    # with a hysteresis threshold (> 0.0) to prevent premature mid-air closing during approach.
                    if cached_chunk is not None:
                        action_to_apply[6] = 0.5 if (cached_chunk[0, 6] > 0.0 or action_to_apply[6] > 0.1) else -1.0
                        action_to_apply[13] = 1.0 if (cached_chunk[0, 13] > 0.0 or action_to_apply[13] > 0.1) else -1.0
                else:
                    action_idx = min(step % args.replan_freq, chunk_size - 1)
                    action_to_apply = cached_chunk[action_idx].copy()
                    action_to_apply[6] = 0.5 if action_to_apply[6] > 0.0 else -1.0
                    action_to_apply[13] = 1.0 if action_to_apply[13] > 0.0 else -1.0

                action_to_apply = np.clip(action_to_apply, -1.0, 1.0)

                # 4. Environment Step (50 Hz control rate)
                next_obs, reward, terminated, truncated, info = env.step(action_to_apply)

                # Check contacts & milestones
                is_held1, f1 = env.controller.check_grasp(1, env.target_name)
                is_held2, f2 = env.controller.check_grasp(2, env.target_name)
                box_pos = env.data.xpos[env.target_body_id]

                if is_held1 and f1 >= 10.0:
                    arm1_grasped = True
                if is_held2 and f2 >= 10.0:
                    arm2_grasped = True
                if info.get("shelf_collision", False):
                    shelf_coll = True
                if info.get("inter_arm_collision", False):
                    inter_coll = True

                ee1_pos, _ = env.controller.get_ee_pose(1)
                dist1 = float(np.linalg.norm(ee1_pos - box_pos)) * 100.0
                if step % 25 == 0:
                    print(f"  Step {step:4d}/{args.max_steps} | EE1: [{ee1_pos[0]:.2f}, {ee1_pos[1]:.2f}, {ee1_pos[2]:.2f}] | Dist: {dist1:4.1f}cm | Grip1: {action_to_apply[6]:.1f} (f1={f1:3.0f}N, held1={is_held1}) | Grip2: {action_to_apply[13]:.1f} (held2={is_held2})", flush=True)

                # Record composite video frame if requested
                if video_writer is not None:
                    from PIL import Image
                    img_top = obs["images"]["top"]
                    img_w1 = obs["images"]["wrist1"]
                    img_w2 = obs["images"]["wrist2"]
                    h_tgt = 240
                    w_tgt = 320
                    t_sm = np.array(Image.fromarray(img_top).resize((w_tgt, h_tgt), Image.BILINEAR))
                    w1_sm = np.array(Image.fromarray(img_w1).resize((w_tgt, h_tgt), Image.BILINEAR))
                    w2_sm = np.array(Image.fromarray(img_w2).resize((w_tgt, h_tgt), Image.BILINEAR))
                    comp_row = np.concatenate([t_sm, w1_sm, w2_sm], axis=1)
                    video_writer.append_data(comp_row)

                # Update Viewer if active
                if viewer_handle is not None:
                    if not viewer_handle.is_running():
                        print(f"\n[VIEWER] Interactive window closed by user at step {step}. Ending trial...")
                        break
                    viewer_handle.sync()
                    elapsed = time.time() - t_step_start
                    delay = max(0.0, (ctrl_dt / max(0.1, args.speed)) - elapsed)
                    if delay > 0:
                        time.sleep(delay)

                obs = next_obs
                step += 1

                # Check if placed in bin
                if env._check_in_bin() and not is_held1 and not is_held2:
                    in_bin_final = True
                    # Let object settle in bin for 50 steps
                    for _ in range(50):
                        env.step(action_to_apply)
                        if viewer_handle is not None:
                            if not viewer_handle.is_running():
                                break
                            viewer_handle.sync()
                    break

            # Evaluate final success metrics
            dist_to_bin = float(np.linalg.norm(box_pos - env.bin_pos)) * 100.0  # in cm
            in_bin_final = env._check_in_bin()
            trial_time = time.time() - t_trial_start

            trial_summary = {
                "trial": trial_i,
                "arm1_picked": arm1_grasped,
                "arm2_received": arm2_grasped,
                "placed_in_bin": in_bin_final,
                "dist_to_bin_cm": round(dist_to_bin, 1),
                "steps": step,
                "duration_s": round(trial_time, 2),
                "inter_collision": inter_coll,
                "shelf_collision": shelf_coll,
            }
            trial_results.append(trial_summary)

            print(f"  Result: {'SUCCESS' if in_bin_final else 'COMPLETED (INSPECTION)'} | "
                  f"Picked: {arm1_grasped} | Handover: {arm2_grasped} | In Bin: {in_bin_final} "
                  f"(Dist: {dist_to_bin:.1f} cm) | Steps: {step} ({step/50:.1f}s)")

    # Execute with viewer or headless
    if args.viewer:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            run_eval_loop(viewer)
            if viewer.is_running():
                print("\n[VIEWER] Simulation evaluation finished. Keeping viewer open (close window to exit)...")
                try:
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.02)
                except KeyboardInterrupt:
                    pass
    else:
        run_eval_loop(None)

    if video_writer is not None:
        video_writer.close()
        print(f"\n[VIDEO] Successfully generated rollout video.")

    # Print Aggregate Scorecard
    n = len(trial_results)
    if n > 0:
        picks = sum(1 for r in trial_results if r["arm1_picked"])
        handovers = sum(1 for r in trial_results if r["arm2_received"])
        places = sum(1 for r in trial_results if r["placed_in_bin"])

        print("\n" + "=" * 80)
        print("                  PHASE 8 CLOSED-LOOP BENCHMARK SCORECARD")
        print("=" * 80)
        print(f"  Trials Evaluated        : {n}")
        print(f"  Arm 1 Grasp Success Rate: {(picks/n)*100.0:.1f}% ({picks}/{n})")
        print(f"  Handover Reception Rate : {(handovers/n)*100.0:.1f}% ({handovers}/{n})")
        print(f"  Bin Placement Success   : {(places/n)*100.0:.1f}% ({places}/{n})")
        print(f"  Mean Duration           : {np.mean([r['duration_s'] for r in trial_results]):.2f} s")
        print(f"  Mean Dist to Bin Center : {np.mean([r['dist_to_bin_cm'] for r in trial_results]):.1f} cm")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
