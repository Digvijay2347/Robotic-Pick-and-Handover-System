"""
Demonstration Episode Recorder for Dual-Arm Manipulation (HDF5 & LeRobot Format).

Provides:
- In-memory episode step buffering for 14-DOF actions and multi-camera observations.
- Standard ALOHA HDF5 dataset logging (Zhao et al., ACT specification).
- Hugging Face LeRobot dataset compatibility (metadata, stats, and episode indices).
- Automatic quality gate filtering (discarding failed trials).
- Dataset-wide normalization statistics calculation (mean, std, min, max).
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import h5py


class EpisodeRecorder:
    """
    Records, buffers, and persists bimanual demonstration episodes to HDF5 / LeRobot formats.
    """

    def __init__(
        self,
        output_dir: str,
        dataset_name: str = "handover_demos.hdf5",
        fps: int = 50,
        save_images: bool = True,
        compression: str = "gzip",
        compression_opts: int = 4,
    ):
        """
        Args:
            output_dir: Directory where datasets and metadata will be saved.
            dataset_name: Name of the consolidated HDF5 file.
            fps: Control frequency of the policy/recording (default 50 Hz).
            save_images: Whether to store rendered camera frames (top, wrist1, wrist2).
            compression: HDF5 compression algorithm ('gzip', 'lzf', or None).
            compression_opts: Compression level for gzip (1-9).
        """
        self.output_dir = os.path.abspath(output_dir)
        self.dataset_path = os.path.join(self.output_dir, dataset_name)
        self.episodes_dir = os.path.join(self.output_dir, "episodes")
        self.meta_dir = os.path.join(self.output_dir, "meta")
        self.fps = fps
        self.save_images = save_images
        self.compression = compression
        self.compression_opts = compression_opts

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.episodes_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)

        # In-memory buffer for current active episode
        self._reset_buffer()

        # Track saved episodes across session
        self.saved_episodes: List[Dict[str, Any]] = []

    def _reset_buffer(self):
        """Resets the temporary episode buffer."""
        self._qpos: List[np.ndarray] = []
        self._qvel: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []
        self._rewards: List[float] = []
        self._terminals: List[bool] = []
        self._images_top: List[np.ndarray] = []
        self._images_wrist1: List[np.ndarray] = []
        self._images_wrist2: List[np.ndarray] = []
        self._metadata: Dict[str, Any] = {}

    def add_step(
        self,
        obs: Dict[str, Any],
        action: np.ndarray,
        reward: float = 0.0,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ):
        """
        Buffers one timestep of data from the environment.

        Args:
            obs: Observation dict containing 'qpos' (14,), 'qvel' (14,), and optional 'images'.
            action: Applied 14-DOF normalized action command [-1.0, 1.0]^14.
            reward: Scalar step reward.
            done: Termination/truncation flag.
            info: Optional auxiliary telemetry dictionary.
        """
        self._qpos.append(np.array(obs["qpos"], dtype=np.float64))
        self._qvel.append(np.array(obs["qvel"], dtype=np.float64))
        self._actions.append(np.array(action, dtype=np.float32))
        self._rewards.append(float(reward))
        self._terminals.append(bool(done))

        if self.save_images and "images" in obs:
            imgs = obs["images"]
            if "top" in imgs:
                self._images_top.append(np.array(imgs["top"], dtype=np.uint8))
            if "wrist1" in imgs:
                self._images_wrist1.append(np.array(imgs["wrist1"], dtype=np.uint8))
            if "wrist2" in imgs:
                self._images_wrist2.append(np.array(imgs["wrist2"], dtype=np.uint8))

    def save_episode(
        self,
        episode_idx: int,
        success: bool,
        task_description: str = "Pick red cube from shelf, handover to Arm 2, and place in storage bin",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Persists the buffered episode to disk if it meets quality criteria.

        Args:
            episode_idx: Integer identifier for the episode.
            success: Whether the episode met the success criterion.
            task_description: Natural language task annotation.
            extra_metadata: Optional dictionary of domain randomization or physics parameters.

        Returns:
            True if episode was successfully saved, False if discarded.
        """
        num_steps = len(self._actions)
        if num_steps == 0:
            print(f"  [WARN] Attempted to save empty episode {episode_idx}!")
            return False

        if not success:
            print(f"  [QUALITY FILTER] Discarding failed episode {episode_idx} (steps={num_steps}).")
            self._reset_buffer()
            return False

        # Convert lists to NumPy arrays
        qpos_arr = np.stack(self._qpos, axis=0)
        qvel_arr = np.stack(self._qvel, axis=0)
        action_arr = np.stack(self._actions, axis=0)
        reward_arr = np.array(self._rewards, dtype=np.float32)
        term_arr = np.array(self._terminals, dtype=bool)

        has_images = self.save_images and len(self._images_top) == num_steps
        if has_images:
            top_arr = np.stack(self._images_top, axis=0)
            w1_arr = np.stack(self._images_wrist1, axis=0)
            w2_arr = np.stack(self._images_wrist2, axis=0)

        # 1. Save to Individual Episode File (LeRobot Standard)
        ep_file = os.path.join(self.episodes_dir, f"episode_{episode_idx:06d}.hdf5")
        with h5py.File(ep_file, "w") as f:
            f.attrs["sim"] = True
            f.attrs["fps"] = self.fps
            f.attrs["num_samples"] = num_steps
            f.attrs["success"] = success
            f.attrs["episode_index"] = episode_idx
            f.attrs["task"] = task_description

            if extra_metadata:
                for k, v in extra_metadata.items():
                    if isinstance(v, (int, float, str, bool)):
                        f.attrs[k] = v
                    elif isinstance(v, np.ndarray):
                        f.attrs[k] = v.tolist()

            obs_grp = f.create_group("observations")
            obs_grp.create_dataset("qpos", data=qpos_arr)
            obs_grp.create_dataset("qvel", data=qvel_arr)

            comp_kwargs = {}
            if self.compression:
                comp_kwargs["compression"] = self.compression
                if self.compression == "gzip" and self.compression_opts is not None:
                    comp_kwargs["compression_opts"] = self.compression_opts

            if has_images:
                img_grp = obs_grp.create_group("images")
                img_grp.create_dataset("top", data=top_arr, **comp_kwargs)
                img_grp.create_dataset("wrist1", data=w1_arr, **comp_kwargs)
                img_grp.create_dataset("wrist2", data=w2_arr, **comp_kwargs)

            f.create_dataset("action", data=action_arr)
            f.create_dataset("rewards", data=reward_arr)
            f.create_dataset("terminals", data=term_arr)

        # 2. Append to Consolidated HDF5 Dataset (ACT Gold Standard)
        mode = "a" if os.path.exists(self.dataset_path) else "w"
        with h5py.File(self.dataset_path, mode) as f:
            group_name = f"data/demo_{episode_idx}"
            if group_name in f:
                del f[group_name]

            demo_grp = f.create_group(group_name)
            demo_grp.attrs["num_samples"] = num_steps
            demo_grp.attrs["success"] = success
            demo_grp.attrs["task"] = task_description

            obs_grp = demo_grp.create_group("observations")
            obs_grp.create_dataset("qpos", data=qpos_arr)
            obs_grp.create_dataset("qvel", data=qvel_arr)

            if has_images:
                img_grp = obs_grp.create_group("images")
                img_grp.create_dataset("top", data=top_arr, **comp_kwargs)
                img_grp.create_dataset("wrist1", data=w1_arr, **comp_kwargs)
                img_grp.create_dataset("wrist2", data=w2_arr, **comp_kwargs)

            demo_grp.create_dataset("action", data=action_arr)
            demo_grp.create_dataset("rewards", data=reward_arr)
            demo_grp.create_dataset("terminals", data=term_arr)

            f.attrs["num_episodes"] = len(f["data"].keys())
            f.attrs["fps"] = self.fps

        # Record metadata summary
        ep_summary = {
            "episode_index": episode_idx,
            "length": num_steps,
            "duration_sec": round(num_steps / self.fps, 2),
            "success": bool(success),
            "task": task_description,
            "file": os.path.basename(ep_file),
        }
        if extra_metadata:
            ep_summary.update(extra_metadata)
        self.saved_episodes.append(ep_summary)

        print(f"  [SAVED] Episode {episode_idx:3d}: {num_steps} steps ({num_steps/self.fps:.2f}s) -> {os.path.basename(ep_file)}")
        self._reset_buffer()
        return True

    def export_lerobot_metadata(self):
        """
        Generates Hugging Face LeRobot dataset metadata files:
        - meta/info.json: Dataset schema, sensor shapes, and robot specifications.
        - meta/episodes.jsonl: Per-episode summary logs.
        - meta/stats.json: Dataset-wide normalization statistics (mean, std, min, max).
        """
        if not self.saved_episodes:
            print("  [WARN] No episodes recorded; skipping LeRobot metadata export.")
            return

        total_frames = sum(ep["length"] for ep in self.saved_episodes)
        total_episodes = len(self.saved_episodes)

        # 1. meta/info.json
        info_data = {
            "codebase_version": "v2.0",
            "robot_type": "dual_ur10e_robotiq_2f85",
            "fps": self.fps,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": 1,
            "features": {
                "action": {
                    "dtype": "float32",
                    "shape": [14],
                    "names": [
                        "arm1_pan", "arm1_lift", "arm1_elbow", "arm1_wrist1", "arm1_wrist2", "arm1_wrist3", "arm1_gripper",
                        "arm2_pan", "arm2_lift", "arm2_elbow", "arm2_wrist1", "arm2_wrist2", "arm2_wrist3", "arm2_gripper",
                    ],
                },
                "observation.state": {
                    "dtype": "float64",
                    "shape": [14],
                    "names": [
                        "arm1_pan_pos", "arm1_lift_pos", "arm1_elbow_pos", "arm1_wrist1_pos", "arm1_wrist2_pos", "arm1_wrist3_pos", "arm1_grip_pos",
                        "arm2_pan_pos", "arm2_lift_pos", "arm2_elbow_pos", "arm2_wrist1_pos", "arm2_wrist2_pos", "arm2_wrist3_pos", "arm2_grip_pos",
                    ],
                },
                "observation.velocity": {
                    "dtype": "float64",
                    "shape": [14],
                },
            },
        }

        if self.save_images:
            # Query actual image shape from dataset
            with h5py.File(self.dataset_path, "r") as f:
                first_demo = next(iter(f["data"].values()))
                h_top, w_top = first_demo["observations/images/top"].shape[1:3]
                h_w1, w_w1 = first_demo["observations/images/wrist1"].shape[1:3]
                h_w2, w_w2 = first_demo["observations/images/wrist2"].shape[1:3]

            info_data["features"]["observation.images.top"] = {
                "dtype": "video" if not self.compression else "image",
                "shape": [h_top, w_top, 3],
            }
            info_data["features"]["observation.images.wrist1"] = {
                "dtype": "video" if not self.compression else "image",
                "shape": [h_w1, w_w1, 3],
            }
            info_data["features"]["observation.images.wrist2"] = {
                "dtype": "video" if not self.compression else "image",
                "shape": [h_w2, w_w2, 3],
            }

        info_path = os.path.join(self.meta_dir, "info.json")
        with open(info_path, "w") as f:
            json.dump(info_data, f, indent=2)

        # 2. meta/episodes.jsonl
        episodes_path = os.path.join(self.meta_dir, "episodes.jsonl")
        with open(episodes_path, "w") as f:
            for ep in self.saved_episodes:
                # Convert numpy types to python native types
                clean_ep = {}
                for k, v in ep.items():
                    if isinstance(v, (np.ndarray, list)):
                        clean_ep[k] = [float(x) if isinstance(x, (np.floating, float)) else bool(x) if isinstance(x, (np.bool_, bool)) else x for x in v]
                    elif isinstance(v, (np.floating, float)):
                        clean_ep[k] = float(v)
                    elif isinstance(v, (np.integer, int)):
                        clean_ep[k] = int(v)
                    elif isinstance(v, (np.bool_, bool)):
                        clean_ep[k] = bool(v)
                    else:
                        clean_ep[k] = str(v) if not isinstance(v, (str, dict)) else v
                f.write(json.dumps(clean_ep) + "\n")

        # 3. meta/stats.json (Compute dataset-wide statistics across all episodes)
        all_actions = []
        all_qpos = []
        all_qvel = []

        with h5py.File(self.dataset_path, "r") as f:
            for demo_key in f["data"].keys():
                all_actions.append(f[f"data/{demo_key}/action"][:])
                all_qpos.append(f[f"data/{demo_key}/observations/qpos"][:])
                all_qvel.append(f[f"data/{demo_key}/observations/qvel"][:])

        if all_actions:
            cat_actions = np.concatenate(all_actions, axis=0)
            cat_qpos = np.concatenate(all_qpos, axis=0)
            cat_qvel = np.concatenate(all_qvel, axis=0)

            stats_data = {
                "action": {
                    "mean": cat_actions.mean(axis=0).tolist(),
                    "std": cat_actions.std(axis=0).tolist(),
                    "min": cat_actions.min(axis=0).tolist(),
                    "max": cat_actions.max(axis=0).tolist(),
                },
                "observation.state": {
                    "mean": cat_qpos.mean(axis=0).tolist(),
                    "std": cat_qpos.std(axis=0).tolist(),
                    "min": cat_qpos.min(axis=0).tolist(),
                    "max": cat_qpos.max(axis=0).tolist(),
                },
                "observation.velocity": {
                    "mean": cat_qvel.mean(axis=0).tolist(),
                    "std": cat_qvel.std(axis=0).tolist(),
                    "min": cat_qvel.min(axis=0).tolist(),
                    "max": cat_qvel.max(axis=0).tolist(),
                },
            }

            stats_path = os.path.join(self.meta_dir, "stats.json")
            with open(stats_path, "w") as f:
                json.dump(stats_data, f, indent=2)

        print(f"  [METADATA] Exported LeRobot metadata to {self.meta_dir}/ (info.json, episodes.jsonl, stats.json).")
