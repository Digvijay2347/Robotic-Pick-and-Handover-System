"""
ACT Dataset Loader with Contiguous In-Memory Preloading.
Preloads and preprocesses synchronized multi-camera RGB streams and joint trajectories
into RAM for maximum training throughput, eliminating disk seek and decompression overhead.
"""

import os
import glob
import json
import time
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from typing import Dict, List, Tuple, Optional, Any


class ACTDataset(Dataset):
    """
    High-Performance PyTorch Dataset for Action Chunking with Transformers (ACT).
    Loads multimodal trajectories with temporal action chunking and dataset-wide normalization.
    Pre-processes images directly into contiguous RAM buffers for instant batch retrieval.
    """

    def __init__(
        self,
        hdf5_path: str = "data/handover_demos.hdf5",
        stats_path: str = "data/meta/stats.json",
        chunk_size: int = 25,
        camera_names: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (112, 112),
        norm_mode: str = "zscore",
        step_stride: int = 2,
        preload: bool = True,
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.stats_path = stats_path
        self.chunk_size = chunk_size
        self.camera_names = camera_names or ["top", "wrist1", "wrist2"]
        self.image_size = tuple(image_size)
        self.norm_mode = norm_mode
        self.step_stride = max(1, step_stride)
        self.preload = preload

        # 1. Load normalization statistics
        self._load_stats()

        # 2. Build sample index across episodes
        self._build_index()

        # 3. Preload into contiguous memory buffers if requested
        if self.preload:
            self._preload_data()

    def _load_stats(self):
        """Loads normalization statistics from stats.json."""
        if os.path.exists(self.stats_path):
            with open(self.stats_path, "r") as f:
                stats = json.load(f)
            self.action_mean = np.array(stats["action"]["mean"], dtype=np.float32)
            self.action_std = np.array(stats["action"]["std"], dtype=np.float32)
            self.action_min = np.array(stats["action"]["min"], dtype=np.float32)
            self.action_max = np.array(stats["action"]["max"], dtype=np.float32)

            self.qpos_mean = np.array(stats["observation.state"]["mean"], dtype=np.float32)
            self.qpos_std = np.array(stats["observation.state"]["std"], dtype=np.float32)
            self.qpos_min = np.array(stats["observation.state"]["min"], dtype=np.float32)
            self.qpos_max = np.array(stats["observation.state"]["max"], dtype=np.float32)
        else:
            self.action_mean = np.zeros(14, dtype=np.float32)
            self.action_std = np.ones(14, dtype=np.float32)
            self.action_min = -np.ones(14, dtype=np.float32)
            self.action_max = np.ones(14, dtype=np.float32)
            self.qpos_mean = np.zeros(14, dtype=np.float32)
            self.qpos_std = np.ones(14, dtype=np.float32)
            self.qpos_min = -np.ones(14, dtype=np.float32)
            self.qpos_max = np.ones(14, dtype=np.float32)

        self.action_std = np.where(self.action_std < 1e-6, 1.0, self.action_std)
        self.qpos_std = np.where(self.qpos_std < 1e-6, 1.0, self.qpos_std)

    def _build_index(self):
        """Discovers episodes and builds (ep_idx, t, num_steps) sample windows."""
        self.samples: List[Tuple[int, int, int]] = []  # (ep_id, t, num_steps)
        self.episodes_meta: List[Dict[str, Any]] = []

        if os.path.isfile(self.hdf5_path):
            with h5py.File(self.hdf5_path, "r") as f:
                if "data" in f:
                    demos = sorted(list(f["data"].keys()))
                    for ep_id, demo_key in enumerate(demos):
                        grp = f[f"data/{demo_key}"]
                        num_steps = grp.attrs.get("num_samples", len(grp["action"]))
                        self.episodes_meta.append({
                            "type": "consolidated",
                            "file": self.hdf5_path,
                            "group": f"data/{demo_key}",
                            "num_steps": num_steps,
                        })
                        for t in range(0, num_steps, self.step_stride):
                            self.samples.append((ep_id, t, num_steps))
        print(f"[ACTDataset] Discovered {len(self.samples)} sample chunks across {len(self.episodes_meta)} episodes.")

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.norm_mode == "minmax":
            denom = np.where(self.action_max - self.action_min < 1e-6, 1.0, self.action_max - self.action_min)
            return 2.0 * (action - self.action_min) / denom - 1.0
        return (action - self.action_mean) / self.action_std

    def unnormalize_action(self, norm_action: np.ndarray) -> np.ndarray:
        if self.norm_mode == "minmax":
            denom = np.where(self.action_max - self.action_min < 1e-6, 1.0, self.action_max - self.action_min)
            return (norm_action + 1.0) / 2.0 * denom + self.action_min
        return norm_action * self.action_std + self.action_mean

    def normalize_qpos(self, qpos: np.ndarray) -> np.ndarray:
        if self.norm_mode == "minmax":
            denom = np.where(self.qpos_max - self.qpos_min < 1e-6, 1.0, self.qpos_max - self.qpos_min)
            return 2.0 * (qpos - self.qpos_min) / denom - 1.0
        return (qpos - self.qpos_mean) / self.qpos_std

    def unnormalize_qpos(self, norm_qpos: np.ndarray) -> np.ndarray:
        if self.norm_mode == "minmax":
            denom = np.where(self.qpos_max - self.qpos_min < 1e-6, 1.0, self.qpos_max - self.qpos_min)
            return (norm_qpos + 1.0) / 2.0 * denom + self.qpos_min
        return norm_qpos * self.qpos_std + self.qpos_mean

    def _preload_data(self):
        """Contiguously preloads and preprocesses images, qpos, and actions into RAM."""
        print(f"[ACTDataset] Preloading episode tensors into RAM buffer (image_size={self.image_size})...")
        t0 = time.time()
        self.preloaded_episodes = []

        with h5py.File(self.hdf5_path, "r") as f:
            for ep_meta in self.episodes_meta:
                grp = f[ep_meta["group"]]
                num_steps = ep_meta["num_steps"]

                # 1. Proprioception & Actions
                raw_qpos = grp["observations/qpos"][:]
                norm_qpos = self.normalize_qpos(raw_qpos)
                qpos_tensor = torch.from_numpy(norm_qpos).float()

                raw_actions = grp["action"][:]
                norm_actions = self.normalize_action(raw_actions)
                actions_tensor = torch.from_numpy(norm_actions).float()

                # 2. Multi-Camera Image Streams
                cams_tensors = {}
                has_imgs = "observations/images" in grp
                for cam_name in self.camera_names:
                    if has_imgs and cam_name in grp["observations/images"]:
                        raw_cams = grp["observations/images"][cam_name][:]  # (N, H, W, 3) uint8
                        # Convert to torch tensor (N, 3, H, W)
                        cam_t = torch.from_numpy(raw_cams).permute(0, 3, 1, 2).float() / 255.0
                        # Batch resize to target size
                        cam_t = TF.resize(cam_t, list(self.image_size), antialias=True)
                        # Normalize with ImageNet mean and std
                        cam_t = TF.normalize(cam_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    else:
                        cam_t = torch.zeros(num_steps, 3, *self.image_size, dtype=torch.float32)
                    cams_tensors[cam_name] = cam_t

                self.preloaded_episodes.append({
                    "qpos": qpos_tensor,
                    "actions": actions_tensor,
                    "images": cams_tensors,
                    "num_steps": num_steps,
                })

        dt = time.time() - t0
        mem_mb = sum(
            ep["qpos"].numel() * 4 + ep["actions"].numel() * 4 + sum(v.numel() * 4 for v in ep["images"].values())
            for ep in self.preloaded_episodes
        ) / (1024**2)
        print(f"[ACTDataset] Preloading Complete in {dt:.2f}s! Memory footprint: {mem_mb:.2f} MB in RAM.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_id, t, num_steps = self.samples[idx]
        ep = self.preloaded_episodes[ep_id]

        # 1. Observation at step t
        qpos_tensor = ep["qpos"][t]
        images_dict = {cam: ep["images"][cam][t] for cam in self.camera_names}

        # 2. Action Chunk [t : t + chunk_size]
        actions_end = min(t + self.chunk_size, num_steps)
        chunk_slice = ep["actions"][t:actions_end]
        k_valid = chunk_slice.shape[0]

        is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
        if k_valid < self.chunk_size:
            pad_len = self.chunk_size - k_valid
            last_act = chunk_slice[-1:] if k_valid > 0 else torch.zeros((1, 14), dtype=torch.float32)
            padding = last_act.repeat(pad_len, 1)
            actions_tensor = torch.cat([chunk_slice, padding], dim=0)
            is_pad[k_valid:] = True
        else:
            actions_tensor = chunk_slice

        return {
            "images": images_dict,
            "qpos": qpos_tensor,
            "actions": actions_tensor,
            "is_pad": is_pad,
        }

    def close(self):
        pass
