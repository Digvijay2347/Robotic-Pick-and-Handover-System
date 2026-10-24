"""
Consolidates all individual episode HDF5 files from data/episodes/ into data/handover_demos.hdf5
and exports dataset-wide LeRobot metadata (info.json, episodes.jsonl, stats.json).
"""

import os
import sys
import glob
import json
import h5py
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataset.recorder import EpisodeRecorder
from dataset.dataset_utils import inspect_dataset

def consolidate(data_dir: str = "data", fps: int = 25):
    data_dir = os.path.abspath(data_dir)
    episodes_dir = os.path.join(data_dir, "episodes")
    master_path = os.path.join(data_dir, "handover_demos.hdf5")
    meta_dir = os.path.join(data_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    ep_files = sorted(glob.glob(os.path.join(episodes_dir, "episode_*.hdf5")))
    print(f"Found {len(ep_files)} individual episode files in {episodes_dir}.")

    if not ep_files:
        print("No episode files found!")
        return

    saved_summaries = []

    with h5py.File(master_path, "w") as f_master:
        data_grp = f_master.create_group("data")

        for ep_file in ep_files:
            fname = os.path.basename(ep_file)
            # Parse episode index from episode_000012.hdf5
            ep_idx = int(fname.replace("episode_", "").replace(".hdf5", ""))
            demo_name = f"demo_{ep_idx}"

            with h5py.File(ep_file, "r") as f_ep:
                # Group copy
                f_master.copy(f_ep["observations"], f"data/{demo_name}/observations")
                f_master.copy(f_ep["action"], f"data/{demo_name}/action")
                f_master.copy(f_ep["rewards"], f"data/{demo_name}/rewards")
                f_master.copy(f_ep["terminals"], f"data/{demo_name}/terminals")

                num_samples = f_ep["action"].shape[0]
                demo_grp = f_master[f"data/{demo_name}"]
                demo_grp.attrs["num_samples"] = num_samples
                demo_grp.attrs["success"] = True
                demo_grp.attrs["task"] = f_ep.attrs.get("task", "Pick red cube from shelf, handover to Arm 2, and place in storage bin")

                # Copy extra attrs
                summary = {
                    "episode_index": ep_idx,
                    "length": num_samples,
                    "duration_sec": round(num_samples / fps, 2),
                    "success": True,
                    "file": fname,
                }
                for k in f_ep.attrs:
                    if k not in ("num_samples", "fps", "sim"):
                        demo_grp.attrs[k] = f_ep.attrs[k]
                        summary[k] = f_ep.attrs[k]
                saved_summaries.append(summary)

        f_master.attrs["num_episodes"] = len(ep_files)
        f_master.attrs["fps"] = fps

    print(f"Consolidated {len(ep_files)} episodes into {master_path}.")

    # Export LeRobot metadata
    recorder = EpisodeRecorder(
        output_dir=data_dir,
        dataset_name="handover_demos.hdf5",
        fps=fps,
        save_images=True,
        compression="lzf",
    )
    recorder.saved_episodes = saved_summaries
    recorder.export_lerobot_metadata()

    inspect_dataset(master_path)

if __name__ == "__main__":
    consolidate()
