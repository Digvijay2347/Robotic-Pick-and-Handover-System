"""
Action Chunking with Transformers (ACT) Training Pipeline.
Trains an ACT policy on bimanual robotic handover demonstrations with multi-camera
ResNet-18 visual feature extractors, C-VAE style encoder, and Transformer action decoder.
"""

import os
import sys
import time
import json
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from typing import Dict, Any

# Ensure project root in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from policies.act_dataset import ACTDataset
from policies.act_model import ACTPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Train ACT Policy for Dual UR10e Handover")
    parser.add_argument("--config", type=str, default="policies/act_config.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override transformer learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda', 'auto')")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save model checkpoints")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_device(requested: str = "auto") -> torch.device:
    if requested in ["cuda", "gpu"] and torch.cuda.is_available():
        device = torch.device("cuda")
    elif requested == "cpu":
        device = torch.device("cpu")
    else:
        # Auto detection
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    # Configure CPU optimization if running on CPU
    if device.type == "cpu":
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        print(f"[Hardware Engine] Configured PyTorch CPU execution with {num_threads} worker threads.")
    else:
        print(f"[Hardware Engine] Configured PyTorch on {torch.cuda.get_device_name(0)}.")
    
    return device


def train_act():
    args = parse_args()
    config = load_config(args.config)

    # Overrides from CLI
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr_transformer"] = args.lr
    if args.save_dir is not None:
        config["training"]["checkpoint_dir"] = args.save_dir

    device = get_device(args.device or config["training"].get("device", "auto"))

    # Set seeds
    seed = config["training"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 80)
    print("      PHASE 7: ACTION CHUNKING WITH TRANSFORMERS (ACT) TRAINING PIPELINE")
    print("=" * 80)
    print(f"  Dataset HDF5    : {config['dataset']['hdf5_path']}")
    print(f"  Chunk Size (K)  : {config['model']['chunk_size']} future action steps (0.50s)")
    print(f"  Cameras         : {config['dataset']['camera_names']}")
    print(f"  Batch Size      : {config['training']['batch_size']}")
    print(f"  Epochs          : {config['training']['num_epochs']}")
    print(f"  KL Weight (beta): {config['training']['kl_weight']}")
    print(f"  Device          : {device}")
    print("=" * 80)

    # 1. Dataset & DataLoader Initialization
    full_dataset = ACTDataset(
        hdf5_path=config["dataset"]["hdf5_path"],
        stats_path=config["dataset"]["stats_path"],
        chunk_size=config["model"]["chunk_size"],
        camera_names=config["dataset"]["camera_names"],
        image_size=tuple(config["dataset"]["input_image_size"]),
        step_stride=config["dataset"].get("step_stride", 1),
    )

    total_samples = len(full_dataset)
    if total_samples == 0:
        raise RuntimeError("No samples found in dataset! Ensure demonstrations exist in data/handover_demos.hdf5")

    val_ratio = 0.1
    val_size = max(1, int(total_samples * val_ratio))
    train_size = total_samples - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    print(f"  Train samples: {train_size} | Validation samples: {val_size}")

    # 2. Model Instantiation
    model = ACTPolicy(
        camera_names=config["dataset"]["camera_names"],
        chunk_size=config["model"]["chunk_size"],
        state_dim=config["model"]["state_dim"],
        action_dim=config["model"]["action_dim"],
        d_model=config["model"]["d_model"],
        nhead=config["model"]["nheads"],
        enc_layers=config["model"]["enc_layers"],
        dec_layers=config["model"]["dec_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        latent_dim=config["model"]["latent_dim"],
        dropout=config["model"]["dropout"],
        kl_weight=config["training"]["kl_weight"],
        freeze_backbone=config["model"].get("freeze_backbone", True),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable Parameters: {total_params:,} ({total_params * 4 / (1024**2):.2f} MB)")

    # 3. Optimizers & Schedulers
    # Differential learning rates: backbones get lower LR
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "backbones" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": config["training"]["lr_backbone"]},
            {"params": other_params, "lr": config["training"]["lr_transformer"]},
        ],
        weight_decay=config["training"]["weight_decay"],
    )

    epochs = config["training"]["num_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )

    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    kl_weight = config["training"]["kl_weight"]

    history = {
        "epoch": [],
        "train_total_loss": [],
        "train_l1_loss": [],
        "train_kl_loss": [],
        "val_total_loss": [],
        "val_l1_loss": [],
        "val_kl_loss": [],
    }

    # 4. Training Loop
    print("\nStarting Training Execution...", flush=True)
    header = f"{'Epoch':^7} | {'Train Total':^12} | {'Train L1':^10} | {'Train KL':^10} | {'Val Total':^11} | {'Val L1':^9} | {'Time (s)':^8}"
    print("-" * len(header), flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)

    t_start_train = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_tot_loss, train_l1_loss, train_kl_loss = 0.0, 0.0, 0.0
        train_batches = 0

        for batch in train_loader:
            images = {k: v.to(device) for k, v in batch["images"].items()}
            qpos = batch["qpos"].to(device)
            actions = batch["actions"].to(device)
            is_pad = batch["is_pad"].to(device)

            optimizer.zero_grad()
            out = model(images=images, qpos=qpos, actions=actions, is_pad=is_pad)

            total_loss = out["total_loss"]
            total_loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            train_tot_loss += total_loss.item()
            train_l1_loss += out["l1_loss"].item()
            train_kl_loss += out["kl_loss"].item()
            train_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_tot_loss, val_l1_loss, val_kl_loss = 0.0, 0.0, 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = {k: v.to(device) for k, v in batch["images"].items()}
                qpos = batch["qpos"].to(device)
                actions = batch["actions"].to(device)
                is_pad = batch["is_pad"].to(device)

                out = model(images=images, qpos=qpos, actions=actions, is_pad=is_pad)
                val_tot_loss += out["total_loss"].item()
                val_l1_loss += out["l1_loss"].item()
                val_kl_loss += out["kl_loss"].item()
                val_batches += 1

        avg_train_tot = train_tot_loss / max(1, train_batches)
        avg_train_l1 = train_l1_loss / max(1, train_batches)
        avg_train_kl = train_kl_loss / max(1, train_batches)

        avg_val_tot = val_tot_loss / max(1, val_batches)
        avg_val_l1 = val_l1_loss / max(1, val_batches)
        avg_val_kl = val_kl_loss / max(1, val_batches)

        dt = time.time() - t0

        print(
            f"{epoch:^7d} | {avg_train_tot:^12.4f} | {avg_train_l1:^10.4f} | {avg_train_kl:^10.4f} | "
            f"{avg_val_tot:^11.4f} | {avg_val_l1:^9.4f} | {dt:^8.2f}",
            flush=True,
        )

        history["epoch"].append(epoch)
        history["train_total_loss"].append(avg_train_tot)
        history["train_l1_loss"].append(avg_train_l1)
        history["train_kl_loss"].append(avg_train_kl)
        history["val_total_loss"].append(avg_val_tot)
        history["val_l1_loss"].append(avg_val_l1)
        history["val_kl_loss"].append(avg_val_kl)

        # Save Best Model Checkpoint
        if avg_val_tot < best_val_loss:
            best_val_loss = avg_val_tot
            best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": config,
                },
                best_ckpt_path,
            )

        # Periodic checkpoint
        if epoch % config["training"]["save_every"] == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"act_epoch_{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": avg_val_tot,
                    "config": config,
                },
                ckpt_path,
            )

    total_time = time.time() - t_start_train
    print("-" * len(header))
    print(f"\nTraining Complete in {total_time:.2f}s ({total_time/60:.2f}m)!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Model Checkpoint: {os.path.join(checkpoint_dir, 'best_model.pt')}")

    # Save training history JSON
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved Training History Telemetry to: {history_path}")

    # Close HDF5 handle
    full_dataset.close()


if __name__ == "__main__":
    train_act()
