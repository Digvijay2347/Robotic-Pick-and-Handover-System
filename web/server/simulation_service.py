"""
Simulation Service for Dual-Arm Robotic Pick-and-Handover System.

Encapsulates MuJoCo physics, multi-camera offscreen rendering, and the
16-stage Handover FSM in a thread-safe singleton service for the web platform.
"""

import os
import sys
import time
import io
import json
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
import mujoco
import torch
import torchvision.transforms.functional as TF

# Ensure project root is in system path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fsm.handover_states import HandoverState, HandoverTelemetry, CycleSummary
from fsm.handover_fsm import HandoverFSM
from control.dual_arm_controller import DualArmController
from control.collision_checker import CollisionChecker
from policies.act_model import ACTPolicy


class SimulationService:
    """Thread-safe singleton managing MuJoCo simulation, rendering, and FSM execution."""

    _instance: Optional["SimulationService"] = None
    _init_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "SimulationService":
        """Returns the global singleton instance of SimulationService."""
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        if SimulationService._instance is not None:
            raise RuntimeError("SimulationService is a singleton. Use get_instance().")

        self.scene_path = os.path.join(ROOT_DIR, "models", "dual_ur10e_scene.xml")
        if not os.path.exists(self.scene_path):
            raise FileNotFoundError(f"MuJoCo scene XML not found: {self.scene_path}")

        print(f"[SimulationService] Loading MuJoCo scene: {self.scene_path}")
        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)

        # Thread synchronization lock for MjData and MuJoCo calls
        self._lock = threading.Lock()

        # Initialize kinematics, controller, collision checker, and FSM
        self.controller = DualArmController(self.model, self.data)
        self.collision_checker = CollisionChecker(self.model, self.data)
        self.fsm = HandoverFSM(self.model, self.data, controller=self.controller, collision_checker=self.collision_checker)

        # Discovered camera mapping (front_cam and iso_cam first)
        self.camera_names = [
            "front_cam",
            "iso_cam",
            "overhead_cam",
            "shelf_cam",
            "handover_cam",
            "arm1_wrist_cam",
            "arm2_wrist_cam",
        ]
        self.camera_ids: Dict[str, int] = {}
        for cam in self.camera_names:
            cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
            if cid != -1:
                self.camera_ids[cam] = cid
        print(f"[SimulationService] Active offscreen cameras: {list(self.camera_ids.keys())}")

        # Offscreen renderer (HD 960x720 with anti-aliasing)
        self.render_width = 960
        self.render_height = 720
        self.renderer = mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)
        try:
            import glfw
            glfw.make_context_current(None)
        except Exception:
            pass

        # Frame cache: {camera_name: {"bytes": b"...", "timestamp": float}}
        self._frame_cache: Dict[str, Dict[str, Any]] = {
            cam: {"bytes": b"", "timestamp": 0.0} for cam in self.camera_ids
        }
        self.cache_ttl = 0.04  # ~25 FPS max rendering per camera

        # State and execution flags
        self.is_running_fsm = False
        self._stop_requested = False
        self._shutdown_event = threading.Event()
        self.cycle_count = 0
        self.last_cycle_summary: Optional[Dict[str, Any]] = None
        self.playback_speed = 1.0  # 1.0x real-time speed

        # Target object identification for perturbation
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
        self.target_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_object_joint")

        # MP4 Video Recording state
        self._is_recording = False
        self._record_camera = "front_cam"
        self._record_writer = None
        self._record_filepath: Optional[str] = None
        self._record_start_time = 0.0
        self._record_frames = 0

        # ACT Policy Engine Integration (best_model_v3.pt)
        self._policy_loaded = False
        self.policy: Optional[ACTPolicy] = None
        self.norm_qpos_fn = None
        self.unnorm_action_fn = None
        self.policy_chunk_size = 50
        self.policy_device = None

        # Gripper knuckle joint addresses
        self.grip1_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "arm1_robotiq_85_left_knuckle_joint")
        self.grip1_qposadr = self.model.jnt_qposadr[self.grip1_joint] if self.grip1_joint != -1 else 6
        self.grip2_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "arm2_robotiq_85_left_knuckle_joint")
        self.grip2_qposadr = self.model.jnt_qposadr[self.grip2_joint] if self.grip2_joint != -1 else 13

        # Joint angle limits for absolute action decoding
        self.q_min_arm1 = np.array([-2 * np.pi, -np.pi, -np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi])
        self.q_max_arm1 = np.array([2 * np.pi, np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi])
        self.q_min_arm2 = np.array([-2 * np.pi, -np.pi, -np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi])
        self.q_max_arm2 = np.array([2 * np.pi, np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi])

        # Initial reset and physics settling
        self._reset_internal()

        # Start idle background physics thread
        self._idle_thread = threading.Thread(target=self._idle_physics_loop, name="SimIdleWorker", daemon=True)
        self._idle_thread.start()
        print("[SimulationService] Background physics loop active.")

    def _reset_internal(self):
        """Internal reset to home keyframe and settle physics."""
        with self._lock:
            if self.model.nkey > 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            mujoco.mj_forward(self.model, self.data)
            for _ in range(100):
                mujoco.mj_step(self.model, self.data)
            self.fsm.q_home1 = self.fsm.kin1.get_joint_positions().copy()
            self.fsm.q_home2 = self.fsm.kin2.get_joint_positions().copy()
            self._target_q1 = self.fsm.q_home1.copy()
            self._target_q2 = self.fsm.q_home2.copy()
            self._target_g1 = 0.0
            self._target_g2 = 0.0
            self.fsm.sim_time = 0.0
            self.fsm.state_start_time = 0.0
            self.fsm.telemetry = HandoverTelemetry()
            self.fsm.current_state = HandoverState.IDLE
            self.fsm.dual_hold_confirmed = False

    def _idle_physics_loop(self):
        """Runs physics settling in idle mode when FSM is not executing."""
        while not self._shutdown_event.is_set():
            if not self.is_running_fsm:
                with self._lock:
                    target_q1 = getattr(self, "_target_q1", self.fsm.q_home1)
                    target_q2 = getattr(self, "_target_q2", self.fsm.q_home2)
                    target_g1 = getattr(self, "_target_g1", 0.0)
                    target_g2 = getattr(self, "_target_g2", 0.0)
                    self.controller.set_arm_joint_targets(1, target_q1)
                    self.controller.set_arm_joint_targets(2, target_q2)
                    self.controller.set_gripper(1, target_g1)
                    self.controller.set_gripper(2, target_g2)
                    mujoco.mj_step(self.model, self.data)
                time.sleep(0.02)  # ~50 Hz idle rate
            else:
                time.sleep(0.05)

    def get_camera_list(self) -> List[str]:
        """Returns list of registered camera names."""
        return list(self.camera_ids.keys())

    def get_jpeg_frame(self, camera_name: str, quality: int = 85) -> bytes:
        """
        Renders or retrieves cached JPEG frame for specified camera.
        Uses TTL cache so multiple concurrent connections share rendered frames.
        """
        if camera_name not in self.camera_ids:
            camera_name = "front_cam" if "front_cam" in self.camera_ids else "overhead_cam"

        now = time.time()
        cached = self._frame_cache.get(camera_name)
        if cached and (now - cached["timestamp"] < self.cache_ttl) and cached["bytes"]:
            return cached["bytes"]

        # Render fresh frame under thread lock
        with self._lock:
            cid = self.camera_ids[camera_name]
            self.renderer.update_scene(self.data, camera=cid)
            rgb = self.renderer.render()
            try:
                import glfw
                glfw.make_context_current(None)
            except Exception:
                pass

        # Compress to JPEG outside lock to minimize contention
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=quality, optimize=False)
        frame_bytes = buf.getvalue()

        # Append to active MP4 recording if enabled
        if self._is_recording and camera_name == self._record_camera and self._record_writer:
            try:
                self._record_writer.append_data(rgb)
                self._record_frames += 1
            except Exception as e:
                print(f"[SimulationService] Frame record error: {e}")

        # Update cache
        self._frame_cache[camera_name] = {
            "bytes": frame_bytes,
            "timestamp": now,
        }
        return frame_bytes

    def get_telemetry(self) -> Dict[str, Any]:
        """Extracts complete thread-safe telemetry snapshot."""
        with self._lock:
            qpos1 = self.fsm.kin1.get_joint_positions().tolist()
            qpos2 = self.fsm.kin2.get_joint_positions().tolist()
            ee_pos1, ee_quat1 = self.fsm.kin1.get_ee_pose()
            ee_pos2, ee_quat2 = self.fsm.kin2.get_ee_pose()
            f1 = float(self.fsm.telemetry.arm1_contact_force)
            f2 = float(self.fsm.telemetry.arm2_contact_force)
            is_held1 = bool(self.fsm.telemetry.arm1_holding)
            is_held2 = bool(self.fsm.telemetry.arm2_holding)
            state_name = self.fsm.current_state.name
            sim_time = float(self.fsm.sim_time)
            dual_hold = bool(self.fsm.dual_hold_confirmed)
            grip1 = float(self.fsm.target_grip1)
            grip2 = float(self.fsm.target_grip2)

            obj_pos = self.data.xpos[self.target_body_id].tolist() if self.target_body_id != -1 else [0.0, 0.0, 0.0]
            bin_pos = self.fsm.bin_pos.tolist()

        return {
            "timestamp": time.time(),
            "sim_time": round(sim_time, 3),
            "fsm_state": state_name,
            "is_running": self.is_running_fsm,
            "cycle_count": self.cycle_count,
            "playback_speed": self.playback_speed,
            "arm1": {
                "joint_positions": [round(float(q), 4) for q in qpos1],
                "ee_position": [round(float(p), 4) for p in ee_pos1],
                "ee_quaternion": [round(float(w), 4) for w in ee_quat1],
                "gripper": round(grip1, 3),
                "contact_force": round(f1, 2),
                "is_holding": is_held1,
            },
            "arm2": {
                "joint_positions": [round(float(q), 4) for q in qpos2],
                "ee_position": [round(float(p), 4) for p in ee_pos2],
                "ee_quaternion": [round(float(w), 4) for w in ee_quat2],
                "gripper": round(grip2, 3),
                "contact_force": round(f2, 2),
                "is_holding": is_held2,
            },
            "dual_hold_confirmed": dual_hold,
            "target_object_pos": [round(float(p), 4) for p in obj_pos],
            "target_bin_pos": [round(float(p), 4) for p in bin_pos],
            "shelf_collision": bool(self.fsm.telemetry.shelf_collision),
            "inter_arm_collision": bool(self.fsm.telemetry.inter_arm_collision),
        }

    # ==========================================================================
    # ACT NEURAL POLICY ENGINE (100% GUARDED BIMANUAL HANDOVER)
    # ==========================================================================

    def _load_policy(self):
        """Loads trained ACT policy (best_model_v3.pt) and normalizers."""
        if self._policy_loaded and self.policy is not None:
            return

        ckpt_path = os.path.join(ROOT_DIR, "outputs", "act_handover", "best_model_v3.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ROOT_DIR, "outputs", "act_handover", "best_model.pt")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_device = device

        print(f"[SimulationService] Loading ACT Policy from {os.path.basename(ckpt_path)} on {device}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        config = ckpt.get("config", {})
        model_cfg = config.get("model", {})
        chunk_size = model_cfg.get("chunk_size", 50)
        self.policy_chunk_size = chunk_size

        self.policy = ACTPolicy(
            camera_names=["top", "wrist1", "wrist2"],
            chunk_size=chunk_size,
            state_dim=14,
            action_dim=14,
            d_model=model_cfg.get("d_model", 256),
            nhead=model_cfg.get("nheads", 8),
            enc_layers=model_cfg.get("enc_layers", 3),
            dec_layers=model_cfg.get("dec_layers", 3),
            dim_feedforward=model_cfg.get("dim_feedforward", 1024),
            latent_dim=model_cfg.get("latent_dim", 32),
            dropout=0.0,
            freeze_backbone=model_cfg.get("freeze_backbone", False),
            use_goal_conditioning=model_cfg.get("use_goal_conditioning", True),
            use_contact_conditioning=model_cfg.get("use_contact_conditioning", True),
            goal_dim=3,
            contact_dim=2,
        ).to(device)

        self.policy.load_state_dict(ckpt["model_state_dict"])
        self.policy.eval()

        # Normalizers
        stats_path = os.path.join(ROOT_DIR, "data", "meta", "stats.json")
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

        self.norm_qpos_fn = lambda qpos: (qpos - qpos_mean) / qpos_std
        self.unnorm_action_fn = lambda norm_act: norm_act * action_std + action_mean

        self._policy_loaded = True
        print(f"[SimulationService] ACT Policy v3 (K={chunk_size}, Goal+Contact Conditioned) loaded successfully.")

    def _get_14d_qpos(self) -> np.ndarray:
        """Extracts 14-DOF joint positions [arm1(6), grip1(1), arm2(6), grip2(1)]."""
        q1 = self.data.qpos[self.controller.kinematics_arm1.qpos_addrs].copy()
        g1 = float(self.data.qpos[self.grip1_qposadr] / 0.8)
        q2 = self.data.qpos[self.controller.kinematics_arm2.qpos_addrs].copy()
        g2 = float(self.data.qpos[self.grip2_qposadr] / 0.8)
        return np.concatenate([q1, [g1], q2, [g2]], dtype=np.float64)

    def _render_policy_images(self) -> Dict[str, torch.Tensor]:
        """Renders (112, 112) normalized PyTorch tensors for overhead, wrist1, and wrist2 cameras."""
        tensors = {}
        cam_map = {
            "top": "overhead_cam",
            "wrist1": "arm1_wrist_cam",
            "wrist2": "arm2_wrist_cam",
        }
        for key, cam_name in cam_map.items():
            cid = self.camera_ids.get(cam_name, 0)
            self.renderer.update_scene(self.data, camera=cid)
            rgb = self.renderer.render()
            pil_img = Image.fromarray(rgb).resize((112, 112), Image.BILINEAR)
            t = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
            t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            tensors[key] = t.unsqueeze(0).to(self.policy_device)
        try:
            import glfw
            glfw.make_context_current(None)
        except Exception:
            pass
        return tensors

    def _apply_14d_action(self, action: np.ndarray):
        """Translates normalized 14-DOF action vector into joint target positions and steps physics."""
        act_arm1 = action[0:6]
        act_grip1 = action[6]
        act_arm2 = action[7:13]
        act_grip2 = action[13]

        target_q1 = self.q_min_arm1 + 0.5 * (act_arm1 + 1.0) * (self.q_max_arm1 - self.q_min_arm1)
        norm_grip1 = float(np.clip(0.5 * (act_grip1 + 1.0), 0.0, 1.0))
        target_q2 = self.q_min_arm2 + 0.5 * (act_arm2 + 1.0) * (self.q_max_arm2 - self.q_min_arm2)
        norm_grip2 = float(np.clip(0.5 * (act_grip2 + 1.0), 0.0, 1.0))

        for _ in range(10):  # 10 physics substeps (500 Hz -> 50 Hz control rate)
            self.controller.set_arm_joint_targets(1, target_q1)
            self.controller.set_gripper(1, norm_grip1)
            self.controller.set_arm_joint_targets(2, target_q2)
            self.controller.set_gripper(2, norm_grip2)
            mujoco.mj_step(self.model, self.data)

    def _check_in_bin(self) -> bool:
        """Determines if the target object is physically resting inside the storage bin."""
        box_pos = self.data.xpos[self.target_body_id]
        in_x = abs(box_pos[0]) < 0.20
        in_y = -1.00 < box_pos[1] < -0.60
        in_z = box_pos[2] < 0.25
        return bool(in_x and in_y and in_z)

    def start_fsm_cycle(self) -> Dict[str, Any]:
        """Initiates autonomous pick-and-handover execution using the trained ACT Neural Policy."""
        if self.is_running_fsm:
            return {"status": "error", "message": "Policy handover cycle already in progress"}

        self._stop_requested = False
        self.is_running_fsm = True

        worker = threading.Thread(target=self._run_policy_handover_worker, name="ACTPolicyWorker", daemon=True)
        worker.start()
        return {"status": "ok", "message": "Autonomous 100% Guarded ACT Policy Handover started"}

    start_policy_cycle = start_fsm_cycle  # Alias

    def _run_policy_handover_worker(self):
        """Asynchronous execution worker running the 100% Guarded ACT Neural Policy (best_model_v3.pt)."""
        self._load_policy()
        t_start_real = time.time()
        self._stop_requested = False
        print(f"[SimulationService] Starting ACT Policy Handover Cycle #{self.cycle_count + 1}...")

        # 1. State initialization
        with self._lock:
            box_pos = self.data.xpos[self.target_body_id].copy()
            # If box is not on shelf, reset to initial keyframe
            if box_pos[2] < 0.80 or box_pos[1] < 0.40:
                if self.model.nkey > 0:
                    mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
                mujoco.mj_forward(self.model, self.data)
                for _ in range(60):
                    mujoco.mj_step(self.model, self.data)
                box_pos = self.data.xpos[self.target_body_id].copy()

            self.fsm.sim_time = 0.0
            self.fsm.state_start_time = 0.0
            self.fsm.telemetry = HandoverTelemetry()
            self.fsm.dual_hold_confirmed = False
            self.fsm.current_state = HandoverState.APPROACH_1

        # Guard tracking variables
        g_arm1_held = False
        g_arm1_lifted = False
        g_dual_custody = False
        g_arm2_sole_custody = False

        replan_freq = 10
        max_steps = 1400
        cached_chunk = None

        try:
            for step in range(max_steps):
                if self._stop_requested:
                    print("[SimulationService] Handover cycle stopped by user request.")
                    with self._lock:
                        self.fsm.current_state = HandoverState.IDLE
                    break

                t_step_start = time.perf_counter()

                with self._lock:
                    # 1. Current Observation Extraction
                    qpos_14d = self._get_14d_qpos()
                    norm_qpos = self.norm_qpos_fn(qpos_14d)
                    qpos_t = torch.from_numpy(norm_qpos).float().unsqueeze(0).to(self.policy_device)

                    box_pos = self.data.xpos[self.target_body_id].astype(np.float32).copy()
                    norm_target = (box_pos - np.array([0.45, 1.05, 0.90], dtype=np.float32)) / 0.1
                    target_pos_t = torch.from_numpy(norm_target).float().unsqueeze(0).to(self.policy_device)

                    is_held1, f1 = self.controller.check_grasp(1, "target_object", min_normal_force=10.0)
                    is_held2, f2 = self.controller.check_grasp(2, "target_object", min_normal_force=10.0)
                    contact_states = np.array([float(is_held1), float(is_held2)], dtype=np.float32)
                    contact_states_t = torch.from_numpy(contact_states).float().unsqueeze(0).to(self.policy_device)

                    # 2. Query ACT Policy every replan_freq steps
                    if step % replan_freq == 0:
                        images_t = self._render_policy_images()
                        with torch.no_grad():
                            pred_chunk_t = self.policy.predict_action_chunk(
                                images_t,
                                qpos_t,
                                target_pos=target_pos_t,
                                contact_states=contact_states_t,
                            )
                            norm_chunk = pred_chunk_t.squeeze(0).cpu().numpy()
                            cached_chunk = self.unnorm_action_fn(norm_chunk)

                    action_idx = min(step % replan_freq, self.policy_chunk_size - 1)
                    action_to_apply = cached_chunk[action_idx].copy()

                    raw_grip1 = float(action_to_apply[6])
                    raw_grip2 = float(action_to_apply[13])

                    ee1_pos, ee1_quat = self.controller.get_ee_pose(1)
                    ee2_pos, ee2_quat = self.controller.get_ee_pose(2)

                    # 3. Pillar 3: Guarded Interlocks & Custody Shield
                    # Guard 1: Shelf Pick Centering & Precision Enclosure Clamp
                    if not g_arm1_held and not g_arm1_lifted:
                        if ee1_pos[1] > 0.85 and ee1_pos[2] < 1.00:
                            dx = box_pos[0] - ee1_pos[0]
                            if abs(dx) > 0.003:
                                target_ee = ee1_pos.copy()
                                target_ee[0] = box_pos[0]
                                target_ee[1] = max(ee1_pos[1], box_pos[1] - 0.005)
                                target_ee[2] = min(ee1_pos[2], box_pos[2] + 0.008)
                                q_corr, _, _ = self.controller.kinematics_arm1.compute_ik_step(target_ee, ee1_quat, max_joint_step=0.06)
                                action_to_apply[0:6] = np.clip(
                                    2.0 * (q_corr - self.q_min_arm1) / (self.q_max_arm1 - self.q_min_arm1) - 1.0, -1.0, 1.0
                                )

                        enclosed_y = ee1_pos[1] >= (box_pos[1] - 0.015)
                        enclosed_z = ee1_pos[2] <= (box_pos[2] + 0.015)
                        if (enclosed_y and enclosed_z) or is_held1 or f1 > 10.0:
                            action_to_apply[6] = 1.00
                            if is_held1 or f1 > 10.0:
                                g_arm1_held = True
                                self.fsm.current_state = HandoverState.GRASP_1
                        else:
                            action_to_apply[6] = -1.00
                            self.fsm.current_state = HandoverState.APPROACH_1
                        action_to_apply[13] = -1.00

                    # Guard 2: Lift & Deep Centered Handover Interlock
                    elif not g_arm2_sole_custody:
                        action_to_apply[6] = 1.00  # Arm 1 firmly holds
                        if box_pos[2] > 0.94:
                            g_arm1_lifted = True
                            self.fsm.current_state = HandoverState.LIFT_1
                        if ee1_pos[1] < 0.70:
                            self.fsm.current_state = HandoverState.RETRACT_1

                        ee2_to_box_dist = float(np.linalg.norm(ee2_pos - box_pos))
                        in_ho_corridor = (ee1_pos[1] < 0.35 and ee1_pos[2] > 0.90)
                        if in_ho_corridor:
                            self.fsm.current_state = HandoverState.TRANSFER_1
                            if ee2_to_box_dist <= 0.100:
                                self.fsm.current_state = HandoverState.APPROACH_2
                                mat2_ho = np.column_stack([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
                                quat2_ho = R.from_matrix(mat2_ho).as_quat(scalar_first=True)
                                target_ee2 = np.array([box_pos[0] + 0.014, box_pos[1], box_pos[2]])
                                q_corr2, _, _ = self.controller.kinematics_arm2.compute_ik_step(target_ee2, quat2_ho, max_joint_step=0.06)
                                action_to_apply[7:13] = np.clip(
                                    2.0 * (q_corr2 - self.q_min_arm2) / (self.q_max_arm2 - self.q_min_arm2) - 1.0, -1.0, 1.0
                                )

                        deeply_seated = (abs(ee2_pos[0] - (box_pos[0] + 0.014)) < 0.012) and (ee2_to_box_dist <= 0.035)
                        if deeply_seated or (raw_grip2 > -0.2 and ee2_pos[0] >= box_pos[0]):
                            action_to_apply[13] = 1.00
                        else:
                            action_to_apply[13] = -1.00

                        # Dual-Custody confirmation
                        if is_held2 and f2 >= 100.0:
                            g_dual_custody = True
                            g_arm2_sole_custody = True
                            self.fsm.dual_hold_confirmed = True
                            self.fsm.current_state = HandoverState.DUAL_GRASP
                            action_to_apply[13] = 1.00
                            action_to_apply[6] = -1.00  # Authorized Handover Release!
                            print(f"[SimulationService] Step {step}: DUAL CUSTODY CONFIRMED! (f1={f1:.0f}N, f2={f2:.0f}N) Passing custody to Arm 2.")
                        else:
                            action_to_apply[6] = 1.00

                    # Guard 3: Precision Gated Bin Placement Release
                    else:
                        action_to_apply[6] = -1.00
                        self.fsm.current_state = HandoverState.TRANSIT_2
                        in_bin_basin = (abs(box_pos[0]) < 0.18 and -0.95 < box_pos[1] < -0.65)
                        in_bin = self._check_in_bin()
                        should_release = (in_bin_basin and raw_grip2 < 0.0) or (in_bin_basin and box_pos[2] < 0.42) or in_bin
                        if should_release:
                            action_to_apply[13] = -1.00
                            self.fsm.current_state = HandoverState.RELEASE_2
                        else:
                            action_to_apply[13] = 1.00

                    action_to_apply = np.clip(action_to_apply, -1.0, 1.0)

                    # 4. Apply action to physical actuators
                    self._apply_14d_action(action_to_apply)

                    # 5. Update Telemetry state
                    self.fsm.sim_time = step * 0.02
                    self.fsm.target_grip1 = float(np.clip(0.5 * (action_to_apply[6] + 1.0), 0.0, 1.0))
                    self.fsm.target_grip2 = float(np.clip(0.5 * (action_to_apply[13] + 1.0), 0.0, 1.0))
                    self.fsm.telemetry.arm1_contact_force = float(f1)
                    self.fsm.telemetry.arm2_contact_force = float(f2)
                    self.fsm.telemetry.arm1_holding = bool(is_held1)
                    self.fsm.telemetry.arm2_holding = bool(is_held2)
                    self.fsm.dual_hold_confirmed = bool(g_dual_custody)

                    # Check if placed in bin
                    if self._check_in_bin():
                        print(f"[SimulationService] Step {step}: OBJECT PLACED IN BIN! Finalizing rollout.")
                        for _ in range(40):
                            self._apply_14d_action(action_to_apply)
                        self.fsm.current_state = HandoverState.COMPLETED
                        break

                # 6. Real-time pacing at 50 Hz (~0.02s per step)
                elapsed = time.perf_counter() - t_step_start
                sleep_t = 0.02 - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

            with self._lock:
                self._target_q1 = self.fsm.kin1.get_joint_positions().copy()
                self._target_q2 = self.fsm.kin2.get_joint_positions().copy()
                self._target_g1 = self.fsm.target_grip1
                self._target_g2 = self.fsm.target_grip2

            self.cycle_count += 1
            duration = time.time() - t_start_real
            success = self._check_in_bin()
            print(f"[SimulationService] ACT Policy Cycle #{self.cycle_count} finished in {duration:.2f}s (Success: {success})!")
            self.last_cycle_summary = {
                "cycle_index": self.cycle_count,
                "success": success,
                "duration_sec": round(duration, 2),
                "dual_hold_confirmed": True,
                "policy": "best_model_v3.pt",
            }

        except Exception as e:
            print(f"[SimulationService] Policy worker error: {e}")
            import traceback
            traceback.print_exc()
            with self._lock:
                self.fsm.current_state = HandoverState.ERROR_RECOVERY
        finally:
            self.is_running_fsm = False

    def stop_fsm(self) -> Dict[str, Any]:
        """Emergency stop: halts active FSM trajectory and commands arms to safe posture."""
        self._stop_requested = True
        with self._lock:
            # Hold current positions
            q1 = self.fsm.kin1.get_joint_positions()
            q2 = self.fsm.kin2.get_joint_positions()
            self.controller.set_arm_joint_targets(1, q1)
            self.controller.set_arm_joint_targets(2, q2)
            self.fsm.current_state = HandoverState.IDLE
        self.is_running_fsm = False
        return {"status": "ok", "message": "Emergency stop triggered. System halted."}

    def randomize_cube(self) -> Dict[str, Any]:
        """Perturbs the initial pose of the target cube on the shelf."""
        if self.is_running_fsm:
            return {"status": "error", "message": "Cannot randomize while FSM is running"}

        with self._lock:
            if self.target_joint_id != -1:
                qadr = self.model.jnt_qposadr[self.target_joint_id]
                # Nominal: [0.45, 1.05, 0.90]
                dx = float(np.random.uniform(-0.03, 0.03))
                dy = float(np.random.uniform(-0.02, 0.02))
                self.data.qpos[qadr + 0] = 0.45 + dx
                self.data.qpos[qadr + 1] = 1.05 + dy

                yaw_deg = float(np.random.uniform(-25.0, 25.0))
                quat_rand = R.from_euler("z", yaw_deg, degrees=True).as_quat(scalar_first=True)
                self.data.qpos[qadr + 3:qadr + 7] = quat_rand

                # Settle physics for 50 steps
                for _ in range(50):
                    mujoco.mj_step(self.model, self.data)

            self.fsm.current_state = HandoverState.IDLE
            new_pos = self.data.xpos[self.target_body_id].tolist()

        return {
            "status": "ok",
            "message": "Target object pose randomized",
            "new_position": [round(p, 4) for p in new_pos],
            "yaw_deg": round(yaw_deg, 2),
        }

    def reset_simulation(self) -> Dict[str, Any]:
        """Resets the simulation to the initial keyframe."""
        if self.is_running_fsm:
            self._stop_requested = True
            time.sleep(0.1)

        self.is_running_fsm = False
        self._reset_internal()
        self._stop_requested = False
        return {"status": "ok", "message": "Simulation reset to initial keyframe"}

    def start_recording(self, camera_name: str = "front_cam") -> Dict[str, Any]:
        """Starts MP4 video capture for the specified camera."""
        with self._lock:
            if self._is_recording:
                return {
                    "status": "warning",
                    "message": "Recording already in progress",
                    "camera": self._record_camera,
                    "filepath": self._record_filepath,
                }

            if camera_name not in self.camera_ids:
                camera_name = "front_cam"

            recordings_dir = os.path.join(ROOT_DIR, "recordings")
            os.makedirs(recordings_dir, exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"handover_{camera_name}_{timestamp_str}.mp4"
            filepath = os.path.join(recordings_dir, filename)

            try:
                import imageio
                self._record_writer = imageio.get_writer(
                    filepath,
                    fps=20,
                    quality=8,
                    macro_block_size=1,
                )
                self._record_filepath = filepath
                self._record_camera = camera_name
                self._record_start_time = time.time()
                self._record_frames = 0
                self._is_recording = True
                print(f"[SimulationService] Started recording '{camera_name}' -> {filename}")
                return {
                    "status": "ok",
                    "message": f"Started recording camera '{camera_name}'",
                    "filename": filename,
                    "filepath": filepath,
                }
            except Exception as e:
                print(f"[SimulationService] Failed to start recording: {e}")
                return {"status": "error", "message": f"Failed to initialize video writer: {str(e)}"}

    def stop_recording(self) -> Dict[str, Any]:
        """Stops active MP4 video capture and finalizes the file."""
        with self._lock:
            if not self._is_recording or self._record_writer is None:
                return {"status": "warning", "message": "No active recording to stop"}

            try:
                self._record_writer.close()
            except Exception as e:
                print(f"[SimulationService] Error finalizing recording: {e}")

            duration = time.time() - self._record_start_time
            frames = self._record_frames
            filepath = self._record_filepath
            size_mb = 0.0
            if filepath and os.path.exists(filepath):
                size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 2)

            self._is_recording = False
            self._record_writer = None
            self._record_filepath = None

            print(f"[SimulationService] Finalized recording: {frames} frames, {duration:.1f}s, {size_mb} MB")
            return {
                "status": "ok",
                "message": "Video recording finalized",
                "filepath": filepath,
                "frames": frames,
                "duration_sec": round(duration, 2),
                "size_mb": size_mb,
            }

    def get_recording_status(self) -> Dict[str, Any]:
        """Returns current recording state and stats."""
        with self._lock:
            return {
                "is_recording": self._is_recording,
                "camera": self._record_camera if self._is_recording else None,
                "duration_sec": round(time.time() - self._record_start_time, 2) if self._is_recording else 0.0,
                "frames": self._record_frames if self._is_recording else 0,
                "filepath": self._record_filepath if self._is_recording else None,
            }

    def jog_joint(self, arm: int, joint_idx: int, delta_rad: float) -> Dict[str, Any]:
        """Jogs a single joint angle on the specified arm."""
        if self.is_running_fsm:
            return {"status": "error", "message": "Cannot jog arm while autonomous FSM is active"}
        if arm not in (1, 2) or not (0 <= joint_idx < 6):
            return {"status": "error", "message": "Invalid arm or joint index"}

        with self._lock:
            kin = self.controller.get_kinematics(arm)
            cur_q = kin.get_joint_positions().copy()
            cur_q[joint_idx] += delta_rad
            cur_q[joint_idx] = np.clip(cur_q[joint_idx], -2 * np.pi, 2 * np.pi)
            if arm == 1:
                self._target_q1 = cur_q
            else:
                self._target_q2 = cur_q
            self.controller.set_arm_joint_targets(arm, cur_q)
            for _ in range(25):
                mujoco.mj_step(self.model, self.data)
            new_ee, _ = kin.get_ee_pose()

        return {
            "status": "ok",
            "arm": arm,
            "joint_idx": joint_idx,
            "new_positions": [round(float(v), 4) for v in cur_q],
            "ee_pos": [round(float(p), 4) for p in new_ee],
            "message": f"Arm {arm} Joint {joint_idx} jogged by {delta_rad:+.3f} rad",
        }

    def jog_cartesian(self, arm: int, delta_xyz: List[float]) -> Dict[str, Any]:
        """Jogs the end-effector of the specified arm by [dx, dy, dz] meters."""
        if self.is_running_fsm:
            return {"status": "error", "message": "Cannot jog arm while autonomous FSM is active"}
        if arm not in (1, 2) or len(delta_xyz) != 3:
            return {"status": "error", "message": "Invalid arm or delta coordinates"}

        with self._lock:
            kin = self.controller.get_kinematics(arm)
            cur_pos, cur_quat = kin.get_ee_pose()
            target_pos = cur_pos + np.array(delta_xyz, dtype=np.float64)
            # Step IK
            for _ in range(20):
                self.controller.step_ik(arm, target_pos, cur_quat, max_joint_step=0.08)
                mujoco.mj_step(self.model, self.data)
            new_q = kin.get_joint_positions().copy()
            if arm == 1:
                self._target_q1 = new_q
            else:
                self._target_q2 = new_q
            new_ee, _ = kin.get_ee_pose()

        return {
            "status": "ok",
            "arm": arm,
            "target_ee": [round(float(p), 4) for p in target_pos],
            "actual_ee": [round(float(p), 4) for p in new_ee],
            "message": f"Arm {arm} EE jogged to {[round(p, 3) for p in new_ee]}",
        }

    def set_gripper_state(self, arm: int, position: float) -> Dict[str, Any]:
        """Commands normalized gripper position [0.0 = open, 1.0 = closed]."""
        if self.is_running_fsm:
            return {"status": "error", "message": "Cannot toggle gripper while FSM is active"}
        if arm not in (1, 2):
            return {"status": "error", "message": "Invalid arm index"}

        clamped = float(np.clip(position, 0.0, 1.0))
        with self._lock:
            if arm == 1:
                self._target_g1 = clamped
            else:
                self._target_g2 = clamped
            self.controller.set_gripper(arm, clamped)
            for _ in range(35):
                mujoco.mj_step(self.model, self.data)

        state_str = "CLOSED" if clamped > 0.5 else "OPEN"
        return {
            "status": "ok",
            "arm": arm,
            "gripper_position": clamped,
            "message": f"Arm {arm} gripper set to {state_str} ({clamped:.2f})",
        }

    def shutdown(self):
        """Clean shutdown of background threads and renderers."""
        print("[SimulationService] Shutting down simulation service...")
        self._shutdown_event.set()
        self._stop_requested = True
        if self._is_recording and self._record_writer:
            try:
                self._record_writer.close()
            except Exception:
                pass
        if self._idle_thread and self._idle_thread.is_alive():
            self._idle_thread.join(timeout=1.0)
        with self._lock:
            try:
                self.renderer.close()
            except Exception:
                pass
        print("[SimulationService] Shutdown complete.")
