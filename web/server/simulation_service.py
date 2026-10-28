"""
Simulation Service for Dual-Arm Robotic Pick-and-Handover System.

Encapsulates MuJoCo physics, multi-camera offscreen rendering, and the
16-stage Handover FSM in a thread-safe singleton service for the web platform.
"""

import os
import sys
import time
import io
import threading
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
import mujoco

# Ensure project root is in system path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fsm.handover_states import HandoverState, HandoverTelemetry, CycleSummary
from fsm.handover_fsm import HandoverFSM
from control.dual_arm_controller import DualArmController
from control.collision_checker import CollisionChecker
from perception import (
    capture_scene_pointcloud,
    SceneSegmenter,
    GeometricGraspPlanner,
    GraspFilter,
)


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
            self.fsm.q_home1 = self.fsm.kin1.get_joint_positions()
            self.fsm.q_home2 = self.fsm.kin2.get_joint_positions()
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
                    self.controller.set_arm_joint_targets(1, self.fsm.q_home1)
                    self.controller.set_arm_joint_targets(2, self.fsm.q_home2)
                    self.controller.set_gripper(1, 0.0)
                    self.controller.set_gripper(2, 0.0)
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

    def start_fsm_cycle(self) -> Dict[str, Any]:
        """Initiates autonomous pick-and-handover execution in worker thread."""
        if self.is_running_fsm:
            return {"status": "error", "message": "FSM cycle already in progress"}

        self._stop_requested = False
        self.is_running_fsm = True

        worker = threading.Thread(target=self._run_fsm_worker, name="FSMExecutionWorker", daemon=True)
        worker.start()
        return {"status": "ok", "message": "Autonomous handover cycle started"}

    def _run_fsm_worker(self):
        """Asynchronous execution worker for the 16-stage Handover FSM."""
        t_start_real = time.time()
        self._stop_requested = False
        print(f"[SimulationService] Starting FSM cycle #{self.cycle_count + 1}...")

        try:
            # Step 0: Reset and settle
            with self._lock:
                if self.model.nkey > 0:
                    mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
                mujoco.mj_forward(self.model, self.data)
                for _ in range(100):
                    mujoco.mj_step(self.model, self.data)
                self.fsm.q_home1 = self.fsm.kin1.get_joint_positions()
                self.fsm.q_home2 = self.fsm.kin2.get_joint_positions()
                self.fsm.sim_time = 0.0
                self.fsm.state_start_time = 0.0
                self.fsm.telemetry = HandoverTelemetry()
                self.fsm.dual_hold_confirmed = False

            if self._stop_requested:
                print("[SimulationService] Cycle cancelled during initialization.")
                return

            # Step 1: Perception scan
            print("[SimulationService] [1/3] Capturing 3D point cloud & segmenting objects...")
            with self._lock:
                self.fsm.transition_to(HandoverState.SCAN)
                pcd = capture_scene_pointcloud(self.model, self.data)
                segmenter = SceneSegmenter()
                clusters = segmenter.segment_objects(pcd)
                target = segmenter.find_target(clusters, color_hint="red")

            if target is None:
                print("[SimulationService] [FATAL] Target perception failed!")
                with self._lock:
                    self.fsm.transition_to(HandoverState.ERROR_RECOVERY)
                return

            if self._stop_requested:
                print("[SimulationService] Cycle cancelled after perception scan.")
                return

            # Step 2: Grasp planning
            print(f"[SimulationService] [2/3] Planning 6-DOF grasps for red box at [{target.centroid[0]:.3f}, {target.centroid[1]:.3f}, {target.centroid[2]:.3f}]...")
            with self._lock:
                self.fsm.transition_to(HandoverState.PLAN_GRASP)
                planner = GeometricGraspPlanner()
                raw_cands = planner.plan_grasps(target.pcd)
                gfilter = GraspFilter(self.model, self.data, arm_id=1, standoff_dist=0.15)
                verified_grasps = gfilter.filter_candidates(raw_cands)

            if not verified_grasps:
                print("[SimulationService] [FATAL] No valid grasps found!")
                with self._lock:
                    self.fsm.transition_to(HandoverState.ERROR_RECOVERY)
                return

            best_grasp = verified_grasps[0]
            with self._lock:
                ok_plan = self.fsm.plan_trajectories(best_grasp)

            if not ok_plan:
                print("[SimulationService] [FATAL] Trajectory resolution failed!")
                with self._lock:
                    self.fsm.transition_to(HandoverState.ERROR_RECOVERY)
                return

            if self._stop_requested:
                print("[SimulationService] Cycle cancelled after trajectory planning.")
                return

            # Step 3: Begin motion trajectory execution
            print("[SimulationService] [3/3] Trajectories generated! Commencing motion -> APPROACH_1")
            with self._lock:
                self.fsm.transition_to(HandoverState.APPROACH_1, duration=2.0)

            batch_size = 5
            is_finished = False
            last_logged_state = HandoverState.APPROACH_1

            while not self._stop_requested and not is_finished:
                t_loop = time.perf_counter()
                with self._lock:
                    for _ in range(batch_size):
                        curr_state, is_finished = self.fsm.step()
                        if curr_state != last_logged_state:
                            print(f"[SimulationService] FSM Transition: {last_logged_state.name} -> {curr_state.name}")
                            last_logged_state = curr_state
                        if is_finished or self._stop_requested:
                            break

                # Pace execution to match target playback speed
                elapsed = time.perf_counter() - t_loop
                target_dt = (self.model.opt.timestep * batch_size) / max(0.1, self.playback_speed)
                sleep_t = target_dt - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

            if self._stop_requested:
                print("[SimulationService] Emergency stop received. Cancelling cycle.")
                with self._lock:
                    self.fsm.transition_to(HandoverState.IDLE)
                return

            # Post-completion settling
            if is_finished and self.fsm.current_state == HandoverState.COMPLETED:
                with self._lock:
                    for _ in range(150):
                        self.controller.set_arm_joint_targets(1, self.fsm.q_home1)
                        self.controller.set_arm_joint_targets(2, self.fsm.q_home2)
                        self.controller.set_gripper(1, 0.0)
                        self.controller.set_gripper(2, 0.0)
                        mujoco.mj_step(self.model, self.data)

                self.cycle_count += 1
                duration = time.time() - t_start_real
                print(f"[SimulationService] Cycle #{self.cycle_count} finished successfully in {duration:.2f}s!")
                self.last_cycle_summary = {
                    "cycle_index": self.cycle_count,
                    "success": True,
                    "duration_sec": round(duration, 2),
                    "dual_hold_confirmed": True,
                }

        except Exception as e:
            print(f"[SimulationService] Unexpected error in FSM worker: {e}")
            with self._lock:
                self.fsm.transition_to(HandoverState.ERROR_RECOVERY)
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

    def shutdown(self):
        """Clean shutdown of background threads and renderers."""
        print("[SimulationService] Shutting down simulation service...")
        self._shutdown_event.set()
        self._stop_requested = True
        if self._idle_thread and self._idle_thread.is_alive():
            self._idle_thread.join(timeout=1.0)
        with self._lock:
            try:
                self.renderer.close()
            except Exception:
                pass
        print("[SimulationService] Shutdown complete.")
