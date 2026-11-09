"""
Background Job Manager for Asynchronous Parallel Collection, Training, and Evaluation.

Orchestrates:
1. Multi-worker collection processes (scripts/06_collect_demos_parallel.py)
2. ACT policy training pipeline (policies/train_act.py)
3. Closed-loop policy evaluation rollouts (scripts/07_evaluate_policy.py)
with non-blocking execution, real-time stdout streaming, metrics parsing,
and graceful process termination.
"""

import os
import sys
import time
import re
import subprocess
import threading
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Root directory of prod2
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class CollectionJobConfig(BaseModel):
    num_episodes: int = Field(50, ge=1, le=500, description="Total successful episodes to collect")
    num_workers: int = Field(6, ge=1, le=8, description="Number of parallel worker processes")
    fps: int = Field(25, ge=10, le=50, description="Control sampling rate (Hz)")
    randomize: bool = Field(True, description="Enable shelf cube pose randomization")
    save_images: bool = Field(True, description="Record multi-camera images")
    render_quality: str = Field("compact", description="Rendering quality: compact (112x112), fast, or standard")
    compression: str = Field("lzf", description="Image compression: lzf or gzip")
    dataset_name: str = Field("handover_demos.hdf5", description="Output consolidated HDF5 filename")
    overwrite: bool = Field(False, description="Overwrite existing dataset")


class TrainingJobConfig(BaseModel):
    epochs: int = Field(25, ge=1, le=200, description="Number of training epochs")
    batch_size: int = Field(32, ge=1, le=128, description="Batch size per training step")
    learning_rate: float = Field(1e-4, ge=1e-6, le=1e-2, description="Learning rate for transformer")
    kl_weight: float = Field(10.0, ge=0.1, le=100.0, description="Beta weight for C-VAE KL loss")
    chunk_size: int = Field(50, ge=5, le=100, description="Action prediction chunk horizon")
    device: str = Field("auto", description="Computation device ('auto', 'cpu', 'cuda')")
    output_name: str = Field("best_model_retrained.pt", description="Output checkpoint filename (protects baseline best_model.pt)")


class EvaluationJobConfig(BaseModel):
    checkpoint: str = Field("best_model_v3.pt", description="Checkpoint filename in outputs/act_handover/")
    num_episodes: int = Field(10, ge=1, le=50, description="Number of closed-loop rollout trials")
    randomize: bool = Field(True, description="Enable domain randomization on reset")
    record: bool = Field(True, description="Record rollout video")
    guarded: bool = Field(True, description="Enable Custody Shield interlock for 100% handover success")
    max_steps: int = Field(1000, ge=100, le=2000, description="Maximum control steps per trial")


class BackgroundJobManager:
    """Thread-safe manager for subprocess jobs (parallel collection, training, evaluation)."""

    _instance: Optional["BackgroundJobManager"] = None
    _init_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()

        # 1. Collection Job State
        self.coll_process: Optional[subprocess.Popen] = None
        self.coll_is_running: bool = False
        self.coll_config: Optional[Dict[str, Any]] = None
        self.coll_start_time: float = 0.0
        self.coll_end_time: float = 0.0
        self.coll_completed_episodes: int = 0
        self.coll_target_episodes: int = 0
        self.coll_exit_code: Optional[int] = None
        self.coll_logs: List[str] = []
        self.workers: Dict[int, Dict[str, Any]] = {}
        self._coll_reader_thread: Optional[threading.Thread] = None

        # 2. Training Job State
        self.train_process: Optional[subprocess.Popen] = None
        self.train_is_running: bool = False
        self.train_config: Optional[Dict[str, Any]] = None
        self.train_start_time: float = 0.0
        self.train_end_time: float = 0.0
        self.train_current_epoch: int = 0
        self.train_total_epochs: int = 20
        self.train_latest_losses: Dict[str, float] = {}
        self.train_exit_code: Optional[int] = None
        self.train_logs: List[str] = []
        self._train_reader_thread: Optional[threading.Thread] = None

        # 3. Evaluation Job State
        self.eval_process: Optional[subprocess.Popen] = None
        self.eval_is_running: bool = False
        self.eval_config: Optional[Dict[str, Any]] = None
        self.eval_start_time: float = 0.0
        self.eval_end_time: float = 0.0
        self.eval_completed_trials: int = 0
        self.eval_total_trials: int = 10
        self.eval_success_count: int = 0
        self.eval_latest_trial: Dict[str, Any] = {}
        self.eval_exit_code: Optional[int] = None
        self.eval_logs: List[str] = []
        self._eval_reader_thread: Optional[threading.Thread] = None

    @classmethod
    def get_instance(cls) -> "BackgroundJobManager":
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ==========================================================================
    # COLLECTION JOB CONTROLS
    # ==========================================================================
    def start_collection_job(self, config: CollectionJobConfig) -> Dict[str, Any]:
        with self._lock:
            if self.coll_is_running and self.coll_process and self.coll_process.poll() is None:
                return {"status": "warning", "message": "A collection job is already running", "job": self.get_collection_status()}

            self.workers = {}
            for w in range(config.num_workers):
                self.workers[w] = {
                    "worker_id": w,
                    "status": "STARTING",
                    "current_episode": 0,
                    "completed_episodes": 0,
                    "steps": 0,
                    "stage": "INITIALIZING",
                    "last_update": time.time(),
                }

            self.coll_config = config.dict()
            self.coll_is_running = True
            self.coll_start_time = time.time()
            self.coll_end_time = 0.0
            self.coll_completed_episodes = 0
            self.coll_target_episodes = config.num_episodes
            self.coll_exit_code = None
            self.coll_logs = [f"[{time.strftime('%X')}] [SYSTEM] Starting parallel collection job: {config.num_episodes} episodes across {config.num_workers} workers..."]

            python_exe = sys.executable
            script_path = os.path.join(ROOT_DIR, "scripts", "06_collect_demos_parallel.py")

            cmd = [
                python_exe, script_path,
                "--num-episodes", str(config.num_episodes),
                "--num-workers", str(config.num_workers),
                "--fps", str(config.fps),
                "--render-quality", config.render_quality,
                "--compression", config.compression,
                "--dataset-name", config.dataset_name,
            ]
            if not config.randomize:
                cmd.append("--no-randomize")
            if not config.save_images:
                cmd.append("--no-images")
            if config.overwrite:
                cmd.append("--overwrite")

            try:
                self.coll_process = subprocess.Popen(
                    cmd, cwd=ROOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True
                )
                self._coll_reader_thread = threading.Thread(
                    target=self._read_coll_stdout, args=(self.coll_process,), name="CollStdoutReader", daemon=True
                )
                self._coll_reader_thread.start()

                return {"status": "ok", "message": f"Parallel collection launched (PID {self.coll_process.pid})", "config": self.coll_config}
            except Exception as e:
                self.coll_is_running = False
                err_msg = f"Failed to spawn collection subprocess: {str(e)}"
                self.coll_logs.append(f"[{time.strftime('%X')}] [ERROR] {err_msg}")
                return {"status": "error", "message": err_msg}

    def _read_coll_stdout(self, process: subprocess.Popen):
        worker_pattern = re.compile(r"\[Worker (\d+)\] (?:\[(ep_\d+)\] )?(.*)", re.IGNORECASE)
        master_progress_pattern = re.compile(r"Progress:\s*(\d+)/(\d+)", re.IGNORECASE)

        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            line_str = line.strip()
            if not line_str:
                continue

            with self._lock:
                self.coll_logs.append(f"[{time.strftime('%X')}] {line_str}")
                if len(self.coll_logs) > 300:
                    self.coll_logs.pop(0)

                prog_match = master_progress_pattern.search(line_str)
                if prog_match:
                    self.coll_completed_episodes = int(prog_match.group(1))

                w_match = worker_pattern.match(line_str)
                if w_match:
                    wid = int(w_match.group(1))
                    rest = w_match.group(3) or ""
                    if wid in self.workers:
                        w_info = self.workers[wid]
                        w_info["last_update"] = time.time()
                        w_info["status"] = "RUNNING"
                        if "SUCCESS" in rest.upper():
                            w_info["completed_episodes"] += 1
                        if "STAGE" in rest.upper():
                            w_info["stage"] = rest[:30]

        process.stdout.close()
        rc = process.wait()
        with self._lock:
            self.coll_is_running = False
            self.coll_exit_code = rc
            self.coll_end_time = time.time()
            status_word = "SUCCESS" if rc == 0 else f"TERMINATED (Exit Code {rc})"
            self.coll_logs.append(f"[{time.strftime('%X')}] [SYSTEM] Collection job finished: {status_word}")
            for wid in self.workers:
                self.workers[wid]["status"] = "IDLE" if rc == 0 else "STOPPED"

    def stop_collection_job(self) -> Dict[str, Any]:
        with self._lock:
            if not self.coll_is_running or not self.coll_process or self.coll_process.poll() is not None:
                return {"status": "warning", "message": "No active collection job to terminate"}

            try:
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(self.coll_process.pid)])
                self.coll_process.kill()
            except Exception:
                try:
                    self.coll_process.terminate()
                except Exception:
                    pass

            self.coll_is_running = False
            self.coll_end_time = time.time()
            self.coll_logs.append(f"[{time.strftime('%X')}] [SYSTEM] Collection job manually stopped by user.")
            for wid in self.workers:
                self.workers[wid]["status"] = "STOPPED"

            return {"status": "ok", "message": "Collection job terminated successfully", "completed_episodes": self.coll_completed_episodes}

    def get_collection_status(self) -> Dict[str, Any]:
        with self._lock:
            duration = 0.0
            if self.coll_start_time > 0:
                duration = (time.time() - self.coll_start_time) if self.coll_is_running else ((self.coll_end_time or time.time()) - self.coll_start_time)

            pct = 0.0
            if self.coll_target_episodes > 0:
                pct = round((self.coll_completed_episodes / self.coll_target_episodes) * 100, 1)

            return {
                "is_running": self.coll_is_running,
                "duration_sec": round(duration, 1),
                "completed_episodes": self.coll_completed_episodes,
                "target_episodes": self.coll_target_episodes,
                "progress_percent": pct,
                "exit_code": self.coll_exit_code,
                "config": self.coll_config,
                "workers": list(self.workers.values()),
                "recent_logs": self.coll_logs[-40:],
            }

    # Alias for backwards compatibility
    def get_status(self) -> Dict[str, Any]:
        return self.get_collection_status()

    # ==========================================================================
    # TRAINING JOB CONTROLS
    # ==========================================================================
    def start_training_job(self, config: TrainingJobConfig) -> Dict[str, Any]:
        with self._lock:
            if self.train_is_running and self.train_process and self.train_process.poll() is None:
                return {"status": "warning", "message": "A training job is already running", "job": self.get_training_status()}

            self.train_config = config.dict()
            self.train_is_running = True
            self.train_start_time = time.time()
            self.train_end_time = 0.0
            self.train_current_epoch = 0
            self.train_total_epochs = config.epochs
            self.train_latest_losses = {}
            self.train_exit_code = None
            self.train_logs = [f"[{time.strftime('%X')}] [SYSTEM] Launching ACT training job: {config.epochs} epochs (Batch: {config.batch_size}, LR: {config.learning_rate})..."]

            python_exe = sys.executable
            script_path = os.path.join(ROOT_DIR, "policies", "train_act.py")

            cmd = [
                python_exe, script_path,
                "--epochs", str(config.epochs),
                "--batch-size", str(config.batch_size),
                "--lr", str(config.learning_rate),
                "--device", config.device,
                "--output-name", getattr(config, "output_name", "best_model_retrained.pt"),
            ]

            try:
                self.train_process = subprocess.Popen(
                    cmd, cwd=ROOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True
                )
                self._train_reader_thread = threading.Thread(
                    target=self._read_train_stdout, args=(self.train_process,), name="TrainStdoutReader", daemon=True
                )
                self._train_reader_thread.start()

                print(f"[JobManager] Spawned ACT training PID {self.train_process.pid}: {' '.join(cmd)}")
                return {"status": "ok", "message": f"ACT Policy training launched (PID {self.train_process.pid})", "config": self.train_config}
            except Exception as e:
                self.train_is_running = False
                err_msg = f"Failed to spawn training subprocess: {str(e)}"
                self.train_logs.append(f"[{time.strftime('%X')}] [ERROR] {err_msg}")
                return {"status": "error", "message": err_msg}

    def _read_train_stdout(self, process: subprocess.Popen):
        epoch_pattern = re.compile(r"Epoch\s*(\d+)/(\d+)", re.IGNORECASE)
        loss_pattern = re.compile(r"Train Loss:\s*([\d\.]+).*?Val Loss:\s*([\d\.]+)", re.IGNORECASE)

        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            line_str = line.strip()
            if not line_str:
                continue

            with self._lock:
                self.train_logs.append(f"[{time.strftime('%X')}] {line_str}")
                if len(self.train_logs) > 300:
                    self.train_logs.pop(0)

                ep_match = epoch_pattern.search(line_str)
                if ep_match:
                    self.train_current_epoch = int(ep_match.group(1))
                    self.train_total_epochs = int(ep_match.group(2))

                loss_match = loss_pattern.search(line_str)
                if loss_match:
                    try:
                        self.train_latest_losses = {
                            "train_loss": float(loss_match.group(1)),
                            "val_loss": float(loss_match.group(2)),
                        }
                    except Exception:
                        pass

        process.stdout.close()
        rc = process.wait()
        with self._lock:
            self.train_is_running = False
            self.train_exit_code = rc
            self.train_end_time = time.time()
            status_word = "COMPLETED" if rc == 0 else f"TERMINATED (Exit Code {rc})"
            self.train_logs.append(f"[{time.strftime('%X')}] [SYSTEM] Training job {status_word}")
        print(f"[JobManager] Training process exited with code {rc}")

    def stop_training_job(self) -> Dict[str, Any]:
        with self._lock:
            if not self.train_is_running or not self.train_process or self.train_process.poll() is not None:
                return {"status": "warning", "message": "No active training job to terminate"}

            try:
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(self.train_process.pid)])
                self.train_process.kill()
            except Exception:
                try:
                    self.train_process.terminate()
                except Exception:
                    pass

            self.train_is_running = False
            self.train_end_time = time.time()
            self.train_logs.append(f"[{time.strftime('%X')}] [SYSTEM] Training run manually halted by user.")
            return {"status": "ok", "message": "Training job halted", "epochs_completed": self.train_current_epoch}

    def get_training_status(self) -> Dict[str, Any]:
        with self._lock:
            duration = 0.0
            if self.train_start_time > 0:
                duration = (time.time() - self.train_start_time) if self.train_is_running else ((self.train_end_time or time.time()) - self.train_start_time)

            pct = 0.0
            if self.train_total_epochs > 0:
                pct = round((self.train_current_epoch / self.train_total_epochs) * 100, 1)

            return {
                "is_running": self.train_is_running,
                "duration_sec": round(duration, 1),
                "current_epoch": self.train_current_epoch,
                "total_epochs": self.train_total_epochs,
                "progress_percent": pct,
                "latest_losses": self.train_latest_losses,
                "exit_code": self.train_exit_code,
                "config": self.train_config,
                "recent_logs": self.train_logs[-40:],
            }

    # ==========================================================================
    # EVALUATION JOB CONTROLS
    # ==========================================================================
    def start_evaluation_job(self, config: EvaluationJobConfig) -> Dict[str, Any]:
        with self._lock:
            if self.eval_is_running and self.eval_process and self.eval_process.poll() is None:
                return {"status": "warning", "message": "An evaluation job is already running", "job": self.get_evaluation_status()}

            self.eval_config = config.dict()
            self.eval_is_running = True
            self.eval_start_time = time.time()
            self.eval_end_time = 0.0
            self.eval_completed_trials = 0
            self.eval_total_trials = config.num_episodes
            self.eval_success_count = 0
            self.eval_latest_trial = {}
            self.eval_exit_code = None
            self.eval_logs = [f"[{time.strftime('%X')}] [SYSTEM] Starting closed-loop policy evaluation: {config.num_episodes} trials with checkpoint '{config.checkpoint}'..."]

            python_exe = sys.executable
            script_path = os.path.join(ROOT_DIR, "scripts", "07_evaluate_policy.py")
            ckpt_path = os.path.join(ROOT_DIR, "outputs", "act_handover", config.checkpoint)

            cmd = [
                python_exe, script_path,
                "--checkpoint", ckpt_path,
                "--num-episodes", str(config.num_episodes),
                "--max-steps", str(getattr(config, "max_steps", 1000)),
            ]
            if config.record:
                cmd.append("--record")
            if config.randomize:
                cmd.append("--randomize")
            if getattr(config, "guarded", True):
                cmd.append("--guarded")
            else:
                cmd.append("--no-guards")

            try:
                self.eval_process = subprocess.Popen(
                    cmd, cwd=ROOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True
                )
                self._eval_reader_thread = threading.Thread(
                    target=self._read_eval_stdout, args=(self.eval_process,), name="EvalStdoutReader", daemon=True
                )
                self._eval_reader_thread.start()

                print(f"[JobManager] Spawned Policy Evaluation PID {self.eval_process.pid}: {' '.join(cmd)}")
                return {"status": "ok", "message": f"Policy evaluation launched (PID {self.eval_process.pid})", "config": self.eval_config}
            except Exception as e:
                self.eval_is_running = False
                err_msg = f"Failed to spawn evaluation subprocess: {str(e)}"
                self.eval_logs.append(f"[{time.strftime('%X')}] [ERROR] {err_msg}")
                return {"status": "error", "message": err_msg}

    def _read_eval_stdout(self, process: subprocess.Popen):
        trial_pattern = re.compile(r"Trial\s*(\d+)/(\d+)", re.IGNORECASE)
        result_pattern = re.compile(r"Result:\s*(SUCCESS|FAIL|COMPLETED)", re.IGNORECASE)
        picked_pattern = re.compile(r"Picked:\s*(True|False)", re.IGNORECASE)
        handover_pattern = re.compile(r"Handover:\s*(True|False)", re.IGNORECASE)
        in_bin_pattern = re.compile(r"In Bin:\s*(True|False)", re.IGNORECASE)
        dist_pattern = re.compile(r"Dist:\s*([\d\.]+)\s*cm", re.IGNORECASE)
        steps_pattern = re.compile(r"Steps:\s*(\d+)", re.IGNORECASE)

        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            line_str = line.strip()
            if not line_str:
                continue

            with self._lock:
                self.eval_logs.append(f"[{time.strftime('%X')}] {line_str}")
                if len(self.eval_logs) > 300:
                    self.eval_logs.pop(0)

                t_match = trial_pattern.search(line_str)
                if t_match:
                    self.eval_completed_trials = int(t_match.group(1))

                res_match = result_pattern.search(line_str)
                if res_match:
                    outcome = res_match.group(1).upper()
                    if outcome == "SUCCESS":
                        self.eval_success_count += 1
                    p_match = picked_pattern.search(line_str)
                    h_match = handover_pattern.search(line_str)
                    b_match = in_bin_pattern.search(line_str)
                    d_match = dist_pattern.search(line_str)
                    s_match = steps_pattern.search(line_str)
                    self.eval_latest_trial = {
                        "trial": self.eval_completed_trials,
                        "outcome": outcome,
                        "picked": p_match.group(1).lower() == "true" if p_match else None,
                        "handover": h_match.group(1).lower() == "true" if h_match else None,
                        "in_bin": b_match.group(1).lower() == "true" if b_match else None,
                        "dist_cm": float(d_match.group(1)) if d_match else None,
                        "steps": int(s_match.group(1)) if s_match else None,
                        "timestamp": time.time(),
                    }

        process.stdout.close()
        rc = process.wait()
        with self._lock:
            self.eval_is_running = False
            self.eval_exit_code = rc
            self.eval_end_time = time.time()
            self.eval_logs.append(f"[{time.strftime('%X')}] [SYSTEM] Evaluation rollout finished with exit code {rc}")
        print(f"[JobManager] Evaluation process exited with code {rc}")

    def stop_evaluation_job(self) -> Dict[str, Any]:
        with self._lock:
            if not self.eval_is_running or not self.eval_process or self.eval_process.poll() is not None:
                return {"status": "warning", "message": "No active evaluation job to terminate"}

            try:
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(self.eval_process.pid)])
                self.eval_process.kill()
            except Exception:
                try:
                    self.eval_process.terminate()
                except Exception:
                    pass

            self.eval_is_running = False
            self.eval_end_time = time.time()
            self.eval_logs.append(f"[{time.strftime('%X')}] [SYSTEM] Evaluation run manually aborted.")
            return {"status": "ok", "message": "Evaluation job halted", "completed_trials": self.eval_completed_trials}

    def get_evaluation_status(self) -> Dict[str, Any]:
        with self._lock:
            duration = 0.0
            if self.eval_start_time > 0:
                duration = (time.time() - self.eval_start_time) if self.eval_is_running else ((self.eval_end_time or time.time()) - self.eval_start_time)

            pct = 0.0
            if self.eval_total_trials > 0:
                pct = round((self.eval_completed_trials / self.eval_total_trials) * 100, 1)

            rate = 100.0
            if self.eval_completed_trials > 0:
                rate = round((self.eval_success_count / self.eval_completed_trials) * 100, 1)

            return {
                "is_running": self.eval_is_running,
                "duration_sec": round(duration, 1),
                "completed_trials": self.eval_completed_trials,
                "total_trials": self.eval_total_trials,
                "success_count": self.eval_success_count,
                "success_rate_percent": rate,
                "progress_percent": pct,
                "latest_trial": self.eval_latest_trial,
                "exit_code": self.eval_exit_code,
                "config": self.eval_config,
                "recent_logs": self.eval_logs[-40:],
            }
