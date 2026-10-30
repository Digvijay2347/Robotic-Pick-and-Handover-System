"""
Background Job Manager for Asynchronous Parallel Data Collection.

Orchestrates multi-worker collection processes (scripts/06_collect_demos_parallel.py)
with non-blocking execution, real-time stdout streaming, worker matrix tracking,
and graceful process termination.
"""

import os
import sys
import time
import re
import signal
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


class BackgroundJobManager:
    """Thread-safe manager for subprocess jobs (parallel collection, training, evaluation)."""

    _instance: Optional["BackgroundJobManager"] = None
    _init_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self.process: Optional[subprocess.Popen] = None
        self.is_running: bool = False
        self.config: Optional[Dict[str, Any]] = None
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.completed_episodes: int = 0
        self.target_episodes: int = 0
        self.exit_code: Optional[int] = None
        self.logs: List[str] = []
        self.max_logs: int = 200

        # Worker status matrix (up to 8 workers)
        self.workers: Dict[int, Dict[str, Any]] = {}
        self._reader_thread: Optional[threading.Thread] = None

    @classmethod
    def get_instance(cls) -> "BackgroundJobManager":
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def start_collection_job(self, config: CollectionJobConfig) -> Dict[str, Any]:
        """Launches scripts/06_collect_demos_parallel.py as an isolated background subprocess."""
        with self._lock:
            if self.is_running and self.process and self.process.poll() is None:
                return {
                    "status": "warning",
                    "message": "A collection job is already running",
                    "job": self.get_status_locked(),
                }

            # Initialize worker matrix for the configured worker count
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

            self.config = config.dict()
            self.is_running = True
            self.start_time = time.time()
            self.end_time = 0.0
            self.completed_episodes = 0
            self.target_episodes = config.num_episodes
            self.exit_code = None
            self.logs = [f"[{time.strftime('%X')}] [SYSTEM] Starting parallel collection job: {config.num_episodes} episodes across {config.num_workers} workers..."]

            # Construct command
            python_exe = sys.executable
            script_path = os.path.join(ROOT_DIR, "scripts", "06_collect_demos_parallel.py")

            cmd = [
                python_exe,
                script_path,
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
                self.process = subprocess.Popen(
                    cmd,
                    cwd=ROOT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )

                self._reader_thread = threading.Thread(
                    target=self._read_stdout_worker,
                    args=(self.process,),
                    name="CollectorStdoutReader",
                    daemon=True,
                )
                self._reader_thread.start()

                print(f"[JobManager] Spawned parallel collection PID {self.process.pid}: {' '.join(cmd)}")
                return {
                    "status": "ok",
                    "message": f"Parallel collection launched (PID {self.process.pid})",
                    "config": self.config,
                }
            except Exception as e:
                self.is_running = False
                err_msg = f"Failed to spawn collection subprocess: {str(e)}"
                self.logs.append(f"[{time.strftime('%X')}] [ERROR] {err_msg}")
                return {"status": "error", "message": err_msg}

    def _read_stdout_worker(self, process: subprocess.Popen):
        """Asynchronously reads and parses stdout lines to track worker progress."""
        worker_pattern = re.compile(r"\[Worker (\d+)\] (?:\[(ep_\d+)\] )?(.*)", re.IGNORECASE)
        master_progress_pattern = re.compile(r"Progress:\s*(\d+)/(\d+)", re.IGNORECASE)

        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            line_str = line.strip()
            if not line_str:
                continue

            with self._lock:
                # Append to logs
                t_str = time.strftime("%X")
                self.logs.append(f"[{t_str}] {line_str}")
                if len(self.logs) > self.max_logs:
                    self.logs.pop(0)

                # Parse Master progress updates
                prog_match = master_progress_pattern.search(line_str)
                if prog_match:
                    self.completed_episodes = int(prog_match.group(1))
                    self.target_episodes = int(prog_match.group(2))

                # Parse individual Worker telemetry lines
                w_match = worker_pattern.search(line_str)
                if w_match:
                    wid = int(w_match.group(1))
                    ep_str = w_match.group(2)
                    rest = w_match.group(3)

                    if wid in self.workers:
                        w_info = self.workers[wid]
                        w_info["status"] = "RUNNING"
                        w_info["last_update"] = time.time()
                        if ep_str:
                            try:
                                ep_num = int(ep_str.replace("ep_", ""))
                                w_info["current_episode"] = ep_num
                            except Exception:
                                pass

                        if "saved" in rest.lower() or "success" in rest.lower():
                            w_info["completed_episodes"] += 1
                            w_info["stage"] = "COMPLETED"
                        elif "failed" in rest.lower() or "aborted" in rest.lower():
                            w_info["stage"] = "RETRYING"
                        elif "step" in rest.lower():
                            w_info["stage"] = "COLLECTING"
                        else:
                            w_info["stage"] = rest[:30]

        process.stdout.close()
        rc = process.wait()
        with self._lock:
            self.is_running = False
            self.exit_code = rc
            self.end_time = time.time()
            t_str = time.strftime("%X")
            status_word = "SUCCESS" if rc == 0 else f"TERMINATED (Exit Code {rc})"
            self.logs.append(f"[{t_str}] [SYSTEM] Collection job finished: {status_word}")
            for wid in self.workers:
                self.workers[wid]["status"] = "IDLE" if rc == 0 else "STOPPED"
        print(f"[JobManager] Parallel collection process exited with code {rc}")

    def stop_collection_job(self) -> Dict[str, Any]:
        """Terminates the active collection subprocess."""
        with self._lock:
            if not self.is_running or not self.process or self.process.poll() is not None:
                return {"status": "warning", "message": "No active collection job to terminate"}

            try:
                # Windows taskkill tree kill
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(self.process.pid)])
                self.process.kill()
            except Exception:
                try:
                    self.process.terminate()
                except Exception:
                    pass

            self.is_running = False
            self.end_time = time.time()
            self.logs.append(f"[{time.strftime('%X')}] [SYSTEM] Collection job manually stopped by user.")
            for wid in self.workers:
                self.workers[wid]["status"] = "STOPPED"

            return {
                "status": "ok",
                "message": "Collection job terminated successfully",
                "completed_episodes": self.completed_episodes,
            }

    def get_status_locked(self) -> Dict[str, Any]:
        """Helper to serialize status dictionary while under lock."""
        duration = 0.0
        if self.start_time > 0:
            if self.is_running:
                duration = time.time() - self.start_time
            else:
                duration = (self.end_time or time.time()) - self.start_time

        pct = 0.0
        if self.target_episodes > 0:
            pct = round((self.completed_episodes / self.target_episodes) * 100, 1)

        return {
            "is_running": self.is_running,
            "duration_sec": round(duration, 1),
            "completed_episodes": self.completed_episodes,
            "target_episodes": self.target_episodes,
            "progress_percent": pct,
            "exit_code": self.exit_code,
            "config": self.config,
            "workers": list(self.workers.values()),
            "recent_logs": self.logs[-40:],
        }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return self.get_status_locked()
