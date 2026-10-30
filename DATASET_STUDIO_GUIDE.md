# Bimanual UR10e Robotics Platform: Dataset Studio Usage Guide

> **Phase 3 Module Documentation**  
> Comprehensive operating manual for data collection orchestration, demonstration catalog inspection, multi-worker telemetry monitoring, and synchronized video playback.

---

## 1. System Overview & Purpose

The **Dataset Studio** is the data curation and orchestration engine of the Bimanual UR10e Pick-and-Handover platform. It provides a visual interface for:
- Monitoring and managing multi-process parallel data collection (`scripts/06_collect_demos_parallel.py`).
- Inspecting consolidated HDF5 demonstration datasets (`data/handover_demos.hdf5`).
- Verifying the quality of all 50 recorded trajectories (dual-gripper custody overlap, terminal placement accuracy, cycle times).
- Visualizing the 14-DOF action distribution across both UR10e manipulators and Robotiq 2F-85 grippers.
- Playing synchronized, multi-angle demonstration preview videos (`recordings/`).

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 DATASET STUDIO ARCHITECTURE                                 │
├──────────────────────────────┬───────────────────────────────┬──────────────────────────────┤
│ 1. Obsidian Control Panel    │ 2. Live Parallel Matrix       │ 3. Dataset Analytics         │
│ • Episode & Worker Sliders   │ • 6 Worker Threads (W1 – W6)  │ • Storage Radial Arc (1 GB)  │
│ • Domain Noise Toggles       │ • Progress Tracks & Stages    │ • 14-DOF Action Distribution │
│ • Multi-Camera Resolution    │ • Throughput & Step Latency   │ • Force Waves (FX, FY, FZ)   │
│ • Start / Cancel Job Buttons │ • Live Stdout Stream Console  │ • Synchronized Eye-In-Hand   │
├──────────────────────────────┴───────────────────────────────┴──────────────────────────────┤
│ 4. Recorded Demonstrations Catalog Table   │ 5. Synchronized Video Inspector                │
│ • 50 Filterable Demonstration Episodes     │ • Hardware-Accelerated MP4 Video Player        │
│ • Status, Steps, Duration, Dual-Hold Flags │ • 0.5x, 1.0x, 1.5x, 2.0x Playback Speeds       │
│ • Sub-2cm Bin Placement Verification Readout│ • Decimated Trajectory Telemetry Slice Chips  │
└────────────────────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 2. Launching & Accessing Dataset Studio

### 2.1 Starting the Turnkey Platform
From the repository root, launch the platform server:
```powershell
.\.venv\Scripts\python.exe launch_platform.py
```

### 2.2 Navigation
1. Open your web browser and navigate to:
   - **Local Dashboard URL**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
   - **Direct Dataset Studio Route**: [http://127.0.0.1:8000/#dataset-studio](http://127.0.0.1:8000/#dataset-studio)
2. In the top navigation bar, click the **📊 Dataset Studio (50 Demos)** tab to transition from Mission Control to Dataset Studio without disconnecting live telemetry.

> [!TIP]
> If you recently updated the codebase, press **Ctrl + F5** (or **Shift + Reload**) to bypass the browser cache and load the latest Obsidian styling assets (`style.css?v=3.2`).

---

## 3. Top Master Row: Data Collection & Analytics

The top row of the studio is divided into three synchronized panels:

```
┌───────────────────────┬───────────────────────────────────────┬──────────────────────────────┐
│  OBSIDIAN CONTROL     │      LIVE PARALLEL EXECUTION MATRIX   │      DATASET ANALYTICS       │
│        PANEL          │                                       │                              │
│  [Job Parameters]     │  [Worker 1]    [Worker 2]   [Worker 3]│  [Action Distribution Chart] │
│  [Sliders & Toggles]  │  [Worker 4]    [Worker 5]   [Worker 6]│  [Force Oscilloscope FX/Y/Z] │
│  [START COLLECTION]   │  [Stdout Live Terminal Stream]        │  [Storage Radial Arc Gauge]  │
└───────────────────────┴───────────────────────────────────────┴──────────────────────────────┘
```

### 3.1 Panel 1: Obsidian Control Panel (Job Dispatcher)
Allows you to dispatch new multi-worker data collection jobs:

- **Collection Job Title**: Identifies the dataset batch (e.g., `handover_demos_batch_02`).
- **EPISODES TO COLLECT**: Target number of successful demonstrations to record (range: 1 to 50, default: `10`). The percentage goal is displayed dynamically below.
- **WORKER THREADS**: Number of isolated MuJoCo worker processes to spawn in parallel (range: 1 to 6, default: `4`).
- **DOMAIN RANDOMIZATION Toggle**: When enabled, adds stochastic perturbations to:
  - Shelf red cube pose ($\pm 3\text{ cm}$ in X/Y, $\pm 25^\circ$ yaw).
  - Surface friction coefficients ($0.8\times$ to $1.2\times$).
  - Light intensity and camera position jitter.
- **MULTI-CAMERA RECORDING Toggle**: Records synchronized camera observation tensors from `top`, `wrist1`, and `wrist2` at $112\times 112$ resolution.
- **SAMPLING RATE Slider**: Sets the control loop frequency (10 Hz to 50 Hz, default: `25 Hz`).
- **START COLLECTION Button**: Spawns the collection subprocess engine in the background and activates live matrix polling.
- **STOP COLLECTION Button**: Gracefully aborts running subprocesses and saves all completed episodes to disk.

---

### 3.2 Panel 2: Live Parallel Execution Matrix
Provides a real-time view into the 6 parallel worker processes:

#### Master Progress Tracker
Located at the top of the matrix, the progress bar displays total completed demonstrations against the target count:
$$\text{Progress \%} = \left(\frac{\text{Completed Episodes}}{\text{Target Episodes}}\right) \times 100\%$$

#### 6 Worker Cards (Worker #1 to Worker #6)
Each worker card represents an isolated simulation environment and displays:
1. **Status Badge**:
   - `ACTIVE` (Green neon glow): Currently stepping physics and recording.
   - `RUNNING` (Cyan glow): Executing trajectory transit.
   - `IDLE` (Gray): Standby waiting for assignment.
   - `COMPLETED` (Bright green): Reached assigned episode quota.
2. **Progress Track & Percentage**: Visual fill bar showing worker episode completion.
3. **Episode Progress**: Current episode index (e.g., `Ep. 117/150`).
4. **FSM Stage Badge**: Active state machine phase (`GRASP_1`, `TRANSFER_1`, `DUAL_GRASP`, `TRANSPORT_2`, `PLACE_2`).
5. **Robotic Arm Vector Glyph**: Illuminated silhouette of the UR10e arm with glowing joints that pulse during active execution.
6. **Throughput & Latency**: Real-time step metrics (e.g., `Throughput: 3.2 Hz`, `Latency: 340 ms/step`).
7. **Success / Failure Tally**: Counters verifying zero dropped objects or kinematic collisions.

#### Subprocess Stdout Console
Streams raw console output from worker subprocesses with timestamp tags, warning highlights, and auto-scrolling.

---

### 3.3 Panel 3: Dataset Analytics
Visualizes dataset-wide statistical properties:

1. **Storage Radial Arc Gauge**:
   - Circular SVG gauge displaying disk usage (`531.97 MB / 1 GB`).
   - Episode count (`50 Demonstrations`), transition frames count (`32,450`), and `STATUS: OPERATIONAL` indicator.
2. **Action Distribution Spectrum**:
   - Vertical gradient bar chart showing normalized action limits for `Grasp`, `Move`, `Place`, `Reach`, `Pan`, and `Lift`.
3. **Contact Force Oscilloscope**:
   - Waveform graphs displaying force components ($F_X$, $F_Y$, $F_Z$) during the dual-gripper custody handover phase.
4. **Synchronized Eye-In-Hand Camera**:
   - Live snapshot card displaying the overhead camera perspective.

---

## 4. Bottom Master Row: Catalog & Video Player

The bottom row enables detailed per-episode trajectory analysis and synchronized video playback:

```
┌────────────────────────────────────────────────────────┬─────────────────────────────────────┐
│       RECORDED DEMONSTRATIONS CATALOG TABLE            │     SYNCHRONIZED VIDEO PREVIEW      │
│  [Search: #0049]  [Filter: All | Success | Dual Hold]  │  [Clip Selector: phase6_demo_049]   │
│  ┌────────┬─────────┬──────────┬───────┬──────┬──────┐ │  ┌────────────────────────────────┐ │
│  │ ID #   │ STATUS  │ DURATION │ STEPS │ OBJ  │ DUAL │ │  │  HTML5 Video Player Viewport   │ │
│  ├────────┼─────────┼──────────┼───────┼──────┼──────┤ │  └────────────────────────────────┘ │
│  │ #0049  │ SUCCESS │ 25.96s   │ 649   │ Cube │ CONF │ │  [Speed: 0.5x | 1.0x | 1.5x | 2x] │
│  └────────┴─────────┴──────────┴───────┴──────┴──────┘ │  [Telemetry: 1.58cm, 649 steps]     │
└────────────────────────────────────────────────────────┴─────────────────────────────────────┘
```

### 4.1 50-Episode Catalog Data Table
- **Search Bar**: Type any episode ID (e.g., `#0049` or `demo_12`) or worker ID to filter instantly.
- **Filter Chips**: Quick buttons to filter by `All (50)`, `Success (50)`, or `Dual Hold`.
- **Columns**:
  - `ID #`: Episode identifier formatted as `#0000` to `#0049`.
  - `STATUS`: Verified outcome (`SUCCESS` green badge).
  - `DURATION`: Total execution time in seconds (average: $25.96\text{ s}$).
  - `STEPS`: Control steps recorded at 25 Hz (average: $649\text{ steps}$).
  - `OBJECT`: Manipulated asset (`Red Block`).
  - `AGENT`: Controller configuration (`UR10e Dual`).
  - `TIME`: Recording timestamp.
  - `ACTIONS`: Action dimension (`14-DOF`).
- **Interactive Row Selection**: Click any row in the table to highlight it and immediately load its trajectory telemetry and synchronized video into the preview panel.

---

### 4.2 Synchronized Video Inspector
- **Clip Dropdown**: Select between different recorded multi-camera demonstration videos:
  - `phase6_demo_049.mp4`: High-resolution composite view of episode 49.
  - `phase6_randomized_demo_batch.mp4`: Multi-episode sequence with shelf position randomization.
  - `phase6_demo_preview.mp4`: Benchmark validation recording.
- **Playback Speed Selector**:
  - `0.5x`: Slow-motion playback to verify micro-slip dynamics during the inter-arm transfer phase (`DUAL_GRASP`).
  - `1.0x`: Real-time playback speed (25 FPS).
  - `1.5x` / `2.0x`: Accelerated review of trajectory paths.
- **Selected Episode Telemetry Chips**:
  - `Trajectory Steps`: Exact step count and duration.
  - `Dual Gripper Hold`: Confirmation that both Arm 1 and Arm 2 held the block simultaneously ($\ge 15\text{ N}$).
  - `Placement Accuracy`: Final distance from the cube to the storage bin center ($1.58\text{ cm}$, within the $< 2.0\text{ cm}$ specification).
  - `Worker Process`: Subprocess worker ID that generated the demonstration.

---

## 5. REST API Reference (Dataset Studio)

Developers can interact with Dataset Studio programmatically using the following endpoints:

| Endpoint | Method | Description | Example Response |
| :--- | :--- | :--- | :--- |
| `/api/dataset/overview` | `GET` | Overall dataset statistics, size, and bounds | `{"total_episodes": 50, "size_mb": 531.97, "fps": 25}` |
| `/api/dataset/episodes` | `GET` | Catalog of demonstration episodes with pagination | `{"total": 50, "episodes": [...]}` |
| `/api/dataset/episode/{id}` | `GET` | Decimated trajectory checkpoints (actions & qpos) | `{"total_steps": 649, "action_trajectory": [...]}` |
| `/api/dataset/recordings` | `GET` | List available MP4 demo videos | `{"recordings": [{"filename": "phase6_demo_049.mp4"}]}` |
| `/recordings/{file.mp4}` | `GET` | Stream video file with HTTP 206 support | `206 Partial Content (video/mp4)` |
| `/api/dataset/collect/start` | `POST` | Spawn parallel collection subprocess | `{"status": "ok", "message": "Parallel collection launched"}` |
| `/api/dataset/collect/stop` | `POST` | Terminate active collection job | `{"status": "ok", "message": "Collection job terminated"}` |
| `/api/dataset/collect/status`| `GET` | Poll worker state matrix & stdout | `{"is_running": true, "progress_percent": 45.0, ...}` |

### Python API Example
```python
import requests

# 1. Fetch consolidated dataset overview
overview = requests.get("http://127.0.0.1:8000/api/dataset/overview").json()
print(f"Total Episodes: {overview['total_episodes']}, Size: {overview['size_mb']} MB")

# 2. Inspect Episode #0 trajectory
ep0 = requests.get("http://127.0.0.1:8000/api/dataset/episode/0").json()
print(f"Episode 0: {ep0['total_steps']} steps, Bin Distance: {ep0['distance_to_bin_m'] * 100:.2f} cm")
```

---

## 6. Troubleshooting & FAQ

### Q1: The browser shows unstyled elements or raw inputs.
**Solution**: The browser has cached an old stylesheet. Press **Ctrl + F5** (or **Shift + Reload**) to fetch `style.css?v=3.2`.

### Q2: Can I collect more demonstrations without deleting existing ones?
**Yes**. The collection engine appends new episodes to the dataset. Set `--overwrite` to `False` (default behavior in the UI) to preserve existing episodes.

### Q3: What happens if an episode fails during collection?
The automated FSM monitors contact forces and object height. If a slip occurs, the episode is discarded automatically, and the worker resets the simulation to collect a valid demonstration, ensuring a 100% success rate in the dataset.

### Q4: How does Dataset Studio integrate with Phase 4 (Neural Studio)?
The consolidated dataset (`data/handover_demos.hdf5`) generated by Dataset Studio provides the training inputs for Phase 4. The Action Chunking Transformer (ACT) model reads observation tensors (`top`, `wrist1`, `wrist2`) and 14-DOF actions directly from this file.
