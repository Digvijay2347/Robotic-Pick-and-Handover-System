# Dual-Arm Robotic Pick-and-Handover System
## Simulation Setup & Interactive User Guide

> **Current Status**: Phase 0 (Environment), Phase 1 (Scene & Assets), Phase 2 (Kinematics & Single-Arm Pick), and Phase 3 (Perception & 6-DOF Grasp Planning) Complete 🟢  
> **Target Environment**: Windows 11 · Python 3.11 · MuJoCo 3.16.0 · Intel Arc 140V (8GB)  
> **Repository**: [`Digvijay2347/Robotic-Pick-and-Handover-System`](https://github.com/Digvijay2347/Robotic-Pick-and-Handover-System)

---

## 1. Quickstart: Run the Autonomous Single-Arm Pick in 3D Viewer

To watch Arm 1 autonomously reach into the wall shelf, grasp the red box with its Robotiq 2F-85 gripper, lift it, and extract it into open space in real time:

### Interactive 3D Viewer Mode (Watch Live):
```powershell
.\.venv\Scripts\python scripts/02_test_ik_reach.py --viewer
```

### Fast Headless Benchmark:
```powershell
.\.venv\Scripts\python scripts/02_test_ik_reach.py
```

---

## 2. Inspecting the Simulation Scene & Passive Viewer

You can also launch a freeform passive viewer to freely manipulate the dual UR10e arms, shelf, and objects with your mouse:

### Option A: Direct Command (No Activation Needed)
```powershell
.\.venv\Scripts\python -m mujoco.viewer --mjcf=models/dual_ur10e_scene.xml
```

### Option B: Enable PowerShell Script Execution & Activate
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\activate
python -m mujoco.viewer --mjcf=models/dual_ur10e_scene.xml
```

### Option C: Scene Health & Sensor Test
```powershell
.\.venv\Scripts\python scripts/01_test_scene.py
```

---

## 2. Interactive Viewer Controls & Shortcuts

When the MuJoCo 3D viewer window opens, you can interact with the robot, shelf, and objects using mouse and keyboard shortcuts:

| Action | Control / Shortcut | Description |
|---|---|---|
| **Rotate Camera** | `Left Click + Drag` | Orbit around the current focal point. |
| **Pan Camera** | `Shift + Right Click + Drag` *(or Middle Click)* | Move camera horizontally and vertically. |
| **Zoom In / Out** | `Right Click + Drag` *(or Mouse Scroll)* | Smooth zoom in/out. |
| **Focus on Element** | `Double Left Click` | Double-click any robot link, shelf, or object to center the camera on it. |
| **Pause / Unpause Physics** | `Spacebar` | Freezes or resumes real-time physics simulation. |
| **Step Single Frame** | `Page Down` | Advances physics by one timestep (`0.002s`) when paused. |
| **Slow Motion** | `Shift + Down Arrow` | Slows simulation down for detailed contact inspection. |
| **Speed Up** | `Shift + Up Arrow` | Accelerates simulation speed. |
| **Apply Perturbation Force** | `Ctrl + Right Click + Drag` | **Grab and pull** any object or robot link in real time to test physics, friction, and gravity! |
| **Apply Torque** | `Ctrl + Left Click + Drag` | Apply rotational torque to bodies. |
| **Toggle Side UI Panels** | `Tab` | Toggles the MuJoCo configuration sidebar. |
| **Reset Simulation** | `Backspace` | Resets the simulation to the initial scene state. |
| **Reload MJCF** | `Ctrl + L` | Live reloads the XML model if you modify scene files. |

> [!TIP]
> **Try this in the viewer:**
> 1. Press `Space` to let physics settle.
> 2. Hold `Ctrl` + `Right Click + Drag` on the **red target box** on the shelf to lift or slide it around. Watch it fall back naturally under realistic gravity and friction!
> 3. Double-click the **handover cyan site** in the center to zoom straight into the shared transfer zone.

---

## 3. What is Currently in the Simulation World?

The simulation scene (`models/dual_ur10e_scene.xml`) consists of:

```
                    ┌───────────────────────────────────────────┐
                    │          4×3 WALL SHELF (Wood)            │
                    │  ┌─────────┬─────────┬─────────┬─────────┐ │
                    │  │ Row 3   │         │         │         │ │
                    │  ├─────────┼─────────┼─────────┼─────────┤ │
                    │  │ 📦 Red  │ 🥫 Green│         │         │ │  ← Target & Can
                    │  ├─────────┼─────────┼─────────┼─────────┤ │
                    │  │ 🧴 Blue │         │         │ 🎲 Yellow│ │  ← Bottle & Cube
                    │  └─────────┴─────────┴─────────┴─────────┘ │
                    └───────────────────────────────────────────┘
                                          ↑
                             ARM 1 Reach Corridor (+Y)
                                          │
                     ┌────────────────────┼────────────────────┐
                     │                                         │
       ┌──────────┐  │             ★ HANDOVER SITE             │  ┌──────────┐
       │  ARM 2   │──┼             [0.0, 0.1, 1.05]            ├──│  ARM 1   │
       │ Receiver │  │                                         │  │  Giver   │
       │ (Amber)  │  └────────────────────┬────────────────────┘  │  (Blue)  │
       └──────────┘                       │                       └──────────┘
                                          │
                             ARM 2 Reach Corridor (-Y)
                                          ↓
                             ┌─────────────────────────┐
                             │    STORAGE BIN 📥       │
                             │ (Chamfered Blue Basin)  │
                             └─────────────────────────┘
```

### Detailed Component Specifications

#### 1. Arm 1 (Giver — Right Side, Elevated Pedestal)
- **Base Position**: `[0.6, 0.2, 0.75]` m, elevated on a cylindrical pedestal (`z = 0.375` m).
- **Orientation**: Rotated $90^\circ$ facing the shelf (`+Y` direction).
- **Visual Accent**: Iconic UR collaborative blue joint caps (`ur_blue`).
- **End-Effector**: Fully articulated **Robotiq 2F-85** parallel-jaw adaptive gripper.
- **EE Site**: `arm1_ee_site` centered exactly between the silicone finger pads (`pos="0 0 0.145"`).
- **Sensors**: Forward-looking wrist camera (`arm1_wrist_cam`) mounted on the flange.

#### 2. Arm 2 (Receiver — Left Side, Elevated Pedestal)
- **Base Position**: `[-0.6, 0.2, 0.75]` m, elevated on a cylindrical pedestal (`z = 0.375` m).
- **Orientation**: Rotated $-90^\circ$ facing the storage bin (`-Y` direction).
- **Visual Accent**: High-visibility amber/orange joint caps (`ur_amber`) to easily distinguish Arm 1 from Arm 2.
- **End-Effector**: Fully articulated **Robotiq 2F-85** parallel-jaw adaptive gripper.
- **EE Site**: `arm2_ee_site` centered between the silicone finger pads.
- **Sensors**: Forward-looking wrist camera (`arm2_wrist_cam`) mounted on the flange.

#### 3. Wall Shelf / Cupboard ([`models/shelf.xml`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/models/shelf.xml))
- **Placement**: `pos="0.0, 1.2, 0.0"` m (centered in front of the arms).
- **Compartments**: 12 independent compartments (4 columns × 3 shelf levels).
- **Dimensions**: Width $1.2\text{ m}$, Depth $0.35\text{ m}$, Height $0.9\text{ m}$.
- **Materials**: Natural wood grain shelf plates with dark graphite vertical dividers and support legs.

#### 4. Physical Manipulable Objects ([`models/objects.xml`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/models/objects.xml))
- 📦 **Target Red Box** (`target_object`): Size $5\times 5\times 10\text{ cm}$, mass $0.15\text{ kg}$, high friction coefficient ($1.2$). Sits on Row 2, Column 1 (`[-0.15, 1.05, 0.90]`).
- 🥫 **Green Soda Can** (`can_object`): Radius $2.8\text{ cm}$, height $9.6\text{ cm}$, mass $0.12\text{ kg}$. Sits on Row 2, Column 2 (`[0.15, 1.05, 0.90]`).
- 🧴 **Blue Flask/Bottle** (`bottle_object`): Dual cylinder geometry with narrow neck, mass $0.10\text{ kg}$. Sits on Row 1, Column 0 (`[-0.45, 1.05, 0.54]`).
- 🎲 **Yellow Cube** (`cube_object`): Size $5\times 5\times 5\text{ cm}$, mass $0.08\text{ kg}$. Sits on Row 1, Column 3 (`[0.45, 1.05, 0.54]`).

#### 5. Storage Drop Bin ([`models/bin.xml`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/models/bin.xml))
- **Placement**: `pos="0.0, -0.8, 0.0"` m (behind the arms).
- **Features**: 4 chamfered slanted guide rims along the top rim so dropped objects slide securely into the basin.
- **Target Site**: `bin_drop_target` marked inside the bottom plate.

#### 6. 14-DOF Actuator Mapping
| Index | Actuator Name | Joint Controlled | Type | Control Range |
|:---:|---|---|:---:|:---:|
| `00` | `arm1_j0_pan` | `arm1_shoulder_pan_joint` | Position | $[-2\pi, +2\pi]$ |
| `01` | `arm1_j1_lift` | `arm1_shoulder_lift_joint` | Position | $[-2\pi, +2\pi]$ |
| `02` | `arm1_j2_elbow` | `arm1_elbow_joint` | Position | $[-\pi, +\pi]$ |
| `03` | `arm1_j3_wrist1` | `arm1_wrist_1_joint` | Position | $[-2\pi, +2\pi]$ |
| `04` | `arm1_j4_wrist2` | `arm1_wrist_2_joint` | Position | $[-2\pi, +2\pi]$ |
| `05` | `arm1_j5_wrist3` | `arm1_wrist_3_joint` | Position | $[-2\pi, +2\pi]$ |
| `06` | `arm1_gripper` | `arm1_split` (tendon) | Force/Pos | $[0, 255]$ ($0=\text{open}, 255=\text{close}$) |
| `07` | `arm2_j0_pan` | `arm2_shoulder_pan_joint` | Position | $[-2\pi, +2\pi]$ |
| `08` | `arm2_j1_lift` | `arm2_shoulder_lift_joint` | Position | $[-2\pi, +2\pi]$ |
| `09` | `arm2_j2_elbow` | `arm2_elbow_joint` | Position | $[-\pi, +\pi]$ |
| `10` | `arm2_j3_wrist1` | `arm2_wrist_1_joint` | Position | $[-2\pi, +2\pi]$ |
| `11` | `arm2_j4_wrist2` | `arm2_wrist_2_joint` | Position | $[-2\pi, +2\pi]$ |
| `12` | `arm2_j5_wrist3` | `arm2_wrist_3_joint` | Position | $[-2\pi, +2\pi]$ |
| `13` | `arm2_gripper` | `arm2_split` (tendon) | Force/Pos | $[0, 255]$ ($0=\text{open}, 255=\text{close}$) |

---

## 4. How to Inspect Camera Sensor Feeds

The system includes 4 camera views that are captured offscreen during simulation:

1. **`overhead_cam`** ($640\times 480$, $75^\circ$ FOV): Top-down bird's-eye perspective of the entire arena.
2. **`shelf_cam`** ($640\times 480$, $60^\circ$ FOV): Direct depth and RGB view focused on the shelf compartments.
3. **`arm1_wrist_cam`** ($640\times 480$, $75^\circ$ FOV): First-person view looking forward from Arm 1's gripper.
4. **`arm2_wrist_cam`** ($640\times 480$, $75^\circ$ FOV): First-person view looking forward from Arm 2's gripper.

Snapshots from all 4 sensors are automatically generated in [`recordings/`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/recordings/):
- [`recordings/overhead_cam.png`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/recordings/overhead_cam.png)
- [`recordings/shelf_cam.png`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/recordings/shelf_cam.png)
- [`recordings/arm1_wrist_cam.png`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/recordings/arm1_wrist_cam.png)
- [`recordings/arm2_wrist_cam.png`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/recordings/arm2_wrist_cam.png)

To re-capture fresh snapshots at any time, run:
```powershell
python scripts/01_test_scene.py
```

---

## 5. Troubleshooting & FAQ

#### Q: The viewer window closes immediately or fails to launch.
Ensure the virtual environment is activated (`.venv\Scripts\activate`). If running through remote desktop or SSH, MuJoCo requires an active Windows desktop session for OpenGL context creation.

#### Q: OSError: [WinError 4551] An Application Control policy has blocked this file
This error is triggered by Windows 11 Smart App Control (SAC) or OneDrive tagging newly extracted `.dll` binaries in `.venv` with the Mark of the Web (`Zone.Identifier`).
To permanently clear this block across all virtual environment binaries:
```powershell
Get-ChildItem -Path '.venv' -Recurse | Unblock-File
```
Additionally, `mujoco/__init__.py` in `.venv` has been patched so that optional bundled plugin DLLs (`sensor.dll`, etc.) caught by Application Control are handled safely without preventing core MuJoCo physics from loading.

#### Q: How fast is physics stepping?
The physics timestep is set to $0.002\text{ s}$ ($500\text{ Hz}$). On this system's multi-threaded CPU and Intel Arc GPU, MuJoCo steps at over **$6,500\text{ Hz}$** ($\approx 13\times$ faster than real-time), enabling fast demonstration collection and rollouts.

---

## 6. Phase 2 Architecture & Autonomous Pick Pipeline

In Phase 2, we implemented the complete autonomous kinematic and Cartesian control stack:
- **[`control/kinematics.py`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/control/kinematics.py)**: Damped Least Squares (DLS) Jacobian IK solver, closed-loop numeric iteration, automatic 2π branch unwrapping, null-space postural biasing, and physical joint limit clamping.
- **[`control/trajectory.py`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/control/trajectory.py)**: 5th-order minimum-jerk quintic polynomial interpolator (`s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5`) and SLERP orientation interpolator.
- **[`control/dual_arm_controller.py`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/control/dual_arm_controller.py)**: Unified 14-DOF actuator coordinator, normalized gripper command interface `[0.0, 1.0]`, and MuJoCo contact dynamics normal force verifier (`check_grasp`).
- **[`control/collision_checker.py`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/control/collision_checker.py)**: Inter-arm proximity and link-to-shelf clearance monitor.
- **[`scripts/02_test_ik_reach.py`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/scripts/02_test_ik_reach.py)**: End-to-end 6-stage autonomous pick test.

### Pick Pipeline Stages:
1. `1_PREGRASP` (2.0s): Arm moves smoothly from home resting pose to 15cm standoff in front of shelf compartment.
2. `2_APPROACH` (1.5s): Horizontal antipodal alignment into the compartment, centering finger pads around the box.
3. `3_GRASP` (1.0s): Closes gripper onto the object; verifies firm antipodal normal contact force (`276.69 N`).
4. `4_LIFT` (1.5s): Smooth vertical lift (+12cm) to break static friction with shelf surface.
5. `5_RETRACT` (2.0s): Horizontally extracts object out of the shelf compartment into free space.
6. `6_HOLD` (1.5s): Holds object steady in free space; final clearance is verified (+12.8cm, zero collisions).

---

## 7. Phase 3: Perception & 6-DOF Grasp Planning

In Phase 3, we implemented an autonomous synthetic perception and 6-DOF grasp planning pipeline driven purely from `shelf_cam` RGB-D sensory input:
- **[`perception/pointcloud.py`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/perception/pointcloud.py)**: Camera intrinsic derivation, synchronized linear metric depth rendering, pinhole back-projection to 3D World coordinates ($P_w = R_{\text{cam}} P_c + T_{\text{cam}}$), and Open3D integration.
- **[`perception/segmentation.py`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/perception/segmentation.py)**: 3D shelf ROI cropping, iterative RANSAC plane stripping for shelf boards and dividers, DBSCAN Euclidean clustering, and color signature matching (`find_target`).
- **[`perception/grasp_planners/`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/perception/grasp_planners/)**:
  - `GeometricGraspPlanner`: Native antipodal parallel-jaw planner using PCA decomposition, approach corridor alignment ($+Y$), horizontal finger pad orientation, and aperture sizing.
  - `GraspGenAdapter`: ML adapter for NVlabs/GraspGen with automatic geometric fallback.
- **[`perception/grasp_filter.py`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/perception/grasp_filter.py)**: DLS numerical IK reachability and shelf collision filtering.
- **[`scripts/03_test_perception.py`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/scripts/03_test_perception.py)**: Full end-to-end perception-driven pick pipeline.

### Running the Perception-Driven Pick:

#### Fast Headless Benchmark:
```powershell
.\.venv\Scripts\python scripts/03_test_perception.py
```

#### Interactive 3D Viewer Mode (Watch Live):
```powershell
.\.venv\Scripts\python scripts/03_test_perception.py --viewer
```

#### With GraspGen ML Adapter Flag:
```powershell
.\.venv\Scripts\python scripts/03_test_perception.py --use-graspgen
```

### Verified Phase 3 Telemetry:
| Metric | Acceptance Gate | Observed Value | Status |
|---|:---:|:---:|:---:|
| **3D Object Localization Error** | $< 15.0\text{ mm}$ | **$6.45\text{ mm}$** | PASSED 🟢 |
| **Grasp Quality Score** | $> 0.50$ | **$0.693$** | PASSED 🟢 |
| **Kinematically Reachable Grasps** | $\ge 1$ | **$2\text{ of }2$** | PASSED 🟢 |
| **Grasp Contact Normal Force** | $> 0.5\text{ N}$ | **$272.04\text{ N}$** | PASSED 🟢 |
| **Vertical Lift Clearance** | $\ge 8.0\text{ cm}$ | **$+13.1\text{ cm}$** | PASSED 🟢 |
| **Link-to-Shelf Collisions** | $0$ | **$0\text{ (CLEAN)}$** | PASSED 🟢 |

### Perception Artifacts Generated in [`recordings/`](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/recordings/):
- `phase3_initial_shelf_view.png`: Offscreen RGB snapshot from `shelf_cam`.
- `phase3_target_cluster.ply`: Segmented 3D point cloud of the target red box.
- `phase3_stage_1_PREGRASP.png` to `phase3_stage_6_HOLD.png`: Pick execution milestones.
- `phase3_final_overhead.png`: Overhead view of Arm 1 holding the retracted object in free space.

---

## 8. Next Step in Roadmap: Phase 4 (Coordinated Dual-Arm Mid-Air Handover FSM)

With perception and single-arm picking fully verified, the next phase will implement:
- **Shared Handover Workspace Analysis**: Determine collision-free SE(3) transfer pose $\mathbf{T}_{\text{handover}}$ at `[0.0, 0.1, 1.05]`.
- **Dual-Arm Master-Follower Trajectory Generation**: Synchronized minimum-jerk motion for Arm 1 and Arm 2.
- **Multi-Phase Handover State Machine**:
  1. `APPROACH_HANDOVER`: Arm 1 moves with object to handover site; Arm 2 moves open gripper to ready pre-transfer pose.
  2. `ALIGNED_INTERCEPT`: Arm 2 aligns finger pads around the exposed upper/lower section of the object.
  3. `DUAL_CONTACT_GRIP`: Arm 2 closes gripper; verify dual-arm force equilibrium ($F_1 > 0, F_2 > 0$).
  4. `TRANSFER_RELEASE`: Arm 1 opens gripper; Arm 2 confirms sole load carriage.
  5. `RETRACT_CLEAR`: Arm 1 retracts to home; Arm 2 transports object to drop bin.
  6. `DEPOSIT`: Arm 2 releases object over storage bin.
