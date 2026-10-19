# Phase 5 Learning Guide: Gym-ALOHA Standard Gymnasium Environment

## 1. Executive Summary & Overview

### What is ALOHA?
**ALOHA** (*A Low-cost Open-source Hardware System for Bimanual Teleoperation*) is an open-source robotic manipulation standard developed by Tony Zhao and researchers at Stanford University (2023). It pioneered modern bimanual imitation learning using deep vision models, notably **Action Chunking with Transformers (ACT)** and **Diffusion Policy**.

The ALOHA architecture consists of:
- **Two 6-DOF robotic manipulator arms** (e.g., ViperX, UR5e, or UR10e).
- **Two parallel-jaw grippers** (e.g., Robotiq 2F-85 or ViperX grippers).
- **A multi-camera visual rig**: typically an overhead global camera plus wrist-mounted cameras on each arm.
- A unified **14-DOF action and state representation** operating at a strict **50 Hz control frequency**.

### What is Gym-ALOHA?
**Gym-ALOHA** is the standardized [Gymnasium](https://gymnasium.farama.org/) (formerly OpenAI Gym) environment specification for ALOHA-style bimanual robotic platforms. It translates raw physics engines (such as MuJoCo) into standard reinforcement learning (RL) and imitation learning (IL) abstractions:

$$\text{Action } a_t \in [-1, 1]^{14} \xrightarrow{\text{step}(a_t)} (\text{Observation } o_{t+1}, \text{Reward } r_{t}, \text{Terminated}, \text{Truncated}, \text{Info})$$

### Why Did We Need Phase 5?
In Phases 1 through 4, we built:
1. The **MuJoCo simulation scene** (`models/dual_ur10e_scene.xml`).
2. The **kinematics and DLS numerical solvers** (`control/kinematics.py`).
3. The **3D perception and grasp synthesizer** (`perception/`).
4. The **scripted 12-state expert handover finite state machine** (`fsm/handover_fsm.py`).

While the scripted FSM orchestrator achieves 100% success, it is a hardcoded procedural state machine. It cannot directly train deep neural networks (like **ACT**, **Diffusion Policy**, or **PPO**). 

Deep learning frameworks (such as [Hugging Face LeRobot](https://github.com/huggingface/lerobot), PyTorch, and Stable-Baselines3) cannot consume raw MuJoCo C-struct pointers (`mjModel*`, `mjData*`). They require a **standardized, modular, reproducible Gymnasium environment** with uniform tensors, image dictionaries, action bounds, and reward functions. 

**Phase 5 bridges physics simulation with deep machine learning.**

```mermaid
graph LR
    subgraph Phase 1-4 [Physics & Control]
        M[MuJoCo Physics Engine<br/>500 Hz Simulation] --> C[DualArmController<br/>Kinematics & Gripper]
        C --> FSM[Scripted Handover FSM<br/>12-Stage Procedural Expert]
    end

    subgraph Phase 5 [Gym-ALOHA Bridge]
        C --> ENV[DualUR10eHandoverEnv<br/>Gymnasium Standard]
        M --> ENV
        ENV --> ACT_SP[14-DOF Action Space<br/>[-1, 1] Normalized]
        ENV --> OBS_SP[Dict Observation Space<br/>qpos, qvel, RGB Images]
    end

    subgraph Phase 6-8 [Machine Learning Pipeline]
        ENV --> REC[Phase 6: Data Recorder<br/>HDF5 / LeRobot Dataset]
        FSM --> REC
        REC --> TRAIN[Phase 7: Policy Training<br/>ACT / Diffusion Policy]
        TRAIN --> EVAL[Phase 8: Autonomous Eval<br/>Closed-Loop Policy Rollout]
        EVAL --> ENV
    end
```

---

## 2. Technical Architecture: How We Built It

The implementation is located across three core files:
- [envs/dual_ur10e_env.py](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/envs/dual_ur10e_env.py): Core environment implementation.
- [envs/__init__.py](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/envs/__init__.py): Gymnasium environment registry.
- [scripts/05_test_gym_env.py](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/scripts/05_test_gym_env.py): Validation suite and acceptance test runner.

---

### A. 14-DOF Action Space

The action space represents the joint target and gripper commands for both arms simultaneously:

```python
self.action_space = spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(14,),
    dtype=np.float32,
)
```

#### Index Mapping:
| Slice | Controlled Subsystem | Range | Function |
| :--- | :--- | :--- | :--- |
| `action[0:6]` | **Arm 1 Joints** (Pan, Lift, Elbow, Wrist 1, Wrist 2, Wrist 3) | $[-1.0, 1.0]$ | Normalized joint angles |
| `action[6]` | **Arm 1 Gripper** (Robotiq 2F-85) | $[-1.0, 1.0]$ | $-1.0 = \text{Fully Open}$, $+1.0 = \text{Fully Closed}$ |
| `action[7:13]` | **Arm 2 Joints** (Pan, Lift, Elbow, Wrist 1, Wrist 2, Wrist 3) | $[-1.0, 1.0]$ | Normalized joint angles |
| `action[13]` | **Arm 2 Gripper** (Robotiq 2F-85) | $[-1.0, 1.0]$ | $-1.0 = \text{Fully Open}$, $+1.0 = \text{Fully Closed}$ |

#### Action Modes:
1. **Absolute Mode (`action_mode="absolute"`, Default)**:
   Maps normalized $[-1, 1]$ directly to the robot's physical joint limits:
   $$q_{\text{target}} = q_{\text{min}} + \frac{\text{action} + 1.0}{2} \times (q_{\text{max}} - q_{\text{min}})$$
   This is the standard format used by **ACT** and **LeRobot** teleoperation.
2. **Delta Mode (`action_mode="delta"`)**:
   Interprets action as bounded joint velocity/increment:
   $$q_{\text{target}} = q_{\text{current}} + \text{action} \times \Delta q_{\text{max}}$$
   Commonly used in classical reinforcement learning (SAC, PPO).

---

### B. Observation Space (ALOHA Dict Standard)

The observation space conforms strictly to the nested dictionary standard expected by bimanual visual policies:

```python
self.observation_space = spaces.Dict({
    "qpos": spaces.Box(low=qpos_low, high=qpos_high, shape=(14,), dtype=np.float64),
    "qvel": spaces.Box(low=qvel_low, high=qvel_high, shape=(14,), dtype=np.float64),
    "images": spaces.Dict({
        "top": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
        "wrist1": spaces.Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8),
        "wrist2": spaces.Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8),
    }),
})
```

#### Observation Contents:
1. **`qpos` (14-DOF float64)**:
   - `qpos[0:6]`: Arm 1 joint angles in radians.
   - `qpos[6]`: Arm 1 normalized gripper position ($0.0 = \text{open}, 1.0 = \text{closed}$).
   - `qpos[7:13]`: Arm 2 joint angles in radians.
   - `qpos[13]`: Arm 2 normalized gripper position ($0.0 = \text{open}, 1.0 = \text{closed}$).
2. **`qvel` (14-DOF float64)**:
   - Angular velocities ($\text{rad/s}$) for all 14 degrees of freedom.
3. **`images` (Visual Sensor Rig)**:
   - **`top`**: Overhead camera snapshot $(480 \times 640 \times 3)$ giving full spatial context of both arms, shelf, and storage bin.
   - **`wrist1`**: Arm 1 wrist-mounted camera $(240 \times 320 \times 3)$ giving egocentric view into shelf compartments.
   - **`wrist2`**: Arm 2 wrist-mounted camera $(240 \times 320 \times 3)$ giving close-up view of the central handover zone.

---

### C. Multi-Rate Temporal Hierarchy (50 Hz Policy vs 500 Hz Physics)

A fundamental challenge in robotic manipulation simulation is the **frequency mismatch**:
- **MuJoCo Physics**: Requires small timesteps ($dt = 0.002\text{ s}$, **$500\text{ Hz}$**) to solve stiff frictional contacts, tendon dynamics, and collision impulses without numerical instability or kinetic explosions.
- **Neural Network Policies**: Deep vision policies (ResNet/Transformer backbones) cannot run at 500 Hz. They run at **$50\text{ Hz}$** ($0.020\text{ s}$ per control decision).

We solved this via **substepping**:
$$\text{Control Rate } (50\text{ Hz}) = \frac{\text{Physics Rate } (500\text{ Hz})}{n_{\text{substeps}} = 10}$$

In `dual_ur10e_env.py`:
```python
# Command actuators and perform substepping
for _ in range(self.n_substeps): # 10 substeps of 0.002s = 0.020s per step
    self.controller.set_arm_joint_targets(1, target_q1)
    self.controller.set_gripper(1, norm_grip1)
    self.controller.set_arm_joint_targets(2, target_q2)
    self.controller.set_gripper(2, norm_grip2)
    mujoco.mj_step(self.model, self.data)
```
Each environment `step()` advances simulation time by exactly $20\text{ ms}$.

---

### D. Zero-Allocation Multi-Camera Rendering Pipeline

Rendering multiple camera angles per step can severely bottleneck simulation speed if OpenGL framebuffers are continually re-allocated.

To maximize throughput, we instantiated **two dedicated, persistent MuJoCo renderers**:
```python
def _init_renderers(self):
    if self.renderer_top is None:
        self.renderer_top = mujoco.Renderer(self.model, height=480, width=640)
    if self.renderer_wrist is None:
        self.renderer_wrist = mujoco.Renderer(self.model, height=240, width=320)
```
- `renderer_top` maintains the $480 \times 640$ buffer for the overhead camera.
- `renderer_wrist` is reused sequentially for `arm1_wrist_cam` and `arm2_wrist_cam` because both share the identical $(240, 320)$ resolution.

---

### E. Reward Function Design

The environment provides both **dense progress shaping** (for RL exploration) and **sparse goal completion**:

$$R = -w_1 d(\text{EE}_1, \text{obj}) + \mathbb{I}_{\text{grasped}_1} \left[ R_{\text{grasp}} - w_2 d(\text{obj}, P_{\text{handover}}) \right] + \mathbb{I}_{\text{grasped}_2} \left[ R_{\text{transfer}} - w_3 d(\text{obj}, P_{\text{bin}}) \right] + \mathbb{I}_{\text{in\_bin}} R_{\text{success}}$$

Where:
- $-w_1 d(\text{EE}_1, \text{obj})$: Steers Arm 1 to reach toward the red box on the shelf ($w_1 = 1.0$).
- $\mathbb{I}_{\text{grasped}_1}$: Boolean indicator from `check_grasp(1, "target_object")`. Awards $+10.0$ bonus upon solid contact, then shapes distance to the central handover point $P_{\text{handover}} = [0.0, 0.10, 1.05]$.
- $\mathbb{I}_{\text{grasped}_2}$: Boolean indicator when Arm 2 takes custody. Awards $+25.0$ bonus, then shapes distance toward the storage bin $P_{\text{bin}} = [0.0, -0.80, 0.07]$.
- $\mathbb{I}_{\text{in\_bin}}$: Awards $+100.0$ terminal success bonus when the object comes to rest inside the bin.

---

### F. Domain Randomization on Reset

To prevent visual policies from overfitting to a fixed coordinate, `reset()` supports stochastic domain randomization:
```python
if self.randomize_on_reset and self.target_joint_id != -1:
    qadr = self.model.jnt_qposadr[self.target_joint_id]
    # Random translation: X in [-3cm, +3cm], Y in [-2cm, +2cm]
    dx = self.np_random.uniform(-0.03, 0.03)
    dy = self.np_random.uniform(-0.02, 0.02)
    self.data.qpos[qadr + 0] = 0.45 + dx
    self.data.qpos[qadr + 1] = 1.05 + dy

    # Random yaw rotation: [-25 deg, +25 deg]
    yaw_deg = self.np_random.uniform(-25.0, 25.0)
    quat_rand = R.from_euler("z", yaw_deg, degrees=True).as_quat(scalar_first=True)
    self.data.qpos[qadr + 3:qadr + 7] = quat_rand
```

---

## 3. Registered Environments in Gymnasium

In [envs/__init__.py](file:///c:/Users/Digvijay%20singh/OneDrive/Desktop/Motion/prod2/envs/__init__.py), we registered two environment flavors:

1. **`DualUR10eHandover-v0` (Visual ALOHA Standard)**:
   - Observation space: `qpos` (14), `qvel` (14), and `images` (`top`, `wrist1`, `wrist2`).
   - Use case: Demonstration recording, visual policy training (ACT, Diffusion Policy).
   ```python
   import gymnasium as gym
   import envs

   env = gym.make("DualUR10eHandover-v0")
   obs, info = env.reset()
   # obs["images"]["top"].shape -> (480, 640, 3)
   ```

2. **`DualUR10eHandover-State-v0` (High-Throughput State RL)**:
   - Observation space: `qpos` (14), `qvel` (14), `target_pos` (3), `ee1_pos` (3), `ee2_pos` (3).
   - Speed: **565+ steps/second** (~5,640 physics Hz) because camera rendering is bypassed.
   - Use case: Fast state-based RL, reward tuning, curriculum debugging.
   ```python
   env = gym.make("DualUR10eHandover-State-v0")
   obs, info = env.reset()
   # State rollout at > 500 FPS
   ```

---

## 4. How Phase 5 Enables the Rest of the Roadmap

With Phase 5 completed, our pipeline directly connects to subsequent deep learning milestones:

| Phase | Milestone | Role of Phase 5 Gymnasium Environment |
| :--- | :--- | :--- |
| **Phase 6** | **Expert Dataset Collection** | The demonstration recorder will wrap `DualUR10eHandover-v0`. The scripted FSM from Phase 4 will drive the environment via `step(action)` to record 50–100 successful episodes with full image streams and 14-DOF actions directly into **HDF5 / LeRobot** format. |
| **Phase 7** | **Imitation Learning (ACT / Diffusion)** | The trained neural network policy (PyTorch / LeRobot) takes `obs` dictionary from `env.reset()` and outputs 14-DOF action chunks executed via `env.step()`. |
| **Phase 8** | **Autonomous Evaluation** | We will evaluate the trained policy across randomized seeds in `DualUR10eHandover-v0` to measure autonomous pick-and-handover success rates without human supervision. |

---

## 5. How to Run & Verify

To execute the complete acceptance gate validation suite:

```powershell
.\.venv\Scripts\python.exe scripts/05_test_gym_env.py
```

This automatically runs:
1. `gymnasium.utils.env_checker.check_env()`: Verifies zero warnings and full Gymnasium API compliance.
2. 500-step random policy rollout: Confirms observation types, reward shapes, and image tensors.
3. Domain randomization test: Verifies distribution variance across resets.
4. State throughput benchmark: Validates high-speed stepping (>500 steps/sec).
5. Visual snapshot capture: Saves current observations to `recordings/phase5_obs_top.png`, `wrist1.png`, and `wrist2.png`.
