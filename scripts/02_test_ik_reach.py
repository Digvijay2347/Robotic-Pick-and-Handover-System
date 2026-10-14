"""
Phase 2 Validation Script: Kinematics, Cartesian Control & Single-Arm Pick Baseline.

Autonomous Ground-Truth Pick Pipeline:
1. Arm 1 starts at home ready pose with Robotiq 2F-85 fully open.
2. Senses ground-truth 3D pose of target box on the shelf.
3. Generates minimum-jerk trajectory to pre-grasp standoff (15cm outside shelf).
4. Executes linear Cartesian approach into shelf compartment with horizontal antipodal alignment.
5. Closes Robotiq gripper and verifies normal contact force (> 0.5 N).
6. Lifts object vertically (+12cm) to break friction with shelf surface.
7. Retracts horizontally out of the shelf compartment into open workspace.
8. Verifies stable hold and saves telemetry and visual confirmation.

Usage:
    python scripts/02_test_ik_reach.py [--viewer] [--speed SPEED]
"""

import os
import sys
import argparse
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# Ensure workspace root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import mujoco
import mujoco.viewer
from control.dual_arm_controller import DualArmController
from control.trajectory import SE3Trajectory, JointTrajectory
from control.collision_checker import CollisionChecker


def save_camera_snapshot(model, data, cam_name: str, out_path: str):
    """Renders and saves an offscreen snapshot from the specified camera."""
    try:
        import imageio
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id != -1:
            renderer.update_scene(data, camera=cam_id)
            img = renderer.render()
            imageio.imwrite(out_path, img)
        renderer.close()
    except Exception as e:
        print(f"  [WARN] Failed to capture snapshot: {e}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Single-Arm Pick Baseline.")
    parser.add_argument("--viewer", action="store_true", help="Launch interactive passive viewer.")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation speed multiplier.")
    args = parser.parse_args()

    scene_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "dual_ur10e_scene.xml"))
    rec_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "recordings"))
    os.makedirs(rec_dir, exist_ok=True)

    print(f"\n========================================================")
    print(f"  PHASE 2: SINGLE-ARM PICK BASELINE & KINEMATICS TEST   ")
    print(f"========================================================")
    print(f"Scene: {scene_path}\n")

    # 1. Load Model & Data
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # Initialize Controller and Collision Checker
    controller = DualArmController(model, data)
    collision_checker = CollisionChecker(model, data)
    kin1 = controller.kinematics_arm1

    # Apply home keyframe state
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # Settle physics for 100 steps
    for _ in range(100):
        mujoco.mj_step(model, data)

    # 2. Query Target Object Position
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    if target_body_id == -1:
        print("[FATAL] target_object body not found in scene!")
        sys.exit(1)

    obj_pos_init = data.xpos[target_body_id].copy()
    print(f"[TARGET] Ground-truth object position: {np.round(obj_pos_init, 3)}")

    # 3. Define Grasping Geometry
    # Horizontal grasp orientation (fingers open horizontally along +X, approach along +Y)
    mat_grasp = np.column_stack([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    quat_grasp = R.from_matrix(mat_grasp).as_quat(scalar_first=True)

    # Waypoints:
    # 1. Grasp target: centered horizontally, 2.5cm above base for wrist-shelf clearance
    pos_grasp = obj_pos_init + np.array([0.0, 0.0, 0.025])
    # 2. Pre-grasp: 15 cm standoff in front of shelf compartment
    pos_pregrasp = pos_grasp + np.array([0.0, -0.15, 0.0])
    # 3. Lift pose: +12 cm vertical lift inside compartment
    pos_lift = pos_grasp + np.array([0.0, 0.0, 0.12])
    # 4. Retract pose: pulled 25 cm horizontally out of the shelf
    pos_retract = pos_lift + np.array([0.0, -0.25, 0.0])

    print(f"[PLAN] Pre-grasp Standoff : {np.round(pos_pregrasp, 3)}")
    print(f"[PLAN] Grasp Target       : {np.round(pos_grasp, 3)}")
    print(f"[PLAN] Vertical Lift      : {np.round(pos_lift, 3)}")
    print(f"[PLAN] Extracted Retract  : {np.round(pos_retract, 3)}\n")

    # Pre-solve sequential unwrapped IK for all 4 key poses
    q_home = kin1.get_joint_positions()
    ok_pre, q_pregrasp = kin1.solve_ik_closed_loop(pos_pregrasp, quat_grasp)
    ok_gr, q_grasp = kin1.solve_ik_closed_loop(pos_grasp, quat_grasp)
    ok_lf, q_lift = kin1.solve_ik_closed_loop(pos_lift, quat_grasp)
    ok_rt, q_retract = kin1.solve_ik_closed_loop(pos_retract, quat_grasp)

    if not (ok_pre and ok_gr and ok_lf and ok_rt):
        print(f"[ERROR] IK solve failed for waypoints: pre={ok_pre}, grasp={ok_gr}, lift={ok_lf}, retract={ok_rt}")
        sys.exit(1)

    print(f"[IK] All 4 trajectory waypoints resolved successfully.")
    print(f"  q_home     : {np.round(q_home, 3)}")
    print(f"  q_pregrasp : {np.round(q_pregrasp, 3)}")
    print(f"  q_grasp    : {np.round(q_grasp, 3)}")
    print(f"  q_lift     : {np.round(q_lift, 3)}")
    print(f"  q_retract  : {np.round(q_retract, 3)}\n")

    # State tracking
    stages = [
        {"name": "1_PREGRASP", "type": "move", "q_from": q_home,     "q_to": q_pregrasp, "des_pos": pos_pregrasp, "gripper": 0.0,  "duration": 2.0},
        {"name": "2_APPROACH", "type": "move", "q_from": q_pregrasp, "q_to": q_grasp,    "des_pos": pos_grasp,    "gripper": 0.0,  "duration": 1.5},
        {"name": "3_GRASP",    "type": "grip", "q_hold": q_grasp,                         "des_pos": pos_grasp,    "gripper": 0.75, "duration": 1.0},
        {"name": "4_LIFT",     "type": "move", "q_from": q_grasp,    "q_to": q_lift,     "des_pos": pos_lift,     "gripper": 0.75, "duration": 1.5},
        {"name": "5_RETRACT",  "type": "move", "q_from": q_lift,     "q_to": q_retract,  "des_pos": pos_retract,  "gripper": 0.75, "duration": 2.0},
        {"name": "6_HOLD",     "type": "hold", "q_hold": q_retract,                       "des_pos": pos_retract,  "gripper": 0.75, "duration": 1.5},
    ]

    current_stage_idx = 0
    stage_start_time = 0.0
    sim_time = 0.0
    dt = model.opt.timestep

    # Initial joint trajectory
    current_traj = JointTrajectory(q_home, q_pregrasp, duration=stages[0]["duration"])

    # Telemetry
    max_tracking_err = 0.0
    grasp_confirmed = False
    pick_successful = False

    def step_simulation(viewer=None):
        nonlocal sim_time, current_stage_idx, stage_start_time, current_traj
        nonlocal max_tracking_err, grasp_confirmed, pick_successful

        stage = stages[current_stage_idx]
        t_in_stage = sim_time - stage_start_time

        # Evaluate target joint positions
        if stage["type"] == "move":
            q_target, _, is_done = current_traj.evaluate(t_in_stage)
        else:
            q_target = stage["q_hold"]
            is_done = t_in_stage >= stage["duration"]

        # Apply arm and gripper control targets
        controller.set_arm_joint_targets(1, q_target)
        controller.set_gripper(1, stage["gripper"])
        controller.set_gripper(2, 0.0)  # Arm 2 stays open in ready pose

        # Track Cartesian positioning error
        curr_ee_pos, _ = controller.get_ee_pose(1)
        cart_err = float(np.linalg.norm(curr_ee_pos - stage["des_pos"]))
        if stage["type"] != "move":
            max_tracking_err = max(max_tracking_err, cart_err)

        # Step MuJoCo physics (500 Hz internal)
        mujoco.mj_step(model, data)
        sim_time += dt

        # Stage Transition Check
        if is_done:
            # Capture milestone image
            snap_path = os.path.join(rec_dir, f"phase2_stage_{stage['name']}.png")
            save_camera_snapshot(model, data, "shelf_cam", snap_path)

            # Special verification checks at key transitions
            if stage["name"] == "3_GRASP":
                is_grasped, f_norm = controller.check_grasp(1, "target_object")
                grasp_confirmed = is_grasped
                print(f"  --> [GRASP CHECK] Firm contact: {is_grasped} (Normal Force: {f_norm:.2f} N)")
                if not is_grasped:
                    print(f"  [WARN] Grasp contact weak or missing!")

            elif stage["name"] == "6_HOLD":
                curr_obj_pos = data.xpos[target_body_id]
                z_delta = curr_obj_pos[2] - obj_pos_init[2]
                is_grasped, f_norm = controller.check_grasp(1, "target_object")
                print(f"  --> [FINAL HOLD] Object Height Delta: +{z_delta*100:.1f} cm | Grasped: {is_grasped} ({f_norm:.2f} N)")
                if z_delta >= 0.08 and is_grasped:
                    pick_successful = True

            # Advance to next stage
            current_stage_idx += 1
            if current_stage_idx < len(stages):
                next_stage = stages[current_stage_idx]
                stage_start_time = sim_time
                print(f"[STAGE] Entering {next_stage['name']} (duration={next_stage['duration']}s)...")

                if next_stage["type"] == "move":
                    current_traj = JointTrajectory(
                        next_stage["q_from"], next_stage["q_to"],
                        duration=next_stage["duration"]
                    )
            return False  # Not finished all stages
        return current_stage_idx >= len(stages)

    # 4. Execution Loop
    print(f"[STAGE] Entering {stages[0]['name']} (duration={stages[0]['duration']}s)...")

    if args.viewer:
        print("[INFO] Running in interactive viewer. Close viewer to end early.\n")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                t0 = time.time()
                finished = step_simulation(viewer)
                viewer.sync()

                if finished:
                    print("\n[COMPLETE] All stages finished. Holding in viewer...")
                    while viewer.is_running():
                        mujoco.mj_step(model, data)
                        viewer.sync()
                        time.sleep(dt)
                    break

                elapsed = time.time() - t0
                sleep_t = (dt / args.speed) - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
    else:
        # Headless fast execution
        t0 = time.time()
        step_count = 0
        while current_stage_idx < len(stages):
            finished = step_simulation()
            step_count += 1
            if finished:
                break
        real_duration = time.time() - t0
        print(f"\n[BENCHMARK] Executed {step_count} physics steps in {real_duration:.3f}s ({step_count/real_duration:.0f} Hz).")

    # 5. Final Evaluation Report
    print("\n========================================================")
    print("               PHASE 2 EVALUATION REPORT                ")
    print("========================================================")
    final_obj_pos = data.xpos[target_body_id]
    final_z_delta = final_obj_pos[2] - obj_pos_init[2]
    is_firmly_held, final_force = controller.check_grasp(1, "target_object")

    print(f"  Target Object Initial Pos : {np.round(obj_pos_init, 3)}")
    print(f"  Target Object Final Pos   : {np.round(final_obj_pos, 3)}")
    print(f"  Vertical Lift Clearance   : +{final_z_delta*100:.1f} cm")
    print(f"  Cartesian Position Error  : {max_tracking_err*1000:.2f} mm")
    print(f"  Grasp Contact Force       : {final_force:.2f} N")
    print(f"  Grasp Confirmed           : {'YES [OK]' if is_firmly_held else 'NO [FAIL]'}")

    has_shelf_collision, coll_pairs = collision_checker.check_shelf_collision(1)
    print(f"  Shelf Collision Status    : {'NONE [CLEAN]' if not has_shelf_collision else f'COLLIDED: {coll_pairs}'}")

    # Also capture final overhead overview
    save_camera_snapshot(model, data, "overhead_cam", os.path.join(rec_dir, "phase2_final_overhead.png"))

    if pick_successful and is_firmly_held and not has_shelf_collision:
        print("\n>>> [ACCEPTANCE GATE PASSED] Single-Arm Pick Baseline Successful! <<<\n")
        return 0
    else:
        print("\n>>> [ACCEPTANCE GATE FAILED] Pick did not meet success criteria. <<<\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
