"""
Phase 4 Validation Script: Coordinated Dual-Arm Handover State Machine (Scripted Expert).

Executes the complete autonomous dual-arm manipulation pipeline:
1. Perception-driven 6-DOF grasp identification and shelf retrieval with Arm 1 (Giver).
2. Transport of the target object to the central handover zone (P = [0.0, 0.1, 1.05]).
3. Lateral intercept with Arm 2 (Receiver) and verified dual-grasp overlapping custody hold.
4. Clean release by Arm 1 and vertical retraction to prevent obstacle collision.
5. Arc transport by Arm 2 to the storage bin (P = [0.0, -0.8, 0.45]) and release.
6. Return to home postures and quantitative mission telemetry verification.

Usage:
    python scripts/04_run_scripted_handover.py [--viewer] [--speed SPEED] [--cycles N] [--save-snapshots]
"""

import os
import sys
import argparse
import time
from typing import List, Optional
import numpy as np

# Ensure workspace root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import mujoco
try:
    import mujoco.viewer
except Exception:
    pass
from fsm.handover_states import HandoverState, CycleSummary
from fsm.handover_fsm import HandoverFSM


def save_camera_snapshot(model: mujoco.MjModel, data: mujoco.MjData, cam_name: str, out_path: str):
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
        print(f"  [WARN] Failed to capture snapshot '{cam_name}': {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Coordinated Dual-Arm Handover State Machine Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--viewer", action="store_true", help="Launch interactive passive MuJoCo viewer")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation playback speed multiplier in viewer")
    parser.add_argument("--cycles", type=int, default=1, help="Number of consecutive pick-and-handover cycles to benchmark")
    parser.add_argument("--save-snapshots", action="store_true", help="Capture offscreen camera snapshots at key milestones")
    args = parser.parse_args()

    scene_path = os.path.join(ROOT_DIR, "models", "dual_ur10e_scene.xml")
    if not os.path.exists(scene_path):
        print(f"[FATAL] Scene file not found: {scene_path}")
        sys.exit(1)

    print("=" * 80)
    print("      PHASE 4: COORDINATED DUAL-ARM HANDOVER STATE MACHINE BENCHMARK")
    print("=" * 80)
    print(f"  Scene Model    : {scene_path}")
    print(f"  Total Cycles   : {args.cycles}")
    print(f"  Interactive    : {args.viewer} (speed={args.speed}x)")
    print(f"  Save Snapshots : {args.save_snapshots}")
    print("=" * 80 + "\n")

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    rec_dir = os.path.join(ROOT_DIR, "recordings")
    if args.save_snapshots:
        os.makedirs(rec_dir, exist_ok=True)

    fsm = HandoverFSM(model, data)
    cycle_summaries: List[CycleSummary] = []

    def run_all(viewer_handle=None):
        for cycle_i in range(1, args.cycles + 1):
            print(f"\n>>> STARTING CYCLE {cycle_i}/{args.cycles} <<<")

            # Milestone callback for snapshots
            def milestone_cb(state_name: str):
                if not args.save_snapshots:
                    return
                # Map key states to snapshot files
                snapshot_map = {
                    "APPROACH_1": ("shelf_cam", f"phase4_c{cycle_i}_01_approach_shelf.png"),
                    "RETRACT_1": ("shelf_cam", f"phase4_c{cycle_i}_02_retract_shelf.png"),
                    "DUAL_GRASP": ("handover_cam", f"phase4_c{cycle_i}_03_dual_grasp.png"),
                    "RETRACT_CLEAR_1": ("handover_cam", f"phase4_c{cycle_i}_04_arm1_retract.png"),
                    "PLACE_2": ("overhead_cam", f"phase4_c{cycle_i}_05_place_bin.png"),
                    "COMPLETED": ("overhead_cam", f"phase4_c{cycle_i}_06_cycle_completed.png"),
                }
                if state_name in snapshot_map:
                    cam, fname = snapshot_map[state_name]
                    out_path = os.path.join(rec_dir, fname)
                    save_camera_snapshot(model, data, cam, out_path)
                    print(f"  [SNAPSHOT] Saved {fname} (from '{cam}')")

            summary = fsm.run_cycle(
                cycle_idx=cycle_i,
                viewer=viewer_handle,
                speed=args.speed,
                milestone_callback=milestone_cb,
            )
            cycle_summaries.append(summary)

            # Print single-cycle evaluation report
            status_str = "SUCCESS" if summary.success else "FAILED"
            print(f"\n--- CYCLE {cycle_i} RESULT: {status_str} ---")
            print(f"  Duration              : {summary.total_duration_sec:.2f} s")
            print(f"  Dual Custody Hold     : {summary.dual_hold_duration_sec:.2f} s")
            print(f"  Mean Grip Force Arm 1 : {summary.arm1_grip_force_mean:.1f} N")
            print(f"  Mean Grip Force Arm 2 : {summary.arm2_grip_force_mean:.1f} N")
            print(f"  Final Box Position    : {np.round(summary.final_object_pos, 4)}")
            print(f"  Distance to Bin Center: {summary.distance_to_bin_center_cm:.1f} cm (threshold: < 15.0 cm)")
            print(f"  Inter-Arm Collision   : {'YES (FAIL)' if summary.inter_arm_collision_detected else 'None (PASS)'}")
            print(f"  Shelf Collision       : {'YES (FAIL)' if summary.shelf_collision_detected else 'None (PASS)'}")

    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run_all(viewer)
            print("\n[VIEWER] Simulation completed. Keeping viewer open (close window to exit)...")
            try:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
            except KeyboardInterrupt:
                pass
    else:
        run_all(None)

    # =========================================================================
    # Aggregated Benchmark Report
    # =========================================================================
    n_total = len(cycle_summaries)
    n_success = sum(1 for c in cycle_summaries if c.success)
    success_rate = (n_success / n_total) * 100.0 if n_total > 0 else 0.0

    durations = [c.total_duration_sec for c in cycle_summaries]
    bin_dists = [c.distance_to_bin_center_cm for c in cycle_summaries]
    f1_means = [c.arm1_grip_force_mean for c in cycle_summaries if c.arm1_grip_force_mean > 0]
    f2_means = [c.arm2_grip_force_mean for c in cycle_summaries if c.arm2_grip_force_mean > 0]
    coll_inter_count = sum(1 for c in cycle_summaries if c.inter_arm_collision_detected)
    coll_shelf_count = sum(1 for c in cycle_summaries if c.shelf_collision_detected)

    print("\n" + "=" * 80)
    print("                    PHASE 4 BENCHMARK SUMMARY REPORT")
    print("=" * 80)
    print(f"  Total Cycles Evaluated        : {n_total}")
    print(f"  Successful Cycles             : {n_success}")
    print(f"  Success Rate                  : {success_rate:.1f}% (Gate: >= 90.0%)")
    print(f"  Mean Cycle Duration           : {np.mean(durations):.2f} +/- {np.std(durations):.2f} s")
    print(f"  Mean Final Distance to Bin    : {np.mean(bin_dists):.1f} +/- {np.std(bin_dists):.1f} cm")
    if f1_means:
        print(f"  Mean Contact Force Arm 1      : {np.mean(f1_means):.1f} N")
    if f2_means:
        print(f"  Mean Contact Force Arm 2      : {np.mean(f2_means):.1f} N")
    print(f"  Inter-Arm Collisions Detected : {coll_inter_count}")
    print(f"  Shelf Collisions Detected     : {coll_shelf_count}")
    print("=" * 80)

    # Verification gate check
    if success_rate >= 90.0:
        print("\n>>> [VERIFICATION PASSED] Phase 4 Coordinated Handover Meets All Quality Gates! <<<\n")
        sys.exit(0)
    else:
        print(f"\n>>> [VERIFICATION FAILED] Success rate ({success_rate:.1f}%) fell below 90% gate! <<<\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
