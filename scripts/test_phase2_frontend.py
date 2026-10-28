"""
Automated Test Suite for Phase 2: Mission Control Frontend & Teleoperation.

Validates:
1. Complete design system static delivery (HTML, CSS, JS).
2. 16-State Interactive Flowchart node hierarchy in the DOM.
3. Dual Radial Force Gauges and Interlock Gate elements.
4. Teleoperation REST Endpoints (Cartesian Jog, Joint Jog, Gripper Toggle).
5. Picture-in-Picture (PIP) camera elements.
"""

import os
import sys
import time

# Ensure project root in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi.testclient import TestClient
from web.server.main import app


def run_phase2_tests():
    print("=" * 80)
    print("      PHASE 2 VALIDATION: MISSION CONTROL FRONTEND & TELEOPERATION")
    print("=" * 80)

    with TestClient(app) as client:
        # 1. Static Web Assets Verification
        print("\n[TEST 1] Verifying static web assets delivery...")
        res_html = client.get("/")
        assert res_html.status_code == 200
        html = res_html.text
        assert "Bimanual UR10e Robotics Platform" in html
        print(f"  --> index.html delivered: {len(html)} bytes")

        res_css = client.get("/css/style.css")
        assert res_css.status_code == 200
        css = res_css.text
        assert "--accent-cyan" in css
        assert ".fsm-node" in css
        assert ".radial-svg-wrap" in css
        print(f"  --> css/style.css delivered: {len(css)} bytes")

        res_js = client.get("/js/mission_control.js")
        assert res_js.status_code == 200
        js = res_js.text
        assert "FSM_STATES_ORDER" in js
        assert "updateForceGauges" in js
        assert "jogCartesian" in js
        print(f"  --> js/mission_control.js delivered: {len(js)} bytes")
        print("  [PASS] Static assets verified.")

        # 2. 16-State Interactive FSM Flowchart DOM Nodes
        print("\n[TEST 2] Verifying 16-State FSM Flowchart nodes in DOM...")
        expected_fsm_nodes = [
            "fsm_node_IDLE",
            "fsm_node_SCAN",
            "fsm_node_PLAN_GRASP",
            "fsm_node_APPROACH_1",
            "fsm_node_GRASP_1",
            "fsm_node_LIFT_1",
            "fsm_node_RETRACT_1",
            "fsm_node_TRANSFER_1",
            "fsm_node_PRE_HANDOVER_2",
            "fsm_node_APPROACH_2",
            "fsm_node_DUAL_GRASP",
            "fsm_node_RELEASE_1",
            "fsm_node_RETRACT_CLEAR_1",
            "fsm_node_TRANSPORT_2",
            "fsm_node_PLACE_2",
            "fsm_node_RETRACT_2",
            "fsm_node_COMPLETED",
        ]
        for node_id in expected_fsm_nodes:
            assert f'id="{node_id}"' in html, f"Missing FSM node: {node_id}"
        print(f"  --> All {len(expected_fsm_nodes)} FSM state nodes confirmed in HTML hierarchy.")
        print("  [PASS] 16-State Flowchart nodes verified.")

        # 3. Dual Radial Force Gauges & Dual Custody Gate
        print("\n[TEST 3] Verifying Dual Radial Force Gauges & Safety Gate...")
        assert 'id="gaugeBar1"' in html
        assert 'id="gaugeBar2"' in html
        assert 'id="custodyGateCard"' in html
        assert 'id="gateStatusPill"' in html
        print("  --> Radial SVG gauge elements and Interlock gate card confirmed.")
        print("  [PASS] Force gauge components verified.")

        # 4. Teleoperation Cartesian Jog API
        print("\n[TEST 4] Testing Cartesian Jog endpoint (POST /api/control/jog_cartesian)...")
        res_jog_z = client.post(
            "/api/control/jog_cartesian",
            json={"arm": 1, "delta_xyz": [0.0, 0.0, 0.02]}
        )
        assert res_jog_z.status_code == 200
        z_data = res_jog_z.json()
        assert z_data["status"] == "ok"
        print(f"  --> Arm 1 jogged +2cm Z: actual EE = {z_data['actual_ee']}")

        res_jog_y = client.post(
            "/api/control/jog_cartesian",
            json={"arm": 2, "delta_xyz": [0.0, -0.01, 0.0]}
        )
        assert res_jog_y.status_code == 200
        y_data = res_jog_y.json()
        assert y_data["status"] == "ok"
        print(f"  --> Arm 2 jogged -1cm Y: actual EE = {y_data['actual_ee']}")
        print("  [PASS] Cartesian Jog verified.")

        # 5. Teleoperation Joint Jog API
        print("\n[TEST 5] Testing Joint Jog endpoint (POST /api/control/jog_joint)...")
        res_joint = client.post(
            "/api/control/jog_joint",
            json={"arm": 1, "joint_idx": 0, "delta_rad": 0.05}
        )
        assert res_joint.status_code == 200
        j_data = res_joint.json()
        assert j_data["status"] == "ok"
        print(f"  --> Arm 1 J0 (Pan) jogged +0.05 rad: {j_data['message']}")
        print("  [PASS] Joint Jog verified.")

        # 6. Teleoperation Gripper Command API
        print("\n[TEST 6] Testing Gripper Command endpoint (POST /api/control/gripper)...")
        res_clamp = client.post(
            "/api/control/gripper",
            json={"arm": 1, "position": 1.0}
        )
        assert res_clamp.status_code == 200
        clamp_data = res_clamp.json()
        assert clamp_data["status"] == "ok"
        print(f"  --> Arm 1 Gripper: {clamp_data['message']}")

        res_open = client.post(
            "/api/control/gripper",
            json={"arm": 1, "position": 0.0}
        )
        assert res_open.status_code == 200
        open_data = res_open.json()
        assert open_data["status"] == "ok"
        print(f"  --> Arm 1 Gripper: {open_data['message']}")
        print("  [PASS] Gripper control verified.")

        # 7. Picture-in-Picture (PIP) & Camera Toolbar
        print("\n[TEST 7] Verifying PIP & Multi-Camera Toolbar...")
        assert 'id="pipFeed"' in html
        assert 'id="pipLabel"' in html
        assert 'onclick="swapPipCamera()"' in html
        for cam in ["front_cam", "iso_cam", "shelf_cam", "handover_cam", "overhead_cam", "arm1_wrist_cam", "arm2_wrist_cam"]:
            assert f'data-cam="{cam}"' in html
        print("  --> All 7 camera tab selectors and PIP swap trigger confirmed.")
        print("  [PASS] Camera controls verified.")

    print("\n" + "=" * 80)
    print("  ALL PHASE 2 MISSION CONTROL FRONTEND & TELEOPERATION TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    run_phase2_tests()
