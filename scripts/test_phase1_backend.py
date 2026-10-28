"""
Phase 1 Validation Script: Backend Foundation & Streaming Engine Verification.

Tests all REST, MJPEG streaming, and WebSocket telemetry endpoints using
FastAPI TestClient.
"""

import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi.testclient import TestClient
from web.server.main import app

def run_tests():
    print("=" * 80)
    print("      PHASE 1 VALIDATION: FASTAPI & STREAMING ENGINE VERIFICATION")
    print("=" * 80)

    with TestClient(app) as client:
        # 1. System Health Status
        print("\n[TEST 1] Querying GET /api/system/status...")
        res = client.get("/api/system/status")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        print(f"  --> Status        : {data['status']}")
        print(f"  --> Platform      : {data['host']['platform']}")
        print(f"  --> Memory Used   : {data['host']['memory_used_gb']} GB / {data['host']['memory_total_gb']} GB")
        print(f"  --> Sim Cameras   : {data['simulation']['cameras']}")
        assert data["status"] == "online"
        assert len(data["simulation"]["cameras"]) == 7
        print("  [PASS] System status endpoint verified.")

        # 2. Camera Feeds Listing
        print("\n[TEST 2] Querying GET /api/stream/cameras...")
        res = client.get("/api/stream/cameras")
        assert res.status_code == 200
        cam_data = res.json()
        expected_cams = ["front_cam", "iso_cam", "overhead_cam", "shelf_cam", "handover_cam", "arm1_wrist_cam", "arm2_wrist_cam"]
        for c in expected_cams:
            assert c in cam_data["cameras"], f"Missing camera {c}"
        print(f"  --> Registered {cam_data['count']} cameras: {cam_data['cameras']}")
        print(f"  --> Resolution: {cam_data['resolution']['width']}x{cam_data['resolution']['height']}")
        print("  [PASS] Camera registry verified.")

        # 3. Snapshot Rendering (JPEG Format & Non-blocking)
        print("\n[TEST 3] Querying GET /api/stream/snapshot/{cam} across all cameras...")
        for cam in expected_cams:
            t0 = time.time()
            res = client.get(f"/api/stream/snapshot/{cam}")
            dt = time.time() - t0
            assert res.status_code == 200
            assert res.headers["content-type"] == "image/jpeg"
            # Verify JPEG SOI marker 0xFF 0xD8
            assert res.content[:2] == b"\xff\xd8", f"Invalid JPEG magic bytes for {cam}"
            print(f"  --> Camera '{cam}': {len(res.content)} bytes in {dt*1000:.1f} ms")
        print("  [PASS] All 5 offscreen camera snapshots verified.")

        # 4. Telemetry REST Endpoint
        print("\n[TEST 4] Querying GET /api/telemetry/current...")
        res = client.get("/api/telemetry/current")
        assert res.status_code == 200
        t_data = res.json()
        print(f"  --> FSM State     : {t_data['fsm_state']}")
        print(f"  --> Arm 1 (6-DOF) : {t_data['arm1']['joint_positions']}")
        print(f"  --> Arm 2 (6-DOF) : {t_data['arm2']['joint_positions']}")
        print(f"  --> Contact Forces: F1={t_data['arm1']['contact_force']} N, F2={t_data['arm2']['contact_force']} N")
        print(f"  --> Object Pos    : {t_data['target_object_pos']}")
        assert len(t_data["arm1"]["joint_positions"]) == 6
        assert len(t_data["arm2"]["joint_positions"]) == 6
        print("  [PASS] Telemetry REST payload verified.")

        # 5. Domain Randomization & Reset APIs
        print("\n[TEST 5] Testing POST /api/fsm/randomize and /api/fsm/reset...")
        res_rand = client.post("/api/fsm/randomize")
        assert res_rand.status_code == 200
        rand_data = res_rand.json()
        print(f"  --> Randomize Result: {rand_data['message']}, Yaw={rand_data.get('yaw_deg')} deg")
        assert rand_data["status"] == "ok"

        res_reset = client.post("/api/fsm/reset")
        assert res_reset.status_code == 200
        reset_data = res_reset.json()
        print(f"  --> Reset Result    : {reset_data['message']}")
        assert reset_data["status"] == "ok"
        print("  [PASS] FSM control endpoints verified.")

        # 6. WebSocket Live Stream Verification
        print("\n[TEST 6] Testing WS /ws/telemetry connection & high-frequency broadcast...")
        with client.websocket_connect("/ws/telemetry") as ws:
            packets = []
            for i in range(5):
                packet = ws.receive_json()
                packets.append(packet)
                print(f"  --> Packet #{i+1}: t={packet['sim_time']}s, State={packet['fsm_state']}, F1={packet['arm1']['contact_force']}N, F2={packet['arm2']['contact_force']}N")
            assert len(packets) == 5
            # Test ping/pong
            ws.send_text('{"event": "ping"}')
            pong_msg = ws.receive_text()
            print(f"  --> Ping Response : {pong_msg}")
            assert "pong" in pong_msg
        print("  [PASS] WebSocket telemetry streaming verified.")

        # 7. Static Dashboard Delivery
        print("\n[TEST 7] Testing GET / (Static Mission Control Dashboard)...")
        res_page = client.get("/")
        assert res_page.status_code == 200
        assert "Bimanual UR10e Robotics Platform" in res_page.text
        assert "cameraFeed" in res_page.text
        assert "ws/telemetry" in res_page.text
        assert "recordBtn" in res_page.text
        print(f"  --> Delivered Dashboard HTML: {len(res_page.text)} bytes")
        print("  [PASS] Static frontend dashboard delivery verified.")

        # 8. Video Recording API Verification
        print("\n[TEST 8] Testing Video Recording APIs (/api/stream/record)...")
        res_start = client.post("/api/stream/record/start?camera_name=front_cam")
        assert res_start.status_code == 200
        start_data = res_start.json()
        assert start_data["status"] == "ok"
        print(f"  --> Record Started : {start_data['filename']}")

        # Trigger 5 frame renders
        for _ in range(5):
            client.get("/api/stream/snapshot/front_cam")

        res_status = client.get("/api/stream/record/status")
        assert res_status.status_code == 200
        status_data = res_status.json()
        assert status_data["is_recording"] is True
        print(f"  --> Record Status  : {status_data['frames']} frames recorded, {status_data['duration_sec']}s")

        res_stop = client.post("/api/stream/record/stop")
        assert res_stop.status_code == 200
        stop_data = res_stop.json()
        assert stop_data["status"] == "ok"
        print(f"  --> Record Stopped : {stop_data['frames']} frames, {stop_data['size_mb']} MB -> {stop_data['filepath']}")
        assert os.path.exists(stop_data["filepath"])
        # Clean up test recording file
        try:
            os.remove(stop_data["filepath"])
        except Exception:
            pass
        print("  [PASS] Video recording lifecycle verified.")

    print("\n" + "=" * 80)
    print("  ALL PHASE 1 BACKEND & STREAMING ENGINE TESTS PASSED SUCCESSFULLY! ")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
