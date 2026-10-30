"""
Unit & Integration Test Suite for Phase 3: Dataset Studio.

Validates:
1. Dataset Overview API (/api/dataset/overview)
2. 50-Episode Catalog API (/api/dataset/episodes)
3. Trajectory Slice API (/api/dataset/episode/0)
4. Demonstration Recordings API (/api/dataset/recordings)
5. Static MP4 Streaming Delivery (/recordings/phase6_demo_049.mp4)
6. Collection Job Engine Status API (/api/dataset/collect/status)
7. HTML & DOM Structure for Dataset Studio Views & Scripts
"""

import os
import sys
from fastapi.testclient import TestClient

# Ensure root prod2 directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from web.server.main import app

client = TestClient(app)


def test_dataset_overview():
    print("[1/7] Testing GET /api/dataset/overview...")
    res = client.get("/api/dataset/overview")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data.get("status") == "ready", f"Expected ready, got {data.get('status')}"
    assert data.get("total_episodes") == 50, f"Expected 50 episodes, got {data.get('total_episodes')}"
    assert data.get("total_frames") == 32450, f"Expected 32450 frames, got {data.get('total_frames')}"
    assert data.get("size_mb", 0) > 500, f"Expected size > 500MB, got {data.get('size_mb')}"
    assert len(data.get("action_distributions", [])) == 14, f"Expected 14 actuator actions, got {len(data.get('action_distributions', []))}"
    assert len(data.get("cameras", [])) == 3, f"Expected 3 synced cameras, got {len(data.get('cameras', []))}"
    print(f"  --> PASSED: 50 episodes, {data['total_frames']} frames, {data['size_mb']} MB, 14 DOF actions verified.")


def test_dataset_episodes_catalog():
    print("[2/7] Testing GET /api/dataset/episodes...")
    res = client.get("/api/dataset/episodes?limit=50")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data.get("total") == 50, f"Expected total 50, got {data.get('total')}"
    episodes = data.get("episodes", [])
    assert len(episodes) == 50, f"Expected 50 episodes in page, got {len(episodes)}"
    ep0 = episodes[0]
    assert ep0.get("episode_id") == "demo_0", f"Unexpected episode_id {ep0.get('episode_id')}"
    assert ep0.get("steps") == 649, f"Unexpected steps {ep0.get('steps')}"
    assert ep0.get("dual_hold_confirmed") is True, "dual_hold_confirmed should be True"
    assert "distance_to_bin_cm" in ep0, "distance_to_bin_cm missing"
    print(f"  --> PASSED: 50 catalog rows verified. Ep 0: {ep0['steps']} steps, dual custody confirmed.")


def test_single_episode_trajectory():
    print("[3/7] Testing GET /api/dataset/episode/0...")
    res = client.get("/api/dataset/episode/0")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data.get("episode_id") == "demo_0"
    assert data.get("total_steps") == 649
    assert len(data.get("action_trajectory", [])) > 0
    assert len(data.get("decimated_steps", [])) == len(data.get("action_trajectory", []))
    print(f"  --> PASSED: Ep 0 trajectory loaded with {len(data['action_trajectory'])} decimated checkpoints.")


def test_recordings_list():
    print("[4/7] Testing GET /api/dataset/recordings...")
    res = client.get("/api/dataset/recordings")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    recordings = data.get("recordings", [])
    assert len(recordings) > 0, "Expected at least 1 recording"
    filenames = [r["filename"] for r in recordings]
    assert "phase6_demo_049.mp4" in filenames, f"phase6_demo_049.mp4 not found in {filenames}"
    print(f"  --> PASSED: {len(recordings)} MP4 preview files found ({', '.join(filenames[:3])}...).")


def test_static_video_streaming():
    print("[5/7] Testing GET /recordings/phase6_demo_049.mp4...")
    res = client.get("/recordings/phase6_demo_049.mp4")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "video/mp4" in res.headers.get("content-type", "")
    content_len = len(res.content)
    assert content_len > 100_000, f"Video content too small: {content_len} bytes"
    print(f"  --> PASSED: Served video/mp4 with size {content_len / (1024*1024):.2f} MB.")


def test_collection_job_status():
    print("[6/7] Testing GET /api/dataset/collect/status...")
    res = client.get("/api/dataset/collect/status")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert "is_running" in data
    assert "workers" in data
    assert "completed_episodes" in data
    assert "progress_percent" in data
    print(f"  --> PASSED: Collection Job status online. Running={data['is_running']}, Progress={data['progress_percent']}%.")


def test_index_html_dom_elements():
    print("[7/7] Testing GET / and DOM structure for Dataset Studio...")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    html = res.text
    assert 'id="datasetStudioView"' in html, "datasetStudioView container missing in index.html"
    assert 'id="missionControlView"' in html, "missionControlView container missing in index.html"
    assert 'id="navBtnDatasetStudio"' in html, "navBtnDatasetStudio button missing in index.html"
    assert 'id="episodesTableBody"' in html, "episodesTableBody missing in index.html"
    assert 'id="demoVideoPlayer"' in html, "demoVideoPlayer missing in index.html"
    assert 'id="actionBarsFlex"' in html, "actionBarsFlex missing in index.html"
    assert 'id="workerMatrixContainer"' in html, "workerMatrixContainer missing in index.html"
    assert 'src="js/dataset_studio.js"' in html, "dataset_studio.js script tag missing in index.html"
    print("  --> PASSED: All Obsidian Dataset Studio DOM containers, IDs, and script tags validated.")


if __name__ == "__main__":
    print("=" * 70)
    print("  RUNNING PHASE 3: DATASET STUDIO INTEGRATION TESTS")
    print("=" * 70)
    test_dataset_overview()
    test_dataset_episodes_catalog()
    test_single_episode_trajectory()
    test_recordings_list()
    test_static_video_streaming()
    test_collection_job_status()
    test_index_html_dom_elements()
    print("=" * 70)
    print("  ALL 7 PHASE 3 DATASET STUDIO TESTS PASSED FLAWLESSLY!")
    print("=" * 70)
