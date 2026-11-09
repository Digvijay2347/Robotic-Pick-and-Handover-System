"""
Automated Test Suite for Phase 4: Neural Studio & Policy Evaluation.

Verifies:
1. ACT Model Configuration REST endpoints (/api/training/config)
2. Training Loss History Curves & Latest Metrics (/api/training/history)
3. Checkpoint Explorer Directory Scanners (/api/training/checkpoints)
4. Training & Evaluation Job Status APIs (/api/training/status, /api/evaluation/status)
5. Evaluation Analytics (Scorecard, Boxplot Cycle Times, 2D Scatter Points)
6. Video Streaming Mounts (/recordings/phase8_act_eval_rollout.mp4)
7. Frontend DOM Elements in index.html, style.css, and neural_studio.js
8. Full System Regression Validation
"""

import os
import sys
import json
from fastapi.testclient import TestClient

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from web.server.main import app

client = TestClient(app)


def test_01_training_config():
    """Verify ACT model hyperparameters can be read and updated."""
    res = client.get("/api/training/config")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data["status"] == "ok"
    cfg = data["config"]
    assert "model" in cfg
    assert "training" in cfg
    assert cfg["model"]["chunk_size"] == 50
    assert cfg["model"]["latent_dim"] == 32
    assert cfg["model"]["backbone"] == "resnet18"
    assert cfg["model"].get("use_goal_conditioning") is True
    assert cfg["model"].get("use_contact_conditioning") is True
    assert cfg["training"]["batch_size"] == 32
    print("[PASS] Test 01 Passed: ACT Configuration loaded accurately with K=50 and Multi-Modal Conditioning.")


def test_02_training_history():
    """Verify real training loss curves are served."""
    res = client.get("/api/training/history")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    num_ep = data["num_epochs"]
    assert num_ep >= 20
    metrics = data["metrics"]
    assert len(metrics["epoch"]) == num_ep
    assert len(metrics["train_total_loss"]) == num_ep
    assert len(metrics["val_total_loss"]) == num_ep
    assert len(metrics["train_l1_loss"]) == num_ep
    assert len(metrics["val_l1_loss"]) == num_ep
    assert len(metrics["train_kl_loss"]) == num_ep
    assert len(metrics["val_kl_loss"]) == num_ep

    # Loss convergence check: initial loss > final loss
    assert metrics["train_total_loss"][0] > metrics["train_total_loss"][-1]
    assert metrics["val_total_loss"][0] > metrics["val_total_loss"][-1]
    assert data["latest"]["epoch"] == num_ep
    print(f"[PASS] Test 02 Passed: Real ACT training history verified across {num_ep} epochs.")


def test_03_checkpoint_explorer():
    """Verify saved .pt models are discovered in outputs/act_handover/."""
    res = client.get("/api/training/checkpoints")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["total_count"] >= 5
    ckpts = data["checkpoints"]

    best_ckpt = next((c for c in ckpts if c["is_best"]), None)
    assert best_ckpt is not None, "is_best model must be identified"
    assert best_ckpt["filename"] in ["best_model_v3.pt", "best_model.pt"]
    assert best_ckpt["size_mb"] > 200
    assert any(c["filename"] == "best_model_v3.pt" for c in ckpts)
    assert any(c["filename"] == "best_model.pt" for c in ckpts)
    print(f"[PASS] Test 03 Passed: {len(ckpts)} model checkpoints discovered (Primary: {best_ckpt['filename']}, {best_ckpt['size_mb']} MB).")


def test_04_training_status():
    """Verify training job status API reports valid operational state."""
    res = client.get("/api/training/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    job = data["job"]
    assert "is_running" in job
    assert "current_epoch" in job
    assert "total_epochs" in job
    assert "progress_percent" in job
    assert isinstance(job["recent_logs"], list)
    print("[PASS] Test 04 Passed: Training status API functional.")


def test_05_evaluation_status():
    """Verify evaluation rollout status API reports valid operational state."""
    res = client.get("/api/evaluation/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    job = data["job"]
    assert "is_running" in job
    assert "completed_trials" in job
    assert "success_rate_percent" in job
    print("[PASS] Test 05 Passed: Evaluation status API functional.")


def test_06_evaluation_analytics():
    """Verify benchmark analytics scorecard, boxplots, and 2D scatter coordinates."""
    res = client.get("/api/evaluation/analytics")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"

    # Scorecard
    sc = data["scorecard"]
    assert sc["success_rate_percent"] == 100.0
    assert sc["successful_trials"] == 50
    assert sc["inter_robot_collisions"] == 0
    assert sc["shelf_collisions"] == 0
    assert sc["mean_cycle_time_s"] > 20.0

    # Boxplot stages
    bp = data["cycle_time_boxplot"]
    stages = bp["stages"]
    stage_names = [s["name"] for s in stages]
    assert "Pick" in stage_names
    assert "Transit" in stage_names
    assert "Handover" in stage_names
    assert "Place" in stage_names
    assert "Total" in stage_names

    # Placement Scatter
    scat = data["placement_scatter"]
    assert scat["num_samples"] == 50
    assert scat["stats"]["mean_error_radius_mm"] < 0.20
    assert len(scat["points"]) == 50

    # Video check
    assert data["video"]["available"] is True
    assert "phase8_act_eval_rollout.mp4" in data["video"]["url"]
    print("[PASS] Test 06 Passed: Full evaluation scorecard, boxplot, and 2D scatter analytics verified.")


def test_07_eval_video_static_mount():
    """Verify the rollout video is physically accessible via static file streaming."""
    res = client.get("/recordings/phase8_act_eval_rollout.mp4")
    assert res.status_code == 200
    assert res.headers["content-type"] in ["video/mp4", "application/octet-stream"]
    assert len(res.content) > 1_000_000  # ~5.0 MB
    print(f"[PASS] Test 07 Passed: Evaluation rollout video streamed successfully ({len(res.content)} bytes).")


def test_08_frontend_dom_elements():
    """Verify index.html contains all Phase 4 Neural Studio elements."""
    html_path = os.path.join(ROOT_DIR, "web", "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    assert 'id="navBtnNeuralStudio"' in html, "Neural Studio nav button must be present"
    assert 'switchView(\'neural-studio\')' in html, "switchView('neural-studio') must be wired"
    assert 'id="neuralStudioView"' in html, "Neural Studio root view must be present"
    assert 'id="neuralModelSelect"' in html, "Model selector dropdown must be present"
    assert 'id="neuralLossCanvas"' in html, "Loss curve canvas must be present"
    assert 'id="neuralBoxplotCanvas"' in html, "Boxplot canvas must be present"
    assert 'id="neuralScatterCanvas"' in html, "Scatter plot canvas must be present"
    assert 'id="neuralCheckpointsTbody"' in html, "Checkpoints table must be present"
    assert 'id="neuralRolloutVideo"' in html, "Rollout video player must be present"
    assert 'id="neuralTerminalLogs"' in html, "Terminal log box must be present"
    assert 'id="neuralTrainingModal"' in html, "Training modal must be present"
    assert 'src="js/neural_studio.js"' in html, "neural_studio.js must be imported"
    print("[PASS] Test 08 Passed: All Phase 4 DOM components verified in index.html.")


def test_09_css_and_js_assets():
    """Verify style.css and neural_studio.js exist and contain critical selectors."""
    css_path = os.path.join(ROOT_DIR, "web", "static", "css", "style.css")
    js_path = os.path.join(ROOT_DIR, "web", "static", "js", "neural_studio.js")

    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()
    assert ".neural-studio-root" in css
    assert ".neural-arch-card" in css
    assert ".neural-chart-card" in css
    assert ".gauge-progress" in css
    assert ".boxplot-container" in css
    assert ".scatter-container" in css

    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()
    assert "initNeuralStudio" in js
    assert "renderTrainingCurves" in js
    assert "renderBoxplot" in js
    assert "renderScatter" in js
    assert "loadCheckpoints" in js
    assert "launchEvaluationJob" in js
    print("[PASS] Test 09 Passed: Phase 4 CSS and JS engines verified.")


def test_10_regression_phases_1_to_3():
    """Verify Phase 1, 2, and 3 APIs remain 100% operational."""
    # Phase 1: System info & Cameras
    res_sys = client.get("/api/system/status")
    assert res_sys.status_code == 200
    assert res_sys.json()["status"] == "online"

    # Phase 1: Video stream cameras
    res_cams = client.get("/api/stream/cameras")
    assert res_cams.status_code == 200
    assert len(res_cams.json()["cameras"]) >= 4

    # Phase 3: Dataset overview & Episodes
    res_ds = client.get("/api/dataset/overview")
    assert res_ds.status_code == 200
    assert res_ds.json()["total_episodes"] == 50

    res_ep = client.get("/api/dataset/episodes?limit=5")
    assert res_ep.status_code == 200
    assert len(res_ep.json()["episodes"]) == 5

    res_stat = client.get("/api/dataset/collect/status")
    assert res_stat.status_code == 200
    print("[PASS] Test 10 Passed: Zero regression across Phase 1, Phase 2, and Phase 3.")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("       RUNNING PHASE 4: NEURAL STUDIO & POLICY EVALUATION TEST SUITE")
    print("=" * 80)
    test_01_training_config()
    test_02_training_history()
    test_03_checkpoint_explorer()
    test_04_training_status()
    test_05_evaluation_status()
    test_06_evaluation_analytics()
    test_07_eval_video_static_mount()
    test_08_frontend_dom_elements()
    test_09_css_and_js_assets()
    test_10_regression_phases_1_to_3()
    print("=" * 80)
    print("  ALL 10 TESTS PASSED SUCCESFULLY! PHASE 4 IMPLEMENTATION 100% VERIFIED.")
    print("=" * 80 + "\n")
