/**
 * BIMANUAL UR10e MISSION CONTROL - INTERACTIVE ENGINE (PHASE 2)
 * High-frequency WebSocket Telemetry, 16-State FSM Flowchart, Radial Gauges & Teleoperation
 */

// 16-State FSM Master Sequence
const FSM_STATES_ORDER = [
    { key: "IDLE",            idx: "01", name: "IDLE",           phase: "1", label: "Ready Home",        desc: "Arms in stable home posture", criteria: "Tracking error < 2mm" },
    { key: "SCAN",            idx: "02", name: "SCAN",           phase: "1", label: "3D Point Cloud",    desc: "RGB-D camera point cloud capture", criteria: "Pinhole backprojection" },
    { key: "PLAN_GRASP",      idx: "03", name: "PLAN_GRASP",     phase: "1", label: "Grasp Synthesis",   desc: "6-DOF antipodal grasp candidates", criteria: "Centroid & collision check" },
    { key: "APPROACH_1",      idx: "04", name: "APPROACH_1",     phase: "1", label: "Shelf Standoff",    desc: "Arm 1 moves to shelf entrance", criteria: "15cm standoff distance" },
    { key: "GRASP_1",         idx: "05", name: "GRASP_1",        phase: "1", label: "Object Clamp",      desc: "Arm 1 closes gripper on red block", criteria: "Contact force >= 15.0 N" },
    { key: "LIFT_1",          idx: "06", name: "LIFT_1",         phase: "1", label: "Friction Break",    desc: "Vertical lift +5cm off shelf", criteria: "Z altitude delta >= +0.04m" },
    { key: "RETRACT_1",       idx: "07", name: "RETRACT_1",      phase: "1", label: "Shelf Extract",     desc: "Horizontal extraction clear of frame", criteria: "X distance clear of shelf" },
    { key: "TRANSFER_1",      idx: "08", name: "TRANSFER_1",     phase: "2", label: "Handover Transit",  desc: "Arm 1 navigates to central handover zone", criteria: "Zone [0.0, 0.45, 1.25]" },
    { key: "PRE_HANDOVER_2",  idx: "09", name: "PRE_HANDOVER_2", phase: "2", label: "Standoff Align",    desc: "Arm 2 reaches pre-handover stance", criteria: "Opposing yaw alignment" },
    { key: "APPROACH_2",      idx: "10", name: "APPROACH_2",     phase: "2", label: "Grasp Enclose",     desc: "Arm 2 surrounds lower half of block", criteria: "Pad distance <= 5mm" },
    { key: "DUAL_GRASP",      idx: "11", name: "DUAL_GRASP",     phase: "3", label: "Dual Custody Gate", desc: "Both grippers clamped simultaneously", criteria: "F1 >= 15N AND F2 >= 15N" },
    { key: "RELEASE_1",       idx: "12", name: "RELEASE_1",      phase: "3", label: "Arm 1 Release",     desc: "Arm 1 opens fingers; custody passes to Arm 2", criteria: "F2 holds full load" },
    { key: "RETRACT_CLEAR_1", idx: "13", name: "RETRACT_CLEAR_1",phase: "3", label: "Arm 1 Retreat",    desc: "Arm 1 ascends safely and returns home", criteria: "Inter-arm distance > 30cm" },
    { key: "TRANSPORT_2",     idx: "14", name: "TRANSPORT_2",    phase: "4", label: "Bin Arc Transit",   desc: "Arm 2 carries block along smooth quintic arc", criteria: "Jerk-limited quintic" },
    { key: "PLACE_2",         idx: "15", name: "PLACE_2",        phase: "4", label: "Bin Deposit",       desc: "Arm 2 opens gripper above bin basin", criteria: "Over bin center [0, -0.8]" },
    { key: "RETRACT_2",       idx: "16", name: "RETRACT_2",      phase: "4", label: "Home Return",       desc: "Arm 2 returns to ready home posture", criteria: "Home posture reached" },
    { key: "COMPLETED",       idx: "17", name: "COMPLETED",      phase: "4", label: "Cycle Complete",    desc: "Handover cycle verified successful", criteria: "Zero slips, zero collisions" },
];

// Global State
let currentCamera = "front_cam";
let pipCamera = "handover_cam";
let activeArm = 1;
let jogStepMeters = 0.01; // 1 cm default
let lastFsmState = "IDLE";
let wsConnection = null;

// Recording state
let isRecording = false;
let mediaRecorder = null;
let recordedChunks = [];
let recordTimerInterval = null;
let recordStartTime = 0;
let recordCanvas = null;
let recordCtx = null;
let recordDrawInterval = null;

// Joint limits for UR10e (radians)
const JOINT_LIMITS = [
    [-6.28, 6.28], // J0 Pan
    [-6.28, 6.28], // J1 Lift
    [-6.28, 6.28], // J2 Elbow
    [-6.28, 6.28], // J3 Wrist 1
    [-6.28, 6.28], // J4 Wrist 2
    [-6.28, 6.28], // J5 Wrist 3
];

const JOINT_NAMES = ["Pan", "Lift", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3"];

// ==========================================================================
// INITIALIZATION
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
    initCameraControls();
    initTeleopControls();
    connectWebSocket();
    logConsole("Mission Control System Initialized. Connecting telemetry...", "log-info");
});

// ==========================================================================
// CAMERA SELECTION & PICTURE-IN-PICTURE (PIP)
// ==========================================================================
function initCameraControls() {
    const tabsContainer = document.getElementById("cameraTabs");
    if (!tabsContainer) return;

    tabsContainer.querySelectorAll(".cam-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            tabsContainer.querySelectorAll(".cam-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            switchMainCamera(btn.dataset.cam);
        });
    });
}

function switchMainCamera(camName) {
    if (camName === currentCamera) return;
    currentCamera = camName;
    const feed = document.getElementById("cameraFeed");
    const camLabel = document.getElementById("camLabel");
    if (feed) feed.src = `/api/stream/feed/${camName}?t=${Date.now()}`;
    if (camLabel) camLabel.textContent = formatCamLabel(camName);
    logConsole(`Viewport switched to [${camName}]`, "log-info");
}

function swapPipCamera() {
    const temp = currentCamera;
    currentCamera = pipCamera;
    pipCamera = temp;

    // Update main feed
    const feed = document.getElementById("cameraFeed");
    const camLabel = document.getElementById("camLabel");
    if (feed) feed.src = `/api/stream/feed/${currentCamera}?t=${Date.now()}`;
    if (camLabel) camLabel.textContent = formatCamLabel(currentCamera);

    // Update PIP feed
    const pipFeed = document.getElementById("pipFeed");
    const pipLabel = document.getElementById("pipLabel");
    if (pipFeed) pipFeed.src = `/api/stream/feed/${pipCamera}?t=${Date.now()}`;
    if (pipLabel) pipLabel.textContent = formatCamLabel(pipCamera);

    // Update active camera tab
    document.querySelectorAll(".cam-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.cam === currentCamera);
    });

    logConsole(`Swapped views: Main=[${currentCamera}], PIP=[${pipCamera}]`, "log-info");
}

function formatCamLabel(name) {
    const map = {
        "front_cam": "3D PERSPECTIVE (FRONT)",
        "iso_cam": "ISOMETRIC 3/4 VIEW",
        "shelf_cam": "SHELF COMPARTMENTS",
        "handover_cam": "HANDOVER ZONE",
        "overhead_cam": "OVERHEAD MAP",
        "arm1_wrist_cam": "ARM 1 WRIST CAM",
        "arm2_wrist_cam": "ARM 2 WRIST CAM",
    };
    return map[name] || name.toUpperCase();
}

function captureSnapshot() {
    window.open(`/api/stream/snapshot/${currentCamera}?quality=95`, "_blank");
    logConsole(`Snapshot captured from [${currentCamera}]`, "log-ok");
}

function toggleFullscreen() {
    const wrapper = document.querySelector(".viewport-wrapper");
    if (!document.fullscreenElement) {
        wrapper.requestFullscreen().catch(err => alert(`Fullscreen error: ${err.message}`));
    } else {
        document.exitFullscreen();
    }
}

// ==========================================================================
// 16-STATE INTERACTIVE FSM DIAGRAM ENGINE
// ==========================================================================
function updateFsmFlowchart(activeStateName) {
    if (activeStateName === lastFsmState) return;

    const fsmActiveBadge = document.getElementById("fsmActiveStage");
    if (fsmActiveBadge) fsmActiveBadge.textContent = activeStateName;

    // Find state index in order
    const targetIdx = FSM_STATES_ORDER.findIndex(s => s.name === activeStateName);

    FSM_STATES_ORDER.forEach((state, i) => {
        const nodeEl = document.getElementById(`fsm_node_${state.name}`);
        if (!nodeEl) return;

        nodeEl.classList.remove("active", "passed");

        if (targetIdx !== -1) {
            if (i < targetIdx) {
                nodeEl.classList.add("passed");
                const icon = nodeEl.querySelector(".fsm-status-icon");
                if (icon) icon.textContent = "✓";
            } else if (i === targetIdx) {
                nodeEl.classList.add("active");
                const icon = nodeEl.querySelector(".fsm-status-icon");
                if (icon) icon.textContent = "●";
                // Auto scroll into view
                nodeEl.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
            } else {
                const icon = nodeEl.querySelector(".fsm-status-icon");
                if (icon) icon.textContent = "○";
            }
        }
    });

    logConsole(`[FSM] State Transition: ${lastFsmState} -> ${activeStateName}`, activeStateName === "COMPLETED" ? "log-ok" : "log-info");
    lastFsmState = activeStateName;
}

function showStateInspector(stateKey) {
    const stateInfo = FSM_STATES_ORDER.find(s => s.key === stateKey);
    if (!stateInfo) return;

    alert(
        `[FSM STAGE ${stateInfo.idx}]: ${stateInfo.name}\n` +
        `-----------------------------------------\n` +
        `Phase        : Phase ${stateInfo.phase}\n` +
        `Description  : ${stateInfo.desc}\n` +
        `Safety Gate  : ${stateInfo.criteria}\n` +
        `Status       : ${stateInfo.name === lastFsmState ? "CURRENTLY ACTIVE" : "STANDBY"}`
    );
}

// ==========================================================================
// RADIAL SVG CONTACT FORCE GAUGES (F1, F2) & DUAL CUSTODY INTERLOCK
// ==========================================================================
function updateForceGauges(f1, f2, dualHoldConfirmed) {
    const maxForce = 200.0;
    const circum = 314; // 2 * pi * r (r=50)

    // Arm 1 Force Gauge
    const f1Clamped = Math.min(Math.max(f1, 0.0), maxForce);
    const offset1 = circum - (f1Clamped / maxForce) * circum;
    const circle1 = document.getElementById("gaugeBar1");
    const num1 = document.getElementById("gaugeNum1");
    const status1 = document.getElementById("gaugeStatus1");

    if (circle1) {
        circle1.style.strokeDashoffset = offset1;
        circle1.style.stroke = f1Clamped >= 50.0 ? "var(--accent-green)" : (f1Clamped >= 15.0 ? "var(--accent-amber)" : "var(--accent-cyan)");
    }
    if (num1) num1.textContent = f1.toFixed(1);
    if (status1) {
        status1.textContent = f1 >= 50.0 ? "SECURE CLAMP (>=50N)" : (f1 >= 10.0 ? "CONTACT ENGAGED" : "OPEN / NO CONTACT");
        status1.style.color = f1 >= 50.0 ? "var(--accent-green)" : (f1 >= 10.0 ? "var(--accent-amber)" : "var(--text-secondary)");
    }

    // Arm 2 Force Gauge
    const f2Clamped = Math.min(Math.max(f2, 0.0), maxForce);
    const offset2 = circum - (f2Clamped / maxForce) * circum;
    const circle2 = document.getElementById("gaugeBar2");
    const num2 = document.getElementById("gaugeNum2");
    const status2 = document.getElementById("gaugeStatus2");

    if (circle2) {
        circle2.style.strokeDashoffset = offset2;
        circle2.style.stroke = f2Clamped >= 50.0 ? "var(--accent-green)" : (f2Clamped >= 15.0 ? "var(--accent-amber)" : "var(--accent-cyan)");
    }
    if (num2) num2.textContent = f2.toFixed(1);
    if (status2) {
        status2.textContent = f2 >= 50.0 ? "SECURE CLAMP (>=50N)" : (f2 >= 10.0 ? "CONTACT ENGAGED" : "OPEN / NO CONTACT");
        status2.style.color = f2 >= 50.0 ? "var(--accent-green)" : (f2 >= 10.0 ? "var(--accent-amber)" : "var(--text-secondary)");
    }

    // Dual Custody Interlock Gate Status
    const gateCard = document.getElementById("custodyGateCard");
    const gatePill = document.getElementById("gateStatusPill");
    const gateIcon = document.getElementById("gateIcon");

    if (dualHoldConfirmed || (f1 >= 15.0 && f2 >= 15.0)) {
        if (gateCard) gateCard.classList.add("confirmed");
        if (gatePill) {
            gatePill.textContent = "DUAL CUSTODY CONFIRMED";
            gatePill.style.color = "var(--accent-green)";
        }
        if (gateIcon) gateIcon.textContent = "🔓";
    } else if (f1 >= 10.0 || f2 >= 10.0) {
        if (gateCard) gateCard.classList.remove("confirmed");
        if (gatePill) {
            gatePill.textContent = "INTERLOCK ENGAGED";
            gatePill.style.color = "var(--accent-amber)";
        }
        if (gateIcon) gateIcon.textContent = "⚡";
    } else {
        if (gateCard) gateCard.classList.remove("confirmed");
        if (gatePill) {
            gatePill.textContent = "STANDBY (SINGLE ARM)";
            gatePill.style.color = "var(--text-muted)";
        }
        if (gateIcon) gateIcon.textContent = "🔒";
    }
}

// ==========================================================================
// 14-DOF JOINT ANGLE PROGRESS BARS & KINEMATICS
// ==========================================================================
function updateJointsUI(q1, q2, ee1, ee2) {
    // Populate or update Arm 1 joints
    const container1 = document.getElementById("arm1JointsList");
    if (container1 && container1.children.length === 0) {
        container1.innerHTML = JOINT_NAMES.map((name, i) => `
            <div class="joint-row">
                <span class="j-label">J${i} ${name}</span>
                <div class="j-bar-wrap"><div class="j-bar-fill" id="jbar1_${i}"></div></div>
                <span class="j-val" id="jval1_${i}">0.000 rad</span>
            </div>
        `).join("");
    }

    if (q1 && q1.length === 6) {
        q1.forEach((val, i) => {
            const bar = document.getElementById(`jbar1_${i}`);
            const text = document.getElementById(`jval1_${i}`);
            const [min, max] = JOINT_LIMITS[i];
            const pct = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
            if (bar) bar.style.width = `${pct}%`;
            if (text) text.textContent = `${val.toFixed(3)} rad`;
        });
    }

    // Populate or update Arm 2 joints
    const container2 = document.getElementById("arm2JointsList");
    if (container2 && container2.children.length === 0) {
        container2.innerHTML = JOINT_NAMES.map((name, i) => `
            <div class="joint-row">
                <span class="j-label">J${i} ${name}</span>
                <div class="j-bar-wrap"><div class="j-bar-fill" id="jbar2_${i}"></div></div>
                <span class="j-val" id="jval2_${i}">0.000 rad</span>
            </div>
        `).join("");
    }

    if (q2 && q2.length === 6) {
        q2.forEach((val, i) => {
            const bar = document.getElementById(`jbar2_${i}`);
            const text = document.getElementById(`jval2_${i}`);
            const [min, max] = JOINT_LIMITS[i];
            const pct = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
            if (bar) bar.style.width = `${pct}%`;
            if (text) text.textContent = `${val.toFixed(3)} rad`;
        });
    }

    // End-Effector Badges
    const eeBadge1 = document.getElementById("arm1Ee");
    const eeBadge2 = document.getElementById("arm2Ee");
    if (eeBadge1 && ee1) eeBadge1.textContent = `EE: [${ee1.map(p => p.toFixed(2)).join(", ")}]`;
    if (eeBadge2 && ee2) eeBadge2.textContent = `EE: [${ee2.map(p => p.toFixed(2)).join(", ")}]`;
}

// ==========================================================================
// TELEOPERATION & MANUAL JOG CONTROLLER
// ==========================================================================
function initTeleopControls() {
    // Arm toggle (Arm 1 vs Arm 2)
    const arm1Btn = document.getElementById("toggleArm1");
    const arm2Btn = document.getElementById("toggleArm2");

    if (arm1Btn && arm2Btn) {
        arm1Btn.addEventListener("click", () => {
            activeArm = 1;
            arm1Btn.classList.add("active");
            arm2Btn.classList.remove("active");
            logConsole("Teleoperation set to Arm 1 (Giver)", "log-info");
        });

        arm2Btn.addEventListener("click", () => {
            activeArm = 2;
            arm2Btn.classList.add("active");
            arm1Btn.classList.remove("active");
            logConsole("Teleoperation set to Arm 2 (Receiver)", "log-info");
        });
    }

    // Step size buttons
    document.querySelectorAll(".step-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".step-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            jogStepMeters = parseFloat(btn.dataset.step);
            logConsole(`Jog step set to ${(jogStepMeters * 100).toFixed(0)} cm`, "log-info");
        });
    });
}

async function jogCartesian(axis, direction) {
    const delta = [0.0, 0.0, 0.0];
    const idx = axis === "x" ? 0 : (axis === "y" ? 1 : 2);
    delta[idx] = direction * jogStepMeters;

    try {
        const res = await fetch("/api/control/jog_cartesian", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                arm: activeArm,
                delta_xyz: delta,
            }),
        });
        const data = await res.json();
        if (data.status === "ok") {
            logConsole(`[JOG] Arm ${activeArm} ${axis.toUpperCase()}${direction > 0 ? "+" : "-"}: EE -> [${data.actual_ee.join(", ")}]`, "log-ok");
        } else {
            logConsole(`[JOG ERROR] ${data.message}`, "log-warn");
        }
    } catch (err) {
        logConsole(`Network jog error: ${err.message}`, "log-err");
    }
}

async function setGripperCommand(position) {
    try {
        const res = await fetch("/api/control/gripper", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                arm: activeArm,
                position: position,
            }),
        });
        const data = await res.json();
        if (data.status === "ok") {
            logConsole(`[GRIPPER] ${data.message}`, "log-ok");
        } else {
            logConsole(`[GRIPPER ERROR] ${data.message}`, "log-warn");
        }
    } catch (err) {
        logConsole(`Network gripper error: ${err.message}`, "log-err");
    }
}

// ==========================================================================
// FSM COMMAND TRIGGERS (Start, Stop, Randomize, Reset)
// ==========================================================================
async function triggerAction(action) {
    try {
        if (action === "start") {
            logConsole("[POLICY] Launching ACT Neural Policy (best_model_v3.pt) with Custody Shield Interlocks...", "log-info");
        }
        const res = await fetch(`/api/fsm/${action}`, { method: "POST" });
        const data = await res.json();
        if (data.status === "ok") {
            logConsole(`Command [${action.toUpperCase()}] acknowledged: ${data.message}`, "log-ok");
        } else {
            logConsole(`Command [${action.toUpperCase()}] rejected: ${data.message}`, "log-warn");
        }
    } catch (err) {
        logConsole(`Network command error [${action}]: ${err.message}`, "log-err");
    }
}

// ==========================================================================
// DUAL VIDEO RECORDING ENGINE (Client MediaRecorder + Server MP4)
// ==========================================================================
async function toggleRecording() {
    if (!isRecording) {
        await startRecording();
    } else {
        await stopRecording();
    }
}

async function startRecording() {
    const feedImg = document.getElementById("cameraFeed");
    if (!feedImg) return;

    if (!recordCanvas) {
        recordCanvas = document.createElement("canvas");
        recordCanvas.width = 960;
        recordCanvas.height = 720;
        recordCtx = recordCanvas.getContext("2d");
    }

    recordedChunks = [];

    // Client-side canvas capture
    try {
        recordDrawInterval = setInterval(() => {
            try {
                if (feedImg.complete && feedImg.naturalWidth > 0) {
                    recordCtx.drawImage(feedImg, 0, 0, recordCanvas.width, recordCanvas.height);
                }
            } catch (e) {}
        }, 40);

        const stream = recordCanvas.captureStream(25);
        let mimeType = "video/webm;codecs=vp9";
        if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = "video/webm;codecs=vp8";
        if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = "video/webm";

        mediaRecorder = new MediaRecorder(stream, { mimeType: mimeType, videoBitsPerSecond: 3000000 });
        mediaRecorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) recordedChunks.push(e.data);
        };
        mediaRecorder.onstop = saveBrowserVideo;
        mediaRecorder.start(200);
    } catch (err) {
        console.warn("[Recorder] Client MediaRecorder error:", err);
    }

    // Server-side MP4 recorder trigger
    try {
        fetch(`/api/stream/record/start?camera_name=${currentCamera}`, { method: "POST" });
    } catch (e) {
        console.warn("[Recorder] Server record trigger error:", e);
    }

    isRecording = true;
    recordStartTime = Date.now();

    // UI Updates
    const recBtn = document.getElementById("recordBtn");
    if (recBtn) recBtn.classList.add("recording-active");
    const recBtnLabel = document.getElementById("recordBtnLabel");
    if (recBtnLabel) recBtnLabel.textContent = "Stop (00:00)";
    const overlayBadge = document.getElementById("recOverlayBadge");
    if (overlayBadge) overlayBadge.style.display = "inline-flex";

    recordTimerInterval = setInterval(() => {
        const elapsedSec = Math.floor((Date.now() - recordStartTime) / 1000);
        const mm = String(Math.floor(elapsedSec / 60)).padStart(2, "0");
        const ss = String(elapsedSec % 60).padStart(2, "0");
        const timeStr = `${mm}:${ss}`;
        const timerText = document.getElementById("recTimerText");
        if (timerText) timerText.textContent = timeStr;
        if (recBtnLabel) recBtnLabel.textContent = `Stop (${timeStr})`;
    }, 500);

    logConsole(`[RECORDER] Started recording [${currentCamera}]...`, "log-ok");
}

async function stopRecording() {
    if (!isRecording) return;
    isRecording = false;

    if (recordTimerInterval) {
        clearInterval(recordTimerInterval);
        recordTimerInterval = null;
    }
    if (recordDrawInterval) {
        clearInterval(recordDrawInterval);
        recordDrawInterval = null;
    }

    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }

    try {
        const res = await fetch("/api/stream/record/stop", { method: "POST" });
        const data = await res.json();
        if (data.status === "ok") {
            logConsole(`[RECORDER] Server MP4 finalized: ${data.duration_sec}s (${data.size_mb} MB) -> ${data.filepath}`, "log-ok");
        }
    } catch (e) {
        console.warn("[Recorder] Server stop error:", e);
    }

    const recBtn = document.getElementById("recordBtn");
    if (recBtn) recBtn.classList.remove("recording-active");
    const recBtnLabel = document.getElementById("recordBtnLabel");
    if (recBtnLabel) recBtnLabel.textContent = "Record";
    const overlayBadge = document.getElementById("recOverlayBadge");
    if (overlayBadge) overlayBadge.style.display = "none";
}

function saveBrowserVideo() {
    if (recordedChunks.length === 0) return;
    const blob = new Blob(recordedChunks, { type: "video/webm" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    a.href = url;
    a.download = `ur10e_handover_${currentCamera}_${ts}.webm`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 100);
    const sizeMb = (blob.size / (1024 * 1024)).toFixed(2);
    logConsole(`[RECORDER] Video clip saved to browser downloads (${sizeMb} MB)`, "log-ok");
}

// ==========================================================================
// HIGH-FREQUENCY WEBSOCKET TELEMETRY LINK
// ==========================================================================
function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
    wsConnection = new WebSocket(wsUrl);

    wsConnection.onopen = () => {
        const wsStatus = document.getElementById("wsStatus");
        const wsDot = document.getElementById("wsDot");
        if (wsStatus) wsStatus.textContent = "ONLINE (20 Hz)";
        if (wsDot) {
            wsDot.style.background = "var(--accent-green)";
            wsDot.style.boxShadow = "0 0 8px var(--accent-green)";
        }
        logConsole("WebSocket telemetry link verified (20 Hz broadcast).", "log-ok");
    };

    wsConnection.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleTelemetryPacket(data);
        } catch (e) {
            console.error("Telemetry parse error:", e);
        }
    };

    wsConnection.onclose = () => {
        const wsStatus = document.getElementById("wsStatus");
        const wsDot = document.getElementById("wsDot");
        if (wsStatus) wsStatus.textContent = "DISCONNECTED";
        if (wsDot) {
            wsDot.style.background = "var(--accent-red)";
            wsDot.style.boxShadow = "0 0 8px var(--accent-red)";
        }
        setTimeout(connectWebSocket, 1500);
    };

    wsConnection.onerror = () => {
        wsConnection.close();
    };
}

function handleTelemetryPacket(pkt) {
    // 1. Simulation clock & cycle
    const simTimeEl = document.getElementById("simTimeVal");
    const cycleEl = document.getElementById("cycleCountVal");
    if (simTimeEl && pkt.sim_time !== undefined) simTimeEl.textContent = `${pkt.sim_time.toFixed(1)}s`;
    if (cycleEl && pkt.cycle_count !== undefined) cycleEl.textContent = `#${pkt.cycle_count}`;

    // 2. FSM Flowchart update
    if (pkt.fsm_state) {
        updateFsmFlowchart(pkt.fsm_state);
    }

    // 3. Radial Force Gauges & Dual Custody
    const f1 = pkt.arm1 ? pkt.arm1.contact_force : 0.0;
    const f2 = pkt.arm2 ? pkt.arm2.contact_force : 0.0;
    updateForceGauges(f1, f2, pkt.dual_hold_confirmed ?? pkt.dual_hold);

    // 4. 14-DOF Joint Bars & EE Positions
    const q1 = pkt.arm1 ? pkt.arm1.joint_positions : [];
    const q2 = pkt.arm2 ? pkt.arm2.joint_positions : [];
    const ee1 = pkt.arm1 ? pkt.arm1.ee_position : null;
    const ee2 = pkt.arm2 ? pkt.arm2.ee_position : null;
    updateJointsUI(q1, q2, ee1, ee2);

    // 5. 3D Spatial Tracking (Cube & Bin)
    const objPosEl = document.getElementById("targetObjPos");
    const binPosEl = document.getElementById("targetBinPos");
    if (objPosEl && pkt.target_object_pos) {
        objPosEl.textContent = `[${pkt.target_object_pos.map(p => p.toFixed(2)).join(", ")}]`;
    }
    const binCoords = pkt.target_bin_pos || pkt.bin_pos;
    if (binPosEl && binCoords) {
        binPosEl.textContent = `[${binCoords.map(p => p.toFixed(2)).join(", ")}]`;
    }
}

// ==========================================================================
// SYSTEM EVENT TERMINAL LOGGING
// ==========================================================================
function logConsole(message, type = "log-info") {
    const consoleEl = document.getElementById("systemConsole");
    if (!consoleEl) return;

    const timeStr = new Date().toLocaleTimeString();
    const entry = document.createElement("div");
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `<span class="log-time">[${timeStr}]</span> ${message}`;
    consoleEl.appendChild(entry);
    consoleEl.scrollTop = consoleEl.scrollHeight;

    // Limit log length to last 60 entries
    while (consoleEl.children.length > 60) {
        consoleEl.removeChild(consoleEl.firstChild);
    }
}
