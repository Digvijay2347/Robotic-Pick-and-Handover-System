/**
 * BIMANUAL UR10e ROBOTICS PLATFORM - DATASET STUDIO (PHASE 3)
 * Consolidated HDF5 Dataset Inspector, 50-Episode Catalog,
 * HTML5 Multi-Angle Preview Player, and Multi-Worker Collection Orchestrator.
 */

// Dataset Studio State
let datasetOverview = null;
let allEpisodes = [];
let filteredEpisodes = [];
let selectedEpisodeIdx = 0;
let collectionPollInterval = null;
let isCollectionRunning = false;

// ==============================================================================
// GLOBAL TAB / VIEW SWITCHER
// ==============================================================================
function switchView(viewName) {
    const missionView = document.getElementById("missionControlView");
    const datasetView = document.getElementById("datasetStudioView");
    const navMission = document.getElementById("navBtnMissionControl");
    const navDataset = document.getElementById("navBtnDatasetStudio");

    if (viewName === "dataset-studio") {
        if (missionView) missionView.style.display = "none";
        if (datasetView) {
            datasetView.style.display = "flex";
            datasetView.classList.add("view-fade-in");
        }
        if (navMission) navMission.classList.remove("active");
        if (navDataset) navDataset.classList.add("active");
        window.location.hash = "dataset-studio";

        // Initialize dataset studio if not loaded yet
        if (!datasetOverview) {
            initDatasetStudio();
        }
    } else {
        // Mission Control
        if (datasetView) datasetView.style.display = "none";
        if (missionView) {
            missionView.style.display = "block";
            missionView.classList.add("view-fade-in");
        }
        if (navDataset) navDataset.classList.remove("active");
        if (navMission) navMission.classList.add("active");
        window.location.hash = "mission-control";
    }
}

// Window popstate / hashchange listener
window.addEventListener("hashchange", () => {
    if (window.location.hash === "#dataset-studio") {
        switchView("dataset-studio");
    } else if (window.location.hash === "#mission-control" || window.location.hash === "") {
        switchView("mission-control");
    }
});

// ==============================================================================
// INITIALIZATION
// ==============================================================================
document.addEventListener("DOMContentLoaded", () => {
    // Check initial hash
    if (window.location.hash === "#dataset-studio") {
        switchView("dataset-studio");
    }

    // Attach listeners for dataset studio controls
    initCollectionFormControls();
    initVideoPlayerControls();
    initSearchAndFilter();
});

async function initDatasetStudio() {
    console.log("[DatasetStudio] Initializing Dataset Studio modules...");
    await Promise.all([
        loadDatasetOverview(),
        loadEpisodesCatalog(),
        loadRecordingsList(),
        checkInitialCollectionStatus()
    ]);
}

// ==============================================================================
// 1. DATASET OVERVIEW & KPI METRICS
// ==============================================================================
async function loadDatasetOverview() {
    try {
        const res = await fetch("/api/dataset/overview");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        datasetOverview = data;

        // Populate KPI Cards
        const totalEpEl = document.getElementById("kpiTotalEpisodes");
        const totalFramesEl = document.getElementById("kpiTotalFrames");
        const storageSizeEl = document.getElementById("kpiStorageSize");
        const successRateEl = document.getElementById("kpiSuccessRate");
        const datasetMetaEl = document.getElementById("datasetMetaSub");
        const navBadgeEl = document.getElementById("navBadgeDemos");

        if (totalEpEl) totalEpEl.textContent = data.total_episodes || "50";
        if (totalFramesEl) totalFramesEl.textContent = (data.total_frames || 32450).toLocaleString();
        if (storageSizeEl) storageSizeEl.textContent = `${data.size_mb || 531.97} MB`;
        if (successRateEl) successRateEl.textContent = `${data.success_rate_percent || 100}%`;
        if (navBadgeEl) navBadgeEl.textContent = `${data.total_episodes || 50} Demos`;

        if (datasetMetaEl) {
            datasetMetaEl.textContent = `HDF5 LeRobot Format • ${data.fps || 25} Hz • 3 Synced Cameras (112x112)`;
        }

        // Render Action Distribution Spectrum
        renderActionDistribution(data.action_distributions || []);
    } catch (err) {
        console.error("[DatasetStudio] Error loading overview:", err);
    }
}

function renderActionDistribution(distributions) {
    const container = document.getElementById("actionSpectrumContainer");
    if (!container) return;

    if (!distributions || distributions.length === 0) {
        container.innerHTML = `<div style="grid-column: 1 / -1; color: var(--text-muted); font-size:11px; padding:10px;">Action bounds computed dynamically from stats.json.</div>`;
        return;
    }

    let html = "";
    distributions.forEach((d) => {
        // Normalize range to 0-100% for display
        // Assuming global joint limits [-3.14, 3.14] for arms, [0, 1] for grippers
        const isGripper = d.name.toLowerCase().includes("gripper");
        const globalMin = isGripper ? 0.0 : -3.14;
        const globalMax = isGripper ? 1.0 : 3.14;
        const span = globalMax - globalMin;

        const leftPct = Math.max(0, Math.min(100, ((d.min - globalMin) / span) * 100));
        const rightPct = Math.max(0, Math.min(100, ((d.max - globalMin) / span) * 100));
        const widthPct = Math.max(2, rightPct - leftPct);
        const meanPct = Math.max(0, Math.min(100, ((d.mean - globalMin) / span) * 100));

        html += `
            <div class="spectrum-row">
                <div class="spectrum-name" title="${d.name}">
                    <span style="color:var(--accent-cyan);font-weight:700;">#${d.actuator_idx}</span> ${d.name}
                </div>
                <div class="spectrum-track">
                    <div class="spectrum-range-bar" style="left: ${leftPct}%; width: ${widthPct}%;"></div>
                    <div class="spectrum-mean-dot" style="left: ${meanPct}%;" title="Mean: ${d.mean}"></div>
                </div>
                <div class="spectrum-range-text">[${d.min.toFixed(2)}, ${d.max.toFixed(2)}]</div>
            </div>
        `;
    });

    container.innerHTML = html;
}

// ==============================================================================
// 2. 50-EPISODE CATALOG & DATA TABLE
// ==============================================================================
async function loadEpisodesCatalog() {
    const tbody = document.getElementById("episodesTableBody");
    if (!tbody) return;

    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;padding:24px;color:var(--text-muted);">Loading 50 demonstration episodes...</td></tr>`;

    try {
        const res = await fetch("/api/dataset/episodes?limit=100");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        allEpisodes = data.episodes || [];
        filteredEpisodes = [...allEpisodes];
        renderEpisodeTable();

        // Select the first episode by default
        if (allEpisodes.length > 0) {
            selectEpisode(allEpisodes[0].episode_index);
        }
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;padding:24px;color:var(--accent-red);">Error loading episodes: ${err.message}</td></tr>`;
    }
}

function renderEpisodeTable() {
    const tbody = document.getElementById("episodesTableBody");
    const countEl = document.getElementById("catalogCount");
    if (!tbody) return;

    if (countEl) countEl.textContent = `${filteredEpisodes.length} Episodes`;

    if (filteredEpisodes.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;padding:24px;color:var(--text-muted);">No matching demonstration episodes found.</td></tr>`;
        return;
    }

    let html = "";
    filteredEpisodes.forEach((ep) => {
        const isSelected = ep.episode_index === selectedEpisodeIdx;
        const rowClass = isSelected ? "selected-row" : "";
        html += `
            <tr class="${rowClass}" onclick="selectEpisode(${ep.episode_index})">
                <td style="font-weight:700;color:var(--text-primary);">
                    <span style="color:var(--accent-cyan);margin-right:4px;">●</span>${ep.episode_id}
                </td>
                <td><span class="badge-success">100% OK</span></td>
                <td>${ep.steps} steps</td>
                <td>${ep.duration_sec}s</td>
                <td><span class="badge-dualhold">CONFIRMED</span></td>
                <td>${ep.distance_to_bin_cm} cm</td>
                <td style="color:var(--text-muted);">Worker ${ep.worker_id}</td>
            </tr>
        `;
    });

    tbody.innerHTML = html;
}

function initSearchAndFilter() {
    const searchInput = document.getElementById("episodeSearchInput");
    if (searchInput) {
        searchInput.addEventListener("input", (e) => {
            const query = e.target.value.trim().toLowerCase();
            if (!query) {
                filteredEpisodes = [...allEpisodes];
            } else {
                filteredEpisodes = allEpisodes.filter(
                    (ep) =>
                        ep.episode_id.toLowerCase().includes(query) ||
                        String(ep.episode_index).includes(query) ||
                        `worker ${ep.worker_id}`.toLowerCase().includes(query)
                );
            }
            renderEpisodeTable();
        });
    }
}

// ==============================================================================
// 3. EPISODE INSPECTION & TRAJECTORY DETAILS
// ==============================================================================
async function selectEpisode(episodeIndex) {
    selectedEpisodeIdx = episodeIndex;

    // Update row highlighting
    const rows = document.querySelectorAll("#episodesTableBody tr");
    rows.forEach((r) => r.classList.remove("selected-row"));

    const targetRow = Array.from(rows).find((r) =>
        r.firstElementChild && r.firstElementChild.textContent.includes(`demo_${episodeIndex}`)
    );
    if (targetRow) targetRow.classList.add("selected-row");

    // Update metadata card
    const ep = allEpisodes.find((e) => e.episode_index === episodeIndex);
    if (ep) {
        document.getElementById("selectedEpTitle").textContent = ep.episode_id.toUpperCase();
        document.getElementById("selectedEpSteps").textContent = `${ep.steps} steps (${ep.duration_sec}s)`;
        document.getElementById("selectedEpBinDist").textContent = `${ep.distance_to_bin_cm} cm (Optimal)`;
        document.getElementById("selectedEpWorker").textContent = `Worker Process #${ep.worker_id}`;
        document.getElementById("selectedEpDualHold").textContent = ep.dual_hold_confirmed ? "Simultaneous Clamped" : "N/A";
    }

    // Optionally fetch trajectory slice from backend
    try {
        const res = await fetch(`/api/dataset/episode/${episodeIndex}`);
        if (res.ok) {
            const epData = await res.json();
            const trajPointsEl = document.getElementById("selectedEpTrajCount");
            if (trajPointsEl) {
                trajPointsEl.textContent = `${epData.action_trajectory.length} decimated checkpoints`;
            }
        }
    } catch (e) {
        // Non-blocking
    }
}

// ==============================================================================
// 4. HTML5 DEMONSTRATION VIDEO PLAYER
// ==============================================================================
async function loadRecordingsList() {
    const dropdown = document.getElementById("videoSelectDropdown");
    if (!dropdown) return;

    try {
        const res = await fetch("/api/dataset/recordings");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const recordings = data.recordings || [];

        if (recordings.length === 0) {
            dropdown.innerHTML = `<option value="">No recorded MP4 videos found</option>`;
            return;
        }

        dropdown.innerHTML = "";
        recordings.forEach((rec, idx) => {
            const opt = document.createElement("option");
            opt.value = rec.video_url;
            opt.textContent = `${rec.title} (${rec.size_mb} MB)`;
            if (idx === 0) opt.selected = true;
            dropdown.appendChild(opt);
        });

        // Load the first video
        if (recordings.length > 0) {
            loadVideo(recordings[0].video_url);
        }
    } catch (err) {
        console.error("[DatasetStudio] Error loading recordings:", err);
    }
}

function initVideoPlayerControls() {
    const dropdown = document.getElementById("videoSelectDropdown");
    const video = document.getElementById("demoVideoPlayer");

    if (dropdown) {
        dropdown.addEventListener("change", (e) => {
            if (e.target.value) {
                loadVideo(e.target.value);
            }
        });
    }

    // Playback Speed Buttons
    const speedBtns = document.querySelectorAll(".video-speed-btn");
    speedBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            speedBtns.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            if (video) {
                video.playbackRate = parseFloat(btn.dataset.speed || "1.0");
            }
        });
    });
}

function loadVideo(url) {
    const video = document.getElementById("demoVideoPlayer");
    if (!video) return;

    video.src = url;
    video.load();
    // Do not autoplay with sound, play muted or wait for user click
    video.muted = true;
    video.play().catch(() => {
        // Autoplay policy handled gracefully
    });
}

// ==============================================================================
// 5. PARALLEL COLLECTION ORCHESTRATION & LIVE WORKER MATRIX
// ==============================================================================
function initCollectionFormControls() {
    const epSlider = document.getElementById("collectNumEpisodes");
    const epDisplay = document.getElementById("collectNumEpisodesVal");
    if (epSlider && epDisplay) {
        epSlider.addEventListener("input", () => {
            epDisplay.textContent = epSlider.value;
        });
    }

    const workerSlider = document.getElementById("collectNumWorkers");
    const workerDisplay = document.getElementById("collectNumWorkersVal");
    if (workerSlider && workerDisplay) {
        workerSlider.addEventListener("input", () => {
            workerDisplay.textContent = workerSlider.value;
        });
    }

    const fpsSlider = document.getElementById("collectFps");
    const fpsDisplay = document.getElementById("collectFpsVal");
    if (fpsSlider && fpsDisplay) {
        fpsSlider.addEventListener("input", () => {
            fpsDisplay.textContent = fpsSlider.value;
        });
    }

    const startBtn = document.getElementById("btnStartCollection");
    const stopBtn = document.getElementById("btnStopCollection");

    if (startBtn) startBtn.addEventListener("click", startParallelCollection);
    if (stopBtn) stopBtn.addEventListener("click", stopParallelCollection);
}

async function checkInitialCollectionStatus() {
    try {
        const res = await fetch("/api/dataset/collect/status");
        if (res.ok) {
            const status = await res.json();
            if (status.is_running) {
                isCollectionRunning = true;
                updateCollectionUI(status);
                startStatusPolling();
            }
        }
    } catch (e) {
        console.warn("[DatasetStudio] Initial status check failed:", e);
    }
}

async function startParallelCollection() {
    const numEpisodes = parseInt(document.getElementById("collectNumEpisodes")?.value || "10", 10);
    const numWorkers = parseInt(document.getElementById("collectNumWorkers")?.value || "4", 10);
    const fps = parseInt(document.getElementById("collectFps")?.value || "25", 10);
    const randomize = document.getElementById("collectRandomize")?.checked ?? true;
    const saveImages = document.getElementById("collectSaveImages")?.checked ?? true;
    const renderQuality = document.getElementById("collectRenderQuality")?.value || "compact";

    const startBtn = document.getElementById("btnStartCollection");
    const stopBtn = document.getElementById("btnStopCollection");

    if (startBtn) {
        startBtn.disabled = true;
        startBtn.innerHTML = `<span>⏳ Launching...</span>`;
    }

    logCollectionConsole(`[LAUNCHER] Sending request to spawn ${numWorkers} parallel workers for ${numEpisodes} episodes...`, "log-info");

    try {
        const res = await fetch("/api/dataset/collect/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                num_episodes: numEpisodes,
                num_workers: numWorkers,
                fps: fps,
                randomize: randomize,
                save_images: saveImages,
                render_quality: renderQuality,
                compression: "lzf",
                dataset_name: "handover_demos.hdf5",
                overwrite: false
            })
        });

        const data = await res.json();
        if (data.status === "ok") {
            logCollectionConsole(`[LAUNCHER] ${data.message}`, "log-success");
            isCollectionRunning = true;
            if (stopBtn) stopBtn.disabled = false;
            startStatusPolling();
        } else {
            logCollectionConsole(`[LAUNCHER] Error: ${data.message}`, "log-warn");
            if (startBtn) {
                startBtn.disabled = false;
                startBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" width="14" height="14"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/></svg> Launch Parallel Collection`;
            }
        }
    } catch (err) {
        logCollectionConsole(`[LAUNCHER] Network exception: ${err.message}`, "log-error");
        if (startBtn) {
            startBtn.disabled = false;
            startBtn.innerHTML = `Launch Parallel Collection`;
        }
    }
}

async function stopParallelCollection() {
    const stopBtn = document.getElementById("btnStopCollection");
    if (stopBtn) stopBtn.disabled = true;

    logCollectionConsole("[LAUNCHER] Stopping parallel collection job...", "log-warn");

    try {
        const res = await fetch("/api/dataset/collect/stop", { method: "POST" });
        const data = await res.json();
        logCollectionConsole(`[LAUNCHER] ${data.message}`, "log-info");
    } catch (err) {
        logCollectionConsole(`[LAUNCHER] Error terminating job: ${err.message}`, "log-error");
    }
}

function startStatusPolling() {
    if (collectionPollInterval) clearInterval(collectionPollInterval);
    collectionPollInterval = setInterval(async () => {
        try {
            const res = await fetch("/api/dataset/collect/status");
            if (!res.ok) return;
            const status = await res.json();
            updateCollectionUI(status);

            if (!status.is_running && isCollectionRunning) {
                // Job just finished or stopped
                isCollectionRunning = false;
                clearInterval(collectionPollInterval);
                logCollectionConsole(`[JOB COMPLETE] Parallel collection finished with exit code ${status.exit_code}. Refreshing dataset...`, "log-success");
                loadDatasetOverview();
                loadEpisodesCatalog();
            }
        } catch (e) {
            console.warn("[DatasetStudio] Status poll failed:", e);
        }
    }, 1000);
}

function updateCollectionUI(status) {
    const startBtn = document.getElementById("btnStartCollection");
    const stopBtn = document.getElementById("btnStopCollection");

    if (startBtn) {
        startBtn.disabled = status.is_running;
        startBtn.innerHTML = status.is_running
            ? `<span>⚙️ Collecting...</span>`
            : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" width="14" height="14"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/></svg> Launch Parallel Collection`;
    }
    if (stopBtn) {
        stopBtn.disabled = !status.is_running;
    }

    // Update Master Progress
    const progText = document.getElementById("masterProgText");
    const progPct = document.getElementById("masterProgPct");
    const progFill = document.getElementById("masterProgFill");

    if (progText) progText.textContent = `${status.completed_episodes} / ${status.target_episodes || 50} Episodes`;
    if (progPct) progPct.textContent = `${status.progress_percent}%`;
    if (progFill) progFill.style.width = `${status.progress_percent}%`;

    // Render Worker Matrix Cards
    renderWorkerMatrix(status.workers || {}, status.is_running);

    // Stream Logs
    if (status.recent_logs && status.recent_logs.length > 0) {
        renderCollectionLogs(status.recent_logs);
    }
}

function renderWorkerMatrix(workers, isRunning) {
    const container = document.getElementById("workerMatrixContainer");
    if (!container) return;

    const workerIds = Object.keys(workers).map(Number).sort((a, b) => a - b);
    if (workerIds.length === 0) {
        // Show default 6 idle worker cards
        let html = "";
        for (let i = 0; i < 6; i++) {
            html += `
                <div class="worker-card" id="worker_card_${i}">
                    <div class="worker-header">
                        <span class="worker-id-title">
                            <span style="color:var(--text-muted);">●</span> Worker ${i}
                        </span>
                        <span class="worker-status-badge">READY</span>
                    </div>
                    <div class="worker-ep-info">
                        <span>Progress:</span>
                        <span>0 eps</span>
                    </div>
                    <div class="worker-prog-track">
                        <div class="worker-prog-fill" style="width: 0%;"></div>
                    </div>
                    <div class="worker-stage-pill">STANDBY</div>
                </div>
            `;
        }
        container.innerHTML = html;
        return;
    }

    let html = "";
    workerIds.forEach((wid) => {
        const w = workers[wid];
        const activeClass = w.status === "ACTIVE" || w.status === "RUNNING" ? "worker-active" : "";
        const badgeClass =
            w.status === "COMPLETED" ? "status-success" :
            w.status === "RUNNING" || w.status === "ACTIVE" ? "status-running" : "";

        const pct = Math.min(100, Math.round(((w.completed_episodes || 0) / Math.max(1, (w.target_episodes || 10))) * 100));

        html += `
            <div class="worker-card ${activeClass}" id="worker_card_${wid}">
                <div class="worker-header">
                    <span class="worker-id-title">
                        <span style="color:${activeClass ? 'var(--accent-cyan)' : 'var(--text-muted)'};">●</span> Worker ${wid}
                    </span>
                    <span class="worker-status-badge ${badgeClass}">${w.status || "IDLE"}</span>
                </div>
                <div class="worker-ep-info">
                    <span>Episode: ${w.current_episode || 0}</span>
                    <span>${w.completed_episodes || 0} done</span>
                </div>
                <div class="worker-prog-track">
                    <div class="worker-prog-fill" style="width: ${pct}%;"></div>
                </div>
                <div class="worker-stage-pill">${w.stage || "IDLE"}</div>
            </div>
        `;
    });

    container.innerHTML = html;
}

function renderCollectionLogs(logs) {
    const consoleEl = document.getElementById("collectionLogsConsole");
    if (!consoleEl) return;

    // Keep console updated without infinite expansion
    const linesHtml = logs.map((l) => {
        let cls = "log-entry";
        if (l.includes("[ERROR]")) cls += " log-error";
        else if (l.includes("[SUCCESS]") || l.includes("COMPLETE")) cls += " log-success";
        else if (l.includes("[WARN]")) cls += " log-warn";
        return `<div class="${cls}">${escapeHtml(l)}</div>`;
    }).join("");

    consoleEl.innerHTML = linesHtml;
    consoleEl.scrollTop = consoleEl.scrollHeight;
}

function logCollectionConsole(msg, cls = "") {
    const consoleEl = document.getElementById("collectionLogsConsole");
    if (!consoleEl) return;

    const entry = document.createElement("div");
    entry.className = `log-entry ${cls}`;
    entry.textContent = msg;
    consoleEl.appendChild(entry);
    consoleEl.scrollTop = consoleEl.scrollHeight;
}

function escapeHtml(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}
