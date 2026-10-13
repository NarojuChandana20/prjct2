/**
 * Student Safety Monitoring System - Frontend Dashboard Script
 * -------------------------------------------------------------
 * Connects UI to FastAPI backend video behaviour & emotion pipeline.
 * Real-time state management, video upload, sample video selection,
 * dual player synchronization, compact telemetry cards with accordion details,
 * live frame scrubber tracking, session scoping, and incident audit review.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements - Video Selection & Analysis
  const dropzone = document.getElementById("videoDropzone");
  const fileInput = document.getElementById("videoFileInput");
  const selectedFileInfo = document.getElementById("selectedFileInfo");
  const selectedFileName = document.getElementById("selectedFileName");
  const selectedFileSpecs = document.getElementById("selectedFileSpecs");
  const btnRemoveFile = document.getElementById("btnRemoveFile");
  const sampleButtonsContainer = document.getElementById("sampleButtonsList");
  const btnAnalyze = document.getElementById("btnAnalyze");
  const selectMaxFrames = document.getElementById("selectMaxFrames");

  const processingPanel = document.getElementById("processingPanel");
  const progressStatusText = document.getElementById("progressStatusText");
  const errorBanner = document.getElementById("errorBanner");
  const errorMessage = document.getElementById("errorMessage");

  const videoPlayerInput = document.getElementById("videoPlayerInput");
  const videoPlayerAnnotated = document.getElementById("videoPlayerAnnotated");
  const inputPlaceholder = document.getElementById("inputPlaceholder");
  const annotatedPlaceholder = document.getElementById("annotatedPlaceholder");

  // Summary Stat Elements
  const statPersistentTracks = document.getElementById("statPersistentTracks");
  const statInFrameCountPill = document.getElementById("statInFrameCountPill");
  const statBehaviourPreds = document.getElementById("statBehaviourPreds");
  const statEmotionPreds = document.getElementById("statEmotionPreds");
  const statProcessingSpeed = document.getElementById("statProcessingSpeed");
  const statIncidentsCount = document.getElementById("statIncidentsCount");
  const statPendingCountPill = document.getElementById("statPendingCountPill");

  // Telemetry Grid & Scrubber
  const liveScrubberFrame = document.getElementById("liveScrubberFrame");
  const liveFramePeopleCount = document.getElementById("liveFramePeopleCount");
  const metricActiveTracks = document.getElementById("metricActiveTracks");
  const totalTrackSessionsCount = document.getElementById("totalTrackSessionsCount");
  const filterTracksAll = document.getElementById("filterTracksAll");
  const filterTracksRisk = document.getElementById("filterTracksRisk");
  const trackedPersonsGrid = document.getElementById("trackedPersonsGrid");
  const emptyResultsPlaceholder = document.getElementById("emptyResultsPlaceholder");
  const statsBreakdownGrid = document.getElementById("statsBreakdownGrid");
  const behaviourDistList = document.getElementById("behaviourDistList");
  const emotionDistList = document.getElementById("emotionDistList");

  // Incident Management Elements
  const btnScopeCurrentVideo = document.getElementById("btnScopeCurrentVideo");
  const btnScopeHistorical = document.getElementById("btnScopeHistorical");
  const countCurrentVideoIncidents = document.getElementById("countCurrentVideoIncidents");
  const countAllHistoricalIncidents = document.getElementById("countAllHistoricalIncidents");
  const incidentScopeNotice = document.getElementById("incidentScopeNotice");

  const incidentsTableBody = document.getElementById("incidentsTableBody");
  const incidentsEmptyPlaceholder = document.getElementById("incidentsEmptyPlaceholder");
  const filterIncidentStatus = document.getElementById("filterIncidentStatus");
  const filterIncidentBehaviour = document.getElementById("filterIncidentBehaviour");
  const inputIncidentSearch = document.getElementById("inputIncidentSearch");
  const btnRefreshIncidents = document.getElementById("btnRefreshIncidents");

  // Incident Review Modal Elements
  const incidentReviewModal = document.getElementById("incidentReviewModal");
  const modalCloseBtn = document.getElementById("modalCloseBtn");
  const btnModalCancel = document.getElementById("btnModalCancel");
  const incidentReviewForm = document.getElementById("incidentReviewForm");
  const modalIncidentTitle = document.getElementById("modalIncidentTitle");
  const modalIncidentSubtitle = document.getElementById("modalIncidentSubtitle");
  const modalEvidenceImg = document.getElementById("modalEvidenceImg");
  const modalEvidencePlaceholder = document.getElementById("modalEvidencePlaceholder");
  const modalMetaBehaviour = document.getElementById("modalMetaBehaviour");
  const modalMetaConfidence = document.getElementById("modalMetaConfidence");
  const modalMetaTrackId = document.getElementById("modalMetaTrackId");
  const modalMetaTimeSpan = document.getElementById("modalMetaTimeSpan");
  const modalMetaEmotion = document.getElementById("modalMetaEmotion");
  const modalMetaSource = document.getElementById("modalMetaSource");
  const inputReviewerName = document.getElementById("inputReviewerName");
  const textareaReviewerNotes = document.getElementById("textareaReviewerNotes");

  // Verified Person Linkage Modal Elements
  const linkPersonModal = document.getElementById("linkPersonModal");
  const linkModalCloseBtn = document.getElementById("linkModalCloseBtn");
  const btnLinkCancel = document.getElementById("btnLinkCancel");
  const linkPersonForm = document.getElementById("linkPersonForm");
  const linkTrackTokenDisplay = document.getElementById("linkTrackTokenDisplay");
  const selectVerifiedPerson = document.getElementById("selectVerifiedPerson");
  const inputLinkReviewer = document.getElementById("inputLinkReviewer");
  const textareaLinkNotes = document.getElementById("textareaLinkNotes");

  const evidenceZoomModal = document.getElementById("evidenceZoomModal");
  const zoomModalCloseBtn = document.getElementById("zoomModalCloseBtn");
  const zoomModalImg = document.getElementById("zoomModalImg");
  const toastContainer = document.getElementById("toastContainer");

  // State
  let currentVideoId = null;
  let currentSessionId = null;
  let currentFile = null;
  let isProcessing = false;
  let activeReviewIncidentId = null;
  let activeLinkTrackId = null;
  let allLoadedIncidents = [];
  let incidentScope = "current"; // "current" | "historical"
  let latestTelemetryTimeline = {};
  let currentFPS = 25.0;
  let latestTrackedPeople = [];
  let trackFilterMode = "all"; // "all" | "risk"

  // Initialize
  checkSystemHealth();
  fetchSampleVideos();
  fetchIncidents();
  fetchIncidentStats();

  // ==========================================================================
  // 1. System Health Check
  // ==========================================================================
  async function checkSystemHealth() {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        const healthBadge = document.getElementById("systemHealthBadge");
        if (healthBadge) {
          healthBadge.innerHTML = `<span class="pulse-dot"></span> 5 Models Online`;
        }
      }
    } catch (err) {
      console.warn("Backend health check warning:", err);
    }
  }

  // ==========================================================================
  // 2. Sample Videos Fetcher
  // ==========================================================================
  async function fetchSampleVideos() {
    try {
      const res = await fetch("/api/sample-videos");
      if (!res.ok) return;
      const data = await res.json();
      const samples = data.samples || [];

      if (samples.length === 0) {
        sampleButtonsContainer.innerHTML = `<div style="font-size:0.78rem;color:var(--text-muted);">No pre-loaded sample videos available.</div>`;
        return;
      }

      sampleButtonsContainer.innerHTML = "";
      samples.forEach(sample => {
        const btn = document.createElement("button");
        btn.className = "sample-btn";
        btn.id = `sample-${sample.filename.replace(/[^a-zA-Z0-9]/g, "_")}`;
        btn.innerHTML = `
          <span>${sample.filename}</span>
          <span class="sample-specs">${sample.frames} frames · ${sample.fps} FPS · ${sample.duration_sec}s</span>
        `;
        btn.addEventListener("click", () => selectSampleVideo(sample.filename, btn));
        sampleButtonsContainer.appendChild(btn);
      });
    } catch (err) {
      console.warn("Could not fetch sample videos:", err);
    }
  }

  async function selectSampleVideo(filename, buttonEl) {
    if (isProcessing) return;
    clearError();

    document.querySelectorAll(".sample-btn").forEach(b => b.classList.remove("active"));
    buttonEl.classList.add("active");

    showProcessing(true, `Selecting sample video: ${filename}...`);

    try {
      const res = await fetch(`/api/use-sample/${filename}`, { method: "POST" });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to select sample video.");
      }

      const meta = await res.json();
      currentVideoId = meta.video_id;
      currentSessionId = null;
      currentFile = null;

      displayFileSelection({
        name: meta.filename,
        size: meta.file_size,
        frames: meta.frames,
        fps: meta.fps,
        duration: meta.duration_sec,
        previewUrl: meta.preview_url,
      });

      setVideoSource(videoPlayerInput, meta.preview_url, inputPlaceholder);
      resetAnnotatedPlayer();
      btnAnalyze.disabled = false;

      // Reset telemetry and refresh scoped incidents
      resetTelemetryView();
      fetchIncidents();
      fetchIncidentStats();
    } catch (err) {
      showError(err.message);
    } finally {
      showProcessing(false);
    }
  }

  // ==========================================================================
  // 3. File Upload & Drag-and-Drop
  // ==========================================================================
  dropzone.addEventListener("click", () => {
    if (!isProcessing) fileInput.click();
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    if (!isProcessing) dropzone.classList.add("drag-over");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-over");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    if (isProcessing) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  btnRemoveFile.addEventListener("click", resetUploadState);

  async function handleFileSelected(file) {
    clearError();
    document.querySelectorAll(".sample-btn").forEach(b => b.classList.remove("active"));

    const validExtensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"];
    const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();

    if (!validExtensions.includes(ext)) {
      showError(`Unsupported file format '${ext}'. Please choose an MP4, AVI, MOV, MKV, or WEBM video.`);
      return;
    }

    if (file.size === 0) {
      showError("The selected video file is empty (0 bytes).");
      return;
    }

    currentFile = file;
    showProcessing(true, `Uploading & validating ${file.name}...`);
    btnAnalyze.disabled = true;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload validation failed.");
      }

      const meta = await res.json();
      currentVideoId = meta.video_id;
      currentSessionId = null;

      displayFileSelection({
        name: meta.filename,
        size: meta.file_size,
        frames: meta.frames,
        fps: meta.fps,
        duration: meta.duration_sec,
        previewUrl: meta.preview_url,
      });

      setVideoSource(videoPlayerInput, meta.preview_url, inputPlaceholder);
      resetAnnotatedPlayer();
      btnAnalyze.disabled = false;

      resetTelemetryView();
      fetchIncidents();
      fetchIncidentStats();
    } catch (err) {
      showError(err.message);
      resetUploadState();
    } finally {
      showProcessing(false);
    }
  }

  function displayFileSelection(meta) {
    selectedFileName.textContent = meta.name;
    const sizeMb = (meta.size / (1024 * 1024)).toFixed(2);
    selectedFileSpecs.textContent = `${sizeMb} MB · ${meta.frames || '?'} frames · ${meta.fps || 25} FPS · ${meta.duration || 0}s`;
    selectedFileInfo.style.display = "flex";
  }

  function resetUploadState() {
    currentVideoId = null;
    currentSessionId = null;
    currentFile = null;
    fileInput.value = "";
    selectedFileInfo.style.display = "none";
    btnAnalyze.disabled = true;
    document.querySelectorAll(".sample-btn").forEach(b => b.classList.remove("active"));

    setVideoSource(videoPlayerInput, null, inputPlaceholder);
    resetAnnotatedPlayer();
    resetTelemetryView();
    fetchIncidents();
    fetchIncidentStats();
    clearError();
  }

  function resetTelemetryView() {
    latestTrackedPeople = [];
    latestTelemetryTimeline = {};
    if (trackedPersonsGrid) trackedPersonsGrid.innerHTML = "";
    if (emptyResultsPlaceholder) {
      emptyResultsPlaceholder.style.display = "block";
      emptyResultsPlaceholder.textContent = "Run video analysis to inspect tracked students, body behaviour classes, and facial emotions.";
    }
    if (statsBreakdownGrid) statsBreakdownGrid.style.display = "none";
    if (liveScrubberFrame) liveScrubberFrame.textContent = "Frame —";
    if (liveFramePeopleCount) liveFramePeopleCount.textContent = "0 detected";
    if (metricActiveTracks) metricActiveTracks.textContent = "0";
    if (totalTrackSessionsCount) totalTrackSessionsCount.textContent = "0";
    if (statInFrameCountPill) statInFrameCountPill.style.display = "none";
  }

  // ==========================================================================
  // 4. Video Analysis Execution
  // ==========================================================================
  btnAnalyze.addEventListener("click", async () => {
    if (!currentVideoId || isProcessing) return;

    clearError();
    const maxFrames = selectMaxFrames.value;
    const isSample = Boolean(maxFrames);
    const msg = isSample 
      ? `Processing Fast Test Sample (${maxFrames} frames)...`
      : `Processing Full Video across all frames (YOLO11 + Pose BiLSTM + BlazeFace + Emotion)...`;
    showProcessing(true, msg);
    btnAnalyze.disabled = true;

    const progressBarFill = document.querySelector(".progress-bar-fill");
    if (progressBarFill) {
      progressBarFill.classList.add("determinate");
      progressBarFill.style.width = "0%";
    }

    const url = maxFrames 
      ? `/api/analyze/${currentVideoId}?max_frames=${maxFrames}`
      : `/api/analyze/${currentVideoId}`;

    const pollInterval = setInterval(async () => {
      try {
        const progRes = await fetch(`/api/progress/${currentVideoId}`);
        if (progRes.ok) {
          const prog = await progRes.json();
          if (prog.status === "processing" && prog.total_frames > 0) {
            const scopeLabel = isSample ? `Sample (${maxFrames}f)` : `Full Video`;
            progressStatusText.textContent = `Analyzing ${scopeLabel}: Frame ${prog.current_frame}/${prog.total_frames} (${prog.percent}%) · ${prog.fps} FPS`;
            if (progressBarFill) {
              progressBarFill.style.width = `${Math.min(100, prog.percent)}%`;
            }
          }
        }
      } catch (e) {}
    }, 500);

    try {
      const res = await fetch(url, { method: "POST" });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Video analysis failed on server.");
      }

      if (progressBarFill) {
        progressBarFill.style.width = "100%";
      }
      const data = await res.json();
      currentSessionId = data.session_id;
      currentFPS = data.processing_fps || 25.0;
      renderAnalysisResults(data);
    } catch (err) {
      showError(err.message);
    } finally {
      clearInterval(pollInterval);
      showProcessing(false);
      btnAnalyze.disabled = false;
      if (progressBarFill) {
        progressBarFill.classList.remove("determinate");
        progressBarFill.style.width = "100%";
      }
    }
  });

  // ==========================================================================
  // 5. Results & Telemetry Rendering (Simplified, Clutter-Free)
  // ==========================================================================
  function renderAnalysisResults(data) {
    // 1. Mount Annotated Video Player
    const annotatedUrl = data.video_urls.annotated;
    setVideoSource(videoPlayerAnnotated, annotatedUrl, annotatedPlaceholder);

    // 2. Populate Header Stats Cards
    const stats = data.statistics;
    statPersistentTracks.textContent = stats.persistent_tracks_count;
    statBehaviourPreds.textContent = stats.total_behaviour_predictions;
    statEmotionPreds.textContent = stats.total_emotion_predictions;
    statProcessingSpeed.textContent = `${data.processing_fps} FPS`;

    // 3. Update Telemetry Context Bar
    if (metricActiveTracks) metricActiveTracks.textContent = stats.persistent_tracks_count;
    if (totalTrackSessionsCount) totalTrackSessionsCount.textContent = stats.total_observed_tracks_count || stats.persistent_tracks_count;
    if (liveFramePeopleCount) liveFramePeopleCount.textContent = `${stats.current_visible_detections || 0} detected`;
    if (liveScrubberFrame) liveScrubberFrame.textContent = `Frame ${data.total_frames_processed}`;

    if (statInFrameCountPill) {
      statInFrameCountPill.textContent = `${stats.current_visible_detections || 0} in Frame`;
      statInFrameCountPill.style.display = "inline-block";
    }

    // 4. Save Timeline and Render Tracked Individuals Grid
    latestTrackedPeople = data.tracked_people || [];
    renderTrackedPeopleGrid();

    // 5. Render Distribution Bars
    renderDistributions(stats.behaviour_distribution, stats.emotion_distribution);
    statsBreakdownGrid.style.display = "grid";

    // 6. Refresh Incident Management Log
    fetchIncidents();
    fetchIncidentStats();
    if (data.recorded_incidents_count && data.recorded_incidents_count > 0) {
      showToast(`Detected & recorded ${data.recorded_incidents_count} high-risk security incident(s) for review.`, "success");
    }
  }

  function renderTrackedPeopleGrid() {
    if (!trackedPersonsGrid) return;

    let filtered = latestTrackedPeople;
    if (trackFilterMode === "risk") {
      filtered = latestTrackedPeople.filter(p => {
        const b = p.latest_behaviour || {};
        const risk = (b.risk_level || "").toLowerCase();
        const cname = (b.class_name || "").toLowerCase();
        return risk === "high" || cname.includes("fight") || cname.includes("fall");
      });
    }

    if (filtered.length === 0) {
      emptyResultsPlaceholder.style.display = "block";
      emptyResultsPlaceholder.textContent = trackFilterMode === "risk"
        ? "No tracks currently exhibiting High-Risk behaviour."
        : "No individuals were detected or tracked in this video segment.";
      trackedPersonsGrid.innerHTML = "";
    } else {
      emptyResultsPlaceholder.style.display = "none";
      trackedPersonsGrid.innerHTML = "";

      filtered.forEach(person => {
        const card = createPersonTelemetryCard(person);
        trackedPersonsGrid.appendChild(card);
      });
    }
  }

  // Filter Chip Event Listeners
  if (filterTracksAll) {
    filterTracksAll.addEventListener("click", () => {
      trackFilterMode = "all";
      filterTracksAll.classList.add("active");
      if (filterTracksRisk) filterTracksRisk.classList.remove("active");
      renderTrackedPeopleGrid();
    });
  }

  if (filterTracksRisk) {
    filterTracksRisk.addEventListener("click", () => {
      trackFilterMode = "risk";
      filterTracksRisk.classList.add("active");
      if (filterTracksAll) filterTracksAll.classList.remove("active");
      renderTrackedPeopleGrid();
    });
  }

  // Live Scrubber Synchronization on Video Playback
  function updateLiveScrubber(videoEl) {
    if (!videoEl || !videoEl.duration) return;
    const curTime = videoEl.currentTime;
    const curFrame = Math.max(1, Math.floor(curTime * currentFPS) + 1);
    if (liveScrubberFrame) {
      liveScrubberFrame.textContent = `Frame ${curFrame} (${curTime.toFixed(1)}s)`;
    }

    // Determine how many persistent tracks were active during this frame
    let activeNow = 0;
    latestTrackedPeople.forEach(p => {
      if (curFrame >= p.first_frame && curFrame <= p.last_frame) {
        activeNow++;
      }
    });

    if (liveFramePeopleCount) {
      liveFramePeopleCount.textContent = `${activeNow} detected in frame`;
    }
    if (statInFrameCountPill) {
      statInFrameCountPill.textContent = `${activeNow} in Frame`;
      statInFrameCountPill.style.display = activeNow > 0 ? "inline-block" : "none";
    }
  }

  videoPlayerAnnotated.addEventListener("timeupdate", () => updateLiveScrubber(videoPlayerAnnotated));
  videoPlayerInput.addEventListener("timeupdate", () => updateLiveScrubber(videoPlayerInput));

  function createPersonTelemetryCard(person) {
    const card = document.createElement("div");
    card.className = "person-card compact";

    const isTemp = person.is_temporary_id;
    const tidLabel = isTemp ? `Ephemeral Detection (${person.track_id})` : `Track ID #${person.track_id}`;
    const badgeClass = isTemp ? "badge-temporary" : "badge-persistent";
    const badgeText = isTemp ? "Untracked" : "Persistent Track";

    // Body behaviour evaluation
    const b = person.latest_behaviour;
    let bClassTag = "tag-collecting";
    let bLabelText = b.class_name || "Unavailable";
    let bSubNote = "";
    let isHighRisk = false;

    if (b.status === "predicted") {
      const risk = (b.risk_level || "").toLowerCase();
      if (risk === "high" || (bLabelText.includes("Fight") && !bLabelText.includes("Non"))) {
        bClassTag = "tag-fight";
        isHighRisk = true;
      } else if (risk === "high" || bLabelText.includes("Fall")) {
        bClassTag = "tag-fight";
        isHighRisk = true;
      } else {
        bClassTag = "tag-normal";
      }
    } else if (b.status === "collecting") {
      const collected = b.collected || 0;
      if (collected > 0) {
        bLabelText = `Collecting ${collected}/30`;
        bSubNote = `Gathering observation sequence (${collected}/30 frames)`;
      } else {
        bLabelText = "Pose Unavailable";
        bSubNote = "Keypoints unresolvable in pose detector (occlusion / partial body)";
      }
    } else {
      bLabelText = "Pose Unavailable";
      bSubNote = "Pose keypoints unresolvable (e.g. occlusion, distant subject, non-frontal view)";
    }

    const bConfidenceText = b.status === "predicted" 
      ? `${(b.confidence * 100).toFixed(1)}%` 
      : `${b.confidence.toFixed(2)}`;

    // Facial emotion evaluation
    const e = person.latest_emotion;
    const isAvailable = e.status === "available";
    let eClassTag = isAvailable ? `tag-${(e.label || "").toLowerCase()}` : "tag-unavailable";
    let eLabelText = isAvailable ? e.label : "Unavailable";
    let eConfidenceText = isAvailable ? `${(e.confidence * 100).toFixed(1)}%` : "0.0%";

    let eReasonText = "";
    if (!isAvailable) {
      if (e.reason === "no_face_detected") {
        eReasonText = "No face detected in upper head zone (occluded or turned)";
      } else if (e.reason === "face_ambiguous") {
        eReasonText = "Ambiguous face match between overlapping individuals (rejected)";
      } else {
        eReasonText = "Face detection unassociated or uncertain";
      }
    }

    const riskBadge = isHighRisk 
      ? `<span class="badge-status-pill dismissed" style="background:rgba(244,63,94,0.2);color:#fda4af;font-size:0.68rem;padding:2px 6px;">Altercation Detected</span>`
      : `<span class="badge-status-pill reviewed" style="font-size:0.68rem;padding:2px 6px;">Normal Activity</span>`;

    card.innerHTML = `
      <!-- Compact Summary Header -->
      <div class="person-card-header">
        <div class="person-id-badge">
          <span>${tidLabel}</span>
          <span class="badge-pill ${badgeClass}">${badgeText}</span>
          ${riskBadge}
        </div>
        <div class="person-frames-meta">
          Frame ${person.first_frame}–${person.last_frame} (${person.first_timestamp_sec}s–${person.last_timestamp_sec}s) · ${person.total_frames_tracked} frames
        </div>
      </div>

      <!-- Compact Row: Behaviour & Emotion Quick Pills -->
      <div class="person-card-compact-row">
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
          <span class="prediction-tag ${bClassTag}" style="font-size:0.78rem;">● Body: ${bLabelText} (${bConfidenceText})</span>
          <span class="prediction-tag ${eClassTag}" style="font-size:0.78rem;">● Face: ${eLabelText} (${isAvailable ? eConfidenceText : '—'})</span>
        </div>
        <button class="btn-track-toggle" id="btn-toggle-${person.track_id}">
          <span>▼ Full Telemetry</span>
        </button>
      </div>

      <!-- Expandable Telemetry Details Accordion (Hidden by default) -->
      <div class="person-card-details" id="details-${person.track_id}" style="display: none;">
        <!-- Branch 1: Body Behaviour -->
        <div class="branch-block">
          <div class="branch-title">
            <span>Body Behaviour Telemetry</span>
            <span>${b.status === "predicted" ? "BiLSTM Model" : "Buffer Gathering"}</span>
          </div>
          <div class="branch-badge-row">
            <span class="prediction-tag ${bClassTag}">● ${bLabelText}</span>
            <span class="conf-tag">Conf: ${bConfidenceText}</span>
          </div>
          ${bSubNote ? `<div class="unavailable-reason">ℹ ${bSubNote}</div>` : ""}
        </div>

        <!-- Branch 2: Facial Emotion -->
        <div class="branch-block">
          <div class="branch-title">
            <span>Facial Emotion Telemetry</span>
            <span>${isAvailable ? "EfficientNetB0" : "Unassociated"}</span>
          </div>
          <div class="branch-badge-row">
            <span class="prediction-tag ${eClassTag}">● ${eLabelText}</span>
            <span class="conf-tag">${isAvailable ? `Conf: ${eConfidenceText}` : "—"}</span>
          </div>
          ${eReasonText ? `<div class="unavailable-reason">ℹ ${eReasonText}</div>` : ""}
        </div>

        <!-- Ethical & Verified Link Section -->
        <div style="display:flex;justify-content:space-between;align-items:center;padding-top:6px;border-top:1px dashed var(--border-subtle);flex-wrap:wrap;gap:6px;">
          <span style="font-size:0.7rem;color:var(--text-muted);">
            ℹ Temporary YOLO11 tracking ID — not a verified student identity.
          </span>
          <button class="btn-link-person" id="btn-link-person-${person.track_id}">
            🔗 Link to Registered Student/Staff
          </button>
        </div>
      </div>
    `;

    // Accordion Toggle Event
    const btnToggle = card.querySelector(`#btn-toggle-${person.track_id}`);
    const detailsContainer = card.querySelector(`#details-${person.track_id}`);
    btnToggle.addEventListener("click", () => {
      const isExpanded = detailsContainer.style.display !== "none";
      detailsContainer.style.display = isExpanded ? "none" : "flex";
      btnToggle.innerHTML = isExpanded ? "<span>▼ Full Telemetry</span>" : "<span>▲ Collapse</span>";
    });

    // Person Linkage Button Event
    const btnLink = card.querySelector(`#btn-link-person-${person.track_id}`);
    if (btnLink) {
      btnLink.addEventListener("click", () => openLinkPersonModal(person.track_id));
    }

    return card;
  }

  function renderDistributions(behDist, emoDist) {
    const totalBeh = Object.values(behDist).reduce((a, b) => a + b, 0) || 1;
    behaviourDistList.innerHTML = Object.entries(behDist).map(([cls, count]) => {
      const pct = ((count / totalBeh) * 100).toFixed(1);
      let color = "var(--accent-primary)";
      if (cls === "Fight") color = "var(--accent-rose)";
      else if (cls === "Fall") color = "var(--accent-orange)";
      else if (cls === "NonFight") color = "var(--accent-emerald)";
      else if (cls === "Collecting") color = "var(--accent-amber)";
      else if (cls === "Unavailable") color = "var(--text-muted)";

      return `
        <div class="dist-item">
          <div class="dist-item-top">
            <span>${cls}</span>
            <span>${count} (${pct}%)</span>
          </div>
          <div class="dist-bar-bg">
            <div class="dist-bar-fill" style="width: ${pct}%; background: ${color};"></div>
          </div>
        </div>
      `;
    }).join("");

    const totalEmo = Object.values(emoDist).reduce((a, b) => a + b, 0) || 1;
    emotionDistList.innerHTML = Object.entries(emoDist).map(([cls, count]) => {
      const pct = ((count / totalEmo) * 100).toFixed(1);
      let color = "var(--accent-cyan)";
      if (cls === "Unavailable") color = "var(--text-muted)";
      else if (cls === "Happy") color = "var(--accent-emerald)";
      else if (cls === "Angry") color = "var(--accent-rose)";

      return `
        <div class="dist-item">
          <div class="dist-item-top">
            <span>${cls}</span>
            <span>${count} (${pct}%)</span>
          </div>
          <div class="dist-bar-bg">
            <div class="dist-bar-fill" style="width: ${pct}%; background: ${color};"></div>
          </div>
        </div>
      `;
    }).join("");
  }

  // ==========================================================================
  // 6. Helpers & UI Utilities
  // ==========================================================================
  function setVideoSource(videoEl, src, placeholderEl) {
    if (src) {
      const cacheBustSrc = src.includes("?") ? `${src}&_t=${Date.now()}` : `${src}?_t=${Date.now()}`;
      videoEl.src = cacheBustSrc;
      videoEl.style.display = "block";
      if (placeholderEl) placeholderEl.style.display = "none";
      videoEl.load();
    } else {
      videoEl.removeAttribute("src");
      videoEl.style.display = "none";
      if (placeholderEl) placeholderEl.style.display = "flex";
    }
  }

  videoPlayerAnnotated.addEventListener("error", () => {
    if (videoPlayerAnnotated.src) {
      console.warn("Annotated player error:", videoPlayerAnnotated.error);
      showError("The browser encountered an error decoding or playing the annotated output video.");
    }
  });

  function resetAnnotatedPlayer() {
    setVideoSource(videoPlayerAnnotated, null, annotatedPlaceholder);
  }

  let processingTimer = null;
  let processingStart = 0;

  function showProcessing(show, text = "Processing...") {
    isProcessing = show;
    if (processingTimer) {
      clearInterval(processingTimer);
      processingTimer = null;
    }
    if (show) {
      processingStart = Date.now();
      progressStatusText.textContent = text;
      processingPanel.style.display = "block";
      processingTimer = setInterval(() => {
        const sec = Math.floor((Date.now() - processingStart) / 1000);
        progressStatusText.textContent = `${text} (${sec}s elapsed)`;
      }, 1000);
    } else {
      processingPanel.style.display = "none";
    }
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    errorBanner.style.display = "block";
  }

  function clearError() {
    errorMessage.textContent = "";
    errorBanner.style.display = "none";
  }

  // ==========================================================================
  // 7. Incident Management & Review Log Logic (Scope-Filtered)
  // ==========================================================================

  if (btnScopeCurrentVideo) {
    btnScopeCurrentVideo.addEventListener("click", () => switchIncidentScope("current"));
  }

  if (btnScopeHistorical) {
    btnScopeHistorical.addEventListener("click", () => switchIncidentScope("historical"));
  }

  function switchIncidentScope(scope) {
    incidentScope = scope;
    if (scope === "current") {
      btnScopeCurrentVideo.classList.add("active");
      btnScopeHistorical.classList.remove("active");
      if (incidentScopeNotice) {
        incidentScopeNotice.textContent = currentVideoId 
          ? `Showing incidents for active video (${currentVideoId})`
          : `Showing current video incidents (None loaded)`;
      }
    } else {
      btnScopeHistorical.classList.add("active");
      btnScopeCurrentVideo.classList.remove("active");
      if (incidentScopeNotice) {
        incidentScopeNotice.textContent = "Showing all historical incidents across all sessions";
      }
    }
    fetchIncidents();
  }

  async function fetchIncidents() {
    try {
      const statusVal = filterIncidentStatus ? filterIncidentStatus.value : "";
      const behVal = filterIncidentBehaviour ? filterIncidentBehaviour.value : "";

      const params = new URLSearchParams();
      if (statusVal) params.append("status", statusVal);
      if (behVal) params.append("behaviour", behVal);
      params.append("limit", "100");

      // Scope filtering: if scope is 'current', filter by active video
      if (incidentScope === "current" && currentVideoId) {
        params.append("video_id", currentVideoId);
      }

      const res = await fetch(`/api/incidents?${params.toString()}`);
      if (!res.ok) return;
      const data = await res.json();
      allLoadedIncidents = data.incidents || [];
      filterAndRenderLocal();

      updateScopeBadgeCounts();
    } catch (err) {
      console.warn("Could not fetch incidents:", err);
    }
  }

  async function updateScopeBadgeCounts() {
    try {
      if (currentVideoId && countCurrentVideoIncidents) {
        const r1 = await fetch(`/api/incidents?video_id=${currentVideoId}&limit=1`);
        if (r1.ok) {
          const d1 = await r1.json();
          countCurrentVideoIncidents.textContent = d1.total_count || 0;
        }
      } else if (countCurrentVideoIncidents) {
        countCurrentVideoIncidents.textContent = "0";
      }

      if (countAllHistoricalIncidents) {
        const r2 = await fetch(`/api/incidents?limit=1`);
        if (r2.ok) {
          const d2 = await r2.json();
          countAllHistoricalIncidents.textContent = d2.total_count || 0;
        }
      }
    } catch (e) {}
  }

  async function fetchIncidentStats() {
    try {
      const res = await fetch("/api/incidents/stats/summary");
      if (!res.ok) return;
      const data = await res.json();
      const stats = data.statistics || {};
      if (statIncidentsCount) {
        statIncidentsCount.textContent = stats.total_incidents || 0;
      }
      if (statPendingCountPill) {
        const pending = stats.pending_review || 0;
        statPendingCountPill.textContent = `${pending} Pending`;
        statPendingCountPill.style.display = pending > 0 ? "inline-block" : "none";
      }
    } catch (err) {
      console.warn("Could not fetch incident stats:", err);
    }
  }

  function filterAndRenderLocal() {
    const q = inputIncidentSearch ? inputIncidentSearch.value.trim().toLowerCase() : "";
    let filtered = allLoadedIncidents;
    if (q) {
      filtered = allLoadedIncidents.filter(inc => {
        const idMatch = inc.id.toLowerCase().includes(q);
        const vidMatch = (inc.video_id || "").toLowerCase().includes(q);
        const trackMatch = inc.track_id !== null && inc.track_id.toString().includes(q);
        const fileMatch = (inc.source_filename || "").toLowerCase().includes(q);
        return idMatch || vidMatch || trackMatch || fileMatch;
      });
    }
    renderIncidentsTable(filtered);
  }

  function renderIncidentsTable(incidents) {
    if (!incidentsTableBody) return;

    if (!incidents || incidents.length === 0) {
      incidentsTableBody.innerHTML = "";
      if (incidentsEmptyPlaceholder) incidentsEmptyPlaceholder.style.display = "flex";
      return;
    }

    if (incidentsEmptyPlaceholder) incidentsEmptyPlaceholder.style.display = "none";
    incidentsTableBody.innerHTML = "";

    incidents.forEach(inc => {
      const row = document.createElement("tr");

      // 1. ID
      const tdId = document.createElement("td");
      tdId.innerHTML = `<span class="badge-inc-id">${inc.id}</span>`;

      // 2. Time & Source
      const tdTime = document.createElement("td");
      const spanFrames = `Frame ${inc.start_frame}–${inc.end_frame}`;
      tdTime.innerHTML = `
        <div style="font-weight:600;color:var(--text-primary);">${inc.event_time_sec}s · ${spanFrames}</div>
        <div style="font-size:0.72rem;color:var(--text-muted);">${inc.source_filename || 'Video'}</div>
      `;

      // 3. Behaviour Event (RED for Fight, ORANGE for Fall)
      const tdEvent = document.createElement("td");
      const confPct = Math.round((inc.confidence || 0) * 100);
      const isFight = inc.behaviour_label.toLowerCase().includes("fight");
      const behBadgeClass = isFight ? "badge-event-fight" : "badge-event-fall";
      tdEvent.innerHTML = `<span class="${behBadgeClass}">
        ${inc.behaviour_label.toUpperCase()} (${confPct}%)
      </span>`;

      // 4. Track Token (Explicit temporary tracker label)
      const tdTrack = document.createElement("td");
      tdTrack.innerHTML = `<span class="badge-track-token" title="Ephemeral YOLO tracker token — not real-world identity">${inc.track_display || 'Track #' + inc.track_id}</span>`;

      // 5. Emotion (Decoupled contextual telemetry)
      const tdEmotion = document.createElement("td");
      const em = inc.associated_emotion || "Unavailable";
      if (em.toLowerCase() !== "unavailable") {
        tdEmotion.innerHTML = `<span class="badge-emotion-tag">${em}</span>`;
      } else {
        tdEmotion.innerHTML = `<span class="badge-emotion-unavail">Unavailable</span>`;
      }

      // 6. Evidence Thumbnail
      const tdEvidence = document.createElement("td");
      if (inc.has_evidence && inc.evidence_url) {
        const thumb = document.createElement("div");
        thumb.className = "evidence-thumb-wrapper";
        thumb.title = "Click to inspect evidence frame screenshot";
        thumb.innerHTML = `<img class="evidence-thumb-img" src="${inc.evidence_url}" alt="Evidence Frame" loading="lazy">`;
        thumb.addEventListener("click", () => openZoomModal(inc.evidence_url));
        tdEvidence.appendChild(thumb);
      } else {
        tdEvidence.innerHTML = `<span class="evidence-none-pill">Unavailable</span>`;
      }

      // 7. Review Status
      const tdStatus = document.createElement("td");
      tdStatus.innerHTML = `<span class="badge-status-pill ${inc.review_status}">${inc.review_status}</span>`;

      // 8. Actions
      const tdActions = document.createElement("td");
      const btnRev = document.createElement("button");
      btnRev.className = "btn-action-review";
      btnRev.textContent = inc.review_status === "pending" ? "Review & Audit" : "Audit Details";
      btnRev.addEventListener("click", () => openReviewModal(inc.id));
      tdActions.appendChild(btnRev);

      row.appendChild(tdId);
      row.appendChild(tdTime);
      row.appendChild(tdEvent);
      row.appendChild(tdTrack);
      row.appendChild(tdEmotion);
      row.appendChild(tdEvidence);
      row.appendChild(tdStatus);
      row.appendChild(tdActions);

      incidentsTableBody.appendChild(row);
    });
  }

  // ==========================================================================
  // 8. Incident Review Modal Handlers
  // ==========================================================================
  async function openReviewModal(incidentId) {
    activeReviewIncidentId = incidentId;
    try {
      const res = await fetch(`/api/incidents/${incidentId}`);
      if (!res.ok) {
        showToast("Failed to load incident details from server.", "error");
        return;
      }
      const data = await res.json();
      const inc = data.incident;

      modalIncidentTitle.textContent = `Review Incident: ${inc.id}`;
      modalIncidentSubtitle.textContent = `High-Risk ${inc.behaviour_label} Detection · Source: ${inc.source_filename}`;

      modalMetaBehaviour.textContent = `${inc.behaviour_label} (${inc.risk_level} Risk)`;
      modalMetaConfidence.textContent = `${(inc.confidence * 100).toFixed(1)}% model confidence`;
      modalMetaTrackId.textContent = inc.track_display || `Track #${inc.track_id}`;
      modalMetaTimeSpan.textContent = `${inc.event_time_sec}s (Frame ${inc.start_frame}–${inc.end_frame})`;
      modalMetaEmotion.textContent = inc.associated_emotion !== "Unavailable" 
        ? `${inc.associated_emotion} (${((inc.emotion_confidence || 0) * 100).toFixed(0)}%)`
        : "Unavailable (No associated face)";
      modalMetaSource.textContent = inc.source_filename || "Video Source";

      if (inc.has_evidence && inc.evidence_url) {
        modalEvidenceImg.src = inc.evidence_url;
        modalEvidenceImg.style.display = "block";
        modalEvidencePlaceholder.style.display = "none";
      } else {
        modalEvidenceImg.removeAttribute("src");
        modalEvidenceImg.style.display = "none";
        modalEvidencePlaceholder.style.display = "flex";
      }

      // Pre-select current review status
      const radio = document.querySelector(`input[name="reviewStatus"][value="${inc.review_status}"]`);
      if (radio) radio.checked = true;

      inputReviewerName.value = inc.reviewed_by || "Authorized Reviewer";
      textareaReviewerNotes.value = inc.reviewer_notes || "";

      incidentReviewModal.style.display = "flex";
    } catch (err) {
      showToast("Error fetching incident record.", "error");
    }
  }

  function closeReviewModal() {
    incidentReviewModal.style.display = "none";
    activeReviewIncidentId = null;
  }

  if (modalCloseBtn) modalCloseBtn.addEventListener("click", closeReviewModal);
  if (btnModalCancel) btnModalCancel.addEventListener("click", closeReviewModal);

  if (incidentReviewForm) {
    incidentReviewForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!activeReviewIncidentId) return;

      const selectedRadio = document.querySelector('input[name="reviewStatus"]:checked');
      const newStatus = selectedRadio ? selectedRadio.value : "reviewed";
      const notes = textareaReviewerNotes.value.trim();
      const reviewer = inputReviewerName.value.trim();

      try {
        const res = await fetch(`/api/incidents/${activeReviewIncidentId}/review`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            review_status: newStatus,
            reviewer_notes: notes,
            reviewer_name: reviewer,
          }),
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || "Failed to update review status.");
        }

        showToast(`Incident ${activeReviewIncidentId} marked as ${newStatus}.`, "success");
        closeReviewModal();
        fetchIncidents();
        fetchIncidentStats();
      } catch (err) {
        showToast(err.message, "error");
      }
    });
  }

  // ==========================================================================
  // 9. Verified Person Linkage Modal (Authorized Auditor Review)
  // ==========================================================================
  async function openLinkPersonModal(trackId) {
    activeLinkTrackId = trackId;
    if (linkTrackTokenDisplay) {
      linkTrackTokenDisplay.value = `Track #${trackId} (Session: ${currentSessionId || 'Current Video'})`;
    }

    // Populate registry options
    try {
      const res = await fetch("/api/verified-persons");
      if (res.ok) {
        const data = await res.json();
        const persons = data.persons || [];
        if (selectVerifiedPerson) {
          selectVerifiedPerson.innerHTML = `<option value="">-- Choose registered student/staff --</option>`;
          persons.forEach(p => {
            const opt = document.createElement("option");
            opt.value = p.id;
            opt.textContent = `${p.full_name} (${p.official_id}) — ${p.role.toUpperCase()}`;
            selectVerifiedPerson.appendChild(opt);
          });
        }
      }
    } catch (e) {}

    if (linkPersonModal) linkPersonModal.style.display = "flex";
  }

  function closeLinkPersonModal() {
    if (linkPersonModal) linkPersonModal.style.display = "none";
    activeLinkTrackId = null;
  }

  if (linkModalCloseBtn) linkModalCloseBtn.addEventListener("click", closeLinkPersonModal);
  if (btnLinkCancel) btnLinkCancel.addEventListener("click", closeLinkPersonModal);

  if (linkPersonForm) {
    linkPersonForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!activeLinkTrackId) return;

      const personId = selectVerifiedPerson.value;
      if (!personId) {
        showToast("Please select a registered person from the list.", "error");
        return;
      }
      const reviewer = inputLinkReviewer.value.trim();
      const notes = textareaLinkNotes.value.trim();
      const sessId = currentSessionId || `ses_legacy_${(currentVideoId || 'vid').substring(0, 8)}`;

      try {
        const res = await fetch(`/api/sessions/${sessId}/tracks/${activeLinkTrackId}/link`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            person_id: personId,
            linked_by: reviewer,
            notes: notes,
          }),
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || "Failed to link track to person.");
        }

        showToast(`Track #${activeLinkTrackId} successfully linked to registered individual.`, "success");
        closeLinkPersonModal();
      } catch (err) {
        showToast(err.message, "error");
      }
    });
  }

  // ==========================================================================
  // 10. Evidence Zoom Modal Handlers & Toast
  // ==========================================================================
  function openZoomModal(imgUrl) {
    if (zoomModalImg) zoomModalImg.src = imgUrl;
    if (evidenceZoomModal) evidenceZoomModal.style.display = "flex";
  }

  function closeZoomModal() {
    if (evidenceZoomModal) evidenceZoomModal.style.display = "none";
  }

  if (zoomModalCloseBtn) zoomModalCloseBtn.addEventListener("click", closeZoomModal);

  window.addEventListener("click", (e) => {
    if (e.target === incidentReviewModal) closeReviewModal();
    if (e.target === evidenceZoomModal) closeZoomModal();
    if (e.target === linkPersonModal) closeLinkPersonModal();
  });

  function showToast(message, type = "success") {
    if (!toastContainer) return;
    const toast = document.createElement("div");
    toast.className = `toast-message ${type}`;
    const icon = type === "success" ? "✓" : "⚠";
    toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(40px)";
      toast.style.transition = "all 0.25s ease-out";
      setTimeout(() => toast.remove(), 250);
    }, 4000);
  }

  // Filter Toolbar Event Handlers
  if (filterIncidentStatus) filterIncidentStatus.addEventListener("change", fetchIncidents);
  if (filterIncidentBehaviour) filterIncidentBehaviour.addEventListener("change", fetchIncidents);
  if (inputIncidentSearch) inputIncidentSearch.addEventListener("input", filterAndRenderLocal);
  if (btnRefreshIncidents) {
    btnRefreshIncidents.addEventListener("click", () => {
      fetchIncidents();
      fetchIncidentStats();
      showToast("Incident log refreshed.", "success");
    });
  }
});
