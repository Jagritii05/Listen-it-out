/* ══════════════════════════════════════════════════════════════════
   Deep Research Agent — Frontend Application Logic
   ══════════════════════════════════════════════════════════════════ */

// ── Step definitions (mirrors server.py STEP_META order) ────────────────────
const STEPS = [];

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  running: false,
  stepStatus: {}, // id → 'idle' | 'active' | 'done'
  activeController: null, // AbortController for the fetch
};

// ── DOM refs ─────────────────────────────────────────────────────────────────
const topicInput = document.getElementById("topic-input");
const videoInput = document.getElementById("video-input");
const researchBtn = document.getElementById("research-btn");
const progressSection = document.getElementById("progress-section");
const stepsGrid = document.getElementById("steps-grid");
const progressBar = document.getElementById("progress-bar");
const progressLabel = document.getElementById("progress-label");
const resultsSection = document.getElementById("results-section");
const reportContent = document.getElementById("report-content");
const podcastScript = document.getElementById("podcast-script");
const audioPlayerWrap = document.getElementById("audio-player-wrap");
const podcastAudio = document.getElementById("podcast-audio");
const errorToast = document.getElementById("error-toast");
const errorMessage = document.getElementById("error-message");
const toastClose = document.getElementById("toast-close");

// ── Fixed pipeline phases shown to user (regardless of how many tool calls happen)
const PHASE_CARDS = [
  { id: "phase_research",  label: "Researching",       icon: "🔍", desc: "Searching the web and gathering source material" },
  { id: "phase_report",   label: "Generating Report",  icon: "📄", desc: "Synthesizing findings into a structured report" },
  { id: "phase_podcast",  label: "Creating Podcast",   icon: "🎙️", desc: "Writing script and rendering audio with Piper TTS" },
];

// Map each tool name to the phase card it belongs to
// create_report_and_podcast handles both report + podcast internally
const TOOL_TO_PHASE = {
  search_the_web:             "phase_research",
  read_youtube_transcript:    "phase_research",
  synthesize_final_report:    "phase_report",      // legacy fallback
  generate_podcast_audio:     "phase_podcast",     // legacy fallback
  create_report_and_podcast:  "phase_report",      // new combined tool starts at report phase
};

function buildStepCards() {
  stepsGrid.innerHTML = "";
  STEPS.length = 0;
  state.stepStatus = {};

  PHASE_CARDS.forEach(phase => {
    STEPS.push({ id: phase.id, label: phase.label, icon: phase.icon });
    state.stepStatus[phase.id] = "idle";

    const card = document.createElement("div");
    card.className = "step-card";
    card.id = `step-card-${phase.id}`;
    card.dataset.status = "idle";
    card.innerHTML = `
      <div class="step-icon-row">
        <span class="step-emoji" aria-hidden="true">${phase.icon}</span>
        <span class="step-status-icon" aria-label="Status">
          <span class="step-spinner" aria-hidden="true"></span>
          <svg class="step-check" viewBox="0 0 18 18" fill="none" aria-hidden="true">
            <circle cx="9" cy="9" r="8.5" stroke="currentColor" stroke-width="1.2"/>
            <path d="M5.5 9 L7.8 11.3 L12.5 6.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </span>
      </div>
      <div class="step-name">${phase.label}</div>
      <div class="step-desc">${phase.desc}</div>
      <div class="thinking-indicator" aria-live="polite">
        <span class="thinking-label">Working</span>
        <div class="thinking-dots" aria-hidden="true"><span></span><span></span><span></span></div>
      </div>
    `;
    stepsGrid.appendChild(card);
  });

  updateProgressBar();
}

// ── Update a single step card ─────────────────────────────────────────────────
function setStepStatus(stepId, status) {
  state.stepStatus[stepId] = status;
  const card = document.getElementById(`step-card-${stepId}`);
  if (card) {
    card.dataset.status = status;
    const statusLabels = { idle: "Pending", active: "In Progress", done: "Complete" };
    card.setAttribute("aria-label", `${STEPS.find(s => s.id === stepId)?.label} — ${statusLabels[status] || status}`);
  }
  updateProgressBar();
}

// ── Progress bar ──────────────────────────────────────────────────────────────
function updateProgressBar() {
  const doneCount = Object.values(state.stepStatus).filter((s) => s === "done").length;
  const total = STEPS.length;
  const pct = total === 0 ? 0 : Math.round((doneCount / total) * 100);
  progressBar.style.width = `${pct}%`;
  progressLabel.textContent = `${pct}%`;
}

// ── Show / hide sections ──────────────────────────────────────────────────────
function showSection(el) {
  el.classList.remove("hidden");
  el.classList.add("fade-in");
}

// ── Render podcast script with colored speaker lines ──────────────────────────
function renderPodcastScript(script) {
  if (!script) {
    podcastScript.textContent = "No script generated.";
    return;
  }

  podcastScript.innerHTML = "";
  const lines = script.split("\n");
  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      podcastScript.appendChild(document.createTextNode("\n"));
      return;
    }

    const span = document.createElement("span");

    if (/^Mike\s*:/i.test(trimmed)) {
      span.className = "line-mike";
      span.textContent = trimmed + "\n";
    } else if (/^Dr\.?\s*Sarah\s*:/i.test(trimmed)) {
      span.className = "line-sarah";
      span.textContent = trimmed + "\n";
    } else {
      span.textContent = trimmed + "\n";
    }
    podcastScript.appendChild(span);
  });
}

// ── Show error toast ──────────────────────────────────────────────────────────
function showError(msg) {
  errorMessage.textContent = msg;
  errorToast.classList.remove("hidden");
}
function hideError() {
  errorToast.classList.add("hidden");
}

toastClose.addEventListener("click", hideError);

// ── Reset UI to initial state ────────────────────────────────────────────────
function resetUI() {
  buildStepCards();
  updateProgressBar();
  reportContent.innerHTML = "";
  podcastScript.innerHTML = "";
  podcastAudio.src = "";
  audioPlayerWrap.classList.add("hidden");
  resultsSection.classList.add("hidden");
  hideError();
}

// ── Toggle button loading state ───────────────────────────────────────────────
function setButtonLoading(loading) {
  state.running = loading;
  researchBtn.classList.toggle("is-loading", loading);
  researchBtn.disabled = loading;
  topicInput.disabled = loading;
  videoInput.disabled = loading;
}

// ── Main: handle SSE stream ───────────────────────────────────────────────────
async function startResearch() {
  const topic = topicInput.value.trim();
  const videoUrl = videoInput.value.trim() || null;

  // Must have at least one of topic or videoUrl
  if (!topic && !videoUrl) {
    // Highlight both fields to hint at least one is needed
    [topicInput, videoInput].forEach((el) => {
      el.style.borderColor = "var(--red)";
      setTimeout(() => (el.style.borderColor = ""), 1800);
    });
    showError("Please enter a research topic or a YouTube URL (or both).");
    return;
  }

  // Cancel any existing in-flight request
  if (state.activeController) {
    state.activeController.abort();
  }
  state.activeController = new AbortController();

  resetUI();
  setButtonLoading(true);
  showSection(progressSection);

  try {
    const response = await fetch("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic, video_url: videoUrl }),
      signal: state.activeController.signal,
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status} ${response.statusText}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Split on double newlines (SSE delimiter)
      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // Keep the incomplete last part

      for (const part of parts) {
        const lines = part.split("\n");
        let eventType = "message";
        let dataStr = "";

        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            dataStr = line.slice(5).trim();
          }
        }

        if (!dataStr || dataStr === "") continue;
        if (dataStr === "[DONE]") break;

        let evt;
        try {
          evt = JSON.parse(dataStr);
        } catch {
          continue; // ignore malformed payloads (e.g. heartbeat comments)
        }

        handleSSEEvent(eventType, evt);
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      showError(err.message || "An unexpected error occurred.");
    }
  } finally {
    setButtonLoading(false);
  }
}

// ── Handle individual SSE events ──────────────────────────────────────────────
function handleSSEEvent(eventType, data) {
  switch (eventType) {
    case "step_start": {
      // Map the tool name to its phase card
      const toolName = data.step?.replace("tool_", "") || data.label?.replace("Using Tool: ", "");
      const phaseId = TOOL_TO_PHASE[toolName] || "phase_research";

      // Only mark the phase active if it isn't done yet
      if (state.stepStatus[phaseId] !== "done") {
        // Mark all earlier phases done if this is a later phase
        PHASE_CARDS.forEach((p) => {
          if (p.id !== phaseId && state.stepStatus[p.id] === "active") {
            setStepStatus(p.id, "done");
          }
        });
        setStepStatus(phaseId, "active");
        const card = document.getElementById(`step-card-${phaseId}`);
        if (card) card.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
      break;
    }

    case "step_done": {
      // step id might be tool_call_id (old) or tool_name — resolve to phase either way
      const rawId = data.step || "";
      const toolName = rawId.startsWith("tool_") ? rawId.slice(5) : rawId;
      const phaseId = TOOL_TO_PHASE[toolName];
      if (phaseId) {
        // Only mark done if the next phase has started (don't close research while still searching)
        const phaseIdx = PHASE_CARDS.findIndex(p => p.id === phaseId);
        const nextPhase = PHASE_CARDS[phaseIdx + 1];
        if (!nextPhase || state.stepStatus[nextPhase.id] === "active" || state.stepStatus[nextPhase.id] === "done") {
          setStepStatus(phaseId, "done");
        }
      }
      break;
    }

    case "result": {
      // Mark all steps done
      STEPS.forEach((s) => setStepStatus(s.id, "done"));

      // Render report
      if (data.report) {
        reportContent.innerHTML = marked.parse(data.report);
      }

      // Render podcast script
      renderPodcastScript(data.podcast_script || "");

      // Podcast audio
      if (data.audio_url) {
        podcastAudio.src = data.audio_url;
        audioPlayerWrap.classList.remove("hidden");
      }

      // Show results section
      showSection(resultsSection);

      // Smooth scroll to results
      setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 300);
      break;
    }

    case "error": {
      const msg = data.message || "An error occurred in the agent pipeline.";
      showError(msg);
      // Mark any active step as idle (to stop the spinner)
      STEPS.forEach((s) => {
        if (state.stepStatus[s.id] === "active") {
          setStepStatus(s.id, "idle");
        }
      });
      break;
    }

    default:
      break;
  }
}

// ── Event listeners ───────────────────────────────────────────────────────────
researchBtn.addEventListener("click", startResearch);

topicInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !state.running) {
    e.preventDefault();
    startResearch();
  }
});

// ── Init ──────────────────────────────────────────────────────────────────────
(function init() {
  buildStepCards();
  topicInput.focus();

  // Animate orbs constantly, no JS needed — CSS handles it.
  // Set marked options for safe + clean rendering
  marked.setOptions({
    breaks: true,
    gfm: true,
  });
})();
