/* ══════════════════════════════════════════════════════════════════
   Deep Research Agent — Frontend Application Logic
   ══════════════════════════════════════════════════════════════════ */

// ── Step definitions (mirrors server.py STEP_META order) ────────────────────
const STEPS = [
  {
    id: "search_research",
    label: "Web Research",
    icon: "🔍",
    desc: "Searching DuckDuckGo + scraping pages",
  },
  {
    id: "analyze_video",
    label: "Video Analysis",
    icon: "🎬",
    desc: "Extracting & summarizing YouTube transcript",
  },
  {
    id: "create_report",
    label: "Generating Report",
    icon: "📄",
    desc: "Synthesizing research into a full report",
  },
  {
    id: "create_podcast",
    label: "Creating Podcast",
    icon: "🎙️",
    desc: "Writing dialogue + rendering audio with Piper",
  },
];

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

// ── Build step cards ─────────────────────────────────────────────────────────
function buildStepCards() {
  stepsGrid.innerHTML = "";
  STEPS.forEach((step) => {
    state.stepStatus[step.id] = "idle";

    const card = document.createElement("div");
    card.className = "step-card";
    card.id = `step-card-${step.id}`;
    card.dataset.status = "idle";

    card.innerHTML = `
      <div class="step-icon-row">
        <span class="step-emoji" aria-hidden="true">${step.icon}</span>
        <span class="step-status-icon" aria-label="Status">
          <span class="step-spinner" aria-hidden="true"></span>
          <svg class="step-check" viewBox="0 0 18 18" fill="none" aria-hidden="true">
            <circle cx="9" cy="9" r="8.5" stroke="currentColor" stroke-width="1.2"/>
            <path d="M5.5 9 L7.8 11.3 L12.5 6.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </span>
      </div>
      <div class="step-name">${step.label}</div>
      <div class="step-desc">${step.desc}</div>
      <div class="thinking-indicator" aria-live="polite">
        <span class="thinking-label">Thinking</span>
        <div class="thinking-dots" aria-hidden="true">
          <span></span><span></span><span></span>
        </div>
      </div>
    `;

    stepsGrid.appendChild(card);
  });
}

// ── Update a single step card ─────────────────────────────────────────────────
function setStepStatus(stepId, status) {
  state.stepStatus[stepId] = status;
  const card = document.getElementById(`step-card-${stepId}`);
  if (card) {
    card.dataset.status = status;
    // Update aria-label for accessibility
    const statusLabels = { idle: "Pending", active: "In Progress", done: "Complete" };
    card.setAttribute("aria-label", `${STEPS.find(s => s.id === stepId)?.label} — ${statusLabels[status] || status}`);
  }
  updateProgressBar();
}

// ── Progress bar ──────────────────────────────────────────────────────────────
function updateProgressBar() {
  const doneCount = Object.values(state.stepStatus).filter((s) => s === "done").length;
  const pct = Math.round((doneCount / STEPS.length) * 100);
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
      const stepId = data.step;
      // Mark all previous steps done if they were still active
      STEPS.forEach((s) => {
        if (state.stepStatus[s.id] === "active") {
          setStepStatus(s.id, "done");
        }
      });
      setStepStatus(stepId, "active");

      // Scroll the active card into view smoothly
      const card = document.getElementById(`step-card-${stepId}`);
      if (card) card.scrollIntoView({ behavior: "smooth", block: "nearest" });
      break;
    }

    case "step_done": {
      const stepId = data.step;
      setStepStatus(stepId, "done");
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
