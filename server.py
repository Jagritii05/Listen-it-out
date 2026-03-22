"""FastAPI server with SSE streaming for the Deep Research Agent UI."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load env before importing agent graph
from dotenv import load_dotenv

load_dotenv()

# Disable LangSmith tracing noise
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

app = FastAPI(title="Deep Research Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static files ───────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent / "frontend"

# ─── Request model ──────────────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    topic: str | None = None
    video_url: str | None = None

    @property
    def effective_topic(self) -> str | None:
        return (self.topic or "").strip() or None

    @property
    def effective_video_url(self) -> str | None:
        return (self.video_url or "").strip() or None


# ─── Step metadata ──────────────────────────────────────────────────────────
STEP_META = {
    "search_research": {
        "id": "search_research",
        "label": "Web Research",
        "description": "Searching DuckDuckGo + scraping pages",
        "icon": "🔍",
    },
    "analyze_video": {
        "id": "analyze_video",
        "label": "Video Analysis",
        "description": "Extracting & summarizing YouTube transcript",
        "icon": "🎬",
    },
    "create_report": {
        "id": "create_report",
        "label": "Generating Report",
        "description": "Synthesizing research into a full report",
        "icon": "📄",
    },
    "create_podcast": {
        "id": "create_podcast",
        "label": "Creating Podcast",
        "description": "Writing dialogue + rendering audio with Piper TTS",
        "icon": "🎙️",
    },
}


def sse_event(event: str, data: dict) -> str:
    """Format a single SSE message."""
    payload = json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


async def run_agent_with_streaming(
    topic: str | None, video_url: str | None
) -> AsyncGenerator[str, None]:
    """
    Run the LangGraph agent in a thread, yielding SSE events for:
      - step_start  : node is beginning
      - thinking    : LLM is working (heartbeat)
      - step_done   : node finished
      - result      : final outputs
      - error       : something went wrong
    """
    import threading
    import queue as _queue

    event_queue: _queue.Queue[dict | None] = _queue.Queue()

    # ── Patch the graph nodes to emit events ────────────────────────────────
    import agent.graph as agent_graph
    from langchain_core.runnables import RunnableConfig

    _original_nodes = {
        "search_research": agent_graph.search_research_node,
        "analyze_video": agent_graph.analyze_youtube_video_node,
        "create_report": agent_graph.create_report_node,
        "create_podcast": agent_graph.create_podcast_node,
    }

    def make_wrapped(node_id: str, original_fn):
        def wrapped(state, config: RunnableConfig = None):
            meta = STEP_META[node_id]
            # emit start
            event_queue.put(
                {
                    "type": "step_start",
                    "step": node_id,
                    "label": meta["label"],
                    "description": meta["description"],
                    "icon": meta["icon"],
                }
            )
            try:
                result = original_fn(state, config) if config is not None else original_fn(state, {})
                # emit done with summary snippet
                summary = ""
                for key in ("search_text", "video_text", "synthesis_text", "podcast_script"):
                    val = result.get(key, "") if isinstance(result, dict) else ""
                    if val:
                        summary = val[:200].strip()
                        break
                event_queue.put(
                    {
                        "type": "step_done",
                        "step": node_id,
                        "label": meta["label"],
                        "preview": summary,
                    }
                )
                return result
            except Exception as exc:
                event_queue.put({"type": "error", "step": node_id, "message": str(exc)})
                raise

        return wrapped

    # Patch
    agent_graph.search_research_node = make_wrapped("search_research", _original_nodes["search_research"])
    agent_graph.analyze_youtube_video_node = make_wrapped("analyze_video", _original_nodes["analyze_video"])
    agent_graph.create_report_node = make_wrapped("create_report", _original_nodes["create_report"])
    agent_graph.create_podcast_node = make_wrapped("create_podcast", _original_nodes["create_podcast"])

    result_holder: dict = {}
    error_holder: dict = {}

    def run_graph():
        try:
            # Rebuild the graph so it picks up the patched node functions
            from langgraph.graph import StateGraph, START, END
            from agent.state import ResearchState, ResearchStateInput, ResearchStateOutput
            from agent.configuration import Configuration

            graph = StateGraph(
                ResearchState,
                input=ResearchStateInput,
                output=ResearchStateOutput,
                config_schema=Configuration,
            )
            graph.add_node("search_research", agent_graph.search_research_node)
            graph.add_node("analyze_video", agent_graph.analyze_youtube_video_node)
            graph.add_node("create_report", agent_graph.create_report_node)
            graph.add_node("create_podcast", agent_graph.create_podcast_node)
            graph.add_edge(START, "search_research")
            graph.add_edge("search_research", "analyze_video")
            graph.add_edge("analyze_video", "create_report")
            graph.add_edge("create_report", "create_podcast")
            graph.add_edge("create_podcast", END)

            compiled = graph.compile()
            result = compiled.invoke({"topic": topic, "video_url": video_url})
            result_holder.update(result)
        except Exception as exc:
            error_holder["message"] = str(exc)
            error_holder["traceback"] = traceback.format_exc()
        finally:
            event_queue.put(None)  # sentinel

    thread = threading.Thread(target=run_graph, daemon=True)
    thread.start()

    # Restore originals when done (so re-runs work)
    try:
        heartbeat_counter = 0
        while True:
            try:
                evt = event_queue.get(timeout=0.5)
            except _queue.Empty:
                # Send a heartbeat comment to keep the connection alive
                heartbeat_counter += 1
                yield ": heartbeat\n\n"
                await asyncio.sleep(0)
                continue

            if evt is None:
                break  # sentinel → graph finished

            yield sse_event(evt["type"], evt)
            await asyncio.sleep(0)

        # Yield error or result
        if error_holder:
            yield sse_event("error", {"message": error_holder.get("message", "Unknown error")})
        else:
            # Build audio URL if a wav was created
            podcast_filename = result_holder.get("podcast_filename")
            audio_url = None
            if podcast_filename:
                wav_path = Path(podcast_filename)
                audio_url = f"/audio/{wav_path.name}"

            yield sse_event(
                "result",
                {
                    "report": result_holder.get("report", ""),
                    "podcast_script": result_holder.get("podcast_script", ""),
                    "podcast_filename": podcast_filename,
                    "audio_url": audio_url,
                },
            )
    finally:
        # Restore original functions
        agent_graph.search_research_node = _original_nodes["search_research"]
        agent_graph.analyze_youtube_video_node = _original_nodes["analyze_video"]
        agent_graph.create_report_node = _original_nodes["create_report"]
        agent_graph.create_podcast_node = _original_nodes["create_podcast"]


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.post("/api/research")
async def research(req: ResearchRequest):
    topic = req.effective_topic
    video_url = req.effective_video_url

    # Validate: need at least a topic or a video URL
    if not topic and not video_url:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail="Provide at least a research topic or a YouTube URL.",
        )

    return StreamingResponse(
        run_agent_with_streaming(topic, video_url),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve a generated WAV podcast file."""
    import tempfile
    
    # Security: only allow .wav files from the OS temp directory
    sys_tmp = Path(tempfile.gettempdir())
    wav_path = (sys_tmp / filename).resolve()

    # Ensure it's within the temp directory and is a .wav
    if not wav_path.suffix == ".wav":
        raise HTTPException(status_code=400, detail="Only .wav files are served.")
    if not wav_path.is_relative_to(sys_tmp):
        raise HTTPException(status_code=403, detail="Access denied.")
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")

    return FileResponse(str(wav_path), media_type="audio/wav")


# Mount static frontend (after API routes so /api takes precedence)
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {"error": "frontend/ directory not found"}
