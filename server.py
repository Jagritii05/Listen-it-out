"""FastAPI server with SSE streaming for the True Autonomous Agent UI."""

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

from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

app = FastAPI(title="Deep Research Agent - Autonomous Mode")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent / "frontend"

class ResearchRequest(BaseModel):
    topic: str | None = None
    video_url: str | None = None

    @property
    def effective_topic(self) -> str | None:
        return (self.topic or "").strip() or None

    @property
    def effective_video_url(self) -> str | None:
        return (self.video_url or "").strip() or None

def sse_event(event: str, data: dict) -> str:
    payload = json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"

async def run_agent_with_streaming(topic: str | None, video_url: str | None) -> AsyncGenerator[str, None]:
    import threading
    import queue as _queue
    event_queue: _queue.Queue[dict | None] = _queue.Queue()

    result_holder = {}
    error_holder = {}

    def run_graph():
        try:
            from run_true_agent import search_the_web, read_youtube_transcript, create_report_and_podcast
            from langchain_groq import ChatGroq
            from langgraph.prebuilt import create_react_agent
            from langchain_core.messages import HumanMessage, SystemMessage

            api_key = os.getenv("GROQ_API_KEY")
            llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key, temperature=0.1)
            tools = [search_the_web, read_youtube_transcript, create_report_and_podcast]
            
            true_agent = create_react_agent(llm, tools)

            # Build a tightly controlled orchestration prompt
            topic_line = f'Research Topic: "{topic}"' if topic else ""
            video_line = f'YouTube Video URL: {video_url}' if video_url else ""

            prompt = f"""You are an efficient autonomous research agent.

{topic_line}
{video_line}

Execute EXACTLY these steps IN ORDER — 2 tool calls total:

STEP 1 — GATHER (1 tool call):
  - If a YouTube URL is given, call `read_youtube_transcript` FIRST, then proceed.
  - Call `search_the_web` ONCE with a comma-separated string of 2 queries covering different angles of the topic (e.g. "<topic> overview, <topic> latest developments 2024").
  - Both queries run in parallel internally.

STEP 2 — DELIVER (1 tool call):
  - Combine all gathered text into `research_context` (max 3500 chars).
  - Call `create_report_and_podcast` ONCE with the topic and context.
  - This generates the report AND podcast audio simultaneously.

IMPORTANT: Exactly 2 tool calls. No retries. No extra steps."""

            sys_msg = SystemMessage(content="You are a disciplined agent. Call search_the_web exactly once, then create_report_and_podcast exactly once. Never repeat. Never skip.")
            messages = [sys_msg, HumanMessage(content=prompt)]

            event_queue.put({
                "type": "step_start",
                "step": "agent_thinking",
                "label": "Agent Thinking",
                "description": "Analyzing objective and planning steps",
                "icon": "🧠"
            })

            # Stream updates — accumulate all messages so we don't need a second invoke()
            all_messages = list(messages)  # start with the initial system + human messages

            for chunk in true_agent.stream({"messages": messages}, {"recursion_limit": 15}):
                if "agent" in chunk:
                    msg = chunk["agent"]["messages"][-1]
                    all_messages.append(msg)  # collect as we go
                    if msg.tool_calls:
                        for tool in msg.tool_calls:
                            event_queue.put({
                                "type": "step_start",
                                "step": f"tool_{tool['name']}",  # use tool name not id for phase mapping
                                "label": f"Using Tool: {tool['name']}",
                                "description": f"Agent is calling {tool['name']}",
                                "icon": "⚙️"
                            })

                if "tools" in chunk:
                    msg = chunk["tools"]["messages"][-1]
                    all_messages.append(msg)  # collect tool result
                    event_queue.put({
                        "type": "step_done",
                        "step": f"tool_{msg.tool_call_id}",
                        "label": "Tool Finished",
                        "preview": "Obtained results."
                    })

            # No second invoke() needed — all messages were collected during the stream above

            # Build map of tool_call_id -> tool_name from AIMessages
            tool_call_map = {}
            for msg in all_messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_call_map[tc['id']] = tc['name']

            # Extract report and podcast from the combined create_report_and_podcast ToolMessage
            final_text = ""
            podcast_filename = None
            podcast_script = ""
            import tempfile

            for msg in all_messages:
                if type(msg).__name__ == 'ToolMessage':
                    content = getattr(msg, 'content', '') or ''
                    tool_name = tool_call_map.get(getattr(msg, 'tool_call_id', ''), '')

                    # New combined tool format: REPORT_TEXT||<report>||PODCAST_CREATED||<path>||<script>
                    #                        or: REPORT_TEXT||<report>||PODCAST_SCRIPT_ONLY||<script>
                    if 'REPORT_TEXT||' in content:
                        parts = content.split('REPORT_TEXT||', 1)[1]
                        # Extract report
                        if '||PODCAST_CREATED||' in parts:
                            report_part, rest = parts.split('||PODCAST_CREATED||', 1)
                            final_text = report_part.strip()
                            wav_path_and_script = rest.split('||', 1)
                            if len(wav_path_and_script) == 2:
                                wav_path = wav_path_and_script[0].strip()
                                podcast_script = wav_path_and_script[1].strip()
                                podcast_filename = os.path.basename(wav_path)
                                result_holder['podcast_wav_path'] = wav_path
                        elif '||PODCAST_SCRIPT_ONLY||' in parts:
                            report_part, script_part = parts.split('||PODCAST_SCRIPT_ONLY||', 1)
                            final_text = report_part.strip()
                            podcast_script = script_part.split('||')[0].strip()

                    # Legacy single-tool format fallback
                    elif tool_name == 'synthesize_final_report' and content.strip():
                        final_text = content.strip()
                    elif 'PODCAST_CREATED||' in content and not podcast_filename:
                        parts = content.split('PODCAST_CREATED||')[-1].split('||', 1)
                        if len(parts) == 2:
                            wav_path, podcast_script = parts[0].strip(), parts[1].strip()
                            podcast_filename = os.path.basename(wav_path)
                            result_holder['podcast_wav_path'] = wav_path

            # Fall back to last AI message if report tool wasn't called
            if not final_text:
                for msg in reversed(all_messages):
                    if type(msg).__name__ == 'AIMessage':
                        c = (getattr(msg, 'content', '') or '').strip()
                        if c and 'PODCAST_CREATED' not in c and 'REPORT_TEXT' not in c:
                            final_text = c
                            break

            result_holder["report"] = final_text
            result_holder["podcast_filename"] = podcast_filename
            result_holder["podcast_script"] = podcast_script
            
        except Exception as exc:
            error_holder["message"] = str(exc)
            error_holder["traceback"] = traceback.format_exc()
        finally:
            event_queue.put(None) 

    thread = threading.Thread(target=run_graph, daemon=True)
    thread.start()

    try:
        heartbeat_counter = 0
        while True:
            try:
                evt = event_queue.get(timeout=0.5)
            except _queue.Empty:
                heartbeat_counter += 1
                yield ": heartbeat\n\n"
                await asyncio.sleep(0)
                continue

            if evt is None:
                break

            yield sse_event(evt["type"], evt)
            await asyncio.sleep(0)

        if error_holder:
            yield sse_event("error", {"message": error_holder.get("message", "Unknown error")})
        else:
            yield sse_event("result", {
                "report": result_holder.get("report", ""),
                "podcast_script": result_holder.get("podcast_script", ""),
                "podcast_filename": result_holder.get("podcast_filename"),
                "audio_url": f"/podcast/{result_holder['podcast_filename']}" if result_holder.get("podcast_filename") else None,
            })
    finally:
        pass

@app.post("/api/research")
async def research(req: ResearchRequest):
    topic = req.effective_topic
    video_url = req.effective_video_url

    if not topic and not video_url:
        raise HTTPException(status_code=422, detail="Provide at least a research topic or a YouTube URL.")

    return StreamingResponse(
        run_agent_with_streaming(topic, video_url),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/podcast/{filename}")
async def serve_podcast(filename: str):
    """Serve podcast WAV directly from the OS temp directory. No permanent storage needed."""
    import tempfile
    wav_path = Path(tempfile.gettempdir()) / filename
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="Podcast audio not found. It may have expired.")
    return FileResponse(
        path=str(wav_path),
        media_type="audio/wav",
        filename=filename,
    )

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {"error": "frontend/ directory not found"}
