import os
from dotenv import load_dotenv
import sys
sys.stdout.reconfigure(encoding='utf-8')

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

load_dotenv()

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq


# Tool 1: Parallel Web Search 
@tool
def search_the_web(queries: str) -> str:
    """Search the internet for information on a topic.
    Input: a comma-separated list of 2 search queries tackling different angles of the topic
    (e.g. "agentic AI overview, agentic AI real world applications").
    All queries are run IN PARALLEL to minimise wait time.
    Returns the top web snippets for all queries combined.
    """
    print(f"\n[Agent Action] 🌐 Parallel web search for: {queries}")
    from duckduckgo_search import DDGS
    from concurrent.futures import ThreadPoolExecutor, as_completed

    query_list = [q.strip() for q in queries.split(",") if q.strip()]
    if not query_list:
        return "No queries provided."

    def _search_one(q: str) -> str:
        try:
            ddgs = DDGS()
            results = list(ddgs.text(q, max_results=3))
            if not results:
                return f"[{q}]: No results found."
            lines = [f"**{r.get('title')}**\nURL: {r.get('href','')}\n{r.get('body','')}" for r in results]
            return f"=== Results for: {q} ===\n" + "\n\n".join(lines)
        except Exception as e:
            return f"[{q}]: Search failed: {e}"

    snippets = []
    with ThreadPoolExecutor(max_workers=len(query_list)) as ex:
        futures = {ex.submit(_search_one, q): q for q in query_list}
        for fut in as_completed(futures):
            snippets.append(fut.result())

    return "\n\n---\n\n".join(snippets)


# ── Tool 2: YouTube Transcript ────────────────────────────────────────────────
@tool
def read_youtube_transcript(url: str) -> str:
    """Fetch a YouTube video's full transcript so you can understand and summarise its content.
    Input should be a valid YouTube URL. Returns the transcript text or video metadata.
    """
    print(f"\n[Agent Action] 🎬 Fetching YouTube transcript for: {url}")
    from urllib.parse import urlparse, parse_qs
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return "youtube_transcript_api is not installed."

    parsed = urlparse(url)
    video_id = None
    if parsed.hostname and "youtube" in parsed.hostname:
        video_id = parse_qs(parsed.query).get("v", [None])[0]
    elif parsed.hostname in {"youtu.be"}:
        video_id = parsed.path.lstrip("/")

    if not video_id:
        return "Error: Could not extract a valid YouTube video ID from the URL."

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = next(iter(transcript_list)).fetch()
        text = " ".join([i['text'] for i in transcript])
        return text[:5000] + ("... (truncated)" if len(text) > 5000 else "")
    except Exception:
        try:
            import yt_dlp
            ydl_opts = {'quiet': True, 'skip_download': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            return (f"Transcript unavailable. Video Metadata:\n"
                    f"Title: {info.get('title')}\n"
                    f"Description: {info.get('description', '')[:800]}")
        except Exception as e:
            return f"Failed entirely: {str(e)}"


# ── Tool 3: Parallel Report + Podcast Generation ──────────────────────────────
@tool
def create_report_and_podcast(topic: str, research_context: str) -> str:
    """Call this tool once ALL research is gathered.
    It SIMULTANEOUSLY generates:
      1. A comprehensive markdown research report
      2. A Mike & Dr. Sarah podcast script
      3. Audio synthesis of the script using Piper TTS
    Both LLM calls run in PARALLEL to cut latency in half.

    Input:
    - topic: The research topic.
    - research_context: All gathered information combined (max 4000 chars).
    Returns a structured result with REPORT and PODCAST sections.
    """
    print(f"\n[Agent Action] 📄🎙️ Generating report + podcast in parallel for: '{topic}'")
    import re, tempfile, wave
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return "Error: Missing GROQ_API_KEY."

    ctx = research_context[:4000]  # keep under token limits

    # ── LLM Prompt: Report ────────────────────────────────────────────────────
    def generate_report() -> str:
        llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0.3)
        prompt = f"""You are a world-class technical journalist writing for MIT Technology Review.
Write the DEFINITIVE research report on: "{topic}"

SOURCE MATERIAL:
{ctx}

Structure (use proper Markdown):
# {topic}: A Deep-Dive Research Report

## Executive Summary
(3 paragraphs. Open with a hook. Define the topic. Why does it matter NOW?)

## Background & Context
(Origins, problem it solves, historical perspective.)

## Key Findings & Analysis
(CORE section. Sub-headings, bullet points, specific facts and numbers from sources.)

## Real-World Applications & Impact
(Concrete use-cases. Industries affected.)

## Challenges & Open Questions
(Downsides, risks, unresolved debates. Be balanced.)

## Future Outlook
(What's next? Bold predictions backed by evidence.)

## Further Reading
(Markdown links for all URLs in the sources: - [Title](URL))

## Conclusion
(1-2 paragraphs. End on a memorable insight.)

Rules: min 600 words. Use **bold** for key terms. Use > blockquotes for stats.
"""
        resp = llm.invoke(prompt)
        return getattr(resp, "content", str(resp)).strip()

    # ── LLM Prompt: Podcast Script ────────────────────────────────────────────
    def generate_script() -> str:
        llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0.5)
        prompt = f"""You are the head writer for a top technology podcast.
Write a BROADCAST-READY podcast episode transcript on: "{topic}"

RESEARCH: {ctx}

CHARACTERS:
- Mike: Charismatic host. Curious, warm, funny. Uses "Wait, hold on", "That's wild!", "So you're telling me..."
- Dr. Sarah: Brilliant expert. Uses analogies. Uses "Here's the thing nobody talks about", "Think of it like...", "And what's counterintuitive is..."

SCRIPT ARC:
1. HOOK (2 lines): Mike opens with a surprising fact or question.
2. EXPLAINER (3-4 lines): Dr. Sarah defines the topic clearly.
3. DEEP DIVE (10-12 lines): Sharp back-and-forth. At least 2 surprising facts woven in.
4. REAL-WORLD (2-3 lines): How this affects the listener listening today.
5. WRAP-UP (2 lines): Memorable closing thought.

FORMAT RULES (strict - audio synthesiser depends on this):
- Every line: "Mike:" or "Dr. Sarah:" prefix. No exceptions.
- 18-24 total lines.
- NO stage directions, markdown, timestamps, or blank lines.
Return ONLY the raw dialogue.
"""
        resp = llm.invoke(prompt)
        return getattr(resp, "content", str(resp)).strip()

    # ── Run both LLM calls in PARALLEL ───────────────────────────────────────
    print("  ⚡ Running report + script generation in parallel...")
    with ThreadPoolExecutor(max_workers=2) as ex:
        report_future = ex.submit(generate_report)
        script_future = ex.submit(generate_script)
        report_text   = report_future.result()
        podcast_script = script_future.result()

    print("  ✅ Both LLM calls done. Starting TTS synthesis...")

    # ── Piper TTS Synthesis ───────────────────────────────────────────────────
    try:
        from piper import PiperVoice
        from piper.download_voices import download_voice
    except ImportError:
        return f"REPORT_TEXT||{report_text}||PODCAST_SCRIPT_ONLY||{podcast_script}"

    safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
    filename = f"research_podcast_{safe_topic.replace(' ', '_')}.wav"

    project_root = Path(__file__).resolve().parent
    voice_dir = (project_root / "piper_voices").resolve()
    voice_dir.mkdir(parents=True, exist_ok=True)

    def ensure_voice(voice_id: str) -> Path:
        onnx_path = voice_dir / f"{voice_id}.onnx"
        if not onnx_path.exists():
            print(f"  Downloading voice: {voice_id}...")
            download_voice(voice_id, voice_dir)
        return onnx_path

    try:
        m_onnx = ensure_voice("en_US-lessac-medium")
        mike_voice = PiperVoice.load(m_onnx, config_path=Path(str(m_onnx) + ".json") if Path(str(m_onnx) + ".json").exists() else None)

        s_onnx = ensure_voice("en_US-amy-medium")
        try:
            sarah_voice = PiperVoice.load(s_onnx, config_path=Path(str(s_onnx) + ".json") if Path(str(s_onnx) + ".json").exists() else None)
        except Exception:
            sarah_voice = mike_voice
    except Exception as e:
        return f"REPORT_TEXT||{report_text}||PODCAST_SCRIPT_ONLY||{podcast_script}||voice_error:{e}"

    line_re = re.compile(r"^\s*(Mike|Dr\.?\s*Sarah)\s*:\s*(.+)$", re.IGNORECASE)
    segments = []
    for raw in podcast_script.splitlines():
        m = line_re.match(raw.strip())
        if not m:
            continue
        speaker = "mike" if m.group(1).lower().startswith("mike") else "sarah"
        text = m.group(2).strip()
        if text:
            segments.append((speaker, text))

    if not segments:
        return f"REPORT_TEXT||{report_text}||PODCAST_SCRIPT_ONLY||{podcast_script}||no_segments"

    sys_tmp = Path(tempfile.gettempdir())
    out_wav = (sys_tmp / filename).resolve()
    tmp_dir = sys_tmp / ".tmp_podcast_audio"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    synth_paths = []
    try:
        for i, (speaker, text) in enumerate(segments, start=1):
            voice = mike_voice if speaker == "mike" else sarah_voice
            seg = tmp_dir / f"seg_{i:04d}.wav"
            with wave.open(str(seg), "wb") as wf:
                voice.synthesize_wav(text, wf)
            synth_paths.append(seg)

        with wave.open(str(synth_paths[0]), "rb") as wf0:
            params = wf0.getparams()
        with wave.open(str(out_wav), "wb") as out_wf:
            out_wf.setparams(params)
            for seg in synth_paths:
                with wave.open(str(seg), "rb") as wf:
                    out_wf.writeframes(wf.readframes(wf.getnframes()))
    except Exception as e:
        return f"REPORT_TEXT||{report_text}||PODCAST_SCRIPT_ONLY||{podcast_script}||tts_error:{e}"
    finally:
        for p in synth_paths:
            try:
                p.unlink()
            except Exception:
                pass

    print(f"  ✅ Podcast audio ready: {out_wav}")
    return f"REPORT_TEXT||{report_text}||PODCAST_CREATED||{str(out_wav)}||{podcast_script}"


# ── CLI Entry Point ───────────────────────────────────────────────────────────
def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Missing GROQ_API_KEY.")
        return

    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key, temperature=0.1)
    tools = [search_the_web, read_youtube_transcript, create_report_and_podcast]
    agent = create_react_agent(llm, tools)

    print("=" * 60)
    print("🤖 STARTING OPTIMISED AUTONOMOUS AGENT")
    print("=" * 60)

    prompt = "Research 'Agentic AI Workflows'. Search the web, then create the report and podcast."
    for s in agent.stream({"messages": [("user", prompt)]}):
        if "agent" in s:
            c = s["agent"]["messages"][-1].content
            if c.strip():
                print(f"[Thinking]: {c[:200]}")

if __name__ == "__main__":
    main()
