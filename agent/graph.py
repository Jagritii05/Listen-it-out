from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
import os

from dotenv import load_dotenv

load_dotenv()

from agent.state import ResearchState, ResearchStateInput, ResearchStateOutput
from agent.configuration import Configuration
from langsmith import traceable

@traceable(run_type="llm", name="Web Research", project_name="Multimodal Researcher")
def search_research_node(state: ResearchState, config:RunnableConfig) -> dict:
    """
    Gemini-free web research:
    - Use DuckDuckGo to gather candidate sources
    - Optionally fetch top pages and extract text excerpts
    - Summarize with Groq (LLM)
    - If no topic is set yet (URL-only mode), skip and let the video node derive it.
    """
    configuration = Configuration.from_runnable_config(config)
    topic = state.get("topic") or ""

    # URL-only mode: no topic yet — skip web search, let video node derive topic.
    if not topic.strip():
        return {
            "search_text": "",
            "search_sources_text": "",
        }

    # Imports are local so the project can load without Gemini deps.
    from duckduckgo_search import DDGS
    import requests
    from bs4 import BeautifulSoup

    from langchain_groq import ChatGroq

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable.")

    llm = ChatGroq(
        model=configuration.groq_search_model,
        groq_api_key=groq_api_key,
        temperature=configuration.search_temperature,
    )

    ddgs = DDGS()
    results = list(
        ddgs.text(topic, max_results=configuration.web_search_max_results)
    )

    # Defensive defaults for varying duckduckgo_search output schemas.
    sources = []
    snippet_blocks = []
    for idx, r in enumerate(results, start=1):
        title = (r.get("title") or "").strip() or "Untitled"
        url = (r.get("href") or r.get("url") or "").strip()
        snippet = (r.get("body") or r.get("snippet") or r.get("description") or "").strip()

        snippet_short = " ".join(snippet.split())[: configuration.web_snippet_char_limit]
        sources.append((idx, title, url))
        snippet_blocks.append(
            f"[{idx}] {title}\nURL: {url}\nSnippet: {snippet_short}"
        )

    # Optionally fetch page excerpts for better summarization.
    page_blocks: list[str] = []
    for idx, title, url in sources[: configuration.web_fetch_top_pages]:
        if not url:
            continue
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": configuration.web_user_agent},
                timeout=configuration.web_http_timeout_s,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(" ", strip=True)
            text = " ".join(text.split())
            text_short = text[: configuration.web_page_char_limit]

            if text_short:
                page_blocks.append(
                    f"[{idx}] {title}\nURL: {url}\nExcerpt: {text_short}"
                )
        except Exception:
            # If fetching fails (blocked sites, network errors), fall back to snippets.
            continue

    used_blocks = page_blocks if page_blocks else snippet_blocks
    sources_text = "\n".join([f"{i}. {t}\n   {u}" for i, t, u in sources])

    prompt = f"""
You are a research assistant. Use the provided sources excerpts to write:
1) A concise overview of: {topic}
2) Key takeaways (bullet list)
3) Reference supporting details using source numbers like [1], [2], ...

Return ONLY the overview + key takeaways. Do not include the sources section.

Sources excerpts:
{chr(10).join(used_blocks)}
"""

    llm_response = llm.invoke(prompt)
    search_text = getattr(llm_response, "content", str(llm_response))

    return {
        "search_text": search_text,
        "search_sources_text": sources_text,
    }


@traceable(run_type="llm", name="Youtube Video Analysis", project_name="Multimodal Researcher")
def analyze_youtube_video_node(state: ResearchState, config: RunnableConfig) -> dict:
    """
    Gemini-free "video analysis":
    - Extract YouTube transcript text
    - If no topic was provided, derive one from the transcript (URL-only mode)
    - Send transcript to Groq to analyze for the given topic
    - Return `video_text` (and optionally a derived `topic`)
    """
    configuration = Configuration.from_runnable_config(config)
    video_url = state.get("video_url")
    topic = state.get("topic") or ""

    if not video_url:
        return {"video_text": ""}

    from urllib.parse import urlparse, parse_qs
    import re

    def extract_youtube_video_id(url: str) -> str | None:
        parsed = urlparse(url)
        if parsed.hostname in {"youtu.be"}:
            return parsed.path.lstrip("/") or None
        if parsed.hostname and "youtube" in parsed.hostname:
            qs = parse_qs(parsed.query)
            vid = qs.get("v", [None])[0]
            if vid:
                return vid
            # /shorts/<id> or /embed/<id>
            m = re.search(r"/(shorts|embed)/([^/?#&]+)", parsed.path)
            if m:
                return m.group(2)
        return None

    video_id = extract_youtube_video_id(video_url)
    if not video_id:
        return {"video_text": "Video transcript unavailable (could not extract YouTube video id)."}

    # Transcript-first approach with yt-dlp metadata fallback
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            try:
                # Try getting the explicitly requested languages
                transcript_obj = transcript_list.find_transcript(list(configuration.video_transcript_languages))
            except Exception:
                # Smart fallback: grab the first available transcript and translate it to English ('en')
                transcript_obj = next(iter(transcript_list)).translate('en')
            
            transcript_items = transcript_obj.fetch()

            # Build a timestamped, size-limited excerpt for the LLM.
            excerpt_parts: list[str] = []
            total_chars = 0
            for entry in transcript_items:
                start = float(entry.get("start", 0.0))
                text = (entry.get("text") or "").strip()
                if not text:
                    continue

                ts = f"{start:.1f}s"
                block = f"[{ts}] {text}"
                if total_chars + len(block) > configuration.video_transcript_char_limit:
                    break
                excerpt_parts.append(block)
                total_chars += len(block) + 1

            transcript_excerpt = "\n".join(excerpt_parts).strip()
            if not transcript_excerpt:
                raise RuntimeError("Empty transcript excerpt")
                
        except Exception:
            # Step 2: Safety net fallback using yt-dlp to scrape metadata
            import yt_dlp
            ydl_opts = {
                'quiet': True,
                'skip_download': True,
                'no_warnings': True,
                'extract_flat': False
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                
            title = info.get('title', 'Unknown Title')
            description = info.get('description', 'No description available')
            
            # tags might be None
            raw_tags = info.get('tags', [])
            tags = ", ".join(raw_tags) if raw_tags else "None"
            
            transcript_excerpt = f"[YOUTUBE VIDEO METADATA FALLBACK]\nTitle: {title}\nTags: {tags}\n\nDescription:\n{description}"
            transcript_excerpt = transcript_excerpt[:configuration.video_transcript_char_limit].strip()
            
            if not transcript_excerpt:
                raise RuntimeError("Empty metadata fallback")
                
    except Exception as e:
        return {"video_text": f"Video transcript unavailable and metadata fallback failed: {str(e)}"}

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable.")

    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model=configuration.groq_video_analysis_model,
        groq_api_key=groq_api_key,
        temperature=configuration.synthesis_temperature,
    )

    # ── URL-only mode: derive topic from transcript ──────────────────────
    derived_topic = None
    if not topic.strip():
        topic_prompt = f"""
Read this YouTube video transcript excerpt and return ONLY a concise topic title
(5-10 words max) that best describes what the video is about. No explanation, just the title.

Transcript excerpt:
{transcript_excerpt[:3000]}
"""
        topic_response = llm.invoke(topic_prompt)
        derived_topic = getattr(topic_response, "content", str(topic_response)).strip().strip('"').strip("'")
        topic = derived_topic

    prompt = f"""
You are a brilliant multimedia analyst. Your task is to extract high-value information from the provided YouTube video transcript excerpt, focusing specifically on: "{topic}".

TRANSCRIPT EXCERPT (timestamped):
{transcript_excerpt}

Task Requirements:
1. NARRATIVE OVERVIEW: Provide a highly engaging, 1-2 paragraph summary of the video's core message regarding the topic.
2. CRITICAL INSIGHTS: Extract 3-5 high-impact bullet points detailing the most important arguments or facts revealed.
3. NUANCES & CAVEATS: Identify any contradictions, biases, uncertainties, or important caveats mentioned or implied in the video.
4. KEY TIMESTAMPS: Provide a structured timeline (3-6 items) referencing approximate timestamps (e.g., [12.3s] - Speaker argues X).

Format your response cleanly. Return ONLY the requested insights without any meta-labels like "Item 1" or conversational filler.
"""

    llm_response = llm.invoke(prompt)
    video_text = getattr(llm_response, "content", str(llm_response))

    result: dict = {"video_text": video_text}
    # Push derived topic back into state so downstream nodes can use it
    if derived_topic:
        result["topic"] = derived_topic
    return result


@traceable(run_type="llm", name="Create Report", project_name="Multimodal Researcher")
def create_report_node(state: ResearchState, config: RunnableConfig) -> dict:
    configuration= Configuration.from_runnable_config(config)
    topic= state.get("topic") or "the researched topic"
    search_text= state.get("search_text", "")
    video_text= state.get("video_text","")
    search_sources_text= state.get("search_sources_text","")
    video_url= state.get("video_url", "")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable.")

    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model=configuration.groq_video_analysis_model,
        groq_api_key=groq_api_key,
        temperature=configuration.synthesis_temperature,
    )

    synthesis_prompt = f"""
You are a senior research analyst and master storyteller. Your task is to weave together web search findings and video insights into a CAPTIVATING, highly readable, and thoroughly structured master report on the topic: "{topic}".

SEARCH FINDINGS:
{search_text}

VIDEO INSIGHTS:
{video_text}

Formatting and Style Requirements:
1. STRUCTURE: Do not just write a boring wall of text. Use rich Markdown formatting! Include engaging section headers (H3/H4), bold key terms, blockquotes for profound insights, and bulleted lists where appropriate.
2. NARRATIVE FLOW: Start with a powerful hook. Weave the search findings and video insights together seamlessly. Compare and contrast different perspectives cleanly.
3. DEPTH & ANALYSIS: Identify the overarching themes, hidden nuances, and future implications. Provide a critical, high-level analysis that feels like a premium newsletter or expert briefing.
4. READABILITY: Make it substantial and comprehensive, but break it up visually so it is effortless to read.

Return ONLY the raw Markdown synthesis text. Do not include a main Title header (H1) as that will be added automatically.
"""

    llm_response = llm.invoke(synthesis_prompt)
    synthesis_text = getattr(llm_response, "content", str(llm_response)).strip()

    report = f"""# Research Report: {topic}

## Executive Summary

{synthesis_text}

## Video Source
- **URL**: {video_url}
- **Analysis**: Transcript-based summarization and synthesis

## Additional Sources
{search_sources_text}

---
*Report generated using Groq-based multimodal research (transcript + web sources).*
"""

    return {
        "report": report,
        "synthesis_text": synthesis_text,
    }

@traceable(run_type="llm", name="Create Podcast", project_name="multi-modal-researcher")
def create_podcast_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that creates a podcast discussion"""
    configuration = Configuration.from_runnable_config(config)
    topic = state.get("topic") or "the researched topic"
    search_text = state.get("search_text", "")
    video_text = state.get("video_text", "")
    search_sources_text = state.get("search_sources_text", "")
    video_url = state.get("video_url", "")
    
    safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
    filename = f"research_podcast_{safe_topic.replace(' ', '_')}.wav"
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable.")

    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model=configuration.groq_video_analysis_model,
        groq_api_key=groq_api_key,
        temperature=configuration.podcast_temperature,
    )

    podcast_prompt = f"""
You are a master scriptwriter for a wildly popular, highly engaging technology and science podcast. 
Create an incredibly natural, dynamic, and fascinating podcast conversation between:
- Mike (the host: curious, enthusiastic, asks great follow-up questions, occasionally mind-blown)
- Dr. Sarah (the resident expert: brilliant, articulate, uses great analogies to explain complex topics)

Topic of the day: "{topic}"

Reference Material to weave into the conversation naturally:
SEARCH FINDINGS:
{search_text}
VIDEO INSIGHTS:
{video_text}
OPTIONAL CONTEXT:
{search_sources_text}

Strict Requirements:
- DIALOGUE RULES: Use EXACT speaker labels: "Mike:" and "Dr. Sarah:". No other text.
- LENGTH: Write 8-10 substantial exchanges (16-20 total lines). Aim for a 3-4 minute listen.
- TONE & DELIVERY: Super conversational. Use natural filler words ("Wow", "Right", "Exactly!", "Wait, so..."). Have them react dynamically to each other, use humor where appropriate, and express genuine amazement.
- SUBSTANCE: Dr. Sarah should break down the complex facts from the research into easily digestible gems. Mike should guide the audience by asking the exact burning questions a curious listener would have.
- FLOW: Give the script a hook at the beginning and a satisfying wrap-up at the end.

Return ONLY the raw dialogue lines. Do not include stage directions, markdown, or introductions.
"""

    llm_response = llm.invoke(podcast_prompt)
    podcast_script = getattr(llm_response, "content", str(llm_response)).strip()

    # --- TTS / podcast audio (Piper, local/offline) ---
    from pathlib import Path
    import re
    import tempfile
    import wave

    from piper import PiperVoice
    from piper.download_voices import download_voice

    project_root = Path(__file__).resolve().parents[1]
    voice_dir = (project_root / configuration.piper_voice_dir).resolve()
    voice_dir.mkdir(parents=True, exist_ok=True)

    def ensure_voice(voice_id: str) -> Path:
        onnx_path = voice_dir / f"{voice_id}.onnx"
        if not onnx_path.exists():
            download_voice(voice_id, voice_dir)
        return onnx_path

    def load_piper_voice(voice_id: str) -> PiperVoice:
        onnx_path = ensure_voice(voice_id)
        config_path = Path(str(onnx_path) + ".json")  # e.g. `en_US-lessac-medium.onnx.json`
        if not config_path.exists():
            config_path = None
        return PiperVoice.load(onnx_path, config_path=config_path)

    mike_voice = load_piper_voice(configuration.piper_mike_voice_id)
    if configuration.piper_single_voice:
        sarah_voice = mike_voice
    else:
        # If the Sarah voice can't be downloaded/loaded, fall back to Mike.
        try:
            sarah_voice = load_piper_voice(configuration.piper_sarah_voice_id)
        except Exception:
            sarah_voice = mike_voice

    # Parse dialogue lines: `Mike: ...` or `Dr. Sarah: ...`
    # Allow optional leading numbering/bullets before the required speaker label.
    line_re = re.compile(
        r"^\s*(?:[-*]|[\d]+\s*[.)])?\s*(Mike|Dr\.?\s*Sarah)\s*:\s*(.*)$",
        re.IGNORECASE,
    )
    segments: list[tuple[str, str]] = []
    for raw in podcast_script.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = line_re.match(line)
        if not m:
            # If the model didn't follow the exact format, ignore the line.
            continue
        speaker_raw = m.group(1).strip()
        text = m.group(2).strip()
        if not text:
            continue
        speaker_key = "mike" if speaker_raw.lower().startswith("mike") else "sarah"
        segments.append((speaker_key, text))

    # Synthesize segments and concatenate WAVs.
    # Use the OS temp directory so the project folder stays clean.
    sys_tmp = Path(tempfile.gettempdir())
    out_wav_path = (sys_tmp / filename).resolve()
    tmp_dir = sys_tmp / ".tmp_podcast_audio"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if not segments:
        # If parsing failed, return script but no audio.
        return {
            "podcast_script": podcast_script,
            "podcast_filename": None,
        }

    synth_paths: list[Path] = []
    try:
        for i, (speaker_key, text) in enumerate(segments, start=1):
            voice = mike_voice if speaker_key == "mike" else sarah_voice
            seg_path = tmp_dir / f"seg_{i:04d}.wav"
            # Write each segment with Piper.
            # wave.open on Windows/Python 3.11 expects str path, not pathlib.Path
            with wave.open(str(seg_path), "wb") as wf:
                voice.synthesize_wav(text, wf)
            synth_paths.append(seg_path)

        # Concatenate WAVs (assumes consistent format across Piper voices).
        with wave.open(str(synth_paths[0]), "rb") as wf0:
            first_params = wf0.getparams()

        with wave.open(str(out_wav_path), "wb") as out_wf:
            out_wf.setparams(first_params)
            for seg_path in synth_paths:
                with wave.open(str(seg_path), "rb") as wf:
                    out_wf.writeframes(wf.readframes(wf.getnframes()))
    finally:
        # Best-effort cleanup of temporary segment files.
        for p in synth_paths:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    podcast_filename = str(out_wav_path)
    
    return {
        "podcast_script": podcast_script,
        "podcast_filename": podcast_filename
    }

def create_research_graph() -> StateGraph:
    graph= StateGraph(
        ResearchState,
        input= ResearchStateInput,
        output= ResearchStateOutput,
        config_schema= Configuration,
    )

    graph.add_node("search_research", search_research_node)
    graph.add_node("analyze_video", analyze_youtube_video_node)
    graph.add_node("create_report", create_report_node)
    graph.add_node("create_podcast", create_podcast_node)

    graph.add_edge(START, "search_research")
    graph.add_edge("search_research", "analyze_video")
    graph.add_edge("analyze_video", "create_report")
    graph.add_edge("create_report", "create_podcast")
    graph.add_edge("create_podcast", END)

    return graph

def create_compiled_graph():
    graph= create_research_graph()
    return graph.compile()