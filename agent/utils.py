"""Legacy Gemini helpers (optional).

The main LangGraph workflow in ``agent.graph`` uses Groq + Piper and does **not**
import this module. These functions remain for reference or if you reinstall
Gemini support. Importing this module does **not** require ``google-genai`` until
you call a Gemini function.
"""

import os
import wave
from typing import Any, Optional

from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv

load_dotenv()

try:
    from google.genai import Client
    from google.genai import types as genai_types

    _HAS_GENAI = True
except ImportError:
    Client = None  # type: ignore[assignment,misc]
    genai_types = None  # type: ignore[assignment,misc]
    _HAS_GENAI = False

_genai_client: Any = None


def get_genai_client() -> Any:
    """Return a cached Gemini client, or raise if unavailable."""
    global _genai_client
    if not _HAS_GENAI or Client is None:
        raise RuntimeError(
            "Gemini helpers require the optional dependency `google-genai`. "
            "Install it (e.g. `pip install google-genai`) and set GEMINI_API_KEY."
        )
    if _genai_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        _genai_client = Client(api_key=api_key)
    return _genai_client


def display_gemini_response(response: Any) -> tuple[str, str]:
    console = Console()

    text = response.candidates[0].content.parts[0].text
    md = Markdown(text)
    console.print(md)

    candidate = response.candidates[0]

    sources_text = ""

    if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
        console.print("\n" + "=" * 50)
        console.print("[bold blue]References & Sources[/bold blue]")
        console.print("=" * 50)

        if candidate.grounding_metadata.grounding_chunks:
            console.print(
                f"\n[bold]Sources ({len(candidate.grounding_metadata.grounding_chunks)}):[/bold]"
            )
            sources_list = []
            for i, chunk in enumerate(candidate.grounding_metadata.grounding_chunks, 1):
                if hasattr(chunk, "web") and chunk.web:
                    title = getattr(chunk.web, "title", "No title") or "No title"
                    uri = getattr(chunk.web, "uri", "No URI") or "No URI"
                    console.print(f"{i}. {title}")
                    console.print(f"   [dim]{uri}[/dim]")
                    sources_list.append(f"{i}. {title}\n   {uri}")

            sources_text = "\n".join(sources_list)

        if candidate.grounding_metadata.grounding_supports:
            console.print(f"\n[bold]Text segments with source backing:[/bold]")
            for support in candidate.grounding_metadata.grounding_supports[:5]:  # Show first 5
                if hasattr(support, "segment") and support.segment:
                    snippet = (
                        support.segment.text[:100] + "..."
                        if len(support.segment.text) > 100
                        else support.segment.text
                    )
                    source_nums = [str(i + 1) for i in support.grounding_chunk_indices]
                    console.print(
                        f'• "{snippet}" [dim](sources: {", ".join(source_nums)})[/dim]'
                    )

    return text, sources_text


def wave_file(
    filename: str,
    pcm: bytes,
    channels: int = 1,
    rate: int = 24000,
    sample_width: int = 2,
) -> None:
    """Save PCM data to a wave file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def create_podcast_discussion(
    topic: str,
    search_text: str,
    video_text: str,
    search_sources_text: str,
    video_url: str,
    filename: str = "research_podcast.wav",
    configuration: Optional[Any] = None,
) -> tuple[str, str]:
    import tempfile
    from pathlib import Path
    
    # Use the OS temp directory
    sys_tmp = Path(tempfile.gettempdir())
    out_wav_path = str((sys_tmp / filename).resolve())
    """Create a 2-speaker podcast discussion explaining the research topic (Gemini)."""
    if genai_types is None:
        raise RuntimeError("google.genai types are not available (install google-genai).")

    if configuration is None:
        from agent.configuration import Configuration

        configuration = Configuration()

    genai_client = get_genai_client()

    script_prompt = f"""
    Create a natural, engaging podcast conversation between Dr. Sarah (research expert) and Mike (curious interviewer) about "{topic}".

    Use this research content:

    SEARCH FINDINGS:
    {search_text}

    VIDEO INSIGHTS:
    {video_text}

    Format as a dialogue with:
    - Mike introducing the topic and asking questions
    - Dr. Sarah explaining key concepts and insights
    - Natural back-and-forth discussion (5-7 exchanges)
    - Mike asking follow-up questions
    - Dr. Sarah synthesizing the main takeaways
    - Keep it conversational and accessible (3-4 minutes when spoken)

    Format exactly like this:
    Mike: [opening question]
    Dr. Sarah: [expert response]
    Mike: [follow-up]
    Dr. Sarah: [explanation]
    [continue...]
    """

    script_response = genai_client.models.generate_content(
        model=configuration.synthesis_model,
        contents=script_prompt,
        config={"temperature": configuration.podcast_temperature},
    )

    podcast_script = script_response.candidates[0].content.parts[0].text

    tts_prompt = f"TTS the following conversation between Mike and Dr. Sarah:\n{podcast_script}"

    response = genai_client.models.generate_content(
        model=configuration.tts_model,
        contents=tts_prompt,
        config=genai_types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=genai_types.SpeechConfig(
                multi_speaker_voice_config=genai_types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        genai_types.SpeakerVoiceConfig(
                            speaker="Mike",
                            voice_config=genai_types.VoiceConfig(
                                prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                                    voice_name=configuration.mike_voice,
                                )
                            ),
                        ),
                        genai_types.SpeakerVoiceConfig(
                            speaker="Dr. Sarah",
                            voice_config=genai_types.VoiceConfig(
                                prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                                    voice_name=configuration.sarah_voice,
                                )
                            ),
                        ),
                    ]
                )
            ),
        ),
    )

    audio_data = response.candidates[0].content.parts[0].inline_data.data
    wave_file(
        out_wav_path,
        audio_data,
        configuration.tts_channels,
        configuration.tts_rate,
        configuration.tts_sample_width,
    )

    print(f"Podcast saved as: {out_wav_path}")
    return podcast_script, out_wav_path


def create_research_report(
    topic: str,
    search_text: str,
    video_text: str,
    search_sources_text: str,
    video_url: str,
    configuration: Optional[Any] = None,
) -> tuple[str, str]:
    """Create a comprehensive research report by synthesizing search and video content (Gemini)."""
    if configuration is None:
        from agent.configuration import Configuration

        configuration = Configuration()

    genai_client = get_genai_client()

    synthesis_prompt = f"""
    You are a research analyst. I have gathered information about "{topic}" from two sources:

    SEARCH RESULTS:
    {search_text}

    VIDEO CONTENT:
    {video_text}

    Please create a comprehensive synthesis that:
    1. Identifies key themes and insights from both sources
    2. Highlights any complementary or contrasting perspectives
    3. Provides an overall analysis of the topic based on this multi-modal research
    4. Keep it concise but thorough (3-4 paragraphs)

    Focus on creating a coherent narrative that brings together the best insights from both sources.
    """

    synthesis_response = genai_client.models.generate_content(
        model=configuration.synthesis_model,
        contents=synthesis_prompt,
        config={
            "temperature": configuration.synthesis_temperature,
        },
    )

    synthesis_text = synthesis_response.candidates[0].content.parts[0].text

    report = f"""# Research Report: {topic}

## Executive Summary

{synthesis_text}

## Video Source
- **URL**: {video_url}

## Additional Sources
{search_sources_text}

---
*Report generated using multi-modal AI research combining web search and video analysis*
"""

    return report, synthesis_text
