import os
from dataclasses import dataclass, fields
from typing import Optional, Any
from langchain_core.runnables import RunnableConfig

@dataclass (kw_only = True)
class Configuration :
    search_model : str= "gemini-2.5-flash"
    synthesis_model: str= "gemini-2.5-flash"
    video_model: str= "gemini-2.5-flash"
    tts_model: str= "gemini-2.5-flash-preview-tts"

    search_temperature: float= 0.0
    synthesis_temperature: float= 0.3
    podcast_temperature: float= 0.4


    mike_voice= str="Kore"
    sarah_voice= str="Puck"
    tts_channels: int= 1
    tts_rate= int= 24000
    tts_sample_width: int=2

    # --- Groq (Gemini-free text) ---
    # Used by the web research node to summarize DuckDuckGo results.
    groq_search_model: str = "llama-3.3-70b-versatile"
    groq_video_analysis_model: str = "llama-3.3-70b-versatile"
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # --- Web search settings ---
    web_search_max_results: int = 5
    web_fetch_top_pages: int = 2  # 0 = summarize only snippets
    web_page_char_limit: int = 4000
    web_snippet_char_limit: int = 600
    web_http_timeout_s: int = 15
    web_user_agent: str = "Mozilla/5.0 (compatible; DeepResearchBot/1.0; +https://example.com/bot)"

    # --- YouTube transcript-based "video analysis" ---
    video_transcript_char_limit: int = 6000
    video_transcript_languages: tuple[str, ...] = ("en", "en-US")
    video_frame_fallback_enabled: bool = False
    video_frame_count: int = 4

    # --- Piper TTS (local/offline) ---
    # These voice identifiers are downloaded as `<voice_id>.onnx` into piper_voice_dir.
    piper_voice_dir: str = "piper_voices"
    piper_mike_voice_id: str = "en_US-lessac-medium"
    piper_sarah_voice_id: str = "en_US-amy-medium"
    piper_single_voice: bool = False

    @classmethod 
    def from_runnable_config(
        cls, config: Optional[RunnableConfig]= None
    ) -> "Configuration":
         configurable= (
            config["configurable"] if config and "configurable" in config else {}
         )
         values: dict[str,Any]={
            f.name:os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
         }

         # Keep falsy-but-valid values (e.g. 0) by filtering only on None.
         return cls(**{k: v for k, v in values.items() if v is not None})

