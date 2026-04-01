# Listen-it-out

An autonomous research system built with **LangGraph** that performs deep-dive investigations by synthesizing web search data and YouTube video insights into comprehensive reports and professional podcast scripts.

The agent operates in a fully autonomous loop, determining when to search the web, when to analyze video transcripts, and how to combine these multi-modal sources into a high-quality final package including an audio podcast generated via local text-to-speech.

## Core Features

- **Autonomous Research Workflow**: Orchestrated by LangGraph to handle search, analysis, and synthesis without manual intervention.
- **Multi-Modal Data Gathering**: 
  - **Web Research**: Parallelized DuckDuckGo searching with automated page excerpt extraction.
  - **Video Analysis**: YouTube transcript extraction with metadata fallback (via `yt-dlp`).
- **High-Quality Synthesis**:
  - **Comprehensive Reports**: Markdown-formatted deep dives optimized for readability and technical depth.
  - **Podcast Generation**: Dynamic, natural-sounding dialogue scripts between two characters (Mike & Dr. Sarah).
- **Local Audio Synthesis**: High-performance, offline text-to-speech using **Piper TTS**.
- **Modern Web Interface**: FastAPI-powered backend with a real-time streaming frontend to visualize the agent's "thinking" process.

## Architecture

The system supports two primary modes:
1.  **StateGraph (Standard)**: A predefined research flow for consistent, structured output.
2.  **ReAct Agent (Autonomous)**: A more flexible agent (in `run_true_agent.py`) that uses tools to gather information and generate reports based on the user's research objective.


```




