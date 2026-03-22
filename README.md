# Deep research agent

LangGraph workflow that:

1. **Web research** — DuckDuckGo + page excerpts, summarized with **Groq**
2. **Video “analysis”** — YouTube **transcript** + Groq summary
3. **Report** — Groq merges search + video insights
4. **Podcast** — Groq dialogue script + **Piper** (local) text-to-speech to `.wav`

## Setup

1. Python 3.11+
2. Create a virtual environment and install dependencies:

   ```bash
   pip install -e .
   ```

3. Copy `.env.example` to `.env` (or create `.env`) and set:

   - `GROQ_API_KEY` — required for all LLM steps

4. Run locally (one-shot test):

   ```bash
   # from project root, with venv activated
   pip install -e .
   python run_once.py
   ```

   This loads `.env`, runs the full graph (search → video → report → podcast + WAV), and prints the final outputs.

5. Or run the LangGraph dev server with [LangGraph CLI](https://github.com/langchain-ai/langgraph) using `langgraph.json`.

## Optional: Gemini helpers

`agent/utils.py` contains **legacy** Gemini helpers. They are not used by the main graph. To use them, install `google-genai` and set `GEMINI_API_KEY`.

## Security

- Never commit `.env` or API keys. Add `.env` to `.gitignore`.
- If a key was ever shared, rotate it in the provider dashboard.
