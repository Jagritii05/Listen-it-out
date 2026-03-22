"""One-shot run of the research agent graph (for local testing)."""

from __future__ import annotations

import os

# Avoid LangSmith 403 noise (force off even if .env sets tracing)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from dotenv import load_dotenv

load_dotenv()

from agent.graph import create_compiled_graph


def main() -> None:
    graph = create_compiled_graph()
    result = graph.invoke(
        {
            "topic": "What is retrieval-augmented generation?",
            "video_url": None,
        }
    )
    # The graph uses `output=ResearchStateOutput`, so invoke() returns only:
    # report, podcast_script, podcast_filename (not intermediate fields).
    print("--- Result keys ---", list(result.keys()))
    for key in ("report", "podcast_script", "podcast_filename"):
        val = result.get(key)
        if val is None:
            print(f"{key}: <None>")
        elif isinstance(val, str) and len(val) > 800:
            print(f"{key}: <{len(val)} chars>\n{val[:800]}...\n")
        else:
            print(f"{key}:\n{val}\n")


if __name__ == "__main__":
    main()
