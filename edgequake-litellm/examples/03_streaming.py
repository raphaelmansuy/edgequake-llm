"""
03_streaming.py — Real-time streaming responses.

Run:
    python examples/03_streaming.py
"""
from __future__ import annotations

import asyncio
import os
import sys

import edgequake_litellm as litellm


async def stream_to_console(model: str, messages: list) -> None:
    """Stream a response and print chunks as they arrive."""
    print(f"\nStreaming from {model}:\n")
    total_chars = 0

    async for chunk in litellm.stream(model, messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            total_chars += len(chunk.content)
        if chunk.finish_reason:
            print(f"\n\n[finished — {total_chars} chars, reason: {chunk.finish_reason}]")


async def main() -> None:
    # Try providers in order of what's available
    providers = [
        ("OPENAI_API_KEY",     "openai/gpt-4o-mini"),
        ("ANTHROPIC_API_KEY",  "anthropic/claude-3-haiku-20240307"),
        ("GEMINI_API_KEY",     "gemini/gemini-2.0-flash"),
        ("MISTRAL_API_KEY",    "mistral/mistral-small-latest"),
    ]

    model_to_use = None
    for env_var, model in providers:
        if os.environ.get(env_var):
            model_to_use = model
            break

    if model_to_use is None:
        print("No API key found. Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, "
              "GEMINI_API_KEY, or MISTRAL_API_KEY.")
        return

    messages = [
        {"role": "user",
         "content": "Write a short poem (4 lines) about open-source software."},
    ]

    await stream_to_console(model_to_use, messages)


if __name__ == "__main__":
    asyncio.run(main())
