"""
02_async_completion.py — Asynchronous completion using asyncio.

Run:
    python examples/02_async_completion.py
"""
from __future__ import annotations

import asyncio
import os

import edgequake_litellm as litellm


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY to run this example.")
        return

    model = "openai/gpt-4o-mini"
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Give me three fun facts about the Moon."},
    ]

    print("Sending async request...")
    resp = await litellm.acompletion(model, messages)

    print(f"\nModel  : {resp.model}")
    print(f"Reply  :\n{resp.content}")
    print(f"\nTokens : {resp.usage.total_tokens}")


# ---------------------------------------------------------------------------
# Run multiple requests concurrently
# ---------------------------------------------------------------------------

async def concurrent_demo() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return

    model = "openai/gpt-4o-mini"
    questions = [
        "What is the speed of light?",
        "Who invented the telephone?",
        "What is the largest planet?",
    ]

    print("\n--- Concurrent requests ---")
    tasks = [
        litellm.acompletion(model, [{"role": "user", "content": q}])
        for q in questions
    ]
    results = await asyncio.gather(*tasks)

    for question, resp in zip(questions, results):
        print(f"\nQ: {question}")
        print(f"A: {resp.content[:120]}...")


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(concurrent_demo())
