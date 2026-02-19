"""
07_ollama_local.py — Running models locally with Ollama (no API key required).

Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Pull models:
       ollama pull llama3.2
       ollama pull nomic-embed-text

Run:
    python examples/07_ollama_local.py
"""
from __future__ import annotations

import asyncio
import urllib.request

import edgequake_litellm as litellm

OLLAMA_HOST = "http://localhost:11434"
CHAT_MODEL = "ollama/llama3.2"
EMBED_MODEL = "ollama/nomic-embed-text"


def check_ollama() -> bool:
    try:
        urllib.request.urlopen(OLLAMA_HOST, timeout=2)
        return True
    except Exception:
        return False


def sync_chat() -> None:
    print("\n--- Sync chat with Ollama ---")
    messages = [{"role": "user", "content": "What is Rust programming language?"}]
    resp = litellm.completion(CHAT_MODEL, messages, api_base=OLLAMA_HOST)
    print(f"Model  : {resp.model}")
    print(f"Reply  : {resp.content[:200]}")


async def async_chat() -> None:
    print("\n--- Async chat with Ollama ---")
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Show me a simple Fibonacci function in Python."},
    ]
    resp = await litellm.acompletion(CHAT_MODEL, messages, api_base=OLLAMA_HOST)
    print(f"Reply:\n{resp.content}")


async def streaming_chat() -> None:
    print("\n--- Streaming with Ollama ---")
    messages = [{"role": "user", "content": "Name five programming languages and their main use cases."}]
    print("Response: ", end="", flush=True)
    async for chunk in litellm.stream(CHAT_MODEL, messages, api_base=OLLAMA_HOST):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


def embeddings() -> None:
    print("\n--- Embeddings with Ollama ---")
    texts = ["The cat sat on the mat.", "Dogs are loyal animals.", "Python is easy to learn."]
    vecs = litellm.embedding(EMBED_MODEL, texts, api_base=OLLAMA_HOST)
    print(f"Embedded {len(vecs)} texts, dimension: {len(vecs[0])}")


def main() -> None:
    if not check_ollama():
        print(f"Ollama not running at {OLLAMA_HOST}.")
        print("Start it with: ollama serve")
        return

    print(f"Ollama is running at {OLLAMA_HOST}")
    sync_chat()
    asyncio.run(async_chat())
    asyncio.run(streaming_chat())
    embeddings()
    print("\nAll Ollama examples completed.")


if __name__ == "__main__":
    main()
