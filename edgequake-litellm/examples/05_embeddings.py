"""
05_embeddings.py — Text embedding example with semantic search.

Run:
    python examples/05_embeddings.py
"""
from __future__ import annotations

import asyncio
import math
import os

import edgequake_litellm as litellm


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def sync_demo(model: str) -> None:
    print(f"\n--- Sync embedding ({model}) ---")

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps above an idle canine.",     # Similar
        "Python is a popular programming language.",          # Different
        "Machine learning is a subset of artificial intelligence.",  # Different
    ]

    vecs = litellm.embedding(model, texts)
    print(f"Embedded {len(vecs)} texts, dimension: {len(vecs[0])}")

    query = vecs[0]
    print(f"\nSimilarity to '{texts[0][:40]}...':")
    for i, (text, vec) in enumerate(zip(texts[1:], vecs[1:]), start=1):
        sim = cosine_similarity(query, vec)
        print(f"  [{i}] {sim:.4f}  {text[:60]}")


async def async_demo(model: str) -> None:
    print(f"\n--- Async embedding ({model}) ---")

    texts = ["hello world", "goodbye world", "hello Python"]
    vecs = await litellm.aembedding(model, texts)
    print(f"Embedded {len(vecs)} texts asynchronously.")

    sim_01 = cosine_similarity(vecs[0], vecs[1])
    sim_02 = cosine_similarity(vecs[0], vecs[2])
    print(f"  'hello world' ↔ 'goodbye world': {sim_01:.4f}")
    print(f"  'hello world' ↔ 'hello Python' : {sim_02:.4f}")


def main() -> None:
    # OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        sync_demo("openai/text-embedding-3-small")
        asyncio.run(async_demo("openai/text-embedding-3-small"))

    # Ollama (local)
    elif os.environ.get("OLLAMA_HOST") or True:
        import urllib.request
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        try:
            urllib.request.urlopen(host, timeout=1)
            sync_demo("ollama/nomic-embed-text")
            asyncio.run(async_demo("ollama/nomic-embed-text"))
        except Exception:
            print("No API key found. Set OPENAI_API_KEY, or run Ollama locally.")


if __name__ == "__main__":
    main()
