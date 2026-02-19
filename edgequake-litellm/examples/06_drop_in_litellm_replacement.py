"""
06_drop_in_litellm_replacement.py — Using edgequake_litellm as a drop-in
replacement for the litellm package.

Existing code that uses `litellm` can be migrated with a single line change:

    Before:  import litellm
    After:   import edgequake_litellm as litellm

Everything else stays the same.

Run:
    python examples/06_drop_in_litellm_replacement.py
"""
from __future__ import annotations

import asyncio
import os

# ─── Drop-in replacement ───────────────────────────────────────────────────
#
#   Before: import litellm
#   After:
import edgequake_litellm as litellm
# ───────────────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# 1. Sync completion — same API as litellm.completion()
# ---------------------------------------------------------------------------
def sync_example() -> None:
    print("\n--- 1. Sync completion ---")
    response = litellm.completion(
        "openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Explain REST in one sentence."}],
        max_tokens=60,
        temperature=0.3,
    )
    print(f"Content : {response.content}")
    print(f"Usage   : {response.usage.total_tokens} tokens")


# ---------------------------------------------------------------------------
# 2. Async completion — same API as await litellm.acompletion()
# ---------------------------------------------------------------------------
async def async_example() -> None:
    print("\n--- 2. Async completion ---")
    response = await litellm.acompletion(
        "openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "What is asyncio?"}],
        max_tokens=80,
    )
    print(f"Content : {response.content}")


# ---------------------------------------------------------------------------
# 3. Streaming — same API as async for chunk in litellm.stream()
# ---------------------------------------------------------------------------
async def streaming_example() -> None:
    print("\n--- 3. Streaming ---")
    print("Response: ", end="", flush=True)
    async for chunk in litellm.stream(
        "openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
    ):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# 4. Embeddings — same API as litellm.embedding()
# ---------------------------------------------------------------------------
def embedding_example() -> None:
    print("\n--- 4. Embeddings ---")
    vecs = litellm.embedding(
        "openai/text-embedding-3-small",
        ["Hello, world!", "Bonjour le monde!"],
    )
    print(f"Vectors : {len(vecs)} × dim {len(vecs[0])}")


# ---------------------------------------------------------------------------
# 5. Exception handling — compatible with litellm exception hierarchy
# ---------------------------------------------------------------------------
def exception_example() -> None:
    print("\n--- 5. Exception handling ---")
    from edgequake_litellm.exceptions import AuthenticationError
    try:
        # Use a deliberately bad key
        old_key = os.environ.get("OPENAI_API_KEY", "")
        os.environ["OPENAI_API_KEY"] = "bad-key"
        litellm.completion(
            "openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Test"}],
        )
    except AuthenticationError as e:
        print(f"Caught AuthenticationError: {e}")
    except Exception as e:
        print(f"Caught {type(e).__name__}: {e}")
    finally:
        os.environ["OPENAI_API_KEY"] = old_key


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sync_example()
    asyncio.run(async_example())
    asyncio.run(streaming_example())
    embedding_example()
    exception_example()
    print("\nAll examples completed.")
