"""
08_multimodal_gemini.py — Multimodal (text + image) example with Google Gemini.

Gemini supports vision inputs using base64-encoded images or public URLs.

Run:
    GEMINI_API_KEY=your_key python examples/08_multimodal_gemini.py
"""
from __future__ import annotations

import asyncio
import os

import edgequake_litellm as litellm

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini/gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Image URL example — use a publicly accessible image
# ---------------------------------------------------------------------------

PUBLIC_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
    "PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
)


def vision_url_example() -> None:
    print("\n--- Multimodal: image URL ---")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": PUBLIC_IMAGE_URL}},
            ],
        }
    ]
    resp = litellm.completion(MODEL, messages)
    print(f"Description: {resp.content}")


async def vision_url_async_example() -> None:
    print("\n--- Multimodal: async image URL ---")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What colours are prominent in this image?"},
                {"type": "image_url", "image_url": {"url": PUBLIC_IMAGE_URL}},
            ],
        }
    ]
    resp = await litellm.acompletion(MODEL, messages)
    print(f"Colours: {resp.content}")


async def vision_streaming_example() -> None:
    print("\n--- Multimodal: streaming vision ---")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "List objects visible in this image."},
                {"type": "image_url", "image_url": {"url": PUBLIC_IMAGE_URL}},
            ],
        }
    ]
    print("Streaming: ", end="", flush=True)
    async for chunk in litellm.stream(MODEL, messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


def main() -> None:
    if not API_KEY:
        print("Set GEMINI_API_KEY to run this example.")
        return

    vision_url_example()
    asyncio.run(vision_url_async_example())
    asyncio.run(vision_streaming_example())
    print("\nAll multimodal examples completed.")


if __name__ == "__main__":
    main()
