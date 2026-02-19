"""
test_e2e_openrouter.py — End-to-end tests against the OpenRouter API.

These tests are automatically skipped when OPENROUTER_API_KEY is not set.

Uses a free model by default to avoid unexpected charges.
"""
from __future__ import annotations

import pytest

from edgequake_litellm import completion, acompletion, stream


# Free tier model on OpenRouter (as of 2025)
MODEL = "openrouter/meta-llama/llama-3.1-8b-instruct:free"
MESSAGES = [{"role": "user", "content": "Reply with exactly one word: pong"}]


@pytest.mark.usefixtures("openrouter_available")
class TestOpenRouterCompletion:
    def test_sync_completion(self):
        resp = completion(MODEL, MESSAGES)
        assert isinstance(resp.content, str)
        assert len(resp.content) > 0

    def test_sync_completion_usage(self):
        resp = completion(MODEL, MESSAGES)
        assert resp.usage.total_tokens > 0

    def test_sync_completion_model_name(self):
        resp = completion(MODEL, MESSAGES)
        assert resp.model != ""

    def test_completion_with_system(self):
        msgs = [{"role": "user", "content": "Who are you?"}]
        resp = completion(MODEL, msgs, system="You are a helpful assistant.")
        assert isinstance(resp.content, str)
        assert len(resp.content) > 0

    def test_completion_with_max_tokens(self):
        resp = completion(MODEL, MESSAGES, max_tokens=20)
        assert isinstance(resp.content, str)

    def test_completion_with_temperature(self):
        resp = completion(MODEL, MESSAGES, temperature=0.2)
        assert isinstance(resp.content, str)

    async def test_async_completion(self):
        resp = await acompletion(MODEL, MESSAGES)
        assert isinstance(resp.content, str)
        assert len(resp.content) > 0

    async def test_async_completion_usage(self):
        resp = await acompletion(MODEL, MESSAGES)
        assert resp.usage.total_tokens > 0

    async def test_streaming(self):
        chunks = []
        async for chunk in stream(MODEL, MESSAGES):
            chunks.append(chunk)
        assert len(chunks) >= 1

    async def test_streaming_accumulates_content(self):
        accumulated = ""
        async for chunk in stream(MODEL, MESSAGES):
            if chunk.content:
                accumulated += chunk.content
        assert isinstance(accumulated, str)

    def test_multi_turn(self):
        msgs = [
            {"role": "user", "content": "My pet is a cat named Whiskers."},
            {"role": "assistant", "content": "Whiskers sounds like a great cat!"},
            {"role": "user", "content": "What is my pet's name?"},
        ]
        resp = completion(MODEL, msgs)
        assert isinstance(resp.content, str)
        assert "Whiskers" in resp.content or "whiskers" in resp.content.lower()

    async def test_async_multi_turn(self):
        msgs = [
            {"role": "user", "content": "The answer to everything is 42."},
            {"role": "assistant", "content": "Noted: 42 is the answer."},
            {"role": "user", "content": "What is the answer to everything?"},
        ]
        resp = await acompletion(MODEL, msgs)
        assert "42" in resp.content
