"""
test_e2e_xai.py — End-to-end tests against the xAI (Grok) API.

These tests are automatically skipped when XAI_API_KEY is not set.
"""
from __future__ import annotations

import pytest

from edgequake_litellm import completion, acompletion, stream


MODEL = "xai/grok-beta"
MESSAGES = [{"role": "user", "content": "Reply with exactly one word: pong"}]


@pytest.mark.usefixtures("xai_available")
class TestXAICompletion:
    def test_sync_completion(self):
        resp = completion(MODEL, MESSAGES)
        assert isinstance(resp.content, str)
        assert len(resp.content) > 0

    def test_sync_completion_usage(self):
        resp = completion(MODEL, MESSAGES)
        assert resp.usage.total_tokens > 0
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.completion_tokens > 0

    def test_sync_completion_model_name(self):
        resp = completion(MODEL, MESSAGES)
        assert resp.model != ""

    def test_completion_with_system(self):
        msgs = [{"role": "user", "content": "Greet me."}]
        resp = completion(MODEL, msgs, system="You are a helpful assistant.")
        assert isinstance(resp.content, str)
        assert len(resp.content) > 0

    def test_completion_with_max_tokens(self):
        resp = completion(MODEL, MESSAGES, max_tokens=20)
        assert isinstance(resp.content, str)

    def test_completion_with_temperature(self):
        resp = completion(MODEL, MESSAGES, temperature=0.1)
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
        assert len(accumulated) >= 1

    def test_multi_turn(self):
        msgs = [
            {"role": "user", "content": "My favourite colour is blue."},
            {"role": "assistant", "content": "Nice, blue is a calming colour!"},
            {"role": "user", "content": "What is my favourite colour?"},
        ]
        resp = completion(MODEL, msgs)
        assert isinstance(resp.content, str)
        assert "blue" in resp.content.lower()

    async def test_async_multi_turn(self):
        msgs = [
            {"role": "user", "content": "The secret code is ALPHA-7."},
            {"role": "assistant", "content": "Noted: ALPHA-7."},
            {"role": "user", "content": "What is the secret code?"},
        ]
        resp = await acompletion(MODEL, msgs)
        assert "ALPHA-7" in resp.content or "alpha" in resp.content.lower()
