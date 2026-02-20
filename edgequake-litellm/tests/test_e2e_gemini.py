"""
test_e2e_gemini.py — End-to-end tests against the Google Gemini API.

These tests are automatically skipped when GEMINI_API_KEY is not set.
"""
from __future__ import annotations

import pytest

from edgequake_litellm import completion, acompletion, stream


MODEL = "gemini/gemini-2.0-flash"
MESSAGES = [{"role": "user", "content": "Reply with exactly one word: pong"}]


@pytest.mark.usefixtures("gemini_available")
class TestGeminiCompletion:
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
        msgs = [{"role": "user", "content": "Greet me."}]
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
        text = "".join(c.content or "" for c in chunks)
        assert isinstance(text, str)

    async def test_streaming_finish_reason(self):
        chunks = []
        async for chunk in stream(MODEL, MESSAGES):
            chunks.append(chunk)
        # Last chunk should carry a finish reason
        finish_chunks = [c for c in chunks if c.finish_reason]
        assert len(finish_chunks) >= 1

    def test_json_like_output(self):
        """Ask for structured-ish response and check it is non-empty."""
        msgs = [{"role": "user", "content": 'Return JSON: {"value": 42}'}]
        resp = completion(MODEL, msgs)
        assert "42" in resp.content or len(resp.content) > 0

    async def test_multi_turn(self):
        msgs = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello, Alice!"},
            {"role": "user", "content": "What is my name?"},
        ]
        resp = await acompletion(MODEL, msgs)
        assert isinstance(resp.content, str)
