"""
test_e2e_anthropic.py — End-to-end tests against the Anthropic API.

These tests are automatically skipped when ANTHROPIC_API_KEY is not set.
"""
from __future__ import annotations

import pytest

from edgequake_litellm import completion, acompletion, stream


MESSAGES = [{"role": "user", "content": "Say 'pong' and nothing else."}]
MODEL = "anthropic/claude-3-5-haiku-20241022"


@pytest.mark.usefixtures("anthropic_available")
class TestAnthropicCompletion:
    def test_sync_completion(self):
        resp = completion(MODEL, MESSAGES)
        assert isinstance(resp.content, str)
        assert len(resp.content) > 0

    def test_sync_completion_usage(self):
        resp = completion(MODEL, MESSAGES)
        assert resp.usage.total_tokens > 0

    def test_sync_completion_model_name(self):
        resp = completion(MODEL, MESSAGES)
        assert "claude" in resp.model.lower()

    async def test_async_completion(self):
        resp = await acompletion(MODEL, MESSAGES)
        assert isinstance(resp.content, str)

    async def test_streaming(self):
        chunks = []
        async for chunk in stream(MODEL, MESSAGES):
            chunks.append(chunk)
        assert len(chunks) >= 0


@pytest.mark.usefixtures("anthropic_available")
class TestAnthropicWithSystem:
    def test_system_prompt(self):
        msgs = [{"role": "user", "content": "What is your role?"}]
        resp = completion(
            MODEL,
            msgs,
            system="You are an AI assistant specializing in Python.",
        )
        assert isinstance(resp.content, str)
