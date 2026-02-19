"""
test_e2e_mistral.py — End-to-end tests against the Mistral AI API.

These tests are automatically skipped when MISTRAL_API_KEY is not set.
"""
from __future__ import annotations

import json
import pytest

from edgequake_litellm import completion, acompletion, stream


MODEL = "mistral/mistral-small-latest"
MESSAGES = [{"role": "user", "content": "Reply with exactly one word: pong"}]


@pytest.mark.usefixtures("mistral_available")
class TestMistralCompletion:
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
        resp = completion(MODEL, msgs, system="You are an enthusiastic assistant.")
        assert isinstance(resp.content, str)
        assert len(resp.content) > 0

    def test_completion_with_max_tokens(self):
        resp = completion(MODEL, MESSAGES, max_tokens=20)
        assert isinstance(resp.content, str)

    def test_completion_with_temperature_zero(self):
        resp1 = completion(MODEL, MESSAGES, temperature=0.0)
        resp2 = completion(MODEL, MESSAGES, temperature=0.0)
        # Deterministic at temperature=0
        assert resp1.content == resp2.content

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

    async def test_streaming_accumulates_content(self):
        accumulated = ""
        async for chunk in stream(MODEL, MESSAGES):
            if chunk.content:
                accumulated += chunk.content
        # We asked for exactly one word so at least 1 char should appear
        assert len(accumulated) >= 1

    def test_tool_calling(self):
        """Mistral supports function/tool calling."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"}
                        },
                        "required": ["city"],
                    },
                },
            }
        ]
        msgs = [{"role": "user", "content": "What's the weather in Paris?"}]
        resp = completion(MODEL, msgs, tools=tools, tool_choice="auto")
        # Either a tool call or a text response is acceptable
        assert isinstance(resp.content, str) or resp.tool_calls is not None

    async def test_multi_turn(self):
        msgs = [
            {"role": "user", "content": "Remember: the magic number is 7."},
            {"role": "assistant", "content": "Got it, the magic number is 7."},
            {"role": "user", "content": "What is the magic number?"},
        ]
        resp = await acompletion(MODEL, msgs)
        assert isinstance(resp.content, str)
        assert "7" in resp.content
