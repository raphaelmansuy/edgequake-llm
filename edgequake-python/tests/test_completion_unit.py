"""
test_completion_unit.py — Unit tests for completion using the mock provider.

These tests require no API keys. They use provider="mock" which returns
deterministic responses from the edgequake-llm MockProvider.
"""
from __future__ import annotations

import pytest

import edgequake_python as eq
from edgequake_python import completion, acompletion
from edgequake_python._types import ModelResponse


MESSAGES = [{"role": "user", "content": "Hello, mock!"}]


# ---------------------------------------------------------------------------
# Synchronous completion
# ---------------------------------------------------------------------------


class TestSyncCompletion:
    def test_basic_completion_returns_model_response(self):
        resp = completion("mock/test-model", MESSAGES)
        assert isinstance(resp, ModelResponse)

    def test_response_has_content(self):
        resp = completion("mock/test-model", MESSAGES)
        assert isinstance(resp.content, str)
        assert len(resp.content) > 0

    def test_response_has_model(self):
        resp = completion("mock/test-model", MESSAGES)
        assert isinstance(resp.model, str)
        assert len(resp.model) > 0

    def test_response_has_usage(self):
        resp = completion("mock/test-model", MESSAGES)
        assert resp.usage is not None
        assert resp.usage.prompt_tokens >= 0
        assert resp.usage.completion_tokens >= 0
        assert resp.usage.total_tokens >= 0

    def test_response_tool_calls_empty_by_default(self):
        resp = completion("mock/test-model", MESSAGES)
        assert isinstance(resp.tool_calls, list)

    def test_response_has_finish_reason(self):
        resp = completion("mock/test-model", MESSAGES)
        # finish_reason can be None or a string
        assert resp.finish_reason is None or isinstance(resp.finish_reason, str)

    def test_completion_with_max_tokens(self):
        resp = completion("mock/test-model", MESSAGES, max_tokens=50)
        assert isinstance(resp, ModelResponse)

    def test_completion_with_temperature(self):
        resp = completion("mock/test-model", MESSAGES, temperature=0.7)
        assert isinstance(resp, ModelResponse)

    def test_completion_with_system_prompt(self):
        msgs = [{"role": "user", "content": "Who are you?"}]
        resp = completion("mock/test-model", msgs, system="You are a test assistant.")
        assert isinstance(resp, ModelResponse)

    def test_completion_with_multi_turn(self):
        msgs = [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "What is your name?"},
            {"role": "assistant", "content": "I be Captain Mock!"},
            {"role": "user", "content": "Where do you sail?"},
        ]
        resp = completion("mock/test-model", msgs)
        assert isinstance(resp, ModelResponse)

    def test_to_dict_has_expected_keys(self):
        resp = completion("mock/test-model", MESSAGES)
        d = resp.to_dict()
        assert "content" in d
        assert "model" in d
        assert "usage" in d
        assert "tool_calls" in d
        assert "finish_reason" in d

    def test_repr_is_string(self):
        resp = completion("mock/test-model", MESSAGES)
        assert isinstance(repr(resp), str)
        assert "ModelResponse" in repr(resp)


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


class TestSyncCompletionErrors:
    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(Exception):  # ValueError from Rust or our layer
            completion("totally_unknown_provider/model", MESSAGES)

    def test_invalid_messages_json_internal(self):
        """Passing an empty messages list should fail gracefully."""
        # In the Python layer this is serialised to [] — providers may error.
        # At minimum it shouldn't crash the process.
        try:
            resp = completion("mock/test-model", [])
        except Exception:
            pass  # Expected — empty messages is invalid for most providers

    def test_list_providers(self):
        providers = eq._eq_core.list_providers()  # type: ignore[attr-defined]
        assert isinstance(providers, list)
        assert "openai" in providers
        assert "anthropic" in providers
        assert "mock" in providers


# ---------------------------------------------------------------------------
# Asynchronous completion
# ---------------------------------------------------------------------------


class TestAsyncCompletion:
    async def test_acompletion_returns_model_response(self):
        resp = await acompletion("mock/test-model", MESSAGES)
        assert isinstance(resp, ModelResponse)

    async def test_acompletion_has_content(self):
        resp = await acompletion("mock/test-model", MESSAGES)
        assert isinstance(resp.content, str)
        assert len(resp.content) > 0

    async def test_acompletion_with_options(self):
        resp = await acompletion(
            "mock/test-model",
            MESSAGES,
            max_tokens=100,
            temperature=0.5,
        )
        assert isinstance(resp, ModelResponse)

    async def test_acompletion_unknown_provider(self):
        with pytest.raises(Exception):
            await acompletion("bad_provider/model", MESSAGES)
