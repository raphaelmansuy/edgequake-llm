"""
test_litellm_compat.py — LiteLLM API compatibility tests.

Tests that edgequake_python's public API matches litellm's API surface so
existing code can migrate with minimal changes.
"""
from __future__ import annotations

import inspect
import pytest


class TestAPICompatibility:
    """Verify the module exports expected names and signatures."""

    def test_completion_is_callable(self):
        from edgequake_python import completion
        assert callable(completion)

    def test_acompletion_is_coroutine_function(self):
        from edgequake_python import acompletion
        assert inspect.iscoroutinefunction(acompletion)

    def test_embedding_is_callable(self):
        from edgequake_python import embedding
        assert callable(embedding)

    def test_aembedding_is_coroutine_function(self):
        from edgequake_python import aembedding
        assert inspect.iscoroutinefunction(aembedding)

    def test_stream_is_async_generator_function(self):
        from edgequake_python import stream
        assert inspect.isasyncgenfunction(stream)

    def test_completion_accepts_model_param(self):
        sig = inspect.signature(__import__("edgequake_python").completion)
        assert "model" in sig.parameters

    def test_completion_accepts_messages_param(self):
        sig = inspect.signature(__import__("edgequake_python").completion)
        assert "messages" in sig.parameters

    def test_completion_accepts_max_tokens(self):
        sig = inspect.signature(__import__("edgequake_python").completion)
        assert "max_tokens" in sig.parameters

    def test_completion_accepts_temperature(self):
        sig = inspect.signature(__import__("edgequake_python").completion)
        assert "temperature" in sig.parameters

    def test_completion_accepts_tools(self):
        sig = inspect.signature(__import__("edgequake_python").completion)
        assert "tools" in sig.parameters

    def test_acompletion_has_same_params_as_completion(self):
        from edgequake_python import completion, acompletion
        sync_sig = inspect.signature(completion)
        async_sig = inspect.signature(acompletion)
        sync_params = set(sync_sig.parameters.keys())
        async_params = set(async_sig.parameters.keys())
        # async should have at least all params that sync has
        assert sync_params == async_params


class TestModelResponseCompatibility:
    """Verify ModelResponse has litellm-compatible attributes."""

    def test_model_response_has_content(self):
        from edgequake_python import completion
        resp = completion("mock/test-model", [{"role": "user", "content": "hi"}])
        assert hasattr(resp, "content")

    def test_model_response_has_model(self):
        from edgequake_python import completion
        resp = completion("mock/test-model", [{"role": "user", "content": "hi"}])
        assert hasattr(resp, "model")

    def test_model_response_has_usage(self):
        from edgequake_python import completion
        resp = completion("mock/test-model", [{"role": "user", "content": "hi"}])
        assert hasattr(resp, "usage")
        assert hasattr(resp.usage, "prompt_tokens")
        assert hasattr(resp.usage, "completion_tokens")
        assert hasattr(resp.usage, "total_tokens")

    def test_model_response_has_finish_reason(self):
        from edgequake_python import completion
        resp = completion("mock/test-model", [{"role": "user", "content": "hi"}])
        assert hasattr(resp, "finish_reason")

    def test_model_response_has_tool_calls(self):
        from edgequake_python import completion
        resp = completion("mock/test-model", [{"role": "user", "content": "hi"}])
        assert hasattr(resp, "tool_calls")
        assert isinstance(resp.tool_calls, list)

    def test_model_response_to_dict(self):
        from edgequake_python import completion
        resp = completion("mock/test-model", [{"role": "user", "content": "hi"}])
        d = resp.to_dict()
        assert isinstance(d, dict)

    def test_model_response_repr(self):
        from edgequake_python import completion
        resp = completion("mock/test-model", [{"role": "user", "content": "hi"}])
        r = repr(resp)
        assert "ModelResponse" in r


class TestProviderRouting:
    """Test the provider/model routing convention."""

    def test_slash_separates_provider_and_model(self):
        from edgequake_python.completion import _parse_model
        provider, model = _parse_model("openai/gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_model_only_uses_default_provider(self):
        from edgequake_python.completion import _parse_model
        from edgequake_python.config import get_config
        provider, model = _parse_model("gpt-4o")
        assert provider == get_config().default_provider
        assert model == "gpt-4o"

    def test_nested_model_path_preserved(self):
        from edgequake_python.completion import _parse_model
        provider, model = _parse_model("openrouter/meta-llama/llama-3.1-70b-instruct")
        assert provider == "openrouter"
        assert model == "meta-llama/llama-3.1-70b-instruct"


class TestConfigCompatibility:
    """Test litellm-compatible config functions."""

    def test_set_default_provider(self):
        from edgequake_python import set_default_provider, get_config
        original = get_config().default_provider
        set_default_provider("anthropic")
        assert get_config().default_provider == "anthropic"
        set_default_provider(original)  # restore

    def test_set_default_model(self):
        from edgequake_python import set_default_model, get_config
        original = get_config().default_model
        set_default_model("claude-3-5-haiku-20241022")
        assert get_config().default_model == "claude-3-5-haiku-20241022"
        set_default_model(original)  # restore
