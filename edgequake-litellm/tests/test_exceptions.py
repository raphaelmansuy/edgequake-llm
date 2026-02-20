"""
test_exceptions.py — Tests for the exception hierarchy and error mapping.
"""
from __future__ import annotations

import pytest

from edgequake_litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    ContextWindowExceededError,
    LiteLLMEdgeError,
    ModelNotFoundError,
    NotImplementedError,
    RateLimitError,
    Timeout,
    _map_builtin,
)


class TestExceptionHierarchy:
    def test_authentication_error_is_edge_quake_error(self):
        exc = AuthenticationError("bad key")
        assert isinstance(exc, LiteLLMEdgeError)
        assert isinstance(exc, Exception)

    def test_rate_limit_error_is_edge_quake_error(self):
        exc = RateLimitError("rate limited")
        assert isinstance(exc, LiteLLMEdgeError)

    def test_timeout_is_edge_quake_error(self):
        exc = Timeout("timed out")
        assert isinstance(exc, LiteLLMEdgeError)

    def test_context_window_is_edge_quake_error(self):
        exc = ContextWindowExceededError("too many tokens")
        assert isinstance(exc, LiteLLMEdgeError)

    def test_api_connection_error_is_edge_quake_error(self):
        exc = APIConnectionError("no route to host")
        assert isinstance(exc, LiteLLMEdgeError)

    def test_model_not_found_is_edge_quake_error(self):
        exc = ModelNotFoundError("cannot find model")
        assert isinstance(exc, LiteLLMEdgeError)

    def test_exception_has_message(self):
        exc = AuthenticationError("bad key", llm_provider="openai", model="gpt-4o")
        assert exc.message == "bad key"
        assert exc.llm_provider == "openai"
        assert exc.model == "gpt-4o"

    def test_exception_str_includes_provider(self):
        exc = AuthenticationError("bad key", llm_provider="openai")
        assert "openai" in str(exc)

    def test_exception_str_includes_model(self):
        exc = AuthenticationError("bad key", model="gpt-4o")
        assert "gpt-4o" in str(exc)


class TestMapBuiltin:
    def test_permission_error_maps_to_authentication(self):
        mapped = _map_builtin(PermissionError("Authentication error: bad key"))
        assert isinstance(mapped, AuthenticationError)

    def test_timeout_error_maps_to_timeout(self):
        mapped = _map_builtin(TimeoutError("Request timed out"))
        assert isinstance(mapped, Timeout)

    def test_connection_error_maps_to_api_connection(self):
        mapped = _map_builtin(ConnectionError("Connection refused"))
        assert isinstance(mapped, APIConnectionError)

    def test_runtime_error_with_rate_limit_maps_correctly(self):
        mapped = _map_builtin(RuntimeError("Rate limit exceeded: try again"))
        assert isinstance(mapped, RateLimitError)

    def test_value_error_with_token_limit_maps_correctly(self):
        mapped = _map_builtin(ValueError("Token limit exceeded: max=100000, got=120000"))
        assert isinstance(mapped, ContextWindowExceededError)

    def test_value_error_with_model_not_found_maps_correctly(self):
        mapped = _map_builtin(ValueError("Model not found: gpt-99"))
        assert isinstance(mapped, ModelNotFoundError)

    def test_generic_runtime_error_maps_to_api_error(self):
        mapped = _map_builtin(RuntimeError("unrecognised error XYZ"))
        assert isinstance(mapped, APIError)

    def test_mapped_error_preserves_provider(self):
        mapped = _map_builtin(PermissionError("bad key"), provider="openai")
        assert mapped.llm_provider == "openai"

    def test_mapped_error_preserves_model(self):
        mapped = _map_builtin(PermissionError("bad key"), model="gpt-4o")
        assert mapped.model == "gpt-4o"


class TestRaisedFromProvider:
    """Ensure that provider calls surface the right exception type."""

    def test_unknown_provider_raises_value_error_or_eq_error(self):
        """Completions with bad providers should raise an LiteLLMEdgeError or ValueError/RuntimeError."""
        from edgequake_litellm import completion
        with pytest.raises((LiteLLMEdgeError, ValueError, RuntimeError)):
            completion("nonexistent_provider_xyz/model", [{"role": "user", "content": "hi"}])
