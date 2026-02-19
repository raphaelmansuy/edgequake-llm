"""
exceptions.py — Custom exception hierarchy for edgequake-python.

Mirrors litellm's exception naming so existing call sites can use the same
``except`` blocks without modification.
"""
from __future__ import annotations


class EdgeQuakeError(Exception):
    """Base exception for all edgequake-python errors."""

    def __init__(self, message: str, llm_provider: str = "", model: str = "", status_code: int = 0) -> None:
        super().__init__(message)
        self.message = message
        self.llm_provider = llm_provider
        self.model = model
        self.status_code = status_code

    def __str__(self) -> str:
        parts = [self.message]
        if self.llm_provider:
            parts.append(f"provider={self.llm_provider}")
        if self.model:
            parts.append(f"model={self.model}")
        return " | ".join(parts)


# litellm-compatible aliases

class AuthenticationError(EdgeQuakeError):
    """Raised when API credentials are invalid or missing (HTTP 401/403)."""
    pass


class RateLimitError(EdgeQuakeError):
    """Raised when the provider rate limit is exceeded (HTTP 429)."""
    pass


class ContextWindowExceededError(EdgeQuakeError):
    """Raised when the prompt exceeds the model context window."""
    pass


class ModelNotFoundError(EdgeQuakeError):
    """Raised when the requested model does not exist."""
    pass


class Timeout(EdgeQuakeError):
    """Raised when the provider request times out."""
    pass


class APIConnectionError(EdgeQuakeError):
    """Raised on network / connectivity failures."""
    pass


class APIError(EdgeQuakeError):
    """Generic API error returned by the provider."""
    pass


class NotImplementedError(EdgeQuakeError):  # noqa: A001
    """Raised when a feature is not supported by this provider/model."""
    pass


def _map_builtin(exc: Exception, provider: str = "", model: str = "") -> EdgeQuakeError:
    """Convert a built-in Python exception (from the Rust layer) to a named EdgeQuakeError."""
    msg = str(exc)
    if isinstance(exc, PermissionError):
        return AuthenticationError(msg, llm_provider=provider, model=model, status_code=401)
    if isinstance(exc, TimeoutError):
        return Timeout(msg, llm_provider=provider, model=model, status_code=408)
    if isinstance(exc, ConnectionError):
        return APIConnectionError(msg, llm_provider=provider, model=model, status_code=503)
    if isinstance(exc, NotImplementedError):  # type: ignore[misc]
        return NotImplementedError(msg, llm_provider=provider, model=model, status_code=501)
    if isinstance(exc, (ValueError, RuntimeError)):
        lower = msg.lower()
        if "rate limit" in lower:
            return RateLimitError(msg, llm_provider=provider, model=model, status_code=429)
        if "token limit" in lower or "context" in lower:
            return ContextWindowExceededError(msg, llm_provider=provider, model=model, status_code=400)
        if "model not found" in lower or "model" in lower:
            return ModelNotFoundError(msg, llm_provider=provider, model=model, status_code=404)
    return APIError(msg, llm_provider=provider, model=model)
