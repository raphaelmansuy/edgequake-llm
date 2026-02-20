"""
_compat.py — litellm-compatible response shims.

Wraps the raw PyO3 types returned by the Rust core so that existing litellm
call-sites work without modification.

Compatibility targets (litellm 1.x):
  - resp.choices[0].message.content
  - resp.choices[0].message.tool_calls
  - resp.choices[0].message.role
  - resp.choices[0].finish_reason
  - resp["choices"][0]["message"]["content"]  (dict-style)
  - resp.id, resp.created, resp.object, resp.system_fingerprint
  - resp.model, resp.usage
  - resp.response_ms  (latency in milliseconds)
  - result.data[0].embedding  (EmbeddingResponseCompat)
  - chunk.choices[0].delta.content  (StreamChunkCompat)
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from typing import Any

# ---------------------------------------------------------------------------
# ModelResponse compat shim
# ---------------------------------------------------------------------------

class _Message:
    """Mirrors litellm's Choices.message object."""

    __slots__ = ("content", "role", "tool_calls", "function_call")

    def __init__(
        self,
        content: str | None,
        role: str = "assistant",
        tool_calls: Any | None = None,
    ) -> None:
        self.content = content
        self.role = role
        self.tool_calls = tool_calls
        self.function_call = None

    # dict-style access
    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return f"Message(role={self.role!r}, content={self.content!r})"


class _Choice:
    """Mirrors litellm's Choices object."""

    __slots__ = ("message", "finish_reason", "index")

    def __init__(
        self,
        message: _Message,
        finish_reason: str = "stop",
        index: int = 0,
    ) -> None:
        self.message = message
        self.finish_reason = finish_reason
        self.index = index

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return (
            f"Choice(index={self.index}, finish_reason={self.finish_reason!r}, "
            f"message={self.message!r})"
        )


class ModelResponseCompat:
    """Wraps the PyO3 ``ModelResponse`` to expose the full litellm-compatible shape.

    Existing code that uses our ``resp.content`` shortcut continues to work.
    Code that follows the litellm / OpenAI convention
    (``resp.choices[0].message.content``) also works.

    Examples::

        resp = completion("openai/gpt-4o-mini", messages)

        # litellm path
        print(resp.choices[0].message.content)

        # our shortcut (still works)
        print(resp.content)

        # dict-style access
        print(resp["choices"][0]["message"]["content"])
    """

    def __init__(self, raw: Any, response_ms: float = 0.0) -> None:
        self._raw = raw

        # Standard litellm / OpenAI fields
        self.id: str = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self.created: int = int(time.time())
        self.object: str = "chat.completion"
        self.system_fingerprint: str | None = None
        self.model: str = getattr(raw, "model", "")
        self.usage: Any = getattr(raw, "usage", None)

        # Latency — populated by completion() / acompletion() wrappers
        self.response_ms: float = response_ms

        # Grab finish_reason from raw if available
        _raw_finish = getattr(raw, "finish_reason", "stop") or "stop"

        # Build the choices list
        _tool_calls = getattr(raw, "tool_calls", None) or None
        _msg = _Message(
            content=getattr(raw, "content", None),
            role="assistant",
            tool_calls=_tool_calls,
        )
        self.choices: list[_Choice] = [
            _Choice(message=_msg, finish_reason=_raw_finish, index=0)
        ]

    # ── Convenience shortcuts (edgequake-litellm extensions) ───────────────

    @property
    def content(self) -> str | None:
        """Shortcut: ``resp.content`` → ``resp.choices[0].message.content``."""
        return self._raw.content

    @property
    def tool_calls(self) -> Any | None:
        """Shortcut: ``resp.tool_calls`` → ``resp.choices[0].message.tool_calls``."""
        return getattr(self._raw, "tool_calls", None)

    @property
    def finish_reason(self) -> str | None:
        """Shortcut: ``resp.finish_reason`` — mirrors ``resp.choices[0].finish_reason``."""
        return self.choices[0].finish_reason if self.choices else None

    @property
    def thinking_content(self) -> str | None:
        """Reasoning/thinking text (Anthropic extended thinking)."""
        return getattr(self._raw, "thinking_content", None)

    def has_tool_calls(self) -> bool:
        """Return True if there are tool calls in the response."""
        tc = self.tool_calls
        return bool(tc)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation (backward compat with PyO3 ModelResponse)."""
        result: dict[str, Any] = {
            "id": self.id,
            "created": self.created,
            "object": self.object,
            "model": self.model,
            "content": self.content,
            "finish_reason": self.finish_reason,
            "tool_calls": self.tool_calls,
            "response_ms": self.response_ms,
        }
        if self.usage is not None:
            try:
                result["usage"] = self.usage.to_dict()
            except AttributeError:
                result["usage"] = {
                    "prompt_tokens": getattr(self.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(self.usage, "completion_tokens", 0),
                    "total_tokens": getattr(self.usage, "total_tokens", 0),
                }
        else:
            result["usage"] = {}
        return result

    # ── Dict-style access ──────────────────────────────────────────────────

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return (
            f"ModelResponse(id={self.id!r}, model={self.model!r}, "
            f"content={self.content!r})"
        )

    def __str__(self) -> str:
        return self.content or ""


# ---------------------------------------------------------------------------
# Streaming chunk compat shim
# ---------------------------------------------------------------------------

class _Delta:
    """Mirrors litellm's streaming delta object."""

    __slots__ = ("content", "role", "tool_calls", "function_call")

    def __init__(
        self,
        content: str | None,
        role: str = "assistant",
        tool_calls: Any | None = None,
    ) -> None:
        self.content = content
        self.role = role
        self.tool_calls = tool_calls
        self.function_call = None

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


class _StreamChoice:
    """Mirrors litellm's streaming choice object."""

    __slots__ = ("delta", "finish_reason", "index")

    def __init__(
        self,
        delta: _Delta,
        finish_reason: str | None = None,
        index: int = 0,
    ) -> None:
        self.delta = delta
        self.finish_reason = finish_reason
        self.index = index

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


class StreamChunkCompat:
    """Wraps the PyO3 ``StreamChunk`` to expose the full litellm-compatible shape.

    Examples::

        # litellm pattern
        for chunk in ...
            content = chunk.choices[0].delta.content or ""

        # edgequake-litellm pattern (still works)
        for chunk in ...
            content = chunk.content or ""
    """

    def __init__(self, raw: Any) -> None:
        self._raw = raw
        self.model: str = ""
        self.object: str = "chat.completion.chunk"
        self.id: str = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self.created: int = int(time.time())

        _delta = _Delta(
            content=getattr(raw, "content", None),
            role="assistant",
        )
        _finish = raw.finish_reason if getattr(raw, "is_finished", False) else None
        self.choices: list[_StreamChoice] = [
            _StreamChoice(delta=_delta, finish_reason=_finish, index=0)
        ]

    @property
    def content(self) -> str | None:
        return self._raw.content

    @property
    def thinking(self) -> str | None:
        return getattr(self._raw, "thinking", None)

    @property
    def is_finished(self) -> bool:
        return getattr(self._raw, "is_finished", False)

    @property
    def finish_reason(self) -> str | None:
        return getattr(self._raw, "finish_reason", None)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __repr__(self) -> str:
        return (
            f"StreamChunk(content={self.content!r}, "
            f"is_finished={self.is_finished!r})"
        )


# ---------------------------------------------------------------------------
# Embedding response compat shim
# ---------------------------------------------------------------------------

class _EmbeddingData:
    """Mirrors litellm's embedding data object."""

    __slots__ = ("embedding", "index", "object")

    def __init__(self, embedding: list[float], index: int = 0) -> None:
        self.embedding = embedding
        self.index = index
        self.object = "embedding"

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


class _EmbeddingUsage:
    __slots__ = ("prompt_tokens", "total_tokens", "completion_tokens")

    def __init__(self, prompt_tokens: int = 0, total_tokens: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.total_tokens = total_tokens
        self.completion_tokens = 0

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


class EmbeddingResponseCompat:
    """Wraps the ``List[List[float]]`` from Rust into a litellm-compatible object.

    Examples::

        # litellm pattern
        result = embedding("openai/text-embedding-3-small", texts)
        vectors = [item.embedding for item in result.data]

        # legacy edgequake-litellm pattern (still works)
        vectors = list(result)   # iterates over List[float] items
        vectors = result[0]       # index access returns List[float]
    """

    def __init__(
        self,
        vectors: list[list[float]],
        model: str = "",
        usage: _EmbeddingUsage | None = None,
    ) -> None:
        self.object = "list"
        self.model = model
        self.data: list[_EmbeddingData] = [
            _EmbeddingData(v, i) for i, v in enumerate(vectors)
        ]
        self.usage = usage or _EmbeddingUsage()

    # ── Backwards-compat: iterate / index returns List[float] directly ──────

    def __iter__(self) -> Iterator[list[float]]:
        """Yield raw ``List[float]`` vectors — maintains old return type behaviour."""
        return (item.embedding for item in self.data)

    def __getitem__(self, idx: int) -> list[float]:
        return self.data[idx].embedding

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f"EmbeddingResponse(model={self.model!r}, "
            f"n={len(self.data)}, dims={len(self.data[0].embedding) if self.data else 0})"
        )


# ---------------------------------------------------------------------------
# stream_chunk_builder helper
# ---------------------------------------------------------------------------

def stream_chunk_builder(
    chunks: list[Any],
    messages: list[Any] | None = None,
) -> ModelResponseCompat:
    """Reconstruct a full ``ModelResponse`` from a list of streaming chunks.

    Mirrors ``litellm.stream_chunk_builder(chunks, messages)``.

    Args:
        chunks:   List of ``StreamChunk`` or ``StreamChunkCompat`` objects.
        messages: Original messages list (unused, for API compat).

    Returns:
        ``ModelResponseCompat`` whose ``.content`` is the concatenated text.
    """
    content_parts = []
    model = ""
    finish_reason = "stop"

    for chunk in chunks:
        # Support both raw StreamChunk (PyO3) and StreamChunkCompat
        raw_content = (
            chunk.content
            if hasattr(chunk, "content")
            else chunk.choices[0].delta.content
            if hasattr(chunk, "choices")
            else None
        )
        if raw_content:
            content_parts.append(raw_content)

        # Try to grab model
        if not model and hasattr(chunk, "model") and chunk.model:
            model = chunk.model

        # Finish reason
        raw_finish = (
            chunk.finish_reason
            if hasattr(chunk, "finish_reason") and chunk.finish_reason
            else None
        )
        if raw_finish:
            finish_reason = raw_finish

    full_content = "".join(content_parts)

    # Build a minimal "raw" object so ModelResponseCompat can wrap it
    class _FakeRaw:
        pass

    raw = _FakeRaw()
    raw.content = full_content  # type: ignore[attr-defined]
    raw.model = model or "unknown"  # type: ignore[attr-defined]
    raw.tool_calls = None  # type: ignore[attr-defined]
    raw.usage = None  # type: ignore[attr-defined]

    result = ModelResponseCompat(raw)
    result.choices[0].finish_reason = finish_reason
    return result


__all__ = [
    "ModelResponseCompat",
    "StreamChunkCompat",
    "EmbeddingResponseCompat",
    "stream_chunk_builder",
]
