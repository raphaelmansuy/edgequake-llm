"""Type stubs for the _elc_core native Rust extension module.

This file provides static typing information for IDEs and type checkers.
The actual implementation is in the compiled ``_elc_core.so`` / ``_elc_core.pyd``
Rust extension.
"""
from __future__ import annotations

from collections.abc import Awaitable
from typing import Any

__version__: str

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_read_input_tokens: int | None
    reasoning_tokens: int | None

    def __repr__(self) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...

class ToolCall:
    id: str
    function_name: str
    function_arguments: str

    def __repr__(self) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...

class ModelResponse:
    content: str
    model: str
    finish_reason: str | None
    usage: Usage
    tool_calls: list[ToolCall]
    thinking_content: str | None

    def __repr__(self) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...
    def has_tool_calls(self) -> bool: ...

class StreamChunk:
    content: str | None
    thinking: str | None
    is_finished: bool
    finish_reason: str | None
    tool_call_delta: dict[str, Any] | None

    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Completion functions
# ---------------------------------------------------------------------------

def completion(
    provider: str,
    model: str,
    messages_json: str,
    options_json: str | None = None,
    tools_json: str | None = None,
    tool_choice_json: str | None = None,
) -> ModelResponse: ...

def acompletion(
    provider: str,
    model: str,
    messages_json: str,
    options_json: str | None = None,
    tools_json: str | None = None,
    tool_choice_json: str | None = None,
) -> Awaitable[ModelResponse]: ...

def stream_completion(
    provider: str,
    model: str,
    messages_json: str,
    options_json: str | None = None,
    tools_json: str | None = None,
    tool_choice_json: str | None = None,
) -> Awaitable[list[StreamChunk]]: ...

# ---------------------------------------------------------------------------
# Provider info
# ---------------------------------------------------------------------------

def list_providers() -> list[str]: ...
def detect_provider() -> str | None: ...

# ---------------------------------------------------------------------------
# Embedding functions
# ---------------------------------------------------------------------------

def embed(
    provider: str,
    model: str,
    texts: list[str],
) -> list[list[float]]: ...

def aembed(
    provider: str,
    model: str,
    texts: list[str],
) -> Awaitable[list[list[float]]]: ...
