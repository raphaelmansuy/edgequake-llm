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

# ---------------------------------------------------------------------------
# Discovery types and functions
# ---------------------------------------------------------------------------

class DiscoveredModel:
    id: str
    name: str
    provider: str
    context_length: int
    max_output_tokens: int
    supports_vision: bool
    supports_tools: bool
    supports_thinking: bool
    supports_streaming: bool
    supports_json_mode: bool
    source: str
    available: bool
    cost_per_m_input: float | None
    cost_per_m_output: float | None
    model_type: str
    tags: list[str]
    deprecated: bool

    def __repr__(self) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...

def discover_all() -> list[DiscoveredModel]: ...
def adiscover_all() -> Awaitable[list[DiscoveredModel]]: ...
def find_models(filter_json: str | None = None) -> list[DiscoveredModel]: ...
def afind_models(filter_json: str | None = None) -> Awaitable[list[DiscoveredModel]]: ...
def get_model(provider: str, model_id: str) -> DiscoveredModel | None: ...
def aget_model(provider: str, model_id: str) -> Awaitable[DiscoveredModel | None]: ...
def find_static_models(filter_json: str | None = None) -> list[DiscoveredModel]: ...
def search_models(
    query: str,
    provider: str | None = None,
    fuzzy: bool = False,
    min_score: float | None = None,
    limit: int | None = None,
    filter_json: str | None = None,
) -> list[ModelSearchMatch]: ...
def search_static_models_by_name(
    query: str,
    provider: str | None = None,
    fuzzy: bool = False,
    min_score: float | None = None,
    limit: int | None = None,
    filter_json: str | None = None,
) -> list[ModelSearchMatch]: ...
def lookup_model_by_name(provider: str, name_or_id: str) -> DiscoveredModel | None: ...
def asearch_models(
    query: str,
    provider: str | None = None,
    fuzzy: bool = False,
    min_score: float | None = None,
    limit: int | None = None,
    filter_json: str | None = None,
) -> Awaitable[list[ModelSearchMatch]]: ...
def discover_provider(provider_id: str) -> list[DiscoveredModel]: ...
def list_discovery_providers() -> list[str]: ...
def discovery_info() -> list[dict[str, str]]: ...

class ModelSearchMatch:
    model: DiscoveredModel
    score: float
    match_kind: str

    def __repr__(self) -> str: ...
