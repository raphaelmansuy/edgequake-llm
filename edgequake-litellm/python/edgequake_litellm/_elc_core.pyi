"""Type stubs for the _elc_core native Rust extension module.

This file provides static typing information for IDEs and type checkers.
The actual implementation is in the compiled ``_elc_core.so`` / ``_elc_core.pyd``
Rust extension.
"""
from __future__ import annotations

from typing import Awaitable, Dict, List, Optional, Any

__version__: str

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_read_input_tokens: Optional[int]
    reasoning_tokens: Optional[int]

    def __repr__(self) -> str: ...
    def to_dict(self) -> Dict[str, Any]: ...

class ToolCall:
    id: str
    function_name: str
    function_arguments: str

    def __repr__(self) -> str: ...
    def to_dict(self) -> Dict[str, Any]: ...

class ModelResponse:
    content: str
    model: str
    finish_reason: Optional[str]
    usage: Usage
    tool_calls: List[ToolCall]
    thinking_content: Optional[str]

    def __repr__(self) -> str: ...
    def to_dict(self) -> Dict[str, Any]: ...
    def has_tool_calls(self) -> bool: ...

class StreamChunk:
    content: Optional[str]
    thinking: Optional[str]
    is_finished: bool
    finish_reason: Optional[str]
    tool_call_delta: Optional[Dict[str, Any]]

    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Completion functions
# ---------------------------------------------------------------------------

def completion(
    provider: str,
    model: str,
    messages_json: str,
    options_json: Optional[str] = None,
    tools_json: Optional[str] = None,
    tool_choice_json: Optional[str] = None,
) -> ModelResponse: ...

def acompletion(
    provider: str,
    model: str,
    messages_json: str,
    options_json: Optional[str] = None,
    tools_json: Optional[str] = None,
    tool_choice_json: Optional[str] = None,
) -> Awaitable[ModelResponse]: ...

def stream_completion(
    provider: str,
    model: str,
    messages_json: str,
    options_json: Optional[str] = None,
    tools_json: Optional[str] = None,
    tool_choice_json: Optional[str] = None,
) -> Awaitable[List[StreamChunk]]: ...

# ---------------------------------------------------------------------------
# Provider info
# ---------------------------------------------------------------------------

def list_providers() -> List[str]: ...
def detect_provider() -> Optional[str]: ...

# ---------------------------------------------------------------------------
# Embedding functions
# ---------------------------------------------------------------------------

def embed(
    provider: str,
    model: str,
    texts: List[str],
) -> List[List[float]]: ...

def aembed(
    provider: str,
    model: str,
    texts: List[str],
) -> Awaitable[List[List[float]]]: ...
