"""
_types.py — Public type definitions for edgequake-litellm.

Re-exports PyO3-generated types and adds Python-native type aliases so that
users get full IDE completions and type-checker support without needing to
import from the private ``_elc_core`` extension.
"""
from __future__ import annotations

from typing import Any

# Import the Rust-backed types from the native extension module.
# These will exist at runtime; stubs (_elc_core.pyi) provide static types.
try:
    from edgequake_litellm._elc_core import (  # type: ignore[import-untyped]
        ModelResponse,
        StreamChunk,
        ToolCall,
        Usage,
    )
except ImportError:
    # Allow importing the module when the extension is not compiled (e.g. docs).
    ModelResponse = Any  # type: ignore[assignment, misc]
    Usage = Any  # type: ignore[assignment, misc]
    ToolCall = Any  # type: ignore[assignment, misc]
    StreamChunk = Any  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Convenience aliases matching litellm naming
# ---------------------------------------------------------------------------

#: Alias for ModelResponse — matches litellm.ModelResponse
LiteLLMModelResponse = ModelResponse

#: Usage stats — matches litellm.Usage
LiteLLMUsage = Usage

__all__ = [
    "ModelResponse",
    "Usage",
    "ToolCall",
    "StreamChunk",
    "LiteLLMModelResponse",
    "LiteLLMUsage",
]
