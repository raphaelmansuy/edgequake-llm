"""
config.py — Library-wide configuration and defaults.

Mirrors litellm's global config so existing code can swap the import with
minimal changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LiteLLMEdgeConfig:
    """Global configuration for edgequake-litellm.

    Attributes:
        default_provider:   Provider to use when none is specified.
        default_model:      Model to use when none is specified.
        timeout:            Default request timeout in seconds.
        max_retries:        Default number of retries on transient errors.
        drop_params:        Drop unknown model params (litellm compatibility).
        verbose:            Enable verbose debug logging.
    """
    default_provider: str = field(
        default_factory=lambda: os.environ.get("LITELLM_EDGE_PROVIDER") or os.environ.get("EDGEQUAKE_PROVIDER", "openai")
    )
    default_model: str = field(
        default_factory=lambda: os.environ.get("LITELLM_EDGE_MODEL") or os.environ.get("EDGEQUAKE_MODEL", "gpt-4o-mini")
    )
    timeout: float = field(
        default_factory=lambda: float(os.environ.get("LITELLM_EDGE_TIMEOUT") or os.environ.get("EDGEQUAKE_TIMEOUT", "60"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.environ.get("LITELLM_EDGE_MAX_RETRIES") or os.environ.get("EDGEQUAKE_MAX_RETRIES", "3"))
    )
    drop_params: bool = True
    verbose: bool = field(
        default_factory=lambda: os.environ.get("LITELLM_EDGE_VERBOSE", os.environ.get("EDGEQUAKE_VERBOSE", "")).lower() in ("1", "true", "yes")
    )


# Singleton config — mutate at startup just like litellm globals
_config = LiteLLMEdgeConfig()


def get_config() -> LiteLLMEdgeConfig:
    """Return the global config singleton."""
    return _config


def set_default_provider(provider: str) -> None:
    """Set the default provider globally."""
    _config.default_provider = provider


def set_default_model(model: str) -> None:
    """Set the default model globally."""
    _config.default_model = model


# ---------------------------------------------------------------------------
# CompletionOptions builder — serialises to the JSON consumed by _elc_core
# ---------------------------------------------------------------------------

def build_options(
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop: Optional[List[str]] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    response_format: Optional[str] = None,
    system_prompt: Optional[str] = None,
    **_kwargs: Any,
) -> Optional[str]:
    """Build a JSON string for CompletionOptions.

    Unknown keyword arguments are silently dropped (litellm drop_params behaviour).

    Returns:
        JSON string or None when no options are specified.
    """
    import json

    opts: Dict[str, Any] = {}
    if max_tokens is not None:
        opts["max_tokens"] = max_tokens
    if temperature is not None:
        opts["temperature"] = temperature
    if top_p is not None:
        opts["top_p"] = top_p
    if stop is not None:
        opts["stop"] = stop
    if frequency_penalty is not None:
        opts["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        opts["presence_penalty"] = presence_penalty
    if response_format is not None:
        opts["response_format"] = response_format
    if system_prompt is not None:
        opts["system_prompt"] = system_prompt

    return json.dumps(opts) if opts else None


__all__ = [
    "LiteLLMEdgeConfig",
    "EdgeQuakeConfig",  # backward-compat alias
    "get_config",
    "set_default_provider",
    "set_default_model",
    "build_options",
]

# Backward-compat alias for code that used edgequake_python
EdgeQuakeConfig = LiteLLMEdgeConfig
