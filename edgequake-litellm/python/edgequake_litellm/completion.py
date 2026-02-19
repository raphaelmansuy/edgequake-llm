"""
completion.py — LiteLLM-compatible completion API.

Public functions
----------------
completion(model, messages, ...)       → ModelResponse
acompletion(model, messages, ...)      → Awaitable[ModelResponse]

Usage
-----
>>> from edgequake_litellm import completion
>>>
>>> resp = completion(
...     model="openai/gpt-4o-mini",
...     messages=[{"role": "user", "content": "Hello!"}],
... )
>>> print(resp.content)
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from edgequake_litellm._types import ModelResponse
from edgequake_litellm.config import build_options
from edgequake_litellm.exceptions import _map_builtin


try:
    from edgequake_litellm import _elc_core  # type: ignore[import]
except ImportError:
    _elc_core = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_model(model: str) -> tuple[str, str]:
    """Split ``provider/model-name`` into ``(provider, model)``.

    If no ``/`` is present the whole string is treated as the model name and
    the default provider from config is used.
    """
    from edgequake_litellm.config import get_config
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return provider, model_name
    return get_config().default_provider, model


def _serialise_messages(messages: List[Dict[str, Any]]) -> str:
    """Serialise litellm-style message dicts to the JSON consumed by Rust."""
    normalised = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        entry: Dict[str, Any] = {"role": role, "content": content or ""}
        if "name" in msg:
            entry["name"] = msg["name"]
        if "tool_calls" in msg and msg["tool_calls"]:
            entry["tool_calls"] = msg["tool_calls"]
        if "tool_call_id" in msg:
            entry["tool_call_id"] = msg["tool_call_id"]
        normalised.append(entry)
    return json.dumps(normalised)


def _serialise_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    if not tools:
        return None
    return json.dumps(tools)


def _serialise_tool_choice(tool_choice: Optional[Any]) -> Optional[str]:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return json.dumps(tool_choice)
    return json.dumps(tool_choice)


# ---------------------------------------------------------------------------
# Synchronous completion
# ---------------------------------------------------------------------------

def completion(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop: Optional[List[str]] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    response_format: Optional[str] = None,
    system: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    **kwargs: Any,
) -> ModelResponse:
    """Call an LLM provider and return the full response.

    The ``model`` argument follows the ``provider/model-name`` convention used
    by litellm::

        "openai/gpt-4o"
        "anthropic/claude-3-5-sonnet-20241022"
        "ollama/llama3.2"
        "gemini/gemini-2.0-flash"

    Args:
        model:             ``provider/model`` or just ``model`` (uses default provider).
        messages:          List of ``{role, content}`` dicts.
        max_tokens:        Maximum tokens to generate.
        temperature:       Sampling temperature.
        top_p:             Nucleus sampling threshold.
        stop:              Stop sequences.
        frequency_penalty: Frequency penalty.
        presence_penalty:  Presence penalty.
        response_format:   Response format hint (e.g. "json_object").
        system:            System prompt override.
        tools:             List of tool definitions.
        tool_choice:       Tool choice configuration.
        **kwargs:          Extra params are ignored (litellm drop_params).

    Returns:
        ModelResponse

    Raises:
        AuthenticationError:       API key missing or invalid.
        RateLimitError:            Provider rate limit exceeded.
        ContextWindowExceededError: Prompt too long.
        Timeout:                   Request timed out.
        APIConnectionError:        Network failure.
        APIError:                  Other provider error.
    """
    if _elc_core is None:
        raise RuntimeError("edgequake_litellm native extension is not installed. Run `pip install edgequake-litellm`.")

    provider, model_name = _parse_model(model)
    messages_json = _serialise_messages(messages)
    options_json = build_options(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        response_format=response_format,
        system_prompt=system,
    )

    try:
        return _elc_core.completion(
            provider,
            model_name,
            messages_json,
            options_json,
            _serialise_tools(tools),
            _serialise_tool_choice(tool_choice),
        )
    except Exception as exc:
        raise _map_builtin(exc, provider=provider, model=model_name) from exc


# ---------------------------------------------------------------------------
# Asynchronous completion
# ---------------------------------------------------------------------------

async def acompletion(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop: Optional[List[str]] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    response_format: Optional[str] = None,
    system: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    **kwargs: Any,
) -> ModelResponse:
    """Asynchronous version of :func:`completion`.

    Must be awaited inside an async context::

        resp = await acompletion("openai/gpt-4o-mini", messages)

    Args/Returns/Raises: same as :func:`completion`.
    """
    if _elc_core is None:
        raise RuntimeError("edgequake_litellm native extension is not installed.")

    provider, model_name = _parse_model(model)
    messages_json = _serialise_messages(messages)
    options_json = build_options(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        response_format=response_format,
        system_prompt=system,
    )

    try:
        return await _elc_core.acompletion(
            provider,
            model_name,
            messages_json,
            options_json,
            _serialise_tools(tools),
            _serialise_tool_choice(tool_choice),
        )
    except Exception as exc:
        raise _map_builtin(exc, provider=provider, model=model_name) from exc


__all__ = ["completion", "acompletion"]
