"""
completion.py — LiteLLM-compatible completion API.

Public functions
----------------
completion(model, messages, ...)       → ModelResponseCompat
acompletion(model, messages, ...)      → Awaitable[ModelResponseCompat]

Usage
-----
>>> from edgequake_litellm import completion
>>>
>>> resp = completion(
...     model="openai/gpt-4o-mini",
...     messages=[{"role": "user", "content": "Hello!"}],
... )
>>> print(resp.content)                        # edgequake shortcut
>>> print(resp.choices[0].message.content)     # litellm / OpenAI path
"""
from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from edgequake_litellm._compat import ModelResponseCompat, StreamChunkCompat
from edgequake_litellm._types import ModelResponse
from edgequake_litellm.config import build_options
from edgequake_litellm.exceptions import NotImplementedError as _NotImplementedError
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


def _normalise_rf(
    response_format: Optional[Union[str, Dict[str, Any]]]
) -> Optional[str]:
    """Normalise response_format to the string the Rust core expects."""
    if response_format is None:
        return None
    if isinstance(response_format, dict):
        return response_format.get("type") or "json_object"
    return response_format


# ---------------------------------------------------------------------------
# Synchronous completion
# ---------------------------------------------------------------------------

def completion(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,  # litellm alias for max_tokens
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop: Optional[List[str]] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    response_format: Optional[Union[str, Dict[str, Any]]] = None,
    system: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    stream: bool = False,
    seed: Optional[int] = None,
    user: Optional[str] = None,
    timeout: Optional[Union[float, int]] = None,
    api_base: Optional[str] = None,
    base_url: Optional[str] = None,  # alias for api_base
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> ModelResponseCompat:
    """Call an LLM provider and return the full response.

    The ``model`` argument follows the ``provider/model-name`` convention used
    by litellm::

        "openai/gpt-4o"
        "anthropic/claude-3-5-sonnet-20241022"
        "ollama/llama3.2"
        "gemini/gemini-2.0-flash"

    Args:
        model:                  ``provider/model`` or just ``model`` (uses default provider).
        messages:               List of ``{role, content}`` dicts.
        max_tokens:             Maximum tokens to generate.
        max_completion_tokens:  Alias for ``max_tokens`` (litellm v1 compat).
        temperature:            Sampling temperature.
        top_p:                  Nucleus sampling threshold.
        stop:                   Stop sequences.
        frequency_penalty:      Frequency penalty.
        presence_penalty:       Presence penalty.
        response_format:        ``"json_object"`` string or ``{"type": "json_object"}`` dict.
        system:                 System prompt override (edgequake extension).
        tools:                  List of tool definitions.
        tool_choice:            Tool choice configuration.
        stream:                 Must be ``False`` — sync streaming not supported.
                                Use ``acompletion(..., stream=True)`` or ``stream()`` async gen.
        seed:                   Random seed for deterministic outputs (silently forwarded).
        user:                   End-user identifier for abuse detection (silently forwarded).
        timeout:                Request timeout in seconds (not yet wired to Rust core).
        api_base:               Per-call base URL override (not yet wired to Rust core).
        base_url:               Alias for ``api_base``.
        api_key:                Per-call API key override (not yet wired to Rust core).
        **kwargs:               Extra params are ignored (litellm drop_params).

    Returns:
        :class:`ModelResponseCompat` — supports both ``resp.content`` and
        ``resp.choices[0].message.content`` access patterns.

    Raises:
        NotImplementedError:           When ``stream=True`` (use async streaming instead).
        AuthenticationError:           API key missing or invalid.
        RateLimitError:                Provider rate limit exceeded.
        ContextWindowExceededError:    Prompt too long.
        Timeout:                       Request timed out.
        APIConnectionError:            Network failure.
        APIError:                      Other provider error.
    """
    if stream:
        raise _NotImplementedError(
            "Synchronous streaming is not supported. "
            "Use acompletion(..., stream=True) or the stream() async generator."
        )

    if _elc_core is None:
        raise RuntimeError(
            "edgequake_litellm native extension is not installed. "
            "Run `pip install edgequake-litellm`."
        )

    # Resolve aliases
    effective_max_tokens = max_tokens or max_completion_tokens
    # api_base / api_key overrides accepted; Rust-level per-call override is on the roadmap.
    # Suppress "unused variable" lint: noqa assigned inline.
    _api_base = api_base or base_url  # noqa: F841

    # Normalise response_format: dict → string for Rust core
    rf = _normalise_rf(response_format)

    provider, model_name = _parse_model(model)
    messages_json = _serialise_messages(messages)
    options_json = build_options(
        max_tokens=effective_max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        response_format=rf,
        system_prompt=system,
    )

    t0 = time.perf_counter()
    try:
        raw = _elc_core.completion(
            provider,
            model_name,
            messages_json,
            options_json,
            _serialise_tools(tools),
            _serialise_tool_choice(tool_choice),
        )
    except Exception as exc:
        raise _map_builtin(exc, provider=provider, model=model_name) from exc

    response_ms = (time.perf_counter() - t0) * 1000
    return ModelResponseCompat(raw, response_ms=response_ms)


# ---------------------------------------------------------------------------
# Asynchronous completion
# ---------------------------------------------------------------------------

async def acompletion(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop: Optional[List[str]] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    response_format: Optional[Union[str, Dict[str, Any]]] = None,
    system: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    stream: bool = False,
    seed: Optional[int] = None,
    user: Optional[str] = None,
    timeout: Optional[Union[float, int]] = None,
    api_base: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> Union[ModelResponseCompat, AsyncGenerator[StreamChunkCompat, None]]:
    """Asynchronous version of :func:`completion`.

    When ``stream=True``, returns an async generator that yields
    :class:`~edgequake_litellm._compat.StreamChunkCompat` objects — supporting
    both ``chunk.content`` and ``chunk.choices[0].delta.content`` access::

        async for chunk in acompletion("openai/gpt-4o-mini", msgs, stream=True):
            print(chunk.choices[0].delta.content or "", end="")

    When ``stream=False`` (default), must be awaited::

        resp = await acompletion("openai/gpt-4o-mini", messages)
        print(resp.choices[0].message.content)

    Args/Returns/Raises: same as :func:`completion` plus ``stream`` kwarg.
    """
    if stream:
        # Delegate to the stream() async generator, wrapping each chunk
        from edgequake_litellm.streaming import stream as _stream_fn

        async def _wrapped_stream() -> AsyncGenerator[StreamChunkCompat, None]:
            async for raw_chunk in _stream_fn(
                model,
                messages,
                max_tokens=max_tokens or max_completion_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                response_format=_normalise_rf(response_format),
                system=system,
                tools=tools,
                tool_choice=tool_choice,
            ):
                yield StreamChunkCompat(raw_chunk)

        return _wrapped_stream()

    if _elc_core is None:
        raise RuntimeError("edgequake_litellm native extension is not installed.")

    # Resolve aliases
    effective_max_tokens = max_tokens or max_completion_tokens
    rf = _normalise_rf(response_format)

    provider, model_name = _parse_model(model)
    messages_json = _serialise_messages(messages)
    options_json = build_options(
        max_tokens=effective_max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        response_format=rf,
        system_prompt=system,
    )

    t0 = time.perf_counter()
    try:
        raw = await _elc_core.acompletion(
            provider,
            model_name,
            messages_json,
            options_json,
            _serialise_tools(tools),
            _serialise_tool_choice(tool_choice),
        )
    except Exception as exc:
        raise _map_builtin(exc, provider=provider, model=model_name) from exc

    response_ms = (time.perf_counter() - t0) * 1000
    return ModelResponseCompat(raw, response_ms=response_ms)


__all__ = ["completion", "acompletion"]
