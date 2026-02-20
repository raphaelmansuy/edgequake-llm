"""
streaming.py — Streaming completion helpers.

Public functions
----------------
stream(model, messages, ...)       → AsyncGenerator[StreamChunk, None]

Usage
-----
>>> async for chunk in stream("openai/gpt-4o-mini", messages):
...     if chunk.content:
...         print(chunk.content, end="", flush=True)
>>>
>>> # litellm-style streaming via acompletion:
>>> async for chunk in acompletion("openai/gpt-4o-mini", messages, stream=True):
...     print(chunk.choices[0].delta.content or "", end="", flush=True)
"""
from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from edgequake_litellm._types import StreamChunk
from edgequake_litellm.config import build_options
from edgequake_litellm.exceptions import _map_builtin

try:
    from edgequake_litellm import _elc_core  # type: ignore[import-untyped]
except ImportError:
    _elc_core = None  # type: ignore[assignment]


def _parse_model(model: str) -> tuple[str, str]:
    from edgequake_litellm.config import get_config
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return provider, model_name
    return get_config().default_provider, model


def _serialise_messages(messages: list[dict[str, Any]]) -> str:
    normalised = []
    for msg in messages:
        entry: dict[str, Any] = {"role": msg["role"], "content": msg.get("content", "")}
        if "name" in msg:
            entry["name"] = msg["name"]
        if "tool_calls" in msg and msg["tool_calls"]:
            entry["tool_calls"] = msg["tool_calls"]
        if "tool_call_id" in msg:
            entry["tool_call_id"] = msg["tool_call_id"]
        normalised.append(entry)
    return json.dumps(normalised)


async def stream(
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int | None = None,
    max_completion_tokens: int | None = None,  # litellm alias
    temperature: float | None = None,
    top_p: float | None = None,
    stop: list[str] | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    response_format: str | None = None,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any | None = None,
    seed: int | None = None,
    user: str | None = None,
    timeout: float | int | None = None,
    api_base: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> AsyncGenerator[StreamChunk, None]:
    """Stream a completion, yielding :class:`StreamChunk` objects.

    Note: The underlying Rust call collects all chunks then releases them
    one-by-one here. True backpressure streaming will be added in a future
    release when pyo3 async iterators are stable.

    Args:
        model:    ``provider/model`` string.
        messages: List of ``{role, content}`` dicts.
        **kwargs: Same options as :func:`~edgequake_litellm.completion.completion`.

    Yields:
        StreamChunk — each yielded object has:
            .content (str | None): incremental text
            .thinking (str | None): reasoning text
            .is_finished (bool): True on the last chunk
            .finish_reason (str | None): stop reason when finished

    Example::

        text = ""
        async for chunk in stream("openai/gpt-4o-mini", messages):
            if chunk.content:
                text += chunk.content
        print(text)
    """
    if _elc_core is None:
        raise RuntimeError("edgequake_litellm native extension is not installed.")

    provider, model_name = _parse_model(model)
    messages_json = _serialise_messages(messages)
    options_json = build_options(
        max_tokens=max_tokens or max_completion_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        response_format=response_format,
        system_prompt=system,
    )

    tools_json = json.dumps(tools) if tools else None
    tc_json = json.dumps(tool_choice) if tool_choice else None

    try:
        chunks: list[StreamChunk] = await _elc_core.stream_completion(
            provider,
            model_name,
            messages_json,
            options_json,
            tools_json,
            tc_json,
        )
    except Exception as exc:
        raise _map_builtin(exc, provider=provider, model=model_name) from exc

    for chunk in chunks:
        yield chunk


__all__ = ["stream"]
