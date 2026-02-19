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
"""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List, Optional

from edgequake_litellm._types import StreamChunk
from edgequake_litellm.config import build_options
from edgequake_litellm.exceptions import _map_builtin

try:
    from edgequake_litellm import _elc_core  # type: ignore[import]
except ImportError:
    _elc_core = None  # type: ignore[assignment]


def _parse_model(model: str) -> tuple[str, str]:
    from edgequake_litellm.config import get_config
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return provider, model_name
    return get_config().default_provider, model


def _serialise_messages(messages: List[Dict[str, Any]]) -> str:
    normalised = []
    for msg in messages:
        entry: Dict[str, Any] = {"role": msg["role"], "content": msg.get("content", "")}
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
        max_tokens=max_tokens,
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
        chunks: List[StreamChunk] = await _elc_core.stream_completion(
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
