"""
embedding.py — LiteLLM-compatible embedding API.

Public functions
----------------
embedding(model, input, ...)       → List[List[float]]
aembedding(model, input, ...)      → Awaitable[List[List[float]]]

Usage
-----
>>> from edgequake_python import embedding
>>>
>>> vectors = embedding("openai/text-embedding-3-small", ["Hello world"])
>>> print(len(vectors), len(vectors[0]))  # 1 1536
"""
from __future__ import annotations

from typing import Any, List, Optional

from edgequake_python.exceptions import _map_builtin

try:
    from edgequake_python import _eq_core  # type: ignore[import]
except ImportError:
    _eq_core = None  # type: ignore[assignment]


def _parse_model(model: str) -> tuple[str, str]:
    from edgequake_python.config import get_config
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return provider, model_name
    return get_config().default_provider, model


def embedding(
    model: str,
    input: List[str],  # noqa: A002
    **kwargs: Any,
) -> List[List[float]]:
    """Generate embeddings synchronously.

    Args:
        model: ``provider/model`` string (e.g. ``"openai/text-embedding-3-small"``).
        input: List of texts to embed.
        **kwargs: Ignored (litellm drop_params).

    Returns:
        List of embedding vectors, one per input text.

    Raises:
        Same exceptions as :func:`~edgequake_python.completion.completion`.
    """
    if _eq_core is None:
        raise RuntimeError("edgequake_python native extension is not installed.")

    provider, model_name = _parse_model(model)
    try:
        return _eq_core.embed(provider, model_name, input)
    except Exception as exc:
        raise _map_builtin(exc, provider=provider, model=model_name) from exc


async def aembedding(
    model: str,
    input: List[str],  # noqa: A002
    **kwargs: Any,
) -> List[List[float]]:
    """Generate embeddings asynchronously.

    Args/Returns/Raises: same as :func:`embedding`.
    """
    if _eq_core is None:
        raise RuntimeError("edgequake_python native extension is not installed.")

    provider, model_name = _parse_model(model)
    try:
        return await _eq_core.aembed(provider, model_name, input)
    except Exception as exc:
        raise _map_builtin(exc, provider=provider, model=model_name) from exc


__all__ = ["embedding", "aembedding"]
