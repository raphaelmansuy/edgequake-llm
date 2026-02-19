"""
embedding.py — LiteLLM-compatible embedding API.

Public functions
----------------
embedding(model, input, ...)       → List[List[float]]
aembedding(model, input, ...)      → Awaitable[List[List[float]]]

Usage
-----
>>> from edgequake_litellm import embedding
>>>
>>> vectors = embedding("openai/text-embedding-3-small", ["Hello world"])
>>> print(len(vectors), len(vectors[0]))  # 1 1536
"""
from __future__ import annotations

from typing import Any, List, Optional

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
        Same exceptions as :func:`~edgequake_litellm.completion.completion`.
    """
    if _elc_core is None:
        raise RuntimeError("edgequake_litellm native extension is not installed.")

    provider, model_name = _parse_model(model)
    try:
        return _elc_core.embed(provider, model_name, input)
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
    if _elc_core is None:
        raise RuntimeError("edgequake_litellm native extension is not installed.")

    provider, model_name = _parse_model(model)
    try:
        return await _elc_core.aembed(provider, model_name, input)
    except Exception as exc:
        raise _map_builtin(exc, provider=provider, model=model_name) from exc


__all__ = ["embedding", "aembedding"]
