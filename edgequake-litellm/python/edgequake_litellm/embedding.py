"""
embedding.py — LiteLLM-compatible embedding API.

Public functions
----------------
embedding(model, input, ...)       → EmbeddingResponseCompat
aembedding(model, input, ...)      → Awaitable[EmbeddingResponseCompat]

The returned object is litellm-compatible::

    result = embedding("openai/text-embedding-3-small", ["Hello world"])

    # litellm / OpenAI path
    vectors = [item.embedding for item in result.data]

    # legacy edgequake path (still works)
    vectors = list(result)   # iterates as List[List[float]]
    vectors = result[0]       # index returns List[float]
"""
from __future__ import annotations

from typing import Any

from edgequake_litellm._compat import EmbeddingResponseCompat
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


def embedding(
    model: str,
    input: list[str],  # noqa: A002
    user: str | None = None,
    dimensions: int | None = None,
    encoding_format: str | None = None,
    timeout: float | int | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> EmbeddingResponseCompat:
    """Generate embeddings synchronously.

    Args:
        model:           ``provider/model`` string (e.g. ``"openai/text-embedding-3-small"``).
        input:           List of texts to embed.
        user:            End-user identifier (forwarded to provider when supported).
        dimensions:      Desired embedding dimensions (silently dropped — roadmap item).
        encoding_format: ``"float"`` or ``"base64"`` (silently dropped — roadmap item).
        timeout:         Request timeout in seconds (silently dropped — roadmap item).
        api_base:        Per-call base URL override (silently dropped — roadmap item).
        api_key:         Per-call API key override (silently dropped — roadmap item).
        **kwargs:        Ignored (litellm drop_params).

    Returns:
        :class:`~edgequake_litellm._compat.EmbeddingResponseCompat` — supports both
        ``result.data[0].embedding`` and ``result[0]`` (legacy ``List[List[float]]``) access.

    Raises:
        Same exceptions as :func:`~edgequake_litellm.completion.completion`.
    """
    if _elc_core is None:
        raise RuntimeError("edgequake_litellm native extension is not installed.")

    provider, model_name = _parse_model(model)
    try:
        vectors: list[list[float]] = _elc_core.embed(provider, model_name, input)
    except Exception as exc:
        raise _map_builtin(exc, provider=provider, model=model_name) from exc

    return EmbeddingResponseCompat(vectors, model=f"{provider}/{model_name}")


async def aembedding(
    model: str,
    input: list[str],  # noqa: A002
    user: str | None = None,
    dimensions: int | None = None,
    encoding_format: str | None = None,
    timeout: float | int | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> EmbeddingResponseCompat:
    """Generate embeddings asynchronously.

    Args/Returns/Raises: same as :func:`embedding`.
    """
    if _elc_core is None:
        raise RuntimeError("edgequake_litellm native extension is not installed.")

    provider, model_name = _parse_model(model)
    try:
        vectors: list[list[float]] = await _elc_core.aembed(provider, model_name, input)
    except Exception as exc:
        raise _map_builtin(exc, provider=provider, model=model_name) from exc

    return EmbeddingResponseCompat(vectors, model=f"{provider}/{model_name}")


__all__ = ["embedding", "aembedding"]
