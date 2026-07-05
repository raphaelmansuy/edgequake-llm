"""
Model Discovery — programmatic model discovery across all providers.

Wraps the Rust ``ModelDiscoveryService`` to let Python code:
- Discover all available models across providers
- Filter by capabilities (vision, tools, thinking, context length, cost)
- Look up specific model metadata

Mirrors the ``litellm.model_list`` / ``litellm.get_model_info()`` API patterns.

Usage
-----
>>> from edgequake_litellm import discovery
>>>
>>> # Find all models
>>> models = discovery.discover_all()
>>>
>>> # Filter by capabilities
>>> vision_models = discovery.find_models(
...     requires_vision=True,
...     requires_tools=True,
...     min_context_length=100_000,
... )
>>>
>>> # Async variant
>>> models = await discovery.adiscover_all()
>>>
>>> # Look up specific model
>>> model = discovery.get_model("openai", "gpt-4.1")
>>> if model:
...     print(f"{model.id}: {model.context_length} tokens")
"""
from __future__ import annotations

import json
from typing import Any

try:
    from edgequake_litellm._elc_core import (  # type: ignore[import-untyped]
        DiscoveredModel,
        ModelSearchMatch,
    )
    from edgequake_litellm._elc_core import (
        adiscover_all as _adiscover_all,
    )
    from edgequake_litellm._elc_core import (
        afind_models as _afind_models,
    )
    from edgequake_litellm._elc_core import (
        aget_model as _aget_model,
    )
    from edgequake_litellm._elc_core import (
        discover_all as _discover_all,
    )
    from edgequake_litellm._elc_core import (
        discover_provider as _discover_provider,
    )
    from edgequake_litellm._elc_core import (
        discovery_info as _discovery_info,
    )
    from edgequake_litellm._elc_core import (
        find_models as _find_models,
    )
    from edgequake_litellm._elc_core import (
        find_static_models as _find_static_models,
    )
    from edgequake_litellm._elc_core import (
        get_model as _get_model,
    )
    from edgequake_litellm._elc_core import (
        list_discovery_providers as _list_discovery_providers,
    )
    from edgequake_litellm._elc_core import (
        lookup_model_by_name as _lookup_model_by_name,
    )
    from edgequake_litellm._elc_core import (
        search_models as _search_models,
    )
    from edgequake_litellm._elc_core import (
        search_static_models_by_name as _search_static_models_by_name,
    )
except ImportError:
    DiscoveredModel = None  # type: ignore[assignment, misc]
    ModelSearchMatch = None  # type: ignore[assignment, misc]

    def _discover_all() -> list:  # type: ignore[misc]
        return []

    def _find_models(filter_json: str | None = None) -> list:  # type: ignore[misc]
        return []

    def _find_static_models(filter_json: str | None = None) -> list:  # type: ignore[misc]
        return []

    def _get_model(provider: str, model_id: str) -> Any:  # type: ignore[misc]
        return None

    def _discover_provider(provider_id: str) -> list:  # type: ignore[misc]
        return []

    def _list_discovery_providers() -> list[str]:  # type: ignore[misc]
        return []

    def _discovery_info() -> list:  # type: ignore[misc]
        return []

    async def _adiscover_all() -> list:  # type: ignore[misc]
        return []

    async def _afind_models(filter_json: str | None = None) -> list:  # type: ignore[misc]
        return []

    async def _aget_model(provider: str, model_id: str) -> Any:  # type: ignore[misc]
        return None

    def _search_models(  # type: ignore[misc]
        query: str,
        provider: str | None = None,
        fuzzy: bool = False,
        min_score: float | None = None,
        limit: int | None = None,
        filter_json: str | None = None,
    ) -> list:
        return []

    def _search_static_models_by_name(  # type: ignore[misc]
        query: str,
        provider: str | None = None,
        fuzzy: bool = False,
        min_score: float | None = None,
        limit: int | None = None,
        filter_json: str | None = None,
    ) -> list:
        return []

    def _lookup_model_by_name(provider: str, name_or_id: str) -> Any:  # type: ignore[misc]
        return None


def _build_filter_json(**kwargs: Any) -> str | None:
    """Build a JSON filter string from keyword arguments."""
    filtered = {k: v for k, v in kwargs.items() if v is not None}
    if not filtered:
        return None
    return json.dumps(filtered)


# ─── Sync API ────────────────────────────────────────────────────────────────


def discover_all() -> list[Any]:
    """Discover all models across all registered providers.

    Returns a list of DiscoveredModel objects with normalized capabilities.
    Falls back to static registry data when API keys are not configured.
    """
    return _discover_all()


def find_models(
    *,
    min_context_length: int | None = None,
    max_context_length: int | None = None,
    min_output_tokens: int | None = None,
    max_output_tokens: int | None = None,
    requires_vision: bool | None = None,
    requires_tools: bool | None = None,
    requires_thinking: bool | None = None,
    requires_streaming: bool | None = None,
    requires_json_mode: bool | None = None,
    provider: str | None = None,
    tags: list[str] | None = None,
    max_cost_per_m_input: float | None = None,
    exclude_deprecated: bool | None = None,
) -> list[Any]:
    """Find models matching capability constraints (AND logic).

    All parameters are optional. Only non-None constraints are applied.

    Parameters
    ----------
    min_context_length : int, optional
        Minimum input context window in tokens.
    requires_vision : bool, optional
        Model must support image/vision input.
    requires_tools : bool, optional
        Model must support function/tool calling.
    requires_thinking : bool, optional
        Model must support extended thinking/reasoning.
    provider : str, optional
        Only return models from this provider.
    max_cost_per_m_input : float, optional
        Maximum cost per million input tokens (USD).
    exclude_deprecated : bool, optional
        Exclude deprecated models.
    """
    filter_json = _build_filter_json(
        min_context_length=min_context_length,
        max_context_length=max_context_length,
        min_output_tokens=min_output_tokens,
        max_output_tokens=max_output_tokens,
        requires_vision=requires_vision,
        requires_tools=requires_tools,
        requires_thinking=requires_thinking,
        requires_streaming=requires_streaming,
        requires_json_mode=requires_json_mode,
        provider=provider,
        tags=tags,
        max_cost_per_m_input=max_cost_per_m_input,
        exclude_deprecated=exclude_deprecated,
    )
    return _find_models(filter_json)


def find_static_models(
    *,
    min_context_length: int | None = None,
    max_context_length: int | None = None,
    min_output_tokens: int | None = None,
    max_output_tokens: int | None = None,
    requires_vision: bool | None = None,
    requires_tools: bool | None = None,
    requires_thinking: bool | None = None,
    requires_streaming: bool | None = None,
    requires_json_mode: bool | None = None,
    provider: str | None = None,
    tags: list[str] | None = None,
    max_cost_per_m_input: float | None = None,
    exclude_deprecated: bool | None = None,
) -> list[Any]:
    """Search the offline static registry by capability (no API calls)."""
    filter_json = _build_filter_json(
        min_context_length=min_context_length,
        max_context_length=max_context_length,
        min_output_tokens=min_output_tokens,
        max_output_tokens=max_output_tokens,
        requires_vision=requires_vision,
        requires_tools=requires_tools,
        requires_thinking=requires_thinking,
        requires_streaming=requires_streaming,
        requires_json_mode=requires_json_mode,
        provider=provider,
        tags=tags,
        max_cost_per_m_input=max_cost_per_m_input,
        exclude_deprecated=exclude_deprecated,
    )
    return _find_static_models(filter_json)


def get_model(provider: str, model_id: str) -> Any | None:
    """Look up a specific model by provider and model ID.

    Returns a DiscoveredModel or None if not found.
    """
    return _get_model(provider, model_id)


def discover_provider(provider_id: str) -> list[Any]:
    """Discover models from a specific provider only."""
    return _discover_provider(provider_id)


def list_discovery_providers() -> list[str]:
    """List all registered discovery provider IDs."""
    return _list_discovery_providers()


def search_models(
    query: str,
    *,
    provider: str | None = None,
    fuzzy: bool = False,
    min_score: float | None = None,
    limit: int | None = None,
    min_context_length: int | None = None,
    max_context_length: int | None = None,
    min_output_tokens: int | None = None,
    max_output_tokens: int | None = None,
    requires_vision: bool | None = None,
    requires_tools: bool | None = None,
    requires_thinking: bool | None = None,
    exclude_deprecated: bool | None = None,
) -> list[Any]:
    """Search models by name or ID across all providers (live discovery)."""
    filter_json = _build_filter_json(
        min_context_length=min_context_length,
        max_context_length=max_context_length,
        min_output_tokens=min_output_tokens,
        max_output_tokens=max_output_tokens,
        requires_vision=requires_vision,
        requires_tools=requires_tools,
        requires_thinking=requires_thinking,
        exclude_deprecated=exclude_deprecated,
    )
    return _search_models(query, provider, fuzzy, min_score, limit, filter_json)


def search_static_models_by_name(
    query: str,
    *,
    provider: str | None = None,
    fuzzy: bool = False,
    min_score: float | None = None,
    limit: int | None = None,
    min_context_length: int | None = None,
    max_context_length: int | None = None,
    min_output_tokens: int | None = None,
    max_output_tokens: int | None = None,
    requires_vision: bool | None = None,
    requires_tools: bool | None = None,
    requires_thinking: bool | None = None,
    exclude_deprecated: bool | None = None,
) -> list[Any]:
    """Search the offline static registry by name or ID (no API calls)."""
    filter_json = _build_filter_json(
        min_context_length=min_context_length,
        max_context_length=max_context_length,
        min_output_tokens=min_output_tokens,
        max_output_tokens=max_output_tokens,
        requires_vision=requires_vision,
        requires_tools=requires_tools,
        requires_thinking=requires_thinking,
        exclude_deprecated=exclude_deprecated,
    )
    return _search_static_models_by_name(
        query, provider, fuzzy, min_score, limit, filter_json
    )


def lookup_model_by_name(provider: str, name_or_id: str) -> Any | None:
    """Resolve a model by exact ID or display name in the static registry."""
    return _lookup_model_by_name(provider, name_or_id)


def discovery_info() -> list[dict[str, str]]:
    """Get information about discovery strategies per provider."""
    return _discovery_info()


# ─── Async API ───────────────────────────────────────────────────────────────


async def adiscover_all() -> list[Any]:
    """Discover all models across all registered providers (async)."""
    return await _adiscover_all()


async def afind_models(
    *,
    min_context_length: int | None = None,
    max_context_length: int | None = None,
    min_output_tokens: int | None = None,
    max_output_tokens: int | None = None,
    requires_vision: bool | None = None,
    requires_tools: bool | None = None,
    requires_thinking: bool | None = None,
    requires_streaming: bool | None = None,
    requires_json_mode: bool | None = None,
    provider: str | None = None,
    tags: list[str] | None = None,
    max_cost_per_m_input: float | None = None,
    exclude_deprecated: bool | None = None,
) -> list[Any]:
    """Find models matching capability constraints (async, AND logic)."""
    filter_json = _build_filter_json(
        min_context_length=min_context_length,
        max_context_length=max_context_length,
        min_output_tokens=min_output_tokens,
        max_output_tokens=max_output_tokens,
        requires_vision=requires_vision,
        requires_tools=requires_tools,
        requires_thinking=requires_thinking,
        requires_streaming=requires_streaming,
        requires_json_mode=requires_json_mode,
        provider=provider,
        tags=tags,
        max_cost_per_m_input=max_cost_per_m_input,
        exclude_deprecated=exclude_deprecated,
    )
    return await _afind_models(filter_json)


async def aget_model(provider: str, model_id: str) -> Any | None:
    """Look up a specific model by provider and model ID (async)."""
    return await _aget_model(provider, model_id)


# ─── Convenience: litellm-compatible model_list ──────────────────────────────


def get_model_info(model: str) -> dict[str, Any] | None:
    """Get model info using litellm's `provider/model` convention.

    >>> info = get_model_info("openai/gpt-4.1")
    >>> info["context_length"]
    1048576
    """
    if "/" in model:
        provider, model_id = model.split("/", 1)
    else:
        provider = "openai"
        model_id = model

    result = get_model(provider, model_id)
    if result is None:
        return None
    return result.to_dict()


__all__ = [
    "DiscoveredModel",
    "ModelSearchMatch",
    "adiscover_all",
    "afind_models",
    "aget_model",
    "discover_all",
    "discover_provider",
    "discovery_info",
    "find_models",
    "find_static_models",
    "get_model",
    "get_model_info",
    "list_discovery_providers",
    "lookup_model_by_name",
    "search_models",
    "search_static_models_by_name",
]
