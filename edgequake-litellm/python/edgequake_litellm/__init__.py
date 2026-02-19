"""
edgequake_litellm — Drop-in LiteLLM replacement backed by Rust.

``edgequake_litellm`` exposes *exactly* the same primary API as ``litellm`` so
existing projects can switch with a one-line import change:

.. code-block:: python

    # Before
    import litellm

    # After (same API, ~10× faster HTTP path)
    import edgequake_litellm as litellm

Backed by ``edgequake-llm`` (Rust + tokio + reqwest) through PyO3 bindings.
No runtime Python dependencies.  Single abi3 wheel covers Python 3.9–3.13+.

Quick start
-----------
>>> import edgequake_litellm as litellm
>>>
>>> # Sync chat (identical to litellm.completion)
>>> resp = litellm.completion(
...     model="openai/gpt-4o-mini",
...     messages=[{"role": "user", "content": "Hello!"}],
... )
>>> print(resp.content)
>>>
>>> # Async chat
>>> resp = await litellm.acompletion("anthropic/claude-3-5-haiku-20241022", messages)
>>>
>>> # Streaming (async generator)
>>> async for chunk in litellm.stream("ollama/llama3.2", messages):
...     print(chunk.content or "", end="", flush=True)
>>>
>>> # Embeddings
>>> vecs = litellm.embedding("openai/text-embedding-3-small", ["Hello world"])

Provider routing
----------------
Use the ``provider/model`` convention — same as litellm::

    "openai/gpt-4o"
    "anthropic/claude-3-5-sonnet-20241022"
    "gemini/gemini-2.0-flash"
    "mistral/mistral-large-latest"
    "openrouter/meta-llama/llama-3.1-70b-instruct"
    "xai/grok-beta"
    "ollama/llama3.2"
    "lmstudio/local-model"
    "azure_openai/gpt-4o"
    "huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1"
    "mock/any"  # testing without API keys

Environment variables
---------------------
``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``, etc. are read
from the environment, same as litellm.  Use ``LITELLM_EDGE_PROVIDER`` /
``LITELLM_EDGE_MODEL`` to set defaults.
"""
from __future__ import annotations

from edgequake_litellm.completion import acompletion, completion
from edgequake_litellm.embedding import aembedding, embedding
from edgequake_litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    ContextWindowExceededError,
    EdgeQuakeError,  # backward-compat alias
    LiteLLMEdgeError,
    ModelNotFoundError,
    RateLimitError,
    Timeout,
)
from edgequake_litellm.streaming import stream
from edgequake_litellm._types import ModelResponse, StreamChunk, ToolCall, Usage
from edgequake_litellm.config import (
    EdgeQuakeConfig,   # backward-compat alias
    LiteLLMEdgeConfig,
    build_options,
    get_config,
    set_default_model,
    set_default_provider,
)

try:
    from edgequake_litellm._elc_core import __version__  # type: ignore[import]
except ImportError:
    __version__ = "0.0.0-dev"

__all__ = [
    # ── Core functions ── same names as litellm ────────────────────────────
    "completion",
    "acompletion",
    "stream",
    "embedding",
    "aembedding",
    # ── Response types ─────────────────────────────────────────────────────
    "ModelResponse",
    "StreamChunk",
    "ToolCall",
    "Usage",
    # ── Exceptions ─ litellm-compatible names ──────────────────────────────
    "LiteLLMEdgeError",
    "EdgeQuakeError",       # backward-compat
    "AuthenticationError",
    "RateLimitError",
    "ContextWindowExceededError",
    "ModelNotFoundError",
    "Timeout",
    "APIConnectionError",
    "APIError",
    # ── Config ─────────────────────────────────────────────────────────────
    "LiteLLMEdgeConfig",
    "EdgeQuakeConfig",      # backward-compat
    "get_config",
    "set_default_provider",
    "set_default_model",
    "build_options",
    # ── Version ────────────────────────────────────────────────────────────
    "__version__",
]

