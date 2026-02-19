"""
edgequake_python — High-performance, LiteLLM-compatible LLM library.

Backed by edgequake-llm (Rust) through PyO3 bindings. Provides the same
primary API surface as litellm so existing projects can migrate with minimal
changes.

Quick start
-----------
>>> from edgequake_python import completion, acompletion, embedding, stream
>>>
>>> # Sync chat
>>> resp = completion("openai/gpt-4o-mini", [{"role": "user", "content": "Hi"}])
>>> print(resp.content)
>>>
>>> # Async chat
>>> resp = await acompletion("anthropic/claude-3-5-haiku-20241022", messages)
>>>
>>> # Streaming
>>> async for chunk in stream("ollama/llama3.2", messages):
...     print(chunk.content or "", end="")
>>>
>>> # Embeddings
>>> vecs = embedding("openai/text-embedding-3-small", ["Hello world"])

Provider routing
----------------
Use the ``provider/model`` convention::

    "openai/gpt-4o"
    "anthropic/claude-3-5-sonnet-20241022"
    "gemini/gemini-2.0-flash"
    "mistral/mistral-large-latest"
    "openrouter/meta-llama/llama-3.1-70b-instruct"
    "xai/grok-beta"
    "ollama/llama3.2"
    "lmstudio/local-model"
    "huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1"
    "mock/any"  (for testing without API keys)
"""
from __future__ import annotations

from edgequake_python.completion import acompletion, completion
from edgequake_python.embedding import aembedding, embedding
from edgequake_python.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    ContextWindowExceededError,
    EdgeQuakeError,
    ModelNotFoundError,
    RateLimitError,
    Timeout,
)
from edgequake_python.streaming import stream
from edgequake_python._types import ModelResponse, StreamChunk, ToolCall, Usage
from edgequake_python.config import (
    EdgeQuakeConfig,
    get_config,
    set_default_model,
    set_default_provider,
)

try:
    from edgequake_python._eq_core import __version__  # type: ignore[import]
except ImportError:
    __version__ = "0.0.0-dev"

__all__ = [
    # Core functions — same names as litellm
    "completion",
    "acompletion",
    "stream",
    "embedding",
    "aembedding",
    # Types
    "ModelResponse",
    "StreamChunk",
    "ToolCall",
    "Usage",
    # Exceptions — litellm-compatible names
    "EdgeQuakeError",
    "AuthenticationError",
    "RateLimitError",
    "ContextWindowExceededError",
    "ModelNotFoundError",
    "Timeout",
    "APIConnectionError",
    "APIError",
    # Config
    "EdgeQuakeConfig",
    "get_config",
    "set_default_provider",
    "set_default_model",
    # Version
    "__version__",
]
