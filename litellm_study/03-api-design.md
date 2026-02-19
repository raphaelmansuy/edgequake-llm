# Python API Design: Classes, Modules, Type Stubs

## 1. Public API Contract

The surface must be a **drop-in replacement** for the following litellm imports:

```python
import edgequake_python as litellm

# or, for true drop-in:
import edgequake_python
import sys
sys.modules["litellm"] = edgequake_python
```

---

## 2. Top-Level `__init__.py`

```python
# edgequake_python/__init__.py

from __future__ import annotations

from .completion import completion, acompletion
from .embedding import embedding, aembedding
from .exceptions import (
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    ServiceUnavailableError,
    ContextWindowExceededError,
    ContentPolicyViolationError,
    Timeout,
    APIConnectionError,
    InternalServerError,
    APIError,
    APIResponseValidationError,
)
from ._types import (
    ModelResponse,
    EmbeddingResponse,
    Message,
    Choices,
    Delta,
    Usage,
    StreamingChoices,
    ChatCompletionMessageToolCall,
    Function,
)
from .config import (
    drop_params,
    num_retries,
    request_timeout,
    # per-provider keys
    openai_key,
    anthropic_key,
    gemini_key,
    mistral_key,
    xai_key,
    openrouter_key,
    # base url overrides
    api_base,
    api_version,
)

__version__ = "0.1.0"
__all__ = [
    "completion", "acompletion",
    "embedding", "aembedding",
    "AuthenticationError", "RateLimitError", "BadRequestError",
    "ContextWindowExceededError", "Timeout", "APIConnectionError",
    "ModelResponse", "EmbeddingResponse", "Message", "Usage",
]
```

---

## 3. Type Definitions (`_types.py`)

```python
# edgequake_python/_types.py
from __future__ import annotations
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
import time


class Function(BaseModel):
    name: str
    arguments: str


class ChatCompletionMessageToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Function


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class Delta(BaseModel):
    """Used in streaming responses."""
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Extended fields (thinking models)
    prompt_tokens_details: Optional[Dict[str, Any]] = None
    completion_tokens_details: Optional[Dict[str, Any]] = None


class Choices(BaseModel):
    finish_reason: Optional[str] = None
    index: int = 0
    message: Message
    logprobs: Optional[Any] = None


class StreamingChoices(BaseModel):
    finish_reason: Optional[str] = None
    index: int = 0
    delta: Delta
    logprobs: Optional[Any] = None


class ModelResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-eq-{int(time.time())}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[Choices] = []
    usage: Usage = Field(default_factory=Usage)
    system_fingerprint: Optional[str] = None
    # edgequake extensions
    _hidden_params: Dict[str, Any] = {}

    # litellm-compat helpers
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    @property
    def content(self) -> Optional[str]:
        """Shortcut to first choice content."""
        if self.choices:
            return self.choices[0].message.content
        return None


class StreamingModelResponse(BaseModel):
    """Streaming delta response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-eq-{int(time.time())}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[StreamingChoices] = []
    usage: Optional[Usage] = None  # only on last chunk


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int = 0


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingObject] = []
    model: str = ""
    usage: Usage = Field(default_factory=Usage)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
```

---

## 4. Completion Functions (`completion.py`)

```python
# edgequake_python/completion.py
from __future__ import annotations

import os
import json
from typing import Optional, List, Dict, Any, Union, Iterator, AsyncIterator

from ._eq_core import complete as _rust_complete, stream as _rust_stream
from ._types import ModelResponse, StreamingModelResponse, Message, Choices, Delta, Usage
from .exceptions import _rust_error_to_python
from .streaming import SyncStreamWrapper, AsyncStreamWrapper
from . import config as _config


def completion(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    # Generation params
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    seed: Optional[int] = None,
    logprobs: Optional[bool] = None,
    # Tool calling
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    parallel_tool_calls: Optional[bool] = None,
    # Output format
    response_format: Optional[Dict[str, Any]] = None,
    # Streaming
    stream: bool = False,
    stream_options: Optional[Dict[str, Any]] = None,
    # Provider overrides
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    # Retry / timeout
    timeout: Optional[float] = None,
    num_retries: Optional[int] = None,
    # Vision
    max_image_size: Optional[int] = None,
    # Misc
    user: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    drop_params: Optional[bool] = None,
    **kwargs: Any,
) -> Union[ModelResponse, SyncStreamWrapper]:
    """
    LiteLLM-compatible completion call backed by edgequake-llm (Rust).

    Examples
    --------
    >>> response = completion(
    ...     model="openai/gpt-4o",
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ... )
    >>> print(response.choices[0].message.content)
    """
    provider, model_name = _parse_model(model)
    resolved_key = api_key or _config._get_key(provider)
    resolved_base = api_base or _config._get_base(provider)

    opts = _build_opts(
        temperature=temperature,
        max_tokens=max_tokens or max_completion_tokens,
        top_p=top_p,
        stop=stop,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        seed=seed,
        timeout=timeout,
        **kwargs,
    )

    if stream:
        rust_iter = _rust_stream(
            provider=provider,
            model=model_name,
            messages=json.dumps(messages),
            opts=json.dumps(opts),
            api_key=resolved_key or "",
            api_base=resolved_base or "",
        )
        return SyncStreamWrapper(rust_iter, model=model)
    else:
        try:
            raw = _rust_complete(
                provider=provider,
                model=model_name,
                messages=json.dumps(messages),
                opts=json.dumps(opts),
                api_key=resolved_key or "",
                api_base=resolved_base or "",
            )
        except Exception as e:
            raise _rust_error_to_python(e) from e
        return _to_model_response(raw, model=model)


async def acompletion(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    stream: bool = False,
    **kwargs: Any,
) -> Union[ModelResponse, AsyncStreamWrapper]:
    """Async version of completion()."""
    import asyncio
    loop = asyncio.get_event_loop()
    if stream:
        # Return async stream wrapper
        provider, model_name = _parse_model(model)
        # ... (same as sync but uses async Rust future via pyo3-asyncio)
        from ._eq_core import astream as _rust_astream
        rust_aiter = await _rust_astream(...)
        return AsyncStreamWrapper(rust_aiter, model=model)
    else:
        return await loop.run_in_executor(
            None,
            lambda: completion(model, messages, **kwargs)
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_model(model: str) -> tuple[str, str]:
    """
    "openai/gpt-4o"        → ("openai", "gpt-4o")
    "anthropic/claude-3.5" → ("anthropic", "claude-3.5")
    "gpt-4o"               → ("openai", "gpt-4o")   # default provider
    """
    if "/" in model:
        provider, _, model_name = model.partition("/")
        return provider.lower(), model_name
    # Fallback heuristics
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
        return "openai", model
    if model.startswith("claude-"):
        return "anthropic", model
    if model.startswith("gemini"):
        return "gemini", model
    if model.startswith("mistral") or model.startswith("mixtral"):
        return "mistral", model
    return "openai", model  # default


def _build_opts(**kwargs) -> dict:
    """Filter None values and map to Rust-expected keys."""
    return {k: v for k, v in kwargs.items() if v is not None}


def _to_model_response(raw, model: str) -> ModelResponse:
    """Convert PyLLMResponse (from Rust) to ModelResponse."""
    return ModelResponse(
        model=model,
        choices=[
            Choices(
                finish_reason=raw.finish_reason or "stop",
                index=0,
                message=Message(
                    role="assistant",
                    content=raw.content,
                    tool_calls=_parse_tool_calls(raw.tool_calls),
                ),
            )
        ],
        usage=Usage(
            prompt_tokens=raw.prompt_tokens,
            completion_tokens=raw.completion_tokens,
            total_tokens=raw.total_tokens,
        ),
    )
```

---

## 5. Streaming Wrappers (`streaming.py`)

```python
# edgequake_python/streaming.py
from __future__ import annotations
from typing import Iterator, AsyncIterator, Optional
from ._types import StreamingModelResponse, StreamingChoices, Delta, Usage


class SyncStreamWrapper:
    """
    Wraps PyAsyncStreamIter (Rust) to provide a synchronous Python iterator.
    Yields StreamingModelResponse objects identical to litellm streaming.

    Usage:
        for chunk in completion(..., stream=True):
            text = chunk.choices[0].delta.content or ""
            print(text, end="", flush=True)
    """
    def __init__(self, rust_iter, model: str):
        self._iter = rust_iter
        self._model = model

    def __iter__(self) -> Iterator[StreamingModelResponse]:
        return self

    def __next__(self) -> StreamingModelResponse:
        chunk = next(self._iter)          # blocks; GIL released in Rust
        if chunk is None:
            raise StopIteration
        return self._to_chunk(chunk)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def _to_chunk(self, raw) -> StreamingModelResponse:
        content = getattr(raw, "content", None)
        finish_reason = getattr(raw, "finish_reason", None)
        usage = None
        if finish_reason:
            usage = Usage(
                prompt_tokens=getattr(raw, "prompt_tokens", 0),
                completion_tokens=getattr(raw, "completion_tokens", 0),
                total_tokens=getattr(raw, "total_tokens", 0),
            )
        return StreamingModelResponse(
            model=self._model,
            choices=[
                StreamingChoices(
                    index=0,
                    finish_reason=finish_reason,
                    delta=Delta(content=content),
                )
            ],
            usage=usage,
        )


class AsyncStreamWrapper:
    """Async version — works with `async for chunk in acompletion(..., stream=True)`."""
    def __init__(self, rust_aiter, model: str):
        self._iter = rust_aiter
        self._model = model

    def __aiter__(self) -> AsyncIterator[StreamingModelResponse]:
        return self

    async def __anext__(self) -> StreamingModelResponse:
        chunk = await self._iter.__anext__()
        if chunk is None:
            raise StopAsyncIteration
        return SyncStreamWrapper._to_chunk(self, chunk)
```

---

## 6. Exception Hierarchy (`exceptions.py`)

```python
# edgequake_python/exceptions.py
from __future__ import annotations


class EQError(Exception):
    """Base class for edgequake-python exceptions."""
    status_code: int = 500
    def __init__(self, message: str = "", status_code: int = 500, **kwargs):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


# Make subclasses also inherit openai.OpenAIError when openai is installed
try:
    from openai import OpenAIError as _OAIError
    _base = (EQError, _OAIError)
except ImportError:
    _base = (EQError,)


class AuthenticationError(*_base):
    status_code = 401

class BadRequestError(*_base):
    status_code = 400

class RateLimitError(*_base):
    status_code = 429

class ServiceUnavailableError(*_base):
    status_code = 503

class ContextWindowExceededError(BadRequestError):
    status_code = 400

class ContentPolicyViolationError(BadRequestError):
    status_code = 400

class Timeout(*_base):
    status_code = 408

class APIConnectionError(*_base):
    status_code = 0

class InternalServerError(*_base):
    status_code = 500

class APIError(*_base):
    status_code = 500

class APIResponseValidationError(*_base):
    status_code = 422


_RUST_ERROR_MAP = {
    "Authentication":       AuthenticationError,
    "RateLimit":            RateLimitError,
    "ContextWindow":        ContextWindowExceededError,
    "InvalidRequest":       BadRequestError,
    "Timeout":              Timeout,
    "Network":              APIConnectionError,
    "ServiceUnavailable":   ServiceUnavailableError,
}

def _rust_error_to_python(err: Exception) -> EQError:
    """Convert a PyO3 RuntimeError from Rust to the correct EQ exception."""
    msg = str(err)
    for key, cls in _RUST_ERROR_MAP.items():
        if key in msg:
            return cls(msg)
    return APIError(msg)
```

---

## 7. Global Config (`config.py`)

```python
# edgequake_python/config.py
"""
litellm-compatible global config variables.

    import edgequake_python as eq
    eq.openai_key = "sk-..."
    eq.anthropic_key = "sk-ant-..."
    eq.num_retries = 3
"""
from __future__ import annotations
import os
from typing import Optional

# Per-provider API keys
openai_key: Optional[str] = None
anthropic_key: Optional[str] = None
gemini_key: Optional[str] = None
mistral_key: Optional[str] = None
xai_key: Optional[str] = None
openrouter_key: Optional[str] = None
ollama_key: Optional[str] = None

# Base URL overrides
api_base: Optional[str] = None

# Behaviour
num_retries: int = 3
request_timeout: float = 600.0
drop_params: bool = False

_PROVIDER_KEY_MAP = {
    "openai":      ("openai_key",      "OPENAI_API_KEY"),
    "anthropic":   ("anthropic_key",   "ANTHROPIC_API_KEY"),
    "gemini":      ("gemini_key",       "GEMINI_API_KEY"),
    "mistral":     ("mistral_key",      "MISTRAL_API_KEY"),
    "xai":         ("xai_key",          "XAI_API_KEY"),
    "openrouter":  ("openrouter_key",   "OPENROUTER_API_KEY"),
}

_PROVIDER_BASE_MAP = {
    "ollama":    ("",  "OLLAMA_API_BASE",    "http://localhost:11434"),
    "lmstudio":  ("",  "LM_STUDIO_API_BASE", "http://localhost:1234"),
}

def _get_key(provider: str) -> Optional[str]:
    if provider not in _PROVIDER_KEY_MAP:
        return None
    attr, env = _PROVIDER_KEY_MAP[provider]
    return globals().get(attr) or os.getenv(env)

def _get_base(provider: str) -> Optional[str]:
    if provider not in _PROVIDER_BASE_MAP:
        return None
    attr, env, default = _PROVIDER_BASE_MAP[provider]
    return globals().get(attr) or os.getenv(env, default)
```

---

## 8. Type Stub File (`edgequake_python.pyi`)

```python
# edgequake_python/__init__.pyi
from typing import overload, Union, List, Dict, Any, Optional, AsyncIterator, Iterator
from ._types import ModelResponse, EmbeddingResponse, StreamingModelResponse

@overload
def completion(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    stream: "Literal[False]" = ...,
    **kwargs: Any,
) -> ModelResponse: ...

@overload
def completion(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    stream: "Literal[True]",
    **kwargs: Any,
) -> Iterator[StreamingModelResponse]: ...

async def acompletion(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    stream: bool = False,
    **kwargs: Any,
) -> Union[ModelResponse, AsyncIterator[StreamingModelResponse]]: ...

def embedding(
    model: str,
    input: Union[str, List[str]],
    **kwargs: Any,
) -> EmbeddingResponse: ...

async def aembedding(
    model: str,
    input: Union[str, List[str]],
    **kwargs: Any,
) -> EmbeddingResponse: ...

# Global config vars
openai_key: Optional[str]
anthropic_key: Optional[str]
num_retries: int
drop_params: bool
```
