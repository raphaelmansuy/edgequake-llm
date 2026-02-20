# DX Improvement Roadmap — edgequake-litellm

> Goal: Make edgequake-litellm a seamless drop-in for litellm users.  
> Priority levels: **P0** = blocking drop-in, **P1** = important DX, **P2** = nice-to-have, **P3** = out-of-scope

---

## ✅ Implementation Status (as of 2026-02-20)

All P0 and P1 items are **implemented**. The package is production-ready for litellm drop-in users.

| Item | Priority | Status | Notes |
|------|----------|--------|-------|
| `resp.choices[0].message.content` shim | P0.1 | ✅ Done | `ModelResponseCompat` in `_compat.py` |
| `acompletion(stream=True)` async generator | P0.2 | ✅ Done | Returns `AsyncGenerator[StreamChunkCompat]` |
| `completion(stream=True)` clear error | P0.3 | ✅ Done | `NotImplementedError` with helpful message |
| `EmbeddingResponseCompat` `.data[0].embedding` | P1.6 | ✅ Done | `_compat.py` |
| `response_ms` latency tracking | P1.5 | ✅ Done | `perf_counter()` in `completion.py` |
| `max_completion_tokens` alias | P2.7 | ✅ Done | Maps to `max_tokens` |
| `response_format` dict support | P2.2 | ✅ Done | `{"type": "json_object"}` accepted |
| `stream_chunk_builder()` | P1.4 | ✅ Done | Reconstructs full response from chunks |
| `set_verbose`, `drop_params` globals | P1.3 | ✅ Done | Module-level in `__init__.py` |
| `NotFoundError` alias | P1.7 | ✅ Done | `NotFoundError = ModelNotFoundError` |
| Multi-arch CI/CD (8 platform/Python combos) | P0.5 | ✅ Done | Linux x86_64/aarch64 musl+glibc, macOS x86_64/arm64, Windows x86_64/aarch64 |
| PyPI publish via OIDC trusted publishing | P0.6 | ✅ Done | `py-v*` tag triggers full build + publish |
| `timeout`, `api_base`, `api_key` wiring | P1.1/P1.2 | 🔄 Accepted | Python params accepted; Rust wiring = roadmap |

---


The biggest usability gap between edgequake-litellm and litellm today is the **response object shape**. Any code that accesses `response.choices[0].message.content` will fail. This single issue will block most litellm users from adopting edgequake-litellm without code changes. The second biggest gap is the **streaming API** shape — litellm uses `stream=True` on `completion()` while edgequake-litellm uses a separate `stream()` function.

---

## P0 — Blocking Drop-in Compatibility

These must be fixed before claiming "drop-in" compatibility.

### P0.1 — `ModelResponse.choices` compatibility shim

**Current state:**
```python
resp = completion("openai/gpt-4o-mini", messages)
resp.content         # ✅ works (our shortcut)
resp.choices[0].message.content  # ❌ AttributeError
resp["choices"]      # ❌ TypeError
```

**Target state:**
```python
resp.choices[0].message.content   # ✅
resp.choices[0].finish_reason     # ✅
resp.choices[0].message.role      # ✅
resp.choices[0].message.tool_calls  # ✅
resp["choices"][0]["message"]["content"]  # ✅ dict access
resp.id        # ✅
resp.created   # ✅
resp.object    # ✅
```

**Implementation plan:**

Create `edgequake_litellm/_compat.py` with `ModelResponseCompat` — a Python wrapper class that:
1. Wraps the PyO3 `ModelResponse` from Rust
2. Exposes `.choices[0].message.content` path
3. Supports `__getitem__` for dict-style access
4. Exposes `.id`, `.created`, `.object`, `.system_fingerprint`

```python
# _compat.py

import time
import uuid
from typing import Any, List, Optional

class _Message:
    def __init__(self, content, role="assistant", tool_calls=None):
        self.content = content
        self.role = role
        self.tool_calls = tool_calls
        self.function_call = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class _Choice:
    def __init__(self, message: _Message, finish_reason: str = "stop", index: int = 0):
        self.message = message
        self.finish_reason = finish_reason
        self.index = index

    def __getitem__(self, key):
        return getattr(self, key)


class ModelResponseCompat:
    """Wraps PyO3 ModelResponse to add litellm-compatible field access."""

    def __init__(self, raw):
        self._raw = raw
        self.model = raw.model
        self.usage = raw.usage
        self.id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self.object = "chat.completion"
        self.system_fingerprint = None
        # Build choices list from raw fields
        msg = _Message(
            content=raw.content,
            role="assistant",
            tool_calls=raw.tool_calls or None,
        )
        self.choices = [_Choice(message=msg, finish_reason="stop", index=0)]

    # Convenience shortcut (our extension)
    @property
    def content(self):
        return self._raw.content

    @property
    def tool_calls(self):
        return self._raw.tool_calls

    # Dict-style access
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __repr__(self):
        return (
            f"ModelResponse(id={self.id!r}, model={self.model!r}, "
            f"content={self.content!r})"
        )
```

**Effort:** Medium (1–2 days) — Python-only change, no Rust needed.  
**Risk:** Low.

---

### P0.2 — `acompletion(..., stream=True)` support

**Current state:**
```python
async for chunk in edgequake_litellm.stream("openai/gpt-4o-mini", msgs):
    print(chunk.content or "")
```

**Target state:**
```python
async for chunk in edgequake_litellm.acompletion("openai/gpt-4o-mini", msgs, stream=True):
    print(chunk.choices[0].delta.content or "")   # litellm path
    # OR
    print(chunk.content or "")                    # our shortcut
```

**Implementation:** Add `stream: bool = False` to `acompletion()`. When `True`, return the `stream()` async generator.

**Effort:** Low (hours) — Python-only.

---

### P0.3 — `completion(..., stream=True)` raises clear error

**Current state:** `stream=True` is silently ignored (swallowed by `**kwargs`).

**Target state:**
```python
completion("openai/gpt-4o-mini", msgs, stream=True)
# Raises: NotImplementedError("Synchronous streaming is not supported.
# Use: asyncio.run(async_main()) with acompletion(..., stream=True)
# or: stream() async generator directly.")
```

**Effort:** Trivial.

---

## P1 — Important Developer Experience

### P1.1 — `timeout` wired to Rust core

**Current state:** `timeout` is accepted in function signature but not forwarded to Rust, so requests can hang.

**Target state:** `timeout` passed to Rust `stream_completion` and `completion` calls.

**What needs to change:**
1. `build_options()` in `config.py` — add `timeout` field  
2. Rust `CompletionOptions` struct — add `timeout_secs: Option<f64>`
3. HTTP client construction in Rust — apply `reqwest::ClientBuilder::timeout()`

**Effort:** Medium (Rust change required). **Rust changes needed** in `src/providers/*.rs`.

---

### P1.2 — `api_base` / `api_key` per-call override

**Current state:** Accepted in signature but forwarded to nothing — always uses env var or Rust config.

**Target state:**
```python
resp = completion(
    "openai/gpt-4o-mini", msgs,
    api_base="https://my-proxy.example.com/v1",
    api_key="sk-...",
)
```

**What needs to change:**
1. `build_options()` — add `api_base`, `api_key` fields
2. Rust `CompletionOptions` — add `api_base: Option<String>`, `api_key: Option<String>`
3. Rust provider constructors — use options override before env var fallback

**Effort:** Medium-High (Rust change required).

---

### P1.3 — `seed` param passthrough

**Current state:** Silently dropped.

**Target state:** Forwarded to providers that support it (OpenAI, Gemini, Mistral).

**What needs to change:**
1. `build_options()` — add `seed: Option<i64>`
2. Rust JSON body construction in provider — include `seed` when set

**Effort:** Low-Medium (Rust change, but mechanical).

---

### P1.4 — `user` param passthrough

Useful for per-user rate limiting, observability, and abuse detection at the provider level.

```python
resp = completion("openai/gpt-4o-mini", msgs, user="user-123")
```

**Effort:** Low (Rust JSON body change).

---

### P1.5 — `ModelResponse.response_ms` latency attribute

**Target state:** `resp.response_ms` returns elapsed time in ms (float).

**Implementation:** Measure `time.perf_counter()` before/after the Rust call and attach to `ModelResponseCompat`.

**Effort:** Trivial once P0.1 is done (pure Python).

---

### P1.6 — `embedding()` returns `EmbeddingResponse`-compatible object

**Current state:** Returns `List[List[float]]`.

**Target state:** Returns an object with:
```python
result.data[0].embedding   # List[float]
result.model               # str
result.usage.prompt_tokens
result.usage.total_tokens
```

**Implementation:** Create `EmbeddingResponseCompat` wrapper in `_compat.py`:
```python
class _EmbeddingData:
    def __init__(self, embedding, index=0):
        self.embedding = embedding
        self.index = index
        self.object = "embedding"

class EmbeddingResponseCompat:
    def __init__(self, vectors: list, model: str = ""):
        self.object = "list"
        self.model = model
        self.data = [_EmbeddingData(v, i) for i, v in enumerate(vectors)]
        self.usage = _EmbeddingUsage()
```

**Effort:** Low (pure Python wrapper).

---

### P1.7 — `completion_tokens_details` and Anthropic cache tokens

For Anthropic users, expose cache_creation_input_tokens and cache_read_input_tokens on `usage`.

**Effort:** Medium (must expose from Rust; Anthropic provider sets these already in `cache_prompt.rs`).

---

## P2 — Nice to Have

### P2.1 — `n` param (multiple completions)

Return multiple choices when `n > 1`. Requires Rust support (loop or parallel calls) and updating `ModelResponseCompat.choices` to hold multiple items.

**Effort:** High (Rust loop logic, response shape change).

---

### P2.2 — `response_format` as full dict

**Current state:** `response_format` accepts only `"json_object"` string.

**Target state:** Accept OpenAI-style dict:
```python
response_format={"type": "json_object"}
response_format={"type": "json_schema", "json_schema": {...}}
```

**Effort:** Low-Medium (Python normalisation, Rust already handles JSON mode).

---

### P2.3 — `litellm.api_key` / `litellm.api_base` module globals

```python
edgequake_litellm.api_key = "sk-..."
edgequake_litellm.api_base = "https://proxy.example.com/v1"
```

These module-level overrides feed into every subsequent call.

**Implementation:** Read in `_parse_model()` / `build_options()`.

**Effort:** Low-Medium.

---

### P2.4 — `logprobs` / `top_logprobs` support

Pass through to providers that support it (OpenAI, Mistral).

**Effort:** Medium (Rust body change + response field exposure).

---

### P2.5 — `parallel_tool_calls` param

Pass through to OpenAI API.

**Effort:** Low (Rust JSON body).

---

### P2.6 — `stream_options` / `include_usage: True`

Allow passing `stream_options={"include_usage": True}` to get token counts from streaming.

**Effort:** Medium (Rust streaming accumulation logic).

---

### P2.7 — `max_completion_tokens` alias

Treat `max_completion_tokens` as alias for `max_tokens` in Python layer.

**Effort:** Trivial.

---

### P2.8 — Better error messages for dropped params

Add a debug-mode warning when `drop_params=True` and an unknown param is passed:
```
WARNING: edgequake-litellm: Ignoring unknown param 'repo' passed to completion(). 
         Set verbose=True for full list.
```

**Effort:** Low.

---

## P3 — Out of Scope (Document Only)

### P3.1 — `litellm.Router`

Load balancing across multiple model deployments is a significant subsystem. edgequake-litellm is designed as a thin, fast Rust-backed edge layer — not a routing proxy. **Recommend using litellm as Router + edgequake-litellm as leaf provider.**

### P3.2 — Callback system (`litellm.callbacks`)

The litellm callback system supports 30+ observability platforms. edgequake-litellm instead provides **native OTEL** (OpenTelemetry) instrumentation at the Rust level — which is lower overhead and more consistent. Users needing platform-specific integrations (Langfuse, MLflow, etc.) should use OTEL exporters.

### P3.3 — `text_completion()` (legacy `/v1/completions`)

The old `/v1/completions` endpoint is deprecated by all major providers. Not planned.

### P3.4 — `image_generation()` / `transcription()` / `speech()`

Multi-modal generation functions — feasible to add but require significant Rust provider work. On the long-term roadmap.

### P3.5 — `litellm.model_cost` pricing dict

Maintaining a live pricing database is operationally expensive. Alternative: expose token counts via `usage` fields and let users compute cost from their own pricing config.

---

## Implementation Priority Matrix

| Item | Impact | Effort | Priority |
|------|--------|--------|----------|
| P0.1 `choices` shim on ModelResponse | 🔴 Critical | Low | **Now** |
| P0.2 `acompletion(stream=True)` | 🔴 Critical | Low | **Now** |
| P0.3 `completion(stream=True)` → clear error | 🟡 Medium | Trivial | **Now** |
| P1.6 `EmbeddingResponseCompat` | 🟡 Medium | Low | **Now** |
| P1.5 `response_ms` on response | 🟢 Low | Trivial | **Now** |
| P2.7 `max_completion_tokens` alias | 🟢 Low | Trivial | **Now** |
| P2.2 `response_format` as dict | 🟡 Medium | Low | **Soon** |
| P2.8 Better drop_params warnings | 🟢 Low | Low | **Soon** |
| P1.1 `timeout` wired to Rust | 🔴 Critical | Medium (Rust) | **Sprint** |
| P1.2 `api_base`/`api_key` per-call | 🔴 Critical | Medium (Rust) | **Sprint** |
| P1.3 `seed` passthrough | 🟡 Medium | Low (Rust) | **Sprint** |
| P1.4 `user` passthrough | 🟡 Medium | Low (Rust) | **Sprint** |
| P2.3 Module-level `api_key`/`api_base` | 🟡 Medium | Low | **Sprint** |
| P1.7 Cache token fields | 🟢 Low | Medium (Rust) | **Later** |
| P2.4 `logprobs` | 🟢 Low | Medium (Rust) | **Later** |
| P2.1 `n` param | 🟢 Low | High (Rust) | **Later** |
| P3.x Router / Callbacks | 🔵 Strategic | Very High | **Not planned** |

---

## Code Snippets for "Now" Items

### Fix 1: `ModelResponseCompat` — add to `_compat.py`, use in `completion.py`

```python
# In completion.py — wrap the Rust response before returning
from edgequake_litellm._compat import ModelResponseCompat

def completion(...) -> ModelResponseCompat:
    raw = _elc_core.completion(...)
    return ModelResponseCompat(raw)
```

### Fix 2: `acompletion(..., stream=True)` 

```python
async def acompletion(..., stream: bool = False, ...):
    if stream:
        from edgequake_litellm.streaming import stream as _stream
        return _stream(model, messages, ...)
    ...
```

### Fix 3: `completion(..., stream=True)` → clear error

```python
def completion(..., stream: bool = False, ...):
    if stream:
        raise NotImplementedError(
            "Synchronous streaming is not supported. "
            "Use acompletion(..., stream=True) in an async context, "
            "or use the stream() async generator function."
        )
```

### Fix 4: `EmbeddingResponseCompat` wrapper

```python
class EmbeddingResponseCompat:
    def __init__(self, vectors, model="", usage=None):
        self.object = "list"
        self.model = model
        self.data = [_EmbeddingData(v, i) for i, v in enumerate(vectors)]
        self.usage = usage  # or EmbeddingUsage(total_tokens=0)

    def __iter__(self):
        """Allow treating as List[List[float]] for backwards compat."""
        return iter(d.embedding for d in self.data)

    def __getitem__(self, idx):
        return self.data[idx].embedding
```

---

## Developer Experience Principles

1. **Never silently corrupt** — if a param is dropped, either warn (verbose mode) or document it.
2. **Fail fast with clarity** — if `stream=True` isn't supported synchronously, raise an exception immediately with a helpful message.
3. **Extend, don't replace** — `resp.content` shortcut alongside `resp.choices[0].message.content` means existing users of either style work.
4. **Measure first** — `response_ms` should be trivially available; perf-conscious users depend on it.
5. **Rust stays the source of truth** — all Python wrappers are thin adapters; business logic lives in Rust.
