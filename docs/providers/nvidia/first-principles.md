# NVIDIA NIM Provider — First Principles Design

> **Deep technical specification for contributors implementing or extending the NVIDIA provider.**

---

## 1. Why First Principles?

Before writing code, we establish **invariants** from first principles to avoid
ad-hoc decisions that would need undoing later.

### Invariant 1 — Single Responsibility

The `NvidiaProvider` struct is **only** responsible for:
- Encapsulating NVIDIA-specific defaults (base URL, API key env var, model catalog)
- Routing streaming/tool-stream requests to `OpenAICompatibleProvider`
- Handling non-streaming chat requests with NVIDIA-specific HTTP 202 polling
- Forwarding `reasoning_effort` consistently for both streaming and non-streaming paths
- Providing a typed `list_models()` method

It is **not** responsible for:
- SSE parsing (delegated to `OpenAICompatibleProvider`)
- Cross-provider retry policy (delegated to caller / `RetryExecutor`)

### Invariant 2 — DRY

No HTTP or SSE code is duplicated. All chat/stream/tool paths go through
`OpenAICompatibleProvider`. Only `list_models()` and async-polling detection
use a raw `reqwest::Client`.

### Invariant 3 — Fail Fast, Fail Clear

All validation happens **at construction time** (`from_env()`, `new()`), not at
request time. A missing API key is a `ConfigError`, not a `NetworkError`.

### Invariant 4 — Open/Closed for Models

The static model catalog `NVIDIA_CHAT_MODELS` is an **append-only** data structure.
New models are added without changing any function signature or match arm.

### Invariant 5 — No env::set_var

We never call `std::env::set_var`. The API key is stored in the `NvidiaProvider`
struct and injected directly into the `ProviderConfig::api_key_env` field, which
`OpenAICompatibleProvider` uses to read from the environment. The key was already
set when `NvidiaProvider::from_env()` reads it.

---

## 2. Parameter Mapping

### thinking / reasoning_effort

```
CompletionOptions::reasoning_effort = Some("high")
  │
  ▼
NvidiaProvider forwards reasoning_effort in its non-streaming request path
and OpenAICompatibleProvider forwards reasoning_effort in streaming paths.
  │
  ▼
POST body includes:
  "reasoning_effort": "high"
```

The implementation passes `reasoning_effort` through the standard `CompletionOptions`
field, and `OpenAICompatibleProvider` already supports forwarding it.

---

## 3. 202 Async Handling

```
POST /v1/chat/completions response status = 202
  │
  ├─ extract `NVCF-REQID` response header
  │
  └─ poll GET https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{nvcf_reqid}
      every NVIDIA_POLL_INTERVAL_MS until non-202
         → 200: parse as normal completion response
         → 4xx/5xx: return LlmError::ApiError
         → max attempts reached: return timeout-style LlmError::ApiError
```

---

## 4. Model ID Format

NVIDIA NIM uses `org/model` format:
- `nvidia/llama-3.3-nemotron-super-49b-v1`
- `meta/llama-3.3-70b-instruct`
- `deepseek-ai/deepseek-v4-flash`

This is different from single-name models like `gpt-4o`. The `/` is part of the
model ID and must be URL-encoded when used in path parameters (though the chat
endpoint takes the model as a JSON body field, not a path param).

---

## 5. Free Tier Logic

NVIDIA does not expose a `free` flag in the models list response. The provider
uses a **static allowlist** of known-free models cross-referenced against the
live model list. If a model appears in both the live list AND the free allowlist,
`NvidiaModelInfo::is_free = true`.

This approach:
- Avoids scraping the web for pricing
- Is deterministic and testable
- Degrades gracefully: unknown models default to `is_free = false`

The free allowlist lives in `NVIDIA_FREE_MODELS` in `nvidia.rs`.

---

## 6. Thread Safety

`NvidiaProvider` is `Send + Sync`:
- `inner: OpenAICompatibleProvider` — Send + Sync
- `model: String` — Send + Sync
- `client: reqwest::Client` — Send + Sync (internally Arc-wrapped)
- `api_key: String` — Send + Sync

No `RefCell`, `Mutex`, or mutable state.

---

## 7. Timeout Strategy

Default timeout: **300 seconds** (5 minutes)

Rationale:
- Standard models respond in <10s
- Reasoning models (DeepSeek V4 Flash with `reasoning_effort=max`) may stream for
  2–3 minutes
- 300s provides headroom without hanging indefinitely

This is lower than xAI's 600s because NVIDIA's reasoning models are generally
faster than Grok 4 extended-thinking mode.

---

## 8. Context Length Fallback

If a model is not in the static catalog (e.g., newly added by NVIDIA), the
`context_length()` helper returns **32_768** (32K) as a conservative fallback.

This is intentionally conservative: returning too-high a value could cause callers
to stuff too much text into a prompt and get a 422 error at inference time.

---

## 9. Capability Flags

```
supports_function_calling = true  (all chat models)
supports_json_mode        = true  (all chat models)
supports_streaming        = true  (all chat models)
supports_system_message   = true  (all chat models)
supports_vision           = model-specific (see catalog)
supports_thinking         = model-specific (see catalog)
```

Vision models: any model whose ID contains `vl`, `vision`, or is in the known VL list
(e.g., `meta/llama-4-maverick-17b-128e-instruct`).

Thinking models: any model in `NVIDIA_CHAT_MODELS` with `thinking = true`.

---

## 10. Cross-References

| Component | Location |
|-----------|----------|
| Core implementation | `src/providers/nvidia.rs` |
| Provider registration | `src/providers/mod.rs` |
| Factory integration | `src/factory.rs` |
| Public API export | `src/lib.rs` |
| E2E tests | `tests/e2e_nvidia.rs` |
| Example | `examples/nvidia/chat.rs` |
| User docs | `docs/providers/nvidia/README.md` |
| Model listing structs | `src/providers/nvidia.rs::NvidiaModelsResponse` |
