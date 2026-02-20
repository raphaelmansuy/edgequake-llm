# edgequake-litellm vs litellm — Compatibility Matrix

> Last updated: 2025-07  
> edgequake-litellm version: **0.3.x**  
> litellm reference version: **1.x (stable)**  
> litellm docs: https://docs.litellm.ai/

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Fully compatible — identical behaviour |
| ⚠️ | Partial — works but with differences (noted below) |
| ❌ | Not implemented |
| 🔧 | Silently dropped (no error, no effect) |
| 📋 | On roadmap |

---

## 1. Top-Level Functions

| Function | Status | Notes |
|----------|--------|-------|
| `completion(model, messages, ...)` | ✅ | Core params supported. See §3. |
| `acompletion(model, messages, ...)` | ✅ | Async version — same params. |
| `completion(..., stream=True)` | ❌ | Only `stream()` async generator available; `stream=True` kwarg raises `NotImplementedError`. |
| `acompletion(..., stream=True)` | ⚠️ | Returns `AsyncGenerator[StreamChunk, None]` — compatible usage but different object type. |
| `stream(model, messages, ...)` | ⚠️ | **edgequake-only** function (not in litellm). litellm uses `stream=True` kwarg on `completion`. |
| `embedding(model, input, ...)` | ⚠️ | Returns `List[List[float]]` — litellm returns `EmbeddingResponse` object. |
| `aembedding(model, input, ...)` | ⚠️ | Same return-type difference as `embedding()`. |
| `text_completion(prompt, model, ...)` | ❌ | Old `/v1/completions` endpoint — not implemented. |
| `image_generation(prompt, model, ...)` | ❌ | Not implemented. |
| `transcription(file, model, ...)` | ❌ | Not implemented. |
| `speech(model, input, voice, ...)` | ❌ | Not implemented. |
| `stream_chunk_builder(chunks, messages)` | ✅ | Provided as `edgequake_litellm.stream_chunk_builder()`. |
| `get_supported_openai_params(model, ...)` | ❌ | Not implemented. |
| `utils.get_model_info(model)` | ❌ | Not implemented. |

---

## 2. Module-Level Globals

| Global | litellm | edgequake-litellm | Status |
|--------|---------|-------------------|--------|
| `litellm.api_key` | Global fallback API key | ❌ | Not implemented |
| `litellm.api_base` | Global base URL override | ❌ | Not implemented |
| `litellm.set_verbose` | Enable debug logging | ✅ | Wraps `_config.verbose` |
| `litellm.drop_params` | Silently drop unsupported params | ✅ | Always `True` (immutable for now) |
| `litellm.model_cost` | Dict of model pricing | ❌ | Not implemented |
| `litellm.callbacks` | List of callback handlers | ❌ | Not implemented |
| `litellm.success_callback` | On-success hooks | ❌ | Not implemented |
| `litellm.failure_callback` | On-failure hooks | ❌ | Not implemented |
| `litellm.REPEATED_STREAMING_CHUNK_LIMIT` | Streaming safety limit | ❌ | Not implemented |

---

## 3. `completion()` / `acompletion()` — Input Parameters

### Core Parameters

| Parameter | litellm | edgequake-litellm | Status |
|-----------|---------|-------------------|--------|
| `model` | ✅ Required | ✅ Required | ✅ |
| `messages` | ✅ Required | ✅ Required | ✅ |
| `max_tokens` | ✅ | ✅ | ✅ |
| `temperature` | ✅ | ✅ | ✅ |
| `top_p` | ✅ | ✅ | ✅ |
| `stop` | ✅ List[str] | ✅ List[str] | ✅ |
| `frequency_penalty` | ✅ | ✅ | ✅ |
| `presence_penalty` | ✅ | ✅ | ✅ |
| `response_format` | ✅ `str` or `dict` | ⚠️ `str` only (`"json_object"`) | ⚠️ |
| `tools` | ✅ | ✅ | ✅ |
| `tool_choice` | ✅ | ✅ | ✅ |
| `stream` | ✅ bool | ❌ Not accepted | ❌ |
| `n` | ✅ int | ❌ Silently dropped | 🔧 |
| `seed` | ✅ int | ❌ Silently dropped | 🔧 |
| `logit_bias` | ✅ dict | ❌ Silently dropped | 🔧 |
| `logprobs` | ✅ bool | ❌ Silently dropped | 🔧 |
| `top_logprobs` | ✅ int | ❌ Silently dropped | 🔧 |
| `parallel_tool_calls` | ✅ bool | ❌ Silently dropped | 🔧 |
| `user` | ✅ str | 🔧 Accepted, silently dropped | 🔧 |
| `timeout` | ✅ float/int | ⚠️ Accepted in signature, not wired to Rust | ⚠️ |
| `max_completion_tokens` | ✅ alias for max_tokens | ❌ Dropped | 🔧 |

### litellm-Specific Parameters (Provider Overrides)

| Parameter | litellm | edgequake-litellm | Status |
|-----------|---------|-------------------|--------|
| `api_base` / `base_url` | ✅ Per-call URL override | ⚠️ Accepted, not wired to Rust core | ⚠️ |
| `api_key` | ✅ Per-call key override | ⚠️ Accepted, not wired to Rust core | ⚠️ |
| `api_version` | ✅ Azure version pin | ❌ Dropped | 🔧 |
| `headers` / `extra_headers` | ✅ Custom HTTP headers | ❌ Dropped | 🔧 |
| `num_retries` | ✅ Per-call retry count | ❌ Dropped (config default used) | 🔧 |
| `fallbacks` | ✅ List of fallback models | ❌ Not implemented | ❌ |
| `metadata` | ✅ Arbitrary logging dict | ❌ Dropped | 🔧 |
| `input_cost_per_token` | ✅ Cost override | ❌ Dropped | 🔧 |
| `output_cost_per_token` | ✅ Cost override | ❌ Dropped | 🔧 |
| `initial_prompt_value` | ✅ | ❌ Dropped | 🔧 |
| `stream_options` | ✅ `{"include_usage": True}` | ❌ Dropped | 🔧 |

### edgequake-litellm-Only Parameters

| Parameter | Description |
|-----------|-------------|
| `system` | Convenience shorthand for adding a system message. Not in litellm's API. |

---

## 4. `ModelResponse` — Output Object

### Field Comparison

| Field | litellm access | edgequake-litellm | Status |
|-------|---------------|-------------------|--------|
| Response ID | `resp.id` | ❌ Not exposed | ❌ |
| Created timestamp | `resp.created` | ❌ Not exposed | ❌ |
| Object type | `resp.object` | ❌ Not exposed | ❌ |
| System fingerprint | `resp.system_fingerprint` | ❌ Not exposed | ❌ |
| Model name | `resp.model` | ✅ `resp.model` | ✅ |
| Message content | `resp.choices[0].message.content` | ⚠️ `resp.content` (shortcut) | ⚠️ |
| Message role | `resp.choices[0].message.role` | ❌ Not exposed | ❌ |
| Finish reason | `resp.choices[0].finish_reason` | ❌ Not exposed | ❌ |
| Tool calls | `resp.choices[0].message.tool_calls` | ⚠️ `resp.tool_calls` (shortcut) | ⚠️ |
| Choices list | `resp.choices` (list, len = n) | ❌ No `choices` attribute | ❌ |
| Prompt tokens | `resp.usage.prompt_tokens` | ✅ `resp.usage.prompt_tokens` | ✅ |
| Completion tokens | `resp.usage.completion_tokens` | ✅ `resp.usage.completion_tokens` | ✅ |
| Total tokens | `resp.usage.total_tokens` | ✅ `resp.usage.total_tokens` | ✅ |
| Cached tokens | `resp.usage.prompt_tokens_details.cached_tokens` | ❌ Not exposed | ❌ |
| Cache creation tokens | `resp.usage.cache_creation_input_tokens` | ❌ Not exposed | ❌ |
| Cache read tokens | `resp.usage.cache_read_input_tokens` | ❌ Not exposed | ❌ |
| Completion token details | `resp.usage.completion_tokens_details` | ❌ Not exposed | ❌ |
| Latency | `resp.response_ms` | ❌ Not exposed | ❌ |
| Dict access | `resp["choices"][0]["message"]["content"]` | ❌ No `__getitem__` | ❌ |

### litellm Example vs edgequake-litellm

```python
# litellm
import litellm
resp = litellm.completion("gpt-4o-mini", messages)
print(resp.choices[0].message.content)   # standard OpenAI path
print(resp["choices"][0]["message"]["content"])  # dict path

# edgequake-litellm
import edgequake_litellm as litellm
resp = litellm.completion("openai/gpt-4o-mini", messages)
print(resp.content)                       # shortened accessor ✅
# resp.choices[0].message.content         ← ❌ raises AttributeError
```

**Impact**: Any code that accesses `response.choices[0].message.content` will break.  
This is the **single largest compatibility gap** today.

---

## 5. Streaming API

### API Shape Difference

| Aspect | litellm | edgequake-litellm |
|--------|---------|-------------------|
| Sync streaming | `for chunk in completion(..., stream=True)` | ❌ Not supported |
| Async streaming | `async for chunk in acompletion(..., stream=True)` | ⚠️ `async for chunk in stream(model, messages)` |
| Chunk type | `ModelResponse` with `choices[0].delta.content` | `StreamChunk` with `.content` |
| Finish detection | `choices[0].finish_reason == "stop"` | `chunk.is_finished == True` |
| Chunk helper | `stream_chunk_builder(chunks, messages)` | ✅ `stream_chunk_builder(chunks)` provided |

### Streaming Chunk Field Comparison

| Field | litellm chunk | edgequake-litellm StreamChunk |
|-------|--------------|-------------------------------|
| Delta content | `chunk.choices[0].delta.content` | ✅ `chunk.content` |
| Delta role | `chunk.choices[0].delta.role` | ❌ Not exposed |
| Delta tool calls | `chunk.choices[0].delta.tool_calls` | ❌ Not exposed |
| Finish reason | `chunk.choices[0].finish_reason` | ⚠️ `chunk.finish_reason` |
| Is finished | (check finish_reason) | ✅ `chunk.is_finished` |
| Thinking/reasoning | N/A (litellm) | ✅ `chunk.thinking` (Anthropic extended) |
| Index | `chunk.choices[0].index` | ❌ Not exposed |

### Migration Pattern

```python
# litellm pattern
for chunk in litellm.completion("gpt-4o-mini", msgs, stream=True):
    print(chunk.choices[0].delta.content or "", end="")

# edgequake-litellm equivalent (must be async)
import asyncio
async def run():
    async for chunk in edgequake_litellm.stream("openai/gpt-4o-mini", msgs):
        print(chunk.content or "", end="")

asyncio.run(run())
```

---

## 6. `embedding()` / `aembedding()`

### Input Parameters

| Parameter | litellm | edgequake-litellm | Status |
|-----------|---------|-------------------|--------|
| `model` | ✅ | ✅ | ✅ |
| `input` | ✅ `str` or `List[str]` | ✅ | ✅ |
| `user` | ✅ | ❌ Dropped | 🔧 |
| `dimensions` | ✅ | ❌ Dropped | 🔧 |
| `encoding_format` | ✅ `"float"` / `"base64"` | ❌ Dropped | 🔧 |
| `timeout` | ✅ | ❌ Dropped | 🔧 |
| `api_base` | ✅ | ❌ Dropped | 🔧 |
| `api_key` | ✅ | ❌ Dropped | 🔧 |

### Return Type Difference

```python
# litellm — returns EmbeddingResponse
result = litellm.embedding("text-embedding-3-small", input=["hello"])
vectors = [item.embedding for item in result.data]   # List[List[float]]

# edgequake-litellm — returns List[List[float]] directly
vectors = edgequake_litellm.embedding("openai/text-embedding-3-small", input=["hello"])
# No .data / .model / .usage attributes available
```

**Impact**: Any code that accesses `result.data`, `result.model`, or `result.usage` will break.

---

## 7. Exception Hierarchy

| litellm exception | edgequake-litellm exception | Status |
|------------------|---------------------------|--------|
| `litellm.AuthenticationError` | `AuthenticationError` | ✅ |
| `litellm.RateLimitError` | `RateLimitError` | ✅ |
| `litellm.ContextWindowExceededError` | `ContextWindowExceededError` | ✅ |
| `litellm.Timeout` | `Timeout` | ✅ |
| `litellm.APIConnectionError` | `APIConnectionError` | ✅ |
| `litellm.APIError` | `APIError` | ✅ |
| `litellm.NotFoundError` | `ModelNotFoundError` | ⚠️ Different name |
| `litellm.BadRequestError` | ❌ Not defined | ❌ |
| `litellm.ServiceUnavailableError` | ❌ Not defined | ❌ |
| `litellm.RouterErrors.RouterLLMNotFoundError` | ❌ Not defined | ❌ |
| `.status_code` attribute | ✅ | ✅ |
| `.llm_provider` attribute | ✅ | ✅ |
| `.model` attribute | ✅ | ✅ |

---

## 8. Router / Load Balancing

The `litellm.Router` class provides horizontal scaling across multiple model deployments — this is an **entirely separate subsystem** not present in edgequake-litellm.

| Feature | litellm.Router | edgequake-litellm |
|---------|---------------|-------------------|
| Multiple model deployments | ✅ | ❌ |
| Load balancing strategies | ✅ (simple-shuffle, latency-based, cost-based, etc.) | ❌ |
| Automatic retries | ✅ | ❌ |
| Deployment cooldowns | ✅ | ❌ |
| Redis caching | ✅ | ❌ |
| In-memory caching | ✅ | ❌ |
| Fallbacks | ✅ | ❌ |
| `router.completion()` | ✅ | ❌ |
| `router.acompletion()` | ✅ | ❌ |

---

## 9. Callbacks & Observability

litellm has a rich callback system for integrating with 30+ observability platforms. edgequake-litellm instead provides native OpenTelemetry instrumentation through the Rust core.

| Feature | litellm | edgequake-litellm |
|---------|---------|-------------------|
| `litellm.success_callback` | ✅ | ❌ |
| `litellm.failure_callback` | ✅ | ❌ |
| `litellm.callbacks = [handler]` | ✅ | ❌ |
| Langfuse integration | ✅ | ❌ |
| MLflow integration | ✅ | ❌ |
| Helicone integration | ✅ | ❌ |
| Lunary integration | ✅ | ❌ |
| OpenTelemetry (OTEL) | ✅ (via callback) | ✅ **Native** (Rust-level) |
| Cost tracking (per call) | ✅ (via `litellm.model_cost`) | ⚠️ (Rust token tracking, no USD cost) |

---

## 10. Provider Coverage

### Text Completion Providers

| Provider | litellm | edgequake-litellm | Model string prefix |
|----------|---------|-------------------|---------------------|
| OpenAI | ✅ | ✅ | `openai/` |
| Anthropic | ✅ | ✅ | `anthropic/` |
| Google Gemini | ✅ | ✅ | `gemini/` |
| Mistral | ✅ | ✅ | `mistral/` |
| xAI (Grok) | ✅ | ✅ | `xai/` |
| OpenRouter | ✅ | ✅ | `openrouter/` |
| Ollama | ✅ | ✅ | `ollama/` |
| LM Studio | ✅ | ✅ | `lmstudio/` |
| Azure OpenAI | ✅ | ✅ | `azure/` |
| Hugging Face | ✅ | ✅ | `huggingface/` |
| Bedrock (AWS) | ✅ | ❌ | — |
| Vertex AI | ✅ | ❌ | — |
| Cohere | ✅ | ❌ | — |
| Together AI | ✅ | ❌ | — |
| Replicate | ✅ | ❌ | — |
| AI21 | ✅ | ❌ | — |
| Groq | ✅ | ❌ | — |

### Embedding Providers

| Provider | litellm | edgequake-litellm |
|----------|---------|-------------------|
| OpenAI | ✅ | ✅ (via `jina/` or `openai/`) |
| Jina AI | ✅ | ✅ |
| Mistral | ✅ | ✅ |
| Cohere | ✅ | ❌ |
| Bedrock | ✅ | ❌ |
| Vertex AI | ✅ | ❌ |

---

## 11. Overall Compatibility Summary

### Compatibility Score by Category

| Category | Score | Notes |
|----------|-------|-------|
| Core `completion()` params | 60% | Missing `stream`, `n`, `seed`, `timeout`, `api_key`, `api_base`, `user` |
| `ModelResponse` structure | 30% | Big gap: no `.choices`, no `.id`/`.created`, no dict access |
| Streaming API shape | 40% | Different function name, different chunk fields |
| `embedding()` return type | 50% | Data is there, wrapper object missing |
| Exception hierarchy | 85% | All critical ones present, a few missing |
| Module globals | 40% | `set_verbose`, `drop_params` present; no `api_key`, `model_cost` |
| Provider coverage | 65% | 10/17+ providers implemented |
| Router / Load balancing | 0% | Not in scope |
| Callbacks / Observability | 10% | OTEL native only |

### Drop-in Compatibility Assessment

**Scenario 1: Basic completion with `.content` access**
```python
resp = litellm.completion("openai/gpt-4o-mini", msgs)
print(resp.content)
```
→ **✅ Works** (edgequake-litellm extends with `.content` shortcut)

**Scenario 2: Standard OpenAI response path**
```python
resp = litellm.completion("openai/gpt-4o-mini", msgs)
print(resp.choices[0].message.content)
```
→ **❌ Fails** — `.choices` not available on `ModelResponse`

**Scenario 3: Synchronous streaming**
```python
for chunk in litellm.completion("openai/gpt-4o-mini", msgs, stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```
→ **❌ Fails** — `stream=True` raises `NotImplementedError`

**Scenario 4: Async streaming**
```python
async for chunk in litellm.acompletion("openai/gpt-4o-mini", msgs, stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```
→ **⚠️ Partial** — works in edgequake-litellm as `stream()` but chunk field path differs

**Scenario 5: Tool calling**
```python
resp = litellm.completion("openai/gpt-4o-mini", msgs, tools=[...])
tool_calls = resp.choices[0].message.tool_calls
```
→ **❌ Fails** — `.choices` not available; use `resp.tool_calls` instead

**Scenario 6: Embeddings with usage stats**
```python
result = litellm.embedding("openai/text-embedding-3-small", input=texts)
print(result.data[0].embedding, result.usage.total_tokens)
```
→ **❌ Fails** — returns `List[List[float]]`, not `EmbeddingResponse`

---

## 12. Quick Migration Reference

For codebases migrating from litellm to edgequake-litellm:

```python
# OLD (litellm)
import litellm

resp = litellm.completion("gpt-4o-mini", messages)
content = resp.choices[0].message.content

# NEW (edgequake-litellm)
import edgequake_litellm as litellm

resp = litellm.completion("openai/gpt-4o-mini", messages)
content = resp.content  # shortcut accessor

# ─── Streaming ───
# OLD: for chunk in litellm.completion(..., stream=True):
# NEW: async for chunk in edgequake_litellm.stream(...):

# ─── Embeddings ───
# OLD: result = litellm.embedding(...); vectors = [d.embedding for d in result.data]
# NEW: vectors = edgequake_litellm.embedding(...)  # already List[List[float]]

# ─── Model string ───
# OLD: "gpt-4o-mini" (litellm resolves provider automatically)
# NEW: "openai/gpt-4o-mini" (explicit provider prefix required)
```
