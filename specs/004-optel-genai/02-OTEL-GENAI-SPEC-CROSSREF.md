# 02 — OTEL GenAI Spec Cross-Reference

> **Purpose**: Single source of truth mapping official OpenTelemetry GenAI semantic conventions to edgequake-llm implementation targets.  
> **Canonical repo**: [open-telemetry/semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai)  
> **Status in spec**: **Development** (not stable as of July 2026)

---

## 1. Version Timeline & Migration

| Release | Date | Breaking changes relevant to edgequake-llm |
|---------|------|---------------------------------------------|
| **v1.36** | 2025 | Baseline: `gen_ai.system`, content on span attributes (`gen_ai.prompt`, `gen_ai.completion`) |
| **v1.37** | 2025-06 | `gen_ai.system` → **`gen_ai.provider.name`**; structured **`gen_ai.input.messages` / `gen_ai.output.messages`**; content moves to **events** |
| **v1.39** | 2025 | MCP tool conventions; `execute_tool` spans |
| **v1.42** | 2026 | All `gen_ai.*` **moved out** of core semconv repo to dedicated GenAI repo |
| **Latest (main)** | 2026-07 | Agent spans, memory ops, cache token split, reasoning tokens rename |

### Transition mechanism (official)

Per [semantic-conventions-genai README](https://github.com/open-telemetry/semantic-conventions-genai):

```bash
# Emit latest experimental conventions only (no legacy)
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental

# Default for existing instrumentations: keep emitting v1.36-style until stable
```

**edgequake-llm mapping**:

| Env var | Values | Default |
|---------|--------|---------|
| `EDGEQUAKE_OTEL_SEMCONV_OPT_IN` | `gen_ai_latest_experimental` \| unset | unset (legacy v1.36 emit) |
| `EDGECODE_CAPTURE_CONTENT` | `true` \| unset | unset (no message content) |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | `true` | — (honor for cross-SDK consistency) |

When either capture env is `true`, enable content in events (and legacy span attrs if in legacy mode).

---

## 2. Signal Inventory

| Signal | Spec doc | edgequake-llm target |
|--------|----------|----------------------|
| **Traces — inference spans** | [gen-ai-spans.md § Inference](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md) | P0 — refactor `TracingProvider` |
| **Traces — embeddings spans** | [gen-ai-spans.md § Embeddings](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md) | P1 — embedding providers |
| **Traces — execute_tool** | [gen-ai-spans.md § Execute tool](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md) | P2 — agent helper API |
| **Events** | [gen-ai-events.md](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md) | P0 — fix `genai_events.rs` |
| **Metrics** | [gen-ai-metrics.md](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md) | P2 — `opentelemetry_sdk` meters |
| **Exceptions** | [gen-ai-exceptions.md](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-exceptions.md) | P1 — map `LlmError` → `error.type` |

---

## 3. Inference Span — Required & Recommended Attributes

Cross-ref: [gen-ai-spans.md — Inference attributes table](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)

### 3.1 Span identity

| Spec requirement | Value | edgequake-llm today | Target |
|------------------|-------|---------------------|--------|
| **Span name** | `{gen_ai.operation.name} {gen_ai.request.model}` | `gen_ai.chat`, `gen_ai.complete`, … | `"chat gpt-4o"`, `"text_completion llama3"` |
| **Span kind** | `CLIENT` (recommended) | unset (INTERNAL default) | Set via `tracing-opentelemetry` span builder or Otel layer |
| **Operation enum** | `chat`, `text_completion`, `generate_content`, `embeddings`, … | `chat`, `complete`, `chat_with_tools`, `stream`, … | Map methods → official enum (see §3.3) |

### 3.2 Required attributes (latest)

| Attribute | Requirement | edgequake-llm constant | Notes |
|-----------|-------------|------------------------|-------|
| `gen_ai.operation.name` | **Required** | `genai_attrs::OPERATION_NAME` ✓ | Fix enum values |
| `gen_ai.provider.name` | **Required** (v1.37+) | Missing — uses `gen_ai.system` | Add `PROVIDER_NAME`; dual-emit `gen_ai.system` in legacy mode |
| `gen_ai.request.model` | Cond. required | ✓ | Set at span creation |
| `gen_ai.request.stream` | Cond. required if streaming | Missing | `true` for `stream*` methods |
| `error.type` | Cond. required on error | Missing | Map from `LlmError` |

### 3.3 Operation name mapping (edgequake method → spec enum)

| `LLMProvider` method | Current `operation.name` | **Target** `gen_ai.operation.name` | Rationale |
|----------------------|--------------------------|-----------------------------------|-----------|
| `complete()` | `complete` | `text_completion` | OpenAI Completions API equivalent |
| `complete_with_options()` | `complete` | `text_completion` | Same |
| `chat()` | `chat` | `chat` | ✓ |
| `chat_with_tools()` | `chat_with_tools` | `chat` | Tools are part of chat API; add `gen_ai.tool.definitions` opt-in |
| `stream()` | `stream` | `text_completion` | + `gen_ai.request.stream=true` |
| `chat_with_tools_stream()` | `stream_with_tools` | `chat` | + `gen_ai.request.stream=true` |

Custom operation names (`chat_with_tools`) violate spec enum — backends filter on `chat` only.

### 3.4 Recommended response / usage attributes

| Attribute | edgequake-llm today | Target |
|-----------|---------------------|--------|
| `gen_ai.response.model` | ✓ | ✓ |
| `gen_ai.response.id` | Event only | Also on span (from `response.metadata["id"]`) |
| `gen_ai.response.finish_reasons` | Single string | **Array** — wrap in `["stop"]` |
| `gen_ai.usage.input_tokens` | ✓ | ✓ |
| `gen_ai.usage.output_tokens` | ✓ | ✓ |
| `gen_ai.usage.cache_read.input_tokens` | `gen_ai.usage.cache_hit_tokens` (custom) | Rename + dual-emit legacy name |
| `gen_ai.usage.cache_creation.input_tokens` | Missing | From `response.cache_write_tokens` |
| `gen_ai.usage.reasoning.output_tokens` | `gen_ai.usage.reasoning_tokens` (custom) | Rename + dual-emit |
| `gen_ai.response.time_to_first_chunk` | Missing | From `InferenceMetrics::ttft_ms()` / stream |
| `gen_ai.request.max_tokens` | ✓ | ✓ |
| `gen_ai.request.temperature` | ✓ | ✓ |
| `gen_ai.request.top_p` | ✓ | ✓ |
| `gen_ai.request.frequency_penalty` | Event only | Span when set |
| `gen_ai.request.presence_penalty` | Event only | Span when set |
| `gen_ai.output.type` | Missing | `text` default; `json` when `response_format` json |

### 3.5 Opt-in content attributes (latest: prefer events)

| Attribute | Spec | edgequake-llm | Target (latest mode) |
|-----------|------|---------------|----------------------|
| `gen_ai.input.messages` | Opt-in, structured | `gen_ai.prompt` JSON on span | **Event only** (or span if backend shim enabled) |
| `gen_ai.output.messages` | Opt-in, structured | `gen_ai.completion.content` on span | **Event only** |
| `gen_ai.system_instructions` | Opt-in | Missing | Extract system role messages |
| `gen_ai.tool.definitions` | Opt-in | Missing | When `chat_with_tools*` + capture enabled |

Spec reference — [Recording content on attributes](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md#recording-content-on-attributes):

> Instrumentations SHOULD NOT record raw content on span attributes in the latest conventions; use events instead.

---

## 4. Message JSON Schema Cross-Reference

Official schemas (repo `model/gen-ai/`):

| Schema | URL |
|--------|-----|
| Input messages | [gen-ai-input-messages.json](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/model/gen-ai/gen-ai-input-messages.json) |
| Output messages | [gen-ai-output-messages.json](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/model/gen-ai/gen-ai-output-messages.json) |
| Tool definitions | [gen-ai-tool-definitions.json](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/model/gen-ai/gen-ai-tool-definitions.json) |

### Structural delta (edgequake today vs spec)

```json
// edgequake-llm today (genai_events.rs)
{ "role": "user", "content": [{ "type": "text", "text": "hello" }] }

// spec (gen-ai-input-messages.json)
{ "role": "user", "parts": [{ "type": "text", "content": "hello" }] }
```

Tool call part types:

| Spec `type` | edgequake today | Action |
|-------------|-----------------|--------|
| `text` | `text` with `text` field | Rename field → `content` |
| `tool_call` | ✓ (partial) | Align `arguments` to object not string when possible |
| `tool_call_response` | Missing | Map `ChatRole::Tool` messages |

---

## 5. Events Cross-Reference

Event name (required): **`gen_ai.client.inference.operation.details`**

Per [gen-ai-events.md](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md):

| Event field | Requirement | edgequake-llm | Target |
|-------------|-------------|---------------|--------|
| `gen_ai.operation.name` | Should match span | Missing on event | Add |
| `gen_ai.provider.name` | Should match span | Missing | Add |
| `gen_ai.input.messages` | Structured | JSON string ✓ | Fix schema to `parts` |
| `gen_ai.output.messages` | Structured | JSON string ✓ | Fix schema |
| `gen_ai.response.id` | Recommended | ✓ | ✓ |
| `gen_ai.usage.*` | Recommended | ✓ | Align cache/reasoning names |

**Emit policy (target)**:

| Content | Legacy mode | Latest mode |
|---------|-------------|-------------|
| Metadata (tokens, model, finish) | Always on span + event | Always on span |
| Messages | Event if capture enabled | Event if capture enabled |
| Span attribute `gen_ai.prompt` | If capture enabled | **Never** (use backend shim flag only) |

---

## 6. Provider Name Registry

Per [gen-ai-spans.md — gen_ai.provider.name enum](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md):

| `LLMProvider::name()` (today) | **Target** `gen_ai.provider.name` | Notes |
|-------------------------------|-----------------------------------|-------|
| `openai` | `openai` | ✓ |
| `azure` | `azure.ai.openai` | Distinguish from Azure AI Inference |
| `anthropic` | `anthropic` | ✓ |
| `gemini` | `gcp.gemini` or `gcp.vertex_ai` | Detect endpoint in provider config |
| `bedrock` | `aws.bedrock` | ✓ |
| `cohere` | `cohere` | ✓ |
| `mistral` | `mistral_ai` | Spec uses underscore |
| `xai` | `x_ai` | ✓ |
| `openrouter` | `openai` | Proxy — document in span event `edgequake.routing.provider=openrouter` |
| `ollama` | `openai` | OpenAI-compatible local; add `server.address=localhost` |
| `lmstudio` | `openai` | Same |
| `nvidia` | `openai` | NIM is OpenAI-compatible |
| `huggingface` | `_OTHER` or custom `huggingface` | Not in enum — use documented custom |
| `mock` | `openai` | Test only |

Implementation: **`ProviderNameRegistry::resolve(provider_name: &str, config: &ProviderConfig) -> &'static str`**

Legacy dual-emit: also set `gen_ai.system` = old short name for LangSmith docs that still reference it.

---

## 7. Metrics Cross-Reference (P2)

Per [gen-ai-metrics.md](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md):

| Metric | Unit | edgequake-llm source |
|--------|------|---------------------|
| `gen_ai.client.token.usage` | `{token}` histogram | `MetricsLLMMiddleware` / span completion |
| `gen_ai.client.operation.duration` | `s` | Span duration (automatic) |
| `gen_ai.client.operation.time_to_first_chunk` | `s` | `InferenceMetrics::ttft_ms()` |
| `gen_ai.client.operation.time_per_output_chunk` | `s` | Stream chunk timestamps |

Bridge strategy: optional `GenAiMetricsRecorder` behind `otel-metrics` feature; record from `TracingProvider` on span close to avoid duplicate middleware.

---

## 8. Application & Tenant Attributes

| Attribute | Standard | edgequake-llm (spec 003) | Status |
|-----------|----------|--------------------------|--------|
| `gen_ai.application.id` | Proposed in GenAI SIG | ✓ `ApplicationContext` | Keep — track SIG issue |
| `gen_ai.application.name` | Proposed | ✓ | Keep |
| `gen_ai.application.url` | Proposed | ✓ | Keep |
| `tenant.id` | [General semconv](https://opentelemetry.io/docs/specs/semconv/general/attributes/) | ✓ | Keep |

---

## 9. Schema URL

When latest mode enabled, set tracer resource / span attribute:

```text
schema.url = "https://opentelemetry.io/schemas/genai/v0.1.0"  # TBD — follow GenAI repo release tags
```

Track [semantic-conventions-genai schema URL TODO](https://github.com/open-telemetry/semantic-conventions-genai).

---

## 10. Conformance Test Checklist

Each inference call must satisfy (automated in `tests/genai_semconv_conformance.rs`):

- [ ] Span name matches `{operation} {model}`
- [ ] `gen_ai.operation.name` ∈ official enum
- [ ] `gen_ai.provider.name` present (latest) or `gen_ai.system` (legacy)
- [ ] `gen_ai.request.model` set at creation time
- [ ] Usage integers ≥ 0 on success
- [ ] `gen_ai.response.finish_reasons` is array when set
- [ ] Event name exactly `gen_ai.client.inference.operation.details` when capture on
- [ ] Input/output JSON validates against official schema (jsonschema crate)
- [ ] No `gen_ai.prompt` on span in latest mode
- [ ] Stream spans have `gen_ai.request.stream=true`
