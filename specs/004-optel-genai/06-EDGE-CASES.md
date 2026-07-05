# 06 — Edge Case Register

> **Purpose**: Exhaustive edge cases for OTel GenAI implementation — each with expected behavior and test ID.  
> **Rule**: Observability adapts; **LLM call semantics never change**.

---

## Category A — Span Identity & Naming

### EC-G01: Empty or unknown model name

**Trigger**: Provider returns `model: ""` or factory default missing.

**Expected**:
- Span name: `"chat "` or `"chat unknown"` (document chosen fallback)
- `gen_ai.request.model` omitted if truly unknown; never fabricate

**Test**: `genai_semconv_conformance::empty_model_name`

---

### EC-G02: Model name with special characters

**Trigger**: Model `meta-llama/Llama-3.1-70B-Instruct`.

**Expected**: Span name includes full string; OTLP UTF-8 safe

**Test**: Unit test span name formatting

---

### EC-G09: Dynamic span names in `tracing` macro

**Trigger**: Rust `tracing` version may require const span names.

**Expected**:
- Primary: dynamic name `{op} {model}`
- Fallback: static `"gen_ai.inference"` + attrs `gen_ai.operation.name`, `gen_ai.request.model`

**Test**: CI matrix on MSRV

---

## Category B — Operation & Method Mapping

### EC-G11: `chat_with_tools` is still `chat` operation

**Trigger**: Agent calls tools via chat API.

**Expected**:
- `gen_ai.operation.name = "chat"` (not `chat_with_tools`)
- Optional: `tool_count` custom attr (non-standard, low cardinality)
- Opt-in: `gen_ai.tool.definitions` on event

**Rationale**: Spec enum has no `chat_with_tools`; LangSmith maps `chat` → LLM run type.

---

### EC-G12: `complete()` maps to `text_completion`

**Trigger**: Legacy completion API usage.

**Expected**: Operation `text_completion`, not `complete`

---

### EC-G13: Gemini multimodal uses `generate_content`

**Trigger**: Provider internal API is Gemini generateContent.

**Expected**: When `ProviderHints::api_family == Gemini`, operation = `generate_content`

**Phase**: P1 — requires provider hint in registry

---

## Category C — Token Usage & Billing

### EC-G03: Reasoning / thinking tokens (OpenAI o-series, Claude)

**Trigger**: `response.thinking_tokens = Some(n)`.

**Expected**:
- Latest: `gen_ai.usage.reasoning.output_tokens = n`
- Legacy dual-emit: `gen_ai.usage.reasoning_tokens = n`
- Value included in `gen_ai.usage.output_tokens` per spec note [20]

---

### EC-G04: Cache read vs cache write tokens

**Trigger**: Anthropic/Gemini cache fields populated.

**Expected**:
| Field | Latest attribute |
|-------|------------------|
| `cache_hit_tokens` | `gen_ai.usage.cache_read.input_tokens` |
| `cache_write_tokens` | `gen_ai.usage.cache_creation.input_tokens` |

Omit when `None` or zero — don't emit sentinel -1 on spans (events currently use -1 for unset — **fix**: omit unset optional attrs entirely).

---

### EC-G14: Provider returns only `total_tokens`

**Trigger**: Local Ollama without breakdown.

**Expected**:
- Set `gen_ai.usage.output_tokens` if derivable; else only total in custom attr `edgequake.usage.total_tokens` (non-standard, debug only)
- Never invent input/output split

---

### EC-G15: Cohere billed vs consumed tokens

**Trigger**: Both `usage.billed_units` and `usage.tokens` in response.

**Expected**: Use **billed** count per spec [18][19]

**Phase**: P1 — provider-specific parsing in response layer

---

## Category D — Finish Reasons & Tool Calls

### EC-G02: Finish reason `tool_calls`

**Trigger**: Model returns tools instead of text.

**Expected**:
- `gen_ai.response.finish_reasons = ["tool_calls"]` (array)
- Output messages contain tool_call parts
- Langfuse profile: output = serialized tool calls JSON

---

### EC-G16: Multiple finish reasons (n>1 choices)

**Trigger**: `choice.count > 1` (rare in edgequake — single choice default).

**Expected**: Array with one entry per choice; edgequake typically `["stop"]`

---

### EC-G17: Empty finish reason

**Trigger**: Stream interrupted before finish event.

**Expected**: Omit `finish_reasons`; set `error.type` if error; span status ERROR

---

## Category E — Content Capture & Privacy

### EC-G06: Multimodal images in messages

**Trigger**: `ChatMessage.images` populated.

**Expected**:
- Latest: text parts only in events unless `EDGEQUAKE_OTEL_CAPTURE_IMAGES=true` (new, default false)
- Never base64 images on span attributes
- Document PII risk

---

### EC-G18: System prompt extraction

**Trigger**: First message role `system`.

**Expected**:
- Latest opt-in: `gen_ai.system_instructions` separate from `gen_ai.input.messages`
- Default: system messages stay in input messages only

---

### EC-G19: Message truncation for large contexts

**Trigger**: 200k context conversation.

**Expected**:
- Env `EDGEQUAKE_OTEL_MAX_MESSAGE_BYTES` (default 65536)
- Truncate with `...[truncated]` marker; UTF-8 safe (reuse thinking content logic from `tracing.rs:409-417`)

---

### EC-G20: Capture enabled but event emission fails

**Trigger**: JSON serialization error.

**Expected**: `tracing::warn!`; LLM response still returned

---

## Category F — Streaming

### EC-G05: Stream error mid-generation

**Trigger**: Provider disconnect after partial tokens.

**Expected**:
- Span status ERROR + `error.type`
- Partial `gen_ai.usage.*` if provider sent usage in last chunk
- `gen_ai.response.time_to_first_chunk` still set if first token arrived

---

### EC-G21: Stream with no usage until end

**Trigger**: OpenAI-compatible SSE without usage mid-stream.

**Expected**: Record usage on stream close; duration covers full stream

---

### EC-G22: Tool streaming (`chat_with_tools_stream`)

**Trigger**: Partial tool call deltas.

**Expected** (P2):
- Single span for entire stream
- Accumulate tool call JSON in guard
- Emit event on drop, not per chunk

---

### EC-G23: Provider doesn't support tool streaming

**Trigger**: `supports_tool_streaming() == false`.

**Expected**: Non-streaming path used; no special case

---

## Category G — Provider & Routing

### EC-G07: OpenRouter — requested vs actual model

**Trigger**: Request `anthropic/claude-3.5-sonnet`, response model differs.

**Expected**:
- `gen_ai.request.model` = requested
- `gen_ai.response.model` = actual from response
- `gen_ai.provider.name` = `openai` (OpenAI-compatible proxy) + `edgequake.routing.provider=openrouter`

---

### EC-G24: Azure OpenAI vs Azure AI Inference

**Trigger**: Azure provider variant.

**Expected**:
- Azure OpenAI: `azure.ai.openai`
- Other Azure AI: `azure.ai.inference`

---

### EC-G25: Gemini API vs Vertex AI

**Trigger**: Different Google endpoints.

**Expected**: `gcp.gemini` vs `gcp.vertex_ai` based on config host

---

### EC-G26: Local Ollama / LM Studio

**Trigger**: localhost provider.

**Expected**:
- `gen_ai.provider.name` = `openai` (compatible API)
- `server.address` = `localhost` or configured host
- `server.port` when non-default

---

## Category H — Backend Compatibility

### EC-G08: Langfuse ignores v1.37 events

**Trigger**: Profile `latest` + capture, no Langfuse shim.

**Expected**: Empty I/O in Langfuse UI — **document**; default examples use `langfuse` profile

---

### EC-G27: LangSmith needs `langsmith.span.kind`

**Trigger**: Profile `langsmith`.

**Expected**: Set `langsmith.span.kind = "LLM"` on inference spans

---

### EC-G28: Legacy OpenLLMetry token attribute names

**Trigger**: Profile `langsmith` or `max_compat`.

**Expected**: Also emit `gen_ai.usage.prompt_tokens`, `gen_ai.usage.completion_tokens`

---

### EC-G29: Datadog requires `gen_ai.provider.name`

**Trigger**: Profile `latest` without provider.name.

**Expected**: Span not classified as GenAI in DD — P0 fix

---

### EC-G30: Array attributes dropped by tracing-opentelemetry

**Trigger**: `finish_reasons` as array.

**Expected**: Test OTLP export; fallback JSON string `'["stop"]'` if exporter flattens

---

## Category I — Errors & Status

### EC-G31: Rate limit error (429)

**Expected**: `error.type = "429"` or provider-specific; span ERROR

---

### EC-G32: Timeout

**Expected**: `error.type = "timeout"`

---

### EC-G33: Auth error (401)

**Expected**: `error.type = "401"`; **never** capture API key in span attrs

---

### EC-G34: Cancelled request (drop future)

**Expected**: Span ends with cancelled status if detectable; else timeout

---

## Category J — Concurrency & Context

### EC-G10: Nested spans (agent wraps LLM)

**Trigger**: User creates parent `agent.run` span.

**Expected**: LLM span as child; `ApplicationContext` still applied

---

### EC-G35: No active tracing subscriber

**Trigger**: User forgets to init OTLP layer.

**Expected**: No-op; no panic; LLM works

---

### EC-G36: ApplicationContext + baggage promotion (spec 003)

**Trigger**: `EDGEQUAKE_OTEL_PROMOTE_APP_TO_BAGGAGE=true`.

**Expected**: `gen_ai.application.id` on span **and** baggage; independent of semconv version

---

## Category K — Embeddings & Future Operations

### EC-G37: Embedding call tracing

**Phase**: P2

**Expected**: Operation `embeddings`; span name `"embeddings {model}"`; input tokens only

---

### EC-G38: execute_tool span from agent code

**Phase**: P2

**Expected**: Separate span `execute_tool {tool_name}`; parent = agent span

---

## Test ID Index

| ID | Priority | Automated |
|----|----------|-----------|
| EC-G01–G04 | P0 | Yes |
| EC-G05–G08 | P0 | Partial (G08 manual) |
| EC-G09 | P0 | CI matrix |
| EC-G10 | P1 | Yes |
| EC-G11–G17 | P0 | Yes |
| EC-G18–G23 | P1 | Partial |
| EC-G24–G30 | P0–P1 | Yes / manual |
| EC-G31–G36 | P0 | Yes |
| EC-G37–G38 | P2 | Planned |

---

## Decision Log

| Date | Decision |
|------|----------|
| 2026-07-05 | Omit unset optional attrs instead of sentinel values (-1, -999) on spans |
| 2026-07-05 | Default backend profile in **docs/examples** = `langfuse` for best DX; library default = `latest` |
| 2026-07-05 | Images excluded from events unless explicit opt-in |
