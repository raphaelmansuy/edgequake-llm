# 03 — Backend Compatibility Matrix

> **Purpose**: Map edgequake-llm OTLP output to Langfuse, LangSmith, Datadog, Jaeger/Tempo, OpenInference consumers — with known gaps and shims.  
> **Strategy**: Emit **standards-first** + optional **backend profile** for maximum compatibility without vendor SDKs.

---

## 1. Integration Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│                     edgequake-llm Application                         │
│  TracingProvider → tracing → tracing-opentelemetry → OTLP exporter   │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ OTLP/HTTP (protobuf or JSON)
                                ▼
              ┌─────────────────────────────────────┐
              │     OpenTelemetry Collector (opt.)     │
              │  transform / filter / multi-export   │
              └─────────┬───────────┬─────────┬───────┘
                        │           │         │
            ┌───────────▼──┐  ┌─────▼────┐  ┌─▼──────────┐
            │   Langfuse   │  │LangSmith │  │  Datadog   │
            │ /api/public/ │  │ /otel/v1 │  │  Agent     │
            │    otel      │  │ /traces  │  │  OTLP mode │
            └──────────────┘  └──────────┘  └────────────┘
```

**Design rule**: edgequake-llm does **not** embed Langfuse/LangSmith SDKs. It emits OTLP; users choose the backend via exporter config.

---

## 2. Backend Summary Matrix

| Backend | OTLP ingest | GenAI semconv version | Primary docs | edgequake target profile |
|---------|-------------|----------------------|--------------|--------------------------|
| **Langfuse** | ✅ `/api/public/otel` | v1.36 attrs + partial v1.37 events | [langfuse.com/.../opentelemetry](https://langfuse.com/integrations/native/opentelemetry) | `langfuse` |
| **LangSmith** | ✅ `api.smith.langchain.com/otel/v1/traces` | v1.36 + `gen_ai.input.messages` | [docs.langchain.com/.../trace-with-opentelemetry](https://docs.langchain.com/langsmith/trace-with-opentelemetry) | `langsmith` |
| **Datadog** | ✅ Agent OTLP + LLM Observability | v1.37+ native | [datadoghq.com/blog/llm-otel](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) | `latest` (default for DD) |
| **Grafana Tempo** | ✅ Generic OTLP | Generic trace search | Grafana docs | `latest` |
| **Jaeger** | ✅ Generic OTLP | No GenAI UI — raw attributes | Jaeger docs | `latest` |
| **Honeycomb / Axiom** | ✅ Generic OTLP | Query on `gen_ai.*` | Vendor docs | `latest` |
| **OpenLLMetry / Traceloop** | ✅ | Legacy token names on some versions | traceloop.com | `legacy` or collector transform |

---

## 3. Langfuse

### 3.1 OTLP setup

```bash
OTEL_EXPORTER_OTLP_ENDPOINT="https://cloud.langfuse.com/api/public/otel"
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic ${LANGFUSE_AUTH},x-langfuse-ingestion-version=4"
# Signal-specific:
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="https://cloud.langfuse.com/api/public/otel/v1/traces"
```

Ref: [Langfuse OTEL docs](https://langfuse.com/integrations/native/opentelemetry)

### 3.2 Attribute precedence (Langfuse)

1. `langfuse.*` namespace (**highest**)
2. `gen_ai.*` semantic conventions
3. OpenInference (`input.value`, `output.value`)
4. Generic attributes

Ref: [Langfuse AttributeMapper](https://langfuse.hexdocs.pm/Langfuse.OpenTelemetry.AttributeMapper.html)

### 3.3 Mapping table — edgequake-llm targets

| Langfuse field | OTel source (priority order) | edgequake action |
|----------------|------------------------------|------------------|
| `input` | `langfuse.observation.input` > `gen_ai.input.messages` > `gen_ai.prompt` | Profile `langfuse`: keep `langfuse.observation.*` when capture on |
| `output` | `langfuse.observation.output` > `gen_ai.output.messages` > `gen_ai.completion` | Same |
| `model` | `gen_ai.request.model`, `gen_ai.response.model` | ✓ already |
| `usage` | `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` | Fix cache/reasoning key names |
| `userId` | `langfuse.user.id`, `user.id` | Document — set via app baggage |
| `sessionId` | `langfuse.session.id`, `session.id` | Document |
| `observation.type` | `langfuse.observation.type=generation` | ✓ on tool spans |

### 3.4 Known Langfuse gaps (July 2026)

| Issue | Impact | edgequake mitigation |
|-------|--------|---------------------|
| [Issue #12657](https://github.com/langfuse/langfuse/issues/12657): `gen_ai.client.inference.operation.details` events not parsed for I/O | Empty input/output when using latest event-only mode | **`BackendProfile::Langfuse`**: dual-write `langfuse.observation.input/output` on span + emit event |
| Filter requires `gen_ai.system` on some versions | Spans dropped from LLM view | Dual-emit `gen_ai.system` alongside `gen_ai.provider.name` |
| Trace-level attrs need propagation | Filtering by session fails | Document BaggageSpanProcessor pattern (see spec 003 §2.2) |

---

## 4. LangSmith

### 4.1 OTLP setup

```bash
OTEL_EXPORTER_OTLP_ENDPOINT="https://api.smith.langchain.com/otel/v1/traces"
OTEL_EXPORTER_OTLP_HEADERS="x-api-key=${LANGSMITH_API_KEY},Langsmith-Project=${PROJECT_NAME}"
# Or hybrid:
LANGSMITH_OTEL_ENABLED=true
```

Ref: [Trace with OpenTelemetry](https://docs.langchain.com/langsmith/trace-with-opentelemetry)

### 4.2 Mapping table (from LangSmith docs)

| LangSmith field | OTel attribute | edgequake action |
|-----------------|----------------|------------------|
| Run type | `langsmith.span.kind` or `gen_ai.operation.name` | Map `chat` → LLM run; set `langsmith.span.kind=LLM` in profile |
| `inputs` | `gen_ai.prompt`, `gen_ai.input.messages`, `gen_ai.prompt.{n}.content` | Profile `langsmith`: populate `gen_ai.input.messages` JSON on span when capture on |
| `outputs` | `gen_ai.completion`, `gen_ai.output.messages` | Same |
| `metadata.ls_provider` | `gen_ai.system` | Dual-emit `gen_ai.system` |
| `invocation_params.*` | `gen_ai.request.*` | ✓ mostly covered |
| `usage_metadata.input_tokens` | `gen_ai.usage.input_tokens` | ✓ |
| `usage_metadata.output_tokens` | `gen_ai.usage.output_tokens` | ✓ |
| Session / experiment | `langsmith.trace.session_id`, `langsmith.reference_example_id` | App responsibility |

### 4.3 LangSmith legacy token names

LangSmith docs still show:

```python
span.set_attribute("gen_ai.usage.prompt_tokens", ...)
span.set_attribute("gen_ai.usage.completion_tokens", ...)
```

**Profile `langsmith`**: optionally dual-emit deprecated token attribute names (OpenLLMetry compat) alongside `input_tokens` / `output_tokens`.

---

## 5. Datadog LLM Observability

Native support for OTel GenAI semconv **v1.37+** (no Datadog SDK required).

Ref: [Datadog blog — LLM OTel semconv](https://www.datadoghq.com/blog/llm-otel-semantic-convention/)

| Requirement | edgequake action |
|-------------|------------------|
| `gen_ai.provider.name` | P0 — required for GenAI span detection |
| `gen_ai.operation.name` | P0 — official enum values |
| Token usage on span | P0 |
| Events for content | Supported in DD — prefer latest mode |

**Recommended profile**: `BackendProfile::Latest` (no legacy shim).

---

## 6. Generic OTLP (Jaeger / Tempo / Grafana)

| Need | edgequake action |
|------|------------------|
| Searchable span names | `{operation} {model}` |
| GenAI filter | `gen_ai.provider.name` attribute |
| Content in timeline | Span events visible in Jaeger "Logs" tab |
| Cost dashboards | Usage attrs on span |

No vendor-specific shims required.

---

## 7. Backend Profile Feature (ADR preview)

```rust
pub enum BackendProfile {
    /// OTel GenAI latest experimental — events for content, provider.name
    Latest,
    /// v1.36 dual-emit for older collectors
    Legacy,
    /// Langfuse OTLP: langfuse.observation.* + gen_ai.* + dual system
    Langfuse,
    /// LangSmith OTLP: input.messages on span + langsmith.span.kind + legacy tokens
    LangSmith,
    /// Maximum compatibility: all of the above (trace size cost)
    MaxCompatibility,
}
```

Configure via:

```bash
EDGEQUAKE_OTEL_BACKEND_PROFILE=langfuse   # default: latest
```

Implementation in `GenAiSpanBuilder::finish()` — **single code path**, profile selects attribute sets (DRY).

---

## 8. OpenInference / Arize / Phoenix (secondary)

| OpenInference attr | GenAI equivalent | Shim |
|--------------------|------------------|------|
| `llm.model_name` | `gen_ai.request.model` | Emit both in `MaxCompatibility` |
| `llm.token_count.prompt` | `gen_ai.usage.input_tokens` | Emit both |
| `openinference.span.kind` | `gen_ai.operation.name` | Map LLM → chat |

Ref: [OpenInference semconv](https://github.com/Arize-ai/openinference)

---

## 9. Collector-Side Normalization (recommended for production)

For multi-backend or MaxCompatibility avoidance, use OTel Collector `transform` processor:

```yaml
processors:
  transform/genai_normalize:
    trace_statements:
      - context: span
        statements:
          - set(attributes["gen_ai.provider.name"], attributes["gen_ai.system"]) where attributes["gen_ai.provider.name"] == nil
          - set(attributes["gen_ai.usage.input_tokens"], attributes["gen_ai.usage.prompt_tokens"]) where attributes["gen_ai.usage.input_tokens"] == nil
```

edgequake-llm should ship **`docs/observability/collector-genai.yaml`** example (P2 docs).

---

## 10. Compatibility Test Matrix (CI targets)

| Test case | Langfuse | LangSmith | Jaeger |
|-----------|----------|-----------|--------|
| Chat completion w/ capture | I/O visible | I/O visible | Events in log tab |
| Chat w/o capture | Model + tokens only | Same | No message attrs |
| Tool call response | Tool JSON in output | outputs.messages | ✓ |
| Streaming completion | TTFT + tokens at end | Partial (usage at end) | stream attr set |
| Error response | error.type set | Error run | Span ERROR status |
| Multi-provider (OpenRouter) | provider.name=openai | ls_provider set | routing attr |

E2E tests use **`otelcol-contrib` test container** + HTTP mock for Langfuse/LangSmith ingest (record payload, assert JSON paths).

---

## 11. User-Facing Quick Start Snippets

### Langfuse (Rust)

```bash
export EDGECODE_CAPTURE_CONTENT=true
export EDGEQUAKE_OTEL_BACKEND_PROFILE=langfuse
export OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel/v1/traces
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic $(echo -n pk:sk | base64),x-langfuse-ingestion-version=4"
```

### LangSmith (Rust)

```bash
export EDGEQUAKE_OTEL_BACKEND_PROFILE=langsmith
export OTEL_EXPORTER_OTLP_ENDPOINT=https://api.smith.langchain.com/otel/v1/traces
export OTEL_EXPORTER_OTLP_HEADERS="x-api-key=$LANGSMITH_API_KEY,Langsmith-Project=my-app"
```

See [05-IMPLEMENTATION-PLAN.md](./05-IMPLEMENTATION-PLAN.md) for example binaries.
