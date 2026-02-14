# Observability

EdgeQuake LLM provides a multi-layered observability stack covering distributed tracing, structured logging, real-time streaming metrics, and aggregate usage tracking. The implementation follows [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) so traces export directly to Jaeger, Grafana Tempo, or any OTLP-compatible backend.

```text
Application
    |
    v
+-------------------+     +-------------------+     +------------------+
| TracingProvider   |---->| LoggingMiddleware |---->| MetricsMiddleware|
| (per-call spans)  |     | (structured logs) |     | (aggregate stats)|
+-------------------+     +-------------------+     +------------------+
    |                                                       |
    v                                                       v
+-------------------+     +-------------------+     +------------------+
| GenAI Events      |     | InferenceMetrics  |     | CostTracker      |
| (span events)     |     | (TTFT, t/s)       |     | (USD per session)|
+-------------------+     +-------------------+     +------------------+
    |                           |                         |
    v                           v                         v
  OTLP Exporter            Display Layer            Budget Alerts
  (Jaeger / Tempo)         (TUI / CLI)             (threshold checks)
```

---

## TracingProvider

`TracingProvider<P>` is a decorator that wraps any `LLMProvider` and emits OpenTelemetry-compatible `tracing` spans for every call.

### Wrapped Operations

| Method | Span Name | Key Attributes |
|--------|-----------|----------------|
| `complete()` | `gen_ai.complete` | operation, system, model, prompt_length |
| `complete_with_options()` | `gen_ai.complete` | + max_tokens, temperature, top_p |
| `chat()` | `gen_ai.chat` | + message_count, reasoning_tokens |
| `chat_with_tools()` | `gen_ai.chat_with_tools` | + tool_count |
| `stream()` | `gen_ai.stream` | operation, system, model |
| `chat_with_tools_stream()` | `gen_ai.stream_with_tools` | + tool_count |

### GenAI Semantic Convention Attributes

All spans carry standardised attributes defined in `genai_attrs`:

| Attribute | Constant | Description |
|-----------|----------|-------------|
| `gen_ai.operation.name` | `OPERATION_NAME` | "chat", "complete", etc. |
| `gen_ai.system` | `SYSTEM` | Provider name (openai, azure, ...) |
| `gen_ai.request.model` | `REQUEST_MODEL` | Requested model |
| `gen_ai.request.max_tokens` | `REQUEST_MAX_TOKENS` | Token limit |
| `gen_ai.request.temperature` | `REQUEST_TEMPERATURE` | Sampling temperature |
| `gen_ai.request.top_p` | `REQUEST_TOP_P` | Nucleus sampling |
| `gen_ai.response.model` | `RESPONSE_MODEL` | Actual model used |
| `gen_ai.usage.input_tokens` | `USAGE_INPUT_TOKENS` | Prompt tokens |
| `gen_ai.usage.output_tokens` | `USAGE_OUTPUT_TOKENS` | Completion tokens |
| `gen_ai.usage.reasoning_tokens` | `USAGE_REASONING_TOKENS` | Thinking tokens |
| `gen_ai.response.finish_reasons` | `RESPONSE_FINISH_REASONS` | Stop reason |
| `gen_ai.reasoning.content` | `REASONING_CONTENT` | Thinking text (opt-in) |

### Usage

```rust
use edgequake_llm::providers::{OpenAIProvider, TracingProvider};

let provider = OpenAIProvider::new("sk-...");
let traced = TracingProvider::new(provider);

// Every call now emits a span with GenAI attributes
let response = traced.chat(&messages, None).await?;

// Access the inner provider if needed
let inner = traced.inner();

// Unwrap the decorator
let provider = traced.into_inner();
```

### Content Capture (Privacy-by-Default)

Prompt and response content is **not** recorded in spans unless you opt in:

```bash
export EDGECODE_CAPTURE_CONTENT=true
```

When enabled, the following fields are populated:
- `gen_ai.prompt` - serialised input messages (JSON)
- `gen_ai.completion.content` - response text
- `gen_ai.reasoning.content` - thinking block (truncated to 10 KB)
- `langfuse.observation.input` / `langfuse.observation.output` - for Langfuse integration

---

## GenAI Events

The `genai_events` module emits structured span events conforming to [OpenTelemetry GenAI Event Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/). These appear as log entries within the parent span timeline in Jaeger.

Event name: `gen_ai.client.inference.operation.details`

### Event Payload

| Field | Type | Description |
|-------|------|-------------|
| `gen_ai.input.messages` | JSON array | Input messages in GenAI format |
| `gen_ai.output.messages` | JSON array | Output messages in GenAI format |
| `gen_ai.response.id` | string | Provider response ID |
| `gen_ai.response.finish_reasons` | string | Why generation stopped |
| `gen_ai.usage.input_tokens` | i64 | Prompt tokens |
| `gen_ai.usage.output_tokens` | i64 | Completion tokens |
| `gen_ai.usage.cache_hit_tokens` | i64 | Tokens served from cache |
| `gen_ai.request.temperature` | f64 | Temperature (-1.0 = unset) |
| `gen_ai.request.max_tokens` | i64 | Max tokens (0 = unset) |
| `gen_ai.request.top_p` | f64 | Top-p (-1.0 = unset) |
| `gen_ai.request.frequency_penalty` | f64 | Frequency penalty (-999.0 = unset) |
| `gen_ai.request.presence_penalty` | f64 | Presence penalty (-999.0 = unset) |

### Message Format

Messages are converted from `ChatMessage` to the GenAI schema with tagged parts:

```json
{
  "role": "assistant",
  "content": [
    { "type": "text", "text": "Let me search for that." },
    { "type": "tool_call", "tool_call": { "id": "call_123", "name": "web_search", "arguments": "{\"query\":\"test\"}" } }
  ]
}
```

Events are only emitted when `EDGECODE_CAPTURE_CONTENT=true`.

---

## Middleware Observability

The middleware stack provides two built-in observability middlewares with `before()` / `after()` hooks that run around every provider call.

### LoggingLLMMiddleware

Structured logging at three levels:

| Level | `before()` | `after()` |
|-------|-----------|----------|
| **Info** | provider, model, message count, tool count | model, total tokens, duration, finish reason |
| **Debug** | + last message preview (100 chars) | + content preview (200 chars), tool call count |
| **Trace** | full message dump | full response dump |

```rust
use edgequake_llm::middleware::{LoggingLLMMiddleware, LogLevel, LLMMiddlewareStack};
use std::sync::Arc;

let mut stack = LLMMiddlewareStack::new();
stack.add(Arc::new(LoggingLLMMiddleware::with_level(LogLevel::Debug)));
```

### MetricsLLMMiddleware

Aggregate counters using `AtomicU64` for lock-free concurrent access:

| Metric | Method | Description |
|--------|--------|-------------|
| Total requests | `get_total_requests()` | Number of LLM calls |
| Total tokens | `get_total_tokens()` | Prompt + completion tokens |
| Prompt tokens | `prompt_tokens` field | Input tokens only |
| Completion tokens | `completion_tokens` field | Output tokens only |
| Average latency | `get_average_latency_ms()` | Mean response time |
| Cache hit tokens | `get_cache_hit_tokens()` | Tokens served from KV cache |
| Cache hit rate | `get_cache_hit_rate()` | Percentage of prompt tokens cached |
| Tool call requests | `tool_call_requests` field | Requests that used tools |

```rust
use edgequake_llm::middleware::{MetricsLLMMiddleware, LLMMiddlewareStack};
use std::sync::Arc;

let metrics = Arc::new(MetricsLLMMiddleware::new());
let mut stack = LLMMiddlewareStack::new();
stack.add(metrics.clone());

// After some calls...
let summary = metrics.get_summary();
println!("Requests: {}", summary.total_requests);
println!("Avg latency: {:.1}ms", metrics.get_average_latency_ms());
println!("Cache hit rate: {:.1}%", metrics.get_cache_hit_rate());
```

---

## InferenceMetrics

`InferenceMetrics` provides real-time streaming metrics for a single LLM inference operation. It tracks timing, token counts, and thinking progress for display layers (TUI, CLI).

```text
Provider Stream --> InferenceMetrics --> Display Layer
     |                    |                   |
     v                    v                   v
 StreamChunk       record_first_token()   ttft_ms()
 ThinkingContent   add_output_tokens()    tokens_per_second()
 Finished          add_thinking_tokens()  format_thinking_progress()
                   set_provider_ttft()    format_ttft()
                                          format_rate()
```

### Key Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `ttft_ms()` | `Option<f64>` | Time to first token (prefers provider-reported) |
| `tokens_per_second()` | `f64` | Output token generation rate |
| `total_tokens_per_second()` | `f64` | Output + thinking token rate |
| `estimated_tokens()` | `usize` | Heuristic: chars_received / 4 |
| `format_ttft()` | `Option<String>` | "850ms" or "1.2s" |
| `format_rate()` | `String` | "42 t/s" or "3.5 t/s" |
| `format_thinking_progress()` | `Option<String>` | "1.5k/10.0k" |

### TTFT Priority

Provider-reported TTFT is preferred over client-measured TTFT because it excludes network latency:

```text
ttft_ms() returns:
  1. provider_ttft_ms if set    (native, most accurate)
  2. measured first_token_time  (client-side, includes network)
```

### Usage

```rust
use edgequake_llm::InferenceMetrics;

let mut metrics = InferenceMetrics::new();

// First token arrives
metrics.record_first_token();

// Accumulate tokens from stream chunks
metrics.add_output_tokens(5);
metrics.add_chars(20);

// Thinking tokens (Claude, o-series)
metrics.add_thinking_tokens(100);
metrics.set_thinking_budget(10_000);

// Display
if let Some(ttft) = metrics.format_ttft() {
    println!("TTFT: {ttft}");
}
println!("Rate: {}", metrics.format_rate());
if let Some(progress) = metrics.format_thinking_progress() {
    println!("Thinking: {progress}");
}
```

---

## Connecting to Jaeger

1. Start Jaeger:
   ```bash
   docker run -d --name jaeger \
     -p 16686:16686 -p 4317:4317 \
     jaegertracing/all-in-one:latest
   ```

2. Configure your application with `tracing-opentelemetry`:
   ```rust
   use tracing_subscriber::prelude::*;
   use opentelemetry::trace::TracerProvider;
   use opentelemetry_otlp::WithExportConfig;

   let exporter = opentelemetry_otlp::SpanExporter::builder()
       .with_tonic()
       .with_endpoint("http://localhost:4317")
       .build()?;

   let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
       .with_batch_exporter(exporter)
       .build();

   let tracer = provider.tracer("edgequake-llm");

   tracing_subscriber::registry()
       .with(tracing_opentelemetry::layer().with_tracer(tracer))
       .with(tracing_subscriber::fmt::layer())
       .init();
   ```

3. Wrap your provider:
   ```rust
   let provider = TracingProvider::new(inner_provider);
   // All calls now appear in Jaeger with GenAI attributes
   ```

4. Open Jaeger UI at `http://localhost:16686` and search for `gen_ai.chat` spans.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGECODE_CAPTURE_CONTENT` | `false` | Enable prompt/response capture in traces |
| `RUST_LOG` | - | Control log levels (`edgequake_llm=debug`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | - | OTLP collector endpoint |

---

## See Also

- [Architecture](architecture.md) - middleware pipeline design
- [Cost Tracking](cost-tracking.md) - financial observability via `SessionCostTracker`
- [Rate Limiting](rate-limiting.md) - backpressure metrics
- [OpenTelemetry GenAI Spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)
