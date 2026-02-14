# Iteration 08 - Observe

## Observation Target
Observability subsystem: `TracingProvider`, `genai_events`, `InferenceMetrics`, and middleware logging/metrics.

## Findings

### TracingProvider (src/providers/tracing.rs, 676 lines)
- Decorator pattern wrapping any `LLMProvider`
- Creates `info_span!` for every operation: complete, chat, chat_with_tools, stream, chat_with_tools_stream
- Uses `genai_attrs` module with 12 OpenTelemetry GenAI semantic convention constants
- Privacy-by-default: content captured only when `EDGECODE_CAPTURE_CONTENT=true`
- Reasoning/thinking token tracking via custom `gen_ai.usage.reasoning_tokens` attribute
- Thinking content truncated to 10 KB to prevent trace bloat
- Langfuse integration via `langfuse.observation.input` / `langfuse.observation.output` fields
- 5 unit tests covering delegation, span creation, and unwrapping

### GenAI Events (src/providers/genai_events.rs, 276 lines)
- Emits `gen_ai.client.inference.operation.details` span events per OpenTelemetry spec
- Converts `ChatMessage` to tagged `GenAIMessage` format (text, tool_call, tool_result parts)
- Captures comprehensive metadata: response_id, finish_reason, cache_hit_tokens, all penalties
- Uses sentinel values (-1.0, -999.0, 0) for optional fields to ensure Jaeger compatibility
- 5 tests covering message conversion, content capture toggle, JSON serialization

### InferenceMetrics (src/inference_metrics.rs, 533 lines)
- Per-stream metrics: TTFT, tokens/second, thinking progress, estimated tokens
- Provider TTFT priority over client-measured TTFT
- Display formatting: `format_ttft()`, `format_rate()`, `format_thinking_progress()`
- Token estimation heuristic: 4 chars per token
- 11 comprehensive tests covering all calculation paths

### Middleware Observability (src/middleware.rs, lines 220-445)
- `LoggingLLMMiddleware`: 3 log levels (Info, Debug, Trace) with structured fields
- `MetricsLLMMiddleware`: 8 atomic counters (requests, tokens, latency, cache, tools)
- `get_cache_hit_rate()` computes percentage of prompt tokens served from cache
- `get_summary()` returns `MetricsSummary` snapshot
