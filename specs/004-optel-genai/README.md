# 004 — OpenTelemetry GenAI Semantic Conventions

> **Date**: 2026-07-05  
> **Status**: 📋 PLANNED  
> **Target Version**: edgequake-llm v0.11.0  
> **Scope**: Align edgequake-llm observability with the latest [OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai), and make Rust LLM applications export traces that work out-of-the-box with Langfuse, LangSmith, Datadog LLM Observability, Grafana Tempo, and other OTLP backends.

---

## Problem Statement

edgequake-llm already ships `TracingProvider`, `genai_events`, and middleware metrics — but the implementation predates several breaking shifts in the GenAI semconv (v1.37+):

1. **`gen_ai.system` → `gen_ai.provider.name`** (required discriminator)
2. **Span naming**: `{operation} {model}` instead of `gen_ai.chat`
3. **Content capture**: span attributes deprecated in favor of **`gen_ai.client.inference.operation.details` events**
4. **Message schema**: `parts[]` with typed content, not ad-hoc `content[]`
5. **Token usage renames**: `cache_read.input_tokens`, `reasoning.output_tokens`
6. **Dual-emit transition**: `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`

Meanwhile, LLM observability platforms (Langfuse, LangSmith, Datadog, OpenLLMetry consumers) are converging on `gen_ai.*` — but each backend still has different attribute/event parsing maturity. Applications using edgequake-llm should **instrument once** and route via standard OTLP without vendor SDK lock-in.

**Non-goals (v1 of this spec):**

- Agent/workflow/MCP span hierarchies (future spec 005)
- Full OTel Metrics SDK integration (histogram instruments) — design included, P2 delivery
- Replacing user-configured OTLP exporters — we emit correct signals; export stays user-owned
- Automatic PII redaction beyond existing privacy defaults

---

## Documents

| # | Document | Purpose |
|---|----------|---------|
| 01 | [Five-WHY Analysis](./01-FIVE-WHY-ANALYSIS.md) | Root cause: why current telemetry is insufficient |
| 02 | [OTEL GenAI Spec Cross-Reference](./02-OTEL-GENAI-SPEC-CROSSREF.md) | Authoritative spec mapping with version timeline |
| 03 | [Backend Compatibility Matrix](./03-BACKEND-COMPATIBILITY-MATRIX.md) | Langfuse, LangSmith, Datadog, Jaeger, OpenInference |
| 04 | [Architecture, Gaps & SOLID Design](./04-ARCHITECTURE-AND-GAP-ANALYSIS.md) | Target module layout, DRY refactor, code anchors |
| 05 | [Implementation Plan](./05-IMPLEMENTATION-PLAN.md) | Phased delivery, ADRs, file map, test strategy |
| 06 | [Edge Case Register](./06-EDGE-CASES.md) | Streaming, tools, privacy, backend quirks |

---

## Current Code Anchors (as-is)

| Module | File | Role |
|--------|------|------|
| Span decorator | `src/providers/tracing.rs` | `TracingProvider`, `genai_attrs` constants |
| Span events | `src/providers/genai_events.rs` | `gen_ai.client.inference.operation.details` |
| Application attribution | `src/application_context.rs` | `gen_ai.application.*` span fields |
| Streaming metrics | `src/inference_metrics.rs` | TTFT, tokens/s (not yet OTel metrics) |
| Middleware | `src/middleware.rs` | Aggregate counters (parallel to OTel) |
| Docs | `docs/observability.md` | User-facing setup guide |

---

## Success Criteria

- [ ] **Spec compliance (latest experimental)**: Required inference span attributes per [gen-ai-spans.md](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)
- [ ] **Backward compatibility**: Default emit v1.36-style attributes; opt-in latest via `EDGEQUAKE_OTEL_SEMCONV_OPT_IN=gen_ai_latest_experimental`
- [ ] **Backend matrix green**: Documented pass/fail for Langfuse OTLP, LangSmith OTLP, Jaeger, with workarounds
- [ ] **DRY**: Single `GenAiSpanBuilder` — no duplicated attribute recording across 6+ `TracingProvider` methods
- [ ] **Privacy-by-default**: Content on events only when `EDGECODE_CAPTURE_CONTENT=true`; span attributes never carry raw prompts by default in latest mode
- [ ] **Provider name registry**: Canonical `gen_ai.provider.name` for all 16+ production providers
- [ ] **Streaming**: `gen_ai.request.stream=true`, TTFT on span, usage recorded when stream completes
- [ ] **Tests**: Snapshot tests for span attributes + event payloads; e2e OTLP export test with mock collector
- [ ] **Examples**: `examples/advanced/otel_langfuse.rs`, `examples/advanced/otel_langsmith.rs`
- [ ] **Docs**: Updated `docs/observability.md` + migration section in `CHANGELOG.md`

---

## Official References (July 2026)

| Resource | URL |
|----------|-----|
| **GenAI semconv repo (canonical)** | https://github.com/open-telemetry/semantic-conventions-genai |
| Spans | https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md |
| Events | https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md |
| Metrics | https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md |
| Attribute registry | https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/registry/attributes/gen-ai.md |
| Migration notice (core repo) | https://opentelemetry.io/docs/specs/semconv/gen-ai/ |
| Stability opt-in | `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` |
| Langfuse OTLP | https://langfuse.com/integrations/native/opentelemetry |
| LangSmith OTLP | https://docs.langchain.com/langsmith/trace-with-opentelemetry |
| Datadog GenAI OTel | https://www.datadoghq.com/blog/llm-otel-semantic-convention/ |

---

## Relationship to Spec 003

Spec [003-application-llm-header](../003-application-llm-header/README.md) covers **provider HTTP attribution** (`ApplicationContext` → upstream headers). Spec 004 covers **telemetry export** (`ApplicationContext` → span attributes). Both read from the same `ApplicationContext`; resolvers remain separate (SRP).

```text
ApplicationContext
       ├──► http::attribution  → provider headers (003)
       └──► GenAiSpanBuilder   → gen_ai.application.* spans (004)
```
