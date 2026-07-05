# 01 — Five-WHY Root Cause Analysis

> **Lens**: Observability / Platform Engineer  
> **Date**: 2026-07-05  
> **Status**: DRAFT

## Problem Statement

edgequake-llm applications cannot reliably plug into Langfuse, LangSmith, Datadog LLM Observability, or generic OTel backends without custom span processors, duplicate instrumentation, or missing input/output in the UI — despite the crate already advertising GenAI semconv support.

---

## Five-WHY Deep Dive

### WHY #1: Why do Langfuse / LangSmith show empty inputs or wrong model names?

**Because** backends map OTLP spans using **specific attribute keys and event formats** that differ from what edgequake-llm emits today. Langfuse prioritizes `langfuse.*` > `gen_ai.*` > OpenInference; LangSmith still documents legacy `gen_ai.prompt.{n}.content` alongside newer `gen_ai.input.messages`. edgequake-llm writes `gen_ai.prompt` (JSON blob) on span attributes and uses deprecated `gen_ai.system` instead of required `gen_ai.provider.name`.

```text
  Backend expects                    edgequake-llm emits (today)
  ─────────────────                  ───────────────────────────
  gen_ai.provider.name = "openai"    gen_ai.system = "openai"
  gen_ai.input.messages (structured)  gen_ai.prompt (JSON string on span)
  event: gen_ai.client.inference...   event exists ✓ but wrong message schema
  span name: "chat gpt-4o"            span name: "gen_ai.chat"
```

### WHY #2: Why does edgequake-llm emit outdated attribute names?

**Because** `TracingProvider` was implemented against **pre-v1.37 semconv** (still linked from `docs/observability.md` to the moved opentelemetry.io path). The GenAI conventions moved to a dedicated repo in v1.42.0 of core semconv; v1.37 replaced `gen_ai.system` with `gen_ai.provider.name` and moved prompt/completion content from span attributes to events. No migration layer or opt-in flag was added.

**Code evidence** (`src/providers/tracing.rs`):

```rust
pub const SYSTEM: &str = "gen_ai.system";  // deprecated since v1.37
// Span: info_span!("gen_ai.chat", ...)   // should be "chat {model}"
// Operation: "chat_with_tools"            // should be "chat" per enum
```

### WHY #3: Why wasn't the instrumentation updated when the spec evolved?

**Because** observability was implemented as a **monolithic decorator** (`tracing.rs`, ~850 lines) with inline attribute strings duplicated across six methods (`complete`, `chat`, `chat_with_tools`, `stream`, …). There is no central `GenAiSpanBuilder`, no semconv version enum, and no conformance tests against the JSON schemas in `semantic-conventions-genai/model/gen-ai/`. Spec drift is invisible until a backend breaks.

```text
  DRY violation (today):

  complete() ──────┐
  complete_with_options() ──┼──► copy-paste span.record(...) blocks
  chat() ──────────┤
  chat_with_tools()┤
  stream() ────────┤
  chat_with_tools_stream()
```

### WHY #4: Why is there no conformance layer or backend compatibility testing?

**Because** the project optimized for **Jaeger + tracing crate** validation (manual eyeballing in UI) rather than **OTLP contract tests**. `genai_events.rs` uses a custom message shape (`content: [{ type: "text", text }]`) that does not match the official [gen-ai-input-messages.json](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/model/gen-ai/gen-ai-input-messages.json) schema (`parts` with `content` field). Events are gated behind `EDGECODE_CAPTURE_CONTENT` but the spec treats event emission as opt-in for *content*, not for metadata.

### WHY #5: Why does the architecture lack a pluggable semconv version strategy?

**Because** edgequake-llm evolved as a **provider abstraction first**, observability second. The decorator pattern (`TracingProvider<P>`) is correct (OCP), but attribute naming lives in a public `genai_attrs` module with no **Strategy** for semconv versions, no **Factory** for span builders, and no **Interface Segregation** between "always emit" (operation, provider, model) vs "opt-in emit" (messages, tools). The root gap is missing **telemetry as a first-class domain module**, not missing tracing altogether.

---

## Root Cause Summary

```text
┌─────────────────────────────────────────────────────────────────────┐
│                     ROOT CAUSE DIAGRAM                               │
│                                                                      │
│  Missing: observability::genai module with versioned semconv strategy  │
│      │                                                               │
│      ├── Stale attribute names (gen_ai.system, custom operations)    │
│      ├── Wrong span naming & missing CLIENT kind                     │
│      ├── Message schema mismatch (content vs parts)                  │
│      ├── No dual-emit / OTEL_SEMCONV_STABILITY_OPT_IN bridge         │
│      ├── Streaming gaps (no usage, no TTFT on span)                  │
│      └── No backend compatibility contract tests                     │
│                                                                      │
│  Impact: vendor-neutral OTLP export fails silently in Langfuse etc.  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Impact Analysis

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| Langfuse / LangSmith adoption | **CRITICAL** | Empty I/O, wrong filters, manual workarounds |
| Multi-backend routing (OTel Collector) | **HIGH** | Same trace, different parsing per vendor |
| Cost / token dashboards | **HIGH** | Wrong cache/reasoning token attribute names |
| Compliance / PII | **HIGH** | Legacy span-attribute content capture vs event-only |
| Agent frameworks on edgequake-llm | **MEDIUM** | No parent/child span helpers for tools (P2) |
| Datadog / Grafana GenAI dashboards | **MEDIUM** | Panel filters expect v1.37+ attribute set |

---

## Design Principles Derived

| Principle | Application |
|-----------|-------------|
| **Instrument once, export anywhere** | Emit standards-compliant OTLP; backends map locally |
| **Code is law** | Every attribute in spec must trace to `GenAiSpanBuilder` code path |
| **DRY** | One builder, one message converter, one provider-name registry |
| **SOLID — SRP** | Split span attrs, events, metrics, backend shims |
| **SOLID — OCP** | New semconv version = new `SemConvVersion` strategy, not fork `tracing.rs` |
| **Privacy-by-default** | Latest mode: metadata on spans, content on events only when opted in |
| **Ascending compatibility** | Default = legacy emit; opt-in = latest experimental |
