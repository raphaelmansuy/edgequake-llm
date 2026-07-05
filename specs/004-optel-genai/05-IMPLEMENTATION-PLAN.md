# 05 — Implementation Plan

> **Version**: 1.0 — 2026-07-05  
> **Target**: edgequake-llm v0.11.0  
> **Principles**: First Principles · DRY · SOLID · Ascending Compatibility · Battle-Tested  
> **Status**: 📋 PLANNED

---

## Table of Contents

1. [Architecture Decisions (ADRs)](#1-architecture-decisions-adrs)
2. [Phase 0 — Foundation](#2-phase-0--foundation)
3. [Phase 1 — Span & Event Compliance (P0)](#3-phase-1--span--event-compliance-p0)
4. [Phase 2 — Streaming & Tools (P0)](#4-phase-2--streaming--tools-p0)
5. [Phase 3 — Backend Profiles (P1)](#5-phase-3--backend-profiles-p1)
6. [Phase 4 — Metrics & Embeddings (P2)](#6-phase-4--metrics--embeddings-p2)
7. [Edge Case Register](#7-edge-case-register)
8. [Testing Strategy](#8-testing-strategy)
9. [Risk Register](#9-risk-register)
10. [File Change Map](#10-file-change-map)

---

## 1. Architecture Decisions (ADRs)

### ADR-G01: Dedicated `observability/` module

**Context**: `tracing.rs` is 850+ lines with duplicated attribute logic.

**Decision**: Extract `src/observability/` per [04-ARCHITECTURE-AND-GAP-ANALYSIS.md](./04-ARCHITECTURE-AND-GAP-ANALYSIS.md).

**Consequences**: Large refactor but single place for semconv changes. `providers/tracing.rs` becomes thin wrapper.

**Rejected**: External crate — premature; keep in-tree until API stabilizes.

---

### ADR-G02: Default legacy emit, opt-in latest

**Context**: OTel spec mandates existing instrumentations keep v1.36 default until stable.

**Decision**:

```rust
SemConvVersion::from_env()  // EDGEQUAKE_OTEL_SEMCONV_OPT_IN
// Default: Legacy (v1.36)
// Opt-in: LatestExperimental
```

**Consequences**: No surprise breaking change for Jaeger users; Datadog/Langfuse users opt in explicitly.

---

### ADR-G03: BackendProfile for vendor shims

**Context**: Langfuse doesn't fully parse v1.37 events yet; LangSmith expects span-level messages.

**Decision**: `BackendProfile` enum controls **additional** attributes; always emit core `gen_ai.*` first.

**Consequences**: `MaxCompatibility` increases trace payload — document tradeoff.

**Rejected**: Separate Langfuse SDK — violates vendor-neutral goal.

---

### ADR-G04: Events for content in Latest mode

**Context**: Spec v1.37+ deprecates content on span attributes.

**Decision**: Latest mode never sets `gen_ai.prompt` / `gen_ai.completion.content` on spans; Langfuse/LangSmith profiles add vendor attrs when capture enabled.

---

### ADR-G05: Observability must never fail LLM calls

**Context**: JSON serialization of messages could panic or error.

**Decision**: All observability code wrapped — log `tracing::warn!` and continue. Unit test: malformed message never returns `Err` from `chat()`.

---

### ADR-G06: Provider name registry centralization

**Context**: 16 providers return different `name()` strings.

**Decision**: `ProviderNameRegistry::resolve(internal_name, ProviderHints)` — single match table, unit tested per provider.

---

## 2. Phase 0 — Foundation

**Goal**: Module skeleton + constants + registry; zero behavior change.

| Task | File(s) | Est. |
|------|---------|------|
| FEAT-100: Create `src/observability/mod.rs` tree | `observability/**` | 0.5d |
| FEAT-101: Move `genai_attrs` → `semconv/attributes.rs` | Move constants | 0.5d |
| FEAT-102: `SemConvVersion` + `BackendProfile` from env | `semconv/version.rs`, `span/backend_profile.rs` | 0.5d |
| FEAT-103: `ProviderNameRegistry` | `semconv/provider_names.rs` | 1d |
| FEAT-104: `OperationKind` enum + mapping | `span/operation.rs` | 0.5d |
| FEAT-105: Re-export back-compat from old paths | `providers/tracing.rs`, `genai_events.rs` | 0.5d |

### Deliverable gate

- [ ] `cargo test` green — no behavior change
- [ ] 30+ unit tests on registry + operation mapping
- [ ] `genai_attrs` constants accessible from old import paths

---

## 3. Phase 1 — Span & Event Compliance (P0)

**Goal**: Correct span names, provider.name, message schema, finish_reasons array.

| Task | File(s) | Est. |
|------|---------|------|
| FEAT-110: `GenAiSpanBuilder` + `ActiveGenAiSpan` | `span/builder.rs` | 2d |
| FEAT-111: Refactor `TracingProvider` to use builder | `providers/tracing.rs` | 1.5d |
| FEAT-112: `MessageConverter` — `parts` schema | `events/messages.rs` | 1d |
| FEAT-113: Refactor `emit_inference_event` | `events/inference_details.rs` | 1d |
| FEAT-114: `record_error` + `error.type` mapping | `error_mapping.rs` | 0.5d |
| FEAT-115: Dual-emit legacy attrs when `SemConvVersion::Legacy` | `builder.rs` | 0.5d |
| FEAT-116: Always emit metadata events (tokens) even w/o content capture | `inference_details.rs` | 0.5d |

### Span builder pseudocode (implement exactly)

```rust
pub fn span_name(operation: &str, model: &str) -> String {
    format!("{} {}", operation, model)
}

pub fn start_span(builder: &GenAiSpanBuilder) -> tracing::Span {
    let provider = ProviderNameRegistry::resolve(builder.provider_internal, &builder.hints);
    let mut span = info_span!(
        target: "gen_ai",
        parent: None,
        // Dynamic name — use tracing span with explicit name:
        &span_name(builder.operation.as_str(), &builder.model),
        { attributes::OPERATION_NAME } = builder.operation.as_str(),
        // ... provider, model at creation
    );
    if builder.semconv.emits_legacy() {
        span.record(attributes::SYSTEM, builder.legacy_system_name());
    }
    if builder.semconv.emits_latest() {
        span.record(attributes::PROVIDER_NAME, provider);
    }
    span
}
```

**Note**: `tracing` macro requires const span names in some versions — use `tracing::info_span!(parent: None, name: ..., ...)` with owned `String` via `Span::current()` pattern or `tracing::span!(Level::INFO, name, ...)` dynamic name support (verify MSRV).

### Deliverable gate

- [ ] Conformance tests pass for `chat`, `complete`, `chat_with_tools`
- [ ] JSON schema validation for messages (dev-dep `jsonschema`)
- [ ] `docs/observability.md` updated attribute table

---

## 4. Phase 2 — Streaming & Tools (P0)

**Goal**: Complete spans for streaming; tool definitions opt-in.

| Task | File(s) | Est. |
|------|---------|------|
| FEAT-120: `StreamObservabilityGuard` — accumulates usage on drop | `span/stream_guard.rs` | 1.5d |
| FEAT-121: Wire `InferenceMetrics` into stream paths | `tracing.rs`, `stream_guard.rs` | 1d |
| FEAT-122: Set `gen_ai.request.stream=true` | `builder.rs` | 0.25d |
| FEAT-123: `gen_ai.response.time_to_first_chunk` from metrics | `builder.rs` | 0.25d |
| FEAT-124: Tool definitions JSON on capture | `events/messages.rs` | 1d |
| FEAT-125: Tool call output in `gen_ai.output.messages` | `messages.rs` | 0.5d |

### Streaming guard pattern

```rust
struct StreamObservabilityGuard {
    span: ActiveGenAiSpan,
    metrics: InferenceMetrics,
    accumulated_text: String,
}

impl Drop for StreamObservabilityGuard {
    fn drop(&mut self) {
        // Record TTFT, gen_ai.request.stream, partial usage if provider sent it
        self.span.record_stream_complete(&self.metrics, &self.accumulated_text);
    }
}
```

**Challenge**: `BoxStream` doesn't easily carry guard — wrap stream with `StreamObservabilityStream` adapter.

### Deliverable gate

- [ ] Streaming chat test records `gen_ai.request.stream=true`
- [ ] TTFT attribute present when provider reports it
- [ ] Tool call e2e: event contains tool_call parts

---

## 5. Phase 3 — Backend Profiles (P1)

**Goal**: Langfuse + LangSmith work out-of-the-box.

| Task | File(s) | Est. |
|------|---------|------|
| FEAT-130: `BackendProfile::Langfuse` shims | `backend_profile.rs` | 1d |
| FEAT-131: `BackendProfile::LangSmith` shims | `backend_profile.rs` | 1d |
| FEAT-132: Legacy token name dual-emit | `backend_profile.rs` | 0.5d |
| FEAT-133: Example `examples/advanced/otel_langfuse.rs` | `examples/` | 0.5d |
| FEAT-134: Example `examples/advanced/otel_langsmith.rs` | `examples/` | 0.5d |
| FEAT-135: Collector example YAML | `docs/observability/collector-genai.yaml` | 0.5d |
| FEAT-136: E2E OTLP payload tests | `tests/e2e_otel_genai_export.rs` | 1.5d |

### Langfuse shim (when profile active + capture)

```rust
if profile.langfuse() && capture {
    span.record("langfuse.observation.input", &input_json);
    span.record("langfuse.observation.output", &output_json);
    span.record("langfuse.observation.type", "generation");
}
// Always also emit gen_ai event for forward compatibility
```

### Deliverable gate

- [ ] Manual verification doc for Langfuse Cloud + LangSmith project
- [ ] CI e2e records OTLP JSON golden files per profile

---

## 6. Phase 4 — Metrics & Embeddings (P2)

| Task | File(s) | Est. |
|------|---------|------|
| FEAT-140: `GenAiMetricsRecorder` histogram | `metrics/recorder.rs` | 2d |
| FEAT-141: `TracingEmbeddingProvider` wrapper | `providers/tracing_embedding.rs` | 1d |
| FEAT-142: Agent `execute_tool` span helper (optional API) | `observability/span/tool.rs` | 2d |
| FEAT-143: Python bindings OTel profile env | `edgequake-litellm` | 1d |

---

## 7. Edge Case Register

See [06-EDGE-CASES.md](./06-EDGE-CASES.md) for full register. Critical P0 items:

| ID | Case | Mitigation |
|----|------|------------|
| EC-G01 | Empty model string | Span name `"chat unknown"`; skip model in name if empty |
| EC-G02 | Finish reason `tool_calls` | `finish_reasons=["tool_calls"]`; output = tool JSON |
| EC-G03 | Reasoning model (o-series) | `gen_ai.usage.reasoning.output_tokens` + opt-in content |
| EC-G04 | Cache tokens partial | Emit only non-zero; don't guess |
| EC-G05 | Stream ends with error mid-way | Record `error.type`; partial usage if available |
| EC-G06 | `EDGECODE_CAPTURE_CONTENT` + huge image multimodal | Truncate binary; omit image bytes from events |
| EC-G07 | OpenRouter actual model ≠ requested | `response.model` on span always |
| EC-G08 | Langfuse event-only → empty UI | Default profile `langfuse` not `latest` in docs |
| EC-G09 | tracing dynamic span names MSRV | Fallback static `"gen_ai.inference"` + attr op/model |
| EC-G10 | Concurrent spans same trace | No change — caller owns parent span context |

---

## 8. Testing Strategy

### Unit tests (~80 new)

| Suite | Location | Coverage |
|-------|----------|----------|
| ProviderNameRegistry | `provider_names.rs` | All 16 providers |
| OperationKind mapping | `operation.rs` | All LLMProvider methods |
| MessageConverter | `messages.rs` | Roles, tools, multimodal text-only |
| BackendProfile shims | `backend_profile.rs` | Attribute keys per profile |
| SemConvVersion | `version.rs` | Env parsing |
| Error mapping | `error_mapping.rs` | Rate limit, timeout, auth |

### Conformance tests

```rust
// tests/genai_semconv_conformance.rs
#[test]
fn chat_span_latest_mode_matches_spec() {
    let recorded = MockSpanHandle::record_chat(...);
    assert_eq!(recorded.name, "chat gpt-4o");
    assert!(recorded.has("gen_ai.provider.name"));
    assert!(!recorded.has("gen_ai.prompt")); // latest mode
}
```

### JSON schema validation

```rust
#[cfg(feature = "genai-schema-tests")]
fn validate_input_messages(json: &str) {
    let schema = include_str!("../specs/004-optel-genai/fixtures/gen-ai-input-messages.json");
    jsonschema::is_valid(schema, json).unwrap();
}
```

Copy official schema to `specs/004-optel-genai/fixtures/` for offline CI.

### E2E tests

| Test | Requirement |
|------|-------------|
| `e2e_otel_genai_export` | `otelcol` + `debugexporter` — parse exported spans |
| `e2e_langfuse_payload` | Mock HTTP server — assert `langfuse.observation.input` |
| Existing mock provider tests | Must still pass |

### Battle testing checklist (manual)

- [ ] Export to Langfuse Cloud — generation shows input/output
- [ ] Export to LangSmith — run appears as LLM type with tokens
- [ ] Jaeger — span name searchable, events in logs tab
- [ ] Toggle `EDGEQUAKE_OTEL_SEMCONV_OPT_IN` — verify dual vs latest
- [ ] Load test 100 RPS — no >5% latency overhead from observability

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Spec changes before stable | High | Medium | SemConvEmitter trait; watch GenAI repo |
| Langfuse fixes event parsing | Medium | Low | Langfuse profile becomes redundant — keep as compat |
| Trace size explosion (MaxCompatibility) | Medium | High | Default `Latest`; warn in docs |
| Dynamic span name unsupported | Low | Medium | EC-G09 fallback |
| tracing-opentelemetry drops array attrs | Medium | Medium | Test export; stringify finish_reasons if needed |
| Performance regression | Low | Medium | Benchmark before/after Phase 1 |

---

## 10. File Change Map

| File | Action | Phase |
|------|--------|-------|
| `src/observability/mod.rs` | **CREATE** | 0 |
| `src/observability/semconv/attributes.rs` | **CREATE** — move constants | 0 |
| `src/observability/semconv/provider_names.rs` | **CREATE** | 0 |
| `src/observability/semconv/version.rs` | **CREATE** | 0 |
| `src/observability/span/builder.rs` | **CREATE** | 1 |
| `src/observability/span/operation.rs` | **CREATE** | 0 |
| `src/observability/span/backend_profile.rs` | **CREATE** | 1 |
| `src/observability/span/stream_guard.rs` | **CREATE** | 2 |
| `src/observability/events/messages.rs` | **CREATE** | 1 |
| `src/observability/events/inference_details.rs` | **CREATE** | 1 |
| `src/observability/error_mapping.rs` | **CREATE** | 1 |
| `src/observability/metrics/recorder.rs` | **CREATE** | 4 |
| `src/providers/tracing.rs` | **REFACTOR** — thin decorator | 1 |
| `src/providers/genai_events.rs` | **DEPRECATE** — re-export | 1 |
| `src/lib.rs` | Export `observability` | 0 |
| `docs/observability.md` | **UPDATE** | 1 |
| `docs/migration-guide.md` | Semconv migration section | 1 |
| `CHANGELOG.md` | v0.11.0 entry | 3 |
| `examples/advanced/otel_langfuse.rs` | **CREATE** | 3 |
| `examples/advanced/otel_langsmith.rs` | **CREATE** | 3 |
| `tests/genai_semconv_conformance.rs` | **CREATE** | 1 |
| `tests/e2e_otel_genai_export.rs` | **CREATE** | 3 |
| `specs/004-optel-genai/fixtures/*.json` | **CREATE** — official schemas | 1 |

---

## Timeline Estimate

| Phase | Duration | Release |
|-------|----------|---------|
| Phase 0 | 3 days | — |
| Phase 1 | 7 days | v0.11.0-alpha |
| Phase 2 | 4 days | v0.11.0-beta |
| Phase 3 | 5 days | v0.11.0 |
| Phase 4 | 6 days | v0.12.0 |

**Total P0 (Phases 0–3)**: ~19 engineering days

---

## Configuration Reference (post-implementation)

| Env var | Default | Description |
|---------|---------|-------------|
| `EDGEQUAKE_OTEL_SEMCONV_OPT_IN` | unset | Set to `gen_ai_latest_experimental` for latest semconv |
| `EDGEQUAKE_OTEL_BACKEND_PROFILE` | `latest` | `legacy` \| `latest` \| `langfuse` \| `langsmith` \| `max_compat` |
| `EDGECODE_CAPTURE_CONTENT` | unset | Capture messages in events (+ profile shims) |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | unset | Alias honored when set |
| `EDGEQUAKE_OTEL_ENABLED` | `true` when otel feature | Master switch for TracingProvider auto-wrap |
