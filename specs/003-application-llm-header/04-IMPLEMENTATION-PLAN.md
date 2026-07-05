# Implementation Plan — Application ID Propagation

> **Version**: 1.0 — 2026-07-05  
> **Target**: edgequake-llm v0.10.0  
> **Principles**: First Principles · DRY · SOLID · Ascending Compatibility · Battle-Tested  
> **Status**: 📋 PLANNED

---

## Table of Contents

1. [Architecture Decisions (ADRs)](#1-architecture-decisions-adrs)
2. [Phase 0 — Foundation](#2-phase-0--foundation)
3. [Phase 1 — Core Providers (P0)](#3-phase-1--core-providers-p0)
4. [Phase 2 — Extended Providers (P1)](#4-phase-2--extended-providers-p1)
5. [Phase 3 — OTEL Integration (P1)](#5-phase-3--otel-integration-p1)
6. [Phase 4 — Python SDK & Docs (P2)](#6-phase-4--python-sdk--docs-p2)
7. [Edge Case Register](#7-edge-case-register)
8. [Testing Strategy](#8-testing-strategy)
9. [Risk Register](#9-risk-register)
10. [File Change Map](#10-file-change-map)

---

## 1. Architecture Decisions (ADRs)

### ADR-001: `ApplicationContext` as single semantic type

**Context**: Headers today are `Vec<(String,String)>` with provider-specific meanings.

**Decision**: Introduce `ApplicationContext` value object; map to provider via `AttributionResolver`.

**Consequences**: One breaking-soft change: `create_llm_provider_with_headers` deprecated but functional.

---

### ADR-002: Decorator over trait modification

**Context**: Could add `fn set_context(&self, ctx)` to `LLMProvider`.

**Decision**: Use `AttributionProvider<P>` wrapper (same pattern as `TracingProvider`).

**Consequences**: No trait breaking change; providers opt-in HTTP injection internally OR via wrapper.

**Rejected**: Extending `LLMProvider` — violates ISP; not all callers need attribution.

---

### ADR-003: Fail loud, not silent

**Context**: `create_llm_provider_with_headers` silently drops headers for unsupported providers (`factory.rs:1792–1801`).

**Decision**:
- `AttributionPolicy::BestEffort` — warn + span event (default)
- `AttributionPolicy::RequireAppId` — return `LlmError::AttributionError`
- Replace `tracing::debug!` with `tracing::warn!` + structured field `attribution.dropped=true`

---

### ADR-004: Body-field attribution for Azure & Bedrock

**Context**: Azure deprecating custom header passthrough; Bedrock metadata is body/header JSON.

**Decision**: `ResolvedAttribution.body_fields` merged at request serialization time in provider, not only HTTP headers.

**Consequences**: Providers need hook `fn apply_attribution(&mut self, resolved: &ResolvedAttribution)` or inline in request builders.

---

### ADR-005: Central reserved-header filter (DRY)

**Context**: Five copies of `RESERVED` header lists across providers.

**Decision**: Single `src/http/attribution.rs::filter_reserved_headers(name, provider_kind)`.

---

## 2. Phase 0 — Foundation

**Goal**: Shared types and utilities without changing provider behavior.

| Task | File(s) | Est. |
|------|---------|------|
| FEAT-070: `ApplicationContext` + builder + validation | `src/application_context.rs`, `src/lib.rs` | 1d |
| FEAT-071: `AttributionResolver` trait + `ResolvedAttribution` | `src/http/attribution.rs` | 0.5d |
| FEAT-072: `filter_reserved_headers`, sanitizers | `src/http/attribution.rs`, `src/http/mod.rs` | 0.5d |
| FEAT-073: `AttributionPolicy`, `AttributionWarning` enum | `src/error.rs` or `src/application_context.rs` | 0.5d |
| FEAT-074: Unit tests for validation edge cases | `src/application_context.rs` tests | 0.5d |

### Validation rules (implement exactly)

```rust
// app_id: ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$
// Reject: empty, whitespace, sk-*, Bearer*, control chars
// request_id: ASCII only, max 512 (OpenAI constraint)
// app_url: url crate parse, https preferred, warn on http
```

### Deliverable gate

- [ ] 40+ unit tests on sanitization
- [ ] No provider behavior change yet
- [ ] `cargo test` green

---

## 3. Phase 1 — Core Providers (P0)

**Goal**: Attribution for providers covering >80% production traffic.

| ID | Provider | Resolver output | Implementation site |
|----|----------|-----------------|---------------------|
| FEAT-080 | OpenAI | `X-Client-Request-Id`, body `user` | `src/providers/openai.rs` — `OpenAIConfig.default_headers` |
| FEAT-081 | Azure OpenAI | body `application_name`, header `x-ms-client-request-id` | `src/providers/azure_openai.rs` |
| FEAT-082 | Anthropic | passthrough + merge in `headers()` | Refactor `anthropic.rs` to use shared filter |
| FEAT-083 | Gemini/Vertex | append `edgequake-app/{id}` to `x-goog-api-client` | `gemini.rs` rebuild client |
| FEAT-084 | OpenRouter | `HTTP-Referer`, `X-OpenRouter-Title` (+ `X-Title` fallback) | `openrouter.rs` |
| FEAT-085 | OpenAI-compatible | `X-Client-Request-Id`, passthrough | Refactor `openai_compatible.rs` |
| FEAT-086 | Factory | `create_with_context`, fix silent drop | `factory.rs` |

### OpenAI implementation notes

```rust
// openai.rs — inject at client build time
let mut default_headers = HeaderMap::new();
if let Some(ref rid) = ctx.request_id {
    default_headers.insert("X-Client-Request-Id", HeaderValue::from_str(rid)?);
}
let config = OpenAIConfig::new()
    .with_api_key(&api_key)
    .with_default_headers(default_headers);
```

For `end_user_id`: thread through `CompletionOptions` or separate per-call context — **ADR-006 pending** (see edge cases).

### OpenRouter fix (breaking-soft)

```rust
// Prefer new header name; keep legacy
headers.insert("X-OpenRouter-Title", ...);
headers.insert("X-Title", ...); // backwards compat
```

### Factory change

Replace match arm `_ => { debug!(); fallback }` with:

```rust
_ => {
    if policy == RequireAppId {
        return Err(LlmError::AttributionError(...));
    }
    tracing::warn!(...);
    if !headers_vec.is_empty() {
        // wrap in AttributionProvider with PassthroughResolver
    }
    Self::create_llm_provider(provider_name, model)
}
```

### Deliverable gate

- [ ] E2E test file `tests/e2e_application_attribution.rs` with wiremock/mockito
- [ ] Each P0 provider: assert outbound header in mock server
- [ ] OpenRouter: test Referer + Title headers

---

## 4. Phase 2 — Extended Providers (P1)

| ID | Provider | Mechanism | File |
|----|----------|-----------|------|
| FEAT-090 | Bedrock | `request_metadata` on Converse | `bedrock.rs` |
| FEAT-091 | Cohere | `X-Client-Name` | `cohere.rs` |
| FEAT-092 | Mistral | shared resolver + dual client | `mistral.rs` |
| FEAT-093 | Nvidia | `X-Request-Id` | `nvidia.rs` |
| FEAT-094 | xAI / HF / LM Studio | OpenAI-compatible resolver via wrapper | wrapper types |
| FEAT-095 | Ollama | passthrough only | `ollama.rs` |
| FEAT-096 | Imagegen | same resolver registry | `imagegen/factory.rs` |
| FEAT-097 | Embeddings | parity with chat for OpenAI, Azure, Cohere, Mistral | various |

### Bedrock specifics

```rust
// bedrock Converse call
let metadata = HashMap::from([
    ("app".into(), ctx.app_id.clone()?),
    ("tenant_id".into(), ctx.tenant_id.clone().unwrap_or_default()),
]);
request = request.request_metadata(aws_sdk_bedrockruntime::types::RequestMetadata::builder()
    .set_custom(metadata).build()?);
```

Verify against [AWS docs](https://docs.aws.amazon.com/bedrock/latest/userguide/cost-mgmt-request-metadata.html).

### Wrapper delegation (DRY)

```rust
// xai.rs
impl XAIProvider {
    pub fn with_context(mut self, ctx: ApplicationContext) -> Self {
        self.inner = self.inner.with_application_context(ctx); // on OpenAICompatibleProvider
        self
    }
}
```

Add `application_context: Option<ApplicationContext>` field to `OpenAICompatibleProvider` — set once at build, applied in `build_client()`.

---

## 5. Phase 3 — OTEL Integration (P1)

| ID | Task | File |
|----|------|------|
| FEAT-100 | `gen_ai.application.*` span attributes | `providers/tracing.rs` |
| FEAT-101 | `TracingProvider::with_application_context()` | `providers/tracing.rs` |
| FEAT-102 | Optional traceparent injection (`EDGEQUAKE_OTEL_INJECT_TRACE_CONTEXT`) | `http/attribution.rs` |
| FEAT-103 | Span events for attribution warnings | `providers/tracing.rs` |
| FEAT-104 | Optional baggage promotion (off by default) | `providers/tracing.rs` |

### Feature gating

```toml
# Cargo.toml — no new deps
[features]
otel = ["opentelemetry", "tracing-opentelemetry"]  # existing
```

Trace injection uses existing `tracing-opentelemetry` when feature enabled; no-op otherwise.

---

## 6. Phase 4 — Python SDK & Docs (P2)

| ID | Task | File |
|----|------|------|
| FEAT-110 | `application_id` param on completion/acompletion/stream | `edgequake-litellm/src/completion.rs` |
| FEAT-111 | PyO3 `ApplicationContext` type | `edgequake-litellm/src/types.rs` |
| FEAT-112 | Migration guide section | `docs/migration-guide.md` |
| FEAT-113 | Observability doc update | `docs/observability.md` |
| FEAT-114 | Provider catalog metadata | `provider_catalog.rs` — `attribution_support: AttributionSupport` |

---

## 7. Edge Case Register

### 7.1 Header & protocol conflicts

| # | Case | Expected behavior | Test |
|---|------|-------------------|------|
| E-001 | Caller sets `Authorization` in extra_headers | Silently dropped | unit |
| E-002 | Caller sets `anthropic-version` override | Dropped for Anthropic | unit |
| E-003 | POE provider + custom User-Agent | Dropped (existing POE quirk) | unit |
| E-004 | Azure >10 custom headers | Prefer body fields; count headers; warn at 8+ | unit |
| E-005 | Invalid header name `X App` | Skip + warning | unit |
| E-006 | Non-ASCII in `X-Client-Request-Id` | Reject at validation (OpenAI 400) | unit |

### 7.2 Semantic conflicts

| # | Case | Expected behavior | Test |
|---|------|-------------------|------|
| E-010 | `app_id` looks like API key `sk-abc...` | Validation error | unit |
| E-011 | OpenRouter without `app_url` | Warn; API call succeeds without ranking | unit |
| E-012 | OpenRouter localhost URL without title | Warn per OpenRouter docs | unit |
| E-013 | Map `app_id` to OpenAI `user` field | **Do not** by default; separate fields | doc + unit |
| E-014 | xAI: app_id → `x-grok-conv-id` | **Forbidden** — changes cache routing | unit |
| E-015 | Gemini: replace vs append `x-goog-api-client` | Must append | integration |

### 7.3 Concurrency & lifecycle

| # | Case | Expected behavior | Test |
|---|------|-------------------|------|
| E-020 | `ApplicationContext` set on shared `Arc<dyn LLMProvider>` | Context is per-provider instance, not global | unit |
| E-021 | Changing context mid-flight | Immutable after provider build; use per-call override API (Phase 2+) | doc |
| E-022 | `with_extra_headers` after `with_context` | Merge with precedence: explicit extra > context | unit |
| E-023 | CachedProvider cache key ignores app_id | Document: cache is content-based; same prompt different apps hits same cache | doc |

### 7.4 Provider-specific

| # | Case | Expected behavior | Test |
|---|------|-------------------|------|
| E-030 | Bedrock InvokeModel vs Converse | Resolver picks correct field per API used | unit |
| E-031 | Bedrock metadata JSON invalid | Validation before send | unit |
| E-032 | VSCode Copilot provider | AttributionPolicy returns Unsupported; OTEL only | unit |
| E-033 | Nvidia HTTP 202 async | Attribution headers on initial POST and poll GET | e2e mock |
| E-034 | Mock provider | Accept context for testing resolver | unit |

### 7.5 OTEL

| # | Case | Expected behavior | Test |
|---|------|-------------------|------|
| E-040 | otel feature disabled | No traceparent injection; span attrs still set locally | unit |
| E-041 | baggage with PII | Not promoted unless allowlist extended | unit |
| E-042 | Provider rejects unknown headers | BestEffort: log upstream 400 with attribution context | e2e |

---

## 8. Testing Strategy

### 8.1 Unit tests (target: 80+ new)

| Module | Focus |
|--------|-------|
| `application_context.rs` | Validation, builder, merge precedence |
| `http/attribution.rs` | Reserved filter, sanitizers, resolver registry |
| Per-resolver | Expected headers/body for fixture context |
| `factory.rs` | No silent drop in RequireAppId mode |

### 8.2 HTTP mock tests (`wiremock` or existing test patterns)

```rust
#[tokio::test]
async fn openai_forwards_client_request_id() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(header("X-Client-Request-Id", "req-123"))
        .respond_with(ResponseTemplate::new(200).set_body_json(...))
        .mount(&mock)
        .await;
    // ...
}
```

### 8.3 E2E file: `tests/e2e_application_attribution.rs`

Scenarios:
1. OpenAI-compatible mock — headers present
2. OpenRouter — Referer + Title
3. Factory RequireAppId — error on unsupported provider
4. Gemini — append client token without clobbering

### 8.4 Regression

- All existing `with_extra_headers` tests in mistral/nvidia must pass unchanged
- Run full suite: `cargo test` (target: 1,400+ existing + 80 new)

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Azure stops header passthrough | Medium | Low if body fields used | ADR-004; monitor FAQ |
| OpenAI rejects custom headers | Low | Medium | Stick to documented headers only |
| Header injection breaks Copilot | Medium | High | Explicit block list for vscode-copilot |
| Cache key collision after attribution | Low | Medium | Document CachedProvider behavior |
| Python API churn | Medium | Low | Optional params with defaults |
| Gemini telemetry corruption | Medium | Medium | Append-only implementation + test |

---

## 10. File Change Map

### New files

| Path | Purpose |
|------|---------|
| `src/application_context.rs` | Core types |
| `src/http/mod.rs` | HTTP utilities module |
| `src/http/attribution.rs` | Resolvers, filters, merge |
| `src/http/resolvers/openai.rs` | OpenAI family |
| `src/http/resolvers/bedrock.rs` | Bedrock metadata |
| `src/http/resolvers/openrouter.rs` | OpenRouter |
| `src/http/resolvers/gemini.rs` | Google client header |
| `src/providers/attribution.rs` | `AttributionProvider` wrapper |
| `tests/e2e_application_attribution.rs` | Integration tests |

### Modified files (priority order)

| Path | Change |
|------|--------|
| `src/factory.rs` | `create_with_context`, fix silent drop |
| `src/providers/openai_compatible.rs` | DRY filter; optional context field |
| `src/providers/openai.rs` | default_headers |
| `src/providers/azure_openai.rs` | body fields |
| `src/providers/openrouter.rs` | X-OpenRouter-Title |
| `src/providers/anthropic.rs` | shared filter |
| `src/providers/gemini.rs` | append x-goog-api-client |
| `src/providers/bedrock.rs` | request_metadata |
| `src/providers/cohere.rs` | X-Client-Name |
| `src/providers/tracing.rs` | gen_ai.application.* |
| `src/provider_catalog.rs` | attribution metadata |
| `src/lib.rs` | re-exports |
| `edgequake-litellm/src/completion.rs` | Python params |
| `docs/migration-guide.md` | usage examples |

---

## Implementation Order (suggested sprint plan)

```text
Week 1: Phase 0 + FEAT-086 factory + FEAT-085 openai-compatible
Week 2: FEAT-080 OpenAI + FEAT-081 Azure + FEAT-084 OpenRouter
Week 3: FEAT-082 Anthropic + FEAT-083 Gemini + FEAT-090 Bedrock
Week 4: Phase 3 OTEL + Phase 4 Python + docs + e2e hardening
```

---

## Acceptance Criteria (v0.10.0)

- [ ] `ApplicationContext` exported from crate root
- [ ] All P0/P1 providers have registered resolver
- [ ] Zero silent header drops when `app_id` set and policy `RequireAppId`
- [ ] `TracingProvider` records `gen_ai.application.id`
- [ ] Python `application_id=` parameter functional
- [ ] Spec matrix doc links verified (manual quarterly check)
- [ ] Migration guide published
