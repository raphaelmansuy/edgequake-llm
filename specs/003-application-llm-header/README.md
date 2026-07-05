# 003 — Application ID Propagation & LLM Provider Attribution

> **Date**: 2026-07-05  
> **Status**: ✅ IMPLEMENTED (v0.10.0)  
> **Target Version**: edgequake-llm v0.10.0  
> **Scope**: Safe propagation of an application identifier from caller → edgequake-llm → upstream LLM provider, with OTEL integration guidance.

---

## Problem Statement

Multi-app platforms (SaaS, internal gateways, agent frameworks) need to attribute LLM usage to the **calling application**, not just the API key owner. Today edgequake-llm has **partial, ad-hoc header support** (`with_extra_headers()` on 6 providers; OpenRouter has separate site URL/name fields) but no canonical `application_id` contract, no ingress header convention, and no OTEL bridge.

Goals:

1. **Safe propagation** — validate, sanitize, and never override auth/protocol headers.
2. **Provider-canonical mapping** — each provider gets the header/field its official API documents for attribution.
3. **First-principles SDK** — one semantic type at the trait boundary; provider specifics hidden behind a resolver.
4. **OTEL clarity** — distinguish trace correlation (`traceparent`) from business attribution (`app_id`) from provider billing tags.

Non-goals (v1):

- HTTP proxy server for edgequake-llm (library-only; ingress header spec is forward-compatible).
- Automatic PII scrubbing beyond documented field constraints.
- Provider-side enforcement of app IDs (most providers treat attribution as opt-in telemetry).

---

## Documents

| # | Document | Purpose |
|---|----------|---------|
| 01 | [First Principles & SDK Design](./01-FIRST-PRINCIPLES-SDK-DESIGN.md) | Semantic model, public API, ingress header contract |
| 02 | [Provider Attribution Matrix](./02-PROVIDER-ATTRIBUTION-MATRIX.md) | Per-provider canonical headers/fields with official doc links |
| 03 | [OTEL & Observability Role](./03-OTEL-AND-OBSERVABILITY.md) | traceparent vs baggage vs app attribution; integration with `TracingProvider` |
| 04 | [Implementation Plan](./04-IMPLEMENTATION-PLAN.md) | Phased plan, ADRs, edge cases, DRY/SOLID, test strategy |

---

## Current State (v0.9.x baseline)

| Mechanism | Coverage | Risk |
|-----------|----------|------|
| `with_extra_headers()` | Anthropic, Gemini/Vertex, Mistral, Nvidia, OpenAI-compatible | Caller must know provider header names |
| `ProviderConfig.headers` | OpenAI-compatible only (TOML) | Not wired to most providers |
| `create_llm_provider_with_headers()` | Same 6 providers; **silent fallback** for others | **Data loss** — headers dropped with only `debug!` log |
| OpenRouter `HTTP-Referer` / `X-Title` | OpenRouter only | Separate API; `X-OpenRouter-Title` not yet used |
| Python `edgequake-litellm` | No header / app_id surface | Cannot propagate from Python callers |
| `TracingProvider` | GenAI spans only | Does not inject W3C headers into HTTP |

Key code anchors:

- Factory fallback: `src/factory.rs` — `create_llm_provider_with_headers()` (lines ~1698–1803)
- OpenAI-compatible headers: `src/providers/openai_compatible.rs` — `with_extra_headers()` (~804)
- OpenRouter attribution: `src/providers/openrouter.rs` — `headers()` (~552–574)
- OTEL docs: `docs/observability.md`

---

## Recommended Canonical Ingress Header

When edgequake-llm is invoked via HTTP (gateway, sidecar, or future proxy):

```http
X-EdgeQuake-App-Id: my-product-backend
```

Optional companions (all validated ASCII, length-bounded):

| Header | Purpose |
|--------|---------|
| `X-EdgeQuake-App-Name` | Human-readable label (maps to OpenRouter title, Cohere client name, etc.) |
| `X-EdgeQuake-App-Url` | Public app URL (maps to OpenRouter `HTTP-Referer`) |
| `X-EdgeQuake-Tenant-Id` | Multi-tenant partition (maps to Bedrock metadata, OTEL baggage) |
| `X-EdgeQuake-Request-Id` | Per-request correlation (maps to `X-Client-Request-Id`, `x-ms-client-request-id`) |

W3C headers (`traceparent`, `tracestate`, `baggage`) pass through **verbatim** when present; they are not reinterpreted as app IDs.

---

## Success Criteria

- [ ] Single Rust type `ApplicationContext` accepted by factory, wrappers, and Python bindings.
- [ ] Provider resolver maps `app_id` → canonical upstream header/field for **all** production providers.
- [ ] Silent header drop replaced with explicit `AttributionWarning` or `Result` when propagation impossible.
- [ ] Unit tests per provider for header name, value sanitization, and reserved-header filtering.
- [ ] Documentation in `docs/migration-guide.md` with before/after examples.
