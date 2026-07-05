# First Principles & SDK Design — Application Attribution

> **Principle**: Separate *what the caller means* (application identity) from *how each provider encodes it* (HTTP header, body field, SDK metadata).

---

## 1. First Principles Analysis

### 1.1 What is being propagated?

Three distinct concepts are often conflated:

| Concept | Example | Purpose | Typical carrier |
|---------|---------|---------|-----------------|
| **Request correlation** | UUID per HTTP call | Debug/support, timeout recovery | `X-Client-Request-Id`, `x-request-id`, `traceparent` |
| **Application attribution** | `billing-app=crm-backend` | Provider dashboards, internal cost allocation | Provider-specific (see matrix doc) |
| **Distributed trace context** | W3C trace | End-to-end latency, span parenting | `traceparent`, `tracestate` |
| **Business context (OTEL Baggage)** | `tenant_id=acme` | Enrich spans/logs in *your* backends | `baggage` header → span attributes |

**Application ID propagation** belongs to **application attribution**. It may reuse the same string as request correlation in small apps, but the SDK must not assume they are identical.

### 1.2 Why a header at the edgequake-llm boundary?

Callers (HTTP gateways, Python services, Rust agents) need a **stable, provider-agnostic** contract:

```http
X-EdgeQuake-App-Id: edgequake-workspace-api
```

edgequake-llm resolves this to upstream canonical forms. Callers never import provider header names.

### 1.3 Safety constraints (non-negotiable)

1. **Never override** auth or protocol headers (`Authorization`, `x-api-key`, `Content-Type`, `Host`, provider version headers).
2. **Validate** header names (RFC 7230 token) and values (printable ASCII; provider-specific max lengths).
3. **Reject or truncate** values that would cause upstream 400s (OpenAI `X-Client-Request-Id` ≤ 512 chars).
4. **No secrets in app_id** — app IDs are identifiers, not API keys.
5. **Explicit failure modes** — if a provider cannot propagate, return `AttributionNotSupported` or attach a structured warning; do not silently drop (fixing current factory behavior).

---

## 2. Core Types (Rust)

### 2.1 `ApplicationContext`

Single value object at the public API boundary (`src/application_context.rs` — new module):

```rust
/// Caller-supplied identity for attribution and correlation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ApplicationContext {
    /// Stable application identifier (slug). Required for attribution.
    /// Example: "edgequake-workspace", "acme-crm-v2"
    pub app_id: Option<String>,

    /// Human-readable name for provider dashboards.
    pub app_name: Option<String>,

    /// Public URL of the application (used by OpenRouter, optional elsewhere).
    pub app_url: Option<String>,

    /// Multi-tenant partition; also promoted to OTEL baggage when configured.
    pub tenant_id: Option<String>,

    /// Per-request correlation ID (distinct from trace ID).
    pub request_id: Option<String>,

    /// Optional end-user ID for providers that support safety/billing user binding.
    /// Maps to OpenAI `user` field, Azure `end_user_id` — never auto-populated.
    pub end_user_id: Option<String>,

    /// Additional passthrough headers (traceparent, custom vendor headers).
    /// Keys validated; reserved keys filtered centrally.
    pub extra_headers: HashMap<String, String>,
}
```

Builder methods: `ApplicationContext::builder().app_id("…").build()` with validation.

### 2.2 `AttributionPolicy`

Controls behavior when propagation is partial:

```rust
pub enum AttributionPolicy {
    /// Best-effort: propagate what the provider supports; log warnings.
    BestEffort,
    /// Strict: fail provider creation if app_id cannot reach upstream.
    RequireAppId,
    /// Disable all attribution injection (testing).
    Disabled,
}
```

### 2.3 `AttributionResolver` trait (SOLID — Interface Segregation)

```rust
pub trait AttributionResolver: Send + Sync {
    /// Map ApplicationContext → outbound HTTP headers + optional body fields.
    fn resolve(&self, ctx: &ApplicationContext) -> ResolvedAttribution;
}

pub struct ResolvedAttribution {
    pub headers: HeaderMap,
    pub body_fields: HashMap<String, serde_json::Value>, // e.g. Azure application_name
    pub warnings: Vec<AttributionWarning>,
}
```

One implementation per provider family (`OpenAiAttribution`, `BedrockAttribution`, …), registered in `ProviderCatalog` metadata — **DRY**: shared logic for OpenAI-compatible family.

### 2.4 Provider integration pattern

**Option A (recommended): `AttributionLayer` wrapper**

```rust
pub struct AttributionProvider<P: LLMProvider> {
    inner: P,
    resolver: Arc<dyn AttributionResolver>,
    context: ApplicationContext,
}
```

Implements `LLMProvider` by delegating and injecting headers at the HTTP layer. Matches existing `TracingProvider`, `CachedProvider`, `RateLimitedProvider` decorator pattern in `src/providers/tracing.rs`.

**Option B: Central HTTP helper** — `HttpAttributionExt` used inside each provider's request builder. Higher touch, duplicates less at runtime.

**Decision**: Option A for new code path; migrate existing `with_extra_headers()` to use shared `filter_and_merge_headers()` utility (DRY).

---

## 3. Factory & Configuration API

### 3.1 New factory methods

```rust
impl ProviderFactory {
    pub fn create_with_context(
        provider: ProviderType,
        model: Option<&str>,
        ctx: ApplicationContext,
    ) -> Result<Arc<dyn LLMProvider>>;

    pub fn create_llm_provider_with_context(
        provider_name: &str,
        model: &str,
        ctx: ApplicationContext,
    ) -> Result<Arc<dyn LLMProvider>>;
}
```

Deprecate (not remove) `create_llm_provider_with_headers()` — implement as:

```rust
ctx.extra_headers.extend(headers);
Self::create_llm_provider_with_context(name, model, ctx)
```

### 3.2 Environment variables (12-factor)

| Variable | Maps to |
|----------|---------|
| `EDGEQUAKE_APP_ID` | `ApplicationContext.app_id` |
| `EDGEQUAKE_APP_NAME` | `app_name` |
| `EDGEQUAKE_APP_URL` | `app_url` |
| `EDGEQUAKE_TENANT_ID` | `tenant_id` |

Per-request values from HTTP headers override env defaults.

### 3.3 TOML config extension

```toml
[defaults.application]
app_id = "my-service"
app_name = "My Service"
app_url = "https://app.example.com"

[[providers]]
name = "openai"
# Provider-level override
application_app_id = "openai-specific-app"
```

Stored in `DefaultsConfig` / `ProviderConfig` — loaded once, merged with per-call context (call wins).

---

## 4. Python SDK (`edgequake-litellm`)

### 4.1 Function signature extension

```python
response = completion(
    model="openai/gpt-5.4-mini",
    messages=[...],
    application_id="my-backend",           # NEW
    application_name="My Backend",       # NEW optional
    application_url="https://...",       # NEW optional
    tenant_id="tenant-42",               # NEW optional
    extra_headers={"traceparent": "..."},  # NEW optional passthrough
)
```

Rust bridge calls `ProviderFactory::create_with_context()`.

### 4.2 `LiteLLMEdgeConfig` extension

```python
@dataclass
class LiteLLMEdgeConfig:
    ...
    application: ApplicationContext | None = None
```

---

## 5. Ingress HTTP Contract (forward-compatible)

For gateways sitting in front of edgequake-llm:

```http
POST /v1/chat/completions
X-EdgeQuake-App-Id: workspace-api
X-EdgeQuake-App-Name: EdgeQuake Workspace
X-EdgeQuake-App-Url: https://app.edgequake.io
X-EdgeQuake-Tenant-Id: org_abc123
X-EdgeQuake-Request-Id: 7c9e6679-7425-40de-944b-e07fc1f90ae7
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
Content-Type: application/json
```

Validation rules:

| Field | Max length | Charset | Required |
|-------|------------|---------|----------|
| `app_id` | 128 | `[a-zA-Z0-9._-]` | Yes (for attribution) |
| `app_name` | 256 | printable ASCII | No |
| `app_url` | 2048 | valid URL | No (required for OpenRouter rankings) |
| `tenant_id` | 128 | `[a-zA-Z0-9._-]` | No |
| `request_id` | 512 | ASCII (OpenAI limit) | No |

Reject `app_id` values matching `(?i)(Bearer|sk-|api[_-]?key)` patterns.

---

## 6. Shared Utilities (DRY)

New module `src/http/attribution.rs`:

| Function | Responsibility |
|----------|----------------|
| `filter_reserved_headers()` | Single reserved list + provider extensions |
| `validate_header_name()` | RFC 7230 token check |
| `sanitize_app_id()` | Slug normalization |
| `merge_attribution(base, resolved)` | Caller extra_headers win over defaults; never win over auth |
| `attribution_resolver_for(provider_type)` | Factory lookup |

Replace duplicated `RESERVED` arrays in:

- `src/providers/anthropic.rs` (~666)
- `src/providers/gemini.rs` (~1074)
- `src/providers/openai_compatible.rs` (~808)
- `src/providers/mistral.rs` (~781)
- `src/providers/nvidia.rs` (~1143)

---

## 7. SOLID Mapping

| Principle | Application |
|-----------|-------------|
| **Single Responsibility** | `ApplicationContext` = data; `AttributionResolver` = mapping; providers = HTTP |
| **Open/Closed** | New provider = new resolver impl; no trait changes |
| **Liskov Substitution** | `AttributionProvider<P>` delegates all `LLMProvider` methods |
| **Interface Segregation** | Separate `AttributionResolver` from `LLMProvider` |
| **Dependency Inversion** | Factory depends on `AttributionResolver` registry, not header strings |

---

## 8. Migration Path

| Before | After |
|--------|-------|
| `.with_extra_headers([("X-Client-Request-Id", id)])` | `.with_context(ApplicationContext::builder().request_id(id).build())` |
| `create_llm_provider_with_headers("openai", …)` | `create_llm_provider_with_context("openai", …, ctx)` — **no silent drop** |
| `OPENROUTER_SITE_URL` env | `EDGEQUAKE_APP_URL` + OpenRouter resolver |
| Manual `traceparent` in extra_headers | Same, or auto from OTEL context (see OTEL doc) |

Ascending compatibility: existing `with_extra_headers()` remains; internally calls shared merge utility.
