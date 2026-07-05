# 04 — Provider Catalog & Capability Search API

**Status:** Implemented  
**Date:** 2026-07-05

## First-Principles Audit (Code Is Law)

### What exists today

| Concern | Location | Gap |
|---------|----------|-----|
| List chat providers | `ProviderRegistry::list_llm()` | Instance-scoped; not crate-level |
| List discovery providers | `ModelDiscoveryService::provider_ids()` | Async service required; subset of LLM providers |
| Parse provider name | `ProviderType::from_str()` | No reverse mapping or enumeration |
| Python provider list | `edgequake-litellm::list_providers()` | Hardcoded; drifted (missing `cohere`, `nvidia`) |
| Search models by capability | `CapabilityFilter` + `find_models().await` | Requires async + network; no offline static search |
| Static model metadata | `all_static_models()` + `static_lookup()` | No filter helper |

### SOLID / DRY violations

1. **DRY** — Provider names duplicated in `ProviderType::from_str`, Python `list_providers()`, docs, and discovery defaults.
2. **SRP** — `ProviderFactory` creates instances; it should not own catalog metadata (new `ProviderCatalog`).
3. **OCP** — Adding a provider requires edits in 3+ places today.
4. **ISP** — Consumers that only need IDs should not pull in factory/env logic.

## Design

### 1. `ProviderCatalog` (single source of truth)

```rust
pub struct ProviderFeatures {
    pub chat: bool,
    pub embedding: bool,
    pub model_discovery: bool,
    pub image_generation: bool,
}

pub struct ProviderDescriptor {
    pub id: &'static str,
    pub aliases: &'static [&'static str],
    pub features: ProviderFeatures,
}

impl ProviderCatalog {
    pub fn all() -> &'static [ProviderDescriptor];
    pub fn list_llm_providers() -> Vec<&'static str>;
    pub fn list_discovery_providers() -> Vec<&'static str>;
    pub fn list_imagegen_providers() -> Vec<&'static str>;
    pub fn resolve_id(input: &str) -> Option<&'static str>;
}
```

`ProviderFactory::list_providers()` and Python `list_providers()` delegate here.

### 2. `ModelCapability` + ergonomic filters

```rust
pub enum ModelCapability {
    Vision, Tools, Thinking, Streaming, JsonMode,
}

impl CapabilityFilter {
    pub fn requiring(self, cap: ModelCapability) -> Self;
    pub fn with_provider(self, id: &str) -> Self;
    // ... builder helpers
}

pub fn find_static_models(filter: &CapabilityFilter) -> Vec<DiscoveredModel>;
```

Offline agents use `find_static_models` without tokio or API keys.

### 3. Service extensions

```rust
impl ModelDiscoveryService {
    pub async fn find_models_for_provider(&self, id: &str, filter: &CapabilityFilter) -> Result<Vec<DiscoveredModel>>;
}
```

## Implementation checklist

- [x] `src/provider_catalog.rs`
- [x] `ProviderType::all()`, `canonical_id()`, `ProviderFactory::list_providers()`
- [x] `ModelCapability`, `CapabilityFilter` builders, `find_static_models()`
- [x] `ModelDiscoveryService::find_models_for_provider()`
- [x] Crate re-exports in `lib.rs`
- [x] Python bindings use catalog
- [x] Tests in `tests/e2e_discovery.rs`

## Model name search (added 2026-07-05)

- [x] `ModelSearchQuery` — exact, substring, `provider/model`, and fuzzy modes
- [x] Input/output length bounds: `min_context_length`, `max_context_length`, `min_output_tokens`, `max_output_tokens`
- [x] `search_models()` / `search_static_models()` — ranked results with score + match kind
- [x] `static_lookup_by_name()` — exact ID then display name
- [x] Python: `search_models()`, `search_static_models_by_name()`, `lookup_model_by_name()`

## Non-goals

- Changing existing async discovery behavior
- Removing `ProviderRegistry` (still valid for runtime registration)
- Auto-detecting credentials (separate concern)
