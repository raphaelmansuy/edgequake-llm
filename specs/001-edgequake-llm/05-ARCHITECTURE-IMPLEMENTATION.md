# 05 — Architecture & Implementation Plan

> **Lens**: Full Stack Developer  
> **Date**: 2026-07-04  
> **Status**: DRAFT  
> **Principles**: DRY, SOLID, Ascending Compatibility

## SOLID Analysis

### S — Single Responsibility

| Component | Responsibility | Files |
|-----------|---------------|-------|
| `ModelDiscoveryProvider` trait | Define discovery contract | `discovery/traits.rs` |
| `DiscoveredModel` | Universal model descriptor | `discovery/types.rs` |
| `CapabilityFilter` | Query/filter interface | `discovery/types.rs` |
| `ModelDiscoveryService` | Orchestrate, cache, merge | `discovery/service.rs` |
| `DiscoveryCache` | TTL-based caching | `discovery/cache.rs` |
| `StaticModelRegistry` | Built-in model data | `discovery/static_registry.rs` |
| Per-provider impl | Provider-specific discovery | `discovery/providers/*.rs` |

### O — Open/Closed

- **Open for extension**: New providers implement `ModelDiscoveryProvider` trait
- **Closed for modification**: Existing `LLMProvider`, `EmbeddingProvider`, `ProviderFactory` unchanged
- **Open for data**: Static registries can be supplemented by TOML config

### L — Liskov Substitution

- Any `ModelDiscoveryProvider` impl returns `Vec<DiscoveredModel>`
- Dynamic, Static, and Hybrid providers are interchangeable through the trait

### I — Interface Segregation

- `ModelDiscoveryProvider` is separate from `LLMProvider`
- Discovery doesn't require chat capability
- Agent harnesses only depend on `discovery` module

### D — Dependency Inversion

- `ModelDiscoveryService` depends on `ModelDiscoveryProvider` trait (abstraction)
- Never depends on concrete provider types
- Provider implementations depend on types module only

---

## DRY Analysis

### Current Duplication to Eliminate

```
  DRY Violations Found:

  1. Model metadata duplicated across:
     ├── model_config.rs::builtin_defaults()  ← static ModelCard
     ├── providers/nvidia.rs  ← NvidiaModelInfo static catalog
     ├── providers/openrouter.rs  ← ModelInfo struct
     └── providers/mistral.rs  ← MistralModel struct
     ACTION: All feed into DiscoveredModel via normalization adapters

  2. Context length defaults duplicated:
     ├── Each provider has its own DEFAULT_CONTEXT_LENGTH const
     ├── model_config.rs has context_length per ModelCard
     └── LLMProvider::max_context_length() per instance
     ACTION: Single source in DiscoveredModel, providers query it

  3. Model listing implementations:
     ├── ollama.rs::list_models() → OllamaModelsResponse
     ├── openrouter.rs::list_models_cached() → Vec<ModelInfo>
     ├── nvidia.rs::list_models() → NvidiaModelsResponse
     ├── mistral.rs::list_models() → Vec<MistralModel>
     └── lmstudio.rs (metadata fetch)
     ACTION: Each wraps existing code, normalizes to DiscoveredModel
            NO duplication of HTTP logic — reuse existing implementations
```

---

## Architecture Diagram

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                  EdgeQuake LLM v0.7 Architecture                        │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                          │
  │  ┌─────────────────────────────────────────────────┐                    │
  │  │                  Public API Layer                 │                    │
  │  │                                                   │                    │
  │  │  LLMProvider     EmbeddingProvider                │                    │
  │  │  ProviderFactory ProviderRegistry                 │                    │
  │  │  ModelDiscoveryService  (NEW)                     │ ◄── Agent entry    │
  │  │  CapabilityFilter       (NEW)                     │     point          │
  │  └────────┬────────────────────┬─────────────────────┘                    │
  │           │                    │                                          │
  │  ┌────────▼────────┐  ┌───────▼──────────────────┐                      │
  │  │  Chat/Stream    │  │  Discovery Module (NEW)   │                      │
  │  │  Path           │  │                            │                      │
  │  │                 │  │  ┌──────────────────────┐  │                      │
  │  │ providers/      │  │  │ ModelDiscoveryProvider│  │ ◄── Trait           │
  │  │ ├─ openai.rs    │  │  │ trait                 │  │                      │
  │  │ ├─ anthropic.rs │  │  └──────────┬───────────┘  │                      │
  │  │ ├─ gemini.rs    │  │             │              │                      │
  │  │ ├─ ollama.rs    │  │  ┌──────────▼───────────┐  │                      │
  │  │ ├─ lmstudio.rs  │  │  │ discovery/providers/  │  │                      │
  │  │ ├─ openrouter.rs│  │  │ ├─ openai.rs (hybrid) │  │                      │
  │  │ ├─ mistral.rs   │  │  │ ├─ anthropic.rs       │  │                      │
  │  │ ├─ nvidia.rs    │  │  │ ├─ gemini.rs (dynamic)│  │                      │
  │  │ ├─ xai.rs       │  │  │ ├─ ollama.rs (dynamic)│  │                      │
  │  │ ├─ bedrock.rs   │  │  │ ├─ lmstudio.rs       │  │                      │
  │  │ └─ ...          │  │  │ ├─ openrouter.rs      │  │                      │
  │  │                 │  │  │ ├─ mistral.rs (hybrid)│  │                      │
  │  │                 │  │  │ ├─ nvidia.rs (hybrid) │  │                      │
  │  │                 │  │  │ ├─ xai.rs (static)    │  │                      │
  │  │                 │  │  │ └─ bedrock.rs (hybrid)│  │                      │
  │  │                 │  │  └──────────────────────┘  │                      │
  │  │                 │  │                            │                      │
  │  │                 │  │  ┌──────────────────────┐  │                      │
  │  │                 │  │  │ DiscoveryCache       │  │ ◄── Per-provider     │
  │  │                 │  │  │ (TTL-based)          │  │     cache            │
  │  │                 │  │  └──────────────────────┘  │                      │
  │  │                 │  │                            │                      │
  │  │                 │  │  ┌──────────────────────┐  │                      │
  │  │                 │  │  │ StaticModelRegistry  │  │ ◄── July 2026 data  │
  │  │                 │  │  │ (built-in fallback)  │  │                      │
  │  │                 │  │  └──────────────────────┘  │                      │
  │  │                 │  │                            │                      │
  │  └─────────────────┘  └────────────────────────────┘                      │
  │                                                                          │
  │  ┌─────────────────────────────────────────────────┐                    │
  │  │  Cross-Cutting Concerns (unchanged)              │                    │
  │  │  ├─ cache.rs        (response caching)           │                    │
  │  │  ├─ cost_tracker.rs (session costs)              │                    │
  │  │  ├─ rate_limiter.rs (API rate limits)            │                    │
  │  │  ├─ retry.rs        (retry with backoff)         │                    │
  │  │  ├─ middleware.rs   (logging, metrics)            │                    │
  │  │  └─ tokenizer.rs   (token counting)              │                    │
  │  └─────────────────────────────────────────────────┘                    │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)

**Goal**: Core types, trait, and static registries

```
  Tasks:
  ┌────────────────────────────────────────────────────────────┐
  │ 1.1  Create discovery/ module structure                     │
  │ 1.2  Define DiscoveredModel, CapabilityFilter types         │
  │ 1.3  Define ModelDiscoveryProvider trait                     │
  │ 1.4  Implement DiscoveryCache with TTL                      │
  │ 1.5  Build StaticModelRegistry with July 2026 data          │
  │      ├─ OpenAI: GPT-4.1, GPT-4o, o3, o4-mini              │
  │      ├─ Anthropic: Claude 4, 4.5                            │
  │      ├─ Gemini: 2.5, 3.x family                            │
  │      ├─ xAI: Grok-3, Grok-4                                │
  │      └─ HuggingFace: common models                         │
  │ 1.6  Extend ModelCapabilities (backward-compatible fields)  │
  │ 1.7  Unit tests for all types and filter logic              │
  └────────────────────────────────────────────────────────────┘
```

**Deliverables**: `discovery/mod.rs`, `discovery/traits.rs`, `discovery/types.rs`, `discovery/cache.rs`, `discovery/static_registry.rs`

### Phase 2: Dynamic Providers (Week 2)

**Goal**: Wire existing discovery implementations

```
  Tasks:
  ┌────────────────────────────────────────────────────────────┐
  │ 2.1  Implement OllamaDiscovery                              │
  │      └─ Wrap existing list_models() + /api/show             │
  │ 2.2  Implement OpenRouterDiscovery                          │
  │      └─ Wrap existing list_models_cached()                  │
  │ 2.3  Implement NvidiaDiscovery                              │
  │      └─ Wrap existing list_models() + static catalog        │
  │ 2.4  Implement MistralDiscovery                             │
  │      └─ Wrap existing list_models() + static                │
  │ 2.5  Implement LMStudioDiscovery                            │
  │      └─ Wrap existing metadata fetch                        │
  │ 2.6  Implement GeminiDiscovery                              │
  │      └─ NEW: call /v1beta/models                            │
  │ 2.7  Integration tests (mocked HTTP)                        │
  └────────────────────────────────────────────────────────────┘
```

**Key principle**: REUSE existing provider code. Discovery providers hold a reference to the existing provider instance and delegate the API calls.

### Phase 3: Hybrid & Static Providers (Week 3)

**Goal**: Complete all providers

```
  Tasks:
  ┌────────────────────────────────────────────────────────────┐
  │ 3.1  Implement OpenAIDiscovery (hybrid)                     │
  │      └─ /v1/models + static registry merge                 │
  │ 3.2  Implement AnthropicDiscovery (hybrid)                  │
  │      └─ /v1/models + static registry merge                 │
  │ 3.3  Implement XAIDiscovery (static)                        │
  │ 3.4  Implement BedrockDiscovery (hybrid, feature-gated)     │
  │ 3.5  Implement OpenAICompatibleDiscovery (try-dynamic)      │
  │ 3.6  Implement VsCodeCopilotDiscovery (dynamic)             │
  │ 3.7  Build ModelDiscoveryService facade                     │
  │ 3.8  Wire into lib.rs exports                               │
  │ 3.9  E2E tests with real providers (optional, env-gated)    │
  └────────────────────────────────────────────────────────────┘
```

### Phase 4: Polish & Integration (Week 4)

**Goal**: Integration with ProviderFactory, documentation, update model_config.rs

```
  Tasks:
  ┌────────────────────────────────────────────────────────────┐
  │ 4.1  Update model_config.rs builtin_defaults()              │
  │      └─ Add all missing model cards                         │
  │ 4.2  Fix known data errors (Codestral context, etc.)        │
  │ 4.3  Bridge ProviderFactory with ModelDiscoveryService       │
  │ 4.4  Update docs/providers.md                                │
  │ 4.5  Update README.md with discovery examples                │
  │ 4.6  Add discovery example (examples/advanced/discovery.rs)  │
  │ 4.7  Benchmark cache performance                             │
  │ 4.8  Update CHANGELOG.md                                     │
  └────────────────────────────────────────────────────────────┘
```

---

## Code Patterns

### Pattern: Discovery Provider Wrapping Existing Provider

```rust
/// OpenRouter discovery — wraps existing OpenRouterProvider.
pub struct OpenRouterDiscovery {
    provider: OpenRouterProvider,
}

#[async_trait]
impl ModelDiscoveryProvider for OpenRouterDiscovery {
    fn provider_id(&self) -> &str { "openrouter" }

    fn discovery_strategy(&self) -> DiscoveryStrategy {
        DiscoveryStrategy::Dynamic
    }

    async fn discover_models(&self) -> Result<Vec<DiscoveredModel>> {
        let models = self.provider
            .list_models_cached(Duration::from_secs(3600))
            .await?;

        Ok(models.iter().map(|m| DiscoveredModel {
            id: m.id.clone(),
            name: m.name.clone(),
            provider: "openrouter".into(),
            context_length: m.context_length,
            max_output_tokens: m.architecture.max_completion_tokens
                .unwrap_or(4096),
            capabilities: ModelCapabilities {
                supports_vision: m.architecture.input_modalities
                    .iter().any(|m| m == "image"),
                supports_function_calling: m.supported_parameters
                    .as_ref()
                    .is_some_and(|p| p.iter().any(|s| s == "tools")),
                supports_thinking: m.supported_parameters
                    .as_ref()
                    .is_some_and(|p| p.iter().any(|s| s == "reasoning")),
                supports_streaming: true,
                ..Default::default()
            },
            source: DiscoverySource::DynamicApi,
            discovered_at: chrono::Utc::now(),
            available: true,
            cost_per_m_input: m.pricing.prompt.parse().ok()
                .map(|p: f64| p * 1_000_000.0),
            cost_per_m_output: m.pricing.completion.parse().ok()
                .map(|p: f64| p * 1_000_000.0),
            model_type: ModelType::Llm,
            tags: vec![],
            deprecated: false,
        }).collect())
    }

    async fn get_model_capabilities(&self, model_id: &str)
        -> Result<Option<DiscoveredModel>>
    {
        let models = self.discover_models().await?;
        Ok(models.into_iter().find(|m| m.id == model_id))
    }
}
```

### Pattern: Hybrid Discovery (Static + API)

```rust
/// OpenAI discovery — hybrid (API for available IDs, static for metadata).
pub struct OpenAIDiscovery {
    api_key: String,
    client: Client,
}

#[async_trait]
impl ModelDiscoveryProvider for OpenAIDiscovery {
    fn discovery_strategy(&self) -> DiscoveryStrategy {
        DiscoveryStrategy::Hybrid
    }

    async fn discover_models(&self) -> Result<Vec<DiscoveredModel>> {
        // 1. Fetch live model IDs from API
        let live_ids = self.fetch_model_ids().await.unwrap_or_default();

        // 2. Get static registry
        let mut registry = openai_static_registry();

        // 3. Mark availability based on API response
        for model in &mut registry {
            model.available = live_ids.is_empty()
                || live_ids.contains(&model.id);
            model.source = if live_ids.is_empty() {
                DiscoverySource::StaticRegistry
            } else {
                DiscoverySource::Hybrid
            };
        }

        Ok(registry)
    }
}
```

### Pattern: Static-Only Discovery

```rust
/// xAI discovery — pure static registry.
pub struct XAIDiscovery;

#[async_trait]
impl ModelDiscoveryProvider for XAIDiscovery {
    fn discovery_strategy(&self) -> DiscoveryStrategy {
        DiscoveryStrategy::Static
    }

    async fn discover_models(&self) -> Result<Vec<DiscoveredModel>> {
        Ok(xai_static_registry())
    }
}
```

---

## Testing Strategy

### Unit Tests

```
  discovery/
  ├── types.rs          → CapabilityFilter::matches() exhaustive tests
  ├── cache.rs          → TTL expiry, concurrent access, invalidation
  ├── static_registry.rs → all providers have non-empty registries
  └── providers/
      ├── openai.rs     → static registry completeness
      ├── anthropic.rs  → model ID format validation
      └── ...
```

### Integration Tests (Mocked HTTP)

```rust
#[tokio::test]
async fn test_openrouter_discovery_normalizes_to_discovered_model() {
    let mock_response = r#"{"data":[{"id":"openai/gpt-4o",...}]}"#;
    let server = MockServer::start().await;
    // ... verify normalization
}
```

### E2E Tests (Env-Gated)

```rust
#[tokio::test]
#[ignore] // Run with: cargo test -- --ignored
async fn test_real_ollama_discovery() {
    if std::env::var("OLLAMA_HOST").is_err() { return; }
    let disc = OllamaDiscovery::new();
    let models = disc.discover_models().await.unwrap();
    assert!(!models.is_empty());
    for m in &models {
        assert!(m.context_length > 0);
    }
}
```

---

## Migration Path

### For Existing Users

```
  Before (v0.6.x):
  let provider = OpenAIProvider::from_env()?;
  let ctx = provider.max_context_length();  // still works

  After (v0.7.x) — additive only:
  let provider = OpenAIProvider::from_env()?;
  let ctx = provider.max_context_length();  // UNCHANGED

  // NEW capability:
  let service = ModelDiscoveryService::new();
  let models = service.discover_for_provider("openai").await?;
  let gpt41 = models.iter().find(|m| m.id == "gpt-4.1");
```

### Versioning

- **v0.7.0**: Add `discovery` module (additive, no breaking changes)
- **v0.7.x**: Iterate on discovery implementations
- **v0.8.0**: Consider making `ModelDiscoveryProvider` a supertrait of `LLMProvider` (breaking)

---

## Roadblocks Proved by Code & Mitigations

> **ADDED 2026-07-04**: Every roadblock below is traced to a specific file and line 
> in the codebase. See `09-IMPLEMENTATION-PLAN-FINAL.md` for the full catalog (RB-01 
> through RB-21).

| # | Roadblock | Code Location | Mitigation |
|---|-----------|---------------|------------|
| RB-01 | Duplicate `ProviderType` enum | `factory.rs:61` vs `model_config.rs:114` | Discovery uses string `ProviderId`, not enum |
| RB-02 | Anthropic `MessagesRequest` has no `thinking` field | `anthropic.rs:178-198` | Phase 0: add `thinking` field before discovery |
| RB-03 | Anthropic `context_length_for_model()` pattern-matches model names | `anthropic.rs:699-729` | Replace with registry lookup |
| RB-04 | Gemini `supports_*` methods use `self.model.contains()` | `gemini.rs:2256-2268` | Replace with `ModelProfile` lookup |
| RB-05 | Gemini `ModelProfile` not exported for discovery | `gemini.rs:560-707` | Wrap in discovery adapter |
| RB-07 | OpenAI uses `async-openai` crate, not raw HTTP | `Cargo.toml:47` | Use crate's `models().list()` method |
| RB-08 | OpenRouter `ModelInfo` missing `supported_parameters` | `openrouter.rs` | Add field to struct |
| RB-10 | Bedrock is feature-gated | `providers/mod.rs:55-58` | Gate discovery with same `#[cfg]` |
| RB-12 | Ollama struct missing `capabilities` field | `ollama.rs` | Add `capabilities: Option<Vec<String>>` |
| RB-17 | Azure API version hardcoded to `2024-10-21` | `azure_openai.rs:65` | Update to `2025-06-01` |
| RB-18 | Azure context length hardcoded to 128K for all models | `azure_openai.rs:155` | Add `context_length_for_model()` with model registry |
| RB-19 | Bedrock `MODEL_RULES` maps `anthropic.claude-4` to 200K | `bedrock.rs:255-267` | Split into version-specific prefixes (4.6+ = 1M) |
| RB-20 | Bedrock discovery needs `aws_sdk_bedrock` (not just runtime) | `Cargo.toml` | Add optional `aws-sdk-bedrock` dependency |
| RB-21 | Bedrock thinking uses `{type: "enabled"}` for all models | `bedrock.rs:1208-1216` | Claude 4.7+ needs `{type: "adaptive"}` |

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Static registries become stale | Models missing | Quarterly update process + TOML override |
| Dynamic API changes format | Parse failures | Defensive parsing + fallback to static |
| Local provider offline | No models returned | Graceful empty response, no panics |
| Rate limiting on discovery calls | Slow startup | Aggressive caching (TTL per strategy) |
| Too many models in memory | Memory pressure | Lazy loading, only discover on request |
| Feature-gated Bedrock | Compilation issues | `#[cfg(feature = "bedrock")]` throughout |
| Anthropic breaks on temperature with Opus 4.7+ | 400 API error | Phase 0 prerequisite fix |
| Discovery unused by factory | No actual benefit | Phase 4 bridges discovery → factory |
