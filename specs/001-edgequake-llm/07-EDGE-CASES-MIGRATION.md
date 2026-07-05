# 07 — Edge Cases, Migration & Ascending Compatibility

> **Lens**: Product Owner  
> **Date**: 2026-07-04  
> **Status**: DRAFT  
> **Principle**: Zero breaking changes, graceful degradation, battle-tested

## Ascending Compatibility Guarantee

### What MUST NOT change (v0.6 → v0.7)

```
  Public API Stability Contract:

  ┌─────────────────────────────────────────────────────┐
  │  FROZEN (no changes to signatures or behavior)       │
  ├─────────────────────────────────────────────────────┤
  │                                                      │
  │  Traits:                                             │
  │  ├─ LLMProvider (all methods)                        │
  │  ├─ EmbeddingProvider (all methods)                  │
  │  └─ All default implementations                      │
  │                                                      │
  │  Types:                                              │
  │  ├─ ChatMessage, ChatRole                            │
  │  ├─ LLMResponse                                      │
  │  ├─ CompletionOptions                                │
  │  ├─ ToolDefinition, ToolCall, ToolResult             │
  │  ├─ StreamChunk, StreamUsage                         │
  │  ├─ ImageData, CacheControl                          │
  │  └─ LlmError, Result                                 │
  │                                                      │
  │  Structs:                                            │
  │  ├─ ProviderFactory, ProviderType                    │
  │  ├─ ProviderRegistry                                 │
  │  ├─ ModelCapabilities (existing fields)              │
  │  └─ ModelCard (existing fields)                      │
  │                                                      │
  │  Provider constructors:                              │
  │  ├─ OpenAIProvider::from_env()                       │
  │  ├─ AnthropicProvider::from_env()                    │
  │  ├─ GeminiProvider::from_env()                       │
  │  ├─ OllamaProvider::from_env()                       │
  │  └─ ... (all from_env() methods)                     │
  │                                                      │
  │  Exports from lib.rs:                                │
  │  └─ All existing `pub use` statements                │
  └─────────────────────────────────────────────────────┘
```

### What CAN be added (additive changes only)

```
  ADDITIVE (new items, existing items untouched):

  ├─ New trait: ModelDiscoveryProvider
  ├─ New types: DiscoveredModel, CapabilityFilter, DiscoverySource
  ├─ New service: ModelDiscoveryService
  ├─ New module: src/discovery/
  ├─ New fields on ModelCapabilities (with #[serde(default)])
  ├─ New default methods on LLMProvider (if any)
  ├─ New exports from lib.rs (pub use discovery::...)
  └─ Updated builtin_defaults() model cards (data-only, same schema)
```

---

## Edge Cases Catalog

### EC-01: Provider Unreachable During Discovery

**Scenario**: Ollama server is not running when `discover_models()` is called.

**Expected Behavior**:
1. HTTP request to `localhost:11434/api/tags` fails with connection refused
2. Discovery logs a warning: `"Ollama unreachable at {host}, returning empty model list"`
3. Returns `Ok(Vec::new())` — NOT an error
4. `ModelDiscoveryService` continues with other providers
5. If cached data exists and is within 2× TTL, return stale cache with `available: false`

**Test**:
```rust
#[tokio::test]
async fn test_ollama_unreachable_returns_empty() {
    let disc = OllamaDiscovery::with_host("http://localhost:99999");
    let result = disc.discover_models().await;
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}
```

### EC-02: API Key Missing for Discovery

**Scenario**: `OPENAI_API_KEY` not set when trying to discover OpenAI models.

**Expected Behavior**:
1. OpenAI discovery cannot call `/v1/models` (auth required)
2. Falls back to static registry (all models listed, `available: unknown`)
3. Sets `source: DiscoverySource::StaticRegistry`
4. Logs info: `"OpenAI API key not set, using static model registry"`

### EC-03: Dynamic Discovery Returns Unknown Models

**Scenario**: OpenRouter adds a new model not in our static registry.

> **REVISED 2026-07-04**: The `infer_capabilities()` heuristic has been removed.
> For providers with rich APIs (like OpenRouter), the API response IS the truth.

**Expected Behavior**:
1. Model appears in `discover_models()` with ID and name from API
2. Capabilities are mapped **directly** from the API response fields:
   - `context_length` from OpenRouter's `context_length` field
   - `supports_function_calling` from `supported_parameters` contains `"tools"`
   - `supports_vision` from `architecture.input_modalities` contains `"image"`
   - `supports_thinking` from `supported_parameters` contains `"reasoning"`
3. For providers with minimal APIs (e.g., OpenAI), if the model ID is not in the
   static registry, capabilities are set to `DiscoverySource::Unknown`:
   - `context_length: 0` (UNKNOWN — caller must handle)
   - All `supports_*: false` (UNKNOWN — caller must handle)
   - `source: DiscoverySource::Unknown` signals to the caller explicitly
4. Model is usable immediately via `ProviderFactory`
5. A `tracing::warn!` is emitted for unknown models to aid debugging

### EC-04: Model ID Aliases and Snapshots

**Scenario**: User passes `"claude-sonnet-4-5-latest"` but registry has `"claude-sonnet-4-5-20250929"`.

**Expected Behavior**:
1. `get_model_capabilities()` first tries exact match
2. If not found, tries alias resolution:
   - Strip `-latest` suffix, find by prefix
   - Map known aliases (e.g., `claude-3.5-sonnet` → `claude-3-5-sonnet-20241022`)
3. If still not found, return `None` (don't guess)

**Alias Registry**:
```rust
fn resolve_alias(model_id: &str) -> Option<&str> {
    match model_id {
        "claude-sonnet-4-5-latest" => Some("claude-sonnet-4-5-20250929"),
        "claude-opus-4-latest" => Some("claude-opus-4-20250514"),
        "claude-sonnet-4-latest" => Some("claude-sonnet-4-20250514"),
        "claude-haiku-4-latest" => Some("claude-haiku-4-20250514"),
        "gpt-4o-latest" => Some("gpt-4o"),
        _ => None,
    }
}
```

### EC-05: Context Length Changes at Runtime (Ollama)

**Scenario**: User changes `num_ctx` parameter in Ollama's Modelfile.

**Expected Behavior**:
1. `OllamaDiscovery::discover_models()` queries `/api/show` per model
2. `num_ctx` from model parameters is used as `context_length`
3. Cache TTL is 60s, so changes are reflected within a minute
4. Existing `refresh_model_metadata()` on `LLMProvider` still works

### EC-06: Concurrent Discovery Calls

**Scenario**: Multiple agent threads call `discover_all()` simultaneously.

**Expected Behavior**:
1. `DiscoveryCache` uses `RwLock` — multiple reads allowed
2. Only one write (cache miss → fetch) proceeds; others wait
3. Consider using `tokio::sync::OnceCell` for per-provider dedup
4. Total time = max(provider_latency) not sum(provider_latency)

```rust
impl ModelDiscoveryService {
    async fn discover_all(&self) -> Result<Vec<DiscoveredModel>> {
        let futures: Vec<_> = self.providers.iter()
            .map(|p| self.discover_cached(p.provider_id()))
            .collect();

        let results = futures::future::join_all(futures).await;

        let mut all = Vec::new();
        for result in results {
            match result {
                Ok(models) => all.extend(models),
                Err(e) => tracing::warn!("Discovery failed: {}", e),
            }
        }
        Ok(all)
    }
}
```

### EC-06b: Vertex AI Token Expiry During Discovery

> **ADDED 2026-07-04**: OAuth2 access tokens expire after ~1 hour.

**Scenario**: A long-running agent process started 2 hours ago. It calls
`discover_for_provider("vertexai")` which uses a stale access token.

**Expected Behavior**:
1. Discovery call to Vertex AI returns HTTP 401 Unauthorized
2. Discovery catches the 401 and attempts token refresh:
   a. Re-read `GOOGLE_ACCESS_TOKEN` env var (may have been updated)
   b. Fallback: call `gcloud auth print-access-token` for fresh token
3. Retry the discovery call with the new token
4. If refresh also fails, return cached stale data (if available)
5. Log warning: `"Vertex AI token expired, serving cached data"`

**Test**:
```rust
#[tokio::test]
async fn test_vertex_discovery_handles_401_with_stale_cache() {
    // Mock server returns 401, cache has stale data
    // Verify: stale data returned, warning logged
}
```

### EC-06c: Vertex AI Without gcloud CLI Installed

**Scenario**: Production deployment on Cloud Run where `gcloud` CLI is not
available. `GOOGLE_ACCESS_TOKEN` is not set either (relies on ADC).

**Expected Behavior** (current — known limitation):
1. `from_env_vertex_ai()` tries `GOOGLE_CLOUD_PROJECT` → found
2. Tries `GOOGLE_ACCESS_TOKEN` → not set
3. Tries `gcloud auth print-access-token` → fails (gcloud not installed)
4. Returns `ConfigError` with clear message about options
5. Discovery returns empty list for Vertex AI, does NOT panic
6. Other providers continue working

**Expected Behavior** (future — with ADC support):
1. Google Auth library auto-detects attached service account via metadata server
2. Token is obtained automatically, refreshed on expiry
3. No gcloud CLI or env vars needed

### EC-06d: Vertex AI Global vs Regional Model Availability

**Scenario**: User's `GOOGLE_CLOUD_REGION=us-central1`. They request discovery
of Gemini 3.x models which are global-endpoint-only.

**Expected Behavior**:
1. Discovery calls regional endpoint → Gemini 2.x models returned
2. Discovery calls global endpoint → Gemini 3.x models returned
3. Results are merged and deduplicated by model ID
4. Each model's metadata includes `effective_region` for routing

### EC-07: Feature-Gated Bedrock Discovery

**Scenario**: Bedrock feature not enabled but user expects AWS models.

**Expected Behavior**:
1. `BedrockDiscovery` is behind `#[cfg(feature = "bedrock")]`
2. When feature disabled, `ModelDiscoveryService` simply doesn't include Bedrock
3. Static registry for Bedrock models can still exist (data only, no AWS SDK dependency)
4. User sees models in static list but `available: false`

### EC-08: OpenAI-Compatible with Unknown Service

**Scenario**: User sets `OPENAI_COMPATIBLE_BASE_URL=https://custom.corp.com/v1`.

**Expected Behavior**:
1. `OpenAICompatibleDiscovery` attempts `GET {base_url}/models`
2. If 200 with JSON: parse model IDs; capabilities set to `DiscoverySource::Unknown`
   (no inference — we cannot know what an unknown service supports)
3. If 404: return empty list, warn "provider doesn't support model listing"
4. If timeout: return empty list, warn
5. If 200 with non-standard format: try parse, warn on failure, return empty

### EC-09: Empty Model Name from Provider

**Scenario**: Dynamic API returns a model with empty `id` or `name`.

**Expected Behavior**:
1. Skip models with empty IDs
2. If `name` is empty but `id` is present, use `id` as `name`
3. Log warning: "Skipping model with empty ID from {provider}"

### EC-10: Rate Limiting During Discovery

**Scenario**: Rapid `discover_all()` calls hit provider rate limits.

**Expected Behavior**:
1. Cache prevents actual API calls within TTL window
2. If API returns 429, cache the error state temporarily (30s)
3. Don't retry discovery on rate limit — serve stale cache or static
4. Log: "Rate limited by {provider}, serving cached data"

### EC-11: TOML Config Overrides Static Registry

**Scenario**: User has `models.toml` with custom model cards.

**Expected Behavior**:
1. User config takes precedence over built-in static data
2. `DiscoverySource::UserConfig` has highest priority
3. If user defines a model, it replaces the static entry
4. If user adds a model not in static, it's appended
5. Dynamic discovery results merge with user overrides

```
  Priority Order (highest to lowest):

  1. User TOML config (models.toml)
  2. Dynamic API discovery
  3. Static built-in registry
  4. Explicit Unknown (DiscoverySource::Unknown)
```

> **REVISED 2026-07-04**: "Capability inference heuristic" has been removed from
> the priority chain. Unknown models are surfaced as explicitly unknown, not guessed.

### EC-12: Deprecated Model Still Works

**Scenario**: Model is marked `deprecated: true` but API still accepts it.

**Expected Behavior**:
1. `discover_models()` includes it with `deprecated: true`
2. `replacement` field points to successor model
3. `find_models()` includes deprecated models by default
4. Agent harnesses can filter: `.filter(|m| !m.deprecated)`
5. Log info on first use: "Model {id} is deprecated, consider {replacement}"

### EC-13: Azure Deployment Name ≠ Model ID

> **ADDED 2026-07-04**: Azure OpenAI uses deployment names (user-chosen) not model IDs.

**Scenario**: Azure deployment `my-smart-model` is backed by `gpt-5.4`. The discovery
system cannot determine capabilities from the deployment name alone.

**Expected Behavior**:
1. If management API available (Azure AD token): resolve deployment → model name + version
2. Map resolved model name to static registry for context length + capabilities
3. If management API unavailable: use data plane `GET /openai/models` for capability flags
4. Fallback: treat deployment name as potential model name (common convention)
5. `DiscoveredModel.id` = deployment name, `DiscoveredModel.tags["azure_model"]` = resolved model

```
  ┌──────────────────────────────────────────────────────────────┐
  │  RESOLUTION CHAIN (Azure Deployment → Capabilities)         │
  │                                                              │
  │  1. Management API (needs Azure AD token)                    │
  │     GET .../deployments → { model: { name: "gpt-5.4" } }    │
  │     ↓                                                        │
  │  2. Static registry lookup                                   │
  │     "gpt-5.4" → { context: 400K, vision: true, tools: true }│
  │     ↓                                                        │
  │  3. Fallback: data plane model catalog                       │
  │     GET /openai/models → capabilities{} (no context lengths) │
  └──────────────────────────────────────────────────────────────┘
```

---

### EC-14: Azure API Version Mismatch

**Scenario**: User has `AZURE_OPENAI_API_VERSION=2024-10-21` but deploys a GPT-5.4 model
which requires a newer API version for full feature support.

**Expected Behavior**:
1. Discovery detects the API version from the provider configuration
2. If API version is < `2025-06-01`, log warning: "API version {v} may not support GPT-5.x features"
3. Model capabilities reported as `Unknown` for features requiring newer API versions
4. Do NOT silently override user's API version — warn and document

---

### EC-15: Bedrock Inference Profile Required for Discovery

**Scenario**: `ListFoundationModels` returns bare model IDs (e.g., `anthropic.claude-sonnet-5`)
but actual inference requires geo-prefixed IDs (e.g., `us.anthropic.claude-sonnet-5`).

**Expected Behavior**:
1. Discovery reports the bare model ID (as returned by API)
2. `DiscoveredModel.metadata["requires_inference_profile"]` = true/false
3. `DiscoveredModel.metadata["inference_profile_prefix"]` set based on region
4. Provider reuses existing `resolve_model_id_for_region()` at invocation time
5. The discovery output is region-agnostic; profile resolution happens at call time

---

### EC-16: Bedrock Feature Gate Affects Discovery

**Scenario**: edgequake-llm built WITHOUT `bedrock` feature. Code references
`aws_sdk_bedrock` for `ListFoundationModels` would cause compilation errors.

**Expected Behavior**:
1. ALL Bedrock discovery code behind `#[cfg(feature = "bedrock")]`
2. `ModelDiscoveryService` skips Bedrock provider when feature disabled
3. Static registry for Bedrock still available (no AWS SDK needed)
4. Conditional import in `discovery/mod.rs`:
   ```rust
   #[cfg(feature = "bedrock")]
   pub mod bedrock;
   ```

---

### EC-17: Bedrock Thinking Type Version Mismatch

**Scenario**: User sends `thinking_budget_tokens: Some(5000)` to Claude Sonnet 5 on Bedrock.
Current code sends `{type: "enabled"}` but Claude 4.7+ requires `{type: "adaptive"}`.

**Expected Behavior**:
1. Provider detects model version from model ID
2. Claude 4.7+, Opus 4.8, Sonnet 5, Fable 5 → `{type: "adaptive"}`
3. Claude 4.5, Sonnet 4.5 → `{type: "enabled", budget_tokens: N}`
4. Models without thinking support → omit the field entirely
5. Error message if version detection fails: "Unknown thinking type for model {id}"

---

## Migration Guide (v0.6 → v0.7)

### For Library Users (No Changes Required)

```rust
// v0.6 code — WORKS UNCHANGED in v0.7
use edgequake_llm::{OpenAIProvider, LLMProvider, ChatMessage};

let provider = OpenAIProvider::from_env()?;
let response = provider.chat(&[ChatMessage::user("Hello")], None).await?;
println!("{}", response.content);
```

### For Agent Harnesses (New Capability)

```rust
// v0.7 — NEW discovery API (opt-in, additive)
use edgequake_llm::discovery::{
    ModelDiscoveryService, CapabilityFilter, DiscoveredModel
};

let service = ModelDiscoveryService::new();

// Discover all available models
let models = service.discover_all().await?;
println!("Found {} models across all providers", models.len());

// Find models for a specific task
let coding_models = service.find_models(&CapabilityFilter {
    requires_tools: Some(true),
    requires_thinking: Some(true),
    min_context_length: Some(100_000),
    ..Default::default()
}).await?;

for model in &coding_models {
    println!("{}/{}: {}K context, ${}/M input",
        model.provider, model.id,
        model.context_length / 1000,
        model.cost_per_m_input.unwrap_or(0.0));
}
```

### For Provider Authors (Implement Trait)

```rust
// v0.7 — implement ModelDiscoveryProvider for your custom provider
use edgequake_llm::discovery::{
    ModelDiscoveryProvider, DiscoveredModel, DiscoveryStrategy
};

struct MyCustomDiscovery { /* ... */ }

#[async_trait]
impl ModelDiscoveryProvider for MyCustomDiscovery {
    fn provider_id(&self) -> &str { "my_custom" }
    fn discovery_strategy(&self) -> DiscoveryStrategy {
        DiscoveryStrategy::Dynamic
    }
    async fn discover_models(&self) -> Result<Vec<DiscoveredModel>> {
        // Call your API, return normalized models
    }
    async fn get_model_capabilities(&self, model_id: &str)
        -> Result<Option<DiscoveredModel>>
    {
        let models = self.discover_models().await?;
        Ok(models.into_iter().find(|m| m.id == model_id))
    }
}

// Register with service
let mut service = ModelDiscoveryService::new();
service.register(Box::new(MyCustomDiscovery::new()));
```

---

## Deprecation Strategy for model_config.rs

The current `model_config.rs::builtin_defaults()` continues to work but becomes the **fallback** data source. The new `discovery/static_registry.rs` is the canonical source for model data.

```
  Phase 1 (v0.7): Both systems coexist
  ├─ model_config.rs: updated with July 2026 data
  ├─ discovery/static_registry.rs: same data, normalized format
  └─ No duplication — static_registry imports from model_config

  Phase 2 (v0.8): static_registry becomes primary
  ├─ model_config.rs: wraps static_registry
  └─ ModelCard generated from DiscoveredModel

  Phase 3 (v1.0): model_config.rs simplified
  └─ Only user TOML config parsing remains
```

---

## Quality Gates

Before merging, the implementation MUST pass:

```
  ┌──────────────────────────────────────────────────────┐
  │  Quality Gate Checklist                               │
  ├──────────────────────────────────────────────────────┤
  │                                                       │
  │  ✓ All existing tests pass (cargo test --workspace)  │
  │  ✓ No clippy warnings (cargo clippy --all-targets)   │
  │  ✓ Formatted (cargo fmt --check)                     │
  │  ✓ Docs build (cargo doc --no-deps)                  │
  │  ✓ No new unsafe code                                │
  │  ✓ All new types are Serialize + Deserialize          │
  │  ✓ All new types are Clone + Debug                    │
  │  ✓ All edge cases have tests (EC-01 through EC-12)    │
  │  ✓ At least one integration test per strategy type    │
  │  ✓ Cache TTL tests (verify expiry behavior)           │
  │  ✓ Concurrent access tests (tokio multi-thread)       │
  │  ✓ Feature-gated Bedrock compiles both ways            │
  │  ✓ Python package (edgequake-litellm) still builds    │
  │  ✓ Examples compile (cargo build --examples)          │
  │  ✓ CHANGELOG.md updated                              │
  │  ✓ README.md updated with discovery example           │
  │                                                       │
  └──────────────────────────────────────────────────────┘
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Model coverage | 95%+ of available models per provider | Compare discovery output vs provider docs |
| Capability accuracy | 99% correct context lengths | Validate against official documentation |
| Discovery latency (cached) | < 1ms per provider | Benchmark with criterion |
| Discovery latency (cold) | < 5s total (all providers) | Parallel fetch, timeout at 3s each |
| Memory overhead | < 1MB for full registry | Measure with `std::mem::size_of_val` |
| API surface simplicity | 3 primary methods | discover_all, find_models, get_model |
| Breaking changes | 0 | Compile existing examples without changes |
