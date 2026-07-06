# Model Discovery

The **Model Discovery** system provides unified, programmatic access to model metadata across all LLM providers. Agent harnesses use this module to discover available models, query capabilities, and select the best model for a given task — without hardcoding model IDs or guessing from name patterns.

## Design Principles

1. **No Heuristics** — capabilities come from API responses or verified documentation, never from model name patterns
2. **DRY** — per-provider adapters normalize to a single `DiscoveredModel` type
3. **SOLID** — `ModelDiscoveryProvider` trait (ISP), `ModelDiscoveryService` depends on trait abstraction (DIP)
4. **Zero Breaking Changes** — additive only, existing APIs untouched
5. **Graceful Degradation** — network failures fall back to static registry or empty lists, never panic

## Quick Start (Rust)

```rust
use edgequake_llm::{ModelDiscoveryService, CapabilityFilter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let service = ModelDiscoveryService::new();

    // Discover all models
    let all_models = service.discover_all().await?;
    println!("Found {} models", all_models.len());

    // Filter: vision + tools + large context
    let filter = CapabilityFilter {
        requires_vision: Some(true),
        requires_tools: Some(true),
        min_context_length: Some(100_000),
        exclude_deprecated: Some(true),
        ..Default::default()
    };
    let candidates = service.find_models(&filter).await?;

    // Look up specific model
    if let Some(model) = service.get_model("openai", "gpt-4.1").await? {
        println!("{}: {} tokens", model.id, model.context_length);
    }

    Ok(())
}
```

## Quick Start (Python)

```python
import edgequake_litellm as litellm

# Discover all models
models = litellm.discovery.discover_all()

# Filter by capabilities
vision_models = litellm.discovery.find_models(
    requires_vision=True,
    requires_tools=True,
    min_context_length=100_000,
    max_output_tokens=32_768,
)

# Offline capability search (no API keys)
static_models = litellm.discovery.find_static_models(requires_thinking=True)

# Search by name / fuzzy with length bounds
hits = litellm.discovery.search_static_models_by_name(
    "claude",
    fuzzy=True,
    min_context_length=200_000,
    min_output_tokens=16_384,
)

# List providers from unified catalog
providers = litellm.list_providers()
discovery_providers = litellm.discovery.list_discovery_providers()

# Look up specific model (litellm convention)
info = litellm.discovery.get_model_info("openai/gpt-4.1")
if info:
    print(f"{info['id']}: {info['context_length']} tokens")

# Async variants
models = await litellm.discovery.adiscover_all()
results = await litellm.discovery.afind_models(requires_thinking=True)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ModelDiscoveryService                        │
│  (facade: parallel discovery, caching, filtering)               │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Anthropic│  │  Gemini  │  │  Ollama  │  │  OpenAI  │  ...  │
│  │ Discovery│  │ Discovery│  │ Discovery│  │ Discovery│       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │              │              │              │             │
│  ModelDiscoveryProvider trait (async discover_models())          │
├─────────────────────────────────────────────────────────────────┤
│                      DiscoveryCache                              │
│  (per-provider TTL, RwLock, stale-on-error fallback)            │
├─────────────────────────────────────────────────────────────────┤
│                     Static Registry                              │
│  (source-cited model data, 30+ models, zero heuristics)         │
└─────────────────────────────────────────────────────────────────┘
```

## Discovery Strategies

Each provider uses one of three strategies:

| Strategy | Description | Providers |
|----------|-------------|-----------|
| **Dynamic** | Live API call returns full capability data | Anthropic, Gemini, Ollama, LM Studio, OpenRouter, Mistral |
| **Hybrid** | API call for availability + static data for capabilities | OpenAI, NVIDIA, Bedrock, **Vertex AI** |
| **Static** | Built-in registry with cited documentation | xAI |

## Provider Reference

| Provider | API Endpoint | Auth Required | Official Docs |
|----------|-------------|---------------|---------------|
| OpenAI | `GET /v1/models` | `OPENAI_API_KEY` | [platform.openai.com/docs/models](https://platform.openai.com/docs/models) |
| Anthropic | `GET /v1/models` | `ANTHROPIC_API_KEY` | [docs.anthropic.com/en/api/models](https://docs.anthropic.com/en/api/models) |
| Gemini | `GET /v1beta/models` | `GEMINI_API_KEY` | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) |
| Vertex AI | `GET /v1beta/models` (regional + global) | `GOOGLE_CLOUD_PROJECT` + ADC / `GOOGLE_ACCESS_TOKEN` | [cloud.google.com/vertex-ai](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models) |
| Ollama | `GET /api/tags` | None (local) | [github.com/ollama/ollama/blob/main/docs/api.md](https://github.com/ollama/ollama/blob/main/docs/api.md) |
| LM Studio | `GET /api/v1/models` (fallback: `/v1/models`) | None (local) | [lmstudio.ai/docs/api](https://lmstudio.ai/docs/api) |
| OpenRouter | `GET /api/v1/models` | `OPENROUTER_API_KEY` (optional) | [openrouter.ai/docs/models](https://openrouter.ai/docs/models) |
| Mistral | `GET /v1/models` | `MISTRAL_API_KEY` | [docs.mistral.ai/api/#tag/models](https://docs.mistral.ai/api/#tag/models) |
| NVIDIA | `GET /v1/models` | `NVIDIA_API_KEY` | [docs.api.nvidia.com](https://docs.api.nvidia.com) |
| xAI | Static registry | `XAI_API_KEY` | [docs.x.ai/docs/models](https://docs.x.ai/docs/models) |
| Bedrock | Static registry | AWS credentials | [docs.aws.amazon.com/bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html) |
| OpenAI-Compatible | `GET {base_url}/models` | Varies | — |

## Core Types

### `DiscoveredModel`

```rust
pub struct DiscoveredModel {
    pub id: String,              // Provider-specific model ID
    pub name: String,            // Human-readable name
    pub provider: String,        // Provider identifier
    pub context_length: usize,   // Max input tokens
    pub max_output_tokens: usize,// Max output tokens
    pub capabilities: ModelCapabilities,
    pub source: DiscoverySource, // How it was discovered
    pub discovered_at: DateTime<Utc>,
    pub available: bool,         // Currently accessible
    pub cost_per_m_input: Option<f64>,  // USD per million input tokens
    pub cost_per_m_output: Option<f64>, // USD per million output tokens
    pub model_type: ModelType,   // Llm, Embedding, Multimodal
    pub tags: Vec<String>,       // ["reasoning", "coding", "fast"]
    pub deprecated: bool,
}
```

### `CapabilityFilter`

All fields use AND logic — a model must satisfy every non-None constraint:

```rust
let filter = CapabilityFilter {
    min_context_length: Some(500_000),    // Min input context (tokens)
    max_context_length: Some(2_000_000),  // Max input context (tokens)
    min_output_tokens: Some(8_192),       // Min max-output (tokens)
    max_output_tokens: Some(65_536),      // Max max-output (tokens)
    requires_vision: Some(true),          // Must support images
    requires_tools: Some(true),           // Must support function calling
    requires_thinking: Some(true),        // Must support reasoning
    provider: Some("anthropic".into()),   // Only from Anthropic
    max_cost_per_m_input: Some(10.0),     // At most $10/M input
    exclude_deprecated: Some(true),       // Skip deprecated models
    ..Default::default()
};
```

Builder API:

```rust
use edgequake_llm::{CapabilityFilter, ModelCapability};

let filter = CapabilityFilter::default()
    .requiring_all([ModelCapability::Vision, ModelCapability::Tools])
    .with_min_context_length(100_000)
    .with_max_output_tokens(32_768)
    .excluding_deprecated();
```

Offline search (no network):

```rust
use edgequake_llm::find_static_models;

let models = find_static_models(&filter);
```

### `ModelSearchQuery`

Search models by name, ID, tag, or `provider/model` with optional fuzzy matching and length bounds:

```rust
use edgequake_llm::{search_static_models, ModelSearchQuery};

let hits = search_static_models(
    &ModelSearchQuery::new("claude sonnet")
        .fuzzy(true)
        .with_provider("anthropic")
        .with_min_context_length(200_000)
        .with_min_output_tokens(16_384)
        .with_limit(10),
);

for hit in hits {
    println!("{} score={:.2} kind={:?}", hit.model.id, hit.score, hit.match_kind);
}
```

Match kinds: `ExactId`, `ExactName`, `ExactQualifiedId`, `Prefix`, `Contains`, `Fuzzy`.

Resolve by exact ID or display name:

```rust
use edgequake_llm::static_lookup_by_name;

if let Some(model) = static_lookup_by_name("openai", "GPT-4.1") {
    println!("{}", model.id);
}
```

### `ProviderCatalog`

Single source of truth for provider listing and alias resolution:

```rust
use edgequake_llm::{ProviderCatalog, ProviderFactory};

ProviderFactory::list_providers();
ProviderCatalog::list_discovery_providers();
ProviderCatalog::resolve_id("lm-studio"); // Some("lmstudio")
```

### `DiscoverySource`

```rust
pub enum DiscoverySource {
    DynamicApi,     // Live API response (freshest)
    StaticRegistry, // Built-in verified data
    Hybrid,         // API + static enrichment
    UserConfig,     // User TOML override
    Unknown,        // Model not found — all caps are 0/false
}
```

## Caching

Discovery results are cached per-provider with configurable TTL:

| Provider Type | Default TTL | Rationale |
|---------------|-------------|-----------|
| Cloud providers | 1 hour | Models change infrequently |
| Local (Ollama, LM Studio) | 60 seconds | Users load/unload models |
| Static (xAI) | 24 hours | Only changes on SDK update |

Stale cache is served when an API call fails (graceful degradation over hard failure).

```rust
// Custom TTL
let service = ModelDiscoveryService::builder()
    .cache_ttl(Duration::from_secs(1800)) // 30 minutes
    .build();

// Force re-discovery
service.invalidate_cache("anthropic").await;
```

## Unknown Model Handling

When a model ID is not found in any registry:

- `context_length: 0` (caller must handle)
- `max_output_tokens: 0`
- All `supports_*: false`
- `source: DiscoverySource::Unknown`

The caller sees explicit "unknown" — never a guess. This is the **anti-heuristic** guarantee.

## Builder Pattern

```rust
use edgequake_llm::discovery::{ModelDiscoveryService, providers::openai_compat::OpenAICompatDiscovery};

let service = ModelDiscoveryService::builder()
    .cache_ttl(Duration::from_secs(900))
    // Add a custom OpenAI-compatible endpoint
    .provider(Box::new(
        OpenAICompatDiscovery::new("https://api.groq.com/openai/v1", "groq")
            .with_api_key("gsk_...".into())
    ))
    .build();
```

## Custom Providers

Implement `ModelDiscoveryProvider` for custom discovery sources:

```rust
use async_trait::async_trait;
use edgequake_llm::discovery::{ModelDiscoveryProvider, DiscoveredModel, DiscoveryStrategy};

struct MyProvider;

#[async_trait]
impl ModelDiscoveryProvider for MyProvider {
    fn provider_id(&self) -> &str { "my-provider" }
    fn discovery_strategy(&self) -> DiscoveryStrategy { DiscoveryStrategy::Static }

    async fn discover_models(&self) -> edgequake_llm::Result<Vec<DiscoveredModel>> {
        Ok(vec![/* your models */])
    }
}
```

## Static Registry

The built-in registry contains 30+ models with verified data. Every entry cites its official documentation source. The registry covers:

- **OpenAI**: GPT-5.5, GPT-5.4, GPT-4.1 family, GPT-4o family, o3, o4-mini, embeddings
- **Anthropic**: Claude Fable 5, Opus 4.8, Sonnet 5, Opus 4.7, Opus 4.6, Sonnet 4.6, Sonnet 4.5, Claude 3.5 Sonnet/Haiku
- **Gemini**: Gemini 3.5 Flash, 3.1 Pro, 2.5 Pro, 2.5 Flash
- **Mistral**: Large, Small, Codestral, Magistral Medium, Pixtral Large
- **xAI**: Grok-4, Grok-3, Grok-3 Mini, Grok-2

## Testing

```bash
# Run discovery unit tests
cargo test --lib discovery

# Run full test suite (1400+ tests)
cargo test --lib

# Run the discovery example
cargo run --example discover_models
```

## Module Structure

```
src/discovery/
├── mod.rs              — Module root with re-exports
├── traits.rs           — ModelDiscoveryProvider trait
├── types.rs            — Core types + CapabilityFilter + unit tests
├── search.rs           — Model name/fuzzy search + length bounds
├── cache.rs            — Per-provider TTL cache + unit tests
├── service.rs          — ModelDiscoveryService facade + tests
├── registry.rs         — Static model data (30+ models, source-cited)
└── providers/
    └── ...

src/provider_catalog.rs — Unified provider listing and alias resolution
```

## Specification

The full design specification is available in [`specs/001-edgequake-llm/`](../specs/001-edgequake-llm/00-INDEX.md):

| Document | Description |
|----------|-------------|
| [00-INDEX](../specs/001-edgequake-llm/00-INDEX.md) | Document index |
| [03-MODEL-DISCOVERY-API](../specs/001-edgequake-llm/03-MODEL-DISCOVERY-API.md) | Core API design |
| [04-PROVIDER-DISCOVERY-APPROACHES](../specs/001-edgequake-llm/04-PROVIDER-DISCOVERY-APPROACHES.md) | Per-provider strategy |
| [06-MODEL-CAPABILITY-REGISTRY](../specs/001-edgequake-llm/06-MODEL-CAPABILITY-REGISTRY.md) | Type system design |
| [09-IMPLEMENTATION-PLAN-FINAL](../specs/001-edgequake-llm/09-IMPLEMENTATION-PLAN-FINAL.md) | Phased plan + roadblocks |
