# 03 — Model Discovery API Design

> **Lens**: API/SDK Designer  
> **Date**: 2026-07-04  
> **Status**: DRAFT  
> **Principle**: Open/Closed, Single Responsibility, DRY

## Design Goals

1. **Agent-First** — AI agent harnesses can discover models before instantiation
2. **Unified Interface** — one trait, many implementations
3. **Hybrid Strategy** — dynamic discovery where possible, static fallback where not
4. **Zero Breaking Changes** — new trait, existing traits untouched
5. **Normalized Output** — all providers return `DiscoveredModel` (extends `ModelCard`)
6. **Cacheable** — built-in TTL-based caching for network calls
7. **Filterable** — query by capability (vision, tools, thinking, min context)

---

## Architecture Overview

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    Model Discovery Architecture                      │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                      │
  │   Agent / Application                                                │
  │        │                                                             │
  │        ▼                                                             │
  │   ┌────────────────────────────────┐                                │
  │   │  ModelDiscoveryService         │  ◄── Facade (caching + merge)  │
  │   │  ├─ discover_all()             │                                │
  │   │  ├─ discover_for_provider()    │                                │
  │   │  ├─ find_by_capability()       │                                │
  │   │  └─ get_model_info()           │                                │
  │   └─────────────┬──────────────────┘                                │
  │                  │                                                   │
  │        ┌─────────┼──────────┐                                       │
  │        ▼         ▼          ▼                                       │
  │   ┌─────────┐ ┌────────┐ ┌────────────┐                            │
  │   │ Dynamic │ │ Static │ │  Hybrid    │                             │
  │   │ Disc.   │ │ Reg.   │ │  (merge)   │                             │
  │   └────┬────┘ └───┬────┘ └─────┬──────┘                            │
  │        │          │            │                                     │
  │   ┌────┴───┐ ┌────┴────┐ ┌────┴─────┐                              │
  │   │Ollama  │ │OpenAI   │ │NVIDIA    │                               │
  │   │LMStudio│ │Anthropic│ │(dyn+stat)│                               │
  │   │OpenRtr │ │Gemini   │ │Mistral   │                               │
  │   │        │ │xAI      │ │(dyn+stat)│                               │
  │   │        │ │Bedrock  │ │          │                                │
  │   └────────┘ └─────────┘ └──────────┘                               │
  │                                                                      │
  │   Strategy per provider:                                             │
  │   ┌──────────────────────────────────────────────────────────┐      │
  │   │ Dynamic: API call → parse → normalize → DiscoveredModel │      │
  │   │ Static:  Built-in registry → DiscoveredModel            │      │
  │   │ Hybrid:  API call → merge with static enrichment        │      │
  │   └──────────────────────────────────────────────────────────┘      │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## Core Types

### `DiscoveredModel` — The Universal Model Descriptor

```rust
/// A discovered model with normalized capabilities.
///
/// This is the single source of truth for model metadata,
/// regardless of whether it came from a dynamic API call
/// or a static registry. Extends the existing `ModelCard`
/// with discovery-specific metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredModel {
    // ── Identity ──────────────────────────────────────────
    /// Provider-specific model ID (e.g., "gpt-4.1", "claude-opus-4")
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Provider that owns this model
    pub provider: String,

    // ── Capabilities ──────────────────────────────────────
    /// Normalized capabilities
    pub capabilities: ModelCapabilities,

    // ── Token Limits ──────────────────────────────────────
    /// Maximum input context window (tokens)
    pub context_length: usize,
    /// Maximum output tokens the model can generate
    pub max_output_tokens: usize,

    // ── Discovery Metadata ────────────────────────────────
    /// How this model was discovered
    pub source: DiscoverySource,
    /// When this information was last verified
    pub discovered_at: chrono::DateTime<chrono::Utc>,
    /// Whether the model is currently available/loaded
    pub available: bool,

    // ── Cost (optional) ───────────────────────────────────
    /// Cost per million input tokens (USD)
    pub cost_per_m_input: Option<f64>,
    /// Cost per million output tokens (USD)
    pub cost_per_m_output: Option<f64>,

    // ── Classification ────────────────────────────────────
    /// Model type
    pub model_type: ModelType,
    /// Tags for filtering (e.g., "reasoning", "coding", "fast")
    pub tags: Vec<String>,
    /// Whether the model is deprecated
    pub deprecated: bool,
}

/// How a model was discovered.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DiscoverySource {
    /// Discovered via live API call (most fresh)
    DynamicApi,
    /// From built-in static registry (may be stale)
    StaticRegistry,
    /// Dynamic API enriched with static metadata
    Hybrid,
    /// User-provided configuration (models.toml)
    UserConfig,
}
```

### `ModelCapabilities` — Extended (Backward Compatible)

The existing `ModelCapabilities` struct in `model_config.rs` already has the right fields. We extend it with new fields while keeping all existing ones:

```rust
/// Extended capabilities (new fields only — existing fields preserved).
pub struct ModelCapabilities {
    // ── EXISTING (unchanged) ──────────────────────────────
    pub context_length: usize,
    pub max_output_tokens: usize,
    pub supports_vision: bool,
    pub supports_function_calling: bool,
    pub supports_json_mode: bool,
    pub supports_streaming: bool,
    pub supports_system_message: bool,
    pub embedding_dimension: usize,
    pub max_embedding_tokens: usize,
    pub supports_thinking: bool,
    pub supports_web_search: bool,
    pub default_temperature: f32,

    // ── NEW FIELDS (all optional, default false/None) ─────
    /// Whether model supports structured output (JSON schema)
    pub supports_structured_output: bool,
    /// Whether model supports prompt caching
    pub supports_prompt_caching: bool,
    /// Whether model supports PDF/document input
    pub supports_document_input: bool,
    /// Whether model supports audio input
    pub supports_audio_input: bool,
    /// Whether model supports image generation
    pub supports_image_generation: bool,
    /// Whether model supports computer use / tool use
    pub supports_computer_use: bool,
    /// Supported modalities as string list (e.g., ["text", "image", "audio"])
    pub input_modalities: Vec<String>,
    /// Output modalities
    pub output_modalities: Vec<String>,
    /// Maximum number of images per request (if vision supported)
    pub max_images: Option<usize>,
    /// Reasoning/thinking token budget range (min, max)
    pub thinking_budget_range: Option<(usize, usize)>,
}
```

---

## Trait Design

### `ModelDiscoveryProvider` — The Core Trait

```rust
/// Trait for providers that can discover available models.
///
/// This trait is SEPARATE from `LLMProvider` — it can be implemented
/// independently, allowing discovery without instantiating a chat provider.
///
/// # Design Decisions
///
/// 1. **Separate trait** (not added to LLMProvider) because:
///    - Discovery doesn't require a model to be selected
///    - Some contexts need discovery without chat capability
///    - Follows Interface Segregation Principle
///
/// 2. **Async** because dynamic discovery requires network calls
///
/// 3. **Returns Vec<DiscoveredModel>** for uniformity across providers
#[async_trait]
pub trait ModelDiscoveryProvider: Send + Sync {
    /// Get the provider identifier (e.g., "openai", "anthropic").
    fn provider_id(&self) -> &str;

    /// Discover all available models from this provider.
    ///
    /// For dynamic providers (Ollama, OpenRouter), this makes an API call.
    /// For static providers (OpenAI, Anthropic), this returns built-in data.
    /// For hybrid providers (NVIDIA, Mistral), this merges both sources.
    async fn discover_models(&self) -> Result<Vec<DiscoveredModel>>;

    /// Get the discovery strategy for this provider.
    fn discovery_strategy(&self) -> DiscoveryStrategy;

    /// Get capabilities for a specific model by ID.
    ///
    /// Returns None if the model is not found.
    async fn get_model_capabilities(&self, model_id: &str) -> Result<Option<DiscoveredModel>>;
}

/// How a provider discovers its models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoveryStrategy {
    /// Models discovered via live API call (Ollama, LM Studio, OpenRouter)
    Dynamic,
    /// Models from a built-in static registry (OpenAI, Anthropic, Gemini)
    Static,
    /// API call enriched with static metadata (NVIDIA, Mistral)
    Hybrid,
}
```

### `CapabilityFilter` — Query Interface

```rust
/// Filter for discovering models by capability.
///
/// Used by agent harnesses to find models that match requirements.
#[derive(Debug, Clone, Default)]
pub struct CapabilityFilter {
    /// Minimum context window required
    pub min_context_length: Option<usize>,
    /// Minimum output tokens required
    pub min_output_tokens: Option<usize>,
    /// Must support vision/images
    pub requires_vision: Option<bool>,
    /// Must support tool/function calling
    pub requires_tools: Option<bool>,
    /// Must support thinking/reasoning
    pub requires_thinking: Option<bool>,
    /// Must support streaming
    pub requires_streaming: Option<bool>,
    /// Must support JSON structured output
    pub requires_json_mode: Option<bool>,
    /// Must support embeddings
    pub requires_embeddings: Option<bool>,
    /// Model type filter
    pub model_type: Option<ModelType>,
    /// Provider filter (e.g., only "openai" models)
    pub provider: Option<String>,
    /// Tags filter (any match)
    pub tags: Option<Vec<String>>,
    /// Maximum cost per million input tokens
    pub max_cost_per_m_input: Option<f64>,
}

impl CapabilityFilter {
    pub fn matches(&self, model: &DiscoveredModel) -> bool {
        // Each filter field is AND-ed
        if let Some(min) = self.min_context_length {
            if model.context_length < min { return false; }
        }
        if let Some(min) = self.min_output_tokens {
            if model.max_output_tokens < min { return false; }
        }
        if let Some(true) = self.requires_vision {
            if !model.capabilities.supports_vision { return false; }
        }
        if let Some(true) = self.requires_tools {
            if !model.capabilities.supports_function_calling { return false; }
        }
        if let Some(true) = self.requires_thinking {
            if !model.capabilities.supports_thinking { return false; }
        }
        // ... etc
        true
    }
}
```

---

## ModelDiscoveryService — The Facade

```rust
/// Service that aggregates discovery across all providers.
///
/// This is the main entry point for agent harnesses.
///
/// # Usage
///
/// ```rust,ignore
/// let service = ModelDiscoveryService::new();
///
/// // Discover all models across all providers
/// let all = service.discover_all().await?;
///
/// // Find models matching requirements
/// let filter = CapabilityFilter {
///     requires_vision: Some(true),
///     requires_tools: Some(true),
///     min_context_length: Some(100_000),
///     ..Default::default()
/// };
/// let matches = service.find_models(&filter).await?;
///
/// // Get specific model info
/// let info = service.get_model("openai", "gpt-4.1").await?;
/// ```
pub struct ModelDiscoveryService {
    providers: Vec<Box<dyn ModelDiscoveryProvider>>,
    cache: Arc<RwLock<DiscoveryCache>>,
    cache_ttl: Duration,
}

struct DiscoveryCache {
    entries: HashMap<String, CacheEntry>,
}

struct CacheEntry {
    models: Vec<DiscoveredModel>,
    fetched_at: Instant,
}

impl ModelDiscoveryService {
    /// Create with default providers and 1-hour cache TTL.
    pub fn new() -> Self { ... }

    /// Create with custom cache TTL.
    pub fn with_cache_ttl(ttl: Duration) -> Self { ... }

    /// Register a custom discovery provider.
    pub fn register(&mut self, provider: Box<dyn ModelDiscoveryProvider>) { ... }

    /// Discover all models across all registered providers.
    /// Results are cached per-provider with configurable TTL.
    pub async fn discover_all(&self) -> Result<Vec<DiscoveredModel>> { ... }

    /// Discover models for a specific provider.
    pub async fn discover_for_provider(&self, provider_id: &str)
        -> Result<Vec<DiscoveredModel>> { ... }

    /// Find models matching a capability filter.
    pub async fn find_models(&self, filter: &CapabilityFilter)
        -> Result<Vec<DiscoveredModel>> { ... }

    /// Get detailed info for a specific model.
    pub async fn get_model(&self, provider_id: &str, model_id: &str)
        -> Result<Option<DiscoveredModel>> { ... }

    /// Invalidate cache for a provider (force re-discovery).
    pub fn invalidate_cache(&self, provider_id: &str) { ... }

    /// Get all registered provider IDs.
    pub fn provider_ids(&self) -> Vec<String> { ... }
}
```

---

## Integration with Existing Architecture

### Non-Breaking Extension Points

```
  How the new API connects to existing code:

  ┌────────────────────────┐
  │   EXISTING (unchanged) │
  │                        │
  │  LLMProvider trait     │──── chat, stream, tools
  │  EmbeddingProvider     │──── embed
  │  ProviderFactory       │──── create providers
  │  ProviderRegistry      │──── runtime registry
  │  ModelCapabilities     │──── capability struct
  │  ModelCard             │──── model metadata
  │  ModelsConfig          │──── TOML config
  └────────┬───────────────┘
           │
           │ EXTENDS (no modifications to above)
           ▼
  ┌────────────────────────┐
  │   NEW                  │
  │                        │
  │  ModelDiscoveryProvider │──── discover_models()
  │  DiscoveredModel       │──── normalized output
  │  DiscoverySource       │──── dynamic/static/hybrid
  │  CapabilityFilter      │──── query interface
  │  ModelDiscoveryService │──── facade + caching
  │  StaticModelRegistry   │──── updated model data
  │                        │
  │  Per-provider impls:   │
  │  ├─ OpenAIDiscovery    │──── static registry
  │  ├─ AnthropicDiscovery │──── static + API
  │  ├─ GeminiDiscovery    │──── API discovery
  │  ├─ OllamaDiscovery    │──── existing list_models
  │  ├─ LMStudioDiscovery  │──── existing metadata
  │  ├─ OpenRouterDiscovery│──── existing list_models
  │  ├─ MistralDiscovery   │──── existing list_models
  │  ├─ NvidiaDiscovery    │──── existing list_models
  │  └─ BedrockDiscovery   │──── AWS SDK
  └────────────────────────┘
```

### Bridge: LLMProvider ↔ ModelDiscoveryProvider

For convenience, providers that implement both traits can expose capabilities at the instance level:

```rust
/// Optional extension to LLMProvider for discovery-aware providers.
///
/// This is a default method added to the EXISTING trait (backward compatible).
impl dyn LLMProvider {
    /// Get capabilities of the current model (if discovery is available).
    ///
    /// Default: returns None. Providers implementing ModelDiscoveryProvider
    /// can override to return rich capability data.
    fn current_model_capabilities(&self) -> Option<&ModelCapabilities> {
        None
    }
}
```

---

## Usage Patterns for Agent Harnesses

### Pattern 1: Find the best model for a task

```rust
let service = ModelDiscoveryService::new();

// "I need a model that can see images, call tools, and has at least 100K context"
let filter = CapabilityFilter {
    requires_vision: Some(true),
    requires_tools: Some(true),
    min_context_length: Some(100_000),
    ..Default::default()
};

let candidates = service.find_models(&filter).await?;

// Sort by cost (cheapest first)
let best = candidates.iter()
    .filter(|m| m.cost_per_m_input.is_some())
    .min_by(|a, b| a.cost_per_m_input.partial_cmp(&b.cost_per_m_input).unwrap());
```

### Pattern 2: Discover what's available locally

```rust
let service = ModelDiscoveryService::new();

// Only check local providers
let ollama_models = service.discover_for_provider("ollama").await?;
let lmstudio_models = service.discover_for_provider("lmstudio").await?;

let local_models: Vec<_> = ollama_models.into_iter()
    .chain(lmstudio_models)
    .filter(|m| m.available)
    .collect();
```

### Pattern 3: Validate model before use

```rust
let service = ModelDiscoveryService::new();

// Agent wants to use tool calling — verify the model supports it
let model = service.get_model("anthropic", "claude-opus-4").await?;
if let Some(m) = model {
    if !m.capabilities.supports_function_calling {
        // Fall back to a different model
    }
    if m.max_output_tokens < required_output {
        // Need a model with larger output budget
    }
}
```

---

## Error Handling

```rust
/// Errors specific to model discovery.
#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    /// Provider is not reachable (network error, server down)
    #[error("Provider '{provider}' unreachable: {source}")]
    ProviderUnreachable {
        provider: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Provider returned unexpected response format
    #[error("Failed to parse discovery response from '{provider}': {details}")]
    ParseError {
        provider: String,
        details: String,
    },

    /// Provider requires authentication for discovery
    #[error("Authentication required for '{provider}' model discovery")]
    AuthRequired {
        provider: String,
    },

    /// Discovery timed out
    #[error("Discovery timed out for '{provider}' after {timeout_ms}ms")]
    Timeout {
        provider: String,
        timeout_ms: u64,
    },
}
```

When a dynamic discovery fails, the service **falls back to static data** (never fails completely):

```
  Fallback Chain:

  Dynamic API ──(fail)──► Static Registry ──(fail)──► Empty Vec + Warning
       │                        │
       └── success ◄────────────┘
```

---

## Caching Strategy

```
  Cache Architecture:

  ┌─────────────────────────────────────────────────┐
  │             DiscoveryCache                       │
  ├─────────────────────────────────────────────────┤
  │                                                  │
  │  Per-provider cache with independent TTLs:       │
  │                                                  │
  │  ┌─────────┬──────────────┬────────────────┐    │
  │  │Provider │ Default TTL  │ Rationale       │    │
  │  ├─────────┼──────────────┼────────────────┤    │
  │  │OpenAI   │ 24h          │ Static, rare    │    │
  │  │Anthropic│ 24h          │ Static, rare    │    │
  │  │Gemini   │ 6h           │ API available   │    │
  │  │Ollama   │ 60s          │ Local, changes  │    │
  │  │LMStudio │ 60s          │ Local, changes  │    │
  │  │OpenRtr  │ 1h           │ API, moderate   │    │
  │  │Mistral  │ 6h           │ API available   │    │
  │  │NVIDIA   │ 6h           │ API available   │    │
  │  │xAI      │ 24h          │ Static fallback │    │
  │  │Bedrock  │ 12h          │ AWS API call    │    │
  │  └─────────┴──────────────┴────────────────┘    │
  │                                                  │
  │  Cache invalidation:                             │
  │  - Manual: invalidate_cache("provider")          │
  │  - TTL expiry (per-provider)                     │
  │  - On provider re-initialization                 │
  └─────────────────────────────────────────────────┘
```

---

## File Organization

```
  src/
  ├── discovery/                    ← NEW module
  │   ├── mod.rs                    ← Module root, re-exports
  │   ├── traits.rs                 ← ModelDiscoveryProvider trait
  │   ├── types.rs                  ← DiscoveredModel, CapabilityFilter, etc.
  │   ├── service.rs                ← ModelDiscoveryService (facade)
  │   ├── cache.rs                  ← DiscoveryCache implementation
  │   ├── static_registry.rs        ← Built-in model data (July 2026)
  │   └── providers/                ← Per-provider discovery implementations
  │       ├── mod.rs
  │       ├── openai.rs             ← Static registry
  │       ├── anthropic.rs          ← Static + API
  │       ├── gemini.rs             ← API discovery
  │       ├── ollama.rs             ← Wraps existing list_models
  │       ├── lmstudio.rs           ← Wraps existing metadata
  │       ├── openrouter.rs         ← Wraps existing list_models
  │       ├── mistral.rs            ← Wraps existing list_models
  │       ├── nvidia.rs             ← Wraps existing list_models
  │       ├── xai.rs                ← Static registry
  │       ├── bedrock.rs            ← AWS SDK (feature-gated)
  │       └── openai_compatible.rs  ← Attempts /v1/models
  │
  ├── model_config.rs              ← EXISTING (ModelCapabilities extended)
  ├── traits.rs                    ← EXISTING (LLMProvider unchanged)
  └── lib.rs                       ← Add `pub mod discovery;`
```

---

## Battle-Testing the Design

### Test 1: Agent selects model for coding task

```
Input: "I need a model for coding with tool calling, 100K+ context, under $5/M tokens"
Expected: Filters → matches GPT-4.1, Claude Sonnet 4, Gemini 2.5 Flash
         Sorts by cost → returns cheapest matching model
```

### Test 2: Local-only discovery when offline

```
Input: discover_all() with no internet
Expected: Ollama (if running) and LM Studio (if running) return models
         Cloud providers return static registry data (cached or built-in)
         No panics, no errors
```

### Test 3: Model added to Ollama mid-session

```
Input: User runs `ollama pull llama3.3:70b` while agent is running
Expected: After cache TTL (60s), discover_for_provider("ollama") includes new model
         Agent can immediately use the newly available model
```

### Test 4: Provider API down

```
Input: OpenRouter API returns 503
Expected: Falls back to cached data (if any) or empty list
         Logs warning but does not error
         Other providers unaffected
```

### Test 5: Model deprecated

```
Input: Model has deprecated=true and replacement set
Expected: find_models() includes it with deprecated flag
         Agent harness can filter out deprecated models
         Replacement model ID is available for migration
```
