//! EdgeQuake LLM - LLM and Embedding Provider Abstraction
//!
//! # Implements
//!
//! - **FEAT0017**: Multi-Provider LLM Support
//! - **FEAT0018**: Embedding Provider Abstraction
//! - **FEAT0019**: LLM Response Caching
//! - **FEAT0020**: API Rate Limiting
//! - **FEAT0005**: Embedding Generation (via providers)
//!
//! # Enforces
//!
//! - **BR0301**: LLM API rate limits (configurable per provider)
//! - **BR0302**: Document size limits (context window awareness)
//! - **BR0303**: Cost tracking per request
//! - **BR0010**: Embedding dimension validated (1536 default)
//!
//! This crate provides traits and implementations for:
//! - Text completion (LLM providers)
//! - Text embedding (embedding providers)
//! - Token counting and management
//! - Rate limiting for API calls
//! - Response caching for cost reduction
//!
//! # Providers
//!
//! | Provider | FEAT0017 | Chat | Embeddings | Notes |
//! |----------|----------|------|------------|-------|
//! | OpenAI | ✓ | ✓ | ✓ | Primary production provider |
//! | Azure OpenAI | ✓ | ✓ | ✓ | Enterprise deployments |
//! | Ollama | ✓ | ✓ | ✓ | Local/on-prem models |
//! | LM Studio | ✓ | ✓ | ✓ | Local OpenAI-compatible API |
//! | Gemini | ✓ | ✓ | ✓ | Google AI |
//! | Mock | ✓ | ✓ | ✓ | Testing (no API calls) |
//!
//! # Architecture
//!
//! The crate uses trait-based abstraction to support multiple LLM backends:
//! - OpenAI (GPT-4, GPT-3.5)
//! - OpenAI-compatible APIs (Ollama, LM Studio, etc.)
//! - Anthropic (Claude 3.5, Claude 3)
//! - Future: Mistral, local models
//!
//! # Example
//!
//! ```ignore
//! use edgequake_llm::{LLMProvider, OpenAIProvider};
//!
//! let provider = OpenAIProvider::new("your-api-key");
//! let response = provider.complete("Hello, world!").await?;
//! ```
//!
//! # See Also
//!
//! - [`crate::traits`] for provider trait definitions
//! - [`crate::providers`] for concrete implementations
//! - [`crate::cache`] for response caching

pub mod cache;
pub mod cache_prompt;
pub mod cost_tracker; // OODA-21: Session-level cost tracking
pub mod error;
pub mod factory;
pub mod inference_metrics; // OODA-33: Unified streaming metrics
pub mod middleware;
pub mod model_config;
pub mod providers;
pub mod rate_limiter;
pub mod registry;
pub mod reranker;
pub mod retry;
pub mod tokenizer;
pub mod traits;

pub use cache::{CacheConfig, CacheStats, CachedProvider, LLMCache};
pub use cache_prompt::{
    apply_cache_control, parse_cache_stats, CachePromptConfig, CacheStats as PromptCacheStats,
};
pub use cost_tracker::{
    format_cost, format_tokens, CostEntry, CostSummary, ModelPricing, SessionCostTracker,
};
pub use error::{LlmError, Result, RetryStrategy};
pub use factory::{ProviderFactory, ProviderType};
pub use inference_metrics::InferenceMetrics; // OODA-33
pub use middleware::{
    LLMMiddleware, LLMMiddlewareStack, LLMRequest, LogLevel, LoggingLLMMiddleware,
    MetricsLLMMiddleware, MetricsSummary,
};
pub use model_config::{
    DefaultsConfig, ModelCapabilities, ModelCard, ModelConfigError, ModelCost, ModelType,
    ModelsConfig, ProviderConfig, ProviderType as ConfigProviderType,
};
pub use providers::azure_openai::AzureOpenAIProvider;
pub use providers::gemini::GeminiProvider;
pub use providers::jina::JinaProvider;
pub use providers::lmstudio::LMStudioProvider;
pub use providers::mock::MockProvider;
pub use providers::ollama::{
    OllamaModelDetails, OllamaModelInfo, OllamaModelsResponse, OllamaProvider,
};
pub use providers::openai::OpenAIProvider;
// FEAT-007: Mistral AI provider
pub use providers::mistral::MistralProvider;
// FEAT-020: AWS Bedrock provider (feature-gated)
#[cfg(feature = "bedrock")]
pub use providers::bedrock::BedrockProvider;
// OODA-01: Anthropic (Claude) provider
pub use providers::anthropic::AnthropicProvider;
// OODA-02: OpenRouter provider (200+ models)
// OODA-72: Dynamic model discovery with caching
pub use providers::openrouter::{
    ModelArchitecture as OpenRouterModelArchitecture, ModelInfo as OpenRouterModelInfo,
    ModelPricing as OpenRouterModelPricing, ModelsResponse as OpenRouterModelsResponse,
    OpenRouterProvider,
};
// OODA-200: Configurable OpenAI-compatible provider
pub use providers::openai_compatible::OpenAICompatibleProvider;
// OODA-71: xAI Grok provider (api.x.ai)
pub use providers::vscode::{
    Model as CopilotModel, ModelsResponse as CopilotModelsResponse, VsCodeCopilotProvider,
};
pub use providers::xai::XAIProvider;
pub use rate_limiter::{RateLimitedProvider, RateLimiter, RateLimiterConfig};
pub use registry::ProviderRegistry;
pub use reranker::{
    BM25Reranker, HttpReranker, HybridReranker, MockReranker, RRFReranker, RerankConfig,
    RerankResult, Reranker, ScoreAggregation, TermOverlapReranker,
};
pub use retry::RetryExecutor;
pub use tokenizer::Tokenizer;
pub use traits::{
    CacheControl, ChatMessage, ChatRole, CompletionOptions, EmbeddingProvider, FunctionCall,
    FunctionDefinition, ImageData, LLMProvider, LLMResponse, ToolCall, ToolChoice, ToolDefinition,
    ToolResult,
};
