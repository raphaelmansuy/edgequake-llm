//! LLM provider implementations.

pub mod openai;

pub mod mock;
// OODA-45: Export MockAgentProvider for E2E testing
pub use mock::{MockAgentProvider, MockProvider, MockResponse};

pub mod gemini;

pub mod azure_openai;

pub mod jina;

pub mod ollama;

pub mod lmstudio;

pub mod vscode;

// OODA-01: Anthropic (Claude) provider
pub mod anthropic;
pub use anthropic::AnthropicProvider;

// OODA-02: OpenRouter provider (200+ models)
pub mod openrouter;
pub use openrouter::OpenRouterProvider;

// OODA-200: Configurable OpenAI-compatible provider
pub mod openai_compatible;
pub use openai_compatible::OpenAICompatibleProvider;

// OODA-71: xAI Grok provider (api.x.ai)
pub mod xai;
pub use xai::XAIProvider;

// OODA-80: HuggingFace Hub provider (open-source models)
pub mod huggingface;
pub use huggingface::HuggingFaceProvider;

// OODA-LOG-03: Tracing wrapper for GenAI observability
pub mod tracing;
pub use self::tracing::TracingProvider;

// OODA-LOG-11: GenAI event emission for OpenTelemetry
pub mod genai_events;

// FEAT-007: Mistral AI provider (chat, embeddings, list-models)
pub mod mistral;
pub use mistral::MistralProvider;
