//! xAI Grok Provider - Direct access to xAI's Grok models.
//!
//! @implements OODA-71: xAI Grok API Integration
//!
//! # Overview
//!
//! This provider connects directly to xAI's API (api.x.ai) for Grok models.
//! xAI's API is OpenAI-compatible, so we leverage `OpenAICompatibleProvider`
//! internally for maximum code reuse and battle-tested functionality.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    xAI Provider Architecture                            │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │   User Request                                                           │
//! │        │                                                                 │
//! │        ▼                                                                 │
//! │   ┌────────────┐     ┌──────────────────────────┐     ┌──────────────┐ │
//! │   │ XAIProvider │────►│ OpenAICompatibleProvider │────►│ api.x.ai     │ │
//! │   │ (wrapper)   │     │ (implementation)         │     │ /v1/chat/*   │ │
//! │   └────────────┘     └──────────────────────────┘     └──────────────┘ │
//! │                                                                          │
//! │   XAIProvider provides:                                                  │
//! │   - XAI_API_KEY environment detection                                   │
//! │   - Default base URL: https://api.x.ai                                  │
//! │   - Default model: grok-4                                               │
//! │   - Model catalog with context sizes                                    │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Environment Variables
//!
//! | Variable | Required | Default | Description |
//! |----------|----------|---------|-------------|
//! | `XAI_API_KEY` | ✅ Yes | - | xAI API key from console.x.ai |
//! | `XAI_MODEL` | ❌ No | `grok-4` | Default model to use |
//! | `XAI_BASE_URL` | ❌ No | `https://api.x.ai` | API endpoint override |
//!
//! # Available Models
//!
//! | Model | Context | Features |
//! |-------|---------|----------|
//! | `grok-4` | 128K | Flagship reasoning model |
//! | `grok-4-0709` | 128K | July 2025 release |
//! | `grok-4.1-fast` | 2M | Fast agentic, tool calling |
//! | `grok-3` | 128K | Previous generation |
//! | `grok-3-mini` | 128K | Smaller, faster |
//! | `grok-2-vision-1212` | 32K | Image understanding |
//! | `grok-code-fast-1` | 128K | Fast coding assistant |
//!
//! # Example
//!
//! ```bash
//! # Set API key
//! export XAI_API_KEY=xai-your-api-key
//!
//! # Use with EdgeCode (auto-detected)
//! edgecode react "Write hello world in Rust"
//!
//! # Explicit provider selection
//! edgecode react --provider xai "Write hello world in Rust"
//!
//! # Use specific model
//! export XAI_MODEL=grok-4.1-fast
//! edgecode react "Build a complex app"
//! ```

use async_trait::async_trait;
use futures::stream::BoxStream;
use tracing::debug;

use crate::error::{LlmError, Result};
use crate::model_config::{ModelCapabilities, ModelCard, ModelType, ProviderConfig, ProviderType as ConfigProviderType};
use crate::traits::StreamChunk;
use crate::providers::openai_compatible::OpenAICompatibleProvider;
use crate::traits::{
    ChatMessage, CompletionOptions, EmbeddingProvider, LLMProvider, LLMResponse,
};

// ============================================================================
// Constants
// ============================================================================

/// Default xAI API base URL (includes /v1 prefix for OpenAI compatibility)
const XAI_BASE_URL: &str = "https://api.x.ai/v1";

/// Default model
const XAI_DEFAULT_MODEL: &str = "grok-4";

/// Provider display name
const XAI_PROVIDER_NAME: &str = "xai";

/// xAI model catalog with context lengths.
///
/// WHY: Pre-defined models ensure users get correct context limits without
/// having to check documentation. Updated from docs.x.ai/docs/models (OODA-15).
///
/// Model specifications (July 2025):
/// - Grok 4: 256K context, reasoning model, multimodal (text, image)
/// - Grok 4.1 Fast: 2M context, optimized for agentic tool calling
/// - Grok 3: 128K context, previous generation
const XAI_MODELS: &[(&str, &str, usize)] = &[
    ("grok-4", "Grok 4 (Flagship, 256K)", 262144),
    ("grok-4-0709", "Grok 4 (July 2025)", 262144),
    ("grok-4-latest", "Grok 4 Latest", 262144),
    ("grok-4-1-fast", "Grok 4.1 Fast (2M context)", 2000000),
    ("grok-4-1-fast-reasoning", "Grok 4.1 Fast Reasoning", 2000000),
    ("grok-4-1-fast-non-reasoning", "Grok 4.1 Fast Non-Reasoning", 2000000),
    ("grok-3", "Grok 3", 131072),
    ("grok-3-latest", "Grok 3 Latest", 131072),
    ("grok-3-mini", "Grok 3 Mini", 131072),
    ("grok-3-mini-latest", "Grok 3 Mini Latest", 131072),
    ("grok-2-vision-1212", "Grok 2 Vision", 32768),
    ("grok-code-fast-1", "Grok Code Fast", 131072),
];

// ============================================================================
// XAI Provider
// ============================================================================

/// xAI Grok provider for direct API access.
///
/// This is a thin wrapper around `OpenAICompatibleProvider` that provides:
/// - Automatic `XAI_API_KEY` detection
/// - Default configuration for xAI's API
/// - Model catalog with correct context sizes
///
/// # Why Wrap OpenAICompatibleProvider?
///
/// xAI's API is 100% OpenAI-compatible, so we get:
/// - Battle-tested HTTP client
/// - Streaming support
/// - Tool/function calling
/// - Vision (image input)
/// - JSON mode
/// - Error handling
/// - Retry logic
///
/// Without code duplication!
#[derive(Debug)]
pub struct XAIProvider {
    /// Inner OpenAI-compatible provider
    inner: OpenAICompatibleProvider,
    /// Current model name
    model: String,
}

impl XAIProvider {
    /// Create provider from environment variables.
    ///
    /// # Environment Variables
    ///
    /// - `XAI_API_KEY`: Required API key
    /// - `XAI_MODEL`: Model name (default: grok-4)
    /// - `XAI_BASE_URL`: Custom base URL (default: <https://api.x.ai>)
    ///
    /// # Errors
    ///
    /// Returns error if `XAI_API_KEY` is not set.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("XAI_API_KEY").map_err(|_| {
            LlmError::ConfigError(
                "XAI_API_KEY environment variable not set. \
                 Get your API key from https://console.x.ai"
                    .to_string(),
            )
        })?;

        if api_key.is_empty() {
            return Err(LlmError::ConfigError(
                "XAI_API_KEY is empty. Please set a valid API key.".to_string(),
            ));
        }

        let model = std::env::var("XAI_MODEL").unwrap_or_else(|_| XAI_DEFAULT_MODEL.to_string());
        let base_url =
            std::env::var("XAI_BASE_URL").unwrap_or_else(|_| XAI_BASE_URL.to_string());

        Self::new(api_key, model, Some(base_url))
    }

    /// Create provider with explicit configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - xAI API key
    /// * `model` - Model name (e.g., "grok-4")
    /// * `base_url` - Optional custom base URL
    pub fn new(api_key: String, model: String, base_url: Option<String>) -> Result<Self> {
        // Build ProviderConfig for OpenAICompatibleProvider
        let config = Self::build_config(&api_key, &model, base_url.as_deref());

        // Create inner provider
        let inner = OpenAICompatibleProvider::from_config(config)?;

        debug!(
            provider = XAI_PROVIDER_NAME,
            model = %model,
            "Created xAI provider"
        );

        Ok(Self { inner, model })
    }

    /// Create with a different model.
    ///
    /// Returns a new provider instance configured for the specified model.
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self.inner = self.inner.with_model(model);
        self
    }

    /// Build ProviderConfig for OpenAICompatibleProvider.
    ///
    /// WHY: We need to set XAI_API_KEY env var before creating the provider because
    /// OpenAICompatibleProvider reads the API key from the environment variable
    /// specified in api_key_env, not from a config field.
    fn build_config(_api_key: &str, model: &str, base_url: Option<&str>) -> ProviderConfig {
        // Build model cards from XAI_MODELS with proper capabilities
        let models: Vec<ModelCard> = XAI_MODELS
            .iter()
            .map(|(name, display, context)| {
                // OODA-15: Grok 4 and 4.1 are multimodal (text + image)
                // Vision support: grok-4.*, grok-4.1.*, grok-2-vision
                let supports_vision = name.starts_with("grok-4")
                    || name.starts_with("grok-4.1")
                    || name.contains("vision");

                // OODA-15: Grok 4 is a reasoning model (always reasons)
                let supports_thinking = name.starts_with("grok-4")
                    && !name.contains("non-reasoning");

                ModelCard {
                    name: name.to_string(),
                    display_name: display.to_string(),
                    model_type: ModelType::Llm,
                    capabilities: ModelCapabilities {
                        context_length: *context,
                        supports_function_calling: true,
                        supports_json_mode: true,
                        supports_streaming: true,
                        supports_system_message: true,
                        supports_vision,
                        supports_thinking,
                        ..Default::default()
                    },
                    ..Default::default()
                }
            })
            .collect();

        ProviderConfig {
            name: XAI_PROVIDER_NAME.to_string(),
            display_name: "xAI Grok".to_string(),
            provider_type: ConfigProviderType::OpenAICompatible,
            api_key_env: Some("XAI_API_KEY".to_string()),
            base_url: Some(base_url.unwrap_or(XAI_BASE_URL).to_string()),
            base_url_env: Some("XAI_BASE_URL".to_string()),
            default_llm_model: Some(model.to_string()),
            default_embedding_model: None, // xAI doesn't provide embeddings yet
            models,
            headers: std::collections::HashMap::new(),
            enabled: true,
            ..Default::default()
        }
    }

    /// Get context length for a model.
    pub fn context_length(model: &str) -> usize {
        XAI_MODELS
            .iter()
            .find(|(name, _, _)| *name == model)
            .map(|(_, _, ctx)| *ctx)
            .unwrap_or(262144) // Default to 256K (Grok 4 standard)
    }

    /// List available models.
    pub fn available_models() -> Vec<(&'static str, &'static str, usize)> {
        XAI_MODELS.to_vec()
    }
}

// ============================================================================
// LLMProvider Implementation (delegates to inner OpenAICompatibleProvider)
// ============================================================================

#[async_trait]
impl LLMProvider for XAIProvider {
    fn name(&self) -> &str {
        XAI_PROVIDER_NAME
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_context_length(&self) -> usize {
        Self::context_length(&self.model)
    }

    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        self.inner.complete(prompt).await
    }

    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<LLMResponse> {
        self.inner.complete_with_options(prompt, options).await
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        self.inner.chat(messages, options).await
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[crate::traits::ToolDefinition],
        tool_choice: Option<crate::traits::ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        self.inner.chat_with_tools(messages, tools, tool_choice, options).await
    }

    async fn chat_with_tools_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[crate::traits::ToolDefinition],
        tool_choice: Option<crate::traits::ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        self.inner.chat_with_tools_stream(messages, tools, tool_choice, options).await
    }

    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        self.inner.stream(prompt).await
    }

    fn supports_function_calling(&self) -> bool {
        self.inner.supports_function_calling()
    }

    fn supports_tool_streaming(&self) -> bool {
        self.inner.supports_tool_streaming()
    }
}

// ============================================================================
// EmbeddingProvider Implementation (not supported - xAI doesn't have embeddings API)
// ============================================================================

#[async_trait]
impl EmbeddingProvider for XAIProvider {
    fn name(&self) -> &str {
        XAI_PROVIDER_NAME
    }

    fn model(&self) -> &str {
        "none"
    }

    fn dimension(&self) -> usize {
        0 // Not supported
    }

    fn max_tokens(&self) -> usize {
        0 // Not supported
    }

    async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // xAI doesn't provide embeddings API yet
        Err(LlmError::ConfigError(
            "xAI does not provide an embeddings API. \
             Use OpenAI or another provider for embeddings."
                .to_string(),
        ))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_length_known_model() {
        // OODA-15: Updated context lengths from docs.x.ai
        assert_eq!(XAIProvider::context_length("grok-4"), 262144); // 256K
        assert_eq!(XAIProvider::context_length("grok-4-1-fast"), 2000000); // 2M
        assert_eq!(XAIProvider::context_length("grok-2-vision-1212"), 32768); // 32K
    }

    #[test]
    fn test_context_length_unknown_model() {
        // Unknown models default to 256K (Grok 4 standard)
        assert_eq!(XAIProvider::context_length("grok-unknown"), 262144);
    }

    #[test]
    fn test_available_models() {
        let models = XAIProvider::available_models();
        assert!(!models.is_empty());
        assert!(models.iter().any(|(name, _, _)| *name == "grok-4"));
        assert!(models.iter().any(|(name, _, _)| *name == "grok-4-1-fast"));
    }

    #[test]
    fn test_build_config() {
        let config = XAIProvider::build_config("test-key", "grok-4", None);
        assert_eq!(config.name, "xai");
        assert_eq!(config.base_url, Some("https://api.x.ai/v1".to_string()));
        assert_eq!(config.default_llm_model, Some("grok-4".to_string()));
    }

    #[test]
    fn test_build_config_custom_url() {
        let config = XAIProvider::build_config("test-key", "grok-3", Some("https://custom.api"));
        assert_eq!(config.base_url, Some("https://custom.api".to_string()));
    }
}
