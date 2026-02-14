//! HuggingFace Hub Provider - Access to open-source models via HuggingFace.
//!
//! @implements OODA-80: HuggingFace Hub Integration
//!
//! # Overview
//!
//! This provider connects to HuggingFace's Inference API for access to
//! open-source models like Llama, Mistral, Qwen, and Gemma. HuggingFace's API
//! is OpenAI-compatible, so we leverage `OpenAICompatibleProvider` internally.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                  HuggingFace Provider Architecture                       │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │   User Request                                                           │
//! │        │                                                                 │
//! │        ▼                                                                 │
//! │   ┌──────────────────┐  ┌──────────────────────┐  ┌──────────────────┐ │
//! │   │ HuggingFaceProvider│─►│ OpenAICompatibleProvider│─►│ api-inference  │ │
//! │   │ (wrapper)          │  │ (implementation)        │  │ .huggingface   │ │
//! │   └──────────────────┘  └──────────────────────┘  │ .co/v1          │ │
//! │                                                     └──────────────────┘ │
//! │                                                                          │
//! │   HuggingFaceProvider provides:                                          │
//! │   - HF_TOKEN environment detection                                       │
//! │   - Dynamic base URL per model                                           │
//! │   - Default model: meta-llama/Meta-Llama-3.1-70B-Instruct               │
//! │   - Model catalog with context sizes                                    │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Environment Variables
//!
//! | Variable | Required | Default | Description |
//! |----------|----------|---------|-------------|
//! | `HF_TOKEN` | ✅ Yes | - | HuggingFace access token |
//! | `HUGGINGFACE_TOKEN` | ❌ Alt | - | Alternative token variable |
//! | `HF_MODEL` | ❌ No | `meta-llama/Meta-Llama-3.1-70B-Instruct` | Default model |
//! | `HF_BASE_URL` | ❌ No | Auto | Custom API endpoint |
//!
//! # Available Models
//!
//! | Model | Context | Description |
//! |-------|---------|-------------|
//! | `meta-llama/Meta-Llama-3.1-70B-Instruct` | 128K | Llama 3.1 70B |
//! | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 128K | Llama 3.1 8B |
//! | `mistralai/Mistral-7B-Instruct-v0.3` | 32K | Mistral 7B |
//! | `Qwen/Qwen2.5-72B-Instruct` | 128K | Qwen 2.5 72B |
//! | `microsoft/Phi-3-medium-4k-instruct` | 4K | Phi-3 Medium |
//! | `google/gemma-7b-it` | 8K | Gemma 7B IT |
//!
//! # Example
//!
//! ```bash
//! # Set HuggingFace token (from https://huggingface.co/settings/tokens)
//! export HF_TOKEN=hf_xxxxxxxxxx
//!
//! # Use with EdgeCode (auto-detected)
//! edgecode react "Write hello world in Rust"
//!
//! # Explicit provider selection
//! edgecode react --provider huggingface "Write hello world in Rust"
//!
//! # Use specific model
//! export HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3
//! edgecode react "Explain quantum computing"
//! ```
//!
//! # Free Tier
//!
//! HuggingFace offers a free tier with limited requests. For higher throughput,
//! consider a Pro subscription or Inference Endpoints for dedicated infrastructure.

use async_trait::async_trait;
use futures::stream::BoxStream;
use tracing::debug;

use crate::error::{LlmError, Result};
use crate::model_config::{
    ModelCapabilities, ModelCard, ModelType, ProviderConfig, ProviderType as ConfigProviderType,
};
use crate::providers::openai_compatible::OpenAICompatibleProvider;
use crate::traits::{ChatMessage, CompletionOptions, EmbeddingProvider, LLMProvider, LLMResponse, StreamChunk};

// ============================================================================
// Constants
// ============================================================================

/// HuggingFace Inference API base URL template
///
/// WHY: HuggingFace uses per-model URLs for the serverless inference API
/// Format: https://api-inference.huggingface.co/models/{model_id}/v1
#[allow(dead_code)]
const HF_BASE_URL_TEMPLATE: &str = "https://api-inference.huggingface.co/models";

/// Alternative: Router-based URL for Inference Providers
/// This provides load balancing and provider selection
const HF_ROUTER_URL: &str = "https://router.huggingface.co/hf-inference/v1";

/// Default model - Llama 3.1 70B is a strong general-purpose model
const HF_DEFAULT_MODEL: &str = "meta-llama/Meta-Llama-3.1-70B-Instruct";

/// Provider display name
const HF_PROVIDER_NAME: &str = "huggingface";

/// HuggingFace model catalog with context lengths.
///
/// WHY: Pre-defined models ensure users get correct context limits without
/// having to check documentation. Updated from HuggingFace Hub.
const HF_MODELS: &[(&str, &str, usize)] = &[
    // Meta Llama models
    (
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "Llama 3.1 70B Instruct",
        128000,
    ),
    (
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Llama 3.1 8B Instruct",
        128000,
    ),
    (
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "Llama 3 8B Instruct",
        8192,
    ),
    (
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "Llama 3 70B Instruct",
        8192,
    ),
    // Mistral models
    (
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Mistral 7B Instruct v0.3",
        32000,
    ),
    (
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Mixtral 8x7B Instruct",
        32000,
    ),
    // Qwen models
    ("Qwen/Qwen2.5-72B-Instruct", "Qwen 2.5 72B Instruct", 128000),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen 2.5 7B Instruct", 128000),
    ("Qwen/Qwen2.5-Coder-32B-Instruct", "Qwen 2.5 Coder 32B", 128000),
    // Microsoft Phi models
    (
        "microsoft/Phi-3-medium-4k-instruct",
        "Phi-3 Medium 4K",
        4096,
    ),
    (
        "microsoft/Phi-3-mini-4k-instruct",
        "Phi-3 Mini 4K",
        4096,
    ),
    // Google Gemma models
    ("google/gemma-7b-it", "Gemma 7B IT", 8192),
    ("google/gemma-2b-it", "Gemma 2B IT", 8192),
    // DeepSeek models
    (
        "deepseek-ai/DeepSeek-Coder-V2-Instruct",
        "DeepSeek Coder V2",
        128000,
    ),
];

// ============================================================================
// HuggingFaceProvider
// ============================================================================

/// HuggingFace Hub provider for open-source model access.
///
/// This is a thin wrapper around `OpenAICompatibleProvider` that provides:
/// - Automatic `HF_TOKEN` detection
/// - Default configuration for HuggingFace's API
/// - Model catalog with correct context sizes
///
/// # Why Wrap OpenAICompatibleProvider?
///
/// HuggingFace's Inference API is OpenAI-compatible, so we get:
/// - Battle-tested HTTP client
/// - Streaming support
/// - Tool/function calling
/// - JSON mode
/// - Error handling
/// - Retry logic
///
/// Without code duplication!
#[derive(Debug)]
pub struct HuggingFaceProvider {
    /// Inner OpenAI-compatible provider
    inner: OpenAICompatibleProvider,
    /// Current model name
    model: String,
}

impl HuggingFaceProvider {
    /// Create provider from environment variables.
    ///
    /// # Environment Variables
    ///
    /// - `HF_TOKEN` or `HUGGINGFACE_TOKEN`: Required API token
    /// - `HF_MODEL`: Model name (default: meta-llama/Meta-Llama-3.1-70B-Instruct)
    /// - `HF_BASE_URL`: Custom base URL (default: auto-generated per model)
    ///
    /// # Errors
    ///
    /// Returns error if neither `HF_TOKEN` nor `HUGGINGFACE_TOKEN` is set.
    pub fn from_env() -> Result<Self> {
        // Check both common token variable names
        let api_key = std::env::var("HF_TOKEN")
            .or_else(|_| std::env::var("HUGGINGFACE_TOKEN"))
            .map_err(|_| {
                LlmError::ConfigError(
                    "HF_TOKEN or HUGGINGFACE_TOKEN environment variable not set. \
                     Get your token from https://huggingface.co/settings/tokens"
                        .to_string(),
                )
            })?;

        if api_key.is_empty() {
            return Err(LlmError::ConfigError(
                "HF_TOKEN is empty. Please set a valid token.".to_string(),
            ));
        }

        let model =
            std::env::var("HF_MODEL").unwrap_or_else(|_| HF_DEFAULT_MODEL.to_string());
        let base_url = std::env::var("HF_BASE_URL").ok();

        Self::new(api_key, model, base_url)
    }

    /// Create provider with explicit configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - HuggingFace access token
    /// * `model` - Model name (e.g., "meta-llama/Meta-Llama-3.1-70B-Instruct")
    /// * `base_url` - Optional custom base URL
    pub fn new(api_key: String, model: String, base_url: Option<String>) -> Result<Self> {
        // Set HF_TOKEN env var for OpenAICompatibleProvider to read
        std::env::set_var("HF_TOKEN", &api_key);

        // Build ProviderConfig
        let config = Self::build_config(&model, base_url.as_deref());

        // Create inner provider
        let inner = OpenAICompatibleProvider::from_config(config)?;

        debug!(
            provider = HF_PROVIDER_NAME,
            model = %model,
            "Created HuggingFace provider"
        );

        Ok(Self { inner, model })
    }

    /// Create with a different model.
    ///
    /// Returns a new provider instance configured for the specified model.
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        // Update inner provider with new model
        // Note: Base URL stays the same (router URL handles routing)
        self.inner = self.inner.with_model(model);
        self
    }

    /// Generate the API URL for a specific model.
    ///
    /// WHY: HuggingFace uses per-model URLs for serverless inference.
    /// We use the router URL for simpler implementation and load balancing.
    #[allow(dead_code)]
    fn model_url(_model: &str) -> String {
        // Use router URL which handles model routing automatically
        // This is simpler than constructing per-model URLs
        HF_ROUTER_URL.to_string()
    }

    /// Build ProviderConfig for OpenAICompatibleProvider.
    fn build_config(model: &str, base_url: Option<&str>) -> ProviderConfig {
        // Build model cards from HF_MODELS with proper capabilities
        let models: Vec<ModelCard> = HF_MODELS
            .iter()
            .map(|(name, display, context)| ModelCard {
                name: name.to_string(),
                display_name: display.to_string(),
                model_type: ModelType::Llm,
                capabilities: ModelCapabilities {
                    context_length: *context,
                    supports_function_calling: true, // Most HF models support this
                    supports_json_mode: true,
                    supports_streaming: true,
                    supports_system_message: true,
                    supports_vision: false, // Set true for vision models if needed
                    ..Default::default()
                },
                ..Default::default()
            })
            .collect();

        // Use provided base_url or default to router URL
        let effective_base_url = base_url
            .map(|s| s.to_string())
            .unwrap_or_else(|| HF_ROUTER_URL.to_string());

        ProviderConfig {
            name: HF_PROVIDER_NAME.to_string(),
            display_name: "HuggingFace Hub".to_string(),
            provider_type: ConfigProviderType::OpenAICompatible,
            api_key_env: Some("HF_TOKEN".to_string()),
            base_url: Some(effective_base_url),
            base_url_env: Some("HF_BASE_URL".to_string()),
            default_llm_model: Some(model.to_string()),
            default_embedding_model: None,
            models,
            headers: std::collections::HashMap::new(),
            enabled: true,
            ..Default::default()
        }
    }

    /// Get context length for a model.
    pub fn context_length(model: &str) -> usize {
        HF_MODELS
            .iter()
            .find(|(name, _, _)| *name == model)
            .map(|(_, _, ctx)| *ctx)
            .unwrap_or(8192) // Conservative default
    }

    /// List available models.
    pub fn available_models() -> Vec<(&'static str, &'static str, usize)> {
        HF_MODELS.to_vec()
    }

    /// Check if a token looks like a HuggingFace token.
    ///
    /// HuggingFace tokens start with "hf_" prefix.
    pub fn is_hf_token(token: &str) -> bool {
        token.starts_with("hf_")
    }
}

// ============================================================================
// LLMProvider Implementation (delegates to inner OpenAICompatibleProvider)
// ============================================================================

#[async_trait]
impl LLMProvider for HuggingFaceProvider {
    fn name(&self) -> &str {
        HF_PROVIDER_NAME
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
        self.inner
            .chat_with_tools(messages, tools, tool_choice, options)
            .await
    }

    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        self.inner.stream(prompt).await
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_function_calling(&self) -> bool {
        self.inner.supports_function_calling()
    }

    fn supports_tool_streaming(&self) -> bool {
        self.inner.supports_tool_streaming()
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
}

// ============================================================================
// EmbeddingProvider Implementation
// ============================================================================

#[async_trait]
impl EmbeddingProvider for HuggingFaceProvider {
    fn name(&self) -> &str {
        HF_PROVIDER_NAME
    }

    fn model(&self) -> &str {
        "none"
    }

    fn dimension(&self) -> usize {
        0 // Embedding support would require a separate implementation
    }

    fn max_tokens(&self) -> usize {
        0
    }

    async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Embeddings require a different API endpoint
        // Could be added in future with a dedicated embedding model
        Err(LlmError::ConfigError(
            "HuggingFace embeddings require a separate provider configuration. \
             Use the HuggingFace Inference API directly for embeddings."
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
        assert_eq!(
            HuggingFaceProvider::context_length("meta-llama/Meta-Llama-3.1-70B-Instruct"),
            128000
        );
        assert_eq!(
            HuggingFaceProvider::context_length("mistralai/Mistral-7B-Instruct-v0.3"),
            32000
        );
        assert_eq!(
            HuggingFaceProvider::context_length("microsoft/Phi-3-medium-4k-instruct"),
            4096
        );
    }

    #[test]
    fn test_context_length_unknown_model() {
        // Unknown models default to 8K
        assert_eq!(
            HuggingFaceProvider::context_length("unknown/model"),
            8192
        );
    }

    #[test]
    fn test_available_models() {
        let models = HuggingFaceProvider::available_models();
        assert!(!models.is_empty());
        assert!(models
            .iter()
            .any(|(name, _, _)| *name == "meta-llama/Meta-Llama-3.1-70B-Instruct"));
        assert!(models
            .iter()
            .any(|(name, _, _)| *name == "Qwen/Qwen2.5-72B-Instruct"));
    }

    #[test]
    fn test_build_config() {
        let config = HuggingFaceProvider::build_config(
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            None,
        );
        assert_eq!(config.name, "huggingface");
        assert_eq!(
            config.base_url,
            Some("https://router.huggingface.co/hf-inference/v1".to_string())
        );
        assert_eq!(
            config.default_llm_model,
            Some("meta-llama/Meta-Llama-3.1-70B-Instruct".to_string())
        );
    }

    #[test]
    fn test_build_config_custom_url() {
        let config = HuggingFaceProvider::build_config(
            "mistralai/Mistral-7B-Instruct-v0.3",
            Some("https://custom.api"),
        );
        assert_eq!(config.base_url, Some("https://custom.api".to_string()));
    }

    #[test]
    fn test_is_hf_token() {
        assert!(HuggingFaceProvider::is_hf_token("hf_xxxxx"));
        assert!(HuggingFaceProvider::is_hf_token("hf_abc123"));
        assert!(!HuggingFaceProvider::is_hf_token("sk-xxxxx"));
        assert!(!HuggingFaceProvider::is_hf_token("xxxxx"));
    }

    #[test]
    fn test_model_url() {
        let url = HuggingFaceProvider::model_url("meta-llama/Meta-Llama-3.1-70B-Instruct");
        assert_eq!(url, "https://router.huggingface.co/hf-inference/v1");
    }
}
