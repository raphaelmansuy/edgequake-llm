//! Mistral AI Provider
//!
//! @implements FEAT-007: Mistral AI provider (chat, embeddings, list-models)
//! Closes https://github.com/raphaelmansuy/edgequake-llm/issues/7
//!
//! # Overview
//!
//! This provider integrates with the Mistral AI API (`api.mistral.ai`) and exposes:
//! - **Chat completions** (sync + SSE streaming, tool/function calling, JSON mode)
//! - **Embeddings** (`mistral-embed`, 1024-dimensional)
//! - **Model listing** (`GET /v1/models`)
//!
//! The chat/tool path is delegated to [`OpenAICompatibleProvider`] because Mistral's
//! `/v1/chat/completions` is fully OpenAI-compatible.  Embeddings are implemented
//! natively because `OpenAICompatibleProvider::embed` is not yet wired up.
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────────────────┐
//! │                      MistralProvider architecture                         │
//! ├───────────────────────────────────────────────────────────────────────────┤
//! │                                                                           │
//! │   User Request                                                            │
//! │        │                                                                  │
//! │        ▼                                                                  │
//! │   ┌────────────────┐    chat/tools    ┌───────────────────────────────┐  │
//! │   │ MistralProvider│ ───────────────► │ OpenAICompatibleProvider      │  │
//! │   │ (wrapper)      │                  │ POST /v1/chat/completions      │  │
//! │   │                │ ◄─────────────── │ (SSE streaming + tool calls)  │  │
//! │   │                │                  └───────────────────────────────┘  │
//! │   │                │    embeddings    ┌───────────────────────────────┐  │
//! │   │                │ ───────────────► │ reqwest POST /v1/embeddings   │  │
//! │   │                │                  │ (native implementation)       │  │
//! │   │                │ ◄─────────────── └───────────────────────────────┘  │
//! │   │                │    list models   ┌───────────────────────────────┐  │
//! │   │                │ ───────────────► │ reqwest GET  /v1/models       │  │
//! │   └────────────────┘ ◄─────────────── └───────────────────────────────┘  │
//! │                                                                           │
//! └───────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Environment Variables
//!
//! | Variable | Required | Default | Description |
//! |----------|----------|---------|-------------|
//! | `MISTRAL_API_KEY` | ✅ Yes | - | API key from console.mistral.ai |
//! | `MISTRAL_MODEL` | ❌ No | `mistral-small-latest` | Default chat model |
//! | `MISTRAL_EMBEDDING_MODEL` | ❌ No | `mistral-embed` | Default embedding model |
//! | `MISTRAL_BASE_URL` | ❌ No | `https://api.mistral.ai/v1` | Endpoint override |
//!
//! # Available Chat Models (2026-02)
//!
//! | Model | Context | Notes |
//! |-------|---------|-------|
//! | `mistral-large-latest` | 128K | Most capable |
//! | `mistral-small-latest` | 32K | Cost-effective |
//! | `codestral-latest` | 256K | Code generation |
//! | `open-mistral-nemo` | 128K | Open weights |
//! | `pixtral-12b-2409` | 128K | Vision capable |
//! | `mistral-embed` | 8K | Embeddings only |
//!
//! # Quick start
//!
//! ```bash
//! export MISTRAL_API_KEY=your-key
//! cargo run --example mistral_chat
//! ```
//!
//! ```rust,no_run
//! use edgequake_llm::{MistralProvider, LLMProvider};
//! use edgequake_llm::traits::ChatMessage;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = MistralProvider::from_env()?;
//! let messages = vec![ChatMessage::user("Hello from Mistral!")];
//! let resp = provider.chat(&messages, None).await?;
//! println!("{}", resp.content);
//! # Ok(())
//! # }
//! ```

use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::debug;

use crate::error::{LlmError, Result};
use crate::model_config::{
    ModelCapabilities, ModelCard, ModelType, ProviderConfig, ProviderType as ConfigProviderType,
};
use crate::providers::openai_compatible::OpenAICompatibleProvider;
use crate::traits::StreamChunk;
use crate::traits::{
    ChatMessage, CompletionOptions, EmbeddingProvider, LLMProvider, LLMResponse, ToolChoice,
    ToolDefinition,
};

// ============================================================================
// Constants
// ============================================================================

/// Default Mistral API base URL (includes /v1 prefix for OpenAI compatibility)
const MISTRAL_BASE_URL: &str = "https://api.mistral.ai/v1";

/// Default chat model
const MISTRAL_DEFAULT_MODEL: &str = "mistral-small-latest";

/// Default embedding model
const MISTRAL_DEFAULT_EMBEDDING_MODEL: &str = "mistral-embed";

/// Embedding dimension for `mistral-embed`
const MISTRAL_EMBED_DIMENSION: usize = 1024;

/// Maximum tokens for `mistral-embed`
const MISTRAL_EMBED_MAX_TOKENS: usize = 8192;

/// Provider display name
const MISTRAL_PROVIDER_NAME: &str = "mistral";

/// Mistral chat model catalog: (id, display_name, context_length, supports_vision, supports_function_calling)
const MISTRAL_CHAT_MODELS: &[(&str, &str, usize, bool, bool)] = &[
    // Premier models
    (
        "mistral-large-latest",
        "Mistral Large (latest)",
        131072,
        false,
        true,
    ),
    (
        "mistral-large-2411",
        "Mistral Large 2411",
        131072,
        false,
        true,
    ),
    (
        "mistral-small-latest",
        "Mistral Small (latest)",
        32768,
        false,
        true,
    ),
    (
        "mistral-small-2501",
        "Mistral Small 2501",
        32768,
        false,
        true,
    ),
    // Code models
    (
        "codestral-latest",
        "Codestral (latest)",
        262144,
        false,
        false,
    ),
    ("codestral-2501", "Codestral 2501", 262144, false, false),
    // Open-weights
    (
        "open-mistral-nemo",
        "Mistral Nemo (open weights)",
        131072,
        false,
        true,
    ),
    (
        "open-mistral-7b",
        "Mistral 7B (open weights)",
        32768,
        false,
        false,
    ),
    (
        "open-mixtral-8x7b",
        "Mixtral 8×7B (open weights)",
        32768,
        false,
        true,
    ),
    (
        "open-mixtral-8x22b",
        "Mixtral 8×22B (open weights)",
        65536,
        false,
        true,
    ),
    // Vision model
    (
        "pixtral-12b-2409",
        "Pixtral 12B (vision)",
        131072,
        true,
        true,
    ),
    (
        "pixtral-large-2411",
        "Pixtral Large (vision)",
        131072,
        true,
        true,
    ),
];

// ============================================================================
// Embedding request/response structs (native implementation)
// ============================================================================

/// Request body for `POST /v1/embeddings`
#[derive(Debug, Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

/// Response from `/v1/embeddings`
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

// ============================================================================
// Model listing structs
// ============================================================================

/// Response from `GET /v1/models`
#[derive(Debug, Deserialize)]
pub struct MistralModelsResponse {
    pub data: Vec<MistralModelInfo>,
}

/// A single model entry from `GET /v1/models`
#[derive(Debug, Deserialize)]
pub struct MistralModelInfo {
    pub id: String,
    #[serde(default)]
    pub created: Option<u64>,
    #[serde(default)]
    pub owned_by: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub max_context_length: Option<usize>,
    #[serde(default)]
    pub capabilities: Option<MistralModelCapabilities>,
}

/// Capabilities returned by the Mistral models endpoint
#[derive(Debug, Deserialize)]
pub struct MistralModelCapabilities {
    #[serde(default)]
    pub completion_chat: bool,
    #[serde(default)]
    pub completion_fim: bool,
    #[serde(default)]
    pub function_calling: bool,
    #[serde(default)]
    pub fine_tuning: bool,
    #[serde(default)]
    pub vision: bool,
}

// ============================================================================
// MistralProvider
// ============================================================================

/// Mistral AI provider for chat completions and embeddings.
///
/// - Chat / streaming / tool-calls → delegates to [`OpenAICompatibleProvider`]
/// - Embeddings → native `reqwest` call to `/v1/embeddings`
/// - Model listing → native `reqwest` call to `GET /v1/models`
#[derive(Debug)]
pub struct MistralProvider {
    /// Inner provider for chat operations (OpenAI-compatible)
    inner: OpenAICompatibleProvider,
    /// Current chat model name
    model: String,
    /// Embedding model name
    embedding_model: String,
    /// Base URL (with `/v1` suffix)
    base_url: String,
    /// API key
    api_key: String,
    /// Shared HTTP client for native requests (embeddings, model listing)
    client: Client,
}

impl MistralProvider {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Create a provider from environment variables.
    ///
    /// Reads:
    /// - `MISTRAL_API_KEY` (required)
    /// - `MISTRAL_MODEL` (optional, default: `mistral-small-latest`)
    /// - `MISTRAL_EMBEDDING_MODEL` (optional, default: `mistral-embed`)
    /// - `MISTRAL_BASE_URL` (optional)
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("MISTRAL_API_KEY").map_err(|_| {
            LlmError::ConfigError(
                "MISTRAL_API_KEY environment variable not set. \
                 Get your API key from https://console.mistral.ai"
                    .to_string(),
            )
        })?;

        if api_key.is_empty() {
            return Err(LlmError::ConfigError(
                "MISTRAL_API_KEY is empty. Please set a valid API key.".to_string(),
            ));
        }

        let model =
            std::env::var("MISTRAL_MODEL").unwrap_or_else(|_| MISTRAL_DEFAULT_MODEL.to_string());
        let embedding_model = std::env::var("MISTRAL_EMBEDDING_MODEL")
            .unwrap_or_else(|_| MISTRAL_DEFAULT_EMBEDDING_MODEL.to_string());
        let base_url =
            std::env::var("MISTRAL_BASE_URL").unwrap_or_else(|_| MISTRAL_BASE_URL.to_string());

        Self::new(api_key, model, embedding_model, Some(base_url))
    }

    /// Create a provider from a [`ProviderConfig`].
    pub fn from_config(config: &ProviderConfig) -> Result<Self> {
        let api_key = if let Some(env_var) = &config.api_key_env {
            std::env::var(env_var).map_err(|_| {
                LlmError::ConfigError(format!(
                    "API key environment variable '{}' not set for Mistral provider.",
                    env_var
                ))
            })?
        } else {
            return Err(LlmError::ConfigError(
                "Mistral provider requires api_key_env to be set.".to_string(),
            ));
        };

        let model = config
            .default_llm_model
            .clone()
            .unwrap_or_else(|| MISTRAL_DEFAULT_MODEL.to_string());
        let embedding_model = config
            .default_embedding_model
            .clone()
            .unwrap_or_else(|| MISTRAL_DEFAULT_EMBEDDING_MODEL.to_string());
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| MISTRAL_BASE_URL.to_string());

        Self::new(api_key, model, embedding_model, Some(base_url))
    }

    /// Create a provider with explicit configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Mistral API key
    /// * `model` - Chat model (e.g., `"mistral-small-latest"`)
    /// * `embedding_model` - Embedding model (e.g., `"mistral-embed"`)
    /// * `base_url` - Optional custom base URL
    pub fn new(
        api_key: String,
        model: String,
        embedding_model: String,
        base_url: Option<String>,
    ) -> Result<Self> {
        let base_url = base_url.unwrap_or_else(|| MISTRAL_BASE_URL.to_string());

        // Build ProviderConfig for the inner OpenAICompatibleProvider
        let config = Self::build_provider_config(&api_key, &model, &embedding_model, &base_url);

        // Set env var so OpenAICompatibleProvider can read it
        // (it reads from `config.api_key_env`)
        std::env::set_var("MISTRAL_API_KEY", &api_key);

        let inner = OpenAICompatibleProvider::from_config(config)?;

        // Build a plain reqwest client for native requests
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|e| LlmError::ConfigError(format!("Failed to build HTTP client: {}", e)))?;

        debug!(
            provider = MISTRAL_PROVIDER_NAME,
            model = %model,
            base_url = %base_url,
            "Created Mistral provider"
        );

        Ok(Self {
            inner,
            model,
            embedding_model,
            base_url,
            api_key,
            client,
        })
    }

    // -----------------------------------------------------------------------
    // Builder methods
    // -----------------------------------------------------------------------

    /// Return a new provider configured for a different chat model.
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self.inner = self.inner.with_model(model);
        self
    }

    /// Return a new provider configured for a different embedding model.
    pub fn with_embedding_model(mut self, model: &str) -> Self {
        self.embedding_model = model.to_string();
        self
    }

    // -----------------------------------------------------------------------
    // Model catalog helpers
    // -----------------------------------------------------------------------

    /// Context length for a given model ID.
    ///
    /// Falls back to 32 768 if the model is not in the static catalog.
    pub fn context_length(model: &str) -> usize {
        MISTRAL_CHAT_MODELS
            .iter()
            .find(|(id, _, _, _, _)| *id == model)
            .map(|(_, _, ctx, _, _)| *ctx)
            .unwrap_or(32768)
    }

    /// List the statically-known chat models.
    pub fn available_models() -> Vec<(&'static str, &'static str, usize)> {
        MISTRAL_CHAT_MODELS
            .iter()
            .map(|(id, name, ctx, _, _)| (*id, *name, *ctx))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Model listing (live API)
    // -----------------------------------------------------------------------

    /// Fetch available models from the Mistral API.
    ///
    /// Calls `GET {base_url}/models` and returns the raw response.
    pub async fn list_models(&self) -> Result<MistralModelsResponse> {
        let url = format!("{}/models", self.base_url.trim_end_matches('/'));

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(format!("Failed to list Mistral models: {}", e)))?;

        let status = response.status();
        let body = response.text().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to read model list response: {}", e))
        })?;

        if !status.is_success() {
            return Err(LlmError::ApiError(format!(
                "Mistral models list failed ({status}): {body}"
            )));
        }

        serde_json::from_str(&body)
            .map_err(|e| LlmError::ApiError(format!("Failed to parse models response: {e}")))
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Construct the [`ProviderConfig`] used by the inner [`OpenAICompatibleProvider`].
    fn build_provider_config(
        _api_key: &str,
        model: &str,
        embedding_model: &str,
        base_url: &str,
    ) -> ProviderConfig {
        let models: Vec<ModelCard> = MISTRAL_CHAT_MODELS
            .iter()
            .map(|(id, display, ctx, vision, fc)| ModelCard {
                name: id.to_string(),
                display_name: display.to_string(),
                model_type: ModelType::Llm,
                capabilities: ModelCapabilities {
                    context_length: *ctx,
                    max_output_tokens: 4096,
                    supports_vision: *vision,
                    supports_function_calling: *fc,
                    supports_json_mode: true,
                    supports_streaming: true,
                    supports_system_message: true,
                    ..Default::default()
                },
                ..Default::default()
            })
            .chain(std::iter::once(ModelCard {
                name: MISTRAL_DEFAULT_EMBEDDING_MODEL.to_string(),
                display_name: "Mistral Embed".to_string(),
                model_type: ModelType::Embedding,
                capabilities: ModelCapabilities {
                    context_length: MISTRAL_EMBED_MAX_TOKENS,
                    embedding_dimension: MISTRAL_EMBED_DIMENSION,
                    ..Default::default()
                },
                ..Default::default()
            }))
            .collect();

        ProviderConfig {
            name: MISTRAL_PROVIDER_NAME.to_string(),
            display_name: "Mistral AI".to_string(),
            provider_type: ConfigProviderType::OpenAICompatible,
            api_key_env: Some("MISTRAL_API_KEY".to_string()),
            base_url: Some(base_url.to_string()),
            base_url_env: Some("MISTRAL_BASE_URL".to_string()),
            default_llm_model: Some(model.to_string()),
            default_embedding_model: Some(embedding_model.to_string()),
            models,
            enabled: true,
            ..Default::default()
        }
    }
}

// ============================================================================
// LLMProvider trait implementation (delegates to inner OpenAICompatibleProvider)
// ============================================================================

#[async_trait]
impl LLMProvider for MistralProvider {
    fn name(&self) -> &str {
        MISTRAL_PROVIDER_NAME
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
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        self.inner
            .chat_with_tools(messages, tools, tool_choice, options)
            .await
    }

    async fn chat_with_tools_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        self.inner
            .chat_with_tools_stream(messages, tools, tool_choice, options)
            .await
    }

    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        self.inner.stream(prompt).await
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_function_calling(&self) -> bool {
        // All "latest" models and nemo support function calling
        let fc_capable = MISTRAL_CHAT_MODELS
            .iter()
            .find(|(id, _, _, _, _)| *id == self.model.as_str())
            .map(|(_, _, _, _, fc)| *fc)
            .unwrap_or(true);
        fc_capable
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    fn supports_tool_streaming(&self) -> bool {
        self.inner.supports_tool_streaming()
    }
}

// ============================================================================
// EmbeddingProvider trait implementation (native reqwest)
// ============================================================================

#[async_trait]
impl EmbeddingProvider for MistralProvider {
    fn name(&self) -> &str {
        MISTRAL_PROVIDER_NAME
    }

    fn model(&self) -> &str {
        &self.embedding_model
    }

    fn dimension(&self) -> usize {
        // Only `mistral-embed` is currently offered; if the user configured a
        // custom embedding model we fall back to the standard dimension.
        if self.embedding_model == MISTRAL_DEFAULT_EMBEDDING_MODEL {
            MISTRAL_EMBED_DIMENSION
        } else {
            MISTRAL_EMBED_DIMENSION // conservative default
        }
    }

    fn max_tokens(&self) -> usize {
        MISTRAL_EMBED_MAX_TOKENS
    }

    /// Embed a batch of texts using `POST /v1/embeddings`.
    ///
    /// The returned vectors are in the same order as the input slice.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/embeddings", self.base_url.trim_end_matches('/'));

        let request_body = EmbeddingRequest {
            model: &self.embedding_model,
            input: texts,
        };

        debug!(
            model = self.embedding_model,
            count = texts.len(),
            "Mistral embed request"
        );

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                LlmError::NetworkError(format!("Mistral embeddings request failed: {}", e))
            })?;

        let status = response.status();
        let body = response.text().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to read Mistral embeddings response: {}", e))
        })?;

        if !status.is_success() {
            return Err(LlmError::ApiError(format!(
                "Mistral embeddings API error ({status}): {body}"
            )));
        }

        let embedding_response: EmbeddingResponse = serde_json::from_str(&body).map_err(|e| {
            LlmError::ApiError(format!(
                "Failed to parse Mistral embeddings response: {e} | body: {}",
                &body[..body.len().min(500)]
            ))
        })?;

        // Sort by index to preserve input order and extract vectors
        let mut data = embedding_response.data;
        data.sort_by_key(|d| d.index);

        let embeddings: Vec<Vec<f32>> = data.into_iter().map(|d| d.embedding).collect();

        // Validate that we got the right number of embeddings
        if embeddings.len() != texts.len() {
            return Err(LlmError::ApiError(format!(
                "Mistral returned {} embeddings for {} inputs",
                embeddings.len(),
                texts.len()
            )));
        }

        Ok(embeddings)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Static catalog tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_available_models_not_empty() {
        let models = MistralProvider::available_models();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_available_models_contains_expected_ids() {
        let models = MistralProvider::available_models();
        let ids: Vec<&str> = models.iter().map(|(id, _, _)| *id).collect();
        assert!(
            ids.contains(&"mistral-large-latest"),
            "Should contain mistral-large-latest"
        );
        assert!(
            ids.contains(&"mistral-small-latest"),
            "Should contain mistral-small-latest"
        );
        assert!(
            ids.contains(&"codestral-latest"),
            "Should contain codestral-latest"
        );
        assert!(
            ids.contains(&"pixtral-12b-2409"),
            "Should contain pixtral-12b-2409"
        );
    }

    #[test]
    fn test_context_length_known_models() {
        assert_eq!(
            MistralProvider::context_length("mistral-large-latest"),
            131072
        );
        assert_eq!(
            MistralProvider::context_length("mistral-small-latest"),
            32768
        );
        assert_eq!(MistralProvider::context_length("codestral-latest"), 262144);
        assert_eq!(MistralProvider::context_length("open-mistral-nemo"), 131072);
    }

    #[test]
    fn test_context_length_unknown_model() {
        // Unknown models should fall back to 32 768
        assert_eq!(MistralProvider::context_length("unknown-model"), 32768);
    }

    #[test]
    fn test_all_models_have_positive_context() {
        for (id, _, ctx) in MistralProvider::available_models() {
            assert!(ctx > 0, "Model {} must have positive context length", id);
        }
    }

    // -----------------------------------------------------------------------
    // Constants tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_provider_name_constant() {
        assert_eq!(MISTRAL_PROVIDER_NAME, "mistral");
    }

    #[test]
    fn test_default_model_constant() {
        assert_eq!(MISTRAL_DEFAULT_MODEL, "mistral-small-latest");
    }

    #[test]
    fn test_default_embedding_model_constant() {
        assert_eq!(MISTRAL_DEFAULT_EMBEDDING_MODEL, "mistral-embed");
    }

    #[test]
    fn test_embed_dimension_constant() {
        assert_eq!(MISTRAL_EMBED_DIMENSION, 1024);
    }

    #[test]
    fn test_base_url_constant() {
        assert_eq!(MISTRAL_BASE_URL, "https://api.mistral.ai/v1");
    }

    // -----------------------------------------------------------------------
    // ProviderConfig construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_provider_config_name() {
        let cfg = MistralProvider::build_provider_config(
            "key",
            "mistral-small-latest",
            "mistral-embed",
            MISTRAL_BASE_URL,
        );
        assert_eq!(cfg.name, "mistral");
        assert_eq!(cfg.display_name, "Mistral AI");
    }

    #[test]
    fn test_build_provider_config_base_url() {
        let cfg = MistralProvider::build_provider_config(
            "key",
            "mistral-small-latest",
            "mistral-embed",
            "https://custom.api/v1",
        );
        assert_eq!(cfg.base_url, Some("https://custom.api/v1".to_string()));
    }

    #[test]
    fn test_build_provider_config_models_include_embedding() {
        let cfg = MistralProvider::build_provider_config(
            "key",
            "mistral-small-latest",
            "mistral-embed",
            MISTRAL_BASE_URL,
        );
        let embedding_cards: Vec<_> = cfg
            .models
            .iter()
            .filter(|m| m.model_type == ModelType::Embedding)
            .collect();
        assert_eq!(embedding_cards.len(), 1);
        assert_eq!(embedding_cards[0].name, "mistral-embed");
        assert_eq!(
            embedding_cards[0].capabilities.embedding_dimension,
            MISTRAL_EMBED_DIMENSION
        );
    }

    #[test]
    fn test_build_provider_config_default_models() {
        let cfg = MistralProvider::build_provider_config(
            "key",
            "mistral-small-latest",
            "mistral-embed",
            MISTRAL_BASE_URL,
        );
        assert_eq!(
            cfg.default_llm_model,
            Some("mistral-small-latest".to_string())
        );
        assert_eq!(
            cfg.default_embedding_model,
            Some("mistral-embed".to_string())
        );
    }

    #[test]
    fn test_build_provider_config_api_key_env() {
        let cfg = MistralProvider::build_provider_config(
            "key",
            "mistral-small-latest",
            "mistral-embed",
            MISTRAL_BASE_URL,
        );
        assert_eq!(cfg.api_key_env, Some("MISTRAL_API_KEY".to_string()));
    }

    // -----------------------------------------------------------------------
    // from_env failures
    // -----------------------------------------------------------------------

    #[test]
    fn test_from_env_missing_api_key() {
        std::env::remove_var("MISTRAL_API_KEY");
        let result = MistralProvider::from_env();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("MISTRAL_API_KEY"));
    }

    // -----------------------------------------------------------------------
    // EmbeddingProvider trait surface (no live calls)
    // -----------------------------------------------------------------------

    #[test]
    fn test_embedding_dimension() {
        std::env::set_var("MISTRAL_API_KEY", "test-key-for-unit-test");
        let p = MistralProvider::new(
            "test-key".to_string(),
            MISTRAL_DEFAULT_MODEL.to_string(),
            MISTRAL_DEFAULT_EMBEDDING_MODEL.to_string(),
            None,
        )
        .unwrap();
        assert_eq!(EmbeddingProvider::dimension(&p), MISTRAL_EMBED_DIMENSION);
        assert_eq!(EmbeddingProvider::max_tokens(&p), MISTRAL_EMBED_MAX_TOKENS);
        assert_eq!(EmbeddingProvider::name(&p), "mistral");
        assert_eq!(EmbeddingProvider::model(&p), "mistral-embed");
        std::env::remove_var("MISTRAL_API_KEY");
    }

    // -----------------------------------------------------------------------
    // LLMProvider trait surface (no live calls)
    // -----------------------------------------------------------------------

    #[test]
    fn test_llm_provider_surface() {
        std::env::set_var("MISTRAL_API_KEY", "test-key-for-unit-test");
        let p = MistralProvider::new(
            "test-key".to_string(),
            "mistral-large-latest".to_string(),
            MISTRAL_DEFAULT_EMBEDDING_MODEL.to_string(),
            None,
        )
        .unwrap();
        assert_eq!(LLMProvider::name(&p), "mistral");
        assert_eq!(LLMProvider::model(&p), "mistral-large-latest");
        assert_eq!(p.max_context_length(), 131072);
        assert!(p.supports_streaming());
        assert!(p.supports_function_calling());
        assert!(p.supports_json_mode());
        std::env::remove_var("MISTRAL_API_KEY");
    }

    #[test]
    fn test_with_model_builder() {
        std::env::set_var("MISTRAL_API_KEY", "test-key-for-unit-test");
        let p = MistralProvider::new(
            "test-key".to_string(),
            MISTRAL_DEFAULT_MODEL.to_string(),
            MISTRAL_DEFAULT_EMBEDDING_MODEL.to_string(),
            None,
        )
        .unwrap()
        .with_model("codestral-latest");
        assert_eq!(p.model, "codestral-latest");
        assert_eq!(p.max_context_length(), 262144);
        std::env::remove_var("MISTRAL_API_KEY");
    }

    #[test]
    fn test_with_embedding_model_builder() {
        std::env::set_var("MISTRAL_API_KEY", "test-key-for-unit-test");
        let p = MistralProvider::new(
            "test-key".to_string(),
            MISTRAL_DEFAULT_MODEL.to_string(),
            MISTRAL_DEFAULT_EMBEDDING_MODEL.to_string(),
            None,
        )
        .unwrap()
        .with_embedding_model("custom-embed");
        assert_eq!(p.embedding_model, "custom-embed");
        std::env::remove_var("MISTRAL_API_KEY");
    }

    // -----------------------------------------------------------------------
    // Serialization / deserialization helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_embedding_request_serialization() {
        let texts = vec!["hello world".to_string(), "foo bar".to_string()];
        let req = EmbeddingRequest {
            model: "mistral-embed",
            input: &texts,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "mistral-embed");
        assert_eq!(json["input"][0], "hello world");
        assert_eq!(json["input"][1], "foo bar");
    }

    #[test]
    fn test_embedding_response_deserialization() {
        let raw = r#"{
            "id": "embd-1",
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
                {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1}
            ],
            "model": "mistral-embed",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        }"#;
        let resp: EmbeddingResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].embedding, vec![0.1, 0.2, 0.3]);
        assert_eq!(resp.data[1].index, 1);
    }

    #[test]
    fn test_model_info_deserialization() {
        let raw = r#"{
            "data": [
                {
                    "id": "mistral-small-latest",
                    "created": 1735689600,
                    "owned_by": "mistralai",
                    "description": "Mistral Small",
                    "max_context_length": 32768,
                    "capabilities": {
                        "completion_chat": true,
                        "completion_fim": false,
                        "function_calling": true,
                        "fine_tuning": false,
                        "vision": false
                    }
                }
            ]
        }"#;
        let resp: MistralModelsResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(resp.data.len(), 1);
        let m = &resp.data[0];
        assert_eq!(m.id, "mistral-small-latest");
        assert_eq!(m.max_context_length, Some(32768));
        let caps = m.capabilities.as_ref().unwrap();
        assert!(caps.function_calling);
        assert!(!caps.vision);
    }

    #[test]
    fn test_vision_model_capabilities() {
        let cfg = MistralProvider::build_provider_config(
            "key",
            "pixtral-12b-2409",
            "mistral-embed",
            MISTRAL_BASE_URL,
        );
        let pixtral = cfg
            .models
            .iter()
            .find(|m| m.name == "pixtral-12b-2409")
            .unwrap();
        assert!(pixtral.capabilities.supports_vision);
        assert!(pixtral.capabilities.supports_function_calling);
    }

    #[test]
    fn test_codestral_no_function_calling_in_catalog() {
        // codestral-latest is listed as fc=false in our catalog
        let cfg = MistralProvider::build_provider_config(
            "key",
            "codestral-latest",
            "mistral-embed",
            MISTRAL_BASE_URL,
        );
        let codestral = cfg
            .models
            .iter()
            .find(|m| m.name == "codestral-latest")
            .unwrap();
        // codestral is a code completion model, not a chat model with FC
        assert!(!codestral.capabilities.supports_function_calling);
    }
}
