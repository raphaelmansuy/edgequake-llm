//! LM Studio provider implementation.
//!
//! This module provides integration with LM Studio's local OpenAI-compatible API.
//! LM Studio runs local models and exposes them via an OpenAI-compatible HTTP API,
//! so this provider is a **thin wrapper** over [`OpenAICompatibleProvider`].
//!
//! ## Architecture (DRY / SOLID)
//!
//! ```text
//!  LMStudioProvider
//!  ├── inner: OpenAICompatibleProvider  ← handles all standard HTTP / JSON logic
//!  ├── rest_client: Client              ← for native /api/v1/chat (reasoning models)
//!  ├── host: String                     ← raw host URL (without /v1)
//!  └── auto_load_models: bool           ← trigger `lms load` on model-not-found
//! ```
//!
//! **Standard operations** (`chat`, `chat_with_tools`, `stream`, `embed`) are
//! fully delegated to `inner`.  Only LM Studio-specific extensions live here:
//!
//! 1. **Reasoning models** (`deepseek-r1`, `qwen3-*`, …): routed to the native
//!    `/api/v1/chat` REST endpoint which exposes explicit `reasoning` and `message`
//!    output items with per-phase token counts.
//! 2. **Auto-load**: when the provider returns a *model-not-loaded* error and
//!    `auto_load_models = true`, the `lms` CLI is invoked and the request is retried.
//! 3. **Health check**: hits `/v1/models` to verify reachability and readiness.
//!
//! # Default Configuration
//!
//! - Base URL: `http://localhost:1234`
//! - Default model: `gemma2-9b-it` (chat), `nomic-embed-text-v1.5` (embeddings, 768 dims)
//!
//! # Environment Variables
//!
//! | Variable                   | Default                   | Description                    |
//! |---------------------------|---------------------------|--------------------------------|
//! | `LMSTUDIO_HOST`           | `http://localhost:1234`   | LM Studio server URL           |
//! | `LMSTUDIO_MODEL`          | `gemma2-9b-it`            | Default chat model             |
//! | `LMSTUDIO_EMBEDDING_MODEL`| `nomic-embed-text-v1.5`   | Default embedding model        |
//! | `LMSTUDIO_EMBEDDING_DIM`  | `768`                     | Embedding dimension            |
//! | `LMSTUDIO_CONTEXT_LENGTH` | `131072`                  | Max context length (128 K)     |
//!
//! # Example
//!
//! ```rust,ignore
//! use edgequake_llm::LMStudioProvider;
//!
//! // Connect to local LM Studio with defaults
//! let provider = LMStudioProvider::from_env()?;
//!
//! // Or specify custom settings
//! let provider = LMStudioProvider::builder()
//!     .host("http://localhost:1234")
//!     .model("mistral-7b-instruct")
//!     .embedding_model("nomic-embed-text-v1.5")
//!     .build()?;
//! ```

use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::debug;

use crate::error::{LlmError, Result};
use crate::model_config::{ModelCapabilities, ModelCard, ModelType, ProviderConfig, ProviderType};
use crate::providers::openai_compatible::OpenAICompatibleProvider;
use crate::traits::{
    ChatMessage, CompletionOptions, EmbeddingProvider, LLMProvider, LLMResponse, StreamChunk,
    ToolChoice, ToolDefinition,
};

// ============================================================================
// Constants
// ============================================================================

/// Default LM Studio host URL.
const DEFAULT_LMSTUDIO_HOST: &str = "http://localhost:1234";

/// Default LM Studio chat model.
const DEFAULT_LMSTUDIO_MODEL: &str = "gemma2-9b-it";

/// Default LM Studio embedding model.
const DEFAULT_LMSTUDIO_EMBEDDING_MODEL: &str = "nomic-embed-text-v1.5";

/// Default embedding dimension for `nomic-embed-text-v1.5`.
const DEFAULT_LMSTUDIO_EMBEDDING_DIM: usize = 768;

// ============================================================================
// Provider Struct
// ============================================================================

/// LM Studio LLM and embedding provider.
///
/// Delegates all OpenAI-compatible operations to an inner
/// [`OpenAICompatibleProvider`].  LM Studio-specific logic (reasoning-model
/// routing, auto-load retry, health check) is implemented directly here.
#[derive(Debug)]
pub struct LMStudioProvider {
    /// Inner OpenAI-compatible provider — handles all standard HTTP / JSON.
    inner: OpenAICompatibleProvider,
    /// Raw host URL **without** the `/v1` suffix (used to build native REST URL).
    host: String,
    /// HTTP client dedicated to the native `/api/v1/chat` REST endpoint.
    /// Kept separate from `inner`'s client to allow distinct timeout / proxy settings.
    rest_client: Client,
    /// When `true` the provider will invoke `lms load <model>` on a
    /// *model-not-loaded* error and then retry the original request once.
    auto_load_models: bool,
    /// Stored max context length (mirrors the value placed in the inner `ProviderConfig`).
    max_context_length: usize,
}

// ============================================================================
// Builder
// ============================================================================

/// Builder for [`LMStudioProvider`].
#[derive(Debug, Clone)]
pub struct LMStudioProviderBuilder {
    host: String,
    model: String,
    embedding_model: String,
    max_context_length: usize,
    embedding_dimension: usize,
    auto_load_models: bool,
}

impl Default for LMStudioProviderBuilder {
    fn default() -> Self {
        Self {
            host: DEFAULT_LMSTUDIO_HOST.to_string(),
            model: DEFAULT_LMSTUDIO_MODEL.to_string(),
            embedding_model: DEFAULT_LMSTUDIO_EMBEDDING_MODEL.to_string(),
            max_context_length: 131_072, // 128 K — matches modern LM Studio defaults
            embedding_dimension: DEFAULT_LMSTUDIO_EMBEDDING_DIM,
            auto_load_models: true,
        }
    }
}

impl LMStudioProviderBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the LM Studio host URL.
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Set the chat model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the embedding model.
    pub fn embedding_model(mut self, model: impl Into<String>) -> Self {
        self.embedding_model = model.into();
        self
    }

    /// Set the maximum context length.
    pub fn max_context_length(mut self, length: usize) -> Self {
        self.max_context_length = length;
        self
    }

    /// Set the embedding dimension.
    pub fn embedding_dimension(mut self, dimension: usize) -> Self {
        self.embedding_dimension = dimension;
        self
    }

    /// Enable or disable automatic model loading via `lms` CLI (default: `true`).
    pub fn auto_load_models(mut self, enabled: bool) -> Self {
        self.auto_load_models = enabled;
        self
    }

    /// Build the [`LMStudioProvider`].
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::NetworkError`] if the HTTP client cannot be
    /// initialised, or [`LlmError::ConfigError`] if the inner provider config
    /// is invalid.
    pub fn build(self) -> Result<LMStudioProvider> {
        // Normalise host: strip trailing `/v1` so we control both API bases.
        let host = self.host.trim_end_matches("/v1").to_string();
        let base_url = format!("{}/v1", host);

        // Construct the ProviderConfig for the inner OpenAICompatibleProvider.
        let config = ProviderConfig {
            name: "lmstudio".to_string(),
            display_name: "LM Studio".to_string(),
            provider_type: ProviderType::LMStudio,
            // LM Studio local server does not require an API key.
            api_key_env: None,
            base_url: Some(base_url),
            default_llm_model: Some(self.model.clone()),
            default_embedding_model: Some(self.embedding_model.clone()),
            // Local inference can be slow — use a long timeout.
            timeout_seconds: 300,
            models: vec![
                // Chat model card
                ModelCard {
                    name: self.model.clone(),
                    display_name: self.model.clone(),
                    model_type: ModelType::Llm,
                    capabilities: ModelCapabilities {
                        context_length: self.max_context_length,
                        supports_function_calling: true,
                        supports_streaming: true,
                        supports_json_mode: false,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                // Embedding model card
                ModelCard {
                    name: self.embedding_model.clone(),
                    display_name: self.embedding_model.clone(),
                    model_type: ModelType::Embedding,
                    capabilities: ModelCapabilities {
                        context_length: 8_192,
                        embedding_dimension: self.embedding_dimension,
                        max_embedding_tokens: 8_192,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let inner = OpenAICompatibleProvider::from_config(config)?;

        let rest_client = Client::builder()
            .timeout(Duration::from_secs(300))
            .no_proxy()
            .build()
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        Ok(LMStudioProvider {
            inner,
            host,
            rest_client,
            auto_load_models: self.auto_load_models,
            max_context_length: self.max_context_length,
        })
    }
}

// ============================================================================
// Factory Methods
// ============================================================================

impl LMStudioProvider {
    /// Create a new `LMStudioProvider` from environment variables.
    pub fn from_env() -> Result<Self> {
        let host =
            std::env::var("LMSTUDIO_HOST").unwrap_or_else(|_| DEFAULT_LMSTUDIO_HOST.to_string());
        let model =
            std::env::var("LMSTUDIO_MODEL").unwrap_or_else(|_| DEFAULT_LMSTUDIO_MODEL.to_string());
        let embedding_model = std::env::var("LMSTUDIO_EMBEDDING_MODEL")
            .unwrap_or_else(|_| DEFAULT_LMSTUDIO_EMBEDDING_MODEL.to_string());
        let embedding_dimension = std::env::var("LMSTUDIO_EMBEDDING_DIM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_LMSTUDIO_EMBEDDING_DIM);
        let max_context_length = std::env::var("LMSTUDIO_CONTEXT_LENGTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(131_072);

        LMStudioProviderBuilder::new()
            .host(host)
            .model(model)
            .embedding_model(embedding_model)
            .embedding_dimension(embedding_dimension)
            .max_context_length(max_context_length)
            .build()
    }

    /// Create a new builder for [`LMStudioProvider`].
    pub fn builder() -> LMStudioProviderBuilder {
        LMStudioProviderBuilder::new()
    }

    /// Create with default settings (`http://localhost:1234`).
    pub fn default_local() -> Result<Self> {
        LMStudioProviderBuilder::new().build()
    }
}

// ============================================================================
// LM Studio–specific helpers (health check, auto-load, reasoning routing)
// ============================================================================

impl LMStudioProvider {
    /// Build the `/v1` OpenAI-compatible base URL.
    fn api_base(&self) -> String {
        format!("{}/v1", self.host)
    }

    /// Build the native REST API base URL (`/api/v1`).
    ///
    /// Used for reasoning-model requests via the LM Studio–proprietary endpoint.
    fn rest_api_base(&self) -> String {
        format!("{}/api/v1", self.host)
    }

    /// Verify that LM Studio is running and has at least one model loaded.
    pub async fn health_check(&self) -> Result<()> {
        let url = format!("{}/models", self.api_base());

        let response = self
            .rest_client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    LlmError::NetworkError(format!(
                        "LM Studio not responding at {}. Is LM Studio running?",
                        self.host
                    ))
                } else if e.is_connect() {
                    LlmError::NetworkError(format!(
                        "Cannot connect to LM Studio at {}. Please start LM Studio and load a model.",
                        self.host
                    ))
                } else {
                    LlmError::NetworkError(format!("LM Studio health check failed: {}", e))
                }
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError(format!(
                "LM Studio returned error status {}: {}. Please check that a model is loaded.",
                status, body
            )));
        }

        let body = response.text().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to read models response: {}", e))
        })?;

        debug!("LM Studio models response: {}", body);

        if !body.contains("\"data\"") && !body.contains("\"object\":") {
            return Err(LlmError::ApiError(
                "LM Studio /v1/models returned unexpected format. \
                 Please ensure LM Studio is properly initialised."
                    .to_string(),
            ));
        }

        Ok(())
    }

    /// Return `true` if the error text indicates the requested model is not loaded.
    fn is_model_not_loaded_error(error_text: &str) -> bool {
        error_text.contains("not a valid model ID")
            || error_text.contains("model not found")
            || error_text.contains("model not loaded")
            || error_text.contains("No model loaded")
    }

    /// Invoke the `lms` CLI to load `model_id` into LM Studio.
    async fn load_model_via_cli(&self, model_id: &str) -> Result<()> {
        eprintln!(
            "⏳ Model '{}' not loaded. Loading automatically via lms CLI...",
            model_id
        );
        eprintln!("   This may take 15–60 seconds depending on model size.");

        let start = std::time::Instant::now();

        let output = tokio::process::Command::new("lms")
            .args(["load", model_id, "--gpu", "max", "-y"])
            .output()
            .await
            .map_err(|e| {
                LlmError::ApiError(format!(
                    "Failed to run 'lms load' command: {}\n\n\
                    Troubleshooting:\n\
                    1. Ensure LM Studio is installed\n\
                    2. Make sure 'lms' CLI is in your PATH\n\
                    3. Run 'lms --help' to verify installation\n\
                    4. Alternatively, manually load the model in LM Studio GUI",
                    e
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(LlmError::ApiError(format!(
                "Failed to load model '{}' via lms CLI:\n{}\n{}\n\n\
                Please manually load the model in LM Studio GUI or check:\n\
                1. Model is downloaded locally (run 'lms ls' to check)\n\
                2. Enough RAM/VRAM available\n\
                3. LM Studio is running",
                model_id, stdout, stderr
            )));
        }

        eprintln!(
            "✅ Model '{}' loaded in {:.1}s",
            model_id,
            start.elapsed().as_secs_f64()
        );

        Ok(())
    }

    /// `chat` with automatic model-load retry on not-loaded errors.
    async fn chat_with_auto_load(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        // Reasoning models use the native REST endpoint.
        if is_reasoning_model(LLMProvider::model(&self.inner)) {
            debug!(
                provider = "lmstudio",
                model = %LLMProvider::model(&self.inner),
                "Routing to native REST API for reasoning model"
            );
            return self.chat_with_reasoning(messages, options).await;
        }

        match self.inner.chat(messages, options).await {
            Ok(r) => Ok(r),
            Err(e) if self.auto_load_models && Self::is_model_not_loaded_error(&e.to_string()) => {
                debug!(
                    provider = "lmstudio",
                    model = %LLMProvider::model(&self.inner),
                    "Model not loaded — attempting automatic load"
                );
                self.load_model_via_cli(LLMProvider::model(&self.inner))
                    .await?;
                self.inner.chat(messages, options).await
            }
            Err(e) => Err(e),
        }
    }

    /// `chat_with_tools` with automatic model-load retry on not-loaded errors.
    async fn chat_with_tools_with_auto_load(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        match self
            .inner
            .chat_with_tools(messages, tools, tool_choice.clone(), options)
            .await
        {
            Ok(r) => Ok(r),
            Err(e) if self.auto_load_models && Self::is_model_not_loaded_error(&e.to_string()) => {
                self.load_model_via_cli(LLMProvider::model(&self.inner))
                    .await?;
                self.inner
                    .chat_with_tools(messages, tools, tool_choice, options)
                    .await
            }
            Err(e) => Err(e),
        }
    }
}

// ============================================================================
// Reasoning-model routing (native /api/v1/chat endpoint)
// ============================================================================

/// Known reasoning-model identifier fragments (explicit allowlist).
///
/// Only models whose inference engines expose a reasoning toggle belong here.
/// Using an allowlist avoids false positives: unknown models default to the
/// safe path (no reasoning flag), which still produces valid output.
///
/// Fixes: <https://github.com/raphaelmansuy/edgequake-llm/issues/19>
const REASONING_MODELS: &[&str] = &[
    // DeepSeek R1 family
    "deepseek-r1",
    // Qwen3 text-reasoning sizes (NOT qwen3-vl, NOT qwen3-coder)
    "qwen3-235b",
    "qwen3-32b",
    "qwen3-30b",
    "qwen3-14b",
    "qwen3-8b",
    "qwen3-4b",
    "qwen3-1.7b",
    "qwen3-0.6b",
    // QwQ (dedicated reasoning model)
    "qwq",
    // Phi-4 reasoning
    "phi4-reasoning",
    // Granite 4 (IBM reasoning)
    "granite-4",
];

/// Return `true` if `model` is a known reasoning / thinking model.
///
/// Uses an explicit size-qualified allowlist rather than broad prefix matching
/// to avoid false positives for non-reasoning variants such as
/// `qwen3-vl-embedding-2b` or `qwen3-coder-14b`.
fn is_reasoning_model(model: &str) -> bool {
    let lower = model.to_lowercase();
    if REASONING_MODELS.iter().any(|&rm| lower.contains(rm)) {
        return true;
    }
    lower.contains("reasoning") || lower.contains("thinking")
}

// ── Native REST API request / response types ──────────────────────────────

/// Request body for `POST /api/v1/chat`.
#[derive(Debug, Serialize)]
struct RestChatRequest {
    model: String,
    input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<i32>,
}

/// Response from `POST /api/v1/chat`.
#[derive(Debug, Deserialize)]
struct RestChatResponse {
    #[allow(dead_code)]
    model_instance_id: String,
    output: Vec<RestOutputItem>,
    stats: RestStats,
}

/// A single output item in the native REST response.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum RestOutputItem {
    #[serde(rename = "reasoning")]
    Reasoning { content: String },
    #[serde(rename = "message")]
    Message { content: String },
}

/// Token statistics returned by the native REST API.
#[derive(Debug, Deserialize)]
struct RestStats {
    input_tokens: usize,
    total_output_tokens: usize,
    #[serde(default)]
    reasoning_output_tokens: usize,
    #[allow(dead_code)]
    tokens_per_second: f64,
    #[allow(dead_code)]
    time_to_first_token_seconds: f64,
}

/// Streaming events from `POST /api/v1/chat` with `stream: true`.
///
/// Reserved for a future streaming implementation of the native REST endpoint.
/// Currently unused because the non-streaming path already provides reasoning
/// and answer separation with token counts. Streaming will be added in a follow-up.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum RestStreamEvent {
    #[serde(rename = "chat.start")]
    ChatStart {
        #[allow(dead_code)]
        model_instance_id: Option<String>,
    },
    #[serde(rename = "reasoning.start")]
    ReasoningStart,
    #[serde(rename = "reasoning.delta")]
    ReasoningDelta { content: String },
    #[serde(rename = "reasoning.end")]
    ReasoningEnd,
    #[serde(rename = "message.start")]
    MessageStart,
    #[serde(rename = "message.delta")]
    MessageDelta { content: String },
    #[serde(rename = "message.end")]
    MessageEnd,
    #[serde(rename = "chat.end")]
    ChatEnd { result: RestChatResponse },
    #[serde(rename = "prompt_processing.start")]
    PromptProcessingStart,
    #[serde(rename = "prompt_processing.progress")]
    PromptProcessingProgress {
        #[allow(dead_code)]
        progress: f64,
    },
    #[serde(rename = "prompt_processing.end")]
    PromptProcessingEnd,
    #[serde(rename = "model_load.start")]
    ModelLoadStart {
        #[allow(dead_code)]
        model_instance_id: Option<String>,
    },
    #[serde(rename = "model_load.progress")]
    ModelLoadProgress {
        #[allow(dead_code)]
        progress: f64,
    },
    #[serde(rename = "model_load.end")]
    ModelLoadEnd {
        #[allow(dead_code)]
        load_time_seconds: Option<f64>,
    },
}

impl LMStudioProvider {
    /// Build a consolidated text `input` block for the native REST API from a
    /// `messages` slice.  Each message is prefixed with its role so the model
    /// can distinguish system instructions from user turns.
    fn build_rest_input(messages: &[ChatMessage]) -> String {
        messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    crate::traits::ChatRole::System => "System",
                    crate::traits::ChatRole::User => "User",
                    crate::traits::ChatRole::Assistant => "Assistant",
                    crate::traits::ChatRole::Tool | crate::traits::ChatRole::Function => "Tool",
                };
                format!("[{}]: {}", role, m.content)
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Call the native `/api/v1/chat` endpoint for reasoning-capable models.
    ///
    /// This endpoint returns explicit `reasoning` and `message` output items
    /// with per-phase token counts — richer than what the OpenAI-compatible
    /// endpoint exposes.
    async fn chat_with_reasoning(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let opts = options.cloned().unwrap_or_default();
        let input = Self::build_rest_input(messages);

        let request = RestChatRequest {
            model: LLMProvider::model(&self.inner).to_string(),
            input,
            reasoning: Some("on".to_string()),
            stream: Some(false),
            temperature: opts.temperature,
            max_output_tokens: opts.max_tokens.map(|t| t as i32),
        };

        let url = format!("{}/chat", self.rest_api_base());

        debug!(
            provider = "lmstudio",
            model = %LLMProvider::model(&self.inner),
            url = %url,
            "Sending native REST chat request for reasoning model"
        );

        let response = self
            .rest_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(format!("LM Studio REST request failed: {}", e)))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| LlmError::NetworkError(format!("Failed to read REST response: {}", e)))?;

        if !status.is_success() {
            return Err(LlmError::ApiError(format!(
                "LM Studio REST API error ({}): {}",
                status, body
            )));
        }

        let rest_resp: RestChatResponse = serde_json::from_str(&body).map_err(|e| {
            LlmError::ApiError(format!(
                "Failed to parse LM Studio REST response: {} — body: {}",
                e,
                &body[..body.len().min(500)]
            ))
        })?;

        // Separate reasoning and answer content.
        let mut thinking = String::new();
        let mut content = String::new();
        for item in &rest_resp.output {
            match item {
                RestOutputItem::Reasoning { content: c } => {
                    if !thinking.is_empty() {
                        thinking.push('\n');
                    }
                    thinking.push_str(c);
                }
                RestOutputItem::Message { content: c } => {
                    if !content.is_empty() {
                        content.push('\n');
                    }
                    content.push_str(c);
                }
            }
        }

        let prompt_tokens = rest_resp.stats.input_tokens;
        let completion_tokens = rest_resp.stats.total_output_tokens;
        let reasoning_tokens = rest_resp.stats.reasoning_output_tokens;

        let mut llm_response = LLMResponse::new(content, LLMProvider::model(&self.inner))
            .with_usage(prompt_tokens, completion_tokens)
            .with_finish_reason("stop".to_string());

        if reasoning_tokens > 0 {
            llm_response = llm_response.with_thinking_tokens(reasoning_tokens);
        }
        if !thinking.is_empty() {
            llm_response = llm_response.with_thinking_content(thinking);
        }

        Ok(llm_response)
    }
}

// ============================================================================
// LLMProvider implementation (delegating to inner)
// ============================================================================

#[async_trait]
impl LLMProvider for LMStudioProvider {
    fn name(&self) -> &str {
        "lmstudio"
    }

    fn model(&self) -> &str {
        LLMProvider::model(&self.inner)
    }

    fn max_context_length(&self) -> usize {
        self.max_context_length
    }

    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        self.complete_with_options(prompt, &CompletionOptions::default())
            .await
    }

    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<LLMResponse> {
        let mut msgs = Vec::new();
        if let Some(sys) = &options.system_prompt {
            msgs.push(ChatMessage::system(sys));
        }
        msgs.push(ChatMessage::user(prompt));
        self.chat(&msgs, Some(options)).await
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        self.chat_with_auto_load(messages, options).await
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        self.chat_with_tools_with_auto_load(messages, tools, tool_choice, options)
            .await
    }

    fn supports_streaming(&self) -> bool {
        self.inner.supports_streaming()
    }

    fn supports_function_calling(&self) -> bool {
        self.inner.supports_function_calling()
    }

    fn supports_json_mode(&self) -> bool {
        // LM Studio JSON mode is model-dependent; conservative default.
        false
    }

    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        self.inner.stream(prompt).await
    }

    fn supports_tool_streaming(&self) -> bool {
        self.inner.supports_tool_streaming()
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
}

// ============================================================================
// EmbeddingProvider implementation (delegating to inner)
// ============================================================================

#[async_trait]
impl EmbeddingProvider for LMStudioProvider {
    fn name(&self) -> &str {
        "lmstudio"
    }

    fn model(&self) -> &str {
        EmbeddingProvider::model(&self.inner)
    }

    fn dimension(&self) -> usize {
        EmbeddingProvider::dimension(&self.inner)
    }

    fn max_tokens(&self) -> usize {
        EmbeddingProvider::max_tokens(&self.inner)
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.inner.embed(texts).await
    }
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Builder tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_LMSTUDIO_HOST, "http://localhost:1234");
        assert_eq!(DEFAULT_LMSTUDIO_MODEL, "gemma2-9b-it");
        assert_eq!(DEFAULT_LMSTUDIO_EMBEDDING_MODEL, "nomic-embed-text-v1.5");
        assert_eq!(DEFAULT_LMSTUDIO_EMBEDDING_DIM, 768);
    }

    #[test]
    fn test_builder_defaults() {
        let builder = LMStudioProviderBuilder::default();
        assert_eq!(builder.host, DEFAULT_LMSTUDIO_HOST);
        assert_eq!(builder.model, DEFAULT_LMSTUDIO_MODEL);
        assert_eq!(builder.embedding_model, DEFAULT_LMSTUDIO_EMBEDDING_MODEL);
        assert_eq!(builder.embedding_dimension, DEFAULT_LMSTUDIO_EMBEDDING_DIM);
        assert!(builder.auto_load_models);
    }

    #[test]
    fn test_builder_auto_load_models_default() {
        let builder = LMStudioProviderBuilder::default();
        assert!(builder.auto_load_models);
    }

    #[test]
    fn test_builder_auto_load_models_disabled() {
        let provider = LMStudioProviderBuilder::new()
            .auto_load_models(false)
            .build()
            .unwrap();
        assert!(!provider.auto_load_models);
    }

    #[test]
    fn test_builder_max_context_length() {
        let provider = LMStudioProviderBuilder::new()
            .max_context_length(65_536)
            .build()
            .unwrap();
        assert_eq!(provider.max_context_length(), 65_536);
    }

    #[test]
    fn test_builder_host_strips_v1_suffix() {
        let provider = LMStudioProviderBuilder::new()
            .host("http://localhost:1234/v1")
            .build()
            .unwrap();
        assert_eq!(provider.host, "http://localhost:1234");
    }

    // ── LLMProvider trait ─────────────────────────────────────────────────────

    #[test]
    fn test_provider_name() {
        let provider = LMStudioProviderBuilder::new().build().unwrap();
        assert_eq!(LLMProvider::name(&provider), "lmstudio");
    }

    #[test]
    fn test_provider_model() {
        let provider = LMStudioProviderBuilder::new()
            .model("mistral-7b-instruct")
            .build()
            .unwrap();
        assert_eq!(LLMProvider::model(&provider), "mistral-7b-instruct");
    }

    #[test]
    fn test_supports_streaming() {
        let provider = LMStudioProviderBuilder::new().build().unwrap();
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_supports_json_mode() {
        let provider = LMStudioProviderBuilder::new().build().unwrap();
        assert!(!provider.supports_json_mode());
    }

    // ── EmbeddingProvider trait ───────────────────────────────────────────────

    #[test]
    fn test_embedding_provider_name() {
        let provider = LMStudioProviderBuilder::new().build().unwrap();
        assert_eq!(EmbeddingProvider::name(&provider), "lmstudio");
    }

    #[test]
    fn test_embedding_provider_model() {
        let provider = LMStudioProviderBuilder::new()
            .embedding_model("custom-embed")
            .build()
            .unwrap();
        assert_eq!(EmbeddingProvider::model(&provider), "custom-embed");
    }

    #[test]
    fn test_embedding_provider_max_tokens() {
        let provider = LMStudioProviderBuilder::new().build().unwrap();
        assert_eq!(EmbeddingProvider::max_tokens(&provider), 8_192);
    }

    #[tokio::test]
    async fn test_embed_empty_input() {
        let provider = LMStudioProviderBuilder::new().build().unwrap();
        let result = provider.embed(&[]).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    // ── REST API URL helpers ──────────────────────────────────────────────────

    #[test]
    fn test_rest_api_base() {
        let provider = LMStudioProviderBuilder::new()
            .host("http://localhost:1234")
            .build()
            .unwrap();
        assert_eq!(provider.rest_api_base(), "http://localhost:1234/api/v1");
    }

    #[test]
    fn test_rest_api_base_with_v1_suffix() {
        let provider = LMStudioProviderBuilder::new()
            .host("http://localhost:1234/v1")
            .build()
            .unwrap();
        assert_eq!(provider.rest_api_base(), "http://localhost:1234/api/v1");
    }

    // ── Reasoning model detection ─────────────────────────────────────────────

    #[test]
    fn test_is_reasoning_model_deepseek() {
        assert!(is_reasoning_model("deepseek-r1-8b"));
        assert!(is_reasoning_model("DeepSeek-R1-Distill-Qwen-14B"));
    }

    #[test]
    fn test_is_reasoning_model_qwen3_sizes() {
        assert!(is_reasoning_model("qwen3-235b"));
        assert!(is_reasoning_model("qwen3-32b"));
        assert!(is_reasoning_model("qwen3-14b"));
        assert!(is_reasoning_model("qwen3-8b"));
        assert!(is_reasoning_model("qwen3-4b"));
        assert!(is_reasoning_model("qwen3-1.7b"));
        assert!(is_reasoning_model("qwen3-0.6b"));
        // Mixed case
        assert!(is_reasoning_model("Qwen3-14B-GGUF"));
        // Models with "thinking" in name
        assert!(is_reasoning_model("Qwen3-Thinking"));
    }

    #[test]
    fn test_is_reasoning_model_qwen3_false_positives_issue_19() {
        // Issue #19: These non-reasoning Qwen3 variants must NOT match.
        assert!(!is_reasoning_model("qwen3-vl-embedding-2b-mlx-nvfp4"));
        assert!(!is_reasoning_model(
            "arthurcollet/Qwen3-VL-Embedding-2B-mlx-nvfp4"
        ));
        assert!(!is_reasoning_model("qwen3-vl-4b"));
        assert!(!is_reasoning_model("Qwen3-VL-72B"));
        assert!(!is_reasoning_model("qwen3-coder-14b"));
    }

    #[test]
    fn test_is_reasoning_model_others() {
        assert!(is_reasoning_model("qwq"));
        assert!(is_reasoning_model("phi4-reasoning"));
        assert!(is_reasoning_model("granite-4"));
        assert!(is_reasoning_model("model-with-reasoning"));
        assert!(is_reasoning_model("model-thinking"));
    }

    #[test]
    fn test_is_reasoning_model_non_reasoning() {
        assert!(!is_reasoning_model("llama-3"));
        assert!(!is_reasoning_model("gpt-oss-20b"));
        assert!(!is_reasoning_model("mistral-7b"));
        assert!(!is_reasoning_model("gemma2-9b"));
        // Bare "qwen3" without a known size suffix should not match.
        assert!(!is_reasoning_model("qwen3"));
    }

    // ── Native REST API type tests ────────────────────────────────────────────

    #[test]
    fn test_rest_output_item_parsing_reasoning() {
        let json = r#"{"type": "reasoning", "content": "Let me think..."}"#;
        let item: RestOutputItem = serde_json::from_str(json).unwrap();
        match item {
            RestOutputItem::Reasoning { content } => assert_eq!(content, "Let me think..."),
            _ => panic!("Expected Reasoning variant"),
        }
    }

    #[test]
    fn test_rest_output_item_parsing_message() {
        let json = r#"{"type": "message", "content": "The answer is 42."}"#;
        let item: RestOutputItem = serde_json::from_str(json).unwrap();
        match item {
            RestOutputItem::Message { content } => assert_eq!(content, "The answer is 42."),
            _ => panic!("Expected Message variant"),
        }
    }

    #[test]
    fn test_rest_stats_parsing() {
        let json = r#"{
            "input_tokens": 100,
            "total_output_tokens": 150,
            "reasoning_output_tokens": 50,
            "tokens_per_second": 43.73,
            "time_to_first_token_seconds": 0.781
        }"#;
        let stats: RestStats = serde_json::from_str(json).unwrap();
        assert_eq!(stats.input_tokens, 100);
        assert_eq!(stats.total_output_tokens, 150);
        assert_eq!(stats.reasoning_output_tokens, 50);
    }

    #[test]
    fn test_rest_chat_request_serialization() {
        let req = RestChatRequest {
            model: "deepseek-r1".to_string(),
            input: "What is 2+2?".to_string(),
            reasoning: Some("on".to_string()),
            stream: Some(false),
            temperature: Some(0.7),
            max_output_tokens: Some(1_000),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"model\":\"deepseek-r1\""));
        assert!(json.contains("\"reasoning\":\"on\""));
        assert!(json.contains("\"stream\":false"));
    }

    #[test]
    fn test_rest_stream_event_parsing_reasoning_delta() {
        let json = r#"{"type": "reasoning.delta", "content": "Step 1..."}"#;
        let event: RestStreamEvent = serde_json::from_str(json).unwrap();
        match event {
            RestStreamEvent::ReasoningDelta { content } => assert_eq!(content, "Step 1..."),
            _ => panic!("Expected ReasoningDelta"),
        }
    }

    #[test]
    fn test_rest_stream_event_parsing_message_delta() {
        let json = r#"{"type": "message.delta", "content": "Hello"}"#;
        let event: RestStreamEvent = serde_json::from_str(json).unwrap();
        match event {
            RestStreamEvent::MessageDelta { content } => assert_eq!(content, "Hello"),
            _ => panic!("Expected MessageDelta"),
        }
    }

    #[test]
    fn test_rest_chat_response_parsing() {
        let json = r#"{
            "model_instance_id": "test-instance",
            "output": [
                {"type": "reasoning", "content": "Thinking..."},
                {"type": "message", "content": "Answer"}
            ],
            "stats": {
                "input_tokens": 10,
                "total_output_tokens": 20,
                "reasoning_output_tokens": 5,
                "tokens_per_second": 30.0,
                "time_to_first_token_seconds": 0.2
            }
        }"#;
        let r: RestChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(r.output.len(), 2);
        assert_eq!(r.stats.reasoning_output_tokens, 5);
    }

    #[test]
    fn test_build_rest_input_user_only() {
        let msgs = vec![ChatMessage::user("hello")];
        let input = LMStudioProvider::build_rest_input(&msgs);
        assert_eq!(input, "[User]: hello");
    }

    #[test]
    fn test_build_rest_input_system_and_user() {
        let msgs = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hi"),
        ];
        let input = LMStudioProvider::build_rest_input(&msgs);
        assert!(input.contains("[System]: You are helpful."));
        assert!(input.contains("[User]: Hi"));
    }

    // ── Model-not-loaded error detection ──────────────────────────────────────

    #[test]
    fn test_is_model_not_loaded_error() {
        assert!(LMStudioProvider::is_model_not_loaded_error(
            "not a valid model ID"
        ));
        assert!(LMStudioProvider::is_model_not_loaded_error(
            "model not found"
        ));
        assert!(LMStudioProvider::is_model_not_loaded_error(
            "model not loaded"
        ));
        assert!(LMStudioProvider::is_model_not_loaded_error(
            "No model loaded"
        ));
        assert!(!LMStudioProvider::is_model_not_loaded_error("timeout"));
        assert!(!LMStudioProvider::is_model_not_loaded_error("Bad Request"));
    }
}
