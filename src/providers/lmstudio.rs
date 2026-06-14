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
//! | `LMSTUDIO_CONTEXT_LENGTH` | `131072`                  | Fallback max context when metadata sync fails |
//! | `LMSTUDIO_TIMEOUT_SECONDS`| `600`                     | HTTP timeout for chat/completions (seconds)   |
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

/// Default HTTP timeout for local inference (overridable via `LMSTUDIO_TIMEOUT_SECONDS`).
const DEFAULT_LMSTUDIO_TIMEOUT_SECS: u64 = 600;

/// Short timeout for metadata probes (`GET /api/v1/models`).
const LMSTUDIO_METADATA_TIMEOUT_SECS: u64 = 5;

/// Fallback completion budget when metadata is unavailable.
const DEFAULT_LMSTUDIO_MAX_OUTPUT_TOKENS: usize = 4096;

// ============================================================================
// Live model metadata (native `/api/v1/models`)
// ============================================================================

/// Resolved limits for the active LM Studio model instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LmStudioModelMetadata {
    /// Context length configured on the loaded instance (UI load setting).
    pub active_context_length: usize,
    /// Model's advertised maximum context length.
    pub max_context_length: usize,
    /// Suggested `max_tokens` when callers omit an explicit budget.
    pub default_max_output_tokens: usize,
}

#[derive(Debug, Clone)]
struct LmStudioRuntimeState {
    active_context_length: usize,
    model_max_context_length: usize,
    default_max_output_tokens: usize,
    metadata_synced: bool,
}

impl LmStudioRuntimeState {
    fn from_builder_default(max_context_length: usize) -> Self {
        Self {
            active_context_length: max_context_length,
            model_max_context_length: max_context_length,
            default_max_output_tokens: default_max_output_from_context(max_context_length),
            metadata_synced: false,
        }
    }

    fn apply_metadata(&mut self, metadata: LmStudioModelMetadata) {
        self.active_context_length = metadata.active_context_length;
        self.model_max_context_length = metadata.max_context_length;
        self.default_max_output_tokens = metadata.default_max_output_tokens;
        self.metadata_synced = true;
    }
}

/// Derive a safe default completion budget from the active context window.
pub fn default_max_output_from_context(active_context_length: usize) -> usize {
    if active_context_length == 0 {
        return DEFAULT_LMSTUDIO_MAX_OUTPUT_TOKENS;
    }
    (active_context_length / 4).clamp(1024, 8192)
}

#[derive(Debug, Deserialize)]
struct NativeModelsResponse {
    #[serde(default)]
    models: Vec<NativeModelEntry>,
}

#[derive(Debug, Deserialize)]
struct NativeModelEntry {
    key: String,
    #[serde(default)]
    max_context_length: Option<u64>,
    #[serde(default)]
    loaded_instances: Vec<NativeLoadedInstance>,
    #[serde(default)]
    variants: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct NativeLoadedInstance {
    #[serde(default)]
    id: String,
    config: NativeLoadedConfig,
}

#[derive(Debug, Deserialize, Default)]
struct NativeLoadedConfig {
    #[serde(default)]
    context_length: Option<u64>,
}

fn model_hint_matches(candidate: &str, hint: &str) -> bool {
    let candidate = candidate.trim();
    let hint = hint.trim();
    if candidate.is_empty() || hint.is_empty() {
        return false;
    }
    if candidate == hint
        || candidate.ends_with(hint)
        || hint.ends_with(candidate)
        || candidate.contains(hint)
        || hint.contains(candidate)
    {
        return true;
    }
    candidate.rsplit('/').next() == hint.rsplit('/').next()
}

/// Parse native LM Studio model metadata from `/api/v1/models` JSON.
pub fn resolve_model_metadata_from_json(
    payload: &str,
    model_hint: &str,
) -> Option<LmStudioModelMetadata> {
    let response: NativeModelsResponse = serde_json::from_str(payload).ok()?;
    let entry = response.models.into_iter().find(|entry| {
        model_hint_matches(&entry.key, model_hint)
            || entry
                .variants
                .iter()
                .any(|variant| model_hint_matches(variant, model_hint))
            || entry
                .loaded_instances
                .iter()
                .any(|instance| model_hint_matches(&instance.id, model_hint))
    })?;

    let model_max = entry
        .max_context_length
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .unwrap_or(131_072);

    let loaded_context = entry
        .loaded_instances
        .iter()
        .find(|instance| model_hint_matches(&instance.id, model_hint))
        .or_else(|| entry.loaded_instances.first())
        .and_then(|instance| instance.config.context_length)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0);

    let active_context_length = loaded_context.unwrap_or(model_max);
    Some(LmStudioModelMetadata {
        active_context_length,
        max_context_length: model_max,
        default_max_output_tokens: default_max_output_from_context(active_context_length),
    })
}

async fn fetch_native_model_metadata(
    host: &str,
    model_hint: &str,
) -> Result<LmStudioModelMetadata> {
    let host = host.trim_end_matches("/v1");
    let url = format!("{host}/api/v1/models");
    let client = Client::builder()
        .timeout(Duration::from_secs(LMSTUDIO_METADATA_TIMEOUT_SECS))
        .no_proxy()
        .build()
        .map_err(|e| LlmError::NetworkError(e.to_string()))?;

    let response =
        client.get(&url).send().await.map_err(|e| {
            LlmError::NetworkError(format!("LM Studio metadata request failed: {e}"))
        })?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "unknown error".to_string());
        return Err(LlmError::ApiError(format!(
            "LM Studio metadata endpoint returned {status}: {body}"
        )));
    }

    let body = response.text().await.map_err(|e| {
        LlmError::NetworkError(format!("Failed to read LM Studio metadata response: {e}"))
    })?;

    resolve_model_metadata_from_json(&body, model_hint).ok_or_else(|| {
        LlmError::ApiError(format!(
            "Could not resolve LM Studio metadata for model hint '{model_hint}'"
        ))
    })
}

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
    /// Live model limits synced from `GET /api/v1/models`.
    runtime: std::sync::RwLock<LmStudioRuntimeState>,
    /// HTTP timeout for chat/completions (seconds).
    timeout_seconds: u64,
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
    timeout_seconds: u64,
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
            timeout_seconds: DEFAULT_LMSTUDIO_TIMEOUT_SECS,
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

    /// HTTP timeout for chat/completions in seconds (default: 600).
    pub fn timeout_seconds(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds.max(30);
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
            // Official OpenAI client examples use a placeholder key; suppresses noisy warnings.
            api_key: Some("lm-studio".to_string()),
            base_url: Some(base_url),
            default_llm_model: Some(self.model.clone()),
            default_embedding_model: Some(self.embedding_model.clone()),
            timeout_seconds: self.timeout_seconds,
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
            .timeout(Duration::from_secs(self.timeout_seconds))
            .no_proxy()
            .build()
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        Ok(LMStudioProvider {
            inner,
            host,
            rest_client,
            auto_load_models: self.auto_load_models,
            runtime: std::sync::RwLock::new(LmStudioRuntimeState::from_builder_default(
                self.max_context_length,
            )),
            timeout_seconds: self.timeout_seconds,
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
        let timeout_seconds = std::env::var("LMSTUDIO_TIMEOUT_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_LMSTUDIO_TIMEOUT_SECS);

        LMStudioProviderBuilder::new()
            .host(host)
            .model(model)
            .embedding_model(embedding_model)
            .embedding_dimension(embedding_dimension)
            .max_context_length(max_context_length)
            .timeout_seconds(timeout_seconds)
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

    /// HTTP timeout used for chat/completions requests.
    pub fn http_timeout_seconds(&self) -> u64 {
        self.timeout_seconds
    }

    /// Pull live context/output limits from LM Studio's native `/api/v1/models` endpoint.
    pub async fn refresh_model_metadata(&self) -> Result<LmStudioModelMetadata> {
        let model_hint = LLMProvider::model(&self.inner).to_string();
        let metadata = fetch_native_model_metadata(&self.host, &model_hint).await?;
        self.runtime
            .write()
            .map_err(|_| LlmError::ProviderError("LM Studio runtime lock poisoned".into()))?
            .apply_metadata(metadata);
        debug!(
            provider = "lmstudio",
            model = %model_hint,
            active_context = metadata.active_context_length,
            max_context = metadata.max_context_length,
            default_max_output = metadata.default_max_output_tokens,
            "Synced LM Studio model metadata"
        );
        Ok(metadata)
    }

    async fn ensure_model_metadata(&self) {
        let already_synced = self
            .runtime
            .read()
            .map(|state| state.metadata_synced)
            .unwrap_or(false);
        if already_synced {
            return;
        }
        let _ = self.refresh_model_metadata().await;
    }

    fn merge_completion_options(&self, options: Option<&CompletionOptions>) -> CompletionOptions {
        let state = self
            .runtime
            .read()
            .expect("LM Studio runtime lock poisoned");
        let mut opts = options.cloned().unwrap_or_default();
        if opts.max_tokens.is_none() {
            opts.max_tokens = Some(state.default_max_output_tokens);
        }
        normalize_lmstudio_completion_options(&mut opts);
        opts
    }
}

/// LM Studio OpenAI-compat knobs — see https://lmstudio.ai/docs/developer/openai-compat
fn normalize_lmstudio_completion_options(opts: &mut CompletionOptions) {
    // LM Studio accepts reasoning_effort on /v1/chat/completions (low/medium/high/none).
    // Trim whitespace so callers can pass `" none "` safely.
    if let Some(effort) = opts.reasoning_effort.as_mut() {
        *effort = effort.trim().to_string();
        if effort.is_empty() {
            opts.reasoning_effort = None;
        }
    }
}

/// Native `/api/v1/chat` reasoning toggle derived from OpenAI-shaped options.
///
/// When `reasoning_effort` is `"none"`, reasoning must stay off so completion budget
/// is available for tool JSON on hybrid models (Qwen3.6, DeepSeek R1, …).
fn native_rest_reasoning_enabled(options: &CompletionOptions, model: &str) -> Option<String> {
    match options.reasoning_effort.as_deref() {
        Some("none") | Some("off") | Some("false") | Some("disabled") => None,
        Some(_) => Some("on".to_string()),
        None if is_reasoning_model(model) => Some("on".to_string()),
        None => None,
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
        self.ensure_model_metadata().await;
        let merged = self.merge_completion_options(options);
        let options = Some(&merged);

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
        self.ensure_model_metadata().await;
        let merged = self.merge_completion_options(options);
        let options = Some(&merged);

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
    PromptProcessingProgress { progress: f64 },
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
            reasoning: native_rest_reasoning_enabled(&opts, LLMProvider::model(&self.inner)),
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

    /// Map a native REST SSE event to a [`StreamChunk`], if applicable.
    fn map_rest_stream_event(event: RestStreamEvent) -> Option<StreamChunk> {
        use crate::traits::StreamUsage;

        match event {
            RestStreamEvent::PromptProcessingProgress { progress } => {
                Some(StreamChunk::PrefillProgress { progress })
            }
            RestStreamEvent::ReasoningDelta { content } => Some(StreamChunk::ThinkingContent {
                text: content,
                tokens_used: None,
                budget_total: None,
            }),
            RestStreamEvent::MessageDelta { content } => Some(StreamChunk::Content(content)),
            RestStreamEvent::ChatEnd { result } => {
                let usage =
                    StreamUsage::new(result.stats.input_tokens, result.stats.total_output_tokens)
                        .with_thinking_tokens(result.stats.reasoning_output_tokens);
                Some(StreamChunk::Finished {
                    reason: "stop".to_string(),
                    ttft_ms: Some(result.stats.time_to_first_token_seconds * 1000.0),
                    usage: Some(usage),
                })
            }
            RestStreamEvent::ChatStart { .. }
            | RestStreamEvent::ReasoningStart
            | RestStreamEvent::ReasoningEnd
            | RestStreamEvent::MessageStart
            | RestStreamEvent::MessageEnd
            | RestStreamEvent::PromptProcessingStart
            | RestStreamEvent::PromptProcessingEnd
            | RestStreamEvent::ModelLoadStart { .. }
            | RestStreamEvent::ModelLoadProgress { .. }
            | RestStreamEvent::ModelLoadEnd { .. } => None,
        }
    }

    /// Stream chat via native `POST /api/v1/chat` (prefill progress + reasoning deltas).
    async fn chat_rest_stream(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        use futures::stream::{self, StreamExt};
        use reqwest_eventsource::{Event, EventSource};

        let opts = options.cloned().unwrap_or_default();
        let input = Self::build_rest_input(messages);
        let model = LLMProvider::model(&self.inner).to_string();
        let reasoning = native_rest_reasoning_enabled(&opts, &model);

        let request = RestChatRequest {
            model,
            input,
            reasoning,
            stream: Some(true),
            temperature: opts.temperature,
            max_output_tokens: opts.max_tokens.map(|t| t as i32),
        };

        let url = format!("{}/chat", self.rest_api_base());
        debug!(
            provider = "lmstudio",
            url = %url,
            "Opening native REST chat stream"
        );

        let req_builder = self.rest_client.post(&url).json(&request);
        let event_source = EventSource::new(req_builder)
            .map_err(|e| LlmError::ApiError(format!("LM Studio REST stream failed: {e}")))?;

        let stream = stream::unfold(event_source, |mut es| async move {
            loop {
                match es.next().await {
                    None => return None,
                    Some(Ok(Event::Open)) => continue,
                    Some(Ok(Event::Message(msg))) => {
                        if msg.data == "[DONE]" {
                            return Some((
                                Ok(StreamChunk::Finished {
                                    reason: "stop".to_string(),
                                    ttft_ms: None,
                                    usage: None,
                                }),
                                es,
                            ));
                        }
                        match serde_json::from_str::<RestStreamEvent>(&msg.data) {
                            Ok(event) => {
                                if let Some(chunk) = Self::map_rest_stream_event(event) {
                                    return Some((Ok(chunk), es));
                                }
                                continue;
                            }
                            Err(e) => {
                                return Some((
                                    Err(LlmError::ApiError(format!(
                                        "Failed to parse LM Studio REST stream event: {e}"
                                    ))),
                                    es,
                                ));
                            }
                        }
                    }
                    Some(Err(e)) => {
                        return Some((
                            Err(LlmError::NetworkError(format!(
                                "LM Studio REST stream error: {e}"
                            ))),
                            es,
                        ));
                    }
                }
            }
        });

        Ok(Box::pin(stream))
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
        self.runtime
            .read()
            .map(|state| state.active_context_length)
            .unwrap_or(131_072)
    }

    async fn refresh_model_metadata(&self) -> Result<()> {
        LMStudioProvider::refresh_model_metadata(self)
            .await
            .map(|_| ())
    }

    fn default_max_output_tokens(&self) -> Option<usize> {
        Some(
            self.runtime
                .read()
                .map(|state| state.default_max_output_tokens)
                .unwrap_or(DEFAULT_LMSTUDIO_MAX_OUTPUT_TOKENS),
        )
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
        self.ensure_model_metadata().await;
        let merged = self.merge_completion_options(options);
        let options = Some(&merged);

        // Tool calls require the OpenAI-compatible endpoint; native REST has no tool schema.
        // Tool-free turns use native REST for `prompt_processing.progress` prefill events.
        if tools.is_empty() {
            return self.chat_rest_stream(messages, options).await;
        }

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
        // Qwen3.6 MoE uses dot versioning — tool turns stay on OpenAI-compat path.
        assert!(!is_reasoning_model("qwen/qwen3.6-35b-a3b"));
    }

    #[test]
    fn test_native_rest_reasoning_disabled_when_effort_none() {
        let opts = CompletionOptions {
            reasoning_effort: Some("none".to_string()),
            ..Default::default()
        };
        assert!(native_rest_reasoning_enabled(&opts, "qwen3-14b").is_none());
    }

    #[test]
    fn test_native_rest_reasoning_enabled_for_reasoning_models_by_default() {
        let opts = CompletionOptions::default();
        assert_eq!(
            native_rest_reasoning_enabled(&opts, "qwen3-14b").as_deref(),
            Some("on")
        );
    }

    #[test]
    fn test_normalize_lmstudio_completion_options_trims_effort() {
        let mut opts = CompletionOptions {
            reasoning_effort: Some(" none ".to_string()),
            ..Default::default()
        };
        normalize_lmstudio_completion_options(&mut opts);
        assert_eq!(opts.reasoning_effort.as_deref(), Some("none"));
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
    fn test_rest_stream_event_parsing_prompt_processing_progress() {
        let json = r#"{"type": "prompt_processing.progress", "progress": 0.73}"#;
        let event: RestStreamEvent = serde_json::from_str(json).unwrap();
        match LMStudioProvider::map_rest_stream_event(event) {
            Some(StreamChunk::PrefillProgress { progress }) => {
                assert!((progress - 0.73).abs() < f64::EPSILON);
            }
            other => panic!("Expected PrefillProgress, got {other:?}"),
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

    #[test]
    fn resolve_metadata_prefers_loaded_instance_context() {
        let payload = r#"{
            "models": [{
                "key": "qwen/qwen3.6-35b-a3b",
                "max_context_length": 262144,
                "loaded_instances": [{
                    "id": "qwen/qwen3.6-35b-a3b",
                    "config": { "context_length": 65536 }
                }],
                "variants": ["qwen/qwen3.6-35b-a3b@q4_k_m"]
            }]
        }"#;
        let metadata =
            resolve_model_metadata_from_json(payload, "qwen/qwen3.6-35b-a3b").expect("metadata");
        assert_eq!(metadata.active_context_length, 65_536);
        assert_eq!(metadata.max_context_length, 262_144);
        assert_eq!(metadata.default_max_output_tokens, 8192);
    }

    #[test]
    fn resolve_metadata_falls_back_to_model_max_when_not_loaded() {
        let payload = r#"{
            "models": [{
                "key": "deepseek-r1",
                "max_context_length": 131072,
                "loaded_instances": []
            }]
        }"#;
        let metadata = resolve_model_metadata_from_json(payload, "deepseek-r1").expect("metadata");
        assert_eq!(metadata.active_context_length, 131_072);
        assert_eq!(metadata.default_max_output_tokens, 8192);
    }
}
