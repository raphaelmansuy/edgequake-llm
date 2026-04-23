//! Mistral AI Provider
//!
//! @implements FEAT-007: Mistral AI provider (chat, embeddings, list-models)
//! Closes <https://github.com/raphaelmansuy/edgequake-llm/issues/7>
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
//! # Available Chat Models (April 2026)
//!
//! | Model alias | Snapshot | Context | Vision | FC |
//! |-------------|----------|---------|--------|----|
//! | `mistral-large-latest` | 2512 (Large 3) | 256 K | ✓ | ✓ |
//! | `mistral-medium-latest` | 2508 (Med 3.1) | 128 K | ✓ | ✓ |
//! | `mistral-small-latest` | 2603 (Small 4) | 256 K | ✓ | ✓ |
//! | `magistral-medium-latest` | Med 1.2 (reasoning) | 128 K | | ✓ |
//! | `magistral-small-latest` | Sm 1.2 (reasoning) | 128 K | | ✓ |
//! | `codestral-latest` | 2508 | 256 K | | |
//! | `devstral-latest` | 2512 | 128 K | | ✓ |
//! | `devstral-small-latest` | | 128 K | | ✓ |
//! | `ministral-3b-latest` | 3B | 128 K | ✓ | ✓ |
//! | `ministral-8b-latest` | 8B | 128 K | ✓ | ✓ |
//! | `ministral-14b-latest` | 14B | 128 K | ✓ | ✓ |
//! | `open-mistral-nemo` | Nemo 12B | 128 K | | ✓ |
//! | `mistral-embed` | — | 8 K | | |
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
use reqwest::{multipart, Client};
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

/// Maximum output tokens for standard Mistral chat models.
///
/// Mistral Large 3 and most frontier models support up to 16 384 output tokens.
/// Models that do not have a specific limit advertised default to 4 096.
const MISTRAL_DEFAULT_MAX_OUTPUT_TOKENS: usize = 4096;

/// Maximum output tokens for larger frontier models (Large / Medium series).
const MISTRAL_FRONTIER_MAX_OUTPUT_TOKENS: usize = 16384;
const MISTRAL_EMBED_DIMENSION: usize = 1024;

/// Maximum tokens for `mistral-embed`
const MISTRAL_EMBED_MAX_TOKENS: usize = 8192;

/// Provider display name
const MISTRAL_PROVIDER_NAME: &str = "mistral";

/// Mistral chat model catalog: (id, display_name, context_length, supports_vision, supports_function_calling)
///
/// Context lengths are sourced from official Mistral documentation (docs.mistral.ai, April 2026).
/// 256 K  = 262_144 tokens, 128 K = 131_072 tokens, 32 K = 32_768 tokens.
///
/// "-latest" aliases always point to the newest stable snapshot.
const MISTRAL_CHAT_MODELS: &[(&str, &str, usize, bool, bool)] = &[
    // ----------------------------------------------------------------
    // Premier frontier — general purpose
    // ----------------------------------------------------------------
    (
        "mistral-large-latest", // → mistral-large-2512 (Mistral Large 3)
        "Mistral Large 3 (latest)",
        262_144, // 256 K
        true,    // vision
        true,    // function calling
    ),
    (
        "mistral-large-2512",
        "Mistral Large 3 (2512)",
        262_144,
        true,
        true,
    ),
    (
        "mistral-medium-latest", // → mistral-medium-2508 (Mistral Medium 3.1)
        "Mistral Medium 3.1 (latest)",
        131_072, // 128 K
        true,
        true,
    ),
    (
        "mistral-medium-2508",
        "Mistral Medium 3.1 (2508)",
        131_072,
        true,
        true,
    ),
    (
        "mistral-small-latest", // → mistral-small-2603 (Mistral Small 4)
        "Mistral Small 4 (latest)",
        262_144, // 256 K
        true,
        true,
    ),
    (
        "mistral-small-2603",
        "Mistral Small 4 (2603)",
        262_144,
        true,
        true,
    ),
    // ----------------------------------------------------------------
    // Reasoning models (Magistral series)
    // ----------------------------------------------------------------
    (
        "magistral-medium-latest", // → Magistral Medium 1.2
        "Magistral Medium 1.2 (latest)",
        131_072,
        false,
        true,
    ),
    (
        "magistral-small-latest", // → Magistral Small 1.2
        "Magistral Small 1.2 (latest)",
        131_072,
        false,
        true,
    ),
    // ----------------------------------------------------------------
    // Code models
    // ----------------------------------------------------------------
    (
        "codestral-latest", // → codestral-2508
        "Codestral (latest)",
        262_144, // 256 K
        false,
        false, // FIM model; tool calling not supported on chat path
    ),
    ("codestral-2508", "Codestral 2508", 262_144, false, false),
    // ----------------------------------------------------------------
    // Code agent models (Devstral series)
    // ----------------------------------------------------------------
    (
        "devstral-latest", // → devstral-2512 (Devstral 2)
        "Devstral 2 (latest)",
        131_072,
        false,
        true,
    ),
    (
        "devstral-small-latest",
        "Devstral Small (latest)",
        131_072,
        false,
        true,
    ),
    // ----------------------------------------------------------------
    // Ministral edge / on-device models
    // ----------------------------------------------------------------
    (
        "ministral-3b-latest",
        "Ministral 3 3B (latest)",
        131_072,
        true,
        true,
    ),
    (
        "ministral-8b-latest",
        "Ministral 3 8B (latest)",
        131_072,
        true,
        true,
    ),
    (
        "ministral-14b-latest",
        "Ministral 3 14B (latest)",
        131_072,
        true,
        true,
    ),
    // ----------------------------------------------------------------
    // Open-weights (now available under new aliases)
    // ----------------------------------------------------------------
    (
        "open-mistral-nemo", // Mistral Nemo 12B — still served
        "Mistral Nemo 12B (open weights)",
        131_072,
        false,
        true,
    ),
    // ----------------------------------------------------------------
    // Legacy / deprecated — kept for backward compatibility only.
    // New code should prefer "-latest" aliases above.
    // ----------------------------------------------------------------
    (
        "mistral-small-2506", // Mistral Small 3.2 snapshot
        "Mistral Small 3.2 (2506, legacy)",
        131_072,
        false,
        true,
    ),
    (
        "mistral-large-2411", // Mistral Large 2.1 snapshot (deprecated)
        "Mistral Large 2411 (deprecated)",
        131_072,
        false,
        true,
    ),
    (
        "pixtral-large-2411", // Pixtral Large snapshot (deprecated)
        "Pixtral Large 2411 (deprecated)",
        131_072,
        true,
        true,
    ),
    (
        "pixtral-12b-2409", // Pixtral 12B snapshot (deprecated)
        "Pixtral 12B 2409 (deprecated)",
        131_072,
        true,
        true,
    ),
    (
        "open-mistral-7b",
        "Mistral 7B (open weights, deprecated)",
        32_768,
        false,
        false,
    ),
    (
        "open-mixtral-8x7b",
        "Mixtral 8x7B (open weights, deprecated)",
        32_768,
        false,
        true,
    ),
    (
        "open-mixtral-8x22b",
        "Mixtral 8x22B (open weights, deprecated)",
        65_536,
        false,
        true,
    ),
    (
        "codestral-2501",
        "Codestral 2501 (deprecated)",
        262_144,
        false,
        false,
    ),
    (
        "mistral-small-2501", // Mistral Small 3.0 snapshot (deprecated)
        "Mistral Small 2501 (deprecated)",
        32_768,
        false,
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
    /// Output format for embedding vectors.
    /// - `"float"` (default): standard 32-bit float arrays.
    /// - `"base64"`: base64-encoded arrays (more compact for large batches).
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<&'a str>,
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
// Audio / OCR request-response structs (native implementation)
// ============================================================================

/// Request body for `POST /v1/audio/speech`.
#[derive(Debug, Serialize)]
pub struct MistralSpeechRequest<'a> {
    /// TTS model ID (e.g. `voxtral-mini-tts-latest`).
    pub model: &'a str,
    /// Text prompt to synthesize into speech.
    pub input: &'a str,
    /// Optional voice ID from `/v1/audio/voices`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_id: Option<&'a str>,
    /// Optional base64 reference audio for voice cloning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ref_audio: Option<&'a str>,
    /// Output codec (`pcm`, `wav`, `mp3`, `flac`, `opus`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<&'a str>,
    /// SSE streaming toggle (false by default).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct MistralSpeechResponse {
    /// Base64-encoded audio payload.
    pub audio_data: String,
}

/// Request body for `POST /v1/audio/transcriptions`.
#[derive(Debug, Serialize)]
pub struct MistralTranscriptionRequest<'a> {
    /// Transcription model ID (e.g. `voxtral-mini-transcribe-26-02`).
    pub model: &'a str,
    /// Public URL to the audio file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_url: Option<&'a str>,
    /// File id previously uploaded to `/v1/files`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<&'a str>,
    /// Optional language hint (e.g. `en`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<&'a str>,
    /// SSE streaming toggle.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct MistralTranscriptionResponse {
    pub text: String,
    pub language: Option<String>,
    pub model: Option<String>,
}

/// Request body for `POST /v1/ocr` with document URL.
#[derive(Debug, Serialize)]
pub struct MistralOcrRequest<'a> {
    /// OCR model ID (e.g. `mistral-ocr-latest`).
    pub model: &'a str,
    pub document: MistralOcrDocument<'a>,
}

#[derive(Debug, Serialize)]
pub struct MistralOcrDocument<'a> {
    #[serde(rename = "type")]
    pub doc_type: &'a str,
    #[serde(rename = "document_url")]
    pub document_url: &'a str,
}

#[derive(Debug, Deserialize)]
pub struct MistralOcrResponse {
    pub model: Option<String>,
    #[serde(default)]
    pub pages: Vec<MistralOcrPage>,
}

#[derive(Debug, Deserialize)]
pub struct MistralOcrPage {
    pub index: Option<usize>,
    pub markdown: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MistralVoicesListResponse {
    pub items: Vec<MistralVoice>,
    pub page: Option<usize>,
    pub page_size: Option<usize>,
    pub total: Option<usize>,
    pub total_pages: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct MistralVoice {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Serialize)]
pub struct MistralCreateVoiceRequest<'a> {
    pub name: &'a str,
    /// Base64-encoded audio sample for cloning.
    pub sample_audio: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_filename: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub languages: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gender: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub age: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slug: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retention_notice: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct MistralUpdateVoiceRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gender: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub age: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub languages: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
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

        // Build ProviderConfig for the inner OpenAICompatibleProvider.
        // We inject the literal API key directly into the config so that
        // OpenAICompatibleProvider never needs to call std::env::var (or
        // force us to call std::env::set_var, which is thread-unsafe).
        let config = Self::build_provider_config(&api_key, &model, &embedding_model, &base_url);

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

    /// Text-to-speech via `POST /v1/audio/speech`.
    pub async fn speech(
        &self,
        request: &MistralSpeechRequest<'_>,
    ) -> Result<MistralSpeechResponse> {
        if request.input.trim().is_empty() {
            return Err(LlmError::InvalidRequest(
                "Mistral speech requires non-empty input text".to_string(),
            ));
        }
        if request.voice_id.is_none() && request.ref_audio.is_none() {
            return Err(LlmError::InvalidRequest(
                "Mistral speech requires either voice_id or ref_audio".to_string(),
            ));
        }
        if request.stream == Some(true) {
            return Err(LlmError::InvalidRequest(
                "Use speech_stream_raw() when stream=true".to_string(),
            ));
        }

        let url = format!("{}/audio/speech", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await
            .map_err(|e| {
                LlmError::NetworkError(format!("Failed to call Mistral speech API: {e}"))
            })?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| LlmError::NetworkError(format!("Failed to read speech response: {e}")))?;

        if !status.is_success() {
            return Err(LlmError::ApiError(format!(
                "Mistral speech API error ({status}): {body}"
            )));
        }

        serde_json::from_str(&body)
            .map_err(|e| LlmError::ApiError(format!("Failed to parse speech response: {e}")))
    }

    /// Text-to-speech streaming via `POST /v1/audio/speech` (event-stream body passthrough).
    pub async fn speech_stream_raw(&self, request: &MistralSpeechRequest<'_>) -> Result<String> {
        if request.input.trim().is_empty() {
            return Err(LlmError::InvalidRequest(
                "Mistral speech requires non-empty input text".to_string(),
            ));
        }
        if request.voice_id.is_none() && request.ref_audio.is_none() {
            return Err(LlmError::InvalidRequest(
                "Mistral speech requires either voice_id or ref_audio".to_string(),
            ));
        }
        if request.stream != Some(true) {
            return Err(LlmError::InvalidRequest(
                "speech_stream_raw requires stream=true".to_string(),
            ));
        }

        let url = format!("{}/audio/speech", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(request)
            .send()
            .await
            .map_err(|e| {
                LlmError::NetworkError(format!("Failed to call Mistral speech API: {e}"))
            })?;

        let status = response.status();
        let body = response.text().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to read speech stream response: {e}"))
        })?;

        if !status.is_success() {
            return Err(LlmError::ApiError(format!(
                "Mistral speech stream API error ({status}): {body}"
            )));
        }

        Ok(body)
    }

    /// Audio transcription via `POST /v1/audio/transcriptions`.
    pub async fn transcribe(
        &self,
        request: &MistralTranscriptionRequest<'_>,
    ) -> Result<MistralTranscriptionResponse> {
        if request.stream == Some(true) {
            return Err(LlmError::InvalidRequest(
                "Use transcribe_stream_raw() when stream=true".to_string(),
            ));
        }
        if request.file_url.is_some() == request.file_id.is_some() {
            return Err(LlmError::InvalidRequest(
                "Mistral transcription requires exactly one source: file_url or file_id"
                    .to_string(),
            ));
        }

        let url = format!(
            "{}/audio/transcriptions",
            self.base_url.trim_end_matches('/')
        );

        // Mistral transcription expects multipart/form-data (even when using file_url).
        let mut form = multipart::Form::new().text("model", request.model.to_string());

        if let Some(file_url) = request.file_url {
            form = form.text("file_url", file_url.to_string());
        }
        if let Some(file_id) = request.file_id {
            form = form.text("file_id", file_id.to_string());
        }

        if let Some(language) = request.language {
            form = form.text("language", language.to_string());
        }
        if let Some(stream) = request.stream {
            form = form.text("stream", stream.to_string());
        }

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .multipart(form)
            .send()
            .await
            .map_err(|e| {
                LlmError::NetworkError(format!("Failed to call Mistral transcription API: {e}"))
            })?;

        let status = response.status();
        let body = response.text().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to read transcription response: {e}"))
        })?;

        if !status.is_success() {
            return Err(LlmError::ApiError(format!(
                "Mistral transcription API error ({status}): {body}"
            )));
        }

        serde_json::from_str(&body)
            .map_err(|e| LlmError::ApiError(format!("Failed to parse transcription response: {e}")))
    }

    /// Multipart file-upload transcription via `POST /v1/audio/transcriptions`.
    pub async fn transcribe_file_upload(
        &self,
        model: &str,
        filename: &str,
        bytes: Vec<u8>,
        language: Option<&str>,
    ) -> Result<MistralTranscriptionResponse> {
        if model.trim().is_empty() {
            return Err(LlmError::InvalidRequest(
                "Mistral transcription requires non-empty model".to_string(),
            ));
        }
        if filename.trim().is_empty() {
            return Err(LlmError::InvalidRequest(
                "Mistral transcription requires non-empty filename".to_string(),
            ));
        }
        if bytes.is_empty() {
            return Err(LlmError::InvalidRequest(
                "Mistral transcription file bytes must not be empty".to_string(),
            ));
        }

        let url = format!(
            "{}/audio/transcriptions",
            self.base_url.trim_end_matches('/')
        );
        let file_part = multipart::Part::bytes(bytes).file_name(filename.to_string());
        let mut form = multipart::Form::new()
            .text("model", model.to_string())
            .part("file", file_part);
        if let Some(lang) = language {
            form = form.text("language", lang.to_string());
        }

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .multipart(form)
            .send()
            .await
            .map_err(|e| {
                LlmError::NetworkError(format!("Failed to call Mistral transcription API: {e}"))
            })?;

        let status = response.status();
        let body = response.text().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to read transcription response: {e}"))
        })?;

        if !status.is_success() {
            return Err(LlmError::ApiError(format!(
                "Mistral transcription API error ({status}): {body}"
            )));
        }

        serde_json::from_str(&body)
            .map_err(|e| LlmError::ApiError(format!("Failed to parse transcription response: {e}")))
    }

    /// Audio transcription streaming via `POST /v1/audio/transcriptions`.
    /// Returns the raw event-stream payload.
    pub async fn transcribe_stream_raw(
        &self,
        request: &MistralTranscriptionRequest<'_>,
    ) -> Result<String> {
        if request.stream != Some(true) {
            return Err(LlmError::InvalidRequest(
                "transcribe_stream_raw requires stream=true".to_string(),
            ));
        }
        if request.file_url.is_some() == request.file_id.is_some() {
            return Err(LlmError::InvalidRequest(
                "Mistral transcription requires exactly one source: file_url or file_id"
                    .to_string(),
            ));
        }

        let url = format!(
            "{}/audio/transcriptions",
            self.base_url.trim_end_matches('/')
        );
        let mut form = multipart::Form::new().text("model", request.model.to_string());
        if let Some(file_url) = request.file_url {
            form = form.text("file_url", file_url.to_string());
        }
        if let Some(file_id) = request.file_id {
            form = form.text("file_id", file_id.to_string());
        }
        if let Some(language) = request.language {
            form = form.text("language", language.to_string());
        }
        form = form.text("stream", "true".to_string());

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "text/event-stream")
            .multipart(form)
            .send()
            .await
            .map_err(|e| {
                LlmError::NetworkError(format!("Failed to call Mistral transcription API: {e}"))
            })?;

        let status = response.status();
        let body = response.text().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to read transcription stream response: {e}"))
        })?;

        if !status.is_success() {
            return Err(LlmError::ApiError(format!(
                "Mistral transcription stream API error ({status}): {body}"
            )));
        }

        Ok(body)
    }

    /// OCR processing via `POST /v1/ocr`.
    pub async fn ocr(&self, request: &MistralOcrRequest<'_>) -> Result<MistralOcrResponse> {
        let url = format!("{}/ocr", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(format!("Failed to call Mistral OCR API: {e}")))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| LlmError::NetworkError(format!("Failed to read OCR response: {e}")))?;

        if !status.is_success() {
            return Err(LlmError::ApiError(format!(
                "Mistral OCR API error ({status}): {body}"
            )));
        }

        serde_json::from_str(&body)
            .map_err(|e| LlmError::ApiError(format!("Failed to parse OCR response: {e}")))
    }

    /// List voices via `GET /v1/audio/voices`.
    pub async fn list_audio_voices(&self) -> Result<MistralVoicesListResponse> {
        let url = format!("{}/audio/voices", self.base_url.trim_end_matches('/'));

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(format!("Failed to list Mistral voices: {}", e)))?;

        let status = response.status();
        let body = response.text().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to read voices list response: {}", e))
        })?;

        if !status.is_success() {
            return Err(LlmError::ApiError(format!(
                "Mistral voices list failed ({status}): {body}"
            )));
        }

        serde_json::from_str(&body)
            .map_err(|e| LlmError::ApiError(format!("Failed to parse voices response: {e}")))
    }

    /// Get a voice metadata record by id.
    pub async fn get_audio_voice(&self, voice_id: &str) -> Result<MistralVoice> {
        if voice_id.trim().is_empty() {
            return Err(LlmError::InvalidRequest(
                "voice_id must not be empty".to_string(),
            ));
        }

        let url = format!(
            "{}/audio/voices/{}",
            self.base_url.trim_end_matches('/'),
            voice_id
        );
        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(format!("Failed to get Mistral voice: {}", e)))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| LlmError::NetworkError(format!("Failed to read voice response: {}", e)))?;

        if !status.is_success() {
            return Err(LlmError::ApiError(format!(
                "Mistral get voice failed ({status}): {body}"
            )));
        }

        serde_json::from_str(&body)
            .map_err(|e| LlmError::ApiError(format!("Failed to parse voice response: {e}")))
    }

    /// Get a base64 voice sample by id.
    pub async fn get_audio_voice_sample(&self, voice_id: &str) -> Result<String> {
        if voice_id.trim().is_empty() {
            return Err(LlmError::InvalidRequest(
                "voice_id must not be empty".to_string(),
            ));
        }

        let url = format!(
            "{}/audio/voices/{}/sample",
            self.base_url.trim_end_matches('/'),
            voice_id
        );
        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| {
                LlmError::NetworkError(format!("Failed to get Mistral voice sample: {}", e))
            })?;

        let status = response.status();
        let body = response.text().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to read voice sample response: {}", e))
        })?;

        if !status.is_success() {
            return Err(LlmError::ApiError(format!(
                "Mistral get voice sample failed ({status}): {body}"
            )));
        }

        // Some deployments return a JSON string (`"..."`), others return
        // raw base64 text. Accept both formats for compatibility.
        if let Ok(sample) = serde_json::from_str::<String>(&body) {
            return Ok(sample);
        }

        let trimmed = body.trim().to_string();
        if trimmed.is_empty() {
            return Err(LlmError::ApiError(
                "Voice sample response was empty".to_string(),
            ));
        }
        Ok(trimmed)
    }

    /// Create a new voice with a base64 encoded sample.
    pub async fn create_audio_voice(
        &self,
        request: &MistralCreateVoiceRequest<'_>,
    ) -> Result<MistralVoice> {
        if request.name.trim().is_empty() {
            return Err(LlmError::InvalidRequest(
                "Mistral voice create requires non-empty name".to_string(),
            ));
        }
        if request.sample_audio.trim().is_empty() {
            return Err(LlmError::InvalidRequest(
                "Mistral voice create requires non-empty sample_audio".to_string(),
            ));
        }

        let url = format!("{}/audio/voices", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await
            .map_err(|e| {
                LlmError::NetworkError(format!("Failed to create Mistral voice metadata: {}", e))
            })?;

        let status = response.status();
        let body = response.text().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to read voice create response: {}", e))
        })?;

        if !status.is_success() {
            return Err(LlmError::ApiError(format!(
                "Mistral create voice failed ({status}): {body}"
            )));
        }

        serde_json::from_str(&body)
            .map_err(|e| LlmError::ApiError(format!("Failed to parse create voice response: {e}")))
    }

    /// Update a voice metadata record.
    pub async fn update_audio_voice(
        &self,
        voice_id: &str,
        request: &MistralUpdateVoiceRequest<'_>,
    ) -> Result<MistralVoice> {
        if voice_id.trim().is_empty() {
            return Err(LlmError::InvalidRequest(
                "voice_id must not be empty".to_string(),
            ));
        }

        let url = format!(
            "{}/audio/voices/{}",
            self.base_url.trim_end_matches('/'),
            voice_id
        );
        let response = self
            .client
            .patch(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await
            .map_err(|e| {
                LlmError::NetworkError(format!("Failed to update Mistral voice metadata: {}", e))
            })?;

        let status = response.status();
        let body = response.text().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to read voice update response: {}", e))
        })?;

        if !status.is_success() {
            return Err(LlmError::ApiError(format!(
                "Mistral update voice failed ({status}): {body}"
            )));
        }

        serde_json::from_str(&body)
            .map_err(|e| LlmError::ApiError(format!("Failed to parse update voice response: {e}")))
    }

    /// Delete a voice record.
    pub async fn delete_audio_voice(&self, voice_id: &str) -> Result<MistralVoice> {
        if voice_id.trim().is_empty() {
            return Err(LlmError::InvalidRequest(
                "voice_id must not be empty".to_string(),
            ));
        }

        let url = format!(
            "{}/audio/voices/{}",
            self.base_url.trim_end_matches('/'),
            voice_id
        );
        let response = self
            .client
            .delete(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| {
                LlmError::NetworkError(format!("Failed to delete Mistral voice: {}", e))
            })?;

        let status = response.status();
        let body = response.text().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to read delete voice response: {}", e))
        })?;

        if !status.is_success() {
            return Err(LlmError::ApiError(format!(
                "Mistral delete voice failed ({status}): {body}"
            )));
        }

        serde_json::from_str(&body)
            .map_err(|e| LlmError::ApiError(format!("Failed to parse delete voice response: {e}")))
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Construct the [`ProviderConfig`] used by the inner [`OpenAICompatibleProvider`].
    fn build_provider_config(
        api_key: &str,
        model: &str,
        embedding_model: &str,
        base_url: &str,
    ) -> ProviderConfig {
        let models: Vec<ModelCard> = MISTRAL_CHAT_MODELS
            .iter()
            .map(|(id, display, ctx, vision, fc)| {
                // Frontier large/medium models support up to 16 384 output tokens;
                // smaller / legacy models default to 4 096.
                let max_output = if id.contains("large") || id.contains("medium") {
                    MISTRAL_FRONTIER_MAX_OUTPUT_TOKENS
                } else {
                    MISTRAL_DEFAULT_MAX_OUTPUT_TOKENS
                };
                ModelCard {
                    name: id.to_string(),
                    display_name: display.to_string(),
                    model_type: ModelType::Llm,
                    capabilities: ModelCapabilities {
                        context_length: *ctx,
                        max_output_tokens: max_output,
                        supports_vision: *vision,
                        supports_function_calling: *fc,
                        supports_json_mode: true,
                        supports_streaming: true,
                        supports_system_message: true,
                        ..Default::default()
                    },
                    ..Default::default()
                }
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
            // Inject the literal key so OpenAICompatibleProvider does not need
            // to read from the environment (avoids std::env::set_var races).
            api_key: Some(api_key.to_string()),
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

    #[allow(clippy::misnamed_getters)]
    fn model(&self) -> &str {
        &self.embedding_model
    }

    fn dimension(&self) -> usize {
        // Only `mistral-embed` is currently offered, which has 1024 dimensions.
        // Return the standard dimension regardless of the configured model name.
        MISTRAL_EMBED_DIMENSION
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
            // Use "float" explicitly so that the response always contains a
            // `Vec<f32>` rather than a base64-encoded blob.  This keeps the
            // deserialization path simple and avoids a client-side decode step.
            encoding_format: Some("float"),
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
        // Current frontier aliases (April 2026)
        assert!(
            ids.contains(&"mistral-large-latest"),
            "Should contain mistral-large-latest"
        );
        assert!(
            ids.contains(&"mistral-medium-latest"),
            "Should contain mistral-medium-latest"
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
            ids.contains(&"devstral-latest"),
            "Should contain devstral-latest"
        );
        // Ministral edge models
        assert!(
            ids.contains(&"ministral-3b-latest"),
            "Should contain ministral-3b-latest"
        );
        assert!(
            ids.contains(&"ministral-8b-latest"),
            "Should contain ministral-8b-latest"
        );
        assert!(
            ids.contains(&"ministral-14b-latest"),
            "Should contain ministral-14b-latest"
        );
        // Reasoning models
        assert!(
            ids.contains(&"magistral-medium-latest"),
            "Should contain magistral-medium-latest"
        );
        assert!(
            ids.contains(&"magistral-small-latest"),
            "Should contain magistral-small-latest"
        );
    }

    #[test]
    fn test_context_length_known_models() {
        // Frontier models — 256 K (262_144)
        assert_eq!(
            MistralProvider::context_length("mistral-large-latest"),
            262_144
        );
        assert_eq!(
            MistralProvider::context_length("mistral-small-latest"),
            262_144
        );
        assert_eq!(MistralProvider::context_length("codestral-latest"), 262_144);
        // Medium — 128 K (131_072)
        assert_eq!(
            MistralProvider::context_length("mistral-medium-latest"),
            131_072
        );
        assert_eq!(
            MistralProvider::context_length("open-mistral-nemo"),
            131_072
        );
        // Ministral edge models — 128 K
        assert_eq!(
            MistralProvider::context_length("ministral-3b-latest"),
            131_072
        );
        assert_eq!(
            MistralProvider::context_length("ministral-8b-latest"),
            131_072
        );
        assert_eq!(
            MistralProvider::context_length("ministral-14b-latest"),
            131_072
        );
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
        assert_eq!(p.max_context_length(), 262_144); // Mistral Large 3 = 256 K
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
            encoding_format: Some("float"),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "mistral-embed");
        assert_eq!(json["input"][0], "hello world");
        assert_eq!(json["input"][1], "foo bar");
        assert_eq!(json["encoding_format"], "float");
    }

    #[test]
    fn test_embedding_request_serialization_no_encoding_format() {
        let texts = vec!["hello".to_string()];
        let req = EmbeddingRequest {
            model: "mistral-embed",
            input: &texts,
            encoding_format: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        // `encoding_format` must be absent when None (skip_serializing_if)
        assert!(
            json.get("encoding_format").is_none(),
            "encoding_format should be absent when None"
        );
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
        // Mistral Large 3 has vision support in latest catalog
        let cfg = MistralProvider::build_provider_config(
            "key",
            "mistral-large-latest",
            "mistral-embed",
            MISTRAL_BASE_URL,
        );
        let large = cfg
            .models
            .iter()
            .find(|m| m.name == "mistral-large-latest")
            .unwrap();
        assert!(large.capabilities.supports_vision);
        assert!(large.capabilities.supports_function_calling);
    }

    #[test]
    fn test_ministral_models_in_catalog() {
        let cfg = MistralProvider::build_provider_config(
            "key",
            "mistral-small-latest",
            "mistral-embed",
            MISTRAL_BASE_URL,
        );
        for id in &[
            "ministral-3b-latest",
            "ministral-8b-latest",
            "ministral-14b-latest",
        ] {
            let card = cfg.models.iter().find(|m| m.name == *id);
            assert!(card.is_some(), "Missing model: {id}");
            let card = card.unwrap();
            assert_eq!(
                card.capabilities.context_length, 131_072,
                "Wrong context for {id}"
            );
            assert!(
                card.capabilities.supports_vision,
                "{id} should support vision"
            );
            assert!(
                card.capabilities.supports_function_calling,
                "{id} should support FC"
            );
        }
    }

    #[test]
    fn test_reasoning_models_in_catalog() {
        let cfg = MistralProvider::build_provider_config(
            "key",
            "magistral-medium-latest",
            "mistral-embed",
            MISTRAL_BASE_URL,
        );
        for id in &["magistral-medium-latest", "magistral-small-latest"] {
            let card = cfg.models.iter().find(|m| m.name == *id);
            assert!(card.is_some(), "Missing reasoning model: {id}");
            let card = card.unwrap();
            assert_eq!(
                card.capabilities.context_length, 131_072,
                "Wrong context for {id}"
            );
            assert!(
                card.capabilities.supports_function_calling,
                "{id} should support FC"
            );
        }
    }

    #[test]
    fn test_frontier_models_have_256k_context() {
        for id in &[
            "mistral-large-latest",
            "mistral-small-latest",
            "codestral-latest",
        ] {
            assert_eq!(
                MistralProvider::context_length(id),
                262_144,
                "Expected 256K context for {id}"
            );
        }
    }

    #[test]
    fn test_provider_config_sets_api_key_directly() {
        // Ensure the literal api_key field is set so OpenAICompatibleProvider
        // does not need to mutate the environment (thread-safety guarantee).
        let cfg = MistralProvider::build_provider_config(
            "secret-api-key",
            "mistral-small-latest",
            "mistral-embed",
            MISTRAL_BASE_URL,
        );
        assert_eq!(cfg.api_key.as_deref(), Some("secret-api-key"));
    }

    #[test]
    fn test_new_does_not_require_env_var() {
        // MistralProvider::new() must not need MISTRAL_API_KEY in the env.
        std::env::remove_var("MISTRAL_API_KEY");
        let result = MistralProvider::new(
            "explicit-key".to_string(),
            MISTRAL_DEFAULT_MODEL.to_string(),
            MISTRAL_DEFAULT_EMBEDDING_MODEL.to_string(),
            None,
        );
        assert!(
            result.is_ok(),
            "MistralProvider::new() should succeed without env var when key is provided directly"
        );
    }

    #[test]
    fn test_frontier_max_output_tokens() {
        let cfg = MistralProvider::build_provider_config(
            "key",
            "mistral-large-latest",
            "mistral-embed",
            MISTRAL_BASE_URL,
        );
        let large = cfg
            .models
            .iter()
            .find(|m| m.name == "mistral-large-latest")
            .unwrap();
        assert_eq!(
            large.capabilities.max_output_tokens, MISTRAL_FRONTIER_MAX_OUTPUT_TOKENS,
            "Frontier large model should have 16 384 output tokens"
        );
        let medium = cfg
            .models
            .iter()
            .find(|m| m.name == "mistral-medium-latest")
            .unwrap();
        assert_eq!(
            medium.capabilities.max_output_tokens, MISTRAL_FRONTIER_MAX_OUTPUT_TOKENS,
            "Frontier medium model should have 16 384 output tokens"
        );
    }

    #[test]
    fn test_speech_request_serialization() {
        let req = MistralSpeechRequest {
            model: "voxtral-mini-tts-latest",
            input: "hello world",
            voice_id: Some("voice_123"),
            ref_audio: None,
            response_format: Some("mp3"),
            stream: Some(false),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "voxtral-mini-tts-latest");
        assert_eq!(json["input"], "hello world");
        assert_eq!(json["voice_id"], "voice_123");
        assert_eq!(json["response_format"], "mp3");
    }

    #[test]
    fn test_transcription_request_serialization() {
        let req = MistralTranscriptionRequest {
            model: "voxtral-mini-transcribe-26-02",
            file_url: Some("https://example.com/sample.wav"),
            file_id: None,
            language: Some("en"),
            stream: Some(false),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "voxtral-mini-transcribe-26-02");
        assert_eq!(json["file_url"], "https://example.com/sample.wav");
        assert!(json.get("file_id").is_none());
        assert_eq!(json["language"], "en");
    }

    #[test]
    fn test_transcription_request_serialization_with_file_id() {
        let req = MistralTranscriptionRequest {
            model: "voxtral-mini-transcribe-2507",
            file_url: None,
            file_id: Some("file-123"),
            language: None,
            stream: Some(false),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["file_id"], "file-123");
        assert!(json.get("file_url").is_none());
    }

    #[test]
    fn test_ocr_request_serialization() {
        let req = MistralOcrRequest {
            model: "mistral-ocr-latest",
            document: MistralOcrDocument {
                doc_type: "document_url",
                document_url: "https://example.com/file.pdf",
            },
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "mistral-ocr-latest");
        assert_eq!(json["document"]["type"], "document_url");
        assert_eq!(
            json["document"]["document_url"],
            "https://example.com/file.pdf"
        );
    }

    #[test]
    fn test_create_voice_request_serialization() {
        let req = MistralCreateVoiceRequest {
            name: "Edgequake Voice",
            sample_audio: "dGVzdA==",
            sample_filename: Some("sample.wav"),
            languages: Some(vec!["en".to_string()]),
            gender: Some("female"),
            age: Some(30),
            color: None,
            slug: None,
            tags: Some(vec!["ci".to_string()]),
            retention_notice: Some(30),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["name"], "Edgequake Voice");
        assert_eq!(json["sample_audio"], "dGVzdA==");
        assert_eq!(json["sample_filename"], "sample.wav");
        assert_eq!(json["languages"][0], "en");
    }

    #[test]
    fn test_speech_requires_voice_or_ref_audio_validation() {
        let req = MistralSpeechRequest {
            model: "voxtral-mini-tts-latest",
            input: "hello",
            voice_id: None,
            ref_audio: None,
            response_format: Some("mp3"),
            stream: Some(false),
        };
        std::env::set_var("MISTRAL_API_KEY", "test-key");
        let p = MistralProvider::new(
            "test-key".to_string(),
            MISTRAL_DEFAULT_MODEL.to_string(),
            MISTRAL_DEFAULT_EMBEDDING_MODEL.to_string(),
            None,
        )
        .unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let err = rt.block_on(async { p.speech(&req).await.unwrap_err() });
        assert!(matches!(err, LlmError::InvalidRequest(_)));
        std::env::remove_var("MISTRAL_API_KEY");
    }

    #[test]
    fn test_transcribe_requires_exactly_one_source_validation() {
        std::env::set_var("MISTRAL_API_KEY", "test-key");
        let p = MistralProvider::new(
            "test-key".to_string(),
            MISTRAL_DEFAULT_MODEL.to_string(),
            MISTRAL_DEFAULT_EMBEDDING_MODEL.to_string(),
            None,
        )
        .unwrap();
        let req_none = MistralTranscriptionRequest {
            model: "voxtral-mini-transcribe-2507",
            file_url: None,
            file_id: None,
            language: None,
            stream: Some(false),
        };
        let rt = tokio::runtime::Runtime::new().unwrap();
        let err_none = rt.block_on(async { p.transcribe(&req_none).await.unwrap_err() });
        assert!(matches!(err_none, LlmError::InvalidRequest(_)));

        let req_both = MistralTranscriptionRequest {
            model: "voxtral-mini-transcribe-2507",
            file_url: Some("https://example.com/a.wav"),
            file_id: Some("file-1"),
            language: None,
            stream: Some(false),
        };
        let err_both = rt.block_on(async { p.transcribe(&req_both).await.unwrap_err() });
        assert!(matches!(err_both, LlmError::InvalidRequest(_)));
        std::env::remove_var("MISTRAL_API_KEY");
    }
}
