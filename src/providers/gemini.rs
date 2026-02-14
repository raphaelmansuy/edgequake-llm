//! Gemini LLM provider implementation.
//!
//! Supports both Google AI Gemini API and VertexAI endpoints.
//!
//! # Environment Variables
//! - `GEMINI_API_KEY`: API key for Google AI Gemini API
//! - `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account JSON for VertexAI
//! - `GOOGLE_CLOUD_PROJECT`: GCP project ID for VertexAI
//! - `GOOGLE_CLOUD_REGION`: GCP region for VertexAI (default: us-central1)

use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, instrument};

use crate::error::{LlmError, Result};
use crate::traits::{
    ChatMessage, ChatRole, CompletionOptions, EmbeddingProvider, LLMProvider, LLMResponse,
    StreamChunk, ToolCall, ToolChoice, ToolDefinition, // OODA-06/07/08: Tool + streaming support
};

/// Gemini API endpoints
const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Default models
// WHY: gemini-2.5-flash is the stable production model as of Jan 2026
// gemini-3-flash exists in docs but may not be available for all API keys
// See: https://ai.google.dev/gemini-api/docs/models
const DEFAULT_GEMINI_MODEL: &str = "gemini-2.5-flash";
const DEFAULT_EMBEDDING_MODEL: &str = "text-embedding-004";

/// Gemini provider configuration
#[derive(Debug, Clone)]
pub enum GeminiEndpoint {
    /// Google AI Gemini API (uses API key)
    GoogleAI { api_key: String },
    /// VertexAI endpoint (uses OAuth2/service account)
    VertexAI {
        project_id: String,
        region: String,
        access_token: String,
    },
}

/// Cache state for interior mutability.
#[derive(Debug, Default)]
struct CacheState {
    content_id: Option<String>,
    system_hash: Option<u64>,
}

/// Gemini LLM provider.
#[derive(Debug)]
pub struct GeminiProvider {
    client: Client,
    endpoint: GeminiEndpoint,
    model: String,
    embedding_model: String,
    max_context_length: usize,
    embedding_dimension: usize,
    /// Cache TTL (time-to-live)
    cache_ttl: String,
    /// Cache state with interior mutability for Arc compatibility
    cache_state: tokio::sync::RwLock<CacheState>,
}

// ============================================================================
// Gemini API Request/Response Types
// ============================================================================

// ============================================================================
// Image Support Types (OODA-54)
// ============================================================================
//
// Gemini uses inline_data for images:
//
// ┌─────────────────────────┐
// │ parts: [                │
// │   { text: "..." },      │
// │   { inlineData: {       │
// │       mimeType: "...",  │
// │       data: "base64..." │
// │     }                   │
// │   }                     │
// │ ]                       │
// └─────────────────────────┘
//
// WHY: Blob struct matches Gemini API Blob type exactly
// ============================================================================

/// Blob for inline media data (images, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Blob {
    pub mime_type: String,  // MIME type, e.g., "image/png"
    pub data: String,       // Base64-encoded data
}

/// Content part for Gemini API (text, inline data, or function call/response)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inline_data: Option<Blob>,
    // OODA-06: Function calling support
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_response: Option<FunctionResponse>,
    // OODA-25: Thinking support - indicates if this part is a thought summary
    // When true, this part contains thinking/reasoning content, not final response
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought: Option<bool>,
}

// ============================================================================
// Function Calling Types (OODA-06)
// ============================================================================
//
// Gemini function calling API structure:
//
// ┌──────────────────────────────────────────────────────────────┐
// │ Request:                                                      │
// │   tools: [{ functionDeclarations: [{ name, description }]}]   │
// │   toolConfig: { functionCallingConfig: { mode: "AUTO" }}      │
// │                                                               │
// │ Response:                                                     │
// │   parts: [{ functionCall: { name, args }}]                    │
// └──────────────────────────────────────────────────────────────┘

/// Function declaration for Gemini tool calling
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Tool wrapper containing function declarations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiTool {
    pub function_declarations: Vec<FunctionDeclaration>,
}

/// Function calling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {
    pub mode: String, // AUTO, ANY, NONE
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

/// Tool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    pub function_calling_config: FunctionCallingConfig,
}

/// Function call in response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCall {
    pub name: String,
    pub args: serde_json::Value,
}

/// Function response for tool results
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionResponse {
    pub name: String,
    pub response: serde_json::Value,
}

/// Content for Gemini API
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Content {
    pub parts: Vec<Part>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
}

/// Generation config for Gemini API
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
}

// ============================================================================
// OODA-25: Thinking Configuration Types
// ============================================================================
//
// Gemini 2.5 and 3.x models support "thinking" - internal reasoning before
// responding. This config controls thinking behavior:
//
// ┌───────────────────────────────────────────────────────────────────────────┐
// │ include_thoughts: true  → Response includes thought summaries            │
// │ thinking_level: "high"  → Maximum reasoning depth (Gemini 3)             │
// │ thinking_budget: 1024   → Token budget for thinking (Gemini 2.5)         │
// └───────────────────────────────────────────────────────────────────────────┘
//
// Reference: https://ai.google.dev/gemini-api/docs/thinking
// ============================================================================

/// Thinking configuration for Gemini 2.5+/3.x models
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingConfig {
    /// Include thought summaries in response (shows model's reasoning)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_thoughts: Option<bool>,
    /// Thinking level for Gemini 3 models: "minimal", "low", "medium", "high"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<String>,
    /// Token budget for thinking (Gemini 2.5): 0 to 24576, -1 for dynamic
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<i32>,
}

/// Safety setting for Gemini API
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SafetySetting {
    pub category: String,
    pub threshold: String,
}

/// Request to create cached content
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CreateCachedContentRequest {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    contents: Option<Vec<Content>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
    ttl: String,
}

/// Response from cachedContents.create
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CachedContentResponse {
    name: String,
    /// Usage metadata for the cached content.
    /// Currently unused but preserved for future cache token tracking.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[allow(dead_code)]
    usage_metadata: Option<UsageMetadata>,
}

/// Request body for generateContent
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerateContentRequest {
    contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safety_settings: Option<Vec<SafetySetting>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cached_content: Option<String>,
    // OODA-06: Tool support
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<ToolConfig>,
    // OODA-25: Thinking config for Gemini 2.5+/3.x models
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<ThinkingConfig>,
}

/// Candidate from Gemini response
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Candidate {
    content: Content,
    finish_reason: Option<String>,
    #[serde(default)]
    safety_ratings: Vec<serde_json::Value>,
}

/// Usage metadata from Gemini response
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct UsageMetadata {
    #[serde(default)]
    prompt_token_count: usize,
    #[serde(default)]
    candidates_token_count: usize,
    #[serde(default)]
    total_token_count: usize,
    #[serde(default)]
    cached_content_token_count: usize,
    // OODA-25: Track thinking tokens used by Gemini 2.5+/3.x
    #[serde(default)]
    thoughts_token_count: usize,
}

/// Response from generateContent
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerateContentResponse {
    candidates: Option<Vec<Candidate>>,
    usage_metadata: Option<UsageMetadata>,
}

/// Request body for embedContent
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct EmbedContentRequest {
    content: Content,
    /// WHY: Required for batchEmbedContents - each request must specify the model
    /// Format: "models/{model}" e.g. "models/text-embedding-004"
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimensionality: Option<usize>,
}

/// Request body for batchEmbedContents
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct BatchEmbedContentsRequest {
    requests: Vec<EmbedContentRequest>,
}

/// Embedding values from Gemini response
#[derive(Debug, Clone, Deserialize)]
struct EmbeddingValues {
    values: Vec<f32>,
}

/// Response from embedContent
#[derive(Debug, Clone, Deserialize)]
struct EmbedContentResponse {
    embedding: EmbeddingValues,
}

/// Response from batchEmbedContents
#[derive(Debug, Clone, Deserialize)]
struct BatchEmbedContentsResponse {
    embeddings: Vec<EmbeddingValues>,
}

/// Error response from Gemini API
#[derive(Debug, Clone, Deserialize)]
struct GeminiErrorResponse {
    error: GeminiError,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct GeminiError {
    code: i32,
    message: String,
    status: String,
}

/// OODA-40: Response structure for Gemini /models endpoint
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiModelsResponse {
    #[serde(default)]
    pub models: Vec<GeminiModelInfo>,
}

/// OODA-40: Individual model info from Gemini API
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiModelInfo {
    /// Full model name (e.g., "models/gemini-2.5-flash")
    pub name: String,
    /// Display name (e.g., "Gemini 2.5 Flash")
    #[serde(default)]
    pub display_name: String,
    /// Description of the model
    #[serde(default)]
    pub description: String,
    /// Max input tokens
    #[serde(default)]
    pub input_token_limit: Option<u32>,
    /// Max output tokens
    #[serde(default)]
    pub output_token_limit: Option<u32>,
    /// Supported generation methods
    #[serde(default)]
    pub supported_generation_methods: Vec<String>,
}

// ============================================================================
// GeminiProvider Implementation
// ============================================================================

impl GeminiProvider {
    /// Create a new Gemini provider using Google AI API key.
    ///
    /// # Arguments
    /// * `api_key` - Google AI API key (from <https://aistudio.google.com/app/apikey>)
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            endpoint: GeminiEndpoint::GoogleAI {
                api_key: api_key.into(),
            },
            model: DEFAULT_GEMINI_MODEL.to_string(),
            embedding_model: DEFAULT_EMBEDDING_MODEL.to_string(),
            max_context_length: 2_000_000, // Gemini 3 supports up to 2M tokens
            embedding_dimension: 768,      // text-embedding-004 default
            cache_ttl: "3600s".to_string(),
            cache_state: tokio::sync::RwLock::new(CacheState::default()),
        }
    }

    /// Create a provider from environment variables.
    ///
    /// Checks for `GEMINI_API_KEY` first, then falls back to VertexAI credentials.
    pub fn from_env() -> Result<Self> {
        // Try Google AI API key first
        if let Ok(api_key) = std::env::var("GEMINI_API_KEY") {
            return Ok(Self::new(api_key));
        }

        // Try VertexAI credentials
        Self::from_env_vertex_ai()
    }

    /// Create a VertexAI provider from environment variables.
    ///
    /// OODA-95: This method is specifically for VertexAI endpoint usage.
    /// It will auto-obtain the access token via `gcloud auth print-access-token`
    /// if `GOOGLE_ACCESS_TOKEN` is not set.
    ///
    /// # Environment Variables
    /// - `GOOGLE_CLOUD_PROJECT`: Required. GCP project ID.
    /// - `GOOGLE_CLOUD_REGION`: Optional. GCP region (default: us-central1).
    /// - `GOOGLE_ACCESS_TOKEN`: Optional. If not set, obtained via gcloud CLI.
    pub fn from_env_vertex_ai() -> Result<Self> {
        let project_id = std::env::var("GOOGLE_CLOUD_PROJECT").map_err(|_| {
            LlmError::ConfigError(
                "VertexAI requires GOOGLE_CLOUD_PROJECT environment variable. \
                 Run: export GOOGLE_CLOUD_PROJECT=your-project-id".to_string()
            )
        })?;

        let region = std::env::var("GOOGLE_CLOUD_REGION")
            .unwrap_or_else(|_| "us-central1".to_string());

        // Try to get access token from env, or obtain via gcloud CLI
        let access_token = match std::env::var("GOOGLE_ACCESS_TOKEN") {
            Ok(token) if !token.is_empty() => token,
            _ => Self::get_access_token_from_gcloud()?,
        };

        Ok(Self::vertex_ai(project_id, region, access_token))
    }

    /// Get access token from gcloud CLI.
    ///
    /// OODA-95: Runs `gcloud auth print-access-token` to obtain OAuth2 token.
    fn get_access_token_from_gcloud() -> Result<String> {
        use std::process::Command;

        debug!("Obtaining access token via gcloud auth print-access-token");

        let output = Command::new("gcloud")
            .args(["auth", "print-access-token"])
            .output()
            .map_err(|e| {
                LlmError::ConfigError(format!(
                    "Failed to run 'gcloud auth print-access-token': {}. \
                     Make sure gcloud CLI is installed and you're authenticated. \
                     Run: gcloud auth login",
                    e
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(LlmError::ConfigError(format!(
                "gcloud auth print-access-token failed: {}. \
                 Run: gcloud auth login",
                stderr.trim()
            )));
        }

        let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if token.is_empty() {
            return Err(LlmError::ConfigError(
                "gcloud auth print-access-token returned empty token. \
                 Run: gcloud auth login".to_string()
            ));
        }

        Ok(token)
    }

    /// Create a VertexAI provider.
    ///
    /// # Arguments
    /// * `project_id` - GCP project ID
    /// * `region` - GCP region (e.g., "us-central1")
    /// * `access_token` - OAuth2 access token
    pub fn vertex_ai(
        project_id: impl Into<String>,
        region: impl Into<String>,
        access_token: impl Into<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            endpoint: GeminiEndpoint::VertexAI {
                project_id: project_id.into(),
                region: region.into(),
                access_token: access_token.into(),
            },
            model: DEFAULT_GEMINI_MODEL.to_string(),
            embedding_model: DEFAULT_EMBEDDING_MODEL.to_string(),
            max_context_length: 1_000_000,
            embedding_dimension: 768,
            cache_ttl: "3600s".to_string(),
            cache_state: tokio::sync::RwLock::new(CacheState::default()),
        }
    }

    /// Configure cache TTL (time-to-live).
    ///
    /// Default: "3600s" (1 hour). Format: duration with 's' suffix (e.g., "7200s" for 2 hours).
    pub fn with_cache_ttl(mut self, ttl: impl Into<String>) -> Self {
        self.cache_ttl = ttl.into();
        self
    }

    /// Set the model to use.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        let model_name = model.into();
        self.max_context_length = Self::context_length_for_model(&model_name);
        self.model = model_name;
        self
    }

    /// Set the embedding model to use.
    pub fn with_embedding_model(mut self, model: impl Into<String>) -> Self {
        let model_name = model.into();
        self.embedding_dimension = Self::dimension_for_model(&model_name);
        self.embedding_model = model_name;
        self
    }

    /// Get context length for a given model.
    pub fn context_length_for_model(model: &str) -> usize {
        match model {
            // Gemini 3 series (2026 models)
            m if m.contains("gemini-3-pro") => 2_000_000,
            m if m.contains("gemini-3-flash") => 2_000_000,

            // Gemini 2.5 series
            m if m.contains("gemini-2.5-pro") => 1_000_000,
            m if m.contains("gemini-2.5-flash") => 1_000_000,

            // Gemini 2.0 series
            m if m.contains("gemini-2.0") => 1_000_000,

            // Gemini 1.5 series
            m if m.contains("gemini-1.5-pro") => 2_000_000,
            m if m.contains("gemini-1.5-flash") => 1_000_000,

            // Gemini 1.0 series
            m if m.contains("gemini-1.0") => 32_000,

            _ => 1_000_000, // Updated default
        }
    }

    /// Get embedding dimension for a given model.
    pub fn dimension_for_model(model: &str) -> usize {
        match model {
            m if m.contains("text-embedding-004") => 768,
            m if m.contains("text-embedding-005") => 768,
            m if m.contains("embedding-001") => 768,
            m if m.contains("text-multilingual-embedding-002") => 768,
            _ => 768, // Default dimension
        }
    }

    /// List available Gemini models via the API.
    ///
    /// # OODA-40: Dynamic Model Discovery
    ///
    /// Fetches the list of available models from the Gemini API.
    /// This enables dynamic model selection instead of relying on a static registry.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let provider = GeminiProvider::from_env()?;
    /// let models = provider.list_models().await?;
    /// for model in models.models {
    ///     println!("Available: {} ({})", model.name, model.display_name);
    /// }
    /// ```
    pub async fn list_models(&self) -> Result<GeminiModelsResponse> {
        let url = match &self.endpoint {
            GeminiEndpoint::GoogleAI { api_key } => {
                format!("{}/models?key={}", GEMINI_API_BASE, api_key)
            }
            GeminiEndpoint::VertexAI {
                project_id,
                region,
                access_token: _,
            } => {
                format!(
                    "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models",
                    region, project_id, region
                )
            }
        };

        debug!(url = %url, "Fetching Gemini models list");

        let mut req = self.client.get(&url);

        // Add auth header for VertexAI
        if let GeminiEndpoint::VertexAI { access_token, .. } = &self.endpoint {
            req = req.bearer_auth(access_token);
        }

        let response = req.send().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to fetch Gemini models: {}", e))
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "Gemini /models returned {}: {}",
                status, body
            )));
        }

        response.json::<GeminiModelsResponse>().await.map_err(|e| {
            LlmError::ProviderError(format!("Failed to parse models response: {}", e))
        })
    }

    /// Create or reuse cached content for system instruction.
    ///
    /// Returns cached_content ID (e.g., "cachedContents/abc123").
    /// Creates new cache if:
    /// - No cache exists (cached_content_id is None)
    /// - System instruction changed (cached_system_hash mismatch)
    ///
    /// Reuses existing cache otherwise.
    #[instrument(skip(self, system_instruction))]
    async fn ensure_cache(&self, system_instruction: &Content) -> Result<String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Hash system instruction to detect changes
        let mut hasher = DefaultHasher::new();
        serde_json::to_string(system_instruction)
            .map_err(LlmError::SerializationError)?
            .hash(&mut hasher);
        let current_hash = hasher.finish();

        // Check if we can reuse existing cache (read lock)
        {
            let cache = self.cache_state.read().await;
            if let (Some(cache_id), Some(cached_hash)) = (&cache.content_id, cache.system_hash) {
                if cached_hash == current_hash {
                    debug!("Reusing cached content: {}", cache_id);
                    return Ok(cache_id.clone());
                } else {
                    debug!("System instruction changed, creating new cache");
                }
            }
        }

        // Create new cache
        debug!("Creating cached content (ttl: {})", self.cache_ttl);

        let request = CreateCachedContentRequest {
            model: format!("models/{}", self.model),
            contents: None,
            system_instruction: Some(system_instruction.clone()),
            tools: None,
            ttl: self.cache_ttl.clone(),
        };

        let url = match &self.endpoint {
            GeminiEndpoint::GoogleAI { api_key } => {
                format!("{}/cachedContents?key={}", GEMINI_API_BASE, api_key)
            }
            GeminiEndpoint::VertexAI {
                project_id, region, ..
            } => {
                format!(
                    "https://{}-aiplatform.googleapis.com/v1beta/projects/{}/locations/{}/cachedContents",
                    region, project_id, region
                )
            }
        };

        let mut req = self.client.post(&url).json(&request);

        // Add auth header for VertexAI
        if let GeminiEndpoint::VertexAI { access_token, .. } = &self.endpoint {
            req = req.bearer_auth(access_token);
        }

        let response = req
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError(format!(
                "Failed to create cached content (status {}): {}",
                status, error_text
            )));
        }

        let cache_response: CachedContentResponse = response.json().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to parse cache response: {}", e))
        })?;

        // Store cache ID and hash (write lock)
        {
            let mut cache = self.cache_state.write().await;
            cache.content_id = Some(cache_response.name.clone());
            cache.system_hash = Some(current_hash);
        }

        debug!("Created cached content: {}", cache_response.name);
        Ok(cache_response.name)
    }

    /// Build the URL for a Gemini API endpoint.
    fn build_url(&self, model: &str, action: &str) -> String {
        // Strip provider prefix from model name (e.g., "vertexai:gemini-2.5-flash" -> "gemini-2.5-flash")
        // This allows the same model ID to work for both GoogleAI and VertexAI endpoints
        let model_without_prefix = model
            .strip_prefix("vertexai:")
            .or_else(|| model.strip_prefix("gemini:"))
            .or_else(|| model.strip_prefix("google:"))
            .unwrap_or(model);
        
        // Gemini 3 models require the "-preview" suffix on VertexAI
        // Auto-add it for user convenience if they forget
        let model_name = if model_without_prefix.starts_with("gemini-3-") 
            && !model_without_prefix.ends_with("-preview") 
        {
            format!("{}-preview", model_without_prefix)
        } else {
            model_without_prefix.to_string()
        };
        
        match &self.endpoint {
            GeminiEndpoint::GoogleAI { api_key } => {
                format!(
                    "{}/models/{}:{}?key={}",
                    GEMINI_API_BASE, model_name, action, api_key
                )
            }
            GeminiEndpoint::VertexAI {
                project_id, region, ..
            } => {
                format!(
                    "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:{}",
                    region, project_id, region, model_name, action
                )
            }
        }
    }

    /// Get authorization headers for the request.
    fn auth_headers(&self) -> Vec<(&'static str, String)> {
        match &self.endpoint {
            GeminiEndpoint::GoogleAI { .. } => {
                // API key is in URL for Google AI
                vec![]
            }
            GeminiEndpoint::VertexAI { access_token, .. } => {
                vec![("Authorization", format!("Bearer {}", access_token))]
            }
        }
    }

    /// Convert ChatMessage to Gemini Content format.
    fn convert_messages(messages: &[ChatMessage]) -> (Option<Content>, Vec<Content>) {
        let mut system_instruction = None;
        let mut contents = Vec::new();

        for msg in messages {
            match msg.role {
                ChatRole::System => {
                    // Gemini uses system_instruction field for system prompts
                    system_instruction = Some(Content {
                        parts: vec![Part {
                            text: Some(msg.content.clone()),
                            ..Default::default()
                        }],
                        role: None,
                    });
                }
                ChatRole::User => {
                    // OODA-54: Check if message has images for multipart content
                    if msg.has_images() {
                        let mut parts = Vec::new();
                        
                        // Add text part first (if non-empty)
                        if !msg.content.is_empty() {
                            parts.push(Part {
                                text: Some(msg.content.clone()),
                                ..Default::default()
                            });
                        }
                        
                        // Add image parts
                        if let Some(ref images) = msg.images {
                            for img in images {
                                parts.push(Part {
                                    inline_data: Some(Blob {
                                        mime_type: img.mime_type.clone(),
                                        data: img.data.clone(),
                                    }),
                                    ..Default::default()
                                });
                            }
                        }
                        
                        contents.push(Content {
                            parts,
                            role: Some("user".to_string()),
                        });
                    } else {
                        contents.push(Content {
                            parts: vec![Part {
                                text: Some(msg.content.clone()),
                                ..Default::default()
                            }],
                            role: Some("user".to_string()),
                        });
                    }
                }
                ChatRole::Assistant => {
                    contents.push(Content {
                        parts: vec![Part {
                            text: Some(msg.content.clone()),
                            ..Default::default()
                        }],
                        role: Some("model".to_string()),
                    });
                }
                ChatRole::Tool | ChatRole::Function => {
                    // Handle tool/function messages as user messages with context
                    contents.push(Content {
                        parts: vec![Part {
                            text: Some(msg.content.clone()),
                            ..Default::default()
                        }],
                        role: Some("user".to_string()),
                    });
                }
            }
        }

        (system_instruction, contents)
    }

    /// Send a request and handle errors.
    async fn send_request<T: for<'de> Deserialize<'de>>(
        &self,
        url: &str,
        body: &impl Serialize,
    ) -> Result<T> {
        let mut request = self.client.post(url).json(body);

        for (key, value) in self.auth_headers() {
            request = request.header(key, value);
        }

        let response = request
            .send()
            .await
            .map_err(|e| LlmError::ApiError(format!("Request failed: {}", e)))?;

        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| LlmError::ApiError(format!("Failed to read response: {}", e)))?;

        if !status.is_success() {
            // Try to parse error response
            if let Ok(error_response) = serde_json::from_str::<GeminiErrorResponse>(&text) {
                return Err(LlmError::ApiError(format!(
                    "Gemini API error ({}): {}",
                    error_response.error.code, error_response.error.message
                )));
            }
            return Err(LlmError::ApiError(format!(
                "Gemini API error ({}): {}",
                status, text
            )));
        }

        serde_json::from_str(&text).map_err(|e| {
            LlmError::ApiError(format!("Failed to parse response: {}. Body: {}", e, text))
        })
    }
    
    // =========================================================================
    // OODA-06: Tool Conversion Methods
    // =========================================================================
    
    /// Remove `$schema` field from JSON object (Gemini doesn't accept it)
    fn sanitize_parameters(mut params: serde_json::Value) -> serde_json::Value {
        if let Some(obj) = params.as_object_mut() {
            obj.remove("$schema");
            
            // Also sanitize nested objects in properties, items, etc.
            for (_key, value) in obj.iter_mut() {
                if value.is_object() || value.is_array() {
                    *value = Self::sanitize_parameters(value.clone());
                }
            }
        } else if let Some(arr) = params.as_array_mut() {
            for item in arr.iter_mut() {
                if item.is_object() || item.is_array() {
                    *item = Self::sanitize_parameters(item.clone());
                }
            }
        }
        params
    }
    
    /// Convert EdgeCode ToolDefinition to Gemini FunctionDeclaration format
    fn convert_tools(tools: &[ToolDefinition]) -> Vec<GeminiTool> {
        let declarations: Vec<FunctionDeclaration> = tools
            .iter()
            .map(|tool| {
                // GEMINI-SCHEMA-FIX: Remove $schema field that Gemini doesn't accept
                let sanitized_params = Self::sanitize_parameters(tool.function.parameters.clone());
                
                FunctionDeclaration {
                    name: tool.function.name.clone(),
                    description: tool.function.description.clone(),
                    parameters: Some(sanitized_params),
                }
            })
            .collect();
        
        vec![GeminiTool { function_declarations: declarations }]
    }
    
    /// Convert ToolChoice to Gemini ToolConfig
    fn convert_tool_choice(tool_choice: Option<ToolChoice>) -> Option<ToolConfig> {
        let mode = match &tool_choice {
            None => "AUTO",
            Some(ToolChoice::Auto(s)) if s == "auto" => "AUTO",
            Some(ToolChoice::Auto(s)) if s == "none" => "NONE",
            Some(ToolChoice::Auto(_)) => "AUTO",
            Some(ToolChoice::Required(_)) => "ANY",
            Some(ToolChoice::Function { function, .. }) => {
                return Some(ToolConfig {
                    function_calling_config: FunctionCallingConfig {
                        mode: "ANY".to_string(),
                        allowed_function_names: Some(vec![function.name.clone()]),
                    },
                });
            }
        };
        
        Some(ToolConfig {
            function_calling_config: FunctionCallingConfig {
                mode: mode.to_string(),
                allowed_function_names: None,
            },
        })
    }
    
    // =========================================================================
    // OODA-25: Thinking Support Detection
    // =========================================================================
    
    /// Check if the current model supports thinking
    /// 
    /// CRITICAL: As of January 2026, NO Gemini models support thinkingConfig via
    /// the Google AI API (generativelanguage.googleapis.com). All models return
    /// "Unknown name 'thinkingConfig'" errors (400 Bad Request).
    /// 
    /// This includes:
    /// - ❌ Gemini 3 Flash (gemini-3-flash)
    /// - ❌ Gemini 3 Pro (gemini-3-pro)  
    /// - ❌ Gemini 2.5 Flash (gemini-2.5-flash)
    /// - ❌ Gemini 2.5 Pro (gemini-2.5-pro)
    /// 
    /// The thinkingConfig feature appears to be:
    /// 1. Documentation-only (not yet in production API)
    /// 2. Preview SDK-only (official Python/Node SDKs only)
    /// 3. Or requiring different API endpoint
    /// 
    /// DISABLE thinking for ALL models until Google enables it in the REST API.
    /// 
    /// See: <https://ai.google.dev/gemini-api/docs/thinking>
    /// API Ref: <https://ai.google.dev/api/generate-content>
    pub fn supports_thinking(&self) -> bool {
        // DISABLED: API doesn't support thinkingConfig as of Jan 2026
        false
    }
}

#[async_trait]
impl LLMProvider for GeminiProvider {
    fn name(&self) -> &str {
        match &self.endpoint {
            GeminiEndpoint::GoogleAI { .. } => "gemini",
            GeminiEndpoint::VertexAI { .. } => "vertex-ai",
        }
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_context_length(&self) -> usize {
        self.max_context_length
    }

    #[instrument(skip(self, prompt), fields(model = %self.model))]
    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        self.complete_with_options(prompt, &CompletionOptions::default())
            .await
    }

    #[instrument(skip(self, prompt, options), fields(model = %self.model))]
    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<LLMResponse> {
        let mut messages = Vec::new();

        if let Some(system) = &options.system_prompt {
            messages.push(ChatMessage::system(system));
        }
        messages.push(ChatMessage::user(prompt));

        self.chat(&messages, Some(options)).await
    }

    #[instrument(skip(self, messages, options), fields(model = %self.model))]
    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let (system_instruction, contents) = Self::convert_messages(messages);

        if contents.is_empty() {
            return Err(LlmError::InvalidRequest(
                "No user messages provided".to_string(),
            ));
        }

        let options = options.cloned().unwrap_or_default();

        // Build generation config
        let mut generation_config = GenerationConfig::default();

        if let Some(max_tokens) = options.max_tokens {
            generation_config.max_output_tokens = Some(max_tokens);
        }
        if let Some(temp) = options.temperature {
            generation_config.temperature = Some(temp);
        }
        if let Some(top_p) = options.top_p {
            generation_config.top_p = Some(top_p);
        }
        if let Some(stop) = options.stop {
            generation_config.stop_sequences = Some(stop);
        }
        if options.response_format.as_deref() == Some("json_object") {
            generation_config.response_mime_type = Some("application/json".to_string());
        }

        // Create or reuse cache if system instruction exists
        let cached_content = if let Some(system_inst) = system_instruction.as_ref() {
            match self.ensure_cache(system_inst).await {
                Ok(cache_id) => Some(cache_id),
                Err(e) => {
                    // Log error but continue without cache
                    debug!(
                        "Failed to create/reuse cache: {}, continuing without cache",
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

        let request = GenerateContentRequest {
            contents,
            generation_config: Some(generation_config),
            system_instruction: if cached_content.is_none() {
                system_instruction
            } else {
                None
            },
            safety_settings: None,
            cached_content,
            // OODA-06: No tools for regular chat
            tools: None,
            tool_config: None,
            // OODA-25: Enable thinking for Gemini 2.5+/3.x models
            thinking_config: if self.supports_thinking() {
                Some(ThinkingConfig {
                    include_thoughts: Some(true),
                    thinking_level: None,
                    thinking_budget: None,
                })
            } else {
                None
            },
        };

        let url = self.build_url(&self.model, "generateContent");
        debug!("Sending request to Gemini: {}", url);

        let response: GenerateContentResponse = self.send_request(&url, &request).await?;

        // Extract content from response
        let candidates = response
            .candidates
            .ok_or_else(|| LlmError::ApiError("No candidates in response".to_string()))?;

        let candidate = candidates
            .first()
            .ok_or_else(|| LlmError::ApiError("Empty candidates array".to_string()))?;

        // OODA-25: Separate thinking content from regular content
        let mut content = String::new();
        let mut thinking_content_parts: Vec<String> = Vec::new();
        
        for part in &candidate.content.parts {
            if let Some(text) = &part.text {
                if part.thought == Some(true) {
                    thinking_content_parts.push(text.clone());
                } else {
                    content.push_str(text);
                }
            }
        }
        
        let thinking_content = if thinking_content_parts.is_empty() {
            None
        } else {
            Some(thinking_content_parts.join(""))
        };

        let usage = response.usage_metadata.unwrap_or_default();

        let mut metadata = HashMap::new();
        if !candidate.safety_ratings.is_empty() {
            metadata.insert(
                "safety_ratings".to_string(),
                serde_json::json!(candidate.safety_ratings),
            );
        }

        Ok(LLMResponse {
            content,
            prompt_tokens: usage.prompt_token_count,
            completion_tokens: usage.candidates_token_count,
            total_tokens: usage.total_token_count,
            model: self.model.clone(),
            finish_reason: candidate.finish_reason.clone(),
            tool_calls: Vec::new(),
            metadata,
            cache_hit_tokens: if usage.cached_content_token_count > 0 {
                Some(usage.cached_content_token_count)
            } else {
                None
            },
            // OODA-25: Track thinking tokens and content from Gemini 2.5+/3.x
            thinking_tokens: if usage.thoughts_token_count > 0 {
                Some(usage.thoughts_token_count)
            } else {
                None
            },
            thinking_content,
        })
    }

    // =========================================================================
    // OODA-07: Tool Calling Implementation
    // =========================================================================
    //
    // Gemini function calling enables the model to call tools/functions.
    // The response may contain functionCall parts that need to be converted
    // to EdgeCode's ToolCall format.
    //
    // Request: { tools: [{ functionDeclarations: [...] }], toolConfig: {...} }
    // Response: { parts: [{ functionCall: { name, args } }] }
    // =========================================================================
    
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let (system_instruction, contents) = Self::convert_messages(messages);
        
        if contents.is_empty() {
            return Err(LlmError::InvalidRequest(
                "No user messages provided".to_string(),
            ));
        }
        
        let options = options.cloned().unwrap_or_default();
        
        // Build generation config
        let mut generation_config = GenerationConfig::default();
        
        if let Some(max_tokens) = options.max_tokens {
            generation_config.max_output_tokens = Some(max_tokens);
        }
        if let Some(temp) = options.temperature {
            generation_config.temperature = Some(temp);
        }
        if let Some(top_p) = options.top_p {
            generation_config.top_p = Some(top_p);
        }
        if let Some(stop) = options.stop {
            generation_config.stop_sequences = Some(stop);
        }
        
        // Convert tools to Gemini format
        let gemini_tools = if tools.is_empty() {
            None
        } else {
            Some(Self::convert_tools(tools))
        };
        
        let gemini_tool_config = Self::convert_tool_choice(tool_choice);
        
        // OODA-25: Enable thinking for Gemini 2.5+/3.x models
        let thinking_config = if self.supports_thinking() {
            Some(ThinkingConfig {
                include_thoughts: Some(true),
                thinking_level: None,
                thinking_budget: None,
            })
        } else {
            None
        };
        
        let request = GenerateContentRequest {
            contents,
            generation_config: Some(generation_config),
            system_instruction,
            safety_settings: None,
            cached_content: None,
            tools: gemini_tools,
            tool_config: gemini_tool_config,
            thinking_config,
        };
        
        let url = self.build_url(&self.model, "generateContent");
        debug!("Sending chat_with_tools request to Gemini: {}", url);
        
        let response: GenerateContentResponse = self.send_request(&url, &request).await?;
        
        // Parse response
        let candidates = response
            .candidates
            .ok_or_else(|| LlmError::ApiError("No candidates in response".to_string()))?;
        
        let candidate = candidates
            .first()
            .ok_or_else(|| LlmError::ApiError("Empty candidates array".to_string()))?;
        
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut thinking_content_parts: Vec<String> = Vec::new();
        
        // OODA-25: Parse all parts - separate thinking content from regular content
        for part in &candidate.content.parts {
            // Text content - check if thinking or regular
            if let Some(text) = &part.text {
                if part.thought == Some(true) {
                    thinking_content_parts.push(text.clone());
                } else {
                    content.push_str(text);
                }
            }
            // Function call - convert to ToolCall
            if let Some(fc) = &part.function_call {
                tool_calls.push(ToolCall {
                    id: format!("call_{}", uuid::Uuid::new_v4().to_string().replace('-', "")),
                    call_type: "function".to_string(),
                    function: crate::traits::FunctionCall {
                        name: fc.name.clone(),
                        arguments: fc.args.to_string(),
                    },
                });
            }
        }
        
        let thinking_content = if thinking_content_parts.is_empty() {
            None
        } else {
            Some(thinking_content_parts.join(""))
        };
        
        let usage = response.usage_metadata.unwrap_or_default();
        
        let mut metadata = HashMap::new();
        if !candidate.safety_ratings.is_empty() {
            metadata.insert(
                "safety_ratings".to_string(),
                serde_json::json!(candidate.safety_ratings),
            );
        }
        
        Ok(LLMResponse {
            content,
            prompt_tokens: usage.prompt_token_count,
            completion_tokens: usage.candidates_token_count,
            total_tokens: usage.total_token_count,
            model: self.model.clone(),
            finish_reason: candidate.finish_reason.clone(),
            tool_calls,
            metadata,
            cache_hit_tokens: if usage.cached_content_token_count > 0 {
                Some(usage.cached_content_token_count)
            } else {
                None
            },
            // OODA-25: Track thinking tokens and content from Gemini 2.5+/3.x
            thinking_tokens: if usage.thoughts_token_count > 0 {
                Some(usage.thoughts_token_count)
            } else {
                None
            },
            thinking_content,
        })
    }

    // ============================================================================
    // Streaming Implementation
    // ============================================================================
    //
    // WHY: Gemini streaming requires `alt=sse` parameter for proper SSE format.
    // Without `alt=sse`, Gemini returns a JSON array which is harder to parse
    // incrementally. With `alt=sse`, we get standard SSE format:
    //
    //   data: {"candidates":[...],"usageMetadata":{...}}
    //   data: {"candidates":[...],"finishReason":"STOP"}
    //
    // Each `data:` line contains a complete GenerateContentResponse JSON object.
    // ============================================================================
    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        use futures::StreamExt;

        let messages = vec![ChatMessage::user(prompt)];
        let (system_instruction, contents) = Self::convert_messages(&messages);

        let request = GenerateContentRequest {
            contents,
            generation_config: None,
            system_instruction,
            safety_settings: None,
            cached_content: None, // TODO: Add caching support for streaming
            // OODA-06: No tools for basic streaming
            tools: None,
            tool_config: None,
            // OODA-25: Enable thinking for streaming
            thinking_config: if self.supports_thinking() {
                Some(ThinkingConfig {
                    include_thoughts: Some(true),
                    thinking_level: None,
                    thinking_budget: None,
                })
            } else {
                None
            },
        };

        // WHY: Add `alt=sse` parameter for proper Server-Sent Events format
        // This makes parsing simpler and more reliable
        let base_url = self.build_url(&self.model, "streamGenerateContent");
        let url = if base_url.contains('?') {
            format!("{}&alt=sse", base_url)
        } else {
            format!("{}?alt=sse", base_url)
        };

        let mut req = self.client.post(&url).json(&request);
        for (key, value) in self.auth_headers() {
            req = req.header(key, value);
        }

        let response = req
            .send()
            .await
            .map_err(|e| LlmError::ApiError(format!("Stream request failed: {}", e)))?;

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!("Stream error: {}", text)));
        }

        let stream = response.bytes_stream();

        // WHY: Parse SSE format - each chunk may contain multiple `data:` lines
        // We need to handle partial chunks and accumulate data across boundaries
        let mapped_stream = stream.map(|result| {
            match result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    let mut content_parts = Vec::new();
                    
                    // Parse each `data:` line in the SSE response
                    for line in text.lines() {
                        if let Some(json_str) = line.strip_prefix("data: ") {
                            if let Ok(chunk) = serde_json::from_str::<GenerateContentResponse>(json_str) {
                                if let Some(candidates) = chunk.candidates {
                                    if let Some(candidate) = candidates.first() {
                                        let content: String = candidate
                                            .content
                                            .parts
                                            .iter()
                                            .filter_map(|p| p.text.clone())
                                            .collect();
                                        if !content.is_empty() {
                                            content_parts.push(content);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    Ok(content_parts.join(""))
                }
                Err(e) => Err(LlmError::ApiError(format!("Stream error: {}", e))),
            }
        });

        Ok(mapped_stream.boxed())
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        // Gemini 1.5+ supports JSON mode
        self.model.contains("gemini-1.5") || self.model.contains("gemini-2")
    }
    
    // OODA-07: Enable function calling for Gemini
    fn supports_function_calling(&self) -> bool {
        // Gemini 1.5+ and 2.x support function calling
        self.model.contains("gemini-1.5") || self.model.contains("gemini-2") || self.model.contains("gemini-3")
    }

    // =========================================================================
    // OODA-08: Streaming with Tool Calling
    // =========================================================================
    //
    // Combines streaming (SSE) with tool calling support.
    // Emits StreamChunk events for content, tool calls, and finish reasons.
    //
    // Response format:
    //   data: {"candidates":[{"content":{"parts":[{"text":"..."}]}}]}
    //   data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"...","args":{...}}}]}}]}
    //
    // =========================================================================
    
    async fn chat_with_tools_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        use futures::StreamExt;
        
        // Convert messages to Gemini format
        let (system_instruction, contents) = Self::convert_messages(messages);
        
        if contents.is_empty() {
            return Err(LlmError::InvalidRequest(
                "No user messages provided".to_string(),
            ));
        }
        
        // Convert tools using OODA-06 helpers
        let gemini_tools = if !tools.is_empty() {
            Some(Self::convert_tools(tools))
        } else {
            None
        };
        
        let tool_config = Self::convert_tool_choice(tool_choice);
        
        // Build generation config
        let options = options.cloned().unwrap_or_default();
        let mut generation_config = GenerationConfig::default();
        
        if let Some(max_tokens) = options.max_tokens {
            generation_config.max_output_tokens = Some(max_tokens);
        }
        if let Some(temp) = options.temperature {
            generation_config.temperature = Some(temp);
        }
        if let Some(top_p) = options.top_p {
            generation_config.top_p = Some(top_p);
        }
        if let Some(stop) = options.stop {
            generation_config.stop_sequences = Some(stop);
        }
        
        // OODA-25: Enable thinking for Gemini 2.5+/3.x models
        let thinking_config = if self.supports_thinking() {
            Some(ThinkingConfig {
                include_thoughts: Some(true),
                thinking_level: None, // Use model default
                thinking_budget: None, // Use model default (-1 dynamic)
            })
        } else {
            None
        };
        
        // Build request with tools
        let request = GenerateContentRequest {
            contents,
            generation_config: Some(generation_config),
            system_instruction,
            safety_settings: None,
            cached_content: None,
            tools: gemini_tools,
            tool_config,
            thinking_config,
        };
        
        // Build streaming URL with alt=sse
        let base_url = self.build_url(&self.model, "streamGenerateContent");
        let url = if base_url.contains('?') {
            format!("{}&alt=sse", base_url)
        } else {
            format!("{}?alt=sse", base_url)
        };
        
        // Send streaming request
        let mut req = self.client.post(&url).json(&request);
        for (key, value) in self.auth_headers() {
            req = req.header(key, value);
        }
        
        let response = req
            .send()
            .await
            .map_err(|e| LlmError::ApiError(format!("Stream request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!("Stream error: {}", text)));
        }
        
        let stream = response.bytes_stream();
        
        // Map SSE stream to StreamChunk events
        let mapped_stream = stream.map(|result| {
            match result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    let mut chunks = Vec::new();
                    
                    // Parse each `data:` line in the SSE response
                    for line in text.lines() {
                        if let Some(json_str) = line.strip_prefix("data: ") {
                            if let Ok(response) = serde_json::from_str::<GenerateContentResponse>(json_str) {
                                if let Some(candidates) = response.candidates {
                                    if let Some(candidate) = candidates.first() {
                                        // Process each part in the response
                                        for part in &candidate.content.parts {
                                            // Handle text content
                                            // OODA-25: Check if this is thinking content
                                            if let Some(text_content) = &part.text {
                                                if !text_content.is_empty() {
                                                    if part.thought == Some(true) {
                                                        // This is thinking/reasoning content
                                                        chunks.push(StreamChunk::ThinkingContent {
                                                            text: text_content.clone(),
                                                            tokens_used: None, // Gemini doesn't report per-chunk tokens
                                                            budget_total: None, // Could be enhanced with thinking_budget
                                                        });
                                                    } else {
                                                        // This is regular content
                                                        chunks.push(StreamChunk::Content(text_content.clone()));
                                                    }
                                                }
                                            }
                                            
                                            // Handle function calls
                                            if let Some(func_call) = &part.function_call {
                                                // Serialize args to JSON string
                                                let args_json = serde_json::to_string(&func_call.args).ok();
                                                
                                                chunks.push(StreamChunk::ToolCallDelta {
                                                    index: 0, // Gemini doesn't use indexing like OpenAI
                                                    id: Some(uuid::Uuid::new_v4().to_string()),
                                                    function_name: Some(func_call.name.clone()),
                                                    function_arguments: args_json,
                                                });
                                            }
                                        }
                                        
                                        // Check for finish reason
                                        if let Some(ref reason) = candidate.finish_reason {
                                            let mapped_reason = match reason.as_str() {
                                                "STOP" => "stop",
                                                "MAX_TOKENS" => "length",
                                                "SAFETY" => "content_filter",
                                                _ => reason.as_str(),
                                            };
                                            chunks.push(StreamChunk::Finished {
                                                reason: mapped_reason.to_string(),
                                                ttft_ms: None,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // Return first chunk or empty content
                    // Note: In real usage, we might want to flatten this better
                    if let Some(chunk) = chunks.into_iter().next() {
                        Ok(chunk)
                    } else {
                        Ok(StreamChunk::Content(String::new()))
                    }
                }
                Err(e) => Err(LlmError::ApiError(format!("Stream error: {}", e))),
            }
        });
        
        Ok(mapped_stream.boxed())
    }
    
    // OODA-08: Enable streaming with tools for Gemini
    fn supports_tool_streaming(&self) -> bool {
        // Same models that support function calling also support streaming with tools
        self.model.contains("gemini-1.5") || self.model.contains("gemini-2") || self.model.contains("gemini-3")
    }
}

#[async_trait]
impl EmbeddingProvider for GeminiProvider {
    fn name(&self) -> &str {
        "gemini"
    }

    /// Returns the embedding model name (not completion model).
    #[allow(clippy::misnamed_getters)]
    fn model(&self) -> &str {
        &self.embedding_model
    }

    fn dimension(&self) -> usize {
        self.embedding_dimension
    }

    fn max_tokens(&self) -> usize {
        2048 // Gemini embedding models max tokens
    }

    #[instrument(skip(self, texts), fields(model = %self.embedding_model, count = texts.len()))]
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Use batch endpoint for multiple texts
        if texts.len() > 1 {
            return self.embed_batch(texts).await;
        }

        // Single text - use embedContent
        let request = EmbedContentRequest {
            content: Content {
                parts: vec![Part {
                    text: Some(texts[0].clone()),
                    ..Default::default()
                }],
                role: None,
            },
            model: None, // Not needed for single embedContent
            task_type: Some("RETRIEVAL_DOCUMENT".to_string()),
            title: None,
            output_dimensionality: None,
        };

        let url = self.build_url(&self.embedding_model, "embedContent");
        debug!("Sending embedding request to Gemini: {}", url);

        let response: EmbedContentResponse = self.send_request(&url, &request).await?;

        Ok(vec![response.embedding.values])
    }

    async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed(&[text.to_string()]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::Unknown("Empty embedding result".to_string()))
    }
}

impl GeminiProvider {
    /// Embed multiple texts using batch endpoint.
    /// Embed multiple texts using batch endpoint.
    ///
    /// WHY: Batch endpoint is more efficient for multiple texts (single API call).
    /// Each request in batch MUST include the model field per Gemini API spec.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // WHY: batchEmbedContents requires model field in each request
        // Format: "models/{model_name}"
        let model_path = format!("models/{}", self.embedding_model);
        
        let requests: Vec<EmbedContentRequest> = texts
            .iter()
            .map(|text| EmbedContentRequest {
                content: Content {
                    parts: vec![Part { text: Some(text.clone()), ..Default::default() }],
                    role: None,
                },
                model: Some(model_path.clone()),
                task_type: Some("RETRIEVAL_DOCUMENT".to_string()),
                title: None,
                output_dimensionality: None,
            })
            .collect();

        let batch_request = BatchEmbedContentsRequest { requests };

        let url = self.build_url(&self.embedding_model, "batchEmbedContents");
        debug!("Sending batch embedding request to Gemini: {}", url);

        let response: BatchEmbedContentsResponse = self.send_request(&url, &batch_request).await?;

        Ok(response.embeddings.into_iter().map(|e| e.values).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_length_detection() {
        assert_eq!(
            GeminiProvider::context_length_for_model("gemini-2.0-flash"),
            1_000_000
        );
        assert_eq!(
            GeminiProvider::context_length_for_model("gemini-1.5-pro"),
            2_000_000
        );
        assert_eq!(
            GeminiProvider::context_length_for_model("gemini-1.0-pro"),
            32_000
        );
    }

    #[test]
    fn test_embedding_dimension_detection() {
        assert_eq!(
            GeminiProvider::dimension_for_model("text-embedding-004"),
            768
        );
        assert_eq!(
            GeminiProvider::dimension_for_model("text-embedding-005"),
            768
        );
    }

    #[test]
    fn test_provider_builder() {
        let provider = GeminiProvider::new("test-key")
            .with_model("gemini-1.5-pro")
            .with_embedding_model("text-embedding-004");

        assert_eq!(LLMProvider::model(&provider), "gemini-1.5-pro");
        assert_eq!(provider.dimension(), 768);
        assert_eq!(provider.max_context_length(), 2_000_000);
    }

    #[test]
    fn test_vertex_ai_provider() {
        let provider = GeminiProvider::vertex_ai("my-project", "us-central1", "test-token");
        assert_eq!(LLMProvider::name(&provider), "vertex-ai");
    }

    #[test]
    fn test_message_conversion() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
        ];

        let (system, contents) = GeminiProvider::convert_messages(&messages);

        assert!(system.is_some());
        assert_eq!(system.unwrap().parts[0].text, Some("You are helpful".to_string()));
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0].role.as_deref(), Some("user"));
        assert_eq!(contents[1].role.as_deref(), Some("model"));
    }

    // =========================================================================
    // Image Support Tests (OODA-54)
    // =========================================================================

    #[test]
    fn test_convert_messages_text_only() {
        // WHY: Verify text-only messages serialize correctly with new Part structure
        let messages = vec![ChatMessage::user("Hello, world!")];
        let (_, contents) = GeminiProvider::convert_messages(&messages);

        assert_eq!(contents.len(), 1);
        
        // Text-only should serialize with text field, no inline_data
        let json = serde_json::to_value(&contents[0]).unwrap();
        let parts = &json["parts"];
        
        assert!(parts.is_array());
        assert_eq!(parts.as_array().unwrap().len(), 1);
        assert_eq!(parts[0]["text"], "Hello, world!");
        assert!(parts[0].get("inlineData").is_none());
    }

    #[test]
    fn test_convert_messages_with_images() {
        use crate::traits::ImageData;
        
        // WHY: Verify images use Gemini's inlineData format
        let images = vec![ImageData::new("base64data", "image/png")];
        let messages = vec![ChatMessage::user_with_images("What's this?", images)];
        let (_, contents) = GeminiProvider::convert_messages(&messages);

        assert_eq!(contents.len(), 1);
        
        // With images should have multiple parts
        let json = serde_json::to_value(&contents[0]).unwrap();
        let parts = &json["parts"];
        
        assert!(parts.is_array());
        assert_eq!(parts.as_array().unwrap().len(), 2);
        
        // First part: text
        assert_eq!(parts[0]["text"], "What's this?");
        
        // Second part: inlineData (Gemini format)
        assert!(parts[1].get("inlineData").is_some(), "Should have inlineData for image");
        assert_eq!(parts[1]["inlineData"]["mimeType"], "image/png");
        assert_eq!(parts[1]["inlineData"]["data"], "base64data");
    }

    #[test]
    fn test_convert_messages_multiple_images() {
        use crate::traits::ImageData;
        
        // WHY: Verify multiple images are handled correctly
        let images = vec![
            ImageData::new("img1data", "image/png"),
            ImageData::new("img2data", "image/jpeg"),
        ];
        let messages = vec![ChatMessage::user_with_images("Compare these", images)];
        let (_, contents) = GeminiProvider::convert_messages(&messages);

        let json = serde_json::to_value(&contents[0]).unwrap();
        let parts = &json["parts"];
        
        assert_eq!(parts.as_array().unwrap().len(), 3); // 1 text + 2 images
        
        // Verify both images
        assert_eq!(parts[1]["inlineData"]["mimeType"], "image/png");
        assert_eq!(parts[2]["inlineData"]["mimeType"], "image/jpeg");
    }

    #[test]
    fn test_build_url_google_ai() {
        let provider = GeminiProvider::new("test-api-key");
        let url = provider.build_url("gemini-2.0-flash", "generateContent");

        assert!(url.contains("generativelanguage.googleapis.com"));
        assert!(url.contains("gemini-2.0-flash"));
        assert!(url.contains("key=test-api-key"));
    }

    #[test]
    fn test_build_url_vertex_ai() {
        let provider = GeminiProvider::vertex_ai("my-project", "us-central1", "token");
        let url = provider.build_url("gemini-2.0-flash", "generateContent");

        assert!(url.contains("aiplatform.googleapis.com"));
        assert!(url.contains("my-project"));
        assert!(url.contains("us-central1"));
    }

    // =========================================================================
    // OODA-25: Thinking Support Tests
    // =========================================================================

    #[test]
    fn test_supports_thinking_gemini_25() {
        // Gemini 2.5 models do NOT support thinking (API rejects thinkingConfig)
        let provider = GeminiProvider::new("key").with_model("gemini-2.5-flash");
        assert!(!provider.supports_thinking());
        
        let provider = GeminiProvider::new("key").with_model("gemini-2.5-pro");
        assert!(!provider.supports_thinking());
    }

    #[test]
    fn test_supports_thinking_gemini_3() {
        // Gemini 3 models also do NOT support thinking via REST API (as of Jan 2026)
        // Documentation shows it, but API rejects with 400 error
        let provider = GeminiProvider::new("key").with_model("gemini-3-flash");
        assert!(!provider.supports_thinking());
        
        let provider = GeminiProvider::new("key").with_model("gemini-3-pro");
        assert!(!provider.supports_thinking());
    }

    #[test]
    fn test_supports_thinking_gemini_1x() {
        // Gemini 1.x models do NOT support thinking
        let provider = GeminiProvider::new("key").with_model("gemini-1.5-flash");
        assert!(!provider.supports_thinking());
        
        let provider = GeminiProvider::new("key").with_model("gemini-1.0-pro");
        assert!(!provider.supports_thinking());
    }

    #[test]
    fn test_thinking_config_serialization() {
        // Verify ThinkingConfig serializes correctly to camelCase
        let config = ThinkingConfig {
            include_thoughts: Some(true),
            thinking_level: Some("high".to_string()),
            thinking_budget: Some(1024),
        };
        
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["includeThoughts"], true);
        assert_eq!(json["thinkingLevel"], "high");
        assert_eq!(json["thinkingBudget"], 1024);
    }

    #[test]
    fn test_part_thought_deserialization() {
        // Verify Part deserializes thought field correctly
        let json = r#"{"text": "thinking...", "thought": true}"#;
        let part: Part = serde_json::from_str(json).unwrap();
        
        assert_eq!(part.text, Some("thinking...".to_string()));
        assert_eq!(part.thought, Some(true));
    }

    #[test]
    fn test_part_thought_defaults_to_none() {
        // Verify Part without thought field defaults to None
        let json = r#"{"text": "response"}"#;
        let part: Part = serde_json::from_str(json).unwrap();
        
        assert_eq!(part.text, Some("response".to_string()));
        assert_eq!(part.thought, None);
    }

    #[test]
    fn test_usage_metadata_thoughts_token_count() {
        // Verify UsageMetadata deserializes thoughtsTokenCount
        let json = r#"{"promptTokenCount": 100, "candidatesTokenCount": 50, "totalTokenCount": 150, "thoughtsTokenCount": 25}"#;
        let usage: UsageMetadata = serde_json::from_str(json).unwrap();
        
        assert_eq!(usage.prompt_token_count, 100);
        assert_eq!(usage.candidates_token_count, 50);
        assert_eq!(usage.thoughts_token_count, 25);
    }

    // =========================================================================
    // OODA-34: Additional Unit Tests
    // =========================================================================

    #[test]
    fn test_constants() {
        // WHY: Verify constants are as expected for API compatibility
        assert_eq!(GEMINI_API_BASE, "https://generativelanguage.googleapis.com/v1beta");
        assert_eq!(DEFAULT_GEMINI_MODEL, "gemini-2.5-flash");
        assert_eq!(DEFAULT_EMBEDDING_MODEL, "text-embedding-004");
    }

    #[test]
    fn test_google_ai_provider_name() {
        // WHY: Google AI endpoint should return "gemini" as name
        let provider = GeminiProvider::new("test-key");
        assert_eq!(LLMProvider::name(&provider), "gemini");
    }

    #[test]
    fn test_supports_streaming() {
        let provider = GeminiProvider::new("test-key");
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_supports_json_mode_gemini_25() {
        // WHY: Gemini 2.x models support JSON mode
        let provider = GeminiProvider::new("key").with_model("gemini-2.5-flash");
        assert!(provider.supports_json_mode());
    }

    #[test]
    fn test_supports_json_mode_gemini_15() {
        // WHY: Gemini 1.5 models support JSON mode
        let provider = GeminiProvider::new("key").with_model("gemini-1.5-pro");
        assert!(provider.supports_json_mode());
    }

    #[test]
    fn test_supports_json_mode_gemini_10() {
        // WHY: Gemini 1.0 does NOT support JSON mode
        let provider = GeminiProvider::new("key").with_model("gemini-1.0-pro");
        assert!(!provider.supports_json_mode());
    }

    #[test]
    fn test_with_cache_ttl() {
        let provider = GeminiProvider::new("key").with_cache_ttl("7200s");
        assert_eq!(provider.cache_ttl, "7200s");
    }

    #[test]
    fn test_embedding_provider_name() {
        let provider = GeminiProvider::new("key");
        assert_eq!(EmbeddingProvider::name(&provider), "gemini");
    }

    #[test]
    fn test_embedding_provider_model() {
        let provider = GeminiProvider::new("key").with_embedding_model("text-embedding-005");
        assert_eq!(EmbeddingProvider::model(&provider), "text-embedding-005");
    }

    #[test]
    fn test_embedding_provider_max_tokens() {
        let provider = GeminiProvider::new("key");
        // Gemini embedding models support large input
        assert!(EmbeddingProvider::max_tokens(&provider) > 0);
    }

    #[tokio::test]
    async fn test_embed_empty_input() {
        let provider = GeminiProvider::new("key");
        let texts: Vec<String> = vec![];
        let result = provider.embed_batch(&texts).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_generation_config_serialization() {
        let config = GenerationConfig {
            max_output_tokens: Some(1000),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            stop_sequences: Some(vec!["END".to_string()]),
            response_mime_type: Some("application/json".to_string()),
        };
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["maxOutputTokens"], 1000);
        // WHY: f32 serialization may have precision differences, check approximate
        let temp = json["temperature"].as_f64().unwrap();
        assert!((temp - 0.7).abs() < 0.001);
        let top_p = json["topP"].as_f64().unwrap();
        assert!((top_p - 0.9).abs() < 0.001);
        assert_eq!(json["topK"], 40);
        assert_eq!(json["stopSequences"], serde_json::json!(["END"]));
        assert_eq!(json["responseMimeType"], "application/json");
    }

    #[test]
    fn test_gemini_models_response_deserialization() {
        let json = r#"{
            "models": [
                {
                    "name": "models/gemini-2.5-flash",
                    "displayName": "Gemini 2.5 Flash",
                    "description": "Fast model",
                    "inputTokenLimit": 1000000,
                    "outputTokenLimit": 8192,
                    "supportedGenerationMethods": ["generateContent"]
                }
            ]
        }"#;
        let response: GeminiModelsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.models.len(), 1);
        assert_eq!(response.models[0].name, "models/gemini-2.5-flash");
        assert_eq!(response.models[0].display_name, "Gemini 2.5 Flash");
        assert_eq!(response.models[0].input_token_limit, Some(1000000));
    }

    #[test]
    fn test_function_call_deserialization() {
        let json = r#"{"name": "get_weather", "args": {"location": "London"}}"#;
        let fc: FunctionCall = serde_json::from_str(json).unwrap();
        assert_eq!(fc.name, "get_weather");
        assert_eq!(fc.args["location"], "London");
    }
}
