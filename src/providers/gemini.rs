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
    ChatMessage,
    ChatRole,
    CompletionOptions,
    EmbeddingProvider,
    LLMProvider,
    LLMResponse,
    StreamChunk,
    StreamUsage,
    ToolCall,
    ToolChoice,
    ToolDefinition, // OODA-06/07/08: Tool + streaming support
};

/// Gemini API endpoints
const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Default models
// WHY: gemini-2.5-flash is the stable production model as of Feb 2026
// gemini-3-flash and gemini-3-pro are available as preview
// See: https://ai.google.dev/gemini-api/docs/models
const DEFAULT_GEMINI_MODEL: &str = "gemini-2.5-flash";

// WHY: gemini-embedding-001 is the current recommended embedding model (Feb 2026)
// It replaces text-embedding-004 and supports dimensions 128-3072
// Default output is 3072 dimensions; recommended: 768, 1536, 3072
// See: https://ai.google.dev/gemini-api/docs/embeddings
const DEFAULT_EMBEDDING_MODEL: &str = "gemini-embedding-001";

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
    pub mime_type: String, // MIME type, e.g., "image/png"
    pub data: String,      // Base64-encoded data
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
    // Gemini 3.x: Encrypted "save state" for the model's reasoning process.
    // Must be echoed back to the model in subsequent requests when the part
    // contains a functionCall.  Omitting it returns HTTP 400 INVALID_ARGUMENT.
    // See: https://cloud.google.com/vertex-ai/generative-ai/docs/thought-signatures
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
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
    // OODA-25/VertexAI: thinkingConfig lives inside generationConfig
    // See: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
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
    /// Included when the prompt itself was blocked by safety filters.
    prompt_feedback: Option<PromptFeedback>,
    /// The actual model version used (e.g. "gemini-2.5-flash-001").
    #[allow(dead_code)]
    model_version: Option<String>,
}

/// Prompt-level feedback from the Gemini API.
/// Present when the request was blocked before any candidates were generated.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptFeedback {
    block_reason: Option<String>,
    #[allow(dead_code)]
    safety_ratings: Option<Vec<serde_json::Value>>,
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

// ============================================================================
// VertexAI Embedding Types
// ============================================================================
//
// VertexAI uses a different API format for embeddings:
//   POST .../models/{model}:predict
//   { "instances": [{ "content": "text" }], "parameters": { ... } }
//
// Response:
//   { "predictions": [{ "embeddings": { "values": [...], "statistics": {...} } }] }
//
// See: https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
// ============================================================================

/// VertexAI embedding request instance
#[derive(Debug, Clone, Serialize)]
struct VertexAIEmbedInstance {
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
}

/// VertexAI embedding request parameters
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexAIEmbedParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimensionality: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    auto_truncate: Option<bool>,
}

/// VertexAI embedding request body
#[derive(Debug, Clone, Serialize)]
struct VertexAIEmbedRequest {
    instances: Vec<VertexAIEmbedInstance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<VertexAIEmbedParameters>,
}

/// VertexAI embedding prediction result
#[derive(Debug, Clone, Deserialize)]
struct VertexAIEmbedPrediction {
    embeddings: VertexAIEmbeddingResult,
}

/// VertexAI embedding values nested in prediction
#[derive(Debug, Clone, Deserialize)]
struct VertexAIEmbeddingResult {
    values: Vec<f32>,
    // statistics is also returned but we don't need it
}

/// VertexAI embedding response
#[derive(Debug, Clone, Deserialize)]
struct VertexAIEmbedResponse {
    predictions: Vec<VertexAIEmbedPrediction>,
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
// Model capability registry — the ONLY place model knowledge lives.
//
// Design goals (SOLID + first principles):
//   • Open/Closed  – add a model by adding a row; no other code changes.
//   • Single Responsibility – each field captures exactly one API-level fact.
//   • No flaky heuristics – `starts_with` prefix matching replaces `contains`
//     substring checks that can match arbitrary substrings in future names.
//
// Ordering rule: list MORE-SPECIFIC prefixes BEFORE less-specific ones so that
// the first-match lookup returns the right row (e.g. "gemini-2.5-flash-lite"
// must come before "gemini-2.5-flash" which must come before "gemini-2.5-").
//
// Sources (April 2026):
//   https://ai.google.dev/gemini-api/docs/models
//   https://ai.google.dev/gemini-api/docs/thinking
// ============================================================================

/// How a model family controls its thinking behaviour.
///
/// First-class enum so dispatch in `build_thinking_config` is type-checked
/// rather than based on fragile name substrings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum ThinkingStyle {
    /// Model does not support thinking (Gemini 1.x, 2.0).
    None,
    /// Gemini 2.5 series: use `thinkingBudget` (int). -1 = dynamic.
    /// Range is the inclusive valid budget window for the model.
    Budget { min: i32, max: i32 },
    /// Gemini 3.x series: use `thinkingLevel` enum string.
    Level,
}

/// Static properties of one Gemini model family, derived from official docs.
pub(super) struct ModelProfile {
    /// Longest prefix that uniquely identifies this family.
    /// Must be `starts_with`-comparable against a bare model ID (provider
    /// prefix already stripped).
    pub prefix: &'static str,
    /// Input token context window (official docs, April 2026).
    pub context_length: usize,
    /// How this model family controls thinking.
    pub thinking: ThinkingStyle,
    /// `true` when the model starts thinking even without explicit config.
    /// `false` for Flash-Lite which requires an explicit opt-in per spec.
    pub thinks_by_default: bool,
    /// `true` when the bare model ID needs a "-preview" suffix appended.
    pub auto_preview_suffix: bool,
    /// `true` when Vertex AI must route to the global (not regional) endpoint.
    pub requires_global_vertex: bool,
}

/// Capability table — indexed by `starts_with(prefix)`, first match wins.
/// Most-specific prefixes MUST come first.
static MODEL_PROFILES: &[ModelProfile] = &[
    // ---- Gemini 3.1 series (most specific of 3.x) --------------------------
    ModelProfile {
        prefix: "gemini-3.1-",
        context_length: 2_000_000,
        thinking: ThinkingStyle::Level,
        thinks_by_default: true,
        auto_preview_suffix: true,
        requires_global_vertex: true,
    },
    // ---- Gemini 3 Flash (shut-down 3-pro excluded by not providing a profile)
    // 3-pro-preview was shut down 2026-03-09; no profile means build_url won't
    // auto-append "-preview" and requests will fail fast at the API level.
    ModelProfile {
        prefix: "gemini-3-flash",
        context_length: 2_000_000,
        thinking: ThinkingStyle::Level,
        thinks_by_default: true,
        auto_preview_suffix: true,
        requires_global_vertex: true,
    },
    // ---- Gemini 3 catch-all (covers gemini-3-pro and other 3.x variants).
    // auto_preview_suffix=false: gemini-3-pro-preview shut down 2026-03-09;
    // do NOT auto-append "-preview" so callers get a clean 404/403 at the API.
    ModelProfile {
        prefix: "gemini-3-",
        context_length: 2_000_000,
        thinking: ThinkingStyle::Level,
        thinks_by_default: true,
        auto_preview_suffix: false,
        requires_global_vertex: true,
    },
    // ---- Gemini 2.5 Flash-Lite (before 2.5-flash so it matches first) ------
    // "Model does not think by default" — official budget table, April 2026.
    ModelProfile {
        prefix: "gemini-2.5-flash-lite",
        context_length: 1_048_576,
        thinking: ThinkingStyle::Budget {
            min: 512,
            max: 24_576,
        },
        thinks_by_default: false,
        auto_preview_suffix: false,
        requires_global_vertex: false,
    },
    // ---- Gemini 2.5 Flash ---------------------------------------------------
    ModelProfile {
        prefix: "gemini-2.5-flash",
        context_length: 1_048_576,
        thinking: ThinkingStyle::Budget {
            min: 0,
            max: 24_576,
        },
        thinks_by_default: true,
        auto_preview_suffix: false,
        requires_global_vertex: false,
    },
    // ---- Gemini 2.5 Pro -----------------------------------------------------
    ModelProfile {
        prefix: "gemini-2.5-pro",
        context_length: 1_048_576,
        thinking: ThinkingStyle::Budget {
            min: 128,
            max: 32_768,
        },
        thinks_by_default: true,
        auto_preview_suffix: false,
        requires_global_vertex: false,
    },
    // ---- Gemini 2.5 catch-all (other 2.5 variants) -------------------------
    ModelProfile {
        prefix: "gemini-2.5-",
        context_length: 1_048_576,
        thinking: ThinkingStyle::Budget {
            min: 0,
            max: 24_576,
        },
        thinks_by_default: true,
        auto_preview_suffix: false,
        requires_global_vertex: false,
    },
    // ---- Gemini 2.0 (deprecated 2026, no thinking support) -----------------
    ModelProfile {
        prefix: "gemini-2.0-",
        context_length: 1_048_576,
        thinking: ThinkingStyle::None,
        thinks_by_default: false,
        auto_preview_suffix: false,
        requires_global_vertex: false,
    },
    // ---- Gemini 1.5 Pro -----------------------------------------------------
    ModelProfile {
        prefix: "gemini-1.5-pro",
        context_length: 2_000_000,
        thinking: ThinkingStyle::None,
        thinks_by_default: false,
        auto_preview_suffix: false,
        requires_global_vertex: false,
    },
    // ---- Gemini 1.5 Flash ---------------------------------------------------
    ModelProfile {
        prefix: "gemini-1.5-flash",
        context_length: 1_000_000,
        thinking: ThinkingStyle::None,
        thinks_by_default: false,
        auto_preview_suffix: false,
        requires_global_vertex: false,
    },
    // ---- Gemini 1.0 / 1.x --------------------------------------------------
    ModelProfile {
        prefix: "gemini-1.0-",
        context_length: 32_000,
        thinking: ThinkingStyle::None,
        thinks_by_default: false,
        auto_preview_suffix: false,
        requires_global_vertex: false,
    },
    // ---- gemini-pro (legacy alias used in older SDKs) ----------------------
    ModelProfile {
        prefix: "gemini-pro",
        context_length: 32_000,
        thinking: ThinkingStyle::None,
        thinks_by_default: false,
        auto_preview_suffix: false,
        requires_global_vertex: false,
    },
];

/// Fallback profile returned for any model string not matching a known prefix.
/// Uses conservative 2.5-class defaults — adequate for future Gemini models.
static DEFAULT_PROFILE: ModelProfile = ModelProfile {
    prefix: "",
    context_length: 1_048_576,
    thinking: ThinkingStyle::None,
    thinks_by_default: false,
    auto_preview_suffix: false,
    requires_global_vertex: false,
};

// ============================================================================
// GeminiProvider Implementation
// ============================================================================

impl GeminiProvider {
    fn stream_usage_from_metadata(usage: Option<&UsageMetadata>) -> Option<StreamUsage> {
        let usage = usage?;
        let mut stream_usage =
            StreamUsage::new(usage.prompt_token_count, usage.candidates_token_count);
        if usage.cached_content_token_count > 0 {
            stream_usage = stream_usage.with_cache_hit_tokens(usage.cached_content_token_count);
        }
        if usage.thoughts_token_count > 0 {
            stream_usage = stream_usage.with_thinking_tokens(usage.thoughts_token_count);
        }
        Some(stream_usage)
    }

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
            max_context_length: 1_000_000, // Gemini 2.5 flash default
            embedding_dimension: 3072,     // gemini-embedding-001 default
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
                 Run: export GOOGLE_CLOUD_PROJECT=your-project-id"
                    .to_string(),
            )
        })?;

        let region =
            std::env::var("GOOGLE_CLOUD_REGION").unwrap_or_else(|_| "us-central1".to_string());

        // Try to get access token from env, or obtain via gcloud CLI
        let access_token = match std::env::var("GOOGLE_ACCESS_TOKEN") {
            Ok(token) if !token.is_empty() => token,
            _ => Self::get_access_token_from_gcloud()?,
        };

        Ok(Self::vertex_ai(project_id, region, access_token))
    }

    /// Get access token from gcloud CLI.
    ///
    /// OODA-95: Tries `gcloud auth print-access-token` first, then falls back to
    /// `gcloud auth application-default print-access-token`.
    fn get_access_token_from_gcloud() -> Result<String> {
        // Try user credentials first
        debug!("Obtaining access token via gcloud auth print-access-token");
        if let Ok(token) = Self::run_gcloud_token_cmd(&["auth", "print-access-token"]) {
            return Ok(token);
        }

        // Fall back to application-default credentials (ADC)
        debug!("Falling back to gcloud auth application-default print-access-token");
        if let Ok(token) =
            Self::run_gcloud_token_cmd(&["auth", "application-default", "print-access-token"])
        {
            return Ok(token);
        }

        Err(LlmError::ConfigError(
            "Could not obtain a Google Cloud access token. \
             Run one of the following and try again:\n  \
             gcloud auth login\n  \
             gcloud auth application-default login"
                .to_string(),
        ))
    }

    /// Run a gcloud subcommand that prints a token to stdout, returning the token or an error.
    fn run_gcloud_token_cmd(args: &[&str]) -> Result<String> {
        use std::process::Command;
        let output = Command::new("gcloud")
            .args(args)
            .output()
            .map_err(|e| LlmError::ConfigError(format!("Failed to run gcloud: {}", e)))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(LlmError::ConfigError(stderr.trim().to_string()));
        }
        let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if token.is_empty() {
            return Err(LlmError::ConfigError("empty token".to_string()));
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
            embedding_dimension: 3072,
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

    /// Set a custom embedding output dimensionality.
    ///
    /// `gemini-embedding-001` supports dimensions 128-3072.
    /// Recommended values: 768, 1536, 3072 (default).
    /// Smaller dimensions save storage/compute with minimal quality loss.
    ///
    /// See: <https://ai.google.dev/gemini-api/docs/embeddings#controlling-embedding-size>
    pub fn with_embedding_dimension(mut self, dimension: usize) -> Self {
        self.embedding_dimension = dimension;
        self
    }

    // =========================================================================
    // Model capability registry — single source of truth (DRY + Open/Closed)
    //
    // All model-family properties live here, keyed by name prefix (longest
    // prefix wins; entries must therefore be listed most-specific first).
    // Adding a new model family only requires adding a row to this table —
    // no other code needs to change.
    //
    // Sources (April 2026 — most recent docs fetched 2026-04-04):
    //   https://ai.google.dev/gemini-api/docs/models
    //   https://ai.google.dev/gemini-api/docs/thinking
    //   https://ai.google.dev/gemini-api/docs/embeddings
    // =========================================================================

    // ---- derived accessors (all expressed in terms of the profile) ----------

    /// Return the input token limit for `model`.
    fn strip_provider_prefix(model: &str) -> &str {
        model
            .strip_prefix("vertexai:")
            .or_else(|| model.strip_prefix("gemini:"))
            .or_else(|| model.strip_prefix("google:"))
            .unwrap_or(model)
    }

    /// Look up the capability profile for a model name.
    ///
    /// Matches by the longest prefix that identifies the model family.
    /// Returns a generic default for unrecognised models.
    fn lookup_profile(model: &str) -> &'static ModelProfile {
        let bare = Self::strip_provider_prefix(model);
        MODEL_PROFILES
            .iter()
            .find(|p| bare.starts_with(p.prefix))
            .unwrap_or(&DEFAULT_PROFILE)
    }

    /// Return the capability profile for the model configured on this provider.
    fn profile(&self) -> &'static ModelProfile {
        Self::lookup_profile(&self.model)
    }

    // ---- derived accessors (all expressed in terms of the profile) ----------

    /// Return the input token limit for `model`.
    pub fn context_length_for_model(model: &str) -> usize {
        Self::lookup_profile(model).context_length
    }

    /// Get embedding dimension for a given model.
    ///
    /// # Supported Models (April 2026):
    /// - `gemini-embedding-2-preview`: 3072 default (128-3072 via `output_dimensionality`)
    /// - `gemini-embedding-001`: 3072 default (128-3072 via `output_dimensionality`)
    /// - `text-embedding-004` / `text-embedding-005`: 768 (legacy)
    ///
    /// See: <https://ai.google.dev/gemini-api/docs/embeddings>
    pub fn dimension_for_model(model: &str) -> usize {
        // Embedding models are not in MODEL_PROFILES (they are only used via the
        // embedding API, not generate-content). Keep a small dedicated lookup.
        match model {
            m if m.contains("gemini-embedding-2") => 3072,
            m if m.contains("gemini-embedding-001") => 3072,
            m if m.contains("text-embedding-004") => 768,
            m if m.contains("text-embedding-005") => 768,
            m if m.contains("text-multilingual-embedding-002") => 768,
            _ => 3072,
        }
    }

    /// Apply a `CompletionOptions` value onto a `GenerationConfig`.
    ///
    /// Centralises the five option-fields that all three call sites
    /// (`chat`, `chat_with_tools`, `chat_with_tools_stream`) used to apply
    /// individually, eliminating the DRY violation.
    fn apply_generation_options(config: &mut GenerationConfig, options: &CompletionOptions) {
        if let Some(max_tokens) = options.max_tokens {
            config.max_output_tokens = Some(max_tokens);
        }
        if let Some(temp) = options.temperature {
            config.temperature = Some(temp);
        }
        if let Some(top_p) = options.top_p {
            config.top_p = Some(top_p);
        }
        if let Some(ref stop) = options.stop {
            config.stop_sequences = Some(stop.clone());
        }
        if options.response_format.as_deref() == Some("json_object") {
            config.response_mime_type = Some("application/json".to_string());
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
                let host = Self::vertex_host(region);
                format!(
                    "https://{}/v1/projects/{}/locations/{}/publishers/google/models",
                    host, project_id, region
                )
            }
        };

        debug!(url = %url, "Fetching Gemini models list");

        let mut req = self.client.get(&url);

        // Add auth header for VertexAI
        if let GeminiEndpoint::VertexAI { access_token, .. } = &self.endpoint {
            req = req.bearer_auth(access_token);
        }

        let response = req
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(format!("Failed to fetch Gemini models: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "Gemini /models returned {}: {}",
                status, body
            )));
        }

        response
            .json::<GeminiModelsResponse>()
            .await
            .map_err(|e| LlmError::ProviderError(format!("Failed to parse models response: {}", e)))
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
                // Gemini 3.x preview models are global-endpoint-only;
                // cached content must also be created at the global endpoint.
                let effective_region: &str = if self.model.contains("gemini-3") {
                    "global"
                } else {
                    region.as_str()
                };
                let host = Self::vertex_host(effective_region);
                format!(
                    "https://{}/v1beta/projects/{}/locations/{}/cachedContents",
                    host, project_id, effective_region
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

    /// Build the Vertex AI base host for a given region.
    ///
    /// The global endpoint does not use a region prefix — it is simply
    /// `aiplatform.googleapis.com`. Regional endpoints use `{region}-aiplatform.googleapis.com`.
    /// See: <https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations>
    fn vertex_host(region: &str) -> String {
        if region == "global" {
            "aiplatform.googleapis.com".to_string()
        } else {
            format!("{}-aiplatform.googleapis.com", region)
        }
    }

    /// Build the URL for a Gemini API endpoint.
    ///
    /// Uses the model capability profile (see `MODEL_PROFILES`) to determine:
    /// - whether "-preview" must be appended (`auto_preview_suffix`)
    /// - whether Vertex AI must use the global endpoint (`requires_global_vertex`)
    ///
    /// Provider namespace prefixes (e.g. `"vertexai:"`) are stripped first so
    /// that callers can use the same model ID for both endpoints.
    fn build_url(&self, model: &str, action: &str) -> String {
        let bare = Self::strip_provider_prefix(model);
        let profile = Self::lookup_profile(bare);

        let model_name = if profile.auto_preview_suffix && !bare.ends_with("-preview") {
            format!("{}-preview", bare)
        } else {
            bare.to_string()
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
                // Gemini 3.x preview models are global-endpoint-only; regional
                // endpoints return HTTP 404 regardless of quota.
                // See: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
                let effective_region: &str = if profile.requires_global_vertex {
                    "global"
                } else {
                    region.as_str()
                };
                let host = Self::vertex_host(effective_region);
                format!(
                    "https://{}/v1/projects/{}/locations/{}/publishers/google/models/{}:{}",
                    host, project_id, effective_region, model_name, action
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

    /// Convert ChatMessage slice to Gemini Content format.
    ///
    /// # Multi-turn Tool Calling (CRITICAL)
    ///
    /// Gemini enforces **strict role alternation** (`user` ↔ `model`) in `contents`.
    /// The correct wire format for a tool-calling turn is:
    ///
    /// ```text
    /// user   → { parts: [{ text }] }
    /// model  → { parts: [{ functionCall: { name, args } }, ...] }
    /// user   → { parts: [{ functionResponse: { name, response } }, ...] }  ← ALL results in ONE Content
    /// model  → { parts: [{ text }] }
    /// ```
    ///
    /// Two bugs existed before this fix:
    ///
    /// 1. **Assistant messages with tool_calls** were serialised as plain text, losing
    ///    the `functionCall` Parts entirely.  Gemini had no record of which functions it
    ///    called, so it couldn't correlate the results on the next turn.
    ///
    /// 2. **Tool-result messages** were each converted to a *separate* `user` Content
    ///    block.  This created consecutive `user` turns which Gemini rejects with a 400,
    ///    and even if it didn't reject them, the results were sent as plain text rather
    ///    than structured `functionResponse` objects.
    ///
    /// This implementation:
    /// - Emits `functionCall` Parts for every `ToolCall` in an assistant message.
    /// - Groups ALL consecutive `Tool`/`Function` messages into **one** `user` Content
    ///   with one `functionResponse` Part per message.
    fn convert_messages(messages: &[ChatMessage]) -> (Option<Content>, Vec<Content>) {
        let mut system_instruction = None;
        let mut contents: Vec<Content> = Vec::new();

        let mut i = 0;
        while i < messages.len() {
            let msg = &messages[i];
            match msg.role {
                ChatRole::System => {
                    // Gemini uses a separate system_instruction field.
                    system_instruction = Some(Content {
                        parts: vec![Part {
                            text: Some(msg.content.clone()),
                            ..Default::default()
                        }],
                        role: None,
                    });
                    i += 1;
                }
                ChatRole::User => {
                    // OODA-54: Check if message has images for multipart content.
                    if msg.has_images() {
                        let mut parts = Vec::new();

                        // Add text part first (if non-empty).
                        if !msg.content.is_empty() {
                            parts.push(Part {
                                text: Some(msg.content.clone()),
                                ..Default::default()
                            });
                        }

                        // Add image parts.
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
                    i += 1;
                }
                ChatRole::Assistant => {
                    // Build Parts: optional text followed by one functionCall Part per
                    // tool call.  Sending both text and functionCall in the same Content
                    // is valid Gemini API (the model sometimes thinks aloud before calling
                    // a function).
                    let mut parts = Vec::new();

                    if !msg.content.is_empty() {
                        parts.push(Part {
                            text: Some(msg.content.clone()),
                            ..Default::default()
                        });
                    }

                    if let Some(ref tool_calls) = msg.tool_calls {
                        for tc in tool_calls {
                            // Parse args to a JSON Value; fall back to empty object on
                            // malformed JSON so we never send invalid wire data.
                            let args: serde_json::Value =
                                serde_json::from_str(&tc.function.arguments).unwrap_or_else(|_| {
                                    serde_json::Value::Object(serde_json::Map::new())
                                });
                            parts.push(Part {
                                function_call: Some(FunctionCall {
                                    name: tc.function.name.clone(),
                                    args,
                                }),
                                // Gemini 3.x: echo the thought_signature back exactly
                                // as received.  Without it the API returns HTTP 400.
                                thought_signature: tc.thought_signature.clone(),
                                ..Default::default()
                            });
                        }
                    }

                    // Gemini requires at least one Part per Content block.
                    if parts.is_empty() {
                        parts.push(Part {
                            text: Some(String::new()),
                            ..Default::default()
                        });
                    }

                    contents.push(Content {
                        parts,
                        role: Some("model".to_string()),
                    });
                    i += 1;
                }
                ChatRole::Tool | ChatRole::Function => {
                    // GROUP all consecutive tool-result messages into ONE user Content.
                    //
                    // WHY: Gemini enforces strict role alternation (user ↔ model).
                    // After the model issues N parallel function calls, each result would
                    // naïvely become its own separate `user` Content, creating N consecutive
                    // "user" turns — the API rejects this with HTTP 400.
                    //
                    // Correct format: one `user` Content with N `functionResponse` Parts,
                    // one per tool result, matching the order of the model's `functionCall`
                    // Parts.
                    //
                    // The function name comes from `msg.name`, which is propagated by
                    // `build_chat_messages` in edgecrab-core (from the internal Message's
                    // `name` field set by `Message::tool_result`).
                    let mut parts = Vec::new();

                    while i < messages.len()
                        && matches!(messages[i].role, ChatRole::Tool | ChatRole::Function)
                    {
                        let tool_msg = &messages[i];

                        // Use msg.name as the Gemini FunctionResponse name.
                        // Falls back to "unknown_function" when the caller didn't set it.
                        let fn_name = tool_msg.name.as_deref().unwrap_or("unknown_function");

                        // Wrap content as a JSON object.  Gemini requires the response
                        // field to be a JSON object, not a scalar or array.
                        // If the content is already valid JSON, use it as-is.
                        // Otherwise wrap in {"content":"..."}.
                        let response_value: serde_json::Value =
                            serde_json::from_str(&tool_msg.content).unwrap_or_else(
                                |_| serde_json::json!({ "content": tool_msg.content.clone() }),
                            );

                        parts.push(Part {
                            function_response: Some(FunctionResponse {
                                name: fn_name.to_string(),
                                response: response_value,
                            }),
                            ..Default::default()
                        });

                        i += 1;
                    }

                    if !parts.is_empty() {
                        contents.push(Content {
                            parts,
                            role: Some("user".to_string()),
                        });
                    }
                    // `i` already points past all consumed tool messages;
                    // do NOT increment again here.
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

    /// Normalize JSON Schema into Gemini's narrower function-declaration subset.
    ///
    /// WHY: EdgeCrab tool schemas target OpenAI/Anthropic strict mode, which
    /// allows constructs Gemini's native Schema proto does not. Gemini function
    /// declarations support a smaller OpenAPI-style schema where `type` is a
    /// single string and nullability is expressed as `nullable: true`.
    fn sanitize_parameters(params: serde_json::Value) -> serde_json::Value {
        match params {
            serde_json::Value::Object(mut obj) => {
                obj.remove("$schema");
                obj.remove("strict");
                obj.remove("additionalProperties");
                obj.remove("anyOf");
                obj.remove("oneOf");
                obj.remove("allOf");
                obj.remove("not");
                obj.remove("if");
                obj.remove("then");
                obj.remove("else");
                obj.remove("dependentRequired");
                obj.remove("dependentSchemas");
                obj.remove("unevaluatedProperties");
                obj.remove("patternProperties");
                obj.remove("propertyNames");

                if let Some(type_value) = obj.get("type").cloned() {
                    match type_value {
                        serde_json::Value::Array(types) => {
                            let mut nullable = false;
                            let mut non_null_types = Vec::new();

                            for entry in types {
                                match entry {
                                    serde_json::Value::String(s) if s == "null" => nullable = true,
                                    serde_json::Value::String(s) => non_null_types.push(s),
                                    _ => {}
                                }
                            }

                            if let Some(primary) = non_null_types.into_iter().next() {
                                obj.insert("type".into(), serde_json::Value::String(primary));
                                if nullable {
                                    obj.insert("nullable".into(), serde_json::Value::Bool(true));
                                }
                            } else {
                                obj.remove("type");
                            }
                        }
                        serde_json::Value::String(_) => {}
                        _ => {
                            obj.remove("type");
                        }
                    }
                }

                let sanitized = obj
                    .into_iter()
                    .map(|(key, value)| (key, Self::sanitize_parameters(value)))
                    .collect::<serde_json::Map<String, serde_json::Value>>();

                serde_json::Value::Object(sanitized)
            }
            serde_json::Value::Array(values) => serde_json::Value::Array(
                values.into_iter().map(Self::sanitize_parameters).collect(),
            ),
            other => other,
        }
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

        vec![GeminiTool {
            function_declarations: declarations,
        }]
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

    /// Check if the current model supports the `thinkingConfig` in `generationConfig`.
    ///
    /// # Per-model behavior (April 2026)
    ///
    /// | Model                  | Default thinking  | Can disable? | Supports `thinkingLevel`? |
    /// |------------------------|-------------------|--------------|--------------------------|
    /// | gemini-2.5-pro         | Dynamic (on)      | No           | No                       |
    /// | gemini-2.5-flash       | Dynamic (on)      | Yes (budget=0)| No                      |
    /// | gemini-2.5-flash-lite  | Off               | Yes (budget=0)| No                      |
    /// | gemini-3-flash-preview | Dynamic (on)      | Partial (minimal) | Yes                 |
    /// | gemini-3.1-pro-preview | Always on         | No           | Yes                      |
    /// | gemini-3.1-flash-lite-preview | Off (minimal) | Yes     | Yes                      |
    ///
    /// Returns `true` when the model accepts `thinkingConfig` in the request.
    pub fn supports_thinking(&self) -> bool {
        !matches!(self.profile().thinking, ThinkingStyle::None)
    }

    /// Returns `true` when the model's *default* API behaviour includes thinking.
    ///
    /// Derived directly from the model capability profile — no heuristics.
    /// Flash-Lite: `thinks_by_default = false` per the official thinking-budget
    /// table ("Model does not think").
    ///
    /// See: <https://ai.google.dev/gemini-api/docs/thinking>
    pub fn model_thinks_by_default(&self) -> bool {
        self.profile().thinks_by_default
    }

    /// Build a `ThinkingConfig` from caller-supplied `CompletionOptions`.
    ///
    /// Returns `None` when the caller has not requested thinking, in which case
    /// the API uses its model-specific defaults (may think, but won't return
    /// thought summaries).  Dispatches on the profile's `ThinkingStyle` so no
    /// model-name substrings appear here.
    pub fn build_thinking_config(
        &self,
        options: &crate::traits::CompletionOptions,
    ) -> Option<ThinkingConfig> {
        let include_thoughts = options.gemini_include_thoughts;
        let budget = options.gemini_thinking_budget;
        let level = options.gemini_thinking_level.clone();

        // Emit a config only when the caller has set at least one thinking field.
        if include_thoughts.is_none() && budget.is_none() && level.is_none() {
            return None;
        }

        match self.profile().thinking {
            // Model family does not accept thinkingConfig — silently drop it so
            // callers don't have to guard against the provider at the call site.
            ThinkingStyle::None => None,

            // Gemini 3.x: `thinkingLevel` enum.
            // Sending `thinkingBudget` to 3.x Pro causes unexpected behaviour.
            ThinkingStyle::Level => Some(ThinkingConfig {
                include_thoughts,
                thinking_level: level
                    .or_else(|| include_thoughts.filter(|&v| v).map(|_| "high".to_string())),
                thinking_budget: None,
            }),

            // Gemini 2.5: `thinkingBudget` integer; -1 = dynamic.
            // `thinkingLevel` is a no-op on 2.5 per the API spec.
            ThinkingStyle::Budget { .. } => Some(ThinkingConfig {
                include_thoughts,
                thinking_level: None,
                thinking_budget: budget.or_else(|| include_thoughts.filter(|&v| v).map(|_| -1i32)),
            }),
        }
    }

    /// Returns the correct `ThinkingConfig` for a given model.
    ///
    /// # Deprecated
    ///
    /// Enables thinking unconditionally.  Prefer [`build_thinking_config`] which
    /// respects the caller's opt-in.  Kept only for unit tests.
    #[cfg(test)]
    pub fn default_thinking_config(model: &str) -> ThinkingConfig {
        let profile = Self::lookup_profile(model);
        match profile.thinking {
            ThinkingStyle::Level => ThinkingConfig {
                include_thoughts: Some(true),
                thinking_level: Some("high".to_string()),
                thinking_budget: None,
            },
            _ => ThinkingConfig {
                include_thoughts: Some(true),
                thinking_level: None,
                thinking_budget: Some(-1),
            },
        }
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

        // Build generation config from caller options (DRY — all option-to-config
        // mapping lives in apply_generation_options; thinking dispatch is in
        // build_thinking_config which uses the model profile, not heuristics).
        let mut generation_config = GenerationConfig::default();
        Self::apply_generation_options(&mut generation_config, &options);
        if let Some(thinking_cfg) = self.build_thinking_config(&options) {
            generation_config.thinking_config = Some(thinking_cfg);
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
        };

        let url = self.build_url(&self.model, "generateContent");
        debug!("Sending request to Gemini: {}", url);

        let response: GenerateContentResponse = self.send_request(&url, &request).await?;

        // Check for prompt-level blocking before extracting candidates.
        // When the prompt itself is blocked by safety filters, Gemini returns
        // an empty candidates list together with promptFeedback.blockReason.
        // Surfacing the block reason avoids a confusing "No candidates" error.
        let candidates = match response.candidates {
            Some(c) if !c.is_empty() => c,
            _ => {
                let block_reason = response
                    .prompt_feedback
                    .as_ref()
                    .and_then(|pf| pf.block_reason.as_deref())
                    .unwrap_or("unknown");
                return Err(LlmError::ApiError(format!(
                    "Gemini blocked the request (promptFeedback.blockReason: {}). \
                     Adjust your prompt or safety settings.",
                    block_reason
                )));
            }
        };

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
            cache_write_tokens: None,
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

        let mut generation_config = GenerationConfig::default();
        Self::apply_generation_options(&mut generation_config, &options);
        if let Some(thinking_cfg) = self.build_thinking_config(&options) {
            generation_config.thinking_config = Some(thinking_cfg);
        }

        // Convert tools to Gemini format
        let gemini_tools = if tools.is_empty() {
            None
        } else {
            Some(Self::convert_tools(tools))
        };

        let gemini_tool_config = Self::convert_tool_choice(tool_choice);

        let request = GenerateContentRequest {
            contents,
            generation_config: Some(generation_config),
            system_instruction,
            safety_settings: None,
            cached_content: None,
            tools: gemini_tools,
            tool_config: gemini_tool_config,
        };

        let url = self.build_url(&self.model, "generateContent");
        debug!("Sending chat_with_tools request to Gemini: {}", url);

        let response: GenerateContentResponse = self.send_request(&url, &request).await?;

        // Check for prompt-level blocking before extracting candidates.
        let candidates = match response.candidates {
            Some(c) if !c.is_empty() => c,
            _ => {
                let block_reason = response
                    .prompt_feedback
                    .as_ref()
                    .and_then(|pf| pf.block_reason.as_deref())
                    .unwrap_or("unknown");
                return Err(LlmError::ApiError(format!(
                    "Gemini blocked the request (promptFeedback.blockReason: {}). \
                     Adjust your prompt or safety settings.",
                    block_reason
                )));
            }
        };

        let candidate = candidates
            .first()
            .ok_or_else(|| LlmError::ApiError("Empty candidates array".to_string()))?;

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut thinking_content_parts: Vec<String> = Vec::new();

        // pending_sig: carries thought_signature from a non-functionCall Part
        // (e.g. thinking Part for Gemini 2.5) forward to the next functionCall Part.
        let mut pending_sig: Option<String> = None;

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
                // Resolve thought_signature: prefer sig on this Part (Gemini 3),
                // fall back to pending_sig from a preceding non-FC Part (Gemini 2.5).
                let sig = part
                    .thought_signature
                    .clone()
                    .or_else(|| pending_sig.take());
                tool_calls.push(ToolCall {
                    id: format!("call_{}", uuid::Uuid::new_v4().to_string().replace('-', "")),
                    call_type: "function".to_string(),
                    function: crate::traits::FunctionCall {
                        name: fc.name.clone(),
                        arguments: fc.args.to_string(),
                    },
                    // Gemini 3.x: capture thought_signature so it can be echoed
                    // back in subsequent requests (required or API returns 400).
                    thought_signature: sig,
                });
            } else if part.thought_signature.is_some() {
                // Capture sig from non-FC Part (Gemini 2.5: sig lives on thinking Part).
                pending_sig = part.thought_signature.clone();
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
            cache_write_tokens: None,
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

        // No thinking config for simple stream — the `stream()` method has no
        // CompletionOptions parameter, so thinking cannot be opted into here.
        // Users who want thinking with streaming should use `chat_with_tools_stream`.
        let generation_config = None::<GenerationConfig>;

        let request = GenerateContentRequest {
            contents,
            generation_config,
            system_instruction,
            safety_settings: None,
            cached_content: None,
            tools: None,
            tool_config: None,
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
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            // Map 429 / RESOURCE_EXHAUSTED to RateLimited so retry logic can act on it.
            // This error is returned before any streaming bytes are emitted, so
            // retrying is fully safe (no duplicate tokens in the TUI).
            if status.as_u16() == 429 || text.contains("RESOURCE_EXHAUSTED") {
                return Err(LlmError::RateLimited(format!("Stream error: {}", text)));
            }
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
                            if let Ok(chunk) =
                                serde_json::from_str::<GenerateContentResponse>(json_str)
                            {
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
        // Gemini 1.5+ supports JSON mode (including 2.x, 3.x)
        self.model.contains("gemini-1.5")
            || self.model.contains("gemini-2")
            || self.model.contains("gemini-3")
    }

    // OODA-07: Enable function calling for Gemini
    fn supports_function_calling(&self) -> bool {
        // Gemini 1.5+ and 2.x support function calling
        self.model.contains("gemini-1.5")
            || self.model.contains("gemini-2")
            || self.model.contains("gemini-3")
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

        let options = options.cloned().unwrap_or_default();
        let mut generation_config = GenerationConfig::default();
        Self::apply_generation_options(&mut generation_config, &options);
        if let Some(thinking_cfg) = self.build_thinking_config(&options) {
            generation_config.thinking_config = Some(thinking_cfg);
        }

        // Build request with tools
        let request = GenerateContentRequest {
            contents,
            generation_config: Some(generation_config),
            system_instruction,
            safety_settings: None,
            cached_content: None,
            tools: gemini_tools,
            tool_config,
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
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            // Map 429 / RESOURCE_EXHAUSTED to RateLimited so retry logic can act on it.
            if status.as_u16() == 429 || text.contains("RESOURCE_EXHAUSTED") {
                return Err(LlmError::RateLimited(format!("Stream error: {}", text)));
            }
            return Err(LlmError::ApiError(format!("Stream error: {}", text)));
        }

        let stream = response.bytes_stream();

        // Stable counter for ToolCallDelta indices across the lifetime of this stream.
        //
        // WHY Arc<AtomicUsize>: Gemini emits each function call as a complete Part
        // (unlike OpenAI which streams arguments incrementally).  The accumulator in
        // `api_call_streaming` (edgecrab-core) uses a BTreeMap keyed by `index`.  If
        // every ToolCallDelta has index=0, only ONE tool call is ever kept — all
        // subsequent ones silently overwrite it.  Using a shared counter ensures each
        // function call gets a unique, monotonically-increasing index regardless of
        // which SSE packet or `data:` line it appears in.
        let fn_call_index = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // WHY pending_sig: The official Gemini API delivers thoughtSignature differently
        // depending on model version and whether the response includes a functionCall:
        //
        //   • Gemini 3 (FC response):  signature is on the FIRST functionCall Part —
        //     must be echoed back (mandatory, else HTTP 400).
        //   • Gemini 2.5 (FC response): signature is on the FIRST Part regardless of
        //     type (often a thinking Part that precedes the functionCall Part).
        //   • Streaming edge case: signature may arrive in a separate empty-text Part
        //     in a DIFFERENT SSE chunk from the functionCall Part.
        //
        // Solution: capture any thought_signature seen on a non-functionCall Part and
        // carry it as `pending_sig` across chunks.  When the next functionCall Part is
        // processed, fall back to pending_sig if the Part itself has no signature.
        let pending_sig: std::sync::Arc<std::sync::Mutex<Option<String>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let latest_usage: std::sync::Arc<std::sync::Mutex<Option<StreamUsage>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));

        // Use flat_map so EVERY StreamChunk from a single SSE packet is emitted.
        //
        // WHY flat_map instead of map: A single `data:` line can contain Parts for
        // text, thinking, AND one or more function calls simultaneously.  The previous
        // `map` returned only the FIRST chunk via `chunks.into_iter().next()`, silently
        // dropping everything else.  flat_map flattens the Vec<Result<StreamChunk>>
        // from each SSE packet into individual stream items.
        let mapped_stream = stream.flat_map(move |result| {
            let fn_call_index = fn_call_index.clone();
            let pending_sig = pending_sig.clone();
            let latest_usage = latest_usage.clone();
            let items: Vec<crate::error::Result<StreamChunk>> = match result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    let mut chunks: Vec<crate::error::Result<StreamChunk>> = Vec::new();

                    // Parse each `data:` line in the SSE response.
                    for line in text.lines() {
                        if let Some(json_str) = line.strip_prefix("data: ") {
                            if let Ok(sse_response) =
                                serde_json::from_str::<GenerateContentResponse>(json_str)
                            {
                                let stream_usage = Self::stream_usage_from_metadata(
                                    sse_response.usage_metadata.as_ref(),
                                );
                                if let Some(ref usage) = stream_usage {
                                    if let Ok(mut latest) = latest_usage.lock() {
                                        *latest = Some(usage.clone());
                                    }
                                }
                                if let Some(candidates) = sse_response.candidates {
                                    if let Some(candidate) = candidates.first() {
                                        // Process every part — text, thinking, and function calls.
                                        for part in &candidate.content.parts {
                                            // OODA-25: distinguish thinking vs regular text.
                                            if let Some(text_content) = &part.text {
                                                if !text_content.is_empty() {
                                                    if part.thought == Some(true) {
                                                        chunks.push(Ok(
                                                            StreamChunk::ThinkingContent {
                                                                text: text_content.clone(),
                                                                tokens_used: None,
                                                                budget_total: None,
                                                            },
                                                        ));
                                                    } else {
                                                        chunks.push(Ok(StreamChunk::Content(
                                                            text_content.clone(),
                                                        )));
                                                    }
                                                }
                                            }

                                            // Emit one ToolCallDelta per function call with a
                                            // unique, monotonically-increasing index.
                                            if let Some(func_call) = &part.function_call {
                                                let args_json =
                                                    serde_json::to_string(&func_call.args).ok();
                                                let idx = fn_call_index.fetch_add(
                                                    1,
                                                    std::sync::atomic::Ordering::Relaxed,
                                                );
                                                // Resolve thought_signature:
                                                // 1. Prefer the signature on this Part (Gemini 3
                                                //    places it directly on the functionCall Part).
                                                // 2. Fall back to pending_sig captured from a
                                                //    preceding non-FC Part (Gemini 2.5 places it
                                                //    on the FIRST Part regardless of type, which
                                                //    is often a thinking Part; streaming can also
                                                //    deliver the sig in a separate SSE chunk
                                                //    before the functionCall chunk arrives).
                                                let sig =
                                                    part.thought_signature.clone().or_else(|| {
                                                        pending_sig
                                                            .lock()
                                                            .ok()
                                                            .and_then(|mut s| s.take())
                                                    });
                                                chunks.push(Ok(StreamChunk::ToolCallDelta {
                                                    index: idx,
                                                    id: Some(uuid::Uuid::new_v4().to_string()),
                                                    function_name: Some(func_call.name.clone()),
                                                    function_arguments: args_json,
                                                    // Preserve thought_signature so the
                                                    // streaming assembler can store it on
                                                    // the ToolCall for history replay.
                                                    thought_signature: sig,
                                                }));
                                            } else if part.thought_signature.is_some() {
                                                // Part has a thoughtSignature but no functionCall.
                                                // Save it as pending so the NEXT functionCall Part
                                                // (possibly in a later SSE chunk) can use it.
                                                // This handles Gemini 2.5 (sig on thinking Part)
                                                // and Gemini 3 streaming edge cases (sig on empty
                                                // text Part preceding the functionCall Part).
                                                if let Ok(mut ps) = pending_sig.lock() {
                                                    *ps = part.thought_signature.clone();
                                                }
                                            }
                                        }

                                        // Emit Finished after all parts of this candidate.
                                        if let Some(ref reason) = candidate.finish_reason {
                                            let mapped_reason = match reason.as_str() {
                                                "STOP" => "stop",
                                                "MAX_TOKENS" => "length",
                                                "SAFETY" => "content_filter",
                                                _ => reason.as_str(),
                                            };
                                            let usage = stream_usage.or_else(|| {
                                                latest_usage
                                                    .lock()
                                                    .ok()
                                                    .and_then(|latest| latest.clone())
                                            });
                                            chunks.push(Ok(StreamChunk::Finished {
                                                reason: mapped_reason.to_string(),
                                                ttft_ms: None,
                                                usage,
                                            }));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Metadata-only SSE packets (e.g. usageMetadata) produce no content
                    // chunks; returning an empty vec is correct here — flat_map skips them.
                    chunks
                }
                Err(e) => vec![Err(LlmError::ApiError(format!("Stream error: {}", e)))],
            };
            futures::stream::iter(items)
        });

        Ok(mapped_stream.boxed())
    }

    // OODA-08: Enable streaming with tools for Gemini
    fn supports_tool_streaming(&self) -> bool {
        // Same models that support function calling also support streaming with tools
        self.model.contains("gemini-1.5")
            || self.model.contains("gemini-2")
            || self.model.contains("gemini-3")
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

        // VertexAI uses a different embedding API format (:predict with instances)
        if matches!(&self.endpoint, GeminiEndpoint::VertexAI { .. }) {
            return self.embed_vertex_ai(texts).await;
        }

        // Google AI: Use batch endpoint for multiple texts
        if texts.len() > 1 {
            return self.embed_batch(texts).await;
        }

        // Google AI: Single text - use embedContent
        // Pass output_dimensionality when non-default dimension is configured
        let output_dim =
            if self.embedding_dimension != Self::dimension_for_model(&self.embedding_model) {
                Some(self.embedding_dimension)
            } else {
                None
            };

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
            output_dimensionality: output_dim,
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
    /// Embed multiple texts using batch endpoint (Google AI only).
    ///
    /// WHY: Batch endpoint is more efficient for multiple texts (single API call).
    /// Each request in batch MUST include the model field per Gemini API spec.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // WHY: batchEmbedContents requires model field in each request
        // Format: "models/{model_name}"
        let model_path = format!("models/{}", self.embedding_model);

        // Pass output_dimensionality when non-default dimension is configured
        let output_dim =
            if self.embedding_dimension != Self::dimension_for_model(&self.embedding_model) {
                Some(self.embedding_dimension)
            } else {
                None
            };

        let requests: Vec<EmbedContentRequest> = texts
            .iter()
            .map(|text| EmbedContentRequest {
                content: Content {
                    parts: vec![Part {
                        text: Some(text.clone()),
                        ..Default::default()
                    }],
                    role: None,
                },
                model: Some(model_path.clone()),
                task_type: Some("RETRIEVAL_DOCUMENT".to_string()),
                title: None,
                output_dimensionality: output_dim,
            })
            .collect();

        let batch_request = BatchEmbedContentsRequest { requests };

        let url = self.build_url(&self.embedding_model, "batchEmbedContents");
        debug!("Sending batch embedding request to Gemini: {}", url);

        let response: BatchEmbedContentsResponse = self.send_request(&url, &batch_request).await?;

        Ok(response.embeddings.into_iter().map(|e| e.values).collect())
    }

    /// Embed texts using VertexAI predict endpoint.
    ///
    /// VertexAI uses `:predict` with `instances` format instead of `:embedContent`.
    /// See: https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
    ///
    /// # API Format
    ///
    /// ```text
    /// POST https://{region}-aiplatform.googleapis.com/v1/projects/{project}/
    ///      locations/{region}/publishers/google/models/{model}:predict
    /// {
    ///   "instances": [{ "content": "text" }],
    ///   "parameters": { "outputDimensionality": 3072 }
    /// }
    /// ```
    async fn embed_vertex_ai(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let instances: Vec<VertexAIEmbedInstance> = texts
            .iter()
            .map(|text| VertexAIEmbedInstance {
                content: text.clone(),
                task_type: Some("RETRIEVAL_DOCUMENT".to_string()),
                title: None,
            })
            .collect();

        let request = VertexAIEmbedRequest {
            instances,
            parameters: Some(VertexAIEmbedParameters {
                output_dimensionality: Some(self.embedding_dimension),
                auto_truncate: Some(true),
            }),
        };

        let url = self.build_url(&self.embedding_model, "predict");
        debug!("Sending VertexAI embedding request: {}", url);

        let response: VertexAIEmbedResponse = self.send_request(&url, &request).await?;

        Ok(response
            .predictions
            .into_iter()
            .map(|p| p.embeddings.values)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_context_length_detection() {
        // Gemini 2.0: official input limit = 1,048,576 (same as 2.5-class)
        assert_eq!(
            GeminiProvider::context_length_for_model("gemini-2.0-flash"),
            1_048_576
        );
        // Gemini 1.5 Pro: 2M input window
        assert_eq!(
            GeminiProvider::context_length_for_model("gemini-1.5-pro"),
            2_000_000
        );
        // Gemini 1.0 Pro: 32k input window
        assert_eq!(
            GeminiProvider::context_length_for_model("gemini-1.0-pro"),
            32_000
        );
        // Gemini 2.5 Flash Lite: official input limit = 1,048,576
        // Source: https://ai.google.dev/gemini-api/docs/models
        assert_eq!(
            GeminiProvider::context_length_for_model("gemini-2.5-flash-lite"),
            1_048_576
        );
        // Gemini 2.5 Flash: official input limit = 1,048,576
        assert_eq!(
            GeminiProvider::context_length_for_model("gemini-2.5-flash"),
            1_048_576
        );
        // Gemini 3.1 series: 2M window
        assert_eq!(
            GeminiProvider::context_length_for_model("gemini-3.1-pro-preview"),
            2_000_000
        );
        // Gemini 3 Flash: 2M window
        assert_eq!(
            GeminiProvider::context_length_for_model("gemini-3-flash"),
            2_000_000
        );
        // Gemini 1.5 Flash: 1M window
        assert_eq!(
            GeminiProvider::context_length_for_model("gemini-1.5-flash"),
            1_000_000
        );
    }

    #[test]
    fn test_embedding_dimension_detection() {
        assert_eq!(
            GeminiProvider::dimension_for_model("gemini-embedding-001"),
            3072
        );
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
    fn test_provider_builder_default_embedding() {
        // Default embedding model should be gemini-embedding-001 with 3072 dims
        let provider = GeminiProvider::new("test-key");
        assert_eq!(EmbeddingProvider::model(&provider), "gemini-embedding-001");
        assert_eq!(provider.dimension(), 3072);
    }

    #[test]
    fn test_sanitize_parameters_converts_nullable_type_array() {
        let sanitized = GeminiProvider::sanitize_parameters(json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "content": {
                    "type": ["string", "null"],
                    "description": "Nullable content"
                },
                "count": {
                    "type": "integer"
                }
            },
            "required": ["content", "count"]
        }));

        assert_eq!(sanitized["type"], "object");
        assert!(sanitized.get("additionalProperties").is_none());
        assert_eq!(sanitized["properties"]["content"]["type"], "string");
        assert_eq!(sanitized["properties"]["content"]["nullable"], true);
        assert_eq!(sanitized["properties"]["count"]["type"], "integer");
        assert!(sanitized["properties"]["count"].get("nullable").is_none());
    }

    #[test]
    fn test_sanitize_parameters_strips_gemini_unsupported_keywords() {
        let sanitized = GeminiProvider::sanitize_parameters(json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "strict": true,
            "oneOf": [{"type": "string"}],
            "properties": {
                "patch": {
                    "type": ["string", "null"],
                    "anyOf": [{"type": "string"}],
                    "allOf": [{"minLength": 1}],
                    "if": {"type": "string"},
                    "then": {"minLength": 2},
                    "else": {"maxLength": 0},
                    "dependentRequired": {"foo": ["bar"]},
                    "patternProperties": {".*": {"type": "string"}}
                }
            }
        }));

        assert!(sanitized.get("$schema").is_none());
        assert!(sanitized.get("strict").is_none());
        assert!(sanitized.get("oneOf").is_none());
        let patch = &sanitized["properties"]["patch"];
        assert_eq!(patch["type"], "string");
        assert_eq!(patch["nullable"], true);
        assert!(patch.get("anyOf").is_none());
        assert!(patch.get("allOf").is_none());
        assert!(patch.get("if").is_none());
        assert!(patch.get("then").is_none());
        assert!(patch.get("else").is_none());
        assert!(patch.get("dependentRequired").is_none());
        assert!(patch.get("patternProperties").is_none());
    }

    #[test]
    fn test_convert_tools_uses_sanitized_gemini_parameters() {
        let tools = vec![ToolDefinition::function(
            "write_file",
            "Write a file",
            json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "content": {
                        "type": ["string", "null"],
                        "description": "Optional scaffold content"
                    }
                },
                "required": ["content"]
            }),
        )];

        let converted = GeminiProvider::convert_tools(&tools);
        let params = converted[0].function_declarations[0]
            .parameters
            .as_ref()
            .expect("parameters should be present");

        assert!(params.get("additionalProperties").is_none());
        assert_eq!(params["properties"]["content"]["type"], "string");
        assert_eq!(params["properties"]["content"]["nullable"], true);
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
        assert_eq!(
            system.unwrap().parts[0].text,
            Some("You are helpful".to_string())
        );
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
        assert!(
            parts[1].get("inlineData").is_some(),
            "Should have inlineData for image"
        );
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

    #[test]
    fn test_build_url_vertex_ai_predict() {
        // VertexAI embedding uses :predict endpoint
        let provider = GeminiProvider::vertex_ai("my-project", "us-central1", "token");
        let url = provider.build_url("gemini-embedding-001", "predict");

        assert!(url.contains("aiplatform.googleapis.com"));
        assert!(url.contains("gemini-embedding-001"));
        assert!(url.contains(":predict"));
        assert!(url.contains("my-project"));
        assert!(url.contains("us-central1"));
    }

    #[test]
    fn test_build_url_vertex_ai_global_region() {
        // Gemini 3.x Preview models require the global endpoint.
        // The URL must NOT be "global-aiplatform.googleapis.com" (invalid).
        // It must be "aiplatform.googleapis.com" (no prefix).
        let provider = GeminiProvider::vertex_ai("my-project", "global", "token");
        let url = provider.build_url("gemini-3-flash-preview", "generateContent");

        assert!(
            url.contains("https://aiplatform.googleapis.com"),
            "global region must use aiplatform.googleapis.com (no prefix), got: {}",
            url
        );
        assert!(
            !url.contains("global-aiplatform"),
            "must NOT contain 'global-aiplatform', got: {}",
            url
        );
        assert!(
            url.contains("/locations/global/"),
            "must contain /locations/global/, got: {}",
            url
        );
        assert!(url.contains("gemini-3-flash-preview"), "got: {}", url);
    }

    #[test]
    fn test_vertex_host_regional() {
        assert_eq!(
            GeminiProvider::vertex_host("us-central1"),
            "us-central1-aiplatform.googleapis.com"
        );
        assert_eq!(
            GeminiProvider::vertex_host("europe-west4"),
            "europe-west4-aiplatform.googleapis.com"
        );
    }

    #[test]
    fn test_vertex_host_global() {
        assert_eq!(
            GeminiProvider::vertex_host("global"),
            "aiplatform.googleapis.com"
        );
    }

    // =========================================================================
    // OODA-25: Thinking Support Tests
    // =========================================================================

    #[test]
    fn test_supports_thinking_gemini_25() {
        // GoogleAI endpoint: thinking IS supported for 2.5+
        let provider = GeminiProvider::new("key").with_model("gemini-2.5-flash");
        assert!(provider.supports_thinking());

        let provider = GeminiProvider::new("key").with_model("gemini-2.5-pro");
        assert!(provider.supports_thinking());

        // VertexAI endpoint: thinking IS supported for 2.5+
        let provider =
            GeminiProvider::vertex_ai("proj", "us-central1", "tok").with_model("gemini-2.5-flash");
        assert!(provider.supports_thinking());

        let provider =
            GeminiProvider::vertex_ai("proj", "us-central1", "tok").with_model("gemini-2.5-pro");
        assert!(provider.supports_thinking());
    }

    #[test]
    fn test_supports_thinking_gemini_3() {
        // GoogleAI endpoint: thinking IS supported for 3.x
        let provider = GeminiProvider::new("key").with_model("gemini-3-flash");
        assert!(provider.supports_thinking());

        let provider = GeminiProvider::new("key").with_model("gemini-3-pro");
        assert!(provider.supports_thinking());

        // VertexAI endpoint: thinking IS supported for 3.x
        let provider =
            GeminiProvider::vertex_ai("proj", "us-central1", "tok").with_model("gemini-3-flash");
        assert!(provider.supports_thinking());

        // Gemini 3.1 also supports thinking
        let provider = GeminiProvider::new("key").with_model("gemini-3.1-pro-preview");
        assert!(provider.supports_thinking());
    }

    #[test]
    fn test_supports_thinking_gemini_1x() {
        // Gemini 1.x models do NOT support thinking on any endpoint
        let provider = GeminiProvider::new("key").with_model("gemini-1.5-flash");
        assert!(!provider.supports_thinking());

        let provider =
            GeminiProvider::vertex_ai("proj", "us-central1", "tok").with_model("gemini-1.5-flash");
        assert!(!provider.supports_thinking());

        let provider = GeminiProvider::new("key").with_model("gemini-1.0-pro");
        assert!(!provider.supports_thinking());

        // Gemini 2.0 models do NOT support thinking
        let provider = GeminiProvider::new("key").with_model("gemini-2.0-flash");
        assert!(!provider.supports_thinking());
    }

    #[test]
    fn test_default_thinking_config_gemini3() {
        // Gemini 3.x must activate thinking via thinking_level (not thinking_budget).
        // Sending include_thoughts without thinking_level returns 400.
        let cfg = GeminiProvider::default_thinking_config("gemini-3-flash-preview");
        assert!(
            cfg.thinking_level.is_some(),
            "Gemini 3 requires thinking_level to be set"
        );
        assert_eq!(cfg.thinking_level.as_deref(), Some("high"));
        assert!(cfg.thinking_budget.is_none());

        let cfg = GeminiProvider::default_thinking_config("gemini-3.1-pro-preview");
        assert_eq!(cfg.thinking_level.as_deref(), Some("high"));
    }

    #[test]
    fn test_default_thinking_config_gemini25() {
        // Gemini 2.5 must activate thinking via thinking_budget (-1 = dynamic).
        let cfg = GeminiProvider::default_thinking_config("gemini-2.5-flash");
        assert!(
            cfg.thinking_budget.is_some(),
            "Gemini 2.5 requires thinking_budget to be set"
        );
        assert_eq!(cfg.thinking_budget, Some(-1));
        assert!(cfg.thinking_level.is_none());

        let cfg = GeminiProvider::default_thinking_config("gemini-2.5-pro");
        assert_eq!(cfg.thinking_budget, Some(-1));
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
        assert_eq!(
            GEMINI_API_BASE,
            "https://generativelanguage.googleapis.com/v1beta"
        );
        assert_eq!(DEFAULT_GEMINI_MODEL, "gemini-2.5-flash");
        assert_eq!(DEFAULT_EMBEDDING_MODEL, "gemini-embedding-001");
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
    fn test_supports_json_mode_gemini_3() {
        // WHY: Gemini 3.x models support JSON mode
        let provider = GeminiProvider::new("key").with_model("gemini-3-flash");
        assert!(provider.supports_json_mode());

        let provider = GeminiProvider::new("key").with_model("gemini-3-pro");
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
            thinking_config: None,
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

    // =========================================================================
    // Multi-turn Tool Calling Tests (fix for VertexAI multi-tool result bug)
    // =========================================================================

    /// Assistant messages that contain tool_calls must be serialised with
    /// functionCall Parts — not plain text.
    #[test]
    fn test_convert_messages_assistant_with_tool_calls() {
        use crate::traits::{FunctionCall as LlmFunctionCall, ToolCall};

        let tc1 = ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: LlmFunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"location":"London"}"#.to_string(),
            },
            thought_signature: None,
        };
        let tc2 = ToolCall {
            id: "call_2".to_string(),
            call_type: "function".to_string(),
            function: LlmFunctionCall {
                name: "get_time".to_string(),
                arguments: r#"{"timezone":"UTC"}"#.to_string(),
            },
            thought_signature: None,
        };

        let msg = ChatMessage::assistant_with_tools("", vec![tc1, tc2]);
        let (_, contents) = GeminiProvider::convert_messages(&[msg]);

        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role.as_deref(), Some("model"));

        let parts = &contents[0].parts;
        // Should have 2 functionCall Parts (empty text is omitted).
        assert_eq!(
            parts.len(),
            2,
            "expected 2 functionCall parts, got {}",
            parts.len()
        );

        let fc1 = parts[0]
            .function_call
            .as_ref()
            .expect("part 0 should be functionCall");
        assert_eq!(fc1.name, "get_weather");
        assert_eq!(fc1.args["location"], "London");

        let fc2 = parts[1]
            .function_call
            .as_ref()
            .expect("part 1 should be functionCall");
        assert_eq!(fc2.name, "get_time");
        assert_eq!(fc2.args["timezone"], "UTC");
    }

    /// Tool-result messages must be converted to functionResponse Parts.
    #[test]
    fn test_convert_messages_tool_result() {
        let mut tool_msg = ChatMessage::tool_result("call_1", r#"{"temp":20}"#);
        tool_msg.name = Some("get_weather".to_string());

        let (_, contents) = GeminiProvider::convert_messages(&[tool_msg]);

        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role.as_deref(), Some("user"));

        let part = &contents[0].parts[0];
        let fr = part
            .function_response
            .as_ref()
            .expect("part should be functionResponse");
        assert_eq!(fr.name, "get_weather");
        assert_eq!(fr.response["temp"], 20);
    }

    /// Plain-text tool results must be wrapped in {"content":"..."} so that
    /// the Gemini API receives a valid JSON object in the response field.
    #[test]
    fn test_convert_messages_tool_result_plain_text() {
        let mut tool_msg = ChatMessage::tool_result("call_1", "done");
        tool_msg.name = Some("run_command".to_string());

        let (_, contents) = GeminiProvider::convert_messages(&[tool_msg]);

        let fr = contents[0].parts[0]
            .function_response
            .as_ref()
            .expect("should be functionResponse");
        assert_eq!(fr.name, "run_command");
        assert_eq!(fr.response["content"], "done");
    }

    /// Multiple consecutive tool results (parallel dispatch) must be grouped
    /// into a SINGLE user Content with one functionResponse Part each.
    /// Two separate user Contents would violate Gemini's alternating-role
    /// constraint and trigger HTTP 400.
    #[test]
    fn test_convert_messages_parallel_tool_results_grouped() {
        let mut tr1 = ChatMessage::tool_result("call_1", r#"{"temp":20}"#);
        tr1.name = Some("get_weather".to_string());

        let mut tr2 = ChatMessage::tool_result("call_2", r#"{"time":"12:00"}"#);
        tr2.name = Some("get_time".to_string());

        let (_, contents) = GeminiProvider::convert_messages(&[tr1, tr2]);

        // Both results must be in ONE Content block.
        assert_eq!(
            contents.len(),
            1,
            "parallel tool results must be grouped into a single user Content"
        );
        assert_eq!(contents[0].role.as_deref(), Some("user"));
        assert_eq!(
            contents[0].parts.len(),
            2,
            "expected one Part per tool result"
        );

        let fr1 = contents[0].parts[0].function_response.as_ref().unwrap();
        let fr2 = contents[0].parts[1].function_response.as_ref().unwrap();

        assert_eq!(fr1.name, "get_weather");
        assert_eq!(fr2.name, "get_time");
    }

    /// Full multi-turn tool-calling conversation must round-trip to the correct
    /// Gemini wire format:
    ///
    ///   user   → text
    ///   model  → functionCall parts
    ///   user   → functionResponse parts (grouped)
    ///   model  → text (final answer)
    #[test]
    fn test_convert_messages_full_tool_calling_turn() {
        use crate::traits::{FunctionCall as LlmFunctionCall, ToolCall};

        let user_msg = ChatMessage::user("What's the weather in London?");

        let tc = ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: LlmFunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"location":"London"}"#.to_string(),
            },
            thought_signature: None,
        };
        let assistant_msg = ChatMessage::assistant_with_tools("", vec![tc]);

        let mut tool_result =
            ChatMessage::tool_result("call_1", r#"{"temp":18,"condition":"cloudy"}"#);
        tool_result.name = Some("get_weather".to_string());

        let final_answer = ChatMessage::assistant("It is 18°C and cloudy in London.");

        let messages = vec![user_msg, assistant_msg, tool_result, final_answer];
        let (_, contents) = GeminiProvider::convert_messages(&messages);

        assert_eq!(contents.len(), 4, "expected 4 Content blocks");

        // 1st: user text
        assert_eq!(contents[0].role.as_deref(), Some("user"));
        assert!(contents[0].parts[0].text.is_some());

        // 2nd: model functionCall
        assert_eq!(contents[1].role.as_deref(), Some("model"));
        assert!(contents[1].parts[0].function_call.is_some());

        // 3rd: user functionResponse
        assert_eq!(contents[2].role.as_deref(), Some("user"));
        assert!(contents[2].parts[0].function_response.is_some());
        let fr = contents[2].parts[0].function_response.as_ref().unwrap();
        assert_eq!(fr.name, "get_weather");

        // 4th: model text
        assert_eq!(contents[3].role.as_deref(), Some("model"));
        assert!(contents[3].parts[0].text.is_some());
    }

    // =========================================================================
    // Model profile registry tests (DRY/SOLID refactor, April 2026)
    // =========================================================================

    /// Every entry in MODEL_PROFILES must have a non-empty prefix and the
    /// table must be ordered most-specific first (longer prefixes before shorter
    /// ones for each family, otherwise the wrong row would match).
    #[test]
    fn test_profile_table_ordering_invariant() {
        for (i, profile_i) in MODEL_PROFILES.iter().enumerate() {
            assert!(
                !profile_i.prefix.is_empty(),
                "profile at index {} has empty prefix",
                i
            );
            // For every subsequent entry j, if the later prefix b is MORE specific
            // than the earlier prefix a (b starts_with a), that is a violation —
            // the more specific entry must appear FIRST in the table.
            for (j, profile_j) in MODEL_PROFILES.iter().enumerate().skip(i + 1) {
                let a = profile_i.prefix; // earlier entry
                let b = profile_j.prefix; // later entry
                if b.starts_with(a) {
                    panic!(
                        "Profile ordering violation: '{}' at index {} is more specific than \
                         '{}' at index {} but appears later. Move more-specific prefixes before \
                         less-specific ones so the first-match lookup returns the right row.",
                        b, j, a, i
                    );
                }
            }
        }
    }

    /// `lookup_profile` must return the most-specific (longest) matching row.
    #[test]
    fn test_lookup_profile_most_specific_wins() {
        // Flash-Lite is more specific than Flash; it must win.
        let p = GeminiProvider::lookup_profile("gemini-2.5-flash-lite");
        assert_eq!(
            p.prefix, "gemini-2.5-flash-lite",
            "flash-lite prefix must match first"
        );
        assert!(!p.thinks_by_default, "Flash-Lite must not think by default");

        let p2 = GeminiProvider::lookup_profile("gemini-2.5-flash");
        assert_eq!(p2.prefix, "gemini-2.5-flash", "flash prefix must match");
        assert!(p2.thinks_by_default, "Flash must think by default");

        let p3 = GeminiProvider::lookup_profile("gemini-3.1-pro-preview");
        assert_eq!(p3.prefix, "gemini-3.1-", "3.1 prefix must match before 3-");
    }

    /// Provider-namespace prefixes must be stripped transparently.
    #[test]
    fn test_lookup_profile_strips_provider_prefix() {
        let bare = GeminiProvider::lookup_profile("gemini-2.5-flash");
        let prefixed = GeminiProvider::lookup_profile("vertexai:gemini-2.5-flash");
        assert_eq!(
            bare.prefix, prefixed.prefix,
            "provider prefix should not affect the resolved model profile"
        );
    }

    /// Unknown model strings fall back to the default profile (no thinking,
    /// 1,048,576 context window).
    #[test]
    fn test_lookup_profile_unknown_returns_default() {
        let p = GeminiProvider::lookup_profile("totally-unknown-model-9000");
        assert_eq!(p.prefix, "", "unknown model must return DEFAULT_PROFILE");
        assert!(
            matches!(p.thinking, ThinkingStyle::None),
            "default profile must not claim thinking support"
        );
    }

    /// `build_thinking_config` must dispatch on `ThinkingStyle`, not on model name.
    #[test]
    fn test_build_thinking_config_uses_style_not_name() {
        use crate::traits::CompletionOptions;

        // 2.5 Flash → Budget style → must emit thinkingBudget, never thinkingLevel.
        let flash = GeminiProvider::new("k").with_model("gemini-2.5-flash");
        let opts = CompletionOptions {
            gemini_include_thoughts: Some(true),
            ..Default::default()
        };
        let cfg = flash
            .build_thinking_config(&opts)
            .expect("should produce config");
        assert!(
            cfg.thinking_budget.is_some(),
            "2.5 must use thinking_budget"
        );
        assert!(
            cfg.thinking_level.is_none(),
            "2.5 must not set thinking_level"
        );

        // Gemini 3 Flash → Level style → must emit thinkingLevel, never thinkingBudget.
        let f3 = GeminiProvider::new("k").with_model("gemini-3-flash");
        let cfg3 = f3
            .build_thinking_config(&opts)
            .expect("should produce config");
        assert!(
            cfg3.thinking_level.is_some(),
            "Gemini 3 must use thinking_level"
        );
        assert!(
            cfg3.thinking_budget.is_none(),
            "Gemini 3 must not set thinking_budget"
        );

        // Flash-Lite → Budget style but thinks_by_default=false; build_thinking_config
        // must still honour an explicit opt-in.
        let lite = GeminiProvider::new("k").with_model("gemini-2.5-flash-lite");
        let cfg_lite = lite
            .build_thinking_config(&opts)
            .expect("flash-lite should accept opt-in");
        assert!(cfg_lite.thinking_budget.is_some());
    }

    /// `build_thinking_config` must return `None` for models with ThinkingStyle::None.
    #[test]
    fn test_build_thinking_config_none_for_no_thinking_model() {
        use crate::traits::CompletionOptions;

        let provider = GeminiProvider::new("k").with_model("gemini-1.5-flash");
        let opts = CompletionOptions {
            gemini_include_thoughts: Some(true),
            gemini_thinking_budget: Some(1024),
            ..Default::default()
        };
        // Silently dropped — caller should not need to guard on provider type.
        assert!(
            provider.build_thinking_config(&opts).is_none(),
            "1.5 models must not emit a ThinkingConfig even when caller opts in"
        );
    }

    /// `build_thinking_config` must return `None` when no thinking option is set,
    /// regardless of model.
    #[test]
    fn test_build_thinking_config_none_when_not_requested() {
        use crate::traits::CompletionOptions;

        let provider = GeminiProvider::new("k").with_model("gemini-2.5-flash");
        let cfg = provider.build_thinking_config(&CompletionOptions::default());
        assert!(
            cfg.is_none(),
            "No thinking fields → None so the API uses its own model defaults"
        );
    }

    /// Flash-Lite profile: `thinks_by_default = false`, budget range 512-24576.
    #[test]
    fn test_flash_lite_profile_correctness() {
        let p = GeminiProvider::lookup_profile("gemini-2.5-flash-lite");
        assert!(!p.thinks_by_default);
        assert!(!p.auto_preview_suffix);
        assert!(!p.requires_global_vertex);
        assert_eq!(p.context_length, 1_048_576);
        assert!(matches!(
            p.thinking,
            ThinkingStyle::Budget {
                min: 512,
                max: 24_576
            }
        ));
    }

    /// Gemini 3.1 profile: global Vertex endpoint, auto-preview suffix.
    #[test]
    fn test_gemini31_profile_correctness() {
        let p = GeminiProvider::lookup_profile("gemini-3.1-pro-preview");
        assert!(p.auto_preview_suffix || p.prefix == "gemini-3.1-"); // prefix matched
        assert_eq!(p.prefix, "gemini-3.1-");
        assert!(p.requires_global_vertex);
        assert!(matches!(p.thinking, ThinkingStyle::Level));
        assert!(p.thinks_by_default);
    }

    /// `apply_generation_options` must apply all five fields onto the config.
    #[test]
    fn test_apply_generation_options_all_fields() {
        use crate::traits::CompletionOptions;

        let opts = CompletionOptions {
            max_tokens: Some(512),
            temperature: Some(0.5),
            top_p: Some(0.9),
            stop: Some(vec!["END".to_string()]),
            response_format: Some("json_object".to_string()),
            ..Default::default()
        };
        let mut config = GenerationConfig::default();
        GeminiProvider::apply_generation_options(&mut config, &opts);

        assert_eq!(config.max_output_tokens, Some(512));
        let temp = config.temperature.expect("temperature must be set");
        assert!((temp - 0.5_f32).abs() < 0.001, "temperature mismatch");
        let top_p = config.top_p.expect("top_p must be set");
        assert!((top_p - 0.9_f32).abs() < 0.001, "top_p mismatch");
        assert_eq!(config.stop_sequences, Some(vec!["END".to_string()]));
        assert_eq!(
            config.response_mime_type.as_deref(),
            Some("application/json")
        );
    }

    /// `apply_generation_options` must not overwrite fields when options are absent.
    #[test]
    fn test_apply_generation_options_missing_fields_not_overwritten() {
        use crate::traits::CompletionOptions;

        let mut config = GenerationConfig {
            max_output_tokens: Some(999), // pre-existing value
            ..Default::default()
        };

        // Empty options: nothing should change.
        GeminiProvider::apply_generation_options(&mut config, &CompletionOptions::default());
        assert_eq!(
            config.max_output_tokens,
            Some(999),
            "pre-existing value must not be clobbered"
        );
        assert!(config.temperature.is_none());
    }

    /// `context_length_for_model` must be consistent with profile table.
    #[test]
    fn test_context_length_via_profile() {
        assert_eq!(
            GeminiProvider::context_length_for_model("gemini-3-flash"),
            2_000_000
        );
        assert_eq!(
            GeminiProvider::context_length_for_model("gemini-3.1-pro-preview"),
            2_000_000
        );
        assert_eq!(
            GeminiProvider::context_length_for_model("gemini-2.5-flash-lite"),
            1_048_576
        );
        assert_eq!(
            GeminiProvider::context_length_for_model("gemini-2.5-pro"),
            1_048_576
        );
        assert_eq!(
            GeminiProvider::context_length_for_model("gemini-2.0-flash"),
            1_048_576
        );
        assert_eq!(
            GeminiProvider::context_length_for_model("gemini-1.5-pro"),
            2_000_000
        );
        assert_eq!(
            GeminiProvider::context_length_for_model("gemini-1.5-flash"),
            1_000_000
        );
        assert_eq!(
            GeminiProvider::context_length_for_model("gemini-1.0-pro"),
            32_000
        );
    }

    /// PromptFeedback deserialization: block_reason must be surfaced.
    #[test]
    fn test_prompt_feedback_deserialization() {
        let json = r#"{
            "candidates": null,
            "promptFeedback": {
                "blockReason": "SAFETY",
                "safetyRatings": []
            }
        }"#;
        let resp: GenerateContentResponse = serde_json::from_str(json).unwrap();
        assert!(resp.candidates.is_none());
        let pf = resp
            .prompt_feedback
            .expect("promptFeedback must deserialize");
        assert_eq!(pf.block_reason.as_deref(), Some("SAFETY"));
    }

    /// Embedding dimension table: gemini-embedding-2 must return 3072.
    #[test]
    fn test_gemini_embedding_2_dimension() {
        assert_eq!(
            GeminiProvider::dimension_for_model("gemini-embedding-2-preview"),
            3072
        );
        assert_eq!(
            GeminiProvider::dimension_for_model("gemini-embedding-001"),
            3072
        );
        assert_eq!(
            GeminiProvider::dimension_for_model("text-embedding-004"),
            768
        );
    }
}
