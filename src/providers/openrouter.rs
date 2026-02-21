//! OpenRouter provider for access to 200+ LLM models.
//!
//! @implements OODA Iteration 02: OpenRouter Provider
//! @implements OODA-72: Dynamic Model Discovery with Caching
//!
//! OpenRouter provides a unified API to access models from:
//! - OpenAI (GPT-4, GPT-4.5, GPT-5)
//! - Anthropic (Claude 3.5, 4, 4.5)
//! - Google (Gemini Pro, Flash)
//! - Meta (LLaMA 3.1, 3.2)
//! - Mistral (Mixtral, Mistral Large)
//! - And 200+ more models
//!
//! # Example
//!
//! ```rust,ignore
//! use edgequake_llm::{OpenRouterProvider, ChatMessage, LLMProvider};
//!
//! // From environment variable
//! let provider = OpenRouterProvider::from_env()?;
//!
//! // With specific model
//! let provider = OpenRouterProvider::new("sk-or-...")
//!     .with_model("anthropic/claude-3.5-sonnet");
//!
//! let response = provider.chat(&[ChatMessage::user("Hello!")], None).await?;
//!
//! // List available models (with caching)
//! let models = provider.list_models_cached(Duration::from_secs(3600)).await?;
//! for model in models.iter().take(5) {
//!     println!("{}: {} ({}K context)", model.id, model.name, model.context_length / 1000);
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, instrument, warn};

use crate::error::{LlmError, Result};
use crate::traits::{
    ChatMessage, ChatRole, CompletionOptions, EmbeddingProvider, FunctionCall, ImageData,
    LLMProvider, LLMResponse, StreamChunk, ToolCall, ToolChoice, ToolDefinition,
};

// ============================================================================
// Vision content helpers
// ============================================================================

/// Build the `content` value for a `RequestMessage`.
///
/// - Text-only: returns a plain JSON string (backward-compatible with OpenRouter API).
/// - With images: returns an OpenAI-compatible content-parts array:
///   `[{"type":"text","text":"…"},{"type":"image_url","image_url":{"url":"data:…"}}]`
///
/// OpenRouter accepts the same multipart format as OpenAI for vision models.
fn openrouter_build_content(msg: &ChatMessage) -> serde_json::Value {
    match &msg.images {
        Some(imgs) if !imgs.is_empty() => {
            let mut parts: Vec<serde_json::Value> = vec![serde_json::json!({
                "type": "text",
                "text": msg.content
            })];
            for img in imgs {
                parts.push(openrouter_build_image_part(img));
            }
            serde_json::Value::Array(parts)
        }
        _ => serde_json::Value::String(msg.content.clone()),
    }
}

/// Build a single OpenAI-compatible `image_url` content part from `ImageData`.
fn openrouter_build_image_part(img: &ImageData) -> serde_json::Value {
    let url = img.to_data_uri();
    let mut image_url = serde_json::json!({ "url": url });
    if let Some(detail) = &img.detail {
        image_url["detail"] = serde_json::Value::String(detail.clone());
    }
    serde_json::json!({ "type": "image_url", "image_url": image_url })
}

// ============================================================================
// Constants
// ============================================================================

/// OpenRouter API base URL.
const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Default model (Claude 3.5 Sonnet via OpenRouter).
const DEFAULT_MODEL: &str = "anthropic/claude-3.5-sonnet";

/// Default max tokens.
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Default max context length
const DEFAULT_MAX_CONTEXT_LENGTH: usize = 128_000;

// ============================================================================
// Request/Response Types (OpenAI-compatible)
// ============================================================================

/// Chat completion request body.
#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<RequestMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<RequestTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct RequestMessage {
    role: String,
    /// Either a plain JSON string (text-only) or an OpenAI-compatible content-parts
    /// array ([{"type":"text",…},{"type":"image_url",…}]) for vision requests.
    content: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<RequestToolCall>>,
}

#[derive(Debug, Serialize)]
struct RequestTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: RequestFunction,
}

#[derive(Debug, Serialize)]
struct RequestFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct RequestToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: RequestFunctionCall,
}

#[derive(Debug, Serialize)]
struct RequestFunctionCall {
    name: String,
    arguments: String,
}

/// Chat completion response.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChatResponse {
    id: String,
    model: String,
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Choice {
    index: usize,
    message: Option<ResponseMessage>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ResponseMessage {
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<ResponseToolCall>>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ResponseToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: Option<String>,
    function: ResponseFunctionCall,
}

#[derive(Debug, Deserialize)]
struct ResponseFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: Option<u32>,
}

/// Error response from OpenRouter.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ErrorDetail {
    message: String,
    code: Option<i32>,
}

/// Stream chunk for SSE responses.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct StreamChunkResponse {
    id: Option<String>,
    model: Option<String>,
    choices: Vec<StreamChoice>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct StreamChoice {
    index: Option<usize>,
    delta: Option<StreamDelta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct StreamDelta {
    role: Option<String>,
    content: Option<String>,
    /// OODA-05: Extended thinking/reasoning content from models like Claude, DeepSeek R1/V3
    /// Uses serde aliases to handle different provider field names
    #[serde(alias = "thinking", alias = "reasoning_content")]
    reasoning: Option<String>,
    tool_calls: Option<Vec<StreamToolCall>>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct StreamToolCall {
    index: Option<usize>,
    id: Option<String>,
    #[serde(rename = "type")]
    call_type: Option<String>,
    function: Option<StreamFunction>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct StreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

// ============================================================================
// Model Discovery Types (OODA-72)
// ============================================================================

/// Response from the /api/v1/models endpoint.
///
/// Contains a list of all available models with their metadata.
#[derive(Debug, Deserialize, Clone)]
pub struct ModelsResponse {
    pub data: Vec<ModelInfo>,
}

/// Information about a single model available through OpenRouter.
///
/// Includes pricing, capabilities, and context length.
#[derive(Debug, Deserialize, Clone)]
pub struct ModelInfo {
    /// Unique model identifier (e.g., "openai/gpt-4o")
    pub id: String,
    /// Human-readable name (e.g., "GPT-4o")
    pub name: String,
    /// Maximum context window in tokens
    #[serde(default)]
    pub context_length: usize,
    /// Pricing information
    #[serde(default)]
    pub pricing: ModelPricing,
    /// Model architecture details
    #[serde(default)]
    pub architecture: ModelArchitecture,
    /// Supported API parameters
    #[serde(default)]
    pub supported_parameters: Vec<String>,
    /// Model description
    #[serde(default)]
    pub description: Option<String>,
    /// Unix timestamp when model was added
    #[serde(default)]
    pub created: Option<u64>,
}

/// Pricing information for an OpenRouter model.
///
/// All values are in USD per token/request.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct ModelPricing {
    /// Cost per input token (as string, e.g., "0.00003")
    #[serde(default)]
    pub prompt: String,
    /// Cost per output token
    #[serde(default)]
    pub completion: String,
    /// Fixed cost per request
    #[serde(default)]
    pub request: Option<String>,
    /// Cost per image input
    #[serde(default)]
    pub image: Option<String>,
}

/// Architecture details for an OpenRouter model.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct ModelArchitecture {
    /// Input modalities (e.g., ["text", "image"])
    #[serde(default)]
    pub input_modalities: Vec<String>,
    /// Output modalities (e.g., ["text"])
    #[serde(default)]
    pub output_modalities: Vec<String>,
    /// Tokenizer used (e.g., "GPT", "Claude")
    #[serde(default)]
    pub tokenizer: Option<String>,
    /// Instruction format type
    #[serde(default)]
    pub instruct_type: Option<String>,
}

/// Internal cache for model list.
///
/// Uses timestamp to determine cache freshness.
#[derive(Debug)]
struct ModelCache {
    models: Vec<ModelInfo>,
    fetched_at: Instant,
}

// ============================================================================
// OpenRouterProvider Implementation
// ============================================================================

/// OpenRouter LLM provider.
///
/// Provides access to 200+ models through a single API.
/// Supports dynamic model discovery with caching.
#[derive(Debug)]
pub struct OpenRouterProvider {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
    max_tokens: u32,
    max_context_length: usize,
    site_url: Option<String>,
    site_name: Option<String>,
    /// Cached model list (thread-safe with interior mutability)
    model_cache: Arc<RwLock<Option<ModelCache>>>,
}

// Manual Clone implementation since RwLock<Option<ModelCache>> doesn't derive Clone
impl Clone for OpenRouterProvider {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            max_context_length: self.max_context_length,
            site_url: self.site_url.clone(),
            site_name: self.site_name.clone(),
            // Share the same cache across clones
            model_cache: Arc::clone(&self.model_cache),
        }
    }
}

impl OpenRouterProvider {
    /// Create a new OpenRouter provider with the given API key.
    ///
    /// # Arguments
    ///
    /// * `api_key` - OpenRouter API key (starts with `sk-or-`)
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("Failed to create HTTP client"),
            api_key: api_key.into(),
            base_url: OPENROUTER_BASE_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            max_context_length: DEFAULT_MAX_CONTEXT_LENGTH,
            site_url: None,
            site_name: None,
            model_cache: Arc::new(RwLock::new(None)),
        }
    }

    /// Create from environment variable `OPENROUTER_API_KEY`.
    ///
    /// # Errors
    ///
    /// Returns error if `OPENROUTER_API_KEY` is not set.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENROUTER_API_KEY").map_err(|_| {
            LlmError::ConfigError(
                "OPENROUTER_API_KEY environment variable not set. \
                 Get your API key at https://openrouter.ai/keys"
                    .to_string(),
            )
        })?;

        let mut provider = Self::new(api_key);

        // Optional model override
        if let Ok(model) = std::env::var("OPENROUTER_MODEL") {
            provider.model = model;
        }

        // Optional site URL and name
        if let Ok(url) = std::env::var("OPENROUTER_SITE_URL") {
            provider.site_url = Some(url);
        }
        if let Ok(name) = std::env::var("OPENROUTER_SITE_NAME") {
            provider.site_name = Some(name);
        }

        Ok(provider)
    }

    /// Set the model to use.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the base URL (for testing or proxies).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set the maximum tokens for responses.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the site URL for OpenRouter dashboard tracking.
    pub fn with_site_url(mut self, url: impl Into<String>) -> Self {
        self.site_url = Some(url.into());
        self
    }

    /// Set the site name for OpenRouter dashboard tracking.
    pub fn with_site_name(mut self, name: impl Into<String>) -> Self {
        self.site_name = Some(name.into());
        self
    }

    /// Get the chat completions endpoint URL.
    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    /// Build headers for API requests.
    fn headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .expect("Invalid API key format"),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        // Optional tracking headers
        if let Some(ref url) = self.site_url {
            if let Ok(value) = HeaderValue::from_str(url) {
                headers.insert("HTTP-Referer", value);
            }
        }
        if let Some(ref name) = self.site_name {
            if let Ok(value) = HeaderValue::from_str(name) {
                headers.insert("X-Title", value);
            }
        }

        headers
    }

    /// Convert EdgeCode messages to OpenRouter format.
    fn convert_messages(messages: &[ChatMessage]) -> Vec<RequestMessage> {
        messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    ChatRole::System => "system",
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                    ChatRole::Tool => "tool",
                    ChatRole::Function => "function",
                };

                // Build content: plain string for text-only, content-parts array for vision.
                let content = openrouter_build_content(msg);

                RequestMessage {
                    role: role.to_string(),
                    content,
                    tool_call_id: msg.tool_call_id.clone(),
                    tool_calls: msg.tool_calls.as_ref().map(|calls| {
                        calls
                            .iter()
                            .map(|tc| RequestToolCall {
                                id: tc.id.clone(),
                                call_type: "function".to_string(),
                                function: RequestFunctionCall {
                                    name: tc.function.name.clone(),
                                    arguments: tc.function.arguments.clone(),
                                },
                            })
                            .collect()
                    }),
                }
            })
            .collect()
    }

    /// Convert EdgeCode tools to OpenRouter format.
    fn convert_tools(tools: &[ToolDefinition]) -> Vec<RequestTool> {
        tools
            .iter()
            .map(|tool| RequestTool {
                tool_type: "function".to_string(),
                function: RequestFunction {
                    name: tool.function.name.clone(),
                    description: tool.function.description.clone(),
                    parameters: tool.function.parameters.clone(),
                },
            })
            .collect()
    }

    /// Convert EdgeCode ToolChoice to OpenRouter format.
    fn convert_tool_choice(choice: &ToolChoice) -> serde_json::Value {
        match choice {
            ToolChoice::Auto(_) => serde_json::json!("auto"),
            ToolChoice::Required(_) => serde_json::json!("required"),
            ToolChoice::Function {
                choice_type: _,
                function,
            } => serde_json::json!({
                "type": "function",
                "function": { "name": function.name }
            }),
        }
    }

    /// Parse response into LLMResponse.
    fn parse_response(response: ChatResponse) -> LLMResponse {
        let choice = response.choices.first();
        let message = choice.and_then(|c| c.message.as_ref());

        let content = message
            .and_then(|m| m.content.as_ref())
            .cloned()
            .unwrap_or_default();

        let tool_calls = message
            .and_then(|m| m.tool_calls.as_ref())
            .map(|tcs| {
                tcs.iter()
                    .map(|tc| ToolCall {
                        id: tc.id.clone(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        },
                    })
                    .collect()
            })
            .unwrap_or_default();

        let usage = response.usage.as_ref();
        let prompt_tokens = usage.map(|u| u.prompt_tokens as usize).unwrap_or(0);
        let completion_tokens = usage.map(|u| u.completion_tokens as usize).unwrap_or(0);

        LLMResponse {
            content,
            tool_calls,
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            model: response.model,
            finish_reason: choice.and_then(|c| c.finish_reason.clone()),
            metadata: HashMap::new(),
            cache_hit_tokens: None,
            thinking_tokens: None,
            thinking_content: None,
        }
    }

    /// Check if an error is retryable.
    ///
    /// OODA-15: Retry logic for rate limits and transient errors.
    #[allow(dead_code)]
    fn is_retryable_error(error: &LlmError) -> bool {
        match error {
            LlmError::RateLimited(_) | LlmError::NetworkError(_) => true,
            LlmError::ApiError(msg) => {
                msg.contains("502") || msg.contains("503") || msg.contains("504")
            }
            _ => false,
        }
    }

    /// Check if HTTP status is retryable.
    fn is_retryable_status(status: reqwest::StatusCode) -> bool {
        matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504)
    }

    /// Handle API error responses with helpful guidance.
    fn handle_error(status: reqwest::StatusCode, body: &str) -> LlmError {
        // Try to parse error response
        if let Ok(error_response) = serde_json::from_str::<ErrorResponse>(body) {
            let message = error_response.error.message;
            match status.as_u16() {
                400 => {
                    // Bad Request - check for common issues
                    if message.to_lowercase().contains("tool")
                        || message.to_lowercase().contains("function")
                        || message.contains("not supported")
                        || message.contains("No endpoints found")
                    {
                        LlmError::InvalidRequest(format!(
                            "Model doesn't support function calling: {}.\n\
                             \n\
                             💡 EdgeCode React agent requires function calling support.\n\
                             \n\
                             Try one of these compatible models:\n\
                             - anthropic/claude-3.5-sonnet (recommended)\n\
                             - openai/gpt-4o\n\
                             - google/gemini-2.0-flash-exp\n\
                             - meta-llama/llama-3.3-70b-instruct\n\
                             \n\
                             Use /model to select a different model.",
                            message
                        ))
                    } else {
                        LlmError::InvalidRequest(format!(
                            "{}. Check that the model name is correct and the request format is valid.",
                            message
                        ))
                    }
                }
                401 => LlmError::AuthError(message),
                402 => {
                    // Insufficient credits
                    LlmError::ApiError(format!(
                        "Insufficient credits: {}. Add credits at https://openrouter.ai/credits",
                        message
                    ))
                }
                403 => {
                    // Forbidden - usually regional restrictions or model not available
                    if message.contains("not available in your region")
                        || message.contains("region")
                    {
                        LlmError::ApiError(format!(
                            "Regional restriction: {}. This model is not available in your geographic region. Try selecting a different model with /model or check OpenRouter's model availability at https://openrouter.ai/docs/models",
                            message
                        ))
                    } else if message.contains("moderation") {
                        LlmError::ApiError(format!(
                            "Content policy violation: {}. Your request was blocked by content moderation. Review OpenRouter's policies or try a different model.",
                            message
                        ))
                    } else {
                        LlmError::ApiError(format!(
                            "Access forbidden: {}. This may be due to model availability, account restrictions, or content policy. Try a different model with /model",
                            message
                        ))
                    }
                }
                404 => {
                    // Not Found - usually incorrect model name
                    LlmError::ApiError(format!(
                        "Model not found: {}. The model name may be incorrect or the model may have been removed. Use /model to see available models.",
                        message
                    ))
                }
                429 => LlmError::RateLimited(message),
                _ => LlmError::ApiError(format!("{}: {}", status, message)),
            }
        } else {
            // Couldn't parse error response - provide basic status code info
            match status.as_u16() {
                403 => LlmError::ApiError(format!(
                    "403 Forbidden: {}. This model may not be available in your region or due to account restrictions. Try selecting a different model with /model",
                    body
                )),
                404 => LlmError::ApiError(format!(
                    "404 Not Found: {}. The model may not exist or has been removed. Use /model to see available models.",
                    body
                )),
                _ => LlmError::ApiError(format!("{}: {}", status, body)),
            }
        }
    }

    /// Send a chat request (non-streaming) with automatic retry.
    ///
    /// OODA-15: Implements exponential backoff for rate limits and transient errors.
    /// Retries up to 3 times with delays: 1s, 2s, 4s.
    #[instrument(skip(self, request))]
    async fn send_request(&self, request: &ChatRequest<'_>) -> Result<ChatResponse> {
        const MAX_RETRIES: u32 = 3;
        const BASE_DELAY_MS: u64 = 1000;

        let mut last_error = None;

        for attempt in 0..=MAX_RETRIES {
            if attempt > 0 {
                let delay_ms = BASE_DELAY_MS * (1 << (attempt - 1)); // 1s, 2s, 4s
                debug!(
                    "Retry attempt {}/{} after {}ms delay",
                    attempt, MAX_RETRIES, delay_ms
                );
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }

            // OODA-16: Enhanced request logging with message count and tools
            debug!(
                "OpenRouter request: model={}, messages={}, tools={}, max_tokens={:?}, stream={:?}",
                request.model,
                request.messages.len(),
                request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
                request.max_tokens,
                request.stream
            );

            let start = std::time::Instant::now();
            let response = match self
                .client
                .post(self.endpoint())
                .headers(self.headers())
                .json(request)
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(e) => {
                    let error = LlmError::NetworkError(e.to_string());
                    if attempt < MAX_RETRIES {
                        warn!(
                            "Network error (attempt {}/{}): {}",
                            attempt + 1,
                            MAX_RETRIES + 1,
                            e
                        );
                        last_error = Some(error);
                        continue;
                    }
                    return Err(error);
                }
            };

            let status = response.status();
            let body = match response.text().await {
                Ok(b) => b,
                Err(e) => {
                    let error = LlmError::NetworkError(e.to_string());
                    if attempt < MAX_RETRIES {
                        last_error = Some(error);
                        continue;
                    }
                    return Err(error);
                }
            };

            if !status.is_success() {
                let error = Self::handle_error(status, &body);

                // Only retry on retryable errors
                if Self::is_retryable_status(status) && attempt < MAX_RETRIES {
                    warn!(
                        "Retryable error (attempt {}/{}): {:?}",
                        attempt + 1,
                        MAX_RETRIES + 1,
                        error
                    );
                    last_error = Some(error);
                    continue;
                }
                return Err(error);
            }

            // OODA-16: Parse response and log success with timing and token usage
            let elapsed = start.elapsed();
            let response: ChatResponse = serde_json::from_str(&body)
                .map_err(|e| LlmError::ApiError(format!("Failed to parse response: {}", e)))?;

            debug!(
                "OpenRouter response: elapsed={}ms, prompt_tokens={}, completion_tokens={}, total_tokens={}, finish_reason={:?}",
                elapsed.as_millis(),
                response.usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0),
                response.usage.as_ref().map(|u| u.completion_tokens).unwrap_or(0),
                response.usage.as_ref().and_then(|u| u.total_tokens).unwrap_or(0),
                response.choices.first().and_then(|c| c.finish_reason.as_deref()).unwrap_or("none")
            );

            return Ok(response);
        }

        // Should not reach here, but just in case
        Err(last_error.unwrap_or_else(|| LlmError::ApiError("Max retries exceeded".to_string())))
    }

    // ========================================================================
    // Model Discovery Methods (OODA-72)
    // ========================================================================

    /// List all available models from the OpenRouter API.
    ///
    /// This method always fetches fresh data from the API.
    /// For cached access, use `list_models_cached()` instead.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let provider = OpenRouterProvider::from_env()?;
    /// let models = provider.list_models().await?;
    /// for model in models.iter().take(10) {
    ///     println!("{}: {}", model.id, model.name);
    /// }
    /// ```
    #[instrument(skip(self))]
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        debug!("Fetching models from OpenRouter API");

        let url = format!("{}/models", self.base_url);

        let response = self
            .client
            .get(&url)
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        if !status.is_success() {
            return Err(Self::handle_error(status, &body));
        }

        let models_response: ModelsResponse = serde_json::from_str(&body)
            .map_err(|e| LlmError::ApiError(format!("Failed to parse models response: {}", e)))?;

        debug!(
            "Fetched {} models from OpenRouter",
            models_response.data.len()
        );

        // Update cache
        {
            let mut cache = self.model_cache.write().await;
            *cache = Some(ModelCache {
                models: models_response.data.clone(),
                fetched_at: Instant::now(),
            });
        }

        Ok(models_response.data)
    }

    /// List models with caching.
    ///
    /// Returns cached models if cache is valid (within `max_age`).
    /// Otherwise fetches fresh data from the API.
    ///
    /// # Arguments
    ///
    /// * `max_age` - Maximum age of cached data before refresh
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::time::Duration;
    ///
    /// let provider = OpenRouterProvider::from_env()?;
    ///
    /// // Use cached data up to 1 hour old
    /// let models = provider.list_models_cached(Duration::from_secs(3600)).await?;
    /// ```
    pub async fn list_models_cached(&self, max_age: Duration) -> Result<Vec<ModelInfo>> {
        // Check cache first
        {
            let cache = self.model_cache.read().await;
            if let Some(ref cached) = *cache {
                if cached.fetched_at.elapsed() < max_age {
                    debug!(
                        "Using cached models ({} models, age: {:?})",
                        cached.models.len(),
                        cached.fetched_at.elapsed()
                    );
                    return Ok(cached.models.clone());
                }
            }
        }

        // Cache miss or expired, fetch fresh
        self.list_models().await
    }

    /// Invalidate the model cache.
    ///
    /// Next call to `list_models_cached()` will fetch fresh data.
    pub async fn invalidate_model_cache(&self) {
        let mut cache = self.model_cache.write().await;
        *cache = None;
        debug!("Model cache invalidated");
    }

    /// Get a specific model by ID.
    ///
    /// Uses cached data if available (1 hour TTL).
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier (e.g., "openai/gpt-4o")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let provider = OpenRouterProvider::from_env()?;
    /// if let Some(model) = provider.get_model("openai/gpt-4o").await? {
    ///     println!("Context length: {}", model.context_length);
    /// }
    /// ```
    pub async fn get_model(&self, model_id: &str) -> Result<Option<ModelInfo>> {
        let models = self.list_models_cached(Duration::from_secs(3600)).await?;
        Ok(models.into_iter().find(|m| m.id == model_id))
    }

    /// Get models filtered by input modality.
    ///
    /// # Arguments
    ///
    /// * `modality` - Input modality (e.g., "text", "image")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let provider = OpenRouterProvider::from_env()?;
    /// let vision_models = provider.get_models_by_modality("image").await?;
    /// ```
    pub async fn get_models_by_modality(&self, modality: &str) -> Result<Vec<ModelInfo>> {
        let models = self.list_models_cached(Duration::from_secs(3600)).await?;
        Ok(models
            .into_iter()
            .filter(|m| {
                m.architecture
                    .input_modalities
                    .contains(&modality.to_string())
            })
            .collect())
    }

    /// Get the number of cached models.
    pub async fn cached_model_count(&self) -> usize {
        let cache = self.model_cache.read().await;
        cache.as_ref().map(|c| c.models.len()).unwrap_or(0)
    }

    /// Check if the model cache is valid.
    pub async fn is_cache_valid(&self, max_age: Duration) -> bool {
        let cache = self.model_cache.read().await;
        cache
            .as_ref()
            .map(|c| c.fetched_at.elapsed() < max_age)
            .unwrap_or(false)
    }
}

// ============================================================================
// LLMProvider Implementation
// ============================================================================

#[async_trait]
impl LLMProvider for OpenRouterProvider {
    fn name(&self) -> &str {
        "openrouter"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_context_length(&self) -> usize {
        self.max_context_length
    }

    #[instrument(skip(self, prompt))]
    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        self.complete_with_options(prompt, &CompletionOptions::default())
            .await
    }

    #[instrument(skip(self, prompt, options))]
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

    #[instrument(skip(self, messages, options))]
    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let options = options.cloned().unwrap_or_default();

        let request = ChatRequest {
            model: &self.model,
            messages: Self::convert_messages(messages),
            stream: Some(false),
            max_tokens: Some(options.max_tokens.unwrap_or(self.max_tokens as usize) as u32),
            temperature: options.temperature,
            top_p: options.top_p,
            stop: options.stop.clone(),
            tools: None,
            tool_choice: None,
        };

        let response = self.send_request(&request).await?;
        Ok(Self::parse_response(response))
    }

    #[instrument(skip(self, messages, tools, options))]
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let options = options.cloned().unwrap_or_default();

        let request = ChatRequest {
            model: &self.model,
            messages: Self::convert_messages(messages),
            stream: Some(false),
            max_tokens: Some(options.max_tokens.unwrap_or(self.max_tokens as usize) as u32),
            temperature: options.temperature,
            top_p: options.top_p,
            stop: options.stop.clone(),
            tools: Some(Self::convert_tools(tools)),
            tool_choice: tool_choice.map(|tc| Self::convert_tool_choice(&tc)),
        };

        let response = self.send_request(&request).await?;
        Ok(Self::parse_response(response))
    }

    #[instrument(skip(self, prompt))]
    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        let messages = vec![ChatMessage::user(prompt)];

        let request = ChatRequest {
            model: &self.model,
            messages: Self::convert_messages(&messages),
            stream: Some(true),
            max_tokens: Some(self.max_tokens),
            temperature: None,
            top_p: None,
            stop: None,
            tools: None,
            tool_choice: None,
        };

        let request_body = serde_json::to_string(&request)
            .map_err(|e| LlmError::InvalidRequest(format!("Failed to serialize request: {}", e)))?;

        let client = self.client.clone();
        let endpoint = self.endpoint();
        let headers = self.headers();

        let response = client
            .post(&endpoint)
            .headers(headers)
            .body(request_body)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Self::handle_error(status, &body));
        }

        // OODA-94: Use proper SSE line buffering (same fix as chat_with_tools_stream)
        let mut line_buffer = String::new();

        let stream = response.bytes_stream().map(move |chunk| {
            let chunk = chunk.map_err(|e| LlmError::NetworkError(e.to_string()))?;
            let text = String::from_utf8_lossy(&chunk);

            // OODA-94: Append new bytes to persistent buffer
            line_buffer.push_str(&text);

            let mut content = String::new();

            // OODA-94: Extract only complete lines (ending with \n)
            while let Some(newline_idx) = line_buffer.find('\n') {
                let line = line_buffer[..newline_idx].trim().to_string();
                line_buffer.drain(..=newline_idx);

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        continue;
                    }

                    if let Ok(chunk) = serde_json::from_str::<StreamChunkResponse>(data) {
                        for choice in chunk.choices {
                            if let Some(delta) = choice.delta {
                                if let Some(c) = delta.content {
                                    content.push_str(&c);
                                }
                            }
                        }
                    }
                }
            }

            Ok(content)
        });

        Ok(stream.boxed())
    }

    async fn chat_with_tools_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        let options = options.cloned().unwrap_or_default();

        let request = ChatRequest {
            model: &self.model,
            messages: Self::convert_messages(messages),
            stream: Some(true),
            max_tokens: Some(options.max_tokens.unwrap_or(self.max_tokens as usize) as u32),
            temperature: options.temperature,
            top_p: options.top_p,
            stop: options.stop.clone(),
            tools: Some(Self::convert_tools(tools)),
            tool_choice: tool_choice.map(|tc| Self::convert_tool_choice(&tc)),
        };

        let request_body = serde_json::to_string(&request)
            .map_err(|e| LlmError::InvalidRequest(format!("Failed to serialize request: {}", e)))?;

        let client = self.client.clone();
        let endpoint = self.endpoint();
        let headers = self.headers();

        let response = client
            .post(&endpoint)
            .headers(headers)
            .body(request_body)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Self::handle_error(status, &body));
        }

        // OODA-94: Use proper SSE line buffering to prevent argument truncation
        // WHY: Network chunks may split SSE lines at arbitrary byte boundaries.
        // Without buffering, incomplete lines like "data: {\"tool\":\"mkdir" get lost
        // when the next chunk contains the rest: " -p ./demo/snake\"}"
        // HOW: Buffer incomplete lines across chunks, only process complete lines (ending in \n)
        let mut line_buffer = String::new();

        let stream = response
            .bytes_stream()
            .map(move |chunk| -> Result<Vec<StreamChunk>> {
                let chunk = chunk.map_err(|e| LlmError::NetworkError(e.to_string()))?;
                let text = String::from_utf8_lossy(&chunk);

                // OODA-94: Append new bytes to persistent buffer
                line_buffer.push_str(&text);

                let mut chunks = Vec::new();

                // OODA-94: Extract and process only complete lines (ending with \n)
                // Keep incomplete lines in buffer for next chunk
                while let Some(newline_idx) = line_buffer.find('\n') {
                    let line = line_buffer[..newline_idx].trim().to_string();
                    line_buffer.drain(..=newline_idx);

                    if line.is_empty() || line.starts_with(':') {
                        continue;
                    }

                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            chunks.push(StreamChunk::Finished {
                                reason: "stop".to_string(),
                                ttft_ms: None,
                            });
                            continue;
                        }

                        if let Ok(chunk_response) =
                            serde_json::from_str::<StreamChunkResponse>(data)
                        {
                            for choice in chunk_response.choices {
                                if let Some(delta) = choice.delta {
                                    // OODA-05: Check for reasoning/thinking content first
                                    // Models like Claude Extended Thinking, DeepSeek R1/V3
                                    // OODA-10: Added budget_total field
                                    if let Some(reasoning) = delta.reasoning {
                                        if !reasoning.is_empty() {
                                            chunks.push(StreamChunk::ThinkingContent {
                                                text: reasoning,
                                                tokens_used: None, // Not available in stream
                                                budget_total: None, // OODA-10: Provider-specific
                                            });
                                        }
                                    }
                                    if let Some(content) = delta.content {
                                        if !content.is_empty() {
                                            chunks.push(StreamChunk::Content(content));
                                        }
                                    }
                                    if let Some(tool_calls) = delta.tool_calls {
                                        for tc in tool_calls {
                                            if let Some(func) = tc.function {
                                                let args = func.arguments.unwrap_or_default();
                                                if !args.is_empty() || tc.id.is_some() {
                                                    chunks.push(StreamChunk::ToolCallDelta {
                                                        index: tc.index.unwrap_or(0),
                                                        id: tc.id,
                                                        function_name: func.name,
                                                        function_arguments: if args.is_empty() {
                                                            None
                                                        } else {
                                                            Some(args)
                                                        },
                                                    });
                                                }
                                            }
                                        }
                                    }
                                }
                                if let Some(reason) = choice.finish_reason {
                                    chunks.push(StreamChunk::Finished {
                                        reason,
                                        ttft_ms: None,
                                    });
                                }
                            }
                        }
                    }
                }
                Ok(chunks)
            })
            .flat_map(|result: Result<Vec<StreamChunk>>| {
                futures::stream::iter(match result {
                    Ok(chunks) => chunks.into_iter().map(Ok).collect::<Vec<_>>(),
                    Err(e) => vec![Err(e)],
                })
            });

        Ok(stream.boxed())
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    fn supports_tool_streaming(&self) -> bool {
        true
    }
}

// OpenRouter doesn't provide embeddings
#[async_trait]
impl EmbeddingProvider for OpenRouterProvider {
    fn name(&self) -> &str {
        "openrouter"
    }

    fn model(&self) -> &str {
        "none"
    }

    fn dimension(&self) -> usize {
        0
    }

    fn max_tokens(&self) -> usize {
        0
    }

    async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Err(LlmError::InvalidRequest(
            "OpenRouter does not support embeddings. Use a dedicated embedding provider."
                .to_string(),
        ))
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_provider() {
        let provider = OpenRouterProvider::new("test-key");
        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.model, DEFAULT_MODEL);
        assert_eq!(provider.base_url, OPENROUTER_BASE_URL);
    }

    #[test]
    fn test_with_model() {
        let provider = OpenRouterProvider::new("test-key").with_model("openai/gpt-4o");
        assert_eq!(provider.model, "openai/gpt-4o");
    }

    #[test]
    fn test_with_base_url() {
        let provider =
            OpenRouterProvider::new("test-key").with_base_url("https://custom.openrouter.ai");
        assert_eq!(provider.base_url, "https://custom.openrouter.ai");
    }

    #[test]
    fn test_with_site_url() {
        let provider = OpenRouterProvider::new("test-key").with_site_url("https://myapp.com");
        assert_eq!(provider.site_url, Some("https://myapp.com".to_string()));
    }

    #[test]
    fn test_with_site_name() {
        let provider = OpenRouterProvider::new("test-key").with_site_name("My App");
        assert_eq!(provider.site_name, Some("My App".to_string()));
    }

    #[test]
    fn test_from_env_missing_key() {
        // Make sure the env var is not set
        std::env::remove_var("OPENROUTER_API_KEY");
        let result = OpenRouterProvider::from_env();
        assert!(result.is_err());
    }

    #[test]
    fn test_endpoint() {
        let provider = OpenRouterProvider::new("test-key");
        assert_eq!(
            provider.endpoint(),
            format!("{}/chat/completions", OPENROUTER_BASE_URL)
        );
    }

    #[test]
    fn test_headers() {
        let provider = OpenRouterProvider::new("test-key")
            .with_site_url("https://example.com")
            .with_site_name("Example");
        let headers = provider.headers();

        assert!(headers.contains_key(AUTHORIZATION));
        assert!(headers.contains_key(CONTENT_TYPE));
        assert!(headers.contains_key("HTTP-Referer"));
        assert!(headers.contains_key("X-Title"));
    }

    #[test]
    fn test_convert_messages() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello!"),
        ];
        let converted = OpenRouterProvider::convert_messages(&messages);

        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].role, "system");
        assert_eq!(converted[0].content, serde_json::json!("You are helpful."));
        assert_eq!(converted[1].role, "user");
        assert_eq!(converted[1].content, serde_json::json!("Hello!"));
    }

    #[test]
    fn test_convert_messages_with_vision_images() {
        use crate::traits::ImageData;

        let img = ImageData::new("iVBORw0KGgo=", "image/png");
        let messages = vec![ChatMessage::user_with_images(
            "What is in this image?",
            vec![img],
        )];
        let converted = OpenRouterProvider::convert_messages(&messages);

        assert_eq!(converted.len(), 1);
        let content = &converted[0].content;
        assert!(
            content.is_array(),
            "Vision message must serialize as content-parts array"
        );
        let parts = content.as_array().unwrap();
        // First part: text
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "What is in this image?");
        // Second part: image_url
        assert_eq!(parts[1]["type"], "image_url");
        let url = &parts[1]["image_url"]["url"];
        assert!(
            url.as_str().unwrap().starts_with("data:image/png;base64,"),
            "Image URL must be a data URI, got: {}",
            url
        );
    }

    #[test]
    fn test_convert_messages_text_only_still_string() {
        // Regression: when there are no images, content must remain a plain string
        // (not an array) for backward compatibility with all OpenRouter models.
        let messages = vec![ChatMessage::user("plain text only")];
        let converted = OpenRouterProvider::convert_messages(&messages);
        let content = &converted[0].content;
        assert!(
            content.is_string(),
            "Text-only message must serialize as a plain JSON string"
        );
        assert_eq!(content.as_str().unwrap(), "plain text only");
    }

    #[test]
    fn test_convert_tool_choice() {
        assert_eq!(
            OpenRouterProvider::convert_tool_choice(&ToolChoice::auto()),
            serde_json::json!("auto")
        );
        assert_eq!(
            OpenRouterProvider::convert_tool_choice(&ToolChoice::required()),
            serde_json::json!("required")
        );
        assert_eq!(
            OpenRouterProvider::convert_tool_choice(&ToolChoice::function("my_func")),
            serde_json::json!({"type": "function", "function": {"name": "my_func"}})
        );
    }

    #[test]
    fn test_parse_response() {
        let response = ChatResponse {
            id: "gen-123".to_string(),
            model: "anthropic/claude-3.5-sonnet".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Some(ResponseMessage {
                    role: "assistant".to_string(),
                    content: Some("Hello!".to_string()),
                    tool_calls: None,
                }),
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: Some(15),
            }),
        };

        let llm_response = OpenRouterProvider::parse_response(response);

        assert_eq!(llm_response.content, "Hello!");
        assert_eq!(llm_response.prompt_tokens, 10);
        assert_eq!(llm_response.completion_tokens, 5);
        assert_eq!(llm_response.total_tokens, 15);
        assert_eq!(llm_response.model, "anthropic/claude-3.5-sonnet");
        assert_eq!(llm_response.finish_reason, Some("stop".to_string()));
    }

    #[test]
    fn test_parse_response_with_tool_calls() {
        let response = ChatResponse {
            id: "gen-456".to_string(),
            model: "openai/gpt-4o".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Some(ResponseMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![ResponseToolCall {
                        id: "call_1".to_string(),
                        call_type: Some("function".to_string()),
                        function: ResponseFunctionCall {
                            name: "get_weather".to_string(),
                            arguments: r#"{"location":"Paris"}"#.to_string(),
                        },
                    }]),
                }),
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: Some(Usage {
                prompt_tokens: 20,
                completion_tokens: 10,
                total_tokens: Some(30),
            }),
        };

        let llm_response = OpenRouterProvider::parse_response(response);

        assert_eq!(llm_response.tool_calls.len(), 1);
        assert_eq!(llm_response.tool_calls[0].id, "call_1");
        assert_eq!(llm_response.tool_calls[0].name(), "get_weather");
        assert!(llm_response.tool_calls[0].arguments().contains("Paris"));
    }

    #[test]
    fn test_provider_trait() {
        let provider = OpenRouterProvider::new("test-key");
        assert_eq!(LLMProvider::name(&provider), "openrouter");
        assert_eq!(LLMProvider::model(&provider), DEFAULT_MODEL);
        assert!(provider.supports_streaming());
        assert!(provider.supports_function_calling());
    }

    // Integration test - requires API key
    #[tokio::test]
    #[ignore]
    async fn test_chat_completion_live() {
        let provider = OpenRouterProvider::from_env().expect("OPENROUTER_API_KEY not set");
        let messages = vec![ChatMessage::user("Say 'hello' and nothing else.")];

        let response = provider.chat(&messages, None).await;
        assert!(response.is_ok());

        let response = response.unwrap();
        assert!(!response.content.is_empty());
        assert!(response.prompt_tokens > 0);
        assert!(response.completion_tokens > 0);
    }

    // ========================================================================
    // Model Discovery Tests (OODA-72)
    // ========================================================================

    #[test]
    fn test_model_info_deserialization() {
        let json = r#"{
            "id": "openai/gpt-4o",
            "name": "GPT-4o",
            "context_length": 128000,
            "pricing": {
                "prompt": "0.000005",
                "completion": "0.000015"
            },
            "architecture": {
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"]
            },
            "supported_parameters": ["temperature", "max_tokens"]
        }"#;

        let model: ModelInfo = serde_json::from_str(json).unwrap();
        assert_eq!(model.id, "openai/gpt-4o");
        assert_eq!(model.name, "GPT-4o");
        assert_eq!(model.context_length, 128000);
        assert_eq!(model.pricing.prompt, "0.000005");
        assert_eq!(model.architecture.input_modalities.len(), 2);
        assert!(model
            .architecture
            .input_modalities
            .contains(&"image".to_string()));
    }

    #[test]
    fn test_models_response_deserialization() {
        let json = r#"{
            "data": [
                {
                    "id": "openai/gpt-4o",
                    "name": "GPT-4o",
                    "context_length": 128000,
                    "pricing": {"prompt": "0.000005", "completion": "0.000015"}
                },
                {
                    "id": "anthropic/claude-3.5-sonnet",
                    "name": "Claude 3.5 Sonnet",
                    "context_length": 200000,
                    "pricing": {"prompt": "0.000003", "completion": "0.000015"}
                }
            ]
        }"#;

        let response: ModelsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.data.len(), 2);
        assert_eq!(response.data[0].id, "openai/gpt-4o");
        assert_eq!(response.data[1].id, "anthropic/claude-3.5-sonnet");
    }

    #[tokio::test]
    async fn test_cache_initially_empty() {
        let provider = OpenRouterProvider::new("test-key");
        assert_eq!(provider.cached_model_count().await, 0);
        assert!(
            !provider
                .is_cache_valid(std::time::Duration::from_secs(3600))
                .await
        );
    }

    #[tokio::test]
    async fn test_provider_clone_shares_cache() {
        let provider1 = OpenRouterProvider::new("test-key");
        let provider2 = provider1.clone();

        // Both should start with empty cache
        assert_eq!(provider1.cached_model_count().await, 0);
        assert_eq!(provider2.cached_model_count().await, 0);

        // Since they share the Arc, they share the cache
        // (We can't test cache population without mocking, but we verify structure)
    }

    // Integration test - requires API key
    #[tokio::test]
    #[ignore]
    async fn test_list_models_live() {
        let provider = OpenRouterProvider::from_env().expect("OPENROUTER_API_KEY not set");

        let models = provider.list_models().await;
        assert!(models.is_ok(), "Failed to list models: {:?}", models);

        let models = models.unwrap();
        assert!(!models.is_empty(), "Should have at least one model");

        // Verify cache was populated
        assert!(provider.cached_model_count().await > 0);
        assert!(
            provider
                .is_cache_valid(std::time::Duration::from_secs(3600))
                .await
        );

        // Find a known model
        let gpt4 = models.iter().find(|m| m.id.contains("gpt-4"));
        assert!(gpt4.is_some(), "Should have a GPT-4 model");
    }

    #[tokio::test]
    #[ignore]
    async fn test_list_models_cached_live() {
        use std::time::Duration;

        let provider = OpenRouterProvider::from_env().expect("OPENROUTER_API_KEY not set");

        // First call should fetch
        let models1 = provider
            .list_models_cached(Duration::from_secs(3600))
            .await
            .unwrap();
        let count1 = models1.len();

        // Second call should use cache
        let models2 = provider
            .list_models_cached(Duration::from_secs(3600))
            .await
            .unwrap();
        assert_eq!(models2.len(), count1, "Cache should return same models");
    }

    #[tokio::test]
    #[ignore]
    async fn test_get_model_live() {
        let provider = OpenRouterProvider::from_env().expect("OPENROUTER_API_KEY not set");

        let model = provider.get_model("openai/gpt-4o").await;
        assert!(model.is_ok());

        let model = model.unwrap();
        assert!(model.is_some(), "Should find gpt-4o model");

        let model = model.unwrap();
        assert_eq!(model.id, "openai/gpt-4o");
        assert!(model.context_length > 0);
    }

    #[tokio::test]
    #[ignore]
    async fn test_get_models_by_modality_live() {
        let provider = OpenRouterProvider::from_env().expect("OPENROUTER_API_KEY not set");

        let vision_models = provider.get_models_by_modality("image").await;
        assert!(vision_models.is_ok());

        let vision_models = vision_models.unwrap();
        // Should have at least some vision models
        assert!(!vision_models.is_empty(), "Should have vision models");

        // All returned models should support image input
        for model in &vision_models {
            assert!(
                model
                    .architecture
                    .input_modalities
                    .contains(&"image".to_string()),
                "Model {} should support image input",
                model.id
            );
        }
    }

    // OODA-94: Test SSE line buffering logic
    // This tests the line buffering algorithm that prevents argument truncation
    // when SSE data is split across network chunks
    #[test]
    fn test_sse_line_buffering_algorithm() {
        // Simulate the line buffering logic used in stream() and chat_with_tools_stream()
        fn process_chunks(chunks: &[&str]) -> Vec<String> {
            let mut line_buffer = String::new();
            let mut complete_lines = Vec::new();

            for chunk in chunks {
                // Append chunk to buffer (simulates: line_buffer.push_str(&text))
                line_buffer.push_str(chunk);

                // Extract complete lines (simulates the while loop)
                while let Some(newline_idx) = line_buffer.find('\n') {
                    let line = line_buffer[..newline_idx].trim().to_string();
                    line_buffer.drain(..=newline_idx);
                    if !line.is_empty() {
                        complete_lines.push(line);
                    }
                }
            }

            complete_lines
        }

        // Test 1: Data split in middle of JSON
        // This is the exact scenario that caused "mkdir ake_gemini406" truncation
        let chunks = vec![
            "data: {\"function\":{\"name\":\"run_command\",\"arguments\":\"{\\\"command\\\":\\\"mkdir",
            " -p ./demo/snake_gemini406\\\"}\"}}",
            "\n",
        ];
        let lines = process_chunks(&chunks);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("mkdir -p ./demo/snake_gemini406"));

        // Test 2: Multiple complete lines in one chunk
        let chunks = vec![
            "data: {\"content\":\"Hello\"}\n",
            "data: {\"content\":\"World\"}\n",
        ];
        let lines = process_chunks(&chunks);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("Hello"));
        assert!(lines[1].contains("World"));

        // Test 3: Partial line at end (should NOT be processed yet)
        let chunks = vec![
            "data: {\"content\":\"Complete\"}\n",
            "data: {\"content\":\"Incomplete",
        ];
        let lines = process_chunks(&chunks);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("Complete"));

        // Test 4: Empty lines and comments should be filtered
        let chunks = vec!["\n", ": comment\n", "data: {\"content\":\"Real\"}\n", "\n"];
        let lines = process_chunks(&chunks);
        assert_eq!(lines.len(), 2); // comment line and data line
        assert!(lines[1].contains("Real"));

        // Test 5: Split exactly at newline character
        let chunks = vec![
            "data: {\"content\":\"First\"}",
            "\ndata: {\"content\":\"Second\"}\n",
        ];
        let lines = process_chunks(&chunks);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("First"));
        assert!(lines[1].contains("Second"));
    }
}
