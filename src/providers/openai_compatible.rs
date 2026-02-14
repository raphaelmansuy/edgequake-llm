//! OpenAI-compatible provider for custom LLM services.
//!
//! @implements OODA-200: Configurable LLM Providers
//!
//! This provider supports any API that follows the OpenAI chat completions format:
//! - Z.ai (GLM models)
//! - DeepSeek
//! - Together AI
//! - Groq
//! - Any OpenAI-compatible endpoint
//!
//! # Configuration Example
//!
//! ```toml
//! [[providers]]
//! name = "zai"
//! display_name = "Z.AI Platform"
//! type = "openai_compatible"
//! api_key_env = "ZAI_API_KEY"
//! base_url = "https://api.z.ai/api/paas/v4"
//! default_llm_model = "glm-4.7"
//!
//! [providers.headers]
//! Accept-Language = "en-US,en"
//!
//! [[providers.models]]
//! name = "glm-4.7"
//! context_length = 128000
//! ```

use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::header::{self, HeaderMap, HeaderName, HeaderValue};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, warn};

use crate::error::{LlmError, Result};
use crate::model_config::{ModelCard, ModelType, ProviderConfig};
use crate::traits::{
    ChatMessage, ChatRole, CompletionOptions, EmbeddingProvider, FunctionCall, LLMProvider,
    LLMResponse, StreamChunk, ToolCall, ToolChoice, ToolDefinition,
};

// ============================================================================
// Request/Response Types (OpenAI-compatible format)
// ============================================================================

/// OpenAI-compatible chat request body.
#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<MessageRequest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    /// Z.ai specific: thinking mode configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
    /// Response format (for JSON mode)
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

// ============================================================================
// Message Content Types for Multimodal Support (OODA-52)
// ============================================================================

/// Request message content - either simple text or multipart with images.
///
/// WHY: Vision-capable models (GPT-4V, GPT-4o) accept content as either:
/// - String: Simple text message
/// - Array: Multipart with text and image_url parts
///
/// Using #[serde(untagged)] allows Serde to serialize the appropriate format
/// based on the variant, maintaining backwards compatibility for text-only.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum RequestContent {
    /// Simple text content (serializes as string)
    Text(String),
    /// Multipart content with text and images (serializes as array)
    Parts(Vec<ContentPart>),
}

/// Content part for multipart messages.
///
/// OpenAI vision API format uses tagged unions with "type" field.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ContentPart {
    /// Text content part
    #[serde(rename = "text")]
    Text { text: String },
    /// Image URL content part (supports base64 data URIs)
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlContent },
}

/// Image URL content for vision models.
///
/// Supports both:
/// - URL references: "https://example.com/image.png"
/// - Base64 data URIs: "data:image/png;base64,..."
#[derive(Debug, Serialize)]
struct ImageUrlContent {
    /// Image URL or data URI
    url: String,
    /// Detail level: "auto" (default), "low", or "high"
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}

// ============================================================================
// Original Request/Response Types
// ============================================================================

#[derive(Debug, Serialize)]
struct MessageRequest {
    role: String,
    /// Content can be string or multipart array for vision models
    content: RequestContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallRequest>>,
}

#[derive(Debug, Serialize)]
struct ToolCallRequest {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: FunctionCallRequest,
}

#[derive(Debug, Serialize)]
struct FunctionCallRequest {
    name: String,
    arguments: String,
}

/// Z.ai thinking mode configuration.
#[derive(Debug, Serialize)]
struct ThinkingConfig {
    #[serde(rename = "type")]
    thinking_type: String, // "enabled" or "disabled"
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String, // "text" or "json_object"
}

/// OpenAI-compatible chat response.
#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[allow(dead_code)]
    id: Option<String>,
    #[allow(dead_code)]
    model: Option<String>,
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    #[allow(dead_code)]
    index: Option<usize>,
    message: Option<MessageContent>,
    // Delta is used for streaming responses (not currently implemented)
    #[allow(dead_code)]
    delta: Option<serde_json::Value>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageContent {
    #[allow(dead_code)]
    role: Option<String>,
    content: Option<String>,
    /// Z.ai specific: reasoning content from thinking mode
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<ToolCallResponse>>,
}

#[derive(Debug, Deserialize)]
struct ToolCallResponse {
    id: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    call_type: Option<String>,
    function: FunctionCallResponse,
}

#[derive(Debug, Deserialize)]
struct FunctionCallResponse {
    name: String,
    arguments: String,
}

/// Completion tokens details (xAI, DeepSeek, OpenAI o-series)
#[derive(Debug, Deserialize, Default)]
struct CompletionTokensDetails {
    /// Tokens used for reasoning/thinking (OODA-28)
    #[serde(default)]
    reasoning_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    #[allow(dead_code)]
    total_tokens: Option<usize>,
    /// Details about completion tokens including reasoning (OODA-28)
    #[serde(default)]
    completion_tokens_details: Option<CompletionTokensDetails>,
}

/// Error response from API.
#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ErrorDetail {
    message: String,
    #[allow(dead_code)]
    code: Option<String>,
}

/// Stream chunk for server-sent events (SSE) streaming.
#[derive(Debug, Deserialize)]
struct ChatStreamChunk {
    #[allow(dead_code)]
    id: Option<String>,
    #[allow(dead_code)]
    model: Option<String>,
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[allow(dead_code)]
    index: Option<usize>,
    delta: Option<StreamDelta>,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
    /// DeepSeek reasoning/thinking content (OODA-27)
    /// Streamed before final content for deepseek-reasoner model
    #[serde(default)]
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct ToolCallDelta {
    index: Option<usize>,
    id: Option<String>,
    function: Option<FunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct FunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

// ============================================================================
// OpenAI-Compatible Provider Implementation
// ============================================================================

/// A configurable provider for any OpenAI-compatible API.
///
/// This provider can connect to any service that implements the OpenAI
/// chat completions API format, including:
/// - Z.ai (GLM models)
/// - DeepSeek
/// - Together AI
/// - Groq
/// - Local services (Ollama, LM Studio)
#[derive(Debug)]
pub struct OpenAICompatibleProvider {
    /// HTTP client with pre-configured headers
    client: Client,
    /// Provider configuration from TOML
    config: ProviderConfig,
    /// API key (resolved from environment)
    api_key: String,
    /// Current model name
    model: String,
    /// Model capabilities from config
    model_card: Option<ModelCard>,
    /// Base URL for API calls
    base_url: String,
}

impl OpenAICompatibleProvider {
    /// Create provider from TOML configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Provider configuration from models.toml
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Required API key environment variable is not set
    /// - Base URL is not configured
    pub fn from_config(config: ProviderConfig) -> Result<Self> {
        // 1. Resolve API key from environment
        let api_key = Self::resolve_api_key(&config)?;

        // 2. Resolve base URL
        let base_url = Self::resolve_base_url(&config)?;

        // 3. Build HTTP client with custom headers
        let client = Self::build_client(&config)?;

        // 4. Get default model
        let model = config
            .default_llm_model
            .clone()
            .unwrap_or_else(|| "default".to_string());

        // 5. Find model card for capabilities
        let model_card = config.models.iter().find(|m| m.name == model).cloned();

        debug!(
            provider = config.name,
            model = model,
            base_url = base_url,
            "Created OpenAI-compatible provider"
        );

        Ok(Self {
            client,
            config,
            api_key,
            model,
            model_card,
            base_url,
        })
    }

    /// Resolve API key from environment variable.
    fn resolve_api_key(config: &ProviderConfig) -> Result<String> {
        if let Some(env_var) = &config.api_key_env {
            std::env::var(env_var).map_err(|_| {
                LlmError::ConfigError(format!(
                    "API key environment variable '{}' not set for provider '{}'. \
                     Please set it with: export {}=your-api-key",
                    env_var, config.name, env_var
                ))
            })
        } else {
            // Some providers don't require API key (local servers)
            Ok(String::new())
        }
    }

    /// Resolve base URL from config or environment.
    fn resolve_base_url(config: &ProviderConfig) -> Result<String> {
        // Check environment variable override first
        if let Some(env_var) = &config.base_url_env {
            if let Ok(url) = std::env::var(env_var) {
                return Ok(url);
            }
        }

        // Use config base_url
        config.base_url.clone().ok_or_else(|| {
            LlmError::ConfigError(format!(
                "Provider '{}' requires 'base_url' or 'base_url_env' to be set",
                config.name
            ))
        })
    }

    /// Build HTTP client with custom headers.
    fn build_client(config: &ProviderConfig) -> Result<Client> {
        let mut headers = HeaderMap::new();

        // Add Content-Type header
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        // NOTE: Not adding User-Agent here - some providers (POE) reject custom agents
        // Reqwest will use its default User-Agent which works better

        // Add custom headers from config
        for (key, value) in &config.headers {
            let header_name = HeaderName::from_bytes(key.as_bytes()).map_err(|e| {
                LlmError::ConfigError(format!("Invalid header name '{}': {}", key, e))
            })?;
            let header_value = HeaderValue::from_str(value).map_err(|e| {
                LlmError::ConfigError(format!("Invalid header value for '{}': {}", key, e))
            })?;
            headers.insert(header_name, header_value);
        }

        Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|e| LlmError::ConfigError(format!("Failed to build HTTP client: {}", e)))
    }

    /// Build the chat completions endpoint URL.
    fn chat_completions_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        format!("{}/chat/completions", base)
    }

    /// Extract content from message, supporting both standard and reasoning content.
    ///
    /// Z.AI and similar providers return content in `reasoning_content` field when
    /// thinking mode is enabled. This function checks both fields and returns the
    /// appropriate content.
    fn extract_content(message: &MessageContent) -> String {
        // Prioritize reasoning_content (Z.AI thinking mode)
        if let Some(ref reasoning) = message.reasoning_content {
            if !reasoning.is_empty() {
                return reasoning.clone();
            }
        }
        // Fall back to standard content field
        message.content.clone().unwrap_or_default()
    }

    /// Convert ChatMessage to request format (OODA-52: supports multimodal).
    ///
    /// WHY: Vision models require content as array of parts when images present.
    /// This function checks for images and builds appropriate format:
    /// - No images: content = "text" (simple string)
    /// - With images: content = [{type: "text", ...}, {type: "image_url", ...}]
    fn convert_messages(messages: &[ChatMessage]) -> Vec<MessageRequest> {
        messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    ChatRole::System => "system",
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                    ChatRole::Tool | ChatRole::Function => "tool",
                };

                // OODA-52: Check if message has images for multipart content
                let content = if msg.has_images() {
                    // Build multipart content with text and images
                    let mut parts = Vec::new();

                    // Add text part first (if not empty)
                    if !msg.content.is_empty() {
                        parts.push(ContentPart::Text {
                            text: msg.content.clone(),
                        });
                    }

                    // Add image parts
                    if let Some(ref images) = msg.images {
                        for img in images {
                            parts.push(ContentPart::ImageUrl {
                                image_url: ImageUrlContent {
                                    url: img.to_data_uri(),
                                    detail: img.detail.clone(),
                                },
                            });
                        }
                    }

                    RequestContent::Parts(parts)
                } else {
                    // Simple text content
                    RequestContent::Text(msg.content.clone())
                };

                MessageRequest {
                    role: role.to_string(),
                    content,
                    tool_call_id: msg.tool_call_id.clone(),
                    tool_calls: None,
                }
            })
            .collect()
    }

    /// Convert ToolChoice to JSON value.
    fn convert_tool_choice(choice: &ToolChoice) -> serde_json::Value {
        match choice {
            ToolChoice::Auto(_) => serde_json::json!("auto"),
            ToolChoice::Required(_) => serde_json::json!("required"),
            ToolChoice::Function { function, .. } => {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": function.name
                    }
                })
            }
        }
    }

    /// Make a non-streaming chat completion request.
    async fn chat_request(&self, request: &ChatRequest<'_>) -> Result<ChatResponse> {
        let url = self.chat_completions_url();

        debug!(
            "OpenAI-compatible API Request: url={} model={} provider={}",
            url, request.model, self.config.name
        );

        let mut req_builder = self.client.post(&url);

        // Add authorization if API key is set
        if !self.api_key.is_empty() {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", self.api_key));
            debug!(
                "API key: {}...",
                &self.api_key[..20.min(self.api_key.len())]
            );
        } else {
            warn!("No API key set for provider: {}", self.config.name);
        }

        let response = req_builder.json(request).send().await.map_err(|e| {
            warn!("Network error calling {} API: {}", self.config.name, e);
            LlmError::NetworkError(format!("Failed to connect to {}: {}", url, e))
        })?;

        let status = response.status();
        debug!("{} API Response: status={}", self.config.name, status);

        let body = response.text().await.map_err(|e| {
            warn!("Failed to read response body: {}", e);
            LlmError::NetworkError(e.to_string())
        })?;

        if !status.is_success() {
            warn!(
                "{} API error: status={} body={}",
                self.config.name, status, body
            );
            // Try to parse error response
            if let Ok(error_resp) = serde_json::from_str::<ErrorResponse>(&body) {
                return Err(LlmError::ApiError(format!(
                    "{} API {}: {}",
                    self.config.name,
                    status.as_u16(),
                    error_resp.error.message
                )));
            }
            return Err(LlmError::ApiError(format!(
                "{} API error {}: {}",
                self.config.name,
                status.as_u16(),
                body
            )));
        }

        debug!(
            "{} API success, parsing response body (length: {})",
            self.config.name,
            body.len()
        );

        // OODA-99.3: Log raw response for GLM debugging
        if self.model.to_lowercase().contains("glm") {
            debug!("GLM response body: {}", &body[..1000.min(body.len())]);
        }

        serde_json::from_str(&body).map_err(|e| {
            warn!(
                "Failed to parse {} response: {} | body: {}",
                self.config.name,
                e,
                &body[..500.min(body.len())]
            );
            LlmError::ApiError(format!(
                "Failed to parse {} response: {} | body preview: {}",
                self.config.name,
                e,
                &body[..500.min(body.len())]
            ))
        })
    }

    /// Set the model for this provider.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        let model_name = model.into();
        self.model_card = self
            .config
            .models
            .iter()
            .find(|m| m.name == model_name)
            .cloned();
        self.model = model_name;
        self
    }

    /// Get the current model card.
    pub fn model_card(&self) -> Option<&ModelCard> {
        self.model_card.as_ref()
    }
}

#[async_trait]
impl LLMProvider for OpenAICompatibleProvider {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_context_length(&self) -> usize {
        self.model_card
            .as_ref()
            .map(|m| m.capabilities.context_length)
            .unwrap_or(128000)
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
        let mut messages = Vec::new();

        if let Some(system) = &options.system_prompt {
            messages.push(ChatMessage::system(system));
        }
        messages.push(ChatMessage::user(prompt));

        self.chat(&messages, Some(options)).await
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let options = options.cloned().unwrap_or_default();
        let messages_req = Self::convert_messages(messages);

        // Check if JSON mode is requested via response_format
        let use_json_mode = options
            .response_format
            .as_ref()
            .map(|f| f == "json_object" || f == "json")
            .unwrap_or(false);

        // Build request (no tools for basic chat)
        let request = ChatRequest {
            model: &self.model,
            messages: messages_req,
            temperature: options.temperature,
            max_tokens: options.max_tokens,
            stream: Some(false),
            tools: None,
            tool_choice: None,
            thinking: if self.config.supports_thinking {
                Some(ThinkingConfig {
                    thinking_type: "enabled".to_string(),
                })
            } else {
                None
            },
            response_format: if use_json_mode {
                Some(ResponseFormat {
                    format_type: "json_object".to_string(),
                })
            } else {
                None
            },
        };

        let response = self.chat_request(&request).await?;

        // Extract content from response
        let choice = response
            .choices
            .first()
            .ok_or_else(|| LlmError::ApiError("No choices in response".to_string()))?;

        let message = choice
            .message
            .as_ref()
            .ok_or_else(|| LlmError::ApiError("No message in choice".to_string()))?;

        let content = message.content.clone().unwrap_or_default();

        // Extract usage including reasoning tokens (OODA-28)
        let (prompt_tokens, completion_tokens, reasoning_tokens) = response
            .usage
            .as_ref()
            .map(|u| {
                let reasoning = u
                    .completion_tokens_details
                    .as_ref()
                    .and_then(|d| d.reasoning_tokens);
                (u.prompt_tokens, u.completion_tokens, reasoning)
            })
            .unwrap_or((0, 0, None));

        let mut llm_response = LLMResponse::new(content, &self.model)
            .with_usage(prompt_tokens, completion_tokens)
            .with_finish_reason(
                choice
                    .finish_reason
                    .clone()
                    .unwrap_or_else(|| "stop".to_string()),
            );

        // Add reasoning tokens if available (xAI, DeepSeek)
        if let Some(tokens) = reasoning_tokens {
            llm_response = llm_response.with_thinking_tokens(tokens);
        }

        Ok(llm_response)
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        debug!(
            "chat_with_tools called: model={} provider={} messages={} tools={}",
            self.model,
            self.config.name,
            messages.len(),
            tools.len()
        );

        let options = options.cloned().unwrap_or_default();
        let messages_req = Self::convert_messages(messages);

        // Build request with tools
        let request = ChatRequest {
            model: &self.model,
            messages: messages_req,
            temperature: options.temperature,
            max_tokens: options.max_tokens,
            stream: Some(false),
            tools: if tools.is_empty() {
                None
            } else {
                Some(tools.to_vec())
            },
            tool_choice: tool_choice.as_ref().map(Self::convert_tool_choice),
            thinking: if self.config.supports_thinking {
                Some(ThinkingConfig {
                    thinking_type: "enabled".to_string(),
                })
            } else {
                None
            },
            response_format: None,
        };

        let response = self.chat_request(&request).await?;

        // Extract content from response
        let choice = response
            .choices
            .first()
            .ok_or_else(|| LlmError::ApiError("No choices in response".to_string()))?;

        let message = choice
            .message
            .as_ref()
            .ok_or_else(|| LlmError::ApiError("No message in choice".to_string()))?;

        let content = Self::extract_content(message);

        // OODA-99.3: Debug log message structure for GLM
        if self.model.to_lowercase().contains("glm") {
            debug!(
                "GLM message structure - content_len={} tool_calls_present={} tool_calls_count={}",
                content.len(),
                message.tool_calls.is_some(),
                message.tool_calls.as_ref().map(|t| t.len()).unwrap_or(0)
            );
        }

        // Extract tool calls if present
        let tool_calls: Vec<ToolCall> = message
            .tool_calls
            .as_ref()
            .map(|calls| {
                calls
                    .iter()
                    .map(|tc| {
                        // OODA-99.3: Debug logging for GLM empty arguments issue
                        if tc.function.arguments.is_empty() || tc.function.arguments == "{}" {
                            warn!(
                                "Empty tool arguments detected - tool={} id={} args='{}' (GLM model may not be providing arguments correctly)",
                                tc.function.name, tc.id, tc.function.arguments
                            );
                        } else {
                            debug!(
                                "Tool call extracted - tool={} args_len={} id={}",
                                tc.function.name, tc.function.arguments.len(), tc.id
                            );
                        }
                        ToolCall {
                            id: tc.id.clone(),
                            call_type: "function".to_string(),
                            function: FunctionCall {
                                name: tc.function.name.clone(),
                                arguments: tc.function.arguments.clone(),
                            },
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Extract usage including reasoning tokens (OODA-28)
        let (prompt_tokens, completion_tokens, reasoning_tokens) = response
            .usage
            .as_ref()
            .map(|u| {
                let reasoning = u
                    .completion_tokens_details
                    .as_ref()
                    .and_then(|d| d.reasoning_tokens);
                (u.prompt_tokens, u.completion_tokens, reasoning)
            })
            .unwrap_or((0, 0, None));

        let mut llm_response = LLMResponse::new(content, &self.model)
            .with_usage(prompt_tokens, completion_tokens)
            .with_tool_calls(tool_calls)
            .with_finish_reason(
                choice
                    .finish_reason
                    .clone()
                    .unwrap_or_else(|| "stop".to_string()),
            );

        // Add reasoning tokens if available (xAI, DeepSeek)
        if let Some(tokens) = reasoning_tokens {
            llm_response = llm_response.with_thinking_tokens(tokens);
        }

        Ok(llm_response)
    }

    fn supports_streaming(&self) -> bool {
        self.model_card
            .as_ref()
            .map(|m| m.capabilities.supports_streaming)
            .unwrap_or(true)
    }

    fn supports_function_calling(&self) -> bool {
        self.model_card
            .as_ref()
            .map(|m| m.capabilities.supports_function_calling)
            .unwrap_or(true)
    }

    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        use futures::StreamExt;

        let messages = vec![MessageRequest {
            role: "user".to_string(),
            content: RequestContent::Text(prompt.to_string()),
            tool_call_id: None,
            tool_calls: None,
        }];

        let request = ChatRequest {
            model: &self.model,
            messages,
            temperature: None,
            max_tokens: None,
            stream: Some(true),
            tools: None,
            tool_choice: None,
            thinking: None,
            response_format: None,
        };

        let url = self.chat_completions_url();

        // DEBUG: Log full request details
        debug!(
            "{} Stream Request: url={} model={}",
            self.config.name, url, &self.model
        );
        debug!(
            "{} Stream Request body: {}",
            self.config.name,
            serde_json::to_string_pretty(&request).unwrap_or_default()
        );

        let mut req_builder = self.client.post(&url);

        if !self.api_key.is_empty() {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", self.api_key));
        }

        let req_builder = req_builder.json(&request);

        use reqwest_eventsource::EventSource;
        let event_source = EventSource::new(req_builder)
            .map_err(|e| {
                let error_msg = e.to_string();
                warn!("Failed to create event source: {}", error_msg);
                // OODA-99: Provide helpful guidance for 400 Bad Request errors
                // OODA-100: Enhanced to also detect tool/function calling issues
                if error_msg.contains("400") && error_msg.contains("Bad Request") {
                    let error_lower = error_msg.to_lowercase();
                    // Check if this is a tool/function calling issue
                    if error_lower.contains("tool") 
                        || error_lower.contains("function")
                        || error_msg.contains("not supported")
                        || error_msg.contains("No endpoints found") {
                        LlmError::ApiError(format!(
                            "stream failed: {}\n\n\
                            💡 Model doesn't support function calling required by EdgeCode React agent.\n\
                            \n\
                            Try one of these compatible models:\n\
                            - anthropic/claude-3.5-sonnet (recommended)\n\
                            - openai/gpt-4o\n\
                            - google/gemini-2.0-flash-exp\n\
                            - meta-llama/llama-3.3-70b-instruct\n\
                            \n\
                            Use /model to select a different model.",
                            error_msg
                        ))
                    } else {
                        // Context window issue
                        LlmError::ApiError(format!(
                            "stream failed: {}\n\n\
                            💡 Troubleshooting 400 Bad Request:\n\
                            \n\
                            If using LMStudio:\n\
                            • The prompt likely exceeds your model's configured context window\n\
                            • Solution 1: Increase context length in LMStudio model settings (32K+ recommended)\n\
                            • Solution 2: Set LMSTUDIO_CONTEXT_LENGTH environment variable (e.g., 32768 or 65536)\n\
                            • Solution 3: Use a model with larger context window\n\
                            • Solution 4: Reduce task complexity or working directory size\n\
                            \n\
                            If using other providers:\n\
                            • Check the model's context limits in the provider's documentation\n\
                            • Reduce the amount of context being sent (files, history, etc.)\n\
                            • Use a model with larger context window",
                            error_msg
                        ))
                    }
                } else {
                    LlmError::ApiError(format!("stream failed: {}", error_msg))
                }
            })?;

        use futures::stream;
        use reqwest_eventsource::Event;

        let stream = stream::unfold(event_source, |mut es| async move {
            match es.next().await {
                Some(Ok(Event::Open)) => {
                    // Connection opened, continue to next event
                    Some((Ok("".to_string()), es))
                }
                Some(Ok(Event::Message(msg))) => {
                    if msg.data == "[DONE]" {
                        es.close();
                        return None;
                    }

                    // Parse SSE chunk
                    match serde_json::from_str::<ChatStreamChunk>(&msg.data) {
                        Ok(chunk) => {
                            if let Some(choice) = chunk.choices.first() {
                                if let Some(ref delta) = choice.delta {
                                    // OODA-27: DeepSeek reasoning_content support
                                    // Note: basic stream() returns String, so we prefix thinking
                                    if let Some(ref reasoning) = delta.reasoning_content {
                                        if !reasoning.is_empty() {
                                            // For basic stream, we emit reasoning as-is
                                            // The consumer can detect it via <think> tags if needed
                                            return Some((Ok(reasoning.clone()), es));
                                        }
                                    }
                                    if let Some(ref content) = delta.content {
                                        if !content.is_empty() {
                                            return Some((Ok(content.clone()), es));
                                        }
                                    }
                                }
                            }
                            // No content in this chunk, continue to next
                            Some((Ok("".to_string()), es))
                        }
                        Err(e) => {
                            warn!("Failed to parse stream chunk: {} | data: {}", e, msg.data);
                            Some((Err(LlmError::ApiError(format!("Parse error: {}", e))), es))
                        }
                    }
                }
                Some(Err(e)) => {
                    es.close();
                    Some((Err(LlmError::ApiError(format!("Stream error: {}", e))), es))
                }
                None => None,
            }
        });

        Ok(stream.boxed())
    }

    fn supports_tool_streaming(&self) -> bool {
        // Tool streaming is supported if both streaming and function calling are supported
        self.supports_streaming() && self.supports_function_calling()
    }

    async fn chat_with_tools_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        use futures::stream::{self, StreamExt};
        use reqwest_eventsource::{Event, EventSource};

        let options = options.cloned().unwrap_or_default();
        let messages_req = Self::convert_messages(messages);

        // Build streaming request with tools
        let request = ChatRequest {
            model: &self.model,
            messages: messages_req,
            temperature: options.temperature,
            max_tokens: options.max_tokens,
            stream: Some(true),
            tools: if tools.is_empty() {
                None
            } else {
                Some(tools.to_vec())
            },
            tool_choice: tool_choice.as_ref().map(Self::convert_tool_choice),
            thinking: if self.config.supports_thinking {
                Some(ThinkingConfig {
                    thinking_type: "enabled".to_string(),
                })
            } else {
                None
            },
            response_format: None,
        };

        let url = self.chat_completions_url();

        debug!(
            "Starting streaming request: model={} url={} tools={}",
            self.model,
            url,
            tools.len()
        );

        let mut req_builder = self.client.post(&url);

        // Add authorization if API key is set
        if !self.api_key.is_empty() {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", self.api_key));
        }

        let req_builder = req_builder.json(&request);

        // Create event source for SSE streaming
        let event_source = EventSource::new(req_builder).map_err(|e| {
            warn!("Failed to create event source: {}", e);
            LlmError::ApiError(format!("Failed to create event source: {}", e))
        })?;

        let stream = stream::unfold(event_source, |mut es| async move {
            match es.next().await {
                Some(Ok(Event::Open)) => {
                    // Connection opened, continue to next event
                    Some((Ok(StreamChunk::Content("".to_string())), es))
                }
                Some(Ok(Event::Message(msg))) => {
                    if msg.data == "[DONE]" {
                        es.close();
                        return None;
                    }

                    // OODA-99.3: Log raw SSE message data for GLM tool debugging
                    if msg.data.contains("tool_calls") || msg.data.contains("write_file") {
                        debug!("RAW SSE message (len={}): {}", msg.data.len(), &msg.data);
                    }

                    // Parse SSE chunk
                    match serde_json::from_str::<ChatStreamChunk>(&msg.data) {
                        Ok(chunk) => {
                            if let Some(choice) = chunk.choices.first() {
                                if let Some(ref delta) = choice.delta {
                                    // OODA-27: DeepSeek reasoning_content (thinking) chunk
                                    // Reasoning content comes first, then final content
                                    if let Some(ref reasoning) = delta.reasoning_content {
                                        if !reasoning.is_empty() {
                                            return Some((
                                                Ok(StreamChunk::ThinkingContent {
                                                    text: reasoning.clone(),
                                                    tokens_used: None, // DeepSeek provides reasoning_tokens in final chunk
                                                    budget_total: None,
                                                }),
                                                es,
                                            ));
                                        }
                                    }

                                    // Content chunk
                                    if let Some(ref content) = delta.content {
                                        if !content.is_empty() {
                                            return Some((
                                                Ok(StreamChunk::Content(content.clone())),
                                                es,
                                            ));
                                        }
                                    }

                                    // Tool call deltas
                                    if let Some(ref tool_calls) = delta.tool_calls {
                                        // OODA-99.3: Log tool call structure for GLM debugging
                                        for (i, tool_call) in tool_calls.iter().enumerate() {
                                            if let Some(ref function) = tool_call.function {
                                                debug!(
                                                    "GLM tool_call[{}] - id={:?} index={:?} name={:?} args_len={} args={:?}",
                                                    i,
                                                    tool_call.id,
                                                    tool_call.index,
                                                    function.name,
                                                    function.arguments.as_ref().map(|s| s.len()).unwrap_or(0),
                                                    function.arguments
                                                );
                                            }
                                        }

                                        for tool_call in tool_calls {
                                            if let Some(ref function) = tool_call.function {
                                                // OODA-99.3: Handle Z.AI GLM models that send name+arguments in single delta
                                                // Must check for BOTH before returning to avoid early return bug
                                                let has_name = function.name.is_some();
                                                let has_args = function.arguments.is_some();

                                                if has_name || has_args {
                                                    return Some((
                                                        Ok(StreamChunk::ToolCallDelta {
                                                            index: tool_call.index.unwrap_or(0),
                                                            id: tool_call.id.clone(),
                                                            function_name: function.name.clone(),
                                                            function_arguments: function
                                                                .arguments
                                                                .clone(),
                                                        }),
                                                        es,
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // Empty chunk, continue
                            Some((Ok(StreamChunk::Content("".to_string())), es))
                        }
                        Err(e) => {
                            warn!("Failed to parse stream chunk: {} | data: {}", e, msg.data);
                            es.close();
                            Some((
                                Err(LlmError::ApiError(format!(
                                    "Failed to parse stream chunk: {} | data: {}",
                                    e, msg.data
                                ))),
                                es,
                            ))
                        }
                    }
                }
                Some(Err(e)) => {
                    // Stream error
                    warn!("Stream error: {}", e);
                    Some((
                        Err(LlmError::NetworkError(format!("Stream error: {}", e))),
                        es,
                    ))
                }
                None => {
                    // Stream ended
                    None
                }
            }
        });

        Ok(Box::pin(stream))
    }
}

// Implement EmbeddingProvider if the provider has embedding models
#[async_trait]
impl EmbeddingProvider for OpenAICompatibleProvider {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn model(&self) -> &str {
        self.config
            .default_embedding_model
            .as_deref()
            .unwrap_or("unknown")
    }

    fn dimension(&self) -> usize {
        // Find embedding model in config
        self.config
            .models
            .iter()
            .find(|m| matches!(m.model_type, ModelType::Embedding))
            .map(|m| m.capabilities.embedding_dimension)
            .unwrap_or(1536)
    }

    fn max_tokens(&self) -> usize {
        // Find embedding model in config and get max tokens
        self.config
            .models
            .iter()
            .find(|m| matches!(m.model_type, ModelType::Embedding))
            .map(|m| m.capabilities.max_output_tokens)
            .unwrap_or(8192)
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Check if we have an embedding model
        let embedding_model = self
            .config
            .default_embedding_model
            .as_ref()
            .ok_or_else(|| {
                LlmError::ConfigError(format!(
                    "Provider '{}' does not have an embedding model configured",
                    self.config.name
                ))
            })?;

        warn!(
            provider = self.config.name,
            model = embedding_model,
            text_count = texts.len(),
            "Embedding not fully implemented for OpenAI-compatible providers"
        );

        Err(LlmError::NotSupported(
            "Embedding support for custom providers is not yet implemented. \
             Use a dedicated embedding provider like OpenAI."
                .to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> ProviderConfig {
        ProviderConfig {
            name: "test-provider".to_string(),
            display_name: "Test Provider".to_string(),
            provider_type: crate::model_config::ProviderType::OpenAICompatible,
            api_key_env: Some("TEST_API_KEY".to_string()),
            base_url: Some("https://api.example.com/v1".to_string()),
            default_llm_model: Some("test-model".to_string()),
            models: vec![ModelCard {
                name: "test-model".to_string(),
                display_name: "Test Model".to_string(),
                model_type: ModelType::Llm,
                capabilities: crate::model_config::ModelCapabilities {
                    context_length: 128000,
                    max_output_tokens: 8192,
                    supports_function_calling: true,
                    supports_streaming: true,
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn test_provider_creation_requires_api_key() {
        let config = create_test_config();

        // Should fail without API key set
        std::env::remove_var("TEST_API_KEY");
        let result = OpenAICompatibleProvider::from_config(config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("TEST_API_KEY"));
    }

    #[test]
    fn test_provider_creation_success() {
        let config = create_test_config();

        std::env::set_var("TEST_API_KEY", "test-key-12345");
        let provider = OpenAICompatibleProvider::from_config(config).unwrap();

        assert_eq!(LLMProvider::name(&provider), "test-provider");
        assert_eq!(LLMProvider::model(&provider), "test-model");
        assert_eq!(provider.max_context_length(), 128000);
        assert!(provider.supports_function_calling());

        std::env::remove_var("TEST_API_KEY");
    }

    #[test]
    fn test_chat_completions_url() {
        std::env::set_var("TEST_API_KEY2", "key");

        let mut config = create_test_config();
        config.api_key_env = Some("TEST_API_KEY2".to_string());
        config.base_url = Some("https://api.z.ai/api/paas/v4".to_string());

        let provider = OpenAICompatibleProvider::from_config(config).unwrap();
        assert_eq!(
            provider.chat_completions_url(),
            "https://api.z.ai/api/paas/v4/chat/completions"
        );

        std::env::remove_var("TEST_API_KEY2");
    }

    #[test]
    fn test_custom_headers() {
        std::env::set_var("TEST_API_KEY3", "key");

        let mut config = create_test_config();
        config.api_key_env = Some("TEST_API_KEY3".to_string());
        config
            .headers
            .insert("Accept-Language".to_string(), "en-US,en".to_string());
        config
            .headers
            .insert("X-Custom-Header".to_string(), "custom-value".to_string());

        let result = OpenAICompatibleProvider::from_config(config);
        assert!(result.is_ok());

        std::env::remove_var("TEST_API_KEY3");
    }

    #[test]
    fn test_convert_messages() {
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("Hello!"),
            ChatMessage::assistant("Hi there!"),
        ];

        let converted = OpenAICompatibleProvider::convert_messages(&messages);

        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0].role, "system");
        assert_eq!(converted[1].role, "user");
        assert_eq!(converted[2].role, "assistant");
    }

    #[test]
    fn test_base_url_env_override() {
        std::env::set_var("TEST_API_KEY4", "key");
        std::env::set_var("CUSTOM_BASE_URL", "https://override.example.com/v1");

        let mut config = create_test_config();
        config.api_key_env = Some("TEST_API_KEY4".to_string());
        config.base_url = Some("https://default.example.com/v1".to_string());
        config.base_url_env = Some("CUSTOM_BASE_URL".to_string());

        let provider = OpenAICompatibleProvider::from_config(config).unwrap();
        assert_eq!(provider.base_url, "https://override.example.com/v1");

        std::env::remove_var("TEST_API_KEY4");
        std::env::remove_var("CUSTOM_BASE_URL");
    }

    #[test]
    fn test_with_model() {
        std::env::set_var("TEST_API_KEY5", "key");

        let mut config = create_test_config();
        config.api_key_env = Some("TEST_API_KEY5".to_string());
        config.models.push(ModelCard {
            name: "another-model".to_string(),
            display_name: "Another Model".to_string(),
            model_type: ModelType::Llm,
            capabilities: crate::model_config::ModelCapabilities {
                context_length: 32000,
                ..Default::default()
            },
            ..Default::default()
        });

        let provider = OpenAICompatibleProvider::from_config(config)
            .unwrap()
            .with_model("another-model");

        assert_eq!(LLMProvider::model(&provider), "another-model");
        assert_eq!(provider.max_context_length(), 32000);

        std::env::remove_var("TEST_API_KEY5");
    }

    // =========================================================================
    // Multipart Message Tests (OODA-52)
    // =========================================================================

    #[test]
    fn test_convert_messages_text_only() {
        let messages = vec![ChatMessage::user("Hello, world!")];
        let converted = OpenAICompatibleProvider::convert_messages(&messages);

        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "user");

        // Text-only should serialize as string
        let json = serde_json::to_value(&converted[0]).unwrap();
        assert_eq!(json["content"], "Hello, world!");
    }

    #[test]
    fn test_convert_messages_with_images() {
        use crate::traits::ImageData;

        let images = vec![ImageData::new("base64data", "image/png")];
        let messages = vec![ChatMessage::user_with_images("What's this?", images)];
        let converted = OpenAICompatibleProvider::convert_messages(&messages);

        assert_eq!(converted.len(), 1);

        // With images should serialize as array
        let json = serde_json::to_value(&converted[0]).unwrap();
        let content = &json["content"];

        assert!(content.is_array());
        assert_eq!(content.as_array().unwrap().len(), 2);

        // First part: text
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "What's this?");

        // Second part: image_url
        assert_eq!(content[1]["type"], "image_url");
        assert!(content[1]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,"));
    }

    #[test]
    fn test_convert_messages_with_image_detail() {
        use crate::traits::ImageData;

        let images = vec![ImageData::new("data", "image/jpeg").with_detail("high")];
        let messages = vec![ChatMessage::user_with_images("Analyze", images)];
        let converted = OpenAICompatibleProvider::convert_messages(&messages);

        let json = serde_json::to_value(&converted[0]).unwrap();
        let content = &json["content"];

        assert_eq!(content[1]["image_url"]["detail"], "high");
    }

    // =========================================================================
    // Reasoning Tokens Tests (OODA-28)
    // =========================================================================

    #[test]
    fn test_usage_with_reasoning_tokens() {
        // Test parsing xAI/DeepSeek usage with reasoning_tokens
        let json = r#"{
            "prompt_tokens": 32,
            "completion_tokens": 9,
            "total_tokens": 135,
            "completion_tokens_details": {
                "reasoning_tokens": 94
            }
        }"#;

        let usage: Usage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.prompt_tokens, 32);
        assert_eq!(usage.completion_tokens, 9);
        assert_eq!(
            usage
                .completion_tokens_details
                .as_ref()
                .unwrap()
                .reasoning_tokens,
            Some(94)
        );
    }

    #[test]
    fn test_usage_without_reasoning_tokens() {
        // Test parsing standard usage without reasoning details
        let json = r#"{
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }"#;

        let usage: Usage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert!(usage.completion_tokens_details.is_none());
    }

    #[test]
    fn test_stream_delta_with_reasoning_content() {
        // Test parsing DeepSeek/xAI streaming delta with reasoning
        let json = r#"{
            "content": null,
            "reasoning_content": "Let me think about this..."
        }"#;

        let delta: StreamDelta = serde_json::from_str(json).unwrap();
        assert!(delta.content.is_none());
        assert_eq!(
            delta.reasoning_content,
            Some("Let me think about this...".to_string())
        );
    }

    // =========================================================================
    // OODA-36: Additional Unit Tests
    // =========================================================================

    #[test]
    fn test_supports_streaming() {
        std::env::set_var("TEST_API_KEY_STREAM", "key");

        let config = create_test_config_with_key("TEST_API_KEY_STREAM");
        let provider = OpenAICompatibleProvider::from_config(config).unwrap();

        // Default model has supports_streaming = true in test config
        assert!(provider.supports_streaming());

        std::env::remove_var("TEST_API_KEY_STREAM");
    }

    #[test]
    fn test_thinking_config_serialization() {
        let config = ThinkingConfig {
            thinking_type: "enabled".to_string(),
        };

        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["type"], "enabled");
    }

    #[test]
    fn test_response_format_serialization() {
        let format = ResponseFormat {
            format_type: "json_object".to_string(),
        };

        let json = serde_json::to_value(&format).unwrap();
        assert_eq!(json["type"], "json_object");
    }

    #[test]
    fn test_embedding_provider_name() {
        std::env::set_var("TEST_API_KEY_EMB1", "key");

        let config = create_test_config_with_key("TEST_API_KEY_EMB1");
        let provider = OpenAICompatibleProvider::from_config(config).unwrap();

        assert_eq!(EmbeddingProvider::name(&provider), "test-provider");

        std::env::remove_var("TEST_API_KEY_EMB1");
    }

    #[test]
    fn test_embedding_provider_model() {
        std::env::set_var("TEST_API_KEY_EMB2", "key");

        let mut config = create_test_config_with_key("TEST_API_KEY_EMB2");
        config.default_embedding_model = Some("text-embedding-ada-002".to_string());
        let provider = OpenAICompatibleProvider::from_config(config).unwrap();

        assert_eq!(
            EmbeddingProvider::model(&provider),
            "text-embedding-ada-002"
        );

        std::env::remove_var("TEST_API_KEY_EMB2");
    }

    #[test]
    fn test_tool_call_request_serialization() {
        let tool_call = ToolCallRequest {
            id: "call_123".to_string(),
            call_type: "function".to_string(),
            function: FunctionCallRequest {
                name: "get_weather".to_string(),
                arguments: r#"{"location":"NYC"}"#.to_string(),
            },
        };

        let json = serde_json::to_value(&tool_call).unwrap();
        assert_eq!(json["id"], "call_123");
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "get_weather");
        assert_eq!(json["function"]["arguments"], r#"{"location":"NYC"}"#);
    }

    #[test]
    fn test_function_call_request_serialization() {
        let func_call = FunctionCallRequest {
            name: "search".to_string(),
            arguments: r#"{"query":"rust programming"}"#.to_string(),
        };

        let json = serde_json::to_value(&func_call).unwrap();
        assert_eq!(json["name"], "search");
        assert_eq!(json["arguments"], r#"{"query":"rust programming"}"#);
    }

    /// Helper to create test config with specific API key env var
    fn create_test_config_with_key(env_var: &str) -> ProviderConfig {
        ProviderConfig {
            name: "test-provider".to_string(),
            display_name: "Test Provider".to_string(),
            provider_type: crate::model_config::ProviderType::OpenAICompatible,
            api_key_env: Some(env_var.to_string()),
            base_url: Some("https://api.test.com/v1".to_string()),
            default_llm_model: Some("test-model".to_string()),
            models: vec![ModelCard {
                name: "test-model".to_string(),
                display_name: "Test Model".to_string(),
                model_type: ModelType::Llm,
                capabilities: crate::model_config::ModelCapabilities {
                    context_length: 128000,
                    supports_streaming: true,
                    supports_function_calling: true,
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        }
    }
}
