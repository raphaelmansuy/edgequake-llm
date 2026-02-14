//! Anthropic (Claude) LLM provider implementation.
//!
//! Supports Anthropic's Messages API for Claude models.
//!
//! # Environment Variables
//! - `ANTHROPIC_API_KEY`: API key for Anthropic API
//!
//! # Models Supported
//! - Claude 4.5 Opus: `claude-opus-4-5-20250929`
//! - Claude 4 Sonnet: `claude-sonnet-4-5-20250929`
//! - Claude 3.5 Sonnet: `claude-3-5-sonnet-20241022`
//! - Claude 3.5 Haiku: `claude-3-5-haiku-20241022`
//! - Claude 3 Opus: `claude-3-opus-20240229`
//!
//! # Example
//! ```ignore
//! use edgequake_llm::AnthropicProvider;
//!
//! let provider = AnthropicProvider::new("your-api-key");
//! let response = provider.chat(&[ChatMessage::user("Hello!")], None).await?;
//! ```

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, instrument, warn};

use crate::error::{LlmError, Result};
use crate::traits::{
    ChatMessage, ChatRole, CompletionOptions, FunctionCall, LLMProvider, LLMResponse, StreamChunk, ToolCall,
    ToolChoice, ToolDefinition,
};

/// Anthropic API base URL
const ANTHROPIC_API_BASE: &str = "https://api.anthropic.com";

/// Anthropic API version (required header)
const ANTHROPIC_API_VERSION: &str = "2023-06-01";

/// Default model
const DEFAULT_MODEL: &str = "claude-sonnet-4-5-20250929";

// ============================================================================
// Anthropic API Request/Response Types
// ============================================================================

/// Message for Anthropic API
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

/// Content can be a string or an array of content blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

// ============================================================================
// Image Support Types (OODA-53)
// ============================================================================
//
// Anthropic uses a different image format than OpenAI:
//
// OpenAI:                            Anthropic:
// ┌─────────────────────────┐        ┌─────────────────────────┐
// │ type: "image_url"       │        │ type: "image"           │
// │ image_url:              │        │ source:                 │
// │   url: "data:..."       │        │   type: "base64"        │
// │   detail: "high"        │        │   media_type: "image/x" │
// └─────────────────────────┘        │   data: "base64..."     │
//                                    └─────────────────────────┘
//
// WHY: Separate struct for clarity and correct serde serialization
// ============================================================================

/// Image source for Anthropic API (base64 encoded images)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImageSource {
    #[serde(rename = "type")]
    source_type: String,  // Always "base64"
    media_type: String,   // MIME type, e.g., "image/png"
    data: String,         // Base64-encoded image data
}

/// Content block for structured content
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_use_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    // OODA-53: Image source for multimodal messages
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<ImageSource>,
}

/// Tool definition for Anthropic API
#[derive(Debug, Clone, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

/// Request body for messages endpoint
#[derive(Debug, Clone, Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

/// Response from messages endpoint
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]  // Fields used for deserialization only
struct MessagesResponse {
    id: String,
    #[serde(rename = "type")]
    response_type: String,
    role: String,
    content: Vec<ContentBlock>,
    model: String,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

/// Usage statistics from Anthropic API
#[derive(Debug, Clone, Deserialize, Default)]
#[allow(dead_code)]  // Fields used for deserialization only
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: Option<u32>,
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
}

/// Error response from Anthropic API
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]  // Fields used for deserialization only
struct AnthropicErrorResponse {
    #[serde(rename = "type")]
    error_type: String,
    error: AnthropicError,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicError {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

/// SSE event for streaming responses
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]  // Variant fields used for deserialization only
enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessagesResponse },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: DeltaBlock },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaData,
        usage: Option<DeltaUsage>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "error")]
    Error { error: AnthropicError },
}

#[derive(Debug, Clone, Deserialize)]
struct DeltaBlock {
    #[serde(rename = "type")]
    delta_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    partial_json: Option<String>,
    /// OODA-03: Extended thinking content from thinking_delta events
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct MessageDeltaData {
    stop_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]  // Fields used for deserialization only
struct DeltaUsage {
    output_tokens: u32,
}

// ============================================================================
// AnthropicProvider Implementation
// ============================================================================

/// Anthropic (Claude) LLM provider.
///
/// Supports Claude 3.5, Claude 4, and Claude 4.5 models via the Messages API.
#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
    max_context_length: usize,
    api_version: String,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider with the given API key.
    ///
    /// # Arguments
    /// * `api_key` - Anthropic API key (from <https://console.anthropic.com/>)
    pub fn new(api_key: impl Into<String>) -> Self {
        let model = DEFAULT_MODEL.to_string();
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            max_context_length: Self::context_length_for_model(&model),
            model,
            base_url: ANTHROPIC_API_BASE.to_string(),
            api_version: ANTHROPIC_API_VERSION.to_string(),
        }
    }

    /// Create a provider from environment variables.
    ///
    /// # Environment Variables
    ///
    /// - `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`: API key (required)
    /// - `ANTHROPIC_BASE_URL`: Custom endpoint, e.g., `http://localhost:11434` for Ollama
    /// - `ANTHROPIC_MODEL`: Default model to use
    ///
    /// # Ollama Compatibility
    ///
    /// To use with Ollama's Anthropic-compatible API (v0.14.0+):
    /// ```bash
    /// export ANTHROPIC_BASE_URL=http://localhost:11434
    /// export ANTHROPIC_API_KEY=ollama  # or ANTHROPIC_AUTH_TOKEN=ollama
    /// ```
    ///
    /// See: <https://docs.ollama.com/api/anthropic-compatibility>
    pub fn from_env() -> Result<Self> {
        // WHY: Support both ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN
        // Ollama documentation uses ANTHROPIC_AUTH_TOKEN, but ANTHROPIC_API_KEY
        // is more common. Support both for maximum compatibility.
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .or_else(|_| std::env::var("ANTHROPIC_AUTH_TOKEN"))
            .map_err(|_| {
                LlmError::ConfigError(
                    "ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN environment variable not set"
                        .to_string(),
                )
            })?;

        let mut provider = Self::new(api_key);

        // WHY: Allow custom base URL for Ollama or proxy servers
        // This enables using the same provider implementation with Ollama's
        // Anthropic-compatible endpoint at http://localhost:11434/v1/messages
        if let Ok(base_url) = std::env::var("ANTHROPIC_BASE_URL") {
            provider = provider.with_base_url(base_url);
        }

        // WHY: Allow model override via environment
        // Useful for testing different models without code changes
        if let Ok(model) = std::env::var("ANTHROPIC_MODEL") {
            provider = provider.with_model(model);
        }

        Ok(provider)
    }

    /// Configure provider for use with Ollama's Anthropic-compatible API.
    ///
    /// Ollama v0.14.0+ provides a `/v1/messages` endpoint that is compatible
    /// with the Anthropic Messages API, enabling use of open-source models
    /// with tools designed for Claude.
    ///
    /// This method sets:
    /// - Base URL to `http://localhost:11434`
    /// - API key to `"ollama"` (required by API but ignored by Ollama)
    /// - Default model to `qwen3-coder`
    ///
    /// # Requirements
    ///
    /// - Ollama v0.14.0 or later
    /// - Model must be pulled: `ollama pull qwen3-coder`
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use edgequake_llm::AnthropicProvider;
    ///
    /// let provider = AnthropicProvider::for_ollama();
    /// let response = provider.chat(&messages, None).await?;
    /// ```
    ///
    /// # Recommended Models
    ///
    /// - `qwen3-coder` - Excellent for coding tasks
    /// - `gpt-oss:20b` - Strong general-purpose model
    /// - `glm-4.7:cloud` - Cloud model (no local GPU needed)
    ///
    /// See: <https://docs.ollama.com/api/anthropic-compatibility>
    pub fn for_ollama() -> Self {
        Self::new("ollama")
            .with_base_url("http://localhost:11434")
            .with_model("qwen3-coder")
    }

    /// Configure provider for Ollama with a specific model.
    ///
    /// # Arguments
    ///
    /// * `model` - Ollama model name (e.g., `gpt-oss:20b`, `qwen3-coder`)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use edgequake_llm::AnthropicProvider;
    ///
    /// // Use gpt-oss:20b model
    /// let provider = AnthropicProvider::for_ollama_with_model("gpt-oss:20b");
    ///
    /// // Use cloud model (no local GPU needed)
    /// let provider = AnthropicProvider::for_ollama_with_model("glm-4.7:cloud");
    /// ```
    pub fn for_ollama_with_model(model: impl Into<String>) -> Self {
        Self::new("ollama")
            .with_base_url("http://localhost:11434")
            .with_model(model)
    }

    /// Configure provider for Ollama at a custom host.
    ///
    /// Use this when Ollama is running on a different host or port,
    /// such as a remote server or Docker container.
    ///
    /// # Arguments
    ///
    /// * `host` - Ollama server URL (e.g., `http://192.168.1.100:11434`)
    /// * `model` - Model name to use
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use edgequake_llm::AnthropicProvider;
    ///
    /// // Connect to remote Ollama instance
    /// let provider = AnthropicProvider::for_ollama_at(
    ///     "http://192.168.1.100:11434",
    ///     "qwen3-coder"
    /// );
    /// ```
    pub fn for_ollama_at(host: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new("ollama")
            .with_base_url(host)
            .with_model(model)
    }

    /// Set the model to use.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        let model_name = model.into();
        self.max_context_length = Self::context_length_for_model(&model_name);
        self.model = model_name;
        self
    }

    /// Set a custom base URL (for proxies or alternative endpoints).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set a custom API version.
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    /// Get context length for a given model.
    pub fn context_length_for_model(model: &str) -> usize {
        match model {
            // Claude 4.5 series (latest 2025)
            m if m.contains("claude-opus-4") || m.contains("opus-4.5") => 200_000,
            m if m.contains("claude-sonnet-4") => 200_000,

            // Claude 3.5 series
            m if m.contains("claude-3-5-sonnet") => 200_000,
            m if m.contains("claude-3-5-haiku") => 200_000,

            // Claude 3 series
            m if m.contains("claude-3-opus") => 200_000,
            m if m.contains("claude-3-sonnet") => 200_000,
            m if m.contains("claude-3-haiku") => 200_000,

            // Legacy models
            m if m.contains("claude-2") => 100_000,
            m if m.contains("claude-instant") => 100_000,

            _ => 200_000, // Default for new models
        }
    }

    /// Build the messages endpoint URL.
    fn endpoint(&self) -> String {
        format!("{}/v1/messages", self.base_url)
    }

    /// Build headers for API requests.
    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "x-api-key",
            self.api_key.parse().expect("Invalid API key"),
        );
        headers.insert(
            "anthropic-version",
            self.api_version.parse().expect("Invalid API version"),
        );
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );
        headers
    }

    /// Convert EdgeCode ChatMessage to Anthropic format.
    ///
    /// Anthropic uses a separate `system` field, so system messages are extracted.
    /// Returns (system_prompt, messages).
    fn convert_messages(messages: &[ChatMessage]) -> (Option<String>, Vec<AnthropicMessage>) {
        let mut system_prompt = None;
        let mut anthropic_messages = Vec::new();

        for msg in messages {
            match msg.role {
                ChatRole::System => {
                    // Anthropic uses a separate system field
                    system_prompt = Some(msg.content.clone());
                }
                ChatRole::User => {
                    // OODA-53: Check if message has images for multipart content
                    if msg.has_images() {
                        let mut blocks = Vec::new();
                        
                        // Add text block first (if non-empty)
                        if !msg.content.is_empty() {
                            blocks.push(ContentBlock {
                                content_type: "text".to_string(),
                                text: Some(msg.content.clone()),
                                id: None,
                                name: None,
                                input: None,
                                tool_use_id: None,
                                content: None,
                                source: None,
                            });
                        }
                        
                        // Add image blocks
                        if let Some(ref images) = msg.images {
                            for img in images {
                                blocks.push(ContentBlock {
                                    content_type: "image".to_string(),
                                    text: None,
                                    id: None,
                                    name: None,
                                    input: None,
                                    tool_use_id: None,
                                    content: None,
                                    source: Some(ImageSource {
                                        source_type: "base64".to_string(),
                                        media_type: img.mime_type.clone(),
                                        data: img.data.clone(),
                                    }),
                                });
                            }
                        }
                        
                        anthropic_messages.push(AnthropicMessage {
                            role: "user".to_string(),
                            content: AnthropicContent::Blocks(blocks),
                        });
                    } else {
                        anthropic_messages.push(AnthropicMessage {
                            role: "user".to_string(),
                            content: AnthropicContent::Text(msg.content.clone()),
                        });
                    }
                }
                ChatRole::Assistant => {
                    anthropic_messages.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content: AnthropicContent::Text(msg.content.clone()),
                    });
                }
                ChatRole::Tool => {
                    // Tool results need special handling
                    if let Some(tool_call_id) = &msg.tool_call_id {
                        anthropic_messages.push(AnthropicMessage {
                            role: "user".to_string(),
                            content: AnthropicContent::Blocks(vec![ContentBlock {
                                content_type: "tool_result".to_string(),
                                tool_use_id: Some(tool_call_id.clone()),
                                content: Some(msg.content.clone()),
                                text: None,
                                id: None,
                                name: None,
                                input: None,
                                source: None,
                            }]),
                        });
                    } else {
                        // Fallback to user message
                        anthropic_messages.push(AnthropicMessage {
                            role: "user".to_string(),
                            content: AnthropicContent::Text(msg.content.clone()),
                        });
                    }
                }
                ChatRole::Function => {
                    // Legacy function role, treat as user message
                    anthropic_messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicContent::Text(msg.content.clone()),
                    });
                }
            }
        }

        (system_prompt, anthropic_messages)
    }

    /// Convert EdgeCode ToolDefinition to Anthropic format.
    fn convert_tools(tools: &[ToolDefinition]) -> Vec<AnthropicTool> {
        tools
            .iter()
            .map(|tool| AnthropicTool {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                input_schema: tool.function.parameters.clone(),
            })
            .collect()
    }

    /// Convert Anthropic tool_choice to JSON value.
    fn convert_tool_choice(choice: &ToolChoice) -> serde_json::Value {
        match choice {
            ToolChoice::Auto(s) if s == "none" => serde_json::json!({"type": "none"}),
            ToolChoice::Auto(_) => serde_json::json!({"type": "auto"}),
            ToolChoice::Required(_) => serde_json::json!({"type": "any"}),
            ToolChoice::Function { function, .. } => {
                serde_json::json!({"type": "tool", "name": function.name})
            }
        }
    }

    /// Parse Anthropic response to LLMResponse.
    fn parse_response(response: MessagesResponse) -> LLMResponse {
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut metadata = HashMap::new();

        for block in &response.content {
            match block.content_type.as_str() {
                "text" => {
                    if let Some(text) = &block.text {
                        content.push_str(text);
                    }
                }
                "tool_use" => {
                    if let (Some(id), Some(name), Some(input)) =
                        (&block.id, &block.name, &block.input)
                    {
                        tool_calls.push(ToolCall {
                            id: id.clone(),
                            call_type: "function".to_string(),
                            function: FunctionCall {
                                name: name.clone(),
                                arguments: input.to_string(),
                            },
                        });
                    }
                }
                _ => {
                    debug!("Unknown content block type: {}", block.content_type);
                }
            }
        }

        metadata.insert("response_id".to_string(), serde_json::json!(response.id));

        // Calculate cache hit tokens if available
        let cache_hit_tokens = response
            .usage
            .cache_read_input_tokens
            .map(|t| t as usize);

        LLMResponse {
            content,
            prompt_tokens: response.usage.input_tokens as usize,
            completion_tokens: response.usage.output_tokens as usize,
            total_tokens: (response.usage.input_tokens + response.usage.output_tokens) as usize,
            model: response.model,
            finish_reason: response.stop_reason,
            tool_calls,
            metadata,
            cache_hit_tokens,
            thinking_tokens: None,
            thinking_content: None,
        }
    }

    /// Send a request and handle errors.
    #[instrument(skip(self, request))]
    async fn send_request(&self, request: &MessagesRequest) -> Result<MessagesResponse> {
        debug!("Sending request to Anthropic API: model={}", request.model);

        let response = self
            .client
            .post(self.endpoint())
            .headers(self.headers())
            .json(request)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        let status = response.status();

        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());

            // Try to parse as Anthropic error
            if let Ok(error_response) = serde_json::from_str::<AnthropicErrorResponse>(&error_text)
            {
                return Err(match status.as_u16() {
                    401 => LlmError::AuthError(error_response.error.message),
                    429 => LlmError::RateLimited(error_response.error.message),
                    400 => LlmError::InvalidRequest(error_response.error.message),
                    _ => LlmError::ApiError(format!(
                        "{}: {}",
                        error_response.error.error_type, error_response.error.message
                    )),
                });
            }

            return Err(LlmError::ApiError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let response_text = response
            .text()
            .await
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        debug!("Anthropic response received: {} bytes", response_text.len());

        serde_json::from_str(&response_text)
            .map_err(|e| LlmError::NetworkError(format!("Failed to parse response: {}", e)))
    }
}

#[async_trait]
impl LLMProvider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
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
        let (system, anthropic_messages) = Self::convert_messages(messages);
        let options = options.cloned().unwrap_or_default();

        let request = MessagesRequest {
            model: self.model.clone(),
            max_tokens: options.max_tokens.unwrap_or(4096) as u32,
            messages: anthropic_messages,
            system,
            stream: None,
            tools: None,
            tool_choice: None,
            temperature: options.temperature,
            top_p: options.top_p,
            stop_sequences: options.stop.clone(),
        };

        let response = self.send_request(&request).await?;
        Ok(Self::parse_response(response))
    }

    #[instrument(skip(self, prompt))]
    async fn stream(
        &self,
        prompt: &str,
    ) -> Result<BoxStream<'static, Result<String>>> {
        let messages = vec![ChatMessage::user(prompt)];
        let (system, anthropic_messages) = Self::convert_messages(&messages);

        let request = MessagesRequest {
            model: self.model.clone(),
            max_tokens: 4096,
            messages: anthropic_messages,
            system,
            stream: Some(true),
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            stop_sequences: None,
        };

        let response = self
            .client
            .post(self.endpoint())
            .headers(self.headers())
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError(format!("HTTP {}: {}", status, error_text)));
        }

        let stream = response
            .bytes_stream()
            .map(move |chunk| {
                let chunk = chunk.map_err(|e| LlmError::NetworkError(e.to_string()))?;
                let text = String::from_utf8_lossy(&chunk);

                let mut result = String::new();
                for line in text.lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        if data.trim() == "[DONE]" {
                            continue;
                        }
                        if let Ok(event) = serde_json::from_str::<StreamEvent>(data) {
                            match event {
                                StreamEvent::ContentBlockDelta { delta, .. } => {
                                    if delta.delta_type == "text_delta" {
                                        if let Some(text) = delta.text {
                                            result.push_str(&text);
                                        }
                                    }
                                }
                                StreamEvent::Error { error } => {
                                    warn!("Stream error: {}", error.message);
                                }
                                _ => {}
                            }
                        }
                    }
                }
                Ok(result)
            })
            .filter(|r| {
                let keep = match r {
                    Ok(s) => !s.is_empty(),
                    Err(_) => true,
                };
                futures::future::ready(keep)
            });

        Ok(stream.boxed())
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    // OODA-44: Enable tool streaming for React agent
    // WHY: AnthropicProvider has a fully working `chat_with_tools_stream()` implementation
    //      but the default `supports_tool_streaming()` returns false.
    //      React agent checks this flag to decide between:
    //        - true  → uses chat_with_tools_stream() (streaming with tools)
    //        - false → uses chat_with_tools() (non-streaming, default ignores tools!)
    //      By returning true, we enable the React agent to use the working stream method.
    //
    //      Flow diagram:
    //      ┌─────────────────┐
    //      │ React Agent     │
    //      └────────┬────────┘
    //               │ supports_tool_streaming()?
    //               ▼
    //         ┌───────────┐
    //         │   true    │─────► chat_with_tools_stream() ✓
    //         └───────────┘
    //         │   false   │─────► chat_with_tools() → chat() (tools ignored!)
    //         └───────────┘
    fn supports_tool_streaming(&self) -> bool {
        true
    }

    #[instrument(skip(self, messages, tools, options))]
    async fn chat_with_tools_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        let (system, anthropic_messages) = Self::convert_messages(messages);
        let anthropic_tools = Self::convert_tools(tools);
        let options = options.cloned().unwrap_or_default();

        let request = MessagesRequest {
            model: self.model.clone(),
            max_tokens: options.max_tokens.unwrap_or(4096) as u32,
            messages: anthropic_messages,
            system,
            stream: Some(true),
            tools: Some(anthropic_tools),
            tool_choice: tool_choice.map(|tc| Self::convert_tool_choice(&tc)),
            temperature: options.temperature,
            top_p: options.top_p,
            stop_sequences: options.stop.clone(),
        };

        let response = self
            .client
            .post(self.endpoint())
            .headers(self.headers())
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError(format!("HTTP {}: {}", status, error_text)));
        }

        // Track current tool call being built
        let stream = response
            .bytes_stream()
            .map(move |chunk| {
                let chunk = chunk.map_err(|e| LlmError::NetworkError(e.to_string()))?;
                let text = String::from_utf8_lossy(&chunk);

                let mut chunks: Vec<StreamChunk> = Vec::new();
                for line in text.lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        if data.trim() == "[DONE]" {
                            continue;
                        }
                        if let Ok(event) = serde_json::from_str::<StreamEvent>(data) {
                            match event {
                                StreamEvent::ContentBlockStart {
                                    index,
                                    content_block,
                                } => {
                                    if content_block.content_type == "tool_use" {
                                        if let (Some(id), Some(name)) =
                                            (content_block.id, content_block.name)
                                        {
                                            // Send tool call start as a delta with id and name
                                            chunks.push(StreamChunk::ToolCallDelta {
                                                index,
                                                id: Some(id),
                                                function_name: Some(name),
                                                function_arguments: None,
                                            });
                                        }
                                    }
                                }
                                StreamEvent::ContentBlockDelta { index, delta } => {
                                    match delta.delta_type.as_str() {
                                        "text_delta" => {
                                            if let Some(text) = delta.text {
                                                chunks.push(StreamChunk::Content(text));
                                            }
                                        }
                                        "input_json_delta" => {
                                            if let Some(json) = delta.partial_json {
                                                chunks.push(StreamChunk::ToolCallDelta {
                                                    index,
                                                    id: None,
                                                    function_name: None,
                                                    function_arguments: Some(json),
                                                });
                                            }
                                        }
                                        // OODA-03: Extended thinking streaming support
                                        "thinking_delta" => {
                                            if let Some(thinking) = delta.thinking {
                                                chunks.push(StreamChunk::ThinkingContent {
                                                    text: thinking,
                                                    tokens_used: None,
                                                    budget_total: None,
                                                });
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                                StreamEvent::ContentBlockStop { .. } => {
                                    // Block completed, no special StreamChunk variant needed
                                }
                                StreamEvent::MessageStop => {
                                    chunks.push(StreamChunk::Finished {
                                        reason: "stop".to_string(),
                                        ttft_ms: None,
                                    });
                                }
                                StreamEvent::MessageDelta { delta, .. } => {
                                    if let Some(reason) = delta.stop_reason {
                                        chunks.push(StreamChunk::Finished { reason, ttft_ms: None });
                                    }
                                }
                                StreamEvent::Error { error } => {
                                    return Err(LlmError::ApiError(error.message));
                                }
                                _ => {}
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
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_provider() {
        let provider = AnthropicProvider::new("test-api-key");
        assert_eq!(provider.name(), "anthropic");
        assert_eq!(provider.model(), DEFAULT_MODEL);
    }

    #[test]
    fn test_with_model() {
        let provider = AnthropicProvider::new("test-api-key")
            .with_model("claude-3-5-sonnet-20241022");
        assert_eq!(provider.model(), "claude-3-5-sonnet-20241022");
        assert_eq!(provider.max_context_length(), 200_000);
    }

    #[test]
    fn test_context_length_for_model() {
        assert_eq!(
            AnthropicProvider::context_length_for_model("claude-opus-4-5-20250929"),
            200_000
        );
        assert_eq!(
            AnthropicProvider::context_length_for_model("claude-sonnet-4-5-20250929"),
            200_000
        );
        assert_eq!(
            AnthropicProvider::context_length_for_model("claude-3-5-sonnet-20241022"),
            200_000
        );
        assert_eq!(
            AnthropicProvider::context_length_for_model("claude-2.1"),
            100_000
        );
    }

    #[test]
    fn test_convert_messages_with_system() {
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("Hello!"),
            ChatMessage::assistant("Hi there!"),
        ];

        let (system, anthropic_messages) = AnthropicProvider::convert_messages(&messages);

        assert_eq!(system, Some("You are a helpful assistant.".to_string()));
        assert_eq!(anthropic_messages.len(), 2);
        assert_eq!(anthropic_messages[0].role, "user");
        assert_eq!(anthropic_messages[1].role, "assistant");
    }

    #[test]
    fn test_convert_messages_without_system() {
        let messages = vec![
            ChatMessage::user("Hello!"),
        ];

        let (system, anthropic_messages) = AnthropicProvider::convert_messages(&messages);

        assert_eq!(system, None);
        assert_eq!(anthropic_messages.len(), 1);
    }

    #[test]
    fn test_convert_tools() {
        use crate::traits::FunctionDefinition;

        let tools = vec![ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "get_weather".to_string(),
                description: "Get the weather for a location".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }),
                strict: None,
            },
        }];

        let anthropic_tools = AnthropicProvider::convert_tools(&tools);

        assert_eq!(anthropic_tools.len(), 1);
        assert_eq!(anthropic_tools[0].name, "get_weather");
        assert_eq!(anthropic_tools[0].description, "Get the weather for a location");
    }

    #[test]
    fn test_convert_tool_choice() {
        let auto = AnthropicProvider::convert_tool_choice(&ToolChoice::auto());
        assert_eq!(auto, serde_json::json!({"type": "auto"}));

        let required = AnthropicProvider::convert_tool_choice(&ToolChoice::required());
        assert_eq!(required, serde_json::json!({"type": "any"}));

        let none = AnthropicProvider::convert_tool_choice(&ToolChoice::none());
        assert_eq!(none, serde_json::json!({"type": "none"}));

        let specific = AnthropicProvider::convert_tool_choice(&ToolChoice::function("my_tool"));
        assert_eq!(specific, serde_json::json!({"type": "tool", "name": "my_tool"}));
    }

    #[test]
    fn test_headers() {
        let provider = AnthropicProvider::new("test-key");
        let headers = provider.headers();

        assert!(headers.contains_key("x-api-key"));
        assert!(headers.contains_key("anthropic-version"));
        assert!(headers.contains_key(reqwest::header::CONTENT_TYPE));
    }

    #[test]
    fn test_endpoint() {
        let provider = AnthropicProvider::new("test-key");
        assert_eq!(provider.endpoint(), "https://api.anthropic.com/v1/messages");

        let custom = provider.with_base_url("https://custom.api.com");
        assert_eq!(custom.endpoint(), "https://custom.api.com/v1/messages");
    }

    #[test]
    fn test_parse_response() {
        let response = MessagesResponse {
            id: "msg_123".to_string(),
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![ContentBlock {
                content_type: "text".to_string(),
                text: Some("Hello, world!".to_string()),
                id: None,
                name: None,
                input: None,
                tool_use_id: None,
                content: None,
                source: None,
            }],
            model: "claude-3-5-sonnet-20241022".to_string(),
            stop_reason: Some("end_turn".to_string()),
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };

        let llm_response = AnthropicProvider::parse_response(response);

        assert_eq!(llm_response.content, "Hello, world!");
        assert_eq!(llm_response.prompt_tokens, 10);
        assert_eq!(llm_response.completion_tokens, 5);
        assert_eq!(llm_response.total_tokens, 15);
        assert_eq!(llm_response.model, "claude-3-5-sonnet-20241022");
        assert_eq!(llm_response.finish_reason, Some("end_turn".to_string()));
        assert!(llm_response.tool_calls.is_empty());
    }

    #[test]
    fn test_parse_response_with_tool_calls() {
        let response = MessagesResponse {
            id: "msg_456".to_string(),
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![ContentBlock {
                content_type: "tool_use".to_string(),
                text: None,
                id: Some("tool_1".to_string()),
                name: Some("get_weather".to_string()),
                input: Some(serde_json::json!({"location": "Paris"})),
                tool_use_id: None,
                content: None,
                source: None,
            }],
            model: "claude-3-5-sonnet-20241022".to_string(),
            stop_reason: Some("tool_use".to_string()),
            usage: AnthropicUsage {
                input_tokens: 20,
                output_tokens: 10,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };

        let llm_response = AnthropicProvider::parse_response(response);

        assert_eq!(llm_response.tool_calls.len(), 1);
        assert_eq!(llm_response.tool_calls[0].id, "tool_1");
        assert_eq!(llm_response.tool_calls[0].name(), "get_weather");
        assert!(llm_response.tool_calls[0].arguments().contains("Paris"));
    }

    // Integration test - requires API key
    #[tokio::test]
    #[ignore]
    async fn test_chat_completion_live() {
        let provider = AnthropicProvider::from_env().expect("ANTHROPIC_API_KEY not set");
        let messages = vec![ChatMessage::user("Say 'hello' and nothing else.")];

        let response = provider.chat(&messages, None).await;
        assert!(response.is_ok());

        let response = response.unwrap();
        assert!(!response.content.is_empty());
        assert!(response.prompt_tokens > 0);
        assert!(response.completion_tokens > 0);
    }

    // =========================================================================
    // Ollama Compatibility Tests
    // =========================================================================

    #[test]
    fn test_for_ollama_defaults() {
        // WHY: Verify that for_ollama() sets correct defaults for Ollama usage
        // These defaults match Ollama's Anthropic compatibility requirements
        let provider = AnthropicProvider::for_ollama();

        assert_eq!(provider.base_url, "http://localhost:11434");
        assert_eq!(provider.api_key, "ollama");
        assert_eq!(provider.model, "qwen3-coder");
    }

    #[test]
    fn test_for_ollama_with_model() {
        // WHY: Users may want to use different Ollama models
        let provider = AnthropicProvider::for_ollama_with_model("gpt-oss:20b");

        assert_eq!(provider.model, "gpt-oss:20b");
        assert_eq!(provider.base_url, "http://localhost:11434");
        assert_eq!(provider.api_key, "ollama");
    }

    #[test]
    fn test_for_ollama_at_custom_host() {
        // WHY: Ollama may run on remote servers or Docker containers
        let provider = AnthropicProvider::for_ollama_at(
            "http://192.168.1.100:11434",
            "llama3"
        );

        assert_eq!(provider.base_url, "http://192.168.1.100:11434");
        assert_eq!(provider.model, "llama3");
        assert_eq!(provider.api_key, "ollama");
    }

    #[test]
    fn test_for_ollama_endpoint() {
        // WHY: Ollama's Anthropic-compatible endpoint is at /v1/messages
        let provider = AnthropicProvider::for_ollama();

        assert_eq!(provider.endpoint(), "http://localhost:11434/v1/messages");
    }

    #[test]
    fn test_with_base_url_chain() {
        // WHY: Ensure builder pattern works correctly for custom endpoints
        let provider = AnthropicProvider::new("test-key")
            .with_base_url("http://localhost:11434")
            .with_model("qwen3-coder");

        assert_eq!(provider.base_url, "http://localhost:11434");
        assert_eq!(provider.model, "qwen3-coder");
        assert_eq!(provider.api_key, "test-key");
    }

    // Integration test - requires Ollama running locally
    #[tokio::test]
    #[ignore]
    async fn test_ollama_chat_completion_live() {
        // WHY: E2E test to verify Ollama integration works
        // Run with: cargo test test_ollama_chat_completion_live -- --ignored
        // Requires: ollama pull qwen3-coder
        let provider = AnthropicProvider::for_ollama();
        let messages = vec![ChatMessage::user("Say 'hello' and nothing else.")];

        let response = provider.chat(&messages, None).await;
        assert!(response.is_ok(), "Ollama chat failed: {:?}", response.err());

        let response = response.unwrap();
        assert!(!response.content.is_empty());
    }

    // =========================================================================
    // Image Support Tests (OODA-53)
    // =========================================================================

    #[test]
    fn test_convert_messages_text_only() {
        // WHY: Verify text-only messages still serialize as simple strings
        let messages = vec![ChatMessage::user("Hello, world!")];
        let (_, anthropic_messages) = AnthropicProvider::convert_messages(&messages);

        assert_eq!(anthropic_messages.len(), 1);
        
        // Text-only should serialize as string
        let json = serde_json::to_value(&anthropic_messages[0]).unwrap();
        assert_eq!(json["content"], "Hello, world!");
    }

    #[test]
    fn test_convert_messages_with_images() {
        use crate::traits::ImageData;
        
        // WHY: Verify images use Anthropic's base64 source format
        let images = vec![ImageData::new("base64data", "image/png")];
        let messages = vec![ChatMessage::user_with_images("What's this?", images)];
        let (_, anthropic_messages) = AnthropicProvider::convert_messages(&messages);

        assert_eq!(anthropic_messages.len(), 1);
        
        // With images should serialize as array of blocks
        let json = serde_json::to_value(&anthropic_messages[0]).unwrap();
        let content = &json["content"];
        
        assert!(content.is_array(), "Content should be an array for images");
        assert_eq!(content.as_array().unwrap().len(), 2);
        
        // First block: text
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "What's this?");
        
        // Second block: image with base64 source (Anthropic format)
        assert_eq!(content[1]["type"], "image");
        assert!(content[1]["source"].is_object(), "Image should have source object");
        assert_eq!(content[1]["source"]["type"], "base64");
        assert_eq!(content[1]["source"]["media_type"], "image/png");
        assert_eq!(content[1]["source"]["data"], "base64data");
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
        let (_, anthropic_messages) = AnthropicProvider::convert_messages(&messages);

        let json = serde_json::to_value(&anthropic_messages[0]).unwrap();
        let content = &json["content"];
        
        assert_eq!(content.as_array().unwrap().len(), 3); // 1 text + 2 images
        
        // Verify both images
        assert_eq!(content[1]["source"]["media_type"], "image/png");
        assert_eq!(content[2]["source"]["media_type"], "image/jpeg");
    }

    // =========================================================================
    // Extended Thinking Tests (OODA-04)
    // =========================================================================

    #[test]
    fn test_delta_block_parses_thinking_delta() {
        // OODA-04: Verify DeltaBlock can parse thinking_delta events
        // WHY: OODA-03 added thinking field to support extended thinking streaming
        let json = r#"{"type":"thinking_delta","thinking":"Let me analyze step by step..."}"#;
        
        let delta: DeltaBlock = serde_json::from_str(json).unwrap();
        
        assert_eq!(delta.delta_type, "thinking_delta");
        assert_eq!(delta.thinking, Some("Let me analyze step by step...".to_string()));
        assert!(delta.text.is_none());
        assert!(delta.partial_json.is_none());
    }

    #[test]
    fn test_delta_block_parses_text_delta() {
        // OODA-04: Verify text_delta still works (regression test)
        let json = r#"{"type":"text_delta","text":"Hello world"}"#;
        
        let delta: DeltaBlock = serde_json::from_str(json).unwrap();
        
        assert_eq!(delta.delta_type, "text_delta");
        assert_eq!(delta.text, Some("Hello world".to_string()));
        assert!(delta.thinking.is_none());
    }

    #[test]
    fn test_delta_block_parses_input_json_delta() {
        // OODA-04: Verify input_json_delta still works (regression test)
        let json = r#"{"type":"input_json_delta","partial_json":"{\"name\":"}"#;
        
        let delta: DeltaBlock = serde_json::from_str(json).unwrap();
        
        assert_eq!(delta.delta_type, "input_json_delta");
        assert_eq!(delta.partial_json, Some("{\"name\":".to_string()));
        assert!(delta.thinking.is_none());
        assert!(delta.text.is_none());
    }
}
