//! Anthropic (Claude) LLM provider implementation.
//!
//! Supports Anthropic's Messages API for Claude models.
//!
//! # Environment Variables
//! - `ANTHROPIC_API_KEY`: API key for Anthropic API
//! - `ANTHROPIC_AUTH_TOKEN`: Alternative API key variable (Ollama compat)
//! - `ANTHROPIC_BASE_URL`: Override base URL (for Ollama or proxies)
//! - `ANTHROPIC_MODEL`: Override default model
//!
//! # Models Supported (latest first)
//! - Claude Opus 4.6:   `claude-opus-4-6`
//! - Claude Sonnet 4.6: `claude-sonnet-4-6`  ← DEFAULT
//! - Claude Haiku 4.5:  `claude-haiku-4-5`
//! - Claude Sonnet 4.5: `claude-sonnet-4-5-20250929`
//! - Claude Opus 4.5:   `claude-opus-4-5-20250929`
//! - Claude Sonnet 3.5: `claude-3-5-sonnet-20241022`
//! - Claude Haiku 3.5:  `claude-3-5-haiku-20241022`
//! - Claude Opus 3:     `claude-3-opus-20240229`
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
    ChatMessage, ChatRole, CompletionOptions, FunctionCall, LLMProvider, LLMResponse, StreamChunk,
    ToolCall, ToolChoice, ToolDefinition,
};

/// Anthropic API base URL
const ANTHROPIC_API_BASE: &str = "https://api.anthropic.com";

/// Anthropic API version (required header)
const ANTHROPIC_API_VERSION: &str = "2023-06-01";

/// Default model — latest Claude Sonnet 4.6 (simplified naming, no date suffix)
const DEFAULT_MODEL: &str = "claude-sonnet-4-6";

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
    source_type: String, // Always "base64"
    media_type: String, // MIME type, e.g., "image/png"
    data: String,       // Base64-encoded image data
}

/// Content block for structured content
///
/// Used for both request blocks (user/assistant messages) and response blocks.
/// The Anthropic API uses a single structure for all block types, with type-specific
/// fields populated based on `content_type`:
/// - `text`: text field
/// - `image`: source field (ImageSource with base64 data)
/// - `tool_use`: id, name, input fields
/// - `tool_result`: tool_use_id, content, is_error fields
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
    /// For tool_result blocks: the ID of the tool_use block being responded to.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_use_id: Option<String>,
    /// For tool_result blocks: the result content (string or content array).
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// For tool_result blocks: set to true if the tool call resulted in an error.
    /// Anthropic uses this to guide the model's error handling behavior.
    #[serde(skip_serializing_if = "Option::is_none")]
    is_error: Option<bool>,
    // Image source for multimodal messages (OODA-53)
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
#[allow(dead_code)] // Fields used for deserialization only
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
#[allow(dead_code)] // Fields used for deserialization only
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
#[allow(dead_code)] // Fields used for deserialization only
struct AnthropicErrorResponse {
    #[serde(rename = "type")]
    error_type: String,
    error: AnthropicError,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicError {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    error_type: String,
    message: String,
}

/// SSE event for streaming responses
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)] // Variant fields used for deserialization only
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
    text: Option<String>,
    partial_json: Option<String>,
    /// Extended thinking content from thinking_delta events
    thinking: Option<String>,
    /// Signature for thinking block integrity verification (signature_delta events)
    #[allow(dead_code)]
    signature: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct MessageDeltaData {
    stop_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // Fields used for deserialization only
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
        Self::new("ollama").with_base_url(host).with_model(model)
    }

    /// Override the API key (useful when the key changes after construction).
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
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

    // ---- Read-only accessors ------------------------------------------------

    /// Return the configured API key.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Return the configured base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Return the configured API version header value.
    pub fn api_version(&self) -> &str {
        &self.api_version
    }

    /// Return the full messages endpoint URL (`<base_url>/v1/messages`).
    pub fn endpoint(&self) -> String {
        format!("{}/v1/messages", self.base_url)
    }

    /// Get context length for a given model.
    pub fn context_length_for_model(model: &str) -> usize {
        match model {
            // Claude 4.6 series (latest 2026) — simplified naming without date suffix
            m if m.contains("claude-opus-4-6") => 200_000,
            m if m.contains("claude-sonnet-4-6") => 200_000,

            // Claude 4.5 series (2025)
            m if m.contains("claude-opus-4-5") || m.contains("opus-4.5") => 200_000,
            m if m.contains("claude-sonnet-4-5") || m.contains("sonnet-4.5") => 200_000,
            m if m.contains("claude-haiku-4-5") || m.contains("haiku-4.5") => 200_000,

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

            _ => 200_000, // Default for new/unknown models
        }
    }

    /// Build headers for API requests.
    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-api-key", self.api_key.parse().expect("Invalid API key"));
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
    /// Multiple system messages are concatenated with a newline separator.
    ///
    /// Assistant messages that contain tool calls are serialized as content-block
    /// arrays so the model can replay them correctly in multi-turn conversations.
    ///
    /// Returns (system_prompt, messages).
    fn convert_messages(messages: &[ChatMessage]) -> (Option<String>, Vec<AnthropicMessage>) {
        let mut system_parts: Vec<String> = Vec::new();
        let mut anthropic_messages = Vec::new();

        for msg in messages {
            match msg.role {
                ChatRole::System => {
                    // Anthropic uses a separate system field.
                    // Multiple system messages are concatenated — this matches the
                    // behaviour of Anthropic's own SDK and avoids silently dropping
                    // earlier system instructions.
                    system_parts.push(msg.content.clone());
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
                                is_error: None,
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
                                    is_error: None,
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
                    // If the assistant message contains tool calls (from a prior turn),
                    // it must be serialized as a content-block array so the API can
                    // correctly replay the conversation. A plain text message drops the
                    // tool_use blocks, which breaks multi-turn tool workflows.
                    if let Some(ref tool_calls) = msg.tool_calls {
                        if !tool_calls.is_empty() {
                            let mut blocks = Vec::new();

                            // Prefix text block (model's reasoning before tool calls)
                            if !msg.content.is_empty() {
                                blocks.push(ContentBlock {
                                    content_type: "text".to_string(),
                                    text: Some(msg.content.clone()),
                                    id: None,
                                    name: None,
                                    input: None,
                                    tool_use_id: None,
                                    content: None,
                                    is_error: None,
                                    source: None,
                                });
                            }

                            // One tool_use block per tool call
                            for tc in tool_calls {
                                let input: serde_json::Value =
                                    serde_json::from_str(&tc.function.arguments)
                                        .unwrap_or(serde_json::Value::Object(Default::default()));
                                blocks.push(ContentBlock {
                                    content_type: "tool_use".to_string(),
                                    text: None,
                                    id: Some(tc.id.clone()),
                                    name: Some(tc.function.name.clone()),
                                    input: Some(input),
                                    tool_use_id: None,
                                    content: None,
                                    is_error: None,
                                    source: None,
                                });
                            }

                            anthropic_messages.push(AnthropicMessage {
                                role: "assistant".to_string(),
                                content: AnthropicContent::Blocks(blocks),
                            });
                            continue;
                        }
                    }

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
                                is_error: None,
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

        let system_prompt = if system_parts.is_empty() {
            None
        } else {
            Some(system_parts.join("\n\n"))
        };

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
                            thought_signature: None,
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
        let cache_hit_tokens = response.usage.cache_read_input_tokens.map(|t| t as usize);

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

    /// Map an HTTP status + optional parsed error response to the correct `LlmError` variant.
    ///
    /// Covers all documented Anthropic HTTP error codes:
    /// - 400 invalid_request_error
    /// - 401 authentication_error
    /// - 402 billing_error
    /// - 403 permission_error
    /// - 404 not_found_error
    /// - 413 request_too_large
    /// - 429 rate_limit_error
    /// - 500 api_error
    /// - 529 overloaded_error
    fn handle_error(status: reqwest::StatusCode, body: &str) -> LlmError {
        // Attempt to extract a structured error message from the body.
        let message = serde_json::from_str::<AnthropicErrorResponse>(body)
            .map(|e| e.error.message)
            .unwrap_or_else(|_| body.to_string());

        match status.as_u16() {
            400 => LlmError::InvalidRequest(message),
            401 => LlmError::AuthError(message),
            402 => LlmError::ApiError(format!("Billing error: {}", message)),
            403 => LlmError::AuthError(format!("Permission denied: {}", message)),
            404 => LlmError::ModelNotFound(message),
            413 => LlmError::TokenLimitExceeded { max: 0, got: 0 },
            429 => LlmError::RateLimited(message),
            500 => LlmError::ApiError(format!("Anthropic internal error: {}", message)),
            529 => LlmError::RateLimited(format!("Service overloaded: {}", message)),
            _ => LlmError::ApiError(format!("HTTP {}: {}", status, message)),
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
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(Self::handle_error(status, &body));
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
    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
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
            let body = response.text().await.unwrap_or_default();
            return Err(Self::handle_error(status, &body));
        }

        // SSE line buffering: network chunks may split lines at arbitrary byte
        // boundaries. Without buffering the incomplete line is silently dropped,
        // truncating response text.  We accumulate bytes into `line_buffer` and
        // only process complete lines (terminated by '\n').
        let mut line_buffer = String::new();

        let stream = response
            .bytes_stream()
            .map(move |chunk| {
                let chunk = chunk.map_err(|e| LlmError::NetworkError(e.to_string()))?;
                let text = String::from_utf8_lossy(&chunk);

                line_buffer.push_str(&text);

                let mut result = String::new();

                while let Some(newline_idx) = line_buffer.find('\n') {
                    let line = line_buffer[..newline_idx].trim().to_string();
                    line_buffer.drain(..=newline_idx);

                    // Skip empty lines and SSE comment lines (':')
                    if line.is_empty() || line.starts_with(':') {
                        continue;
                    }

                    if let Some(data) = line.strip_prefix("data: ") {
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

    fn supports_function_calling(&self) -> bool {
        true
    }

    // OODA-44: Enable tool streaming for React agent
    fn supports_tool_streaming(&self) -> bool {
        true
    }

    /// Non-streaming chat with tool/function calling support.
    ///
    /// This override properly includes the tools in the Messages API request.
    /// Without this override the default trait implementation ignores tools
    /// entirely (calls plain `chat()`), making non-streaming tool use broken.
    #[instrument(skip(self, messages, tools, options))]
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let (system, anthropic_messages) = Self::convert_messages(messages);
        let anthropic_tools = Self::convert_tools(tools);
        let options = options.cloned().unwrap_or_default();

        let request = MessagesRequest {
            model: self.model.clone(),
            max_tokens: options.max_tokens.unwrap_or(4096) as u32,
            messages: anthropic_messages,
            system,
            stream: None,
            tools: if tools.is_empty() {
                None
            } else {
                Some(anthropic_tools)
            },
            tool_choice: tool_choice.map(|tc| Self::convert_tool_choice(&tc)),
            temperature: options.temperature,
            top_p: options.top_p,
            stop_sequences: options.stop.clone(),
        };

        let response = self.send_request(&request).await?;
        Ok(Self::parse_response(response))
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
            tools: if tools.is_empty() {
                None
            } else {
                Some(anthropic_tools)
            },
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
            let body = response.text().await.unwrap_or_default();
            return Err(Self::handle_error(status, &body));
        }

        // SSE line buffering: accumulate bytes across network chunks so we only
        // process complete lines. Without this, a line split at a chunk boundary
        // is silently dropped, truncating tool call arguments or response text.
        let mut line_buffer = String::new();
        // WHY: track across byte-chunk boundaries. MessageDelta (tool_use stop_reason)
        // and MessageStop arrive in separate HTTP chunks, so the guard must be
        // captured in the FnMut closure state rather than recomputed from the
        // current chunk's local vec.
        let mut finished_emitted = false;

        let stream = response
            .bytes_stream()
            .map(move |chunk| -> Result<Vec<StreamChunk>> {
                let chunk = chunk.map_err(|e| LlmError::NetworkError(e.to_string()))?;
                let text = String::from_utf8_lossy(&chunk);

                line_buffer.push_str(&text);

                let mut chunks: Vec<StreamChunk> = Vec::new();

                while let Some(newline_idx) = line_buffer.find('\n') {
                    let line = line_buffer[..newline_idx].trim().to_string();
                    line_buffer.drain(..=newline_idx);

                    // Skip empty lines and SSE comment lines (':')
                    if line.is_empty() || line.starts_with(':') {
                        continue;
                    }

                    if let Some(data) = line.strip_prefix("data: ") {
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
                                            // Signal start of a new tool call with its id and name.
                                            chunks.push(StreamChunk::ToolCallDelta {
                                                index,
                                                id: Some(id),
                                                function_name: Some(name),
                                                function_arguments: None,
                                                thought_signature: None,
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
                                                    thought_signature: None,
                                                });
                                            }
                                        }
                                        // Extended thinking streaming (OODA-03)
                                        "thinking_delta" => {
                                            if let Some(thinking) = delta.thinking {
                                                chunks.push(StreamChunk::ThinkingContent {
                                                    text: thinking,
                                                    tokens_used: None,
                                                    budget_total: None,
                                                });
                                            }
                                        }
                                        // signature_delta is informational; no StreamChunk needed
                                        _ => {}
                                    }
                                }
                                StreamEvent::ContentBlockStop { .. } => {
                                    // Block closed — consumer accumulates via deltas above
                                }
                                StreamEvent::MessageDelta { delta, .. } => {
                                    // Emit Finished with the authoritative stop_reason from
                                    // message_delta.  Do NOT also emit from MessageStop to
                                    // avoid duplicate Finished chunks.
                                    if let Some(reason) = delta.stop_reason {
                                        if !finished_emitted {
                                            finished_emitted = true;
                                            chunks.push(StreamChunk::Finished {
                                                reason,
                                                ttft_ms: None,
                                            });
                                        }
                                    }
                                }
                                StreamEvent::MessageStop => {
                                    // message_stop arrives after message_delta which already
                                    // emitted Finished.  Only emit here if message_delta
                                    // did not carry a stop_reason (error recovery path).
                                    if !finished_emitted {
                                        finished_emitted = true;
                                        chunks.push(StreamChunk::Finished {
                                            reason: "stop".to_string(),
                                            ttft_ms: None,
                                        });
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
        let provider =
            AnthropicProvider::new("test-api-key").with_model("claude-3-5-sonnet-20241022");
        assert_eq!(provider.model(), "claude-3-5-sonnet-20241022");
        assert_eq!(provider.max_context_length(), 200_000);
    }

    #[test]
    fn test_context_length_for_model() {
        assert_eq!(
            AnthropicProvider::context_length_for_model("claude-opus-4-6"),
            200_000
        );
        assert_eq!(
            AnthropicProvider::context_length_for_model("claude-sonnet-4-6"),
            200_000
        );
        assert_eq!(
            AnthropicProvider::context_length_for_model("claude-opus-4-5-20250929"),
            200_000
        );
        assert_eq!(
            AnthropicProvider::context_length_for_model("claude-sonnet-4-5-20250929"),
            200_000
        );
        assert_eq!(
            AnthropicProvider::context_length_for_model("claude-haiku-4-5"),
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
        let messages = vec![ChatMessage::user("Hello!")];

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
        assert_eq!(
            anthropic_tools[0].description,
            "Get the weather for a location"
        );
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
        assert_eq!(
            specific,
            serde_json::json!({"type": "tool", "name": "my_tool"})
        );
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
        let provider = AnthropicProvider::for_ollama_at("http://192.168.1.100:11434", "llama3");

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
        assert!(
            content[1]["source"].is_object(),
            "Image should have source object"
        );
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
        assert_eq!(
            delta.thinking,
            Some("Let me analyze step by step...".to_string())
        );
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

    // =========================================================================
    // OODA-35: Additional Unit Tests
    // =========================================================================

    #[test]
    fn test_constants() {
        // WHY: Verify constants are as expected for API compatibility
        assert_eq!(ANTHROPIC_API_BASE, "https://api.anthropic.com");
        assert_eq!(ANTHROPIC_API_VERSION, "2023-06-01");
        assert_eq!(DEFAULT_MODEL, "claude-sonnet-4-6");
    }

    #[test]
    fn test_supports_streaming() {
        let provider = AnthropicProvider::new("key");
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_supports_tool_streaming() {
        // WHY: OODA-44 enabled tool streaming for React agent
        let provider = AnthropicProvider::new("key");
        assert!(provider.supports_tool_streaming());
    }

    #[test]
    fn test_anthropic_usage_with_cache_tokens() {
        // WHY: Anthropic prompt caching adds cache_creation_input_tokens and cache_read_input_tokens
        let json = r#"{
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 25,
            "cache_read_input_tokens": 10
        }"#;
        let usage: AnthropicUsage = serde_json::from_str(json).unwrap();

        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.cache_creation_input_tokens, Some(25));
        assert_eq!(usage.cache_read_input_tokens, Some(10));
    }

    #[test]
    fn test_anthropic_error_response_deserialization() {
        let json = r#"{
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "messages: Required field missing"
            }
        }"#;
        let error: AnthropicErrorResponse = serde_json::from_str(json).unwrap();

        assert_eq!(error.error_type, "error");
        assert_eq!(error.error.error_type, "invalid_request_error");
        assert_eq!(error.error.message, "messages: Required field missing");
    }

    #[test]
    fn test_stream_event_message_start() {
        // WHY: StreamEvent uses tagged enum - verify it deserializes correctly
        let json = r#"{
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-3-5-sonnet",
                "stop_reason": null,
                "usage": {"input_tokens": 10, "output_tokens": 0}
            }
        }"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();

        match event {
            StreamEvent::MessageStart { message } => {
                assert_eq!(message.id, "msg_123");
                assert_eq!(message.role, "assistant");
            }
            _ => panic!("Expected MessageStart event"),
        }
    }

    #[test]
    fn test_stream_event_ping() {
        let json = r#"{"type": "ping"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();

        matches!(event, StreamEvent::Ping);
    }

    #[test]
    fn test_image_source_serialization() {
        // WHY: Verify ImageSource serializes correctly for Anthropic API
        let source = ImageSource {
            source_type: "base64".to_string(),
            media_type: "image/png".to_string(),
            data: "aGVsbG8=".to_string(),
        };

        let json = serde_json::to_value(&source).unwrap();
        assert_eq!(json["type"], "base64");
        assert_eq!(json["media_type"], "image/png");
        assert_eq!(json["data"], "aGVsbG8=");
    }

    #[test]
    fn test_content_block_tool_use() {
        let block = ContentBlock {
            content_type: "tool_use".to_string(),
            text: None,
            id: Some("tool_123".to_string()),
            name: Some("get_weather".to_string()),
            input: Some(serde_json::json!({"location": "NYC"})),
            tool_use_id: None,
            content: None,
            is_error: None,
            source: None,
        };

        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "tool_use");
        assert_eq!(json["id"], "tool_123");
        assert_eq!(json["name"], "get_weather");
        assert_eq!(json["input"]["location"], "NYC");
    }

    // =========================================================================
    // Bug Fix Tests — added by Anthropic API audit 2026-04
    // =========================================================================

    #[test]
    fn test_multiple_system_messages_are_concatenated() {
        // Bug fix: Previously only the last system message was kept.
        // Correct: Multiple system messages must be concatenated with \n\n.
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::system("Always respond in JSON."),
            ChatMessage::user("Hello!"),
        ];

        let (system, anthropic_messages) = AnthropicProvider::convert_messages(&messages);

        assert_eq!(
            system,
            Some("You are a helpful assistant.\n\nAlways respond in JSON.".to_string()),
            "Multiple system messages must be joined with \\n\\n"
        );
        assert_eq!(anthropic_messages.len(), 1);
        assert_eq!(anthropic_messages[0].role, "user");
    }

    #[test]
    fn test_assistant_message_with_tool_calls_serializes_as_blocks() {
        // Bug fix: Assistant messages that include prior tool calls must be
        // serialized as content blocks (not plain text) for multi-turn correctness.
        use crate::traits::{FunctionCall, ToolCall};

        let tool_call = ToolCall {
            id: "toolu_01".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"location":"Paris"}"#.to_string(),
            },
            thought_signature: None,
        };

        let mut msg = ChatMessage::assistant("Let me check the weather.");
        msg.tool_calls = Some(vec![tool_call]);

        let (_, anthropic_messages) = AnthropicProvider::convert_messages(&[msg]);
        assert_eq!(anthropic_messages.len(), 1);

        let json = serde_json::to_value(&anthropic_messages[0]).unwrap();
        assert!(
            json["content"].is_array(),
            "Assistant message with tool calls must serialize as a block array"
        );

        let blocks = json["content"].as_array().unwrap();
        assert_eq!(
            blocks.len(),
            2,
            "Expect one text block + one tool_use block"
        );

        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "Let me check the weather.");

        assert_eq!(blocks[1]["type"], "tool_use");
        assert_eq!(blocks[1]["id"], "toolu_01");
        assert_eq!(blocks[1]["name"], "get_weather");
        assert_eq!(blocks[1]["input"]["location"], "Paris");
    }

    #[test]
    fn test_assistant_message_without_tool_calls_serializes_as_text() {
        // Regression: Plain assistant messages (no tool calls) must still be strings.
        let msg = ChatMessage::assistant("Hello!");
        let (_, anthropic_messages) = AnthropicProvider::convert_messages(&[msg]);

        let json = serde_json::to_value(&anthropic_messages[0]).unwrap();
        assert_eq!(
            json["content"], "Hello!",
            "Plain assistant message must serialize as a JSON string"
        );
    }

    #[test]
    fn test_content_block_is_error_field() {
        // The is_error field on tool_result blocks must serialize correctly.
        let block = ContentBlock {
            content_type: "tool_result".to_string(),
            tool_use_id: Some("toolu_01".to_string()),
            content: Some("File not found".to_string()),
            is_error: Some(true),
            text: None,
            id: None,
            name: None,
            input: None,
            source: None,
        };

        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "tool_result");
        assert_eq!(json["tool_use_id"], "toolu_01");
        assert_eq!(json["content"], "File not found");
        assert_eq!(json["is_error"], true);
    }

    #[test]
    fn test_content_block_is_error_not_serialized_when_none() {
        // is_error must not appear in JSON when None (API rejects unknown fields).
        let block = ContentBlock {
            content_type: "tool_result".to_string(),
            tool_use_id: Some("toolu_01".to_string()),
            content: Some("42°C".to_string()),
            is_error: None,
            text: None,
            id: None,
            name: None,
            input: None,
            source: None,
        };

        let json = serde_json::to_value(&block).unwrap();
        assert!(
            json.get("is_error").is_none(),
            "is_error must be omitted from JSON when None"
        );
    }

    #[test]
    fn test_delta_block_no_spurious_serialize_attrs() {
        // DeltaBlock only derives Deserialize, so skip_serializing_if was dead
        // code that could confuse readers. Verify it parses correctly instead.
        let json = r#"{"type":"text_delta","text":"hello","partial_json":null,"thinking":null}"#;
        let delta: DeltaBlock = serde_json::from_str(json).unwrap();
        assert_eq!(delta.delta_type, "text_delta");
        assert_eq!(delta.text, Some("hello".to_string()));
        assert_eq!(delta.partial_json, None);
        assert_eq!(delta.thinking, None);
    }

    #[test]
    fn test_delta_block_parses_signature_delta() {
        // signature_delta events must deserialize without error.
        let json = r#"{"type":"signature_delta","signature":"EqQBCgIYAhIM1gbcDa9GJwZA2b3hGg=="}"#;
        let delta: DeltaBlock = serde_json::from_str(json).unwrap();
        assert_eq!(delta.delta_type, "signature_delta");
        assert_eq!(
            delta.signature,
            Some("EqQBCgIYAhIM1gbcDa9GJwZA2b3hGg==".to_string())
        );
    }

    #[test]
    fn test_supports_function_calling() {
        let provider = AnthropicProvider::new("key");
        assert!(
            provider.supports_function_calling(),
            "AnthropicProvider must advertise function-calling support"
        );
    }

    #[test]
    fn test_handle_error_401() {
        let body = r#"{"type":"error","error":{"type":"authentication_error","message":"Invalid API key"}}"#;
        let err = AnthropicProvider::handle_error(reqwest::StatusCode::UNAUTHORIZED, body);
        assert!(matches!(err, crate::error::LlmError::AuthError(_)));
    }

    #[test]
    fn test_handle_error_400() {
        let body =
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"Bad request"}}"#;
        let err = AnthropicProvider::handle_error(reqwest::StatusCode::BAD_REQUEST, body);
        assert!(matches!(err, crate::error::LlmError::InvalidRequest(_)));
    }

    #[test]
    fn test_handle_error_402() {
        let body =
            r#"{"type":"error","error":{"type":"billing_error","message":"Payment required"}}"#;
        let err = AnthropicProvider::handle_error(reqwest::StatusCode::PAYMENT_REQUIRED, body);
        assert!(matches!(err, crate::error::LlmError::ApiError(_)));
        if let crate::error::LlmError::ApiError(msg) = err {
            assert!(msg.contains("Billing error"));
        }
    }

    #[test]
    fn test_handle_error_403() {
        let body =
            r#"{"type":"error","error":{"type":"permission_error","message":"No permission"}}"#;
        let err = AnthropicProvider::handle_error(reqwest::StatusCode::FORBIDDEN, body);
        assert!(matches!(err, crate::error::LlmError::AuthError(_)));
        if let crate::error::LlmError::AuthError(msg) = err {
            assert!(msg.contains("Permission denied"));
        }
    }

    #[test]
    fn test_handle_error_404() {
        let body =
            r#"{"type":"error","error":{"type":"not_found_error","message":"Model not found"}}"#;
        let err = AnthropicProvider::handle_error(reqwest::StatusCode::NOT_FOUND, body);
        assert!(matches!(err, crate::error::LlmError::ModelNotFound(_)));
    }

    #[test]
    fn test_handle_error_413() {
        let body = r#"{"type":"error","error":{"type":"request_too_large","message":"Request too large"}}"#;
        let err = AnthropicProvider::handle_error(reqwest::StatusCode::PAYLOAD_TOO_LARGE, body);
        assert!(matches!(
            err,
            crate::error::LlmError::TokenLimitExceeded { .. }
        ));
    }

    #[test]
    fn test_handle_error_429() {
        let body =
            r#"{"type":"error","error":{"type":"rate_limit_error","message":"Too many requests"}}"#;
        let err = AnthropicProvider::handle_error(reqwest::StatusCode::TOO_MANY_REQUESTS, body);
        assert!(matches!(err, crate::error::LlmError::RateLimited(_)));
    }

    #[test]
    fn test_handle_error_500() {
        let body = r#"{"type":"error","error":{"type":"api_error","message":"Internal error"}}"#;
        let err = AnthropicProvider::handle_error(reqwest::StatusCode::INTERNAL_SERVER_ERROR, body);
        assert!(matches!(err, crate::error::LlmError::ApiError(_)));
        if let crate::error::LlmError::ApiError(msg) = err {
            assert!(msg.contains("Anthropic internal error"));
        }
    }

    #[test]
    fn test_handle_error_529_overloaded() {
        // Anthropic uses 529 for overloaded (not a standard HTTP status, but used by the API)
        let body = r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
        let status = reqwest::StatusCode::from_u16(529).unwrap();
        let err = AnthropicProvider::handle_error(status, body);
        assert!(
            matches!(err, crate::error::LlmError::RateLimited(_)),
            "529 overloaded_error should map to RateLimited"
        );
        if let crate::error::LlmError::RateLimited(msg) = err {
            assert!(msg.contains("overloaded"));
        }
    }

    #[test]
    fn test_handle_error_fallback_non_json_body() {
        // If the body is not valid JSON, use raw text as the message.
        let err = AnthropicProvider::handle_error(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            "Service temporarily unavailable",
        );
        assert!(matches!(err, crate::error::LlmError::ApiError(_)));
    }

    #[test]
    fn test_sse_line_buffer_logic() {
        // Simulate what happens when an SSE line is split across two network chunks.
        // This replicates the fix: buffer must accumulate until '\n' is found.
        let chunk1 = "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hel";
        let chunk2 = "lo\"}}\n";

        let mut buffer = String::new();
        buffer.push_str(chunk1);
        // After chunk1: no complete line yet, nothing extracted
        assert!(buffer.find('\n').is_none());

        buffer.push_str(chunk2);
        // After chunk2: one complete line available
        let newline_idx = buffer.find('\n').unwrap();
        let line = buffer[..newline_idx].trim().to_string();
        buffer.drain(..=newline_idx);

        // Parse the complete line
        let data = line.strip_prefix("data: ").unwrap();
        let event: StreamEvent = serde_json::from_str(data).unwrap();
        match event {
            StreamEvent::ContentBlockDelta { delta, .. } => {
                assert_eq!(delta.delta_type, "text_delta");
                assert_eq!(delta.text, Some("hello".to_string()));
            }
            _ => panic!("Expected ContentBlockDelta"),
        }
    }

    #[test]
    fn test_sse_multiple_events_in_one_chunk() {
        // Verify the buffer correctly processes multiple complete events in one chunk.
        let chunk = "data: {\"type\":\"ping\"}\ndata: {\"type\":\"message_stop\"}\n";
        let mut buffer = String::new();
        buffer.push_str(chunk);

        let mut events: Vec<StreamEvent> = Vec::new();
        while let Some(newline_idx) = buffer.find('\n') {
            let line = buffer[..newline_idx].trim().to_string();
            buffer.drain(..=newline_idx);
            if let Some(data) = line.strip_prefix("data: ") {
                if let Ok(event) = serde_json::from_str::<StreamEvent>(data) {
                    events.push(event);
                }
            }
        }

        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], StreamEvent::Ping));
        assert!(matches!(events[1], StreamEvent::MessageStop));
    }

    #[test]
    fn test_convert_messages_tool_role_without_id_falls_back_to_user() {
        // A Tool role message without tool_call_id must fall back gracefully.
        let mut msg = ChatMessage::user("result of tool");
        msg.role = crate::traits::ChatRole::Tool;
        // No tool_call_id set

        let (_, anthropic_messages) = AnthropicProvider::convert_messages(&[msg]);
        assert_eq!(anthropic_messages.len(), 1);
        assert_eq!(anthropic_messages[0].role, "user");
        let json = serde_json::to_value(&anthropic_messages[0]).unwrap();
        assert_eq!(json["content"], "result of tool");
    }
}
