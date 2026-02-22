//! LLM provider traits for text completion and embedding.
//!
//! # Implements
//!
//! @implements FEAT0006 (Vector Embedding Generation via EmbeddingProvider trait)
//! @implements FEAT0017 (Multi-Provider LLM Support via LLMProvider trait)
//! @implements FEAT0018 (Embedding Provider Abstraction)
//!
//! # Enforces
//!
//! - **BR0303**: Token usage tracked in [`LLMResponse`]
//! - **BR0010**: Embedding dimension validated by providers
//!
//! # WHY: Trait-Based Provider Abstraction
//!
//! Using traits instead of concrete types enables:
//! - **Testing**: MockProvider for unit tests (no API calls)
//! - **Flexibility**: Swap providers without code changes
//! - **Cost control**: Route to different providers based on request type
//! - **Resilience**: Fallback providers when primary is unavailable
//!
//! # Key Traits
//!
//! - [`LLMProvider`]: Text completion (chat, extraction prompts)
//! - [`EmbeddingProvider`]: Vector embedding generation

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;

use crate::error::Result;

use futures::stream::BoxStream;

// ============================================================================
// Function/Tool Calling Types (OpenAI-compatible)
// ============================================================================

/// Definition of a tool that the model can call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Type of tool (always "function" for function tools).
    #[serde(rename = "type")]
    pub tool_type: String,

    /// Function definition.
    pub function: FunctionDefinition,
}

impl ToolDefinition {
    /// Create a new function tool definition.
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: JsonValue,
    ) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.into(),
                description: description.into(),
                parameters,
                strict: Some(true),
            },
        }
    }
}

/// Definition of a function that can be called by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Name of the function.
    pub name: String,

    /// Description of what the function does.
    pub description: String,

    /// JSON Schema defining the function parameters.
    pub parameters: JsonValue,

    /// Whether to enforce strict mode for schema validation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// A tool call request from the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,

    /// Type of tool (always "function").
    #[serde(rename = "type")]
    pub call_type: String,

    /// Function call details.
    pub function: FunctionCall,
}

impl ToolCall {
    /// Parse the function arguments as JSON.
    pub fn parse_arguments<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_str(&self.function.arguments).map_err(|e| {
            crate::error::LlmError::InvalidRequest(format!("Failed to parse tool arguments: {}", e))
        })
    }

    /// Get the function name.
    pub fn name(&self) -> &str {
        &self.function.name
    }

    /// Get the raw arguments string.
    pub fn arguments(&self) -> &str {
        &self.function.arguments
    }
}

/// Details of a function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Name of the function to call.
    pub name: String,

    /// JSON-encoded arguments for the function.
    pub arguments: String,
}

/// Tool choice configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// Let the model decide (default).
    Auto(String),

    /// Force the model to use tools.
    Required(String),

    /// Force a specific function.
    Function {
        #[serde(rename = "type")]
        choice_type: String,
        function: ToolChoiceFunction,
    },
}

impl ToolChoice {
    /// Auto mode - model decides when to use tools.
    pub fn auto() -> Self {
        ToolChoice::Auto("auto".to_string())
    }

    /// Required mode - model must use at least one tool.
    pub fn required() -> Self {
        ToolChoice::Required("required".to_string())
    }

    /// Force a specific function to be called.
    pub fn function(name: impl Into<String>) -> Self {
        ToolChoice::Function {
            choice_type: "function".to_string(),
            function: ToolChoiceFunction { name: name.into() },
        }
    }

    /// None mode - disable tool calling.
    pub fn none() -> Self {
        ToolChoice::Auto("none".to_string())
    }
}

/// Specific function choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    /// Name of the function to call.
    pub name: String,
}

/// Result of a tool execution to send back to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// ID of the tool call this result is for.
    pub tool_call_id: String,

    /// Role (always "tool").
    pub role: String,

    /// Content/output of the tool execution.
    pub content: String,
}

impl ToolResult {
    /// Create a new tool result.
    pub fn new(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            role: "tool".to_string(),
            content: content.into(),
        }
    }

    /// Create an error result.
    pub fn error(tool_call_id: impl Into<String>, error: impl std::fmt::Display) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            role: "tool".to_string(),
            content: format!("Error: {}", error),
        }
    }
}

// ============================================================================
// Streaming Types
// ============================================================================

/// Chunk of a streaming response with tool call support.
///
/// OODA-04: Added ThinkingContent for extended thinking/reasoning streaming.
/// OODA-10: Added budget_remaining for thinking budget display.
#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// Partial content/reasoning text.
    Content(String),

    /// Extended thinking/reasoning content (OODA-04, OODA-10).
    ///
    /// Emitted by models supporting extended thinking (Claude, Gemini 2.0 Flash Thinking,
    /// DeepSeek R1/V3). Allows real-time display of model reasoning process.
    ThinkingContent {
        /// The thinking/reasoning text fragment
        text: String,
        /// Tokens used for this thinking chunk (if provider reports it)
        tokens_used: Option<usize>,
        /// Total thinking budget (OODA-10: for budget display like "1.2k/10k")
        budget_total: Option<usize>,
    },

    /// Incremental tool call data.
    ToolCallDelta {
        /// Index of the tool call (for multiple parallel calls).
        index: usize,
        /// Tool call ID (may be sent once at start).
        id: Option<String>,
        /// Function name (may be sent once at start).
        function_name: Option<String>,
        /// Incremental function arguments (JSON fragment).
        function_arguments: Option<String>,
    },

    /// Stream finished with reason.
    ///
    /// OODA-35: Extended with optional provider metrics.
    Finished {
        /// Finish reason (e.g., "stop", "tool_calls", "length").
        reason: String,
        /// Time to first token in milliseconds (if provider reports it).
        /// OODA-35: Added for provider-native TTFT.
        #[allow(dead_code)]
        ttft_ms: Option<f64>,
    },
}

// ============================================================================
// LLM Response with Tool Calls
// ============================================================================

/// Response from an LLM completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    /// The generated text content.
    pub content: String,

    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,

    /// Number of tokens in the completion.
    pub completion_tokens: usize,

    /// Total tokens used.
    pub total_tokens: usize,

    /// Model used for the request.
    pub model: String,

    /// Finish reason (e.g., "stop", "length", "content_filter", "tool_calls").
    pub finish_reason: Option<String>,

    /// Tool calls requested by the model (if any).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,

    /// Additional metadata from the provider.
    pub metadata: HashMap<String, serde_json::Value>,

    /// Number of tokens served from cache (if provider supports caching).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_hit_tokens: Option<usize>,

    /// Number of reasoning/thinking tokens used by the model.
    ///
    /// OODA-15: Extended thinking/reasoning mode capture
    ///
    /// OpenAI o-series: Extracted from `output_tokens_details.reasoning_tokens`
    /// Anthropic Claude: Derived from thinking block token count
    ///
    /// These tokens are billed as output tokens but represent internal reasoning
    /// that precedes the visible response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_tokens: Option<usize>,

    /// Reasoning/thinking content from the model (if available).
    ///
    /// OODA-15: Extended thinking content capture
    ///
    /// Only populated when:
    /// 1. The model supports visible thinking (e.g., Claude extended thinking)
    /// 2. Content capture is enabled (EDGECODE_CAPTURE_CONTENT=true for tracing)
    ///
    /// OpenAI o-series: Reasoning is hidden (not returned via API)
    /// Anthropic Claude: Thinking content returned in thinking blocks
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_content: Option<String>,
}

impl LLMResponse {
    /// Create a new LLM response.
    pub fn new(content: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            model: model.into(),
            finish_reason: None,
            tool_calls: Vec::new(),
            metadata: HashMap::new(),
            cache_hit_tokens: None,
            thinking_tokens: None,
            thinking_content: None,
        }
    }

    /// Set token usage.
    pub fn with_usage(mut self, prompt: usize, completion: usize) -> Self {
        self.prompt_tokens = prompt;
        self.completion_tokens = completion;
        self.total_tokens = prompt + completion;
        self
    }

    /// Set finish reason.
    pub fn with_finish_reason(mut self, reason: impl Into<String>) -> Self {
        self.finish_reason = Some(reason.into());
        self
    }

    /// Add tool calls to the response.
    pub fn with_tool_calls(mut self, calls: Vec<ToolCall>) -> Self {
        self.tool_calls = calls;
        self
    }

    /// Set the number of tokens served from cache.
    ///
    /// # Context Engineering Note
    /// Cache hit tracking is critical for measuring the effectiveness of
    /// prompt caching strategies. Providers like OpenAI, Anthropic, and Gemini
    /// support KV-cache and report cached token counts in their responses.
    ///
    /// A high cache hit rate (>80%) indicates effective context engineering:
    /// - Stable prompt prefixes (no timestamps at start)
    /// - Deterministic message serialization
    /// - Append-only history patterns
    pub fn with_cache_hit_tokens(mut self, tokens: usize) -> Self {
        self.cache_hit_tokens = Some(tokens);
        self
    }

    /// Add metadata to the response.
    ///
    /// # OODA-13: Response ID Capture
    /// Providers should call this to add response IDs and other metadata
    /// for OpenTelemetry GenAI semantic conventions compliance.
    ///
    /// Common keys: "id" (response ID), "system_fingerprint", etc.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set the number of reasoning/thinking tokens.
    ///
    /// # OODA-15: Extended Thinking Token Capture
    /// Use this to record the number of tokens the model used for internal
    /// reasoning before generating the visible response.
    ///
    /// OpenAI o-series: `output_tokens_details.reasoning_tokens`
    /// Anthropic Claude: Derived from thinking block sizes
    ///
    /// These tokens are billed as output tokens but represent hidden reasoning.
    pub fn with_thinking_tokens(mut self, tokens: usize) -> Self {
        self.thinking_tokens = Some(tokens);
        self
    }

    /// Set the reasoning/thinking content.
    ///
    /// # OODA-15: Extended Thinking Content Capture
    /// Use this to record the model's visible thinking/reasoning text.
    ///
    /// Only applicable for models that expose thinking content:
    /// - Anthropic Claude: Returns thinking blocks with visible reasoning
    /// - OpenAI o-series: Reasoning is hidden (do not use this method)
    ///
    /// Content should be captured only when opt-in is enabled
    /// (EDGECODE_CAPTURE_CONTENT=true) due to potential sensitivity.
    pub fn with_thinking_content(mut self, content: impl Into<String>) -> Self {
        self.thinking_content = Some(content.into());
        self
    }

    /// Check if the response has tool calls.
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Check if the response has thinking/reasoning tokens.
    ///
    /// Returns true if the model used extended thinking capabilities.
    pub fn has_thinking(&self) -> bool {
        self.thinking_tokens.is_some() || self.thinking_content.is_some()
    }
}

/// Options for LLM completion requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompletionOptions {
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<usize>,

    /// Temperature for sampling (0.0 = deterministic, 1.0 = creative).
    pub temperature: Option<f32>,

    /// Top-p (nucleus) sampling.
    pub top_p: Option<f32>,

    /// Stop sequences.
    pub stop: Option<Vec<String>>,

    /// Frequency penalty.
    pub frequency_penalty: Option<f32>,

    /// Presence penalty.
    pub presence_penalty: Option<f32>,

    /// Response format (e.g., "json").
    pub response_format: Option<String>,

    /// System prompt to prepend.
    pub system_prompt: Option<String>,
}

impl CompletionOptions {
    /// Create options with a specific temperature.
    pub fn with_temperature(temperature: f32) -> Self {
        Self {
            temperature: Some(temperature),
            ..Default::default()
        }
    }

    /// Create options for JSON output.
    pub fn json_mode() -> Self {
        Self {
            response_format: Some("json_object".to_string()),
            ..Default::default()
        }
    }
}

/// Trait for LLM providers that can generate text completions.
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Get the name of this provider.
    fn name(&self) -> &str;

    /// Get the current model.
    fn model(&self) -> &str;

    /// Get the maximum context length for the model.
    fn max_context_length(&self) -> usize;

    /// Generate a completion for the given prompt.
    async fn complete(&self, prompt: &str) -> Result<LLMResponse>;

    /// Generate a completion with custom options.
    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<LLMResponse>;

    /// Generate a chat completion with messages.
    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse>;

    /// Generate a chat completion with tool/function calling support.
    ///
    /// This method allows the model to call tools/functions defined in the `tools` parameter.
    /// The model may respond with tool_calls in the response, which should be executed
    /// and the results sent back via ToolResult messages.
    ///
    /// # Arguments
    /// * `messages` - The conversation messages
    /// * `tools` - Available tools the model can call
    /// * `tool_choice` - How the model should select tools (auto, required, or specific)
    /// * `options` - Additional completion options
    ///
    /// # Returns
    /// An LLMResponse that may contain tool_calls if the model wants to use tools.
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        // Default implementation: ignore tools and use regular chat
        // Providers that support function calling should override this
        let _ = (tools, tool_choice);
        self.chat(messages, options).await
    }

    /// Generate a streaming completion.
    async fn stream(&self, _prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        Err(crate::error::LlmError::NotSupported(
            "Streaming not supported".to_string(),
        ))
    }

    /// Stream chat completion with tool calling support.
    /// Returns a stream of events containing content chunks, tool call deltas, and finish reasons.
    ///
    /// # Arguments
    /// * `messages` - Chat messages for context
    /// * `tools` - Available tools the model can call
    /// * `tool_choice` - How the model should select tools
    /// * `options` - Additional completion options
    ///
    /// # Returns
    /// A stream of [`StreamChunk`] events that must be accumulated by the consumer.
    async fn chat_with_tools_stream(
        &self,
        _messages: &[ChatMessage],
        _tools: &[ToolDefinition],
        _tool_choice: Option<ToolChoice>,
        _options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        Err(crate::error::LlmError::NotSupported(
            "Streaming tool calls not supported by this provider".to_string(),
        ))
    }

    /// Check if the model supports streaming.
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Check if the provider supports streaming with tool calls.
    fn supports_tool_streaming(&self) -> bool {
        false
    }

    /// Check if the model supports JSON mode.
    fn supports_json_mode(&self) -> bool {
        false
    }

    /// Check if the model supports function/tool calling.
    fn supports_function_calling(&self) -> bool {
        false
    }

    /// Get the model name as an `Option<String>`.
    ///
    /// This is a convenience method for systems that need an optional model name.
    /// Returns Some(model_name) if the model is set, None otherwise.
    ///
    /// # OODA-27: Model-Specific Edit Format Selection
    /// This method is used to determine the optimal edit format based on model capabilities:
    /// - Claude Haiku → WholeFile (format errors common)
    /// - Claude Sonnet → SearchReplace (excellent reliability)
    /// - GPT-4 Turbo → UnifiedDiff (reduces lazy coding)
    fn model_name(&self) -> Option<String> {
        let m = self.model();
        if m.is_empty() {
            None
        } else {
            Some(m.to_string())
        }
    }
}

// ============================================================================
// Image Data for Multimodal Messages (OODA-51)
// ============================================================================

/// Image data for multimodal messages.
///
/// WHY: Vision-capable LLMs (GPT-4V, Claude 3, Gemini Pro Vision) accept images
/// as part of the conversation. This struct provides a provider-agnostic way
/// to attach images to messages, which providers then convert to their specific
/// format (OpenAI: image_url, Anthropic: source.base64).
///
/// # Example
/// ```
/// use edgequake_llm::traits::ImageData;
///
/// let image = ImageData::new("iVBORw0KGgo...", "image/png");
/// assert_eq!(image.mime_type, "image/png");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageData {
    /// Base64-encoded image data (without data: URI prefix).
    pub data: String,

    /// MIME type of the image (e.g., "image/png", "image/jpeg", "image/gif", "image/webp").
    pub mime_type: String,

    /// Optional detail level for vision models.
    /// - "auto": Let the model decide (default)
    /// - "low": Lower resolution, faster, cheaper
    /// - "high": Higher resolution, better for detailed images
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl ImageData {
    /// Create new image data from base64 string and MIME type.
    pub fn new(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            mime_type: mime_type.into(),
            detail: None,
        }
    }

    /// Create image data with specific detail level.
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    /// Create a data URI for the image (OpenAI format).
    ///
    /// Returns: `data:image/png;base64,iVBORw0KGgo...`
    pub fn to_data_uri(&self) -> String {
        format!("data:{};base64,{}", self.mime_type, self.data)
    }

    /// Create image data from a public HTTPS URL.
    ///
    /// The URL is passed directly to the vision API instead of being base64-encoded,
    /// which is more efficient for large images and avoids encoding overhead.
    ///
    /// # Example
    /// ```
    /// use edgequake_llm::traits::ImageData;
    /// let img = ImageData::from_url("https://example.com/photo.jpg");
    /// assert!(img.is_url());
    /// ```
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            data: url.into(),
            mime_type: "url".to_string(),
            detail: None,
        }
    }

    /// Returns true if this image was constructed from a URL (not base64 data).
    pub fn is_url(&self) -> bool {
        self.mime_type == "url"
    }

    /// Returns the URL string for display/URL images, or the data URI for base64 images.
    pub fn to_api_url(&self) -> String {
        if self.is_url() {
            self.data.clone()
        } else {
            self.to_data_uri()
        }
    }

    /// Check if MIME type is supported by most vision APIs.
    pub fn is_supported_mime(&self) -> bool {
        matches!(
            self.mime_type.as_str(),
            "image/png" | "image/jpeg" | "image/gif" | "image/webp" | "url"
        )
    }
}

/// A message in a chat conversation.
/// Cache control hint for providers that support prompt caching (e.g., Anthropic).
///
/// Some LLM providers (notably Anthropic Claude) support explicit cache breakpoints
/// to optimize KV-cache hits and reduce costs by ~90% for cached tokens.
///
/// # Example
/// ```
/// use edgequake_llm::traits::CacheControl;
///
/// let cache = CacheControl::ephemeral();
/// assert_eq!(cache.cache_type, "ephemeral");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheControl {
    /// Cache type. Currently supports "ephemeral" (Anthropic's cache_control.type).
    #[serde(rename = "type")]
    pub cache_type: String,
}

impl CacheControl {
    /// Create an ephemeral cache control (Anthropic's default).
    ///
    /// Ephemeral caches persist for ~5 minutes and are shared across API calls
    /// with the same prefix.
    pub fn ephemeral() -> Self {
        Self {
            cache_type: "ephemeral".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message sender.
    pub role: ChatRole,

    /// Content of the message.
    pub content: String,

    /// Optional name for the message sender.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Tool calls made by the assistant (only for assistant role).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Tool call ID this message is responding to (only for tool role).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// Cache control hint for providers that support prompt caching.
    ///
    /// When set, this tells the provider to establish a cache breakpoint at this message.
    /// Currently supported by Anthropic Claude (cache_control) and Gemini (cachedContent).
    ///
    /// # Example
    /// ```
    /// use edgequake_llm::traits::{ChatMessage, CacheControl};
    ///
    /// let mut msg = ChatMessage::system("You are a helpful assistant");
    /// msg.cache_control = Some(CacheControl::ephemeral());
    /// ```
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,

    /// Optional images for multimodal messages (OODA-51).
    ///
    /// WHY: Vision-capable models accept images alongside text. This field enables
    /// sending images to models like GPT-4V, Claude 3, and Gemini Pro Vision.
    /// Providers convert these to their specific multipart format during serialization.
    ///
    /// # Example
    /// ```
    /// use edgequake_llm::traits::{ChatMessage, ImageData};
    ///
    /// let mut msg = ChatMessage::user("What's in this image?");
    /// msg.images = Some(vec![ImageData::new("iVBORw0...", "image/png")]);
    /// ```
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<ImageData>>,
}

impl ChatMessage {
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            cache_control: None,
            images: None,
        }
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            cache_control: None,
            images: None,
        }
    }

    /// Create a user message with images (OODA-51).
    ///
    /// Use this for multimodal conversations with vision models.
    pub fn user_with_images(content: impl Into<String>, images: Vec<ImageData>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            cache_control: None,
            images: if images.is_empty() {
                None
            } else {
                Some(images)
            },
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            cache_control: None,
            images: None,
        }
    }

    /// Create an assistant message with tool calls.
    pub fn assistant_with_tools(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
            name: None,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
            cache_control: None,
            images: None,
        }
    }

    /// Create a tool response message.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Tool,
            content: content.into(),
            name: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            cache_control: None,
            images: None,
        }
    }

    /// Check if this message has images attached.
    pub fn has_images(&self) -> bool {
        self.images.as_ref().map(|v| !v.is_empty()).unwrap_or(false)
    }
}

/// Role of a chat message sender.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    /// System message for setting context.
    System,
    /// User input message.
    User,
    /// Assistant response message.
    Assistant,
    /// Tool/function result message.
    Tool,
    /// Function/tool result message (deprecated, use Tool).
    Function,
}

impl ChatRole {
    /// Convert role to string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "tool",
            ChatRole::Function => "function",
        }
    }
}

/// Trait for providers that can generate text embeddings.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Get the name of this provider.
    fn name(&self) -> &str;

    /// Get the embedding model.
    fn model(&self) -> &str;

    /// Get the dimension of the embeddings.
    fn dimension(&self) -> usize;

    /// Get the maximum number of tokens per input.
    fn max_tokens(&self) -> usize;

    /// Generate embeddings for a batch of texts.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Generate embedding for a single text.
    async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed(&[text.to_string()]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| crate::error::LlmError::Unknown("Empty embedding result".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_response_builder() {
        let response = LLMResponse::new("Hello, world!", "gpt-4")
            .with_usage(10, 5)
            .with_finish_reason("stop");

        assert_eq!(response.content, "Hello, world!");
        assert_eq!(response.model, "gpt-4");
        assert_eq!(response.prompt_tokens, 10);
        assert_eq!(response.completion_tokens, 5);
        assert_eq!(response.total_tokens, 15);
        assert_eq!(response.finish_reason, Some("stop".to_string()));
    }

    #[test]
    fn test_llm_response_with_cache_hit_tokens() {
        // Test cache hit tracking for context engineering
        let response = LLMResponse::new("cached response", "gemini-pro")
            .with_usage(1000, 50)
            .with_cache_hit_tokens(800);

        assert_eq!(response.cache_hit_tokens, Some(800));
        assert_eq!(response.prompt_tokens, 1000);
        // Verify 80% cache hit rate
        let cache_rate = response.cache_hit_tokens.unwrap() as f64 / response.prompt_tokens as f64;
        assert!((cache_rate - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_llm_response_no_cache_hit_tokens() {
        // Default should be None when not set
        let response = LLMResponse::new("no cache", "gpt-4").with_usage(100, 20);

        assert_eq!(response.cache_hit_tokens, None);
    }

    #[test]
    fn test_chat_message_constructors() {
        let system = ChatMessage::system("You are helpful");
        assert_eq!(system.role, ChatRole::System);

        let user = ChatMessage::user("Hello");
        assert_eq!(user.role, ChatRole::User);

        let assistant = ChatMessage::assistant("Hi there!");
        assert_eq!(assistant.role, ChatRole::Assistant);
    }

    #[test]
    fn test_cache_control_ephemeral() {
        let cache = CacheControl::ephemeral();
        assert_eq!(cache.cache_type, "ephemeral");
    }

    #[test]
    fn test_cache_control_serialization() {
        let cache = CacheControl::ephemeral();
        let json = serde_json::to_value(&cache).unwrap();

        // Should serialize with "type" key (not "cache_type")
        assert_eq!(json["type"], "ephemeral");
        assert!(!json.as_object().unwrap().contains_key("cache_type"));
    }

    #[test]
    fn test_message_with_cache_control() {
        let mut msg = ChatMessage::system("System prompt");
        msg.cache_control = Some(CacheControl::ephemeral());

        let json = serde_json::to_value(&msg).unwrap();

        // Should include cache_control in JSON
        assert!(json.as_object().unwrap().contains_key("cache_control"));
        assert_eq!(json["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_message_without_cache_control() {
        let msg = ChatMessage::user("Hello");

        let json = serde_json::to_value(&msg).unwrap();

        // Should omit cache_control if None (skip_serializing_if)
        assert!(!json.as_object().unwrap().contains_key("cache_control"));
    }

    #[test]
    fn test_cache_control_roundtrip() {
        let original = CacheControl {
            cache_type: "ephemeral".to_string(),
        };

        // Serialize
        let json_str = serde_json::to_string(&original).unwrap();

        // Deserialize
        let deserialized: CacheControl = serde_json::from_str(&json_str).unwrap();

        assert_eq!(original.cache_type, deserialized.cache_type);
    }

    // =========================================================================
    // ImageData Tests (OODA-51)
    // =========================================================================

    #[test]
    fn test_image_data_new() {
        let image = ImageData::new("iVBORw0KGgo...", "image/png");
        assert_eq!(image.mime_type, "image/png");
        assert_eq!(image.data, "iVBORw0KGgo...");
        assert_eq!(image.detail, None);
    }

    #[test]
    fn test_image_data_with_detail() {
        let image = ImageData::new("data123", "image/jpeg").with_detail("high");
        assert_eq!(image.detail, Some("high".to_string()));
    }

    #[test]
    fn test_image_data_to_data_uri() {
        let image = ImageData::new("base64data", "image/png");
        assert_eq!(image.to_data_uri(), "data:image/png;base64,base64data");
    }

    #[test]
    fn test_image_data_supported_mime() {
        assert!(ImageData::new("", "image/png").is_supported_mime());
        assert!(ImageData::new("", "image/jpeg").is_supported_mime());
        assert!(ImageData::new("", "image/gif").is_supported_mime());
        assert!(ImageData::new("", "image/webp").is_supported_mime());
        assert!(!ImageData::new("", "image/bmp").is_supported_mime());
        assert!(!ImageData::new("", "text/plain").is_supported_mime());
    }

    #[test]
    fn test_chat_message_user_with_images() {
        let images = vec![ImageData::new("data1", "image/png")];
        let msg = ChatMessage::user_with_images("What's this?", images);

        assert_eq!(msg.role, ChatRole::User);
        assert_eq!(msg.content, "What's this?");
        assert!(msg.has_images());
        assert_eq!(msg.images.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_chat_message_user_with_empty_images() {
        let msg = ChatMessage::user_with_images("Hello", vec![]);

        assert!(!msg.has_images());
        assert!(msg.images.is_none());
    }

    #[test]
    fn test_image_data_serialization() {
        let image = ImageData::new("base64", "image/png").with_detail("low");
        let json = serde_json::to_value(&image).unwrap();

        assert_eq!(json["data"], "base64");
        assert_eq!(json["mime_type"], "image/png");
        assert_eq!(json["detail"], "low");
    }

    // ---- Iteration 24: Additional traits tests ----

    #[test]
    fn test_tool_definition_function_constructor() {
        let tool = ToolDefinition::function(
            "my_func",
            "Does something",
            serde_json::json!({"type": "object"}),
        );
        assert_eq!(tool.tool_type, "function");
        assert_eq!(tool.function.name, "my_func");
        assert_eq!(tool.function.description, "Does something");
        assert_eq!(tool.function.strict, Some(true));
    }

    #[test]
    fn test_tool_definition_serialization() {
        let tool = ToolDefinition::function(
            "search",
            "Search the web",
            serde_json::json!({"type": "object", "properties": {}}),
        );
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "search");
    }

    #[test]
    fn test_tool_call_name_and_arguments() {
        let tc = ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"city": "Paris"}"#.to_string(),
            },
        };
        assert_eq!(tc.name(), "get_weather");
        assert_eq!(tc.arguments(), r#"{"city": "Paris"}"#);
    }

    #[test]
    fn test_tool_call_parse_arguments() {
        let tc = ToolCall {
            id: "call_2".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "add".to_string(),
                arguments: r#"{"a": 1, "b": 2}"#.to_string(),
            },
        };
        let parsed: serde_json::Value = tc.parse_arguments().unwrap();
        assert_eq!(parsed["a"], 1);
        assert_eq!(parsed["b"], 2);
    }

    #[test]
    fn test_tool_call_parse_arguments_invalid() {
        let tc = ToolCall {
            id: "call_3".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "bad".to_string(),
                arguments: "not json".to_string(),
            },
        };
        let result: std::result::Result<serde_json::Value, _> = tc.parse_arguments();
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_choice_auto() {
        let tc = ToolChoice::auto();
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json, "auto");
    }

    #[test]
    fn test_tool_choice_required() {
        let tc = ToolChoice::required();
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json, "required");
    }

    #[test]
    fn test_tool_choice_none() {
        let tc = ToolChoice::none();
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json, "none");
    }

    #[test]
    fn test_tool_choice_function() {
        let tc = ToolChoice::function("get_weather");
        if let ToolChoice::Function {
            choice_type,
            function,
        } = tc
        {
            assert_eq!(choice_type, "function");
            assert_eq!(function.name, "get_weather");
        } else {
            panic!("Expected ToolChoice::Function");
        }
    }

    #[test]
    fn test_tool_result_new() {
        let tr = ToolResult::new("call_1", "sunny, 20C");
        assert_eq!(tr.tool_call_id, "call_1");
        assert_eq!(tr.role, "tool");
        assert_eq!(tr.content, "sunny, 20C");
    }

    #[test]
    fn test_tool_result_error() {
        let tr = ToolResult::error("call_2", "City not found");
        assert_eq!(tr.tool_call_id, "call_2");
        assert_eq!(tr.content, "Error: City not found");
    }

    #[test]
    fn test_llm_response_with_tool_calls() {
        let tc = vec![ToolCall {
            id: "c1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "search".to_string(),
                arguments: "{}".to_string(),
            },
        }];
        let resp = LLMResponse::new("", "gpt-4").with_tool_calls(tc);
        assert!(resp.has_tool_calls());
        assert_eq!(resp.tool_calls.len(), 1);
    }

    #[test]
    fn test_llm_response_no_tool_calls() {
        let resp = LLMResponse::new("hello", "gpt-4");
        assert!(!resp.has_tool_calls());
    }

    #[test]
    fn test_llm_response_with_metadata() {
        let resp =
            LLMResponse::new("hi", "gpt-4").with_metadata("id", serde_json::json!("resp_123"));
        assert_eq!(
            resp.metadata.get("id"),
            Some(&serde_json::json!("resp_123"))
        );
    }

    #[test]
    fn test_llm_response_with_thinking() {
        let resp = LLMResponse::new("answer", "claude-3")
            .with_thinking_tokens(500)
            .with_thinking_content("Let me think...");
        assert!(resp.has_thinking());
        assert_eq!(resp.thinking_tokens, Some(500));
        assert_eq!(resp.thinking_content, Some("Let me think...".to_string()));
    }

    #[test]
    fn test_llm_response_has_thinking_tokens_only() {
        let resp = LLMResponse::new("x", "o1").with_thinking_tokens(100);
        assert!(resp.has_thinking());
    }

    #[test]
    fn test_llm_response_has_thinking_content_only() {
        let resp = LLMResponse::new("x", "claude").with_thinking_content("hmm");
        assert!(resp.has_thinking());
    }

    #[test]
    fn test_llm_response_no_thinking() {
        let resp = LLMResponse::new("x", "gpt-4");
        assert!(!resp.has_thinking());
    }

    #[test]
    fn test_completion_options_default() {
        let opts = CompletionOptions::default();
        assert!(opts.max_tokens.is_none());
        assert!(opts.temperature.is_none());
        assert!(opts.response_format.is_none());
    }

    #[test]
    fn test_completion_options_with_temperature() {
        let opts = CompletionOptions::with_temperature(0.7);
        assert_eq!(opts.temperature, Some(0.7));
        assert!(opts.max_tokens.is_none());
    }

    #[test]
    fn test_completion_options_json_mode() {
        let opts = CompletionOptions::json_mode();
        assert_eq!(opts.response_format, Some("json_object".to_string()));
    }

    #[test]
    fn test_chat_role_as_str() {
        assert_eq!(ChatRole::System.as_str(), "system");
        assert_eq!(ChatRole::User.as_str(), "user");
        assert_eq!(ChatRole::Assistant.as_str(), "assistant");
        assert_eq!(ChatRole::Tool.as_str(), "tool");
        assert_eq!(ChatRole::Function.as_str(), "function");
    }

    #[test]
    fn test_chat_role_serialization() {
        let json = serde_json::to_value(ChatRole::User).unwrap();
        assert_eq!(json, "user");
        let json = serde_json::to_value(ChatRole::Tool).unwrap();
        assert_eq!(json, "tool");
    }

    #[test]
    fn test_chat_message_assistant_with_tools() {
        let tc = vec![ToolCall {
            id: "c1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "search".to_string(),
                arguments: "{}".to_string(),
            },
        }];
        let msg = ChatMessage::assistant_with_tools("I'll search", tc);
        assert_eq!(msg.role, ChatRole::Assistant);
        assert!(msg.tool_calls.is_some());
        assert_eq!(msg.tool_calls.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_chat_message_assistant_with_empty_tools() {
        let msg = ChatMessage::assistant_with_tools("just text", vec![]);
        assert!(msg.tool_calls.is_none());
    }

    #[test]
    fn test_chat_message_tool_result() {
        let msg = ChatMessage::tool_result("call_1", "result data");
        assert_eq!(msg.role, ChatRole::Tool);
        assert_eq!(msg.tool_call_id, Some("call_1".to_string()));
        assert_eq!(msg.content, "result data");
    }

    #[test]
    fn test_chat_message_has_images_false() {
        let msg = ChatMessage::user("hello");
        assert!(!msg.has_images());
    }

    #[test]
    fn test_image_data_equality() {
        let a = ImageData::new("data", "image/png");
        let b = ImageData::new("data", "image/png");
        assert_eq!(a, b);

        let c = ImageData::new("data", "image/jpeg");
        assert_ne!(a, c);
    }

    #[test]
    fn test_stream_chunk_content() {
        let chunk = StreamChunk::Content("hello".to_string());
        if let StreamChunk::Content(text) = chunk {
            assert_eq!(text, "hello");
        } else {
            panic!("Expected Content");
        }
    }

    #[test]
    fn test_stream_chunk_thinking() {
        let chunk = StreamChunk::ThinkingContent {
            text: "reasoning...".to_string(),
            tokens_used: Some(50),
            budget_total: Some(10000),
        };
        if let StreamChunk::ThinkingContent {
            text,
            tokens_used,
            budget_total,
        } = chunk
        {
            assert_eq!(text, "reasoning...");
            assert_eq!(tokens_used, Some(50));
            assert_eq!(budget_total, Some(10000));
        }
    }

    #[test]
    fn test_stream_chunk_finished() {
        let chunk = StreamChunk::Finished {
            reason: "stop".to_string(),
            ttft_ms: Some(120.5),
        };
        if let StreamChunk::Finished { reason, ttft_ms } = chunk {
            assert_eq!(reason, "stop");
            assert_eq!(ttft_ms, Some(120.5));
        }
    }

    #[test]
    fn test_stream_chunk_tool_call_delta() {
        let chunk = StreamChunk::ToolCallDelta {
            index: 0,
            id: Some("call_1".to_string()),
            function_name: Some("search".to_string()),
            function_arguments: Some(r#"{"q":"#.to_string()),
        };
        if let StreamChunk::ToolCallDelta {
            index,
            id,
            function_name,
            function_arguments,
        } = chunk
        {
            assert_eq!(index, 0);
            assert_eq!(id, Some("call_1".to_string()));
            assert_eq!(function_name, Some("search".to_string()));
            assert!(function_arguments.is_some());
        }
    }
}
