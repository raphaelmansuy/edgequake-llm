//! VSCode Copilot LLM provider.
//!
//! This provider integrates with GitHub Copilot via the copilot-api proxy.
//!
//! # Phase 1: Proxy-Based Integration
//!
//! This implementation connects to a local copilot-api proxy server running
//! on localhost:4141 (by default). The proxy handles GitHub authentication
//! and token management.
//!
//! # Examples
//!
//! ```no_run
//! use edgequake_llm::{VsCodeCopilotProvider, LLMProvider};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = VsCodeCopilotProvider::new()
//!     .model("gpt-4o-mini")
//!     .build()?;
//!
//! let response = provider.complete("Hello, world!").await?;
//! println!("Response: {}", response.content);
//! # Ok(())
//! # }
//! ```
//!
//! # Setup
//!
//! 1. Install and authenticate with copilot-api:
//!    ```bash
//!    cd copilot-api
//!    bun install
//!    bun run auth
//!    ```
//!
//! 2. Start the proxy server:
//!    ```bash
//!    bun run start
//!    ```
//!
//! 3. Use the provider in your Rust code.
//!
//! # See Also
//!
//! - [copilot-api repository](https://github.com/ericc-ch/copilot-api)
//! - [LLMProvider trait](../../traits/trait.LLMProvider.html)

pub mod auth;
mod client;
mod error;
mod stream;
pub mod token;
pub mod types;

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use std::time::Duration;
use tracing::debug;

pub use client::{AccountType, VsCodeCopilotClient};
pub use error::{Result, VsCodeError};
pub use types::{Model, ModelsResponse};

use crate::error::Result as LlmResult;
use crate::traits::{
    ChatMessage, ChatRole, CompletionOptions, EmbeddingProvider, FunctionCall, LLMProvider,
    LLMResponse, StreamChunk, ToolCall, ToolChoice, ToolDefinition,
};
use types::{
    ChatCompletionRequest, ContentPart, EmbeddingInput, EmbeddingRequest, ImageUrlContent,
    RequestContent, RequestFunction, RequestMessage, RequestTool, ResponseFormat,
};

/// VSCode Copilot LLM provider (proxy-based).
///
/// Connects to copilot-api proxy for GitHub Copilot access.
#[derive(Clone)]
pub struct VsCodeCopilotProvider {
    /// HTTP client for proxy communication.
    client: VsCodeCopilotClient,

    /// Model identifier (e.g., "gpt-4o-mini", "gpt-4o").
    model: String,

    /// Maximum context window size.
    max_context_length: usize,

    /// Whether vision mode is supported/enabled.
    #[allow(dead_code)]
    supports_vision: bool,

    /// Embedding model to use.
    embedding_model: String,

    /// Embedding dimension for the selected model.
    embedding_dimension: usize,
}

impl VsCodeCopilotProvider {
    /// Create a new provider builder with default settings.
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> VsCodeCopilotProviderBuilder {
        VsCodeCopilotProviderBuilder::default()
    }

    /// Create a provider builder with custom proxy URL.
    pub fn with_proxy(proxy_url: impl Into<String>) -> VsCodeCopilotProviderBuilder {
        VsCodeCopilotProviderBuilder::new().proxy_url(proxy_url)
    }

    /// Get a reference to the HTTP client for advanced operations.
    pub fn get_client(&self) -> &VsCodeCopilotClient {
        &self.client
    }

    /// List available models from the Copilot API.
    ///
    /// # OODA-79: Dynamic Model Discovery
    ///
    /// Delegates to the underlying client to fetch available models.
    /// Returns models that are available for the authenticated user.
    pub async fn list_models(&self) -> Result<types::ModelsResponse> {
        self.client.list_models().await
    }

    /// Convert internal messages to API format.
    ///
    /// # OODA-55: Multipart Image Support
    ///
    /// Handles messages with images using multipart content format:
    /// ```text
    /// ┌─────────────────────────────────────────────┐
    /// │ RequestMessage                              │
    /// ├─────────────────────────────────────────────┤
    /// │ content: RequestContent                     │
    /// │   ├── Text(String)        ← text-only      │
    /// │   └── Parts(Vec<ContentPart>)              │
    /// │         ├── Text { text }                  │
    /// │         └── ImageUrl { image_url: {        │
    /// │               url: "data:image/png;..."    │
    /// │               detail: "auto"               │
    /// │             }}                             │
    /// └─────────────────────────────────────────────┘
    /// ```
    fn convert_messages(messages: &[ChatMessage]) -> Vec<RequestMessage> {
        messages
            .iter()
            .map(|msg| {
                // Convert tool calls if present (for assistant messages)
                let tool_calls = msg.tool_calls.as_ref().map(|calls| {
                    calls
                        .iter()
                        .map(|tc| types::ResponseToolCall {
                            id: tc.id.clone(),
                            call_type: "function".to_string(),
                            function: types::ResponseFunctionCall {
                                name: tc.name().to_string(),
                                arguments: tc.arguments().to_string(),
                            },
                        })
                        .collect()
                });

                // Convert cache control if present
                let cache_control =
                    msg.cache_control
                        .as_ref()
                        .map(|cc| types::RequestCacheControl {
                            cache_type: cc.cache_type.clone(),
                        });

                // OODA-55: Build content based on whether images are present
                // User messages with images use multipart format, others use simple text
                let content = if msg.content.is_empty() && tool_calls.is_some() {
                    None // OpenAI API expects no content when there are only tool calls
                } else if msg.has_images() {
                    // Build multipart content with text + images
                    let mut parts: Vec<ContentPart> = Vec::new();

                    // Add text part first (if not empty)
                    if !msg.content.is_empty() {
                        parts.push(ContentPart::Text {
                            text: msg.content.clone(),
                        });
                    }

                    // Add image parts
                    if let Some(images) = &msg.images {
                        for img in images {
                            // Build data URI: data:<mime_type>;base64,<data>
                            let data_uri = format!(
                                "data:{};base64,{}",
                                img.mime_type, img.data
                            );
                            parts.push(ContentPart::ImageUrl {
                                image_url: ImageUrlContent {
                                    url: data_uri,
                                    detail: img.detail.clone(),
                                },
                            });
                        }
                    }

                    Some(RequestContent::Parts(parts))
                } else {
                    // Simple text content
                    Some(RequestContent::Text(msg.content.clone()))
                };

                RequestMessage {
                    role: match msg.role {
                        ChatRole::System => "system".to_string(),
                        ChatRole::User => "user".to_string(),
                        ChatRole::Assistant => "assistant".to_string(),
                        ChatRole::Tool => "tool".to_string(),
                        ChatRole::Function => "tool".to_string(),
                    },
                    content,
                    name: msg.name.clone(),
                    tool_calls,
                    tool_call_id: msg.tool_call_id.clone(),
                    cache_control,
                }
            })
            .collect()
    }

    /// Convert tool definitions to API format.
    fn convert_tools(tools: &[ToolDefinition]) -> Vec<RequestTool> {
        tools
            .iter()
            .map(|tool| RequestTool {
                tool_type: "function".to_string(),
                function: RequestFunction {
                    name: tool.function.name.clone(),
                    description: tool.function.description.clone(),
                    parameters: tool.function.parameters.clone(),
                    strict: tool.function.strict,
                },
            })
            .collect()
    }

    /// Convert tool choice to API format.
    fn convert_tool_choice(choice: Option<ToolChoice>) -> Option<serde_json::Value> {
        choice.map(|c| match c {
            ToolChoice::Auto(s) | ToolChoice::Required(s) => serde_json::Value::String(s),
            ToolChoice::Function { function, .. } => {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": function.name
                    }
                })
            }
        })
    }

    /// Convert response tool calls to internal format.
    fn convert_response_tool_calls(calls: Option<Vec<types::ResponseToolCall>>) -> Vec<ToolCall> {
        calls
            .unwrap_or_default()
            .into_iter()
            .map(|tc| ToolCall {
                id: tc.id,
                call_type: tc.call_type,
                function: FunctionCall {
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                },
            })
            .collect()
    }
}

impl Default for VsCodeCopilotProvider {
    fn default() -> Self {
        Self::new()
            .build()
            .expect("Failed to build default VsCodeCopilotProvider")
    }
}

/// Builder for VsCodeCopilotProvider.
///
/// # Example
///
/// ```rust,no_run
/// use edgequake_llm::VsCodeCopilotProvider;
///
/// // Direct mode (default - recommended)
/// let provider = VsCodeCopilotProvider::new()
///     .direct()  // Use direct API (default)
///     .model("gpt-4o")
///     .build()?;
///
/// // Proxy mode (legacy)
/// let provider = VsCodeCopilotProvider::new()
///     .proxy_url("http://localhost:4141")
///     .model("gpt-4o-mini")
///     .build()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone)]
pub struct VsCodeCopilotProviderBuilder {
    /// Base URL for the API (proxy URL or direct API URL).
    base_url: Option<String>,
    /// Model name.
    model: String,
    /// Maximum context length.
    max_context_length: usize,
    /// Whether vision is supported.
    supports_vision: bool,
    /// Request timeout.
    timeout: Duration,
    /// Whether to use direct API mode.
    direct_mode: bool,
    /// Account type for direct mode.
    account_type: client::AccountType,
    /// Embedding model to use.
    embedding_model: String,
    /// Embedding dimension.
    embedding_dimension: usize,
}

impl Default for VsCodeCopilotProviderBuilder {
    fn default() -> Self {
        // Check environment variable for direct mode preference
        let direct_mode = std::env::var("VSCODE_COPILOT_DIRECT")
            .map(|v| v.to_lowercase() != "false" && v != "0")
            .unwrap_or(true); // Default to direct mode

        // Check environment for account type
        let account_type = std::env::var("VSCODE_COPILOT_ACCOUNT_TYPE")
            .ok()
            .and_then(|s| client::AccountType::from_str(&s))
            .unwrap_or_default();

        // Check environment for embedding model
        let embedding_model = std::env::var("VSCODE_COPILOT_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "text-embedding-3-small".to_string());

        // Set dimension based on model
        let embedding_dimension = Self::dimension_for_embedding_model(&embedding_model);

        Self {
            base_url: None,
            model: "gpt-4o-mini".to_string(),
            max_context_length: 128_000,
            supports_vision: false,
            timeout: Duration::from_secs(120),
            direct_mode,
            account_type,
            embedding_model,
            embedding_dimension,
        }
    }
}

impl VsCodeCopilotProviderBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the proxy URL (enables proxy mode).
    ///
    /// This disables direct mode and connects through a local copilot-api proxy.
    pub fn proxy_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self.direct_mode = false;
        self
    }

    /// Enable direct API mode (default).
    ///
    /// Connects directly to api.githubcopilot.com without a proxy.
    pub fn direct(mut self) -> Self {
        self.direct_mode = true;
        self.base_url = None;
        self
    }

    /// Set the account type for direct mode.
    ///
    /// - `Individual` - Personal GitHub Copilot subscription
    /// - `Business` - GitHub Copilot Business
    /// - `Enterprise` - GitHub Copilot Enterprise
    pub fn account_type(mut self, account_type: client::AccountType) -> Self {
        self.account_type = account_type;
        self
    }

    /// Set the model to use.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        let model_str = model.into();
        self.max_context_length = Self::context_length_for_model(&model_str);

        // Increase timeout for Grok models (known to be slower)
        if model_str.contains("grok") {
            self.timeout = Duration::from_secs(300); // 5 minutes for Grok
        }

        self.model = model_str;
        self
    }

    /// Set the embedding model to use.
    ///
    /// Supported models include:
    /// - `text-embedding-3-small` (default, 1536 dimensions)
    /// - `text-embedding-3-large` (3072 dimensions)
    /// - `text-embedding-ada-002` (1536 dimensions)
    pub fn embedding_model(mut self, model: impl Into<String>) -> Self {
        let model_str = model.into();
        self.embedding_dimension = Self::dimension_for_embedding_model(&model_str);
        self.embedding_model = model_str;
        self
    }

    /// Enable or disable vision support.
    pub fn with_vision(mut self, enabled: bool) -> Self {
        self.supports_vision = enabled;
        self
    }

    /// Set the request timeout.
    pub fn timeout(mut self, duration: Duration) -> Self {
        self.timeout = duration;
        self
    }

    /// Build the provider.
    pub fn build(self) -> Result<VsCodeCopilotProvider> {
        let client = if let Some(url) = &self.base_url {
            // Proxy mode with custom URL
            VsCodeCopilotClient::with_base_url(url, self.timeout)?
        } else if self.direct_mode {
            // Direct mode
            VsCodeCopilotClient::new_with_options(self.timeout, true, self.account_type)?
                .with_vision(self.supports_vision)
        } else {
            // Proxy mode with default URL
            let proxy_url = std::env::var("VSCODE_COPILOT_PROXY_URL")
                .unwrap_or_else(|_| "http://localhost:4141".to_string());
            VsCodeCopilotClient::with_base_url(&proxy_url, self.timeout)?
        };

        let mode_str = if self.direct_mode { "direct" } else { "proxy" };

        debug!(
            model = %self.model,
            max_context = self.max_context_length,
            mode = mode_str,
            account_type = ?self.account_type,
            embedding_model = %self.embedding_model,
            "Built VsCodeCopilotProvider"
        );

        Ok(VsCodeCopilotProvider {
            client,
            model: self.model,
            max_context_length: self.max_context_length,
            supports_vision: self.supports_vision,
            embedding_model: self.embedding_model,
            embedding_dimension: self.embedding_dimension,
        })
    }

    /// Get context length for a model.
    fn context_length_for_model(model: &str) -> usize {
        match model {
            m if m.contains("grok") => 131_072, // Grok models have 131K context window
            m if m.contains("gpt-4o") => 128_000,
            m if m.contains("gpt-4-turbo") => 128_000,
            m if m.contains("gpt-4-32k") => 32_768,
            m if m.contains("gpt-4") => 8_192,
            m if m.contains("gpt-3.5-turbo-16k") => 16_384,
            m if m.contains("gpt-3.5") => 4_096,
            m if m.contains("o1") || m.contains("o3") => 200_000,
            _ => 128_000, // Conservative default
        }
    }

    /// Get embedding dimension for a model.
    fn dimension_for_embedding_model(model: &str) -> usize {
        match model {
            m if m.contains("text-embedding-3-large") => 3072,
            m if m.contains("text-embedding-3-small") => 1536,
            m if m.contains("text-embedding-ada") => 1536,
            m if m.contains("copilot-text-embedding") => 1536,
            _ => 1536, // Conservative default
        }
    }
}

#[async_trait]
impl LLMProvider for VsCodeCopilotProvider {
    fn name(&self) -> &str {
        "vscode-copilot"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_context_length(&self) -> usize {
        self.max_context_length
    }

    async fn complete(&self, prompt: &str) -> LlmResult<LLMResponse> {
        self.complete_with_options(prompt, &CompletionOptions::default())
            .await
    }

    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> LlmResult<LLMResponse> {
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
    ) -> LlmResult<LLMResponse> {
        // Convert messages
        let request_messages = Self::convert_messages(messages);

        // Build request
        let opts = options.cloned().unwrap_or_default();
        let request = ChatCompletionRequest {
            messages: request_messages,
            model: self.model.clone(),
            temperature: opts.temperature,
            top_p: opts.top_p,
            max_tokens: opts.max_tokens,
            stop: opts.stop,
            stream: Some(false),
            frequency_penalty: opts.frequency_penalty,
            presence_penalty: opts.presence_penalty,
            response_format: opts
                .response_format
                .map(|fmt| ResponseFormat { format_type: fmt }),
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
        };

        debug!(
            model = %self.model,
            message_count = messages.len(),
            "Sending chat request"
        );

        // Send request
        let response = self.client.chat_completion(request).await?;

        // Extract result
        let choice = response
            .choices
            .first()
            .ok_or_else(|| crate::error::LlmError::ApiError("No choices in response".into()))?;

        let content = choice.message.content.clone().unwrap_or_default();

        let usage = response.usage.unwrap_or(types::Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            prompt_tokens_details: None,
            extra: None,
        });

        debug!(
            prompt_tokens = usage.prompt_tokens,
            completion_tokens = usage.completion_tokens,
            "Chat request completed"
        );

        // Convert any tool calls in the response
        let tool_calls = Self::convert_response_tool_calls(choice.message.tool_calls.clone());

        // OODA-24: Extract cached_tokens for KV cache hit tracking
        let cache_hit_tokens = usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens);

        // OODA-13: Capture response ID for OpenTelemetry GenAI semantic conventions
        let mut response_builder = LLMResponse::new(content, response.model.clone())
            .with_usage(usage.prompt_tokens, usage.completion_tokens)
            .with_finish_reason(choice.finish_reason.clone().unwrap_or_default())
            .with_tool_calls(tool_calls)
            .with_metadata("id", serde_json::json!(response.id));

        // Add cache hit tokens if available (OODA-24)
        if let Some(cached) = cache_hit_tokens {
            response_builder = response_builder.with_cache_hit_tokens(cached);
        }

        Ok(response_builder)
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> LlmResult<LLMResponse> {
        // Convert messages
        let request_messages = Self::convert_messages(messages);

        // Convert tools
        let request_tools = if tools.is_empty() {
            None
        } else {
            Some(Self::convert_tools(tools))
        };

        // Convert tool choice
        let request_tool_choice = Self::convert_tool_choice(tool_choice);

        // Build request
        let opts = options.cloned().unwrap_or_default();
        let request = ChatCompletionRequest {
            messages: request_messages,
            model: self.model.clone(),
            temperature: opts.temperature,
            top_p: opts.top_p,
            max_tokens: opts.max_tokens,
            stop: opts.stop,
            stream: Some(false),
            frequency_penalty: opts.frequency_penalty,
            presence_penalty: opts.presence_penalty,
            response_format: opts
                .response_format
                .map(|fmt| ResponseFormat { format_type: fmt }),
            tools: request_tools,
            tool_choice: request_tool_choice,
            parallel_tool_calls: Some(true),
        };

        debug!(
            model = %self.model,
            message_count = messages.len(),
            tool_count = tools.len(),
            "Sending chat request with tools"
        );

        // Send request
        let response = self.client.chat_completion(request).await?;

        // Extract result
        let choice = response
            .choices
            .first()
            .ok_or_else(|| crate::error::LlmError::ApiError("No choices in response".into()))?;

        let content = choice.message.content.clone().unwrap_or_default();
        let tool_calls = Self::convert_response_tool_calls(choice.message.tool_calls.clone());

        let usage = response.usage.unwrap_or(types::Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            prompt_tokens_details: None,
            extra: None,
        });

        debug!(
            prompt_tokens = usage.prompt_tokens,
            completion_tokens = usage.completion_tokens,
            tool_call_count = tool_calls.len(),
            "Chat with tools request completed"
        );

        // OODA-24: Extract cached_tokens for KV cache hit tracking
        let cache_hit_tokens = usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens);

        // OODA-13: Capture response ID for OpenTelemetry GenAI semantic conventions
        let mut response_builder = LLMResponse::new(content, response.model.clone())
            .with_usage(usage.prompt_tokens, usage.completion_tokens)
            .with_finish_reason(choice.finish_reason.clone().unwrap_or_default())
            .with_tool_calls(tool_calls)
            .with_metadata("id", serde_json::json!(response.id));

        // Add cache hit tokens if available (OODA-24)
        if let Some(cached) = cache_hit_tokens {
            response_builder = response_builder.with_cache_hit_tokens(cached);
        }

        Ok(response_builder)
    }

    async fn stream(&self, prompt: &str) -> LlmResult<BoxStream<'static, LlmResult<String>>> {
        let request_messages = vec![RequestMessage {
            role: "user".to_string(),
            content: Some(RequestContent::Text(prompt.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            cache_control: None,
        }];

        let request = ChatCompletionRequest {
            messages: request_messages,
            model: self.model.clone(),
            stream: Some(true),
            ..Default::default()
        };

        debug!(model = %self.model, "Sending streaming request");

        let response = self.client.chat_completion_stream(request).await?;
        let stream = stream::parse_sse_stream(response);

        // Map VsCodeError to LlmError
        let mapped = stream.map(|result| result.map_err(|e| e.into()));

        Ok(Box::pin(mapped))
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    fn supports_function_calling(&self) -> bool {
        true
    }
    
    /// OODA-05: Enable streaming with tool calls for real-time token display.
    fn supports_tool_streaming(&self) -> bool {
        true
    }
    
    /// Stream LLM response with tool calls (OODA-05).
    ///
    /// Returns a stream of `StreamChunk` events for real-time:
    /// - Content display
    /// - Tool call progress
    /// - Token counting and rate display
    ///
    /// This enables the React agent to use `StreamingProgress` instead of
    /// `SpinnerGuard`, providing `⚡ N tokens (M t/s)` display.
    async fn chat_with_tools_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> LlmResult<BoxStream<'static, LlmResult<StreamChunk>>> {
        // Convert messages
        let request_messages = Self::convert_messages(messages);

        // Convert tools
        let request_tools = if tools.is_empty() {
            None
        } else {
            Some(Self::convert_tools(tools))
        };

        // Convert tool choice
        let request_tool_choice = Self::convert_tool_choice(tool_choice);

        // Build request with streaming enabled
        let opts = options.cloned().unwrap_or_default();
        let request = ChatCompletionRequest {
            messages: request_messages,
            model: self.model.clone(),
            temperature: opts.temperature,
            top_p: opts.top_p,
            max_tokens: opts.max_tokens,
            stop: opts.stop,
            stream: Some(true), // Enable streaming
            frequency_penalty: opts.frequency_penalty,
            presence_penalty: opts.presence_penalty,
            response_format: opts
                .response_format
                .map(|fmt| ResponseFormat { format_type: fmt }),
            tools: request_tools,
            tool_choice: request_tool_choice,
            parallel_tool_calls: Some(true),
        };

        debug!(
            model = %self.model,
            message_count = messages.len(),
            tool_count = tools.len(),
            "Sending streaming chat request with tools (OODA-05)"
        );

        // Send streaming request
        let response = self.client.chat_completion_stream(request).await?;
        
        // Parse SSE stream with tool call support
        let stream = stream::parse_sse_stream_with_tools(response);

        // Map VsCodeError to LlmError
        let mapped = stream.map(|result| result.map_err(|e| e.into()));

        Ok(Box::pin(mapped))
    }
}

#[async_trait]
impl EmbeddingProvider for VsCodeCopilotProvider {
    fn name(&self) -> &str {
        "vscode-copilot"
    }

    #[allow(clippy::misnamed_getters)]
    fn model(&self) -> &str {
        // Note: Returns embedding_model, not the chat model - this is intentional
        // as per EmbeddingProvider trait contract
        &self.embedding_model
    }

    fn dimension(&self) -> usize {
        self.embedding_dimension
    }

    fn max_tokens(&self) -> usize {
        8192 // OpenAI embedding models support up to 8192 tokens
    }

    async fn embed(&self, texts: &[String]) -> LlmResult<Vec<Vec<f32>>> {
        let input = if texts.len() == 1 {
            EmbeddingInput::Single(texts[0].clone())
        } else {
            EmbeddingInput::Multiple(texts.to_vec())
        };

        let request = EmbeddingRequest::new(input, &self.embedding_model);

        debug!(
            model = %self.embedding_model,
            input_count = texts.len(),
            "Sending embedding request"
        );

        let response = self.client.create_embeddings(request).await?;

        debug!(
            prompt_tokens = response.usage.prompt_tokens,
            total_tokens = response.usage.total_tokens,
            embedding_count = response.data.len(),
            "Embedding request completed"
        );

        // Return embeddings in order
        let embeddings: Vec<Vec<f32>> = response
            .data
            .into_iter()
            .map(|e| (e.index, e.embedding))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|(_, e)| e)
            .collect();

        // Sort by index to maintain order (API should return in order, but be safe)
        if embeddings.len() != texts.len() {
            return Err(crate::error::LlmError::ApiError(format!(
                "Expected {} embeddings, got {}",
                texts.len(),
                embeddings.len()
            )));
        }

        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::{ResponseFunctionCall, ResponseToolCall};

    // =========================================================================
    // Tool Conversion Tests
    // WHY: Tool calling is core to coding agent functionality
    // =========================================================================

    #[test]
    fn test_convert_single_tool() {
        // WHY: Verify basic tool definition conversion
        let tools = vec![ToolDefinition::function(
            "read_file",
            "Read contents of a file",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }),
        )];

        let converted = VsCodeCopilotProvider::convert_tools(&tools);

        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].tool_type, "function");
        assert_eq!(converted[0].function.name, "read_file");
        assert_eq!(converted[0].function.description, "Read contents of a file");
        assert!(converted[0].function.strict.is_some());
    }

    #[test]
    fn test_convert_multiple_tools() {
        // WHY: Agent uses multiple tools - order must be preserved
        let tools = vec![
            ToolDefinition::function("tool_a", "First tool", serde_json::json!({})),
            ToolDefinition::function("tool_b", "Second tool", serde_json::json!({})),
            ToolDefinition::function("tool_c", "Third tool", serde_json::json!({})),
        ];

        let converted = VsCodeCopilotProvider::convert_tools(&tools);

        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0].function.name, "tool_a");
        assert_eq!(converted[1].function.name, "tool_b");
        assert_eq!(converted[2].function.name, "tool_c");
    }

    #[test]
    fn test_convert_tool_with_complex_parameters() {
        // WHY: Real tools have nested parameter schemas
        let params = serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "options": {
                    "type": "object",
                    "properties": {
                        "regex": {"type": "boolean"},
                        "case_sensitive": {"type": "boolean"}
                    }
                }
            },
            "required": ["query"]
        });

        let tools = vec![ToolDefinition::function(
            "grep_search",
            "Search codebase",
            params.clone(),
        )];

        let converted = VsCodeCopilotProvider::convert_tools(&tools);

        assert_eq!(converted[0].function.parameters, params);
    }

    // =========================================================================
    // Tool Choice Tests
    // WHY: Tool choice controls model's tool usage behavior
    // =========================================================================

    #[test]
    fn test_tool_choice_none() {
        // WHY: None means let API use defaults
        let result = VsCodeCopilotProvider::convert_tool_choice(None);
        assert!(result.is_none());
    }

    #[test]
    fn test_tool_choice_auto() {
        // WHY: Auto lets model decide when to use tools
        let choice = ToolChoice::auto();
        let result = VsCodeCopilotProvider::convert_tool_choice(Some(choice));

        assert_eq!(result, Some(serde_json::Value::String("auto".to_string())));
    }

    #[test]
    fn test_tool_choice_required() {
        // WHY: Required forces model to use at least one tool
        let choice = ToolChoice::required();
        let result = VsCodeCopilotProvider::convert_tool_choice(Some(choice));

        assert_eq!(
            result,
            Some(serde_json::Value::String("required".to_string()))
        );
    }

    #[test]
    fn test_tool_choice_function() {
        // WHY: Can force model to call specific function
        let choice = ToolChoice::function("read_file");
        let result = VsCodeCopilotProvider::convert_tool_choice(Some(choice));

        let expected = serde_json::json!({
            "type": "function",
            "function": {
                "name": "read_file"
            }
        });

        assert_eq!(result, Some(expected));
    }

    #[test]
    fn test_tool_choice_none_value() {
        // WHY: "none" string disables tool calling entirely
        let choice = ToolChoice::none();
        let result = VsCodeCopilotProvider::convert_tool_choice(Some(choice));

        assert_eq!(result, Some(serde_json::Value::String("none".to_string())));
    }

    // =========================================================================
    // Response Tool Call Conversion Tests
    // WHY: Must correctly parse tool calls from API responses
    // =========================================================================

    #[test]
    fn test_response_tool_calls_none() {
        // WHY: Response may have no tool calls
        let result = VsCodeCopilotProvider::convert_response_tool_calls(None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_response_tool_calls_single() {
        // WHY: Most common case - one tool call
        let calls = vec![ResponseToolCall {
            id: "call_123".to_string(),
            call_type: "function".to_string(),
            function: ResponseFunctionCall {
                name: "read_file".to_string(),
                arguments: r#"{"path":"src/main.rs"}"#.to_string(),
            },
        }];

        let result = VsCodeCopilotProvider::convert_response_tool_calls(Some(calls));

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "call_123");
        assert_eq!(result[0].call_type, "function");
        assert_eq!(result[0].function.name, "read_file");
        assert_eq!(result[0].function.arguments, r#"{"path":"src/main.rs"}"#);
    }

    #[test]
    fn test_response_tool_calls_multiple() {
        // WHY: Model can request multiple tool calls in parallel
        let calls = vec![
            ResponseToolCall {
                id: "call_1".to_string(),
                call_type: "function".to_string(),
                function: ResponseFunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                },
            },
            ResponseToolCall {
                id: "call_2".to_string(),
                call_type: "function".to_string(),
                function: ResponseFunctionCall {
                    name: "search_code".to_string(),
                    arguments: "{}".to_string(),
                },
            },
        ];

        let result = VsCodeCopilotProvider::convert_response_tool_calls(Some(calls));

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, "call_1");
        assert_eq!(result[1].id, "call_2");
    }

    // =========================================================================
    // Message Conversion with Tool Calls Tests
    // WHY: Assistant messages can include tool calls
    // =========================================================================

    #[test]
    fn test_message_with_tool_calls() {
        // WHY: Assistant can respond with tool calls
        let mut msg = ChatMessage::assistant("I'll read that file for you.");
        msg.tool_calls = Some(vec![ToolCall {
            id: "call_abc".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: r#"{"path":"Cargo.toml"}"#.to_string(),
            },
        }]);

        let converted = VsCodeCopilotProvider::convert_messages(&[msg]);

        assert_eq!(converted.len(), 1);
        assert!(converted[0].tool_calls.is_some());

        let tool_calls = converted[0].tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_abc");
        assert_eq!(tool_calls[0].function.name, "read_file");
    }

    #[test]
    fn test_tool_message_conversion() {
        // WHY: Tool results are sent as "tool" role messages
        let msg = ChatMessage {
            role: ChatRole::Tool,
            content: "File contents: ...".to_string(),
            name: Some("read_file".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_xyz".to_string()),
            cache_control: None,
            images: None,
        };

        let converted = VsCodeCopilotProvider::convert_messages(&[msg]);

        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "tool");
        // OODA-55: content is now RequestContent::Text
        assert_eq!(
            converted[0].content,
            Some(RequestContent::Text("File contents: ...".to_string()))
        );
        assert_eq!(converted[0].tool_call_id, Some("call_xyz".to_string()));
    }

    #[test]
    fn test_assistant_message_with_only_tool_calls() {
        // WHY: OpenAI API expects null content when only tool calls present
        let mut msg = ChatMessage::assistant("");
        msg.tool_calls = Some(vec![ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "list_files".to_string(),
                arguments: "{}".to_string(),
            },
        }]);

        let converted = VsCodeCopilotProvider::convert_messages(&[msg]);

        // Content should be None (not empty string) when only tool calls
        assert!(converted[0].content.is_none());
        assert!(converted[0].tool_calls.is_some());
    }

    // =========================================================================
    // OODA-55: Image Serialization Tests
    // =========================================================================
    // WHY: VS Code Copilot uses OpenAI-compatible multipart format for images.
    // These tests verify the correct serialization of image data URIs.

    #[test]
    fn test_convert_messages_text_only() {
        // WHY: Text-only messages should use simple RequestContent::Text
        let messages = vec![ChatMessage::user("Hello, world!")];
        let converted = VsCodeCopilotProvider::convert_messages(&messages);

        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "user");
        match &converted[0].content {
            Some(RequestContent::Text(text)) => {
                assert_eq!(text, "Hello, world!");
            }
            _ => panic!("Expected RequestContent::Text"),
        }
    }

    #[test]
    fn test_convert_messages_with_images() {
        // WHY: Messages with images must use multipart format
        use crate::traits::ImageData;

        let msg = ChatMessage::user_with_images(
            "What's in this image?",
            vec![ImageData {
                data: "iVBORw0KGgo=".to_string(),
                mime_type: "image/png".to_string(),
                detail: None,
            }],
        );

        let converted = VsCodeCopilotProvider::convert_messages(&[msg]);

        assert_eq!(converted.len(), 1);
        match &converted[0].content {
            Some(RequestContent::Parts(parts)) => {
                assert_eq!(parts.len(), 2); // text + image

                // Verify text part
                match &parts[0] {
                    ContentPart::Text { text } => {
                        assert_eq!(text, "What's in this image?");
                    }
                    _ => panic!("First part should be text"),
                }

                // Verify image part
                match &parts[1] {
                    ContentPart::ImageUrl { image_url } => {
                        assert!(image_url.url.starts_with("data:image/png;base64,"));
                        assert!(image_url.url.contains("iVBORw0KGgo="));
                    }
                    _ => panic!("Second part should be image_url"),
                }
            }
            _ => panic!("Expected RequestContent::Parts for image message"),
        }
    }

    #[test]
    fn test_convert_messages_with_image_detail() {
        // WHY: Detail level must be preserved for vision API control
        use crate::traits::ImageData;

        let msg = ChatMessage::user_with_images(
            "Describe in detail",
            vec![ImageData {
                data: "base64data".to_string(),
                mime_type: "image/jpeg".to_string(),
                detail: Some("high".to_string()),
            }],
        );

        let converted = VsCodeCopilotProvider::convert_messages(&[msg]);

        match &converted[0].content {
            Some(RequestContent::Parts(parts)) => {
                assert_eq!(parts.len(), 2);

                match &parts[1] {
                    ContentPart::ImageUrl { image_url } => {
                        assert_eq!(image_url.detail, Some("high".to_string()));
                    }
                    _ => panic!("Expected ImageUrl part"),
                }
            }
            _ => panic!("Expected Parts content"),
        }
    }

    // =========================================================================
    // Original Tests
    // =========================================================================

    #[test]
    fn test_context_length_detection() {
        assert_eq!(
            VsCodeCopilotProviderBuilder::context_length_for_model("gpt-4o"),
            128_000
        );
        assert_eq!(
            VsCodeCopilotProviderBuilder::context_length_for_model("gpt-4o-mini"),
            128_000
        );
        assert_eq!(
            VsCodeCopilotProviderBuilder::context_length_for_model("gpt-4"),
            8_192
        );
        assert_eq!(
            VsCodeCopilotProviderBuilder::context_length_for_model("gpt-3.5-turbo"),
            4_096
        );
        assert_eq!(
            VsCodeCopilotProviderBuilder::context_length_for_model("o1-preview"),
            200_000
        );
    }

    #[test]
    fn test_message_conversion() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello!"),
            ChatMessage::assistant("Hi there!"),
        ];

        let converted = VsCodeCopilotProvider::convert_messages(&messages);

        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0].role, "system");
        // OODA-55: content is now RequestContent::Text
        assert_eq!(
            converted[0].content,
            Some(RequestContent::Text("You are helpful.".to_string()))
        );
        assert_eq!(converted[1].role, "user");
        assert_eq!(converted[2].role, "assistant");
    }

    #[test]
    fn test_builder_defaults() {
        // Set env to ensure consistent test behavior
        std::env::set_var("VSCODE_COPILOT_DIRECT", "true");
        let builder = VsCodeCopilotProviderBuilder::default();
        assert_eq!(builder.model, "gpt-4o-mini");
        assert_eq!(builder.max_context_length, 128_000);
        assert!(!builder.supports_vision);
        assert!(builder.direct_mode); // Direct mode is now default
        std::env::remove_var("VSCODE_COPILOT_DIRECT");
    }

    #[test]
    fn test_builder_proxy_mode() {
        let provider = VsCodeCopilotProvider::new()
            .proxy_url("http://localhost:8080")
            .model("gpt-4")
            .with_vision(true)
            .build()
            .unwrap();

        assert_eq!(provider.model, "gpt-4");
        assert_eq!(provider.max_context_length, 8_192);
        assert!(provider.supports_vision);
    }

    #[test]
    fn test_builder_direct_mode() {
        let provider = VsCodeCopilotProvider::new()
            .direct()
            .model("gpt-4o")
            .build()
            .unwrap();

        assert_eq!(provider.model, "gpt-4o");
        assert_eq!(provider.max_context_length, 128_000);
    }

    #[test]
    fn test_account_type_base_url() {
        assert_eq!(
            client::AccountType::Individual.base_url(),
            "https://api.githubcopilot.com"
        );
        assert_eq!(
            client::AccountType::Business.base_url(),
            "https://api.business.githubcopilot.com"
        );
        assert_eq!(
            client::AccountType::Enterprise.base_url(),
            "https://api.enterprise.githubcopilot.com"
        );
    }

    #[test]
    fn test_embedding_dimension_detection() {
        assert_eq!(
            VsCodeCopilotProviderBuilder::dimension_for_embedding_model("text-embedding-3-small"),
            1536
        );
        assert_eq!(
            VsCodeCopilotProviderBuilder::dimension_for_embedding_model("text-embedding-3-large"),
            3072
        );
        assert_eq!(
            VsCodeCopilotProviderBuilder::dimension_for_embedding_model("text-embedding-ada-002"),
            1536
        );
        assert_eq!(
            VsCodeCopilotProviderBuilder::dimension_for_embedding_model("unknown-model"),
            1536 // Default
        );
    }

    #[test]
    fn test_builder_embedding_model() {
        let provider = VsCodeCopilotProvider::new()
            .direct()
            .embedding_model("text-embedding-3-large")
            .build()
            .unwrap();

        assert_eq!(provider.embedding_model, "text-embedding-3-large");
        assert_eq!(provider.embedding_dimension, 3072);
    }

    // =========================================================================
    // Vision Mode Tests
    // =========================================================================
    //
    // WHY: Vision mode enables image processing via the `copilot-vision-request`
    // header. The TypeScript proxy auto-detects image content in messages,
    // but our Rust implementation uses explicit `with_vision(true)`.
    //
    // Future: Auto-detect vision based on message content type checking.
    // See: copilot-api/src/services/copilot/create-chat-completions.ts:15-17

    #[test]
    fn test_builder_vision_disabled_by_default() {
        // WHY: Vision should be opt-in to avoid header overhead
        let builder = VsCodeCopilotProvider::new().direct();

        // We can build and it should work
        let provider = builder.build();
        assert!(provider.is_ok());

        // Vision is off by default (checked via internal state)
        let provider = provider.unwrap();
        assert!(!provider.supports_vision);
    }

    #[test]
    fn test_builder_with_vision_true() {
        // WHY: User explicitly enables vision for image-containing requests
        let provider = VsCodeCopilotProvider::new()
            .direct()
            .with_vision(true)
            .build()
            .unwrap();

        assert!(provider.supports_vision);
    }

    #[test]
    fn test_builder_with_vision_false() {
        // WHY: User can explicitly disable vision
        let provider = VsCodeCopilotProvider::new()
            .direct()
            .with_vision(true)
            .with_vision(false) // Disable after enabling
            .build()
            .unwrap();

        assert!(!provider.supports_vision);
    }

    #[test]
    fn test_builder_vision_with_model() {
        // WHY: Vision should work with any compatible model
        let provider = VsCodeCopilotProvider::new()
            .direct()
            .model("gpt-4o") // Vision-capable model
            .with_vision(true)
            .build()
            .unwrap();

        assert!(provider.supports_vision);
        assert_eq!(provider.model, "gpt-4o");
    }

    #[test]
    fn test_builder_vision_with_proxy_mode() {
        // WHY: Vision should also work in proxy mode
        let builder = VsCodeCopilotProvider::new()
            .proxy_url("http://localhost:4141")
            .with_vision(true);

        // Build would fail without proper auth, but builder pattern works
        assert!(builder.supports_vision);
    }

    // =========================================================================
    // Builder Chain Tests
    // WHY: Verify all builder options can be chained together
    // =========================================================================

    #[test]
    fn test_builder_chain_all_options() {
        // WHY: Full chain should work without panicking
        use std::time::Duration;

        let builder = VsCodeCopilotProvider::new()
            .model("claude-3.5-sonnet")
            .embedding_model("text-embedding-3-large")
            .with_vision(true)
            .timeout(Duration::from_secs(120));

        // Verify all options were set correctly
        assert_eq!(builder.model, "claude-3.5-sonnet");
        assert_eq!(builder.embedding_model, "text-embedding-3-large");
        assert!(builder.supports_vision);
        assert_eq!(builder.timeout.as_secs(), 120);
    }

    #[test]
    fn test_builder_account_type_business() {
        // WHY: Business accounts use different API endpoint
        use client::AccountType;

        let builder = VsCodeCopilotProvider::new().account_type(AccountType::Business);

        assert!(matches!(builder.account_type, AccountType::Business));
    }

    #[test]
    fn test_builder_account_type_enterprise() {
        // WHY: Enterprise accounts use different API endpoint
        use client::AccountType;

        let builder = VsCodeCopilotProvider::new().account_type(AccountType::Enterprise);

        assert!(matches!(builder.account_type, AccountType::Enterprise));
    }

    // ========================================================================
    // Default Configuration Tests (Iteration 32)
    // ========================================================================
    // WHY: Default values are critical for user experience and must be
    // documented through tests. Changes to defaults affect all users.

    #[test]
    fn test_builder_default_embedding_model() {
        // WHY: Default embedding model affects dimension calculations
        // and compatibility with existing embeddings databases.
        // Value: text-embedding-3-small (matches OpenAI default)
        std::env::remove_var("VSCODE_COPILOT_EMBEDDING_MODEL"); // Clear env override
        let builder = VsCodeCopilotProviderBuilder::default();
        assert_eq!(builder.embedding_model, "text-embedding-3-small");
        assert_eq!(builder.embedding_dimension, 1536);
    }

    #[test]
    fn test_builder_default_timeout() {
        // WHY: Default timeout must be long enough for model responses
        // but not so long that failures hang indefinitely.
        // Value: 120 seconds (2 minutes)
        let builder = VsCodeCopilotProviderBuilder::default();
        assert_eq!(builder.timeout.as_secs(), 120);
    }

    #[test]
    fn test_builder_default_context_length() {
        // WHY: Context length determines how many tokens can be sent.
        // Default 128k matches modern models like gpt-4o-mini.
        std::env::set_var("VSCODE_COPILOT_DIRECT", "true");
        let builder = VsCodeCopilotProviderBuilder::default();
        assert_eq!(builder.max_context_length, 128_000);
        std::env::remove_var("VSCODE_COPILOT_DIRECT");
    }
}
