//! Cohere LLM and Embedding Provider
//!
//! @implements FEAT-012: Cohere provider (chat, embeddings, tool calling)
//!
//! # Overview
//!
//! This provider integrates with the Cohere v2 API (`api.cohere.com/v2`) and exposes:
//! - **Chat completions** (sync + SSE streaming, tool/function calling)
//! - **Embeddings** (`embed-v4.0`, configurable dimensions)
//!
//! The implementation uses raw `reqwest` HTTP calls (no OpenAI-compatible SDK)
//! because Cohere v2's response format differs from OpenAI's.
//!
//! # Environment Variables
//!
//! | Variable | Required | Default | Description |
//! |----------|----------|---------|-------------|
//! | `COHERE_API_KEY` | Yes | - | API key from dashboard.cohere.com |
//!
//! # Available Models (July 2026)
//!
//! | Model | Context | Features |
//! |-------|---------|----------|
//! | `command-a-plus-05-2026` | 128K | Flagship (tools, JSON mode) |
//! | `command-a-03-2025` | 128K | Previous gen |
//! | `command-r-plus` | 128K | Legacy |
//! | `command-r` | 128K | Legacy |
//!
//! # Embedding Models
//!
//! | Model | Dimensions | Max Tokens |
//! |-------|-----------|------------|
//! | `embed-v4.0` | 1536 | 512 |
//! | `embed-multilingual-v3.0` | 1024 | 512 |
//! | `embed-english-v3.0` | 1024 | 512 |
//!
//! # Example
//!
//! ```rust,no_run
//! use edgequake_llm::traits::{ChatMessage, LLMProvider, EmbeddingProvider};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Provider reads COHERE_API_KEY from env
//! // let provider = CohereProvider::from_env()?;
//! // let messages = vec![ChatMessage::user("Hello from Cohere!")];
//! // let resp = provider.chat(&messages, None).await?;
//! // println!("{}", resp.content);
//! # Ok(())
//! # }
//! ```

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::debug;

use crate::error::{LlmError, Result};
use crate::traits::{
    ChatMessage, ChatRole, CompletionOptions, EmbeddingProvider, FunctionCall, LLMProvider,
    LLMResponse, StreamChunk, StreamUsage, ToolCall, ToolChoice, ToolDefinition,
};

// ============================================================================
// Constants
// ============================================================================

const COHERE_BASE_URL: &str = "https://api.cohere.com/v2";
const COHERE_DEFAULT_MODEL: &str = "command-a-plus-05-2026";
const COHERE_DEFAULT_EMBED_MODEL: &str = "embed-v4.0";
const COHERE_DEFAULT_EMBED_DIMENSIONS: usize = 1536;
const COHERE_DEFAULT_MAX_CONTEXT: usize = 128_000;
const COHERE_EMBED_MAX_BATCH_SIZE: usize = 96;
const COHERE_EMBED_MAX_TOKENS: usize = 512;
const COHERE_PROVIDER_NAME: &str = "cohere";

// ============================================================================
// CohereProvider
// ============================================================================

/// Cohere LLM and embedding provider.
///
/// Connects to Cohere's v2 API for chat completions and embeddings.
/// Supports streaming, tool calling, and batch embeddings.
#[derive(Debug, Clone)]
pub struct CohereProvider {
    api_key: String,
    base_url: String,
    model: String,
    embed_model: String,
    embed_dimensions: usize,
    max_context: usize,
    http_client: Client,
    attribution_headers: std::collections::HashMap<String, String>,
}

impl CohereProvider {
    /// Create a new provider with default settings.
    ///
    /// Uses:
    /// - Model: `command-a-plus-05-2026`
    /// - Embed model: `embed-v4.0`
    /// - Base URL: `https://api.cohere.com/v2`
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            api_key: api_key.into(),
            base_url: COHERE_BASE_URL.to_string(),
            model: COHERE_DEFAULT_MODEL.to_string(),
            embed_model: COHERE_DEFAULT_EMBED_MODEL.to_string(),
            embed_dimensions: COHERE_DEFAULT_EMBED_DIMENSIONS,
            max_context: COHERE_DEFAULT_MAX_CONTEXT,
            http_client: client,
            attribution_headers: std::collections::HashMap::new(),
        }
    }

    /// Create a provider from the `COHERE_API_KEY` environment variable.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("COHERE_API_KEY").map_err(|_| {
            LlmError::ConfigError(
                "COHERE_API_KEY environment variable not set. \
                 Get your API key from https://dashboard.cohere.com"
                    .to_string(),
            )
        })?;

        if api_key.is_empty() {
            return Err(LlmError::ConfigError(
                "COHERE_API_KEY is empty. Please set a valid API key.".to_string(),
            ));
        }

        Ok(Self::new(api_key))
    }

    /// Set the chat model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the embedding model.
    pub fn with_embed_model(mut self, model: impl Into<String>) -> Self {
        self.embed_model = model.into();
        self
    }

    /// Set a custom base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the embedding dimensions.
    pub fn with_embed_dimensions(mut self, dimensions: usize) -> Self {
        self.embed_dimensions = dimensions;
        self
    }

    /// Attach [`ApplicationContext`] for Cohere `X-Client-Name` attribution.
    pub fn with_application_context(
        mut self,
        ctx: crate::application_context::ApplicationContext,
    ) -> Self {
        if ctx.is_empty() {
            return self;
        }
        let resolved = crate::http::attribution::resolve_attribution(
            crate::http::attribution::AttributionProviderKind::Cohere,
            &ctx,
        );
        self.attribution_headers.extend(resolved.headers);
        crate::http::attribution::merge_extra_headers(
            &mut self.attribution_headers,
            &ctx.extra_headers,
            crate::http::attribution::AttributionProviderKind::Cohere,
        );
        crate::http::attribution::log_attribution_warnings("cohere", &resolved.warnings);
        self
    }

    fn apply_attribution_headers(
        &self,
        builder: reqwest::RequestBuilder,
    ) -> reqwest::RequestBuilder {
        let mut b = builder;
        for (k, v) in &self.attribution_headers {
            b = b.header(k.as_str(), v.as_str());
        }
        b
    }

    // -----------------------------------------------------------------------
    // Internal: Chat API
    // -----------------------------------------------------------------------

    async fn send_chat_request(&self, request: &CohereChatRequest) -> Result<CohereChatResponse> {
        let url = format!("{}/chat", self.base_url.trim_end_matches('/'));

        debug!(
            model = %request.model,
            stream = request.stream,
            "Cohere chat request"
        );

        let response = self
            .apply_attribution_headers(
                self.http_client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .header("Content-Type", "application/json"),
            )
            .json(request)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(format!("Cohere chat request failed: {}", e)))?;

        let status = response.status();
        let body = response.text().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to read Cohere chat response: {}", e))
        })?;

        if !status.is_success() {
            return Err(Self::map_http_error(status, &body));
        }

        serde_json::from_str(&body).map_err(|e| {
            LlmError::ProviderError(format!(
                "Failed to parse Cohere chat response: {} | body: {}",
                e,
                &body[..body.len().min(500)]
            ))
        })
    }

    async fn send_streaming_request(
        &self,
        request: &CohereChatRequest,
    ) -> Result<reqwest::Response> {
        let url = format!("{}/chat", self.base_url.trim_end_matches('/'));

        let response = self
            .apply_attribution_headers(
                self.http_client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .header("Content-Type", "application/json"),
            )
            .json(request)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(format!("Cohere stream request failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Self::map_http_error(status, &body));
        }

        Ok(response)
    }

    fn build_chat_request(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<&ToolChoice>,
        options: Option<&CompletionOptions>,
        stream: bool,
    ) -> CohereChatRequest {
        let cohere_messages = Self::convert_messages(messages, options);
        let options = options.cloned().unwrap_or_default();

        let cohere_tools = if tools.is_empty() {
            None
        } else {
            Some(Self::convert_tools(tools))
        };

        CohereChatRequest {
            model: self.model.clone(),
            messages: cohere_messages,
            max_tokens: options.max_tokens.map(|t| t as u32),
            temperature: options.temperature,
            top_p: options.top_p,
            stop: options.stop.clone(),
            frequency_penalty: options.frequency_penalty,
            presence_penalty: options.presence_penalty,
            tools: cohere_tools,
            tool_choice: tool_choice.map(Self::convert_tool_choice),
            stream,
            response_format: options.response_format.as_ref().map(|f| {
                if f == "json_object" {
                    CohereResponseFormat {
                        format_type: "json_object".to_string(),
                    }
                } else {
                    CohereResponseFormat {
                        format_type: "text".to_string(),
                    }
                }
            }),
        }
    }

    fn convert_messages(
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Vec<CohereMessage> {
        let mut result = Vec::new();

        // Prepend system prompt from options if provided
        if let Some(opts) = options {
            if let Some(system) = &opts.system_prompt {
                result.push(CohereMessage {
                    role: "system".to_string(),
                    content: Some(CohereContent::Text(system.clone())),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
        }

        for msg in messages {
            let role = match msg.role {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
                ChatRole::Tool => "tool",
                ChatRole::Function => "tool",
            };

            // Handle assistant messages with tool calls
            if msg.role == ChatRole::Assistant {
                if let Some(tool_calls) = &msg.tool_calls {
                    let cohere_tool_calls: Vec<CohereToolCall> = tool_calls
                        .iter()
                        .map(|tc| CohereToolCall {
                            id: tc.id.clone(),
                            call_type: "function".to_string(),
                            function: CohereFunction {
                                name: tc.function.name.clone(),
                                arguments: tc.function.arguments.clone(),
                            },
                        })
                        .collect();

                    result.push(CohereMessage {
                        role: role.to_string(),
                        content: if msg.content.is_empty() {
                            None
                        } else {
                            Some(CohereContent::Text(msg.content.clone()))
                        },
                        tool_calls: Some(cohere_tool_calls),
                        tool_call_id: None,
                    });
                    continue;
                }
            }

            // Handle tool result messages
            if msg.role == ChatRole::Tool {
                result.push(CohereMessage {
                    role: role.to_string(),
                    content: Some(CohereContent::Text(msg.content.clone())),
                    tool_calls: None,
                    tool_call_id: msg.tool_call_id.clone(),
                });
                continue;
            }

            result.push(CohereMessage {
                role: role.to_string(),
                content: Some(CohereContent::Text(msg.content.clone())),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        result
    }

    fn convert_tools(tools: &[ToolDefinition]) -> Vec<CohereTool> {
        tools
            .iter()
            .map(|t| CohereTool {
                tool_type: "function".to_string(),
                function: CohereToolFunction {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    parameters: t.function.parameters.clone(),
                },
            })
            .collect()
    }

    fn convert_tool_choice(choice: &ToolChoice) -> CohereToolChoice {
        match choice {
            ToolChoice::Auto(_) => CohereToolChoice::String("auto".to_string()),
            ToolChoice::Required(_) => CohereToolChoice::String("required".to_string()),
            ToolChoice::Function { function, .. } => {
                CohereToolChoice::Object(CohereToolChoiceObject {
                    choice_type: "function".to_string(),
                    function: CohereToolChoiceFunction {
                        name: function.name.clone(),
                    },
                })
            }
        }
    }

    fn parse_chat_response(response: CohereChatResponse) -> LLMResponse {
        let content = response
            .message
            .content
            .as_ref()
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|b| {
                        if b.block_type == "text" {
                            b.text.as_ref()
                        } else {
                            None
                        }
                    })
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();

        let tool_calls: Vec<ToolCall> = response
            .message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|tc| ToolCall {
                id: tc.id,
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                },
                thought_signature: None,
            })
            .collect();

        let (prompt_tokens, completion_tokens) = response
            .usage
            .as_ref()
            .map(|u| {
                let input = u
                    .tokens
                    .as_ref()
                    .map(|t| t.input_tokens)
                    .or_else(|| u.billed_units.as_ref().map(|b| b.input_tokens))
                    .unwrap_or(0);
                let output = u
                    .tokens
                    .as_ref()
                    .map(|t| t.output_tokens)
                    .or_else(|| u.billed_units.as_ref().map(|b| b.output_tokens))
                    .unwrap_or(0);
                (input as usize, output as usize)
            })
            .unwrap_or((0, 0));

        let finish_reason = response.finish_reason.map(|r| match r.as_str() {
            "COMPLETE" => "stop".to_string(),
            "MAX_TOKENS" => "length".to_string(),
            "TOOL_CALL" => "tool_calls".to_string(),
            other => other.to_lowercase(),
        });

        let mut llm_response = LLMResponse::new(content, response.model.unwrap_or_default())
            .with_usage(prompt_tokens, completion_tokens);

        if let Some(reason) = finish_reason {
            llm_response = llm_response.with_finish_reason(reason);
        }

        if !tool_calls.is_empty() {
            llm_response = llm_response.with_tool_calls(tool_calls);
        }

        if let Some(id) = response.id {
            llm_response = llm_response.with_metadata("id", serde_json::json!(id));
        }

        llm_response
    }

    fn map_http_error(status: reqwest::StatusCode, body: &str) -> LlmError {
        match status.as_u16() {
            401 => LlmError::AuthError(format!("Cohere authentication failed: {}", body)),
            429 => LlmError::RateLimited(format!("Cohere rate limited: {}", body)),
            400 => LlmError::InvalidRequest(format!("Cohere bad request: {}", body)),
            404 => LlmError::ModelNotFound(format!("Cohere resource not found: {}", body)),
            500..=599 => {
                LlmError::ProviderError(format!("Cohere server error ({}): {}", status, body))
            }
            _ => LlmError::ApiError(format!("Cohere API error ({}): {}", status, body)),
        }
    }
}

// ============================================================================
// LLMProvider Implementation
// ============================================================================

#[async_trait]
impl LLMProvider for CohereProvider {
    fn name(&self) -> &str {
        COHERE_PROVIDER_NAME
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_context_length(&self) -> usize {
        self.max_context
    }

    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        let messages = vec![ChatMessage::user(prompt)];
        self.chat(&messages, None).await
    }

    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<LLMResponse> {
        let messages = vec![ChatMessage::user(prompt)];
        self.chat(&messages, Some(options)).await
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let request = self.build_chat_request(messages, &[], None, options, false);
        let response = self.send_chat_request(&request).await?;
        Ok(Self::parse_chat_response(response))
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let request =
            self.build_chat_request(messages, tools, tool_choice.as_ref(), options, false);
        let response = self.send_chat_request(&request).await?;
        Ok(Self::parse_chat_response(response))
    }

    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        let messages = vec![ChatMessage::user(prompt)];
        let request = self.build_chat_request(&messages, &[], None, None, true);
        let response = self.send_streaming_request(&request).await?;

        let mut line_buffer = String::new();

        let stream = response
            .bytes_stream()
            .chain(futures::stream::once(async {
                Ok::<_, reqwest::Error>("\n".into())
            }))
            .map(move |chunk| {
                let chunk = chunk.map_err(|e| LlmError::NetworkError(e.to_string()))?;
                let text = String::from_utf8_lossy(&chunk);
                line_buffer.push_str(&text);

                let mut result = String::new();

                for data in drain_sse_data_lines(&mut line_buffer) {
                    if let Ok(CohereStreamEvent::ContentDelta { delta, .. }) =
                        serde_json::from_str::<CohereStreamEvent>(&data)
                    {
                        if let Some(msg) = delta.message {
                            if let Some(content) = msg.content {
                                if let Some(text) = content.text {
                                    result.push_str(&text);
                                }
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

    async fn chat_with_tools_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        let request = self.build_chat_request(messages, tools, tool_choice.as_ref(), options, true);
        let response = self.send_streaming_request(&request).await?;

        let mut line_buffer = String::new();
        let mut finished_emitted = false;

        let stream = response
            .bytes_stream()
            .chain(futures::stream::once(async {
                Ok::<_, reqwest::Error>("\n".into())
            }))
            .map(move |chunk| -> Result<Vec<StreamChunk>> {
                let chunk = chunk.map_err(|e| LlmError::NetworkError(e.to_string()))?;
                let text = String::from_utf8_lossy(&chunk);
                line_buffer.push_str(&text);

                let mut chunks: Vec<StreamChunk> = Vec::new();

                for data in drain_sse_data_lines(&mut line_buffer) {
                    if let Ok(event) = serde_json::from_str::<CohereStreamEvent>(&data) {
                        match event {
                            CohereStreamEvent::ContentDelta { delta, .. } => {
                                if let Some(msg) = delta.message {
                                    if let Some(content) = msg.content {
                                        if let Some(text) = content.text {
                                            chunks.push(StreamChunk::Content(text));
                                        }
                                    }
                                }
                            }
                            CohereStreamEvent::ToolCallStart { index, delta, .. } => {
                                if let Some(msg) = delta.message {
                                    if let Some(tool_calls) = msg.tool_calls {
                                        if let Some(tc) = tool_calls.into_iter().next() {
                                            chunks.push(StreamChunk::ToolCallDelta {
                                                index: index.unwrap_or(0),
                                                id: Some(tc.id),
                                                function_name: Some(tc.function.name),
                                                function_arguments: None,
                                                thought_signature: None,
                                            });
                                        }
                                    }
                                }
                            }
                            CohereStreamEvent::ToolCallDelta { index, delta, .. } => {
                                if let Some(msg) = delta.message {
                                    if let Some(tool_calls) = msg.tool_calls {
                                        if let Some(tc) = tool_calls.into_iter().next() {
                                            chunks.push(StreamChunk::ToolCallDelta {
                                                index: index.unwrap_or(0),
                                                id: None,
                                                function_name: None,
                                                function_arguments: Some(tc.function.arguments),
                                                thought_signature: None,
                                            });
                                        }
                                    }
                                }
                            }
                            CohereStreamEvent::MessageEnd { delta, .. } if !finished_emitted => {
                                finished_emitted = true;
                                let reason = delta
                                    .as_ref()
                                    .and_then(|d| d.finish_reason.as_ref())
                                    .map(|r| match r.as_str() {
                                        "COMPLETE" => "stop".to_string(),
                                        "MAX_TOKENS" => "length".to_string(),
                                        "TOOL_CALL" => "tool_calls".to_string(),
                                        other => other.to_lowercase(),
                                    })
                                    .unwrap_or_else(|| "stop".to_string());

                                let usage =
                                    delta.as_ref().and_then(|d| d.usage.as_ref()).map(|u| {
                                        let input = u
                                            .tokens
                                            .as_ref()
                                            .map(|t| t.input_tokens as usize)
                                            .unwrap_or(0);
                                        let output = u
                                            .tokens
                                            .as_ref()
                                            .map(|t| t.output_tokens as usize)
                                            .unwrap_or(0);
                                        StreamUsage::new(input, output)
                                    });

                                chunks.push(StreamChunk::Finished {
                                    reason,
                                    ttft_ms: None,
                                    usage,
                                });
                            }
                            _ => {}
                        }
                    }
                }

                Ok(chunks)
            })
            .flat_map(|result| match result {
                Ok(chunks) => {
                    futures::stream::iter(chunks.into_iter().map(Ok).collect::<Vec<_>>()).boxed()
                }
                Err(e) => futures::stream::once(async move { Err(e) }).boxed(),
            });

        Ok(stream.boxed())
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_tool_streaming(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    fn supports_function_calling(&self) -> bool {
        true
    }
}

// ============================================================================
// EmbeddingProvider Implementation
// ============================================================================

#[async_trait]
impl EmbeddingProvider for CohereProvider {
    fn name(&self) -> &str {
        COHERE_PROVIDER_NAME
    }

    #[allow(clippy::misnamed_getters)]
    fn model(&self) -> &str {
        &self.embed_model
    }

    fn dimension(&self) -> usize {
        self.embed_dimensions
    }

    fn max_tokens(&self) -> usize {
        COHERE_EMBED_MAX_TOKENS
    }

    fn max_batch_size(&self) -> usize {
        std::env::var("EDGEQUAKE_EMBEDDING_BATCH_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(COHERE_EMBED_MAX_BATCH_SIZE)
            .min(COHERE_EMBED_MAX_BATCH_SIZE)
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        if texts.len() <= COHERE_EMBED_MAX_BATCH_SIZE {
            return self.embed_batch_http(texts).await;
        }

        // Split oversized batches
        debug!(
            total = texts.len(),
            batch_size = COHERE_EMBED_MAX_BATCH_SIZE,
            "Cohere embed: splitting oversized batch"
        );
        let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(COHERE_EMBED_MAX_BATCH_SIZE) {
            let batch = self.embed_batch_http(chunk).await?;
            all_embeddings.extend(batch);
        }
        Ok(all_embeddings)
    }
}

impl CohereProvider {
    async fn embed_batch_http(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embed", self.base_url.trim_end_matches('/'));

        let request_body = CohereEmbedRequest {
            model: &self.embed_model,
            texts,
            input_type: "search_document",
            embedding_types: &["float"],
        };

        debug!(
            model = self.embed_model,
            count = texts.len(),
            "Cohere embed request"
        );

        let response = self
            .apply_attribution_headers(
                self.http_client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .header("Content-Type", "application/json"),
            )
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                LlmError::NetworkError(format!("Cohere embeddings request failed: {}", e))
            })?;

        let status = response.status();
        let body = response.text().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to read Cohere embeddings response: {}", e))
        })?;

        if !status.is_success() {
            return Err(Self::map_http_error(status, &body));
        }

        let embed_response: CohereEmbedResponse = serde_json::from_str(&body).map_err(|e| {
            LlmError::ProviderError(format!(
                "Failed to parse Cohere embeddings response: {} | body: {}",
                e,
                &body[..body.len().min(500)]
            ))
        })?;

        let embeddings = embed_response.embeddings.float.unwrap_or_default();

        if embeddings.len() != texts.len() {
            return Err(LlmError::ProviderError(format!(
                "Cohere returned {} embeddings for {} inputs",
                embeddings.len(),
                texts.len()
            )));
        }

        Ok(embeddings)
    }
}

// ============================================================================
// SSE Helper
// ============================================================================

fn drain_sse_data_lines(line_buffer: &mut String) -> Vec<String> {
    let mut data_lines = Vec::new();

    while let Some(newline_idx) = line_buffer.find('\n') {
        let line = line_buffer[..newline_idx].trim().to_string();
        line_buffer.drain(..=newline_idx);

        if line.is_empty() || line.starts_with(':') || line.starts_with("event:") {
            continue;
        }

        if let Some(data) = line.strip_prefix("data: ") {
            if data != "[DONE]" {
                data_lines.push(data.to_string());
            }
        }
    }

    data_lines
}

// ============================================================================
// Request/Response Types - Chat
// ============================================================================

#[derive(Debug, Serialize)]
struct CohereChatRequest {
    model: String,
    messages: Vec<CohereMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<CohereTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<CohereToolChoice>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<CohereResponseFormat>,
}

#[derive(Debug, Serialize)]
struct CohereResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CohereMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<CohereContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<CohereToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum CohereContent {
    Text(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CohereToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: CohereFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CohereFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct CohereTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: CohereToolFunction,
}

#[derive(Debug, Serialize)]
struct CohereToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum CohereToolChoice {
    String(String),
    Object(CohereToolChoiceObject),
}

#[derive(Debug, Serialize)]
struct CohereToolChoiceObject {
    #[serde(rename = "type")]
    choice_type: String,
    function: CohereToolChoiceFunction,
}

#[derive(Debug, Serialize)]
struct CohereToolChoiceFunction {
    name: String,
}

// ============================================================================
// Response Types - Chat
// ============================================================================

#[derive(Debug, Deserialize)]
struct CohereChatResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    message: CohereResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    usage: Option<CohereUsage>,
}

#[derive(Debug, Deserialize)]
struct CohereResponseMessage {
    #[allow(dead_code)]
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<Vec<CohereContentBlock>>,
    #[serde(default)]
    tool_calls: Option<Vec<CohereToolCall>>,
}

#[derive(Debug, Deserialize)]
struct CohereContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CohereUsage {
    #[serde(default)]
    billed_units: Option<CohereBilledUnits>,
    #[serde(default)]
    tokens: Option<CohereTokens>,
}

#[derive(Debug, Deserialize)]
struct CohereBilledUnits {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct CohereTokens {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
}

// ============================================================================
// Streaming Event Types
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum CohereStreamEvent {
    #[serde(rename = "message-start")]
    MessageStart {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        delta: Option<CohereStreamDelta>,
    },
    #[serde(rename = "content-start")]
    ContentStart {
        #[serde(default)]
        index: Option<usize>,
    },
    #[serde(rename = "content-delta")]
    ContentDelta {
        #[serde(default)]
        index: Option<usize>,
        delta: CohereStreamContentDelta,
    },
    #[serde(rename = "content-end")]
    ContentEnd {
        #[serde(default)]
        index: Option<usize>,
    },
    #[serde(rename = "tool-call-start")]
    ToolCallStart {
        #[serde(default)]
        index: Option<usize>,
        delta: CohereStreamToolDelta,
    },
    #[serde(rename = "tool-call-delta")]
    ToolCallDelta {
        #[serde(default)]
        index: Option<usize>,
        delta: CohereStreamToolDelta,
    },
    #[serde(rename = "tool-call-end")]
    ToolCallEnd {
        #[serde(default)]
        index: Option<usize>,
    },
    #[serde(rename = "message-end")]
    MessageEnd {
        #[serde(default)]
        delta: Option<CohereMessageEndDelta>,
    },
}

#[derive(Debug, Deserialize)]
struct CohereStreamDelta {
    #[serde(default)]
    #[allow(dead_code)]
    message: Option<CohereStreamStartMessage>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CohereStreamStartMessage {
    #[serde(default)]
    role: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CohereStreamContentDelta {
    #[serde(default)]
    message: Option<CohereStreamContentMessage>,
}

#[derive(Debug, Deserialize)]
struct CohereStreamContentMessage {
    #[serde(default)]
    content: Option<CohereStreamContentText>,
}

#[derive(Debug, Deserialize)]
struct CohereStreamContentText {
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CohereStreamToolDelta {
    #[serde(default)]
    message: Option<CohereStreamToolMessage>,
}

#[derive(Debug, Deserialize)]
struct CohereStreamToolMessage {
    #[serde(default)]
    tool_calls: Option<Vec<CohereStreamToolCallItem>>,
}

#[derive(Debug, Deserialize)]
struct CohereStreamToolCallItem {
    #[serde(default)]
    id: String,
    function: CohereStreamToolCallFunction,
}

#[derive(Debug, Deserialize)]
struct CohereStreamToolCallFunction {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct CohereMessageEndDelta {
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    usage: Option<CohereUsage>,
}

// ============================================================================
// Request/Response Types - Embed
// ============================================================================

#[derive(Debug, Serialize)]
struct CohereEmbedRequest<'a> {
    model: &'a str,
    texts: &'a [String],
    input_type: &'a str,
    embedding_types: &'a [&'a str],
}

#[derive(Debug, Deserialize)]
struct CohereEmbedResponse {
    #[allow(dead_code)]
    #[serde(default)]
    id: Option<String>,
    embeddings: CohereEmbeddings,
}

#[derive(Debug, Deserialize)]
struct CohereEmbeddings {
    #[serde(default)]
    float: Option<Vec<Vec<f32>>>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Constructor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_creates_provider_with_defaults() {
        let provider = CohereProvider::new("test-key");
        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.base_url, COHERE_BASE_URL);
        assert_eq!(provider.model, COHERE_DEFAULT_MODEL);
        assert_eq!(provider.embed_model, COHERE_DEFAULT_EMBED_MODEL);
        assert_eq!(provider.embed_dimensions, COHERE_DEFAULT_EMBED_DIMENSIONS);
        assert_eq!(provider.max_context, COHERE_DEFAULT_MAX_CONTEXT);
    }

    #[test]
    fn test_with_model_builder() {
        let provider = CohereProvider::new("key").with_model("command-r-plus");
        assert_eq!(provider.model, "command-r-plus");
    }

    #[test]
    fn test_with_embed_model_builder() {
        let provider = CohereProvider::new("key").with_embed_model("embed-english-v3.0");
        assert_eq!(provider.embed_model, "embed-english-v3.0");
    }

    #[test]
    fn test_with_base_url_builder() {
        let provider = CohereProvider::new("key").with_base_url("https://custom.api/v2");
        assert_eq!(provider.base_url, "https://custom.api/v2");
    }

    #[test]
    fn test_from_env_missing_key() {
        std::env::remove_var("COHERE_API_KEY");
        let result = CohereProvider::from_env();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("COHERE_API_KEY"));
    }

    #[test]
    fn test_from_env_empty_key() {
        std::env::set_var("COHERE_API_KEY", "");
        let result = CohereProvider::from_env();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
        std::env::remove_var("COHERE_API_KEY");
    }

    // -----------------------------------------------------------------------
    // Trait surface tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_llm_provider_surface() {
        let provider = CohereProvider::new("key");
        assert_eq!(LLMProvider::name(&provider), "cohere");
        assert_eq!(LLMProvider::model(&provider), COHERE_DEFAULT_MODEL);
        assert_eq!(provider.max_context_length(), COHERE_DEFAULT_MAX_CONTEXT);
        assert!(provider.supports_streaming());
        assert!(provider.supports_function_calling());
        assert!(provider.supports_json_mode());
        assert!(provider.supports_tool_streaming());
    }

    #[test]
    fn test_embedding_provider_surface() {
        let provider = CohereProvider::new("key");
        assert_eq!(EmbeddingProvider::name(&provider), "cohere");
        assert_eq!(
            EmbeddingProvider::model(&provider),
            COHERE_DEFAULT_EMBED_MODEL
        );
        assert_eq!(
            EmbeddingProvider::dimension(&provider),
            COHERE_DEFAULT_EMBED_DIMENSIONS
        );
        assert_eq!(
            EmbeddingProvider::max_tokens(&provider),
            COHERE_EMBED_MAX_TOKENS
        );
        assert_eq!(
            EmbeddingProvider::max_batch_size(&provider),
            COHERE_EMBED_MAX_BATCH_SIZE
        );
    }

    // -----------------------------------------------------------------------
    // Request serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_chat_request_serialization_minimal() {
        let request = CohereChatRequest {
            model: "command-a-plus-05-2026".to_string(),
            messages: vec![CohereMessage {
                role: "user".to_string(),
                content: Some(CohereContent::Text("Hello".to_string())),
                tool_calls: None,
                tool_call_id: None,
            }],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            tools: None,
            tool_choice: None,
            stream: false,
            response_format: None,
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["model"], "command-a-plus-05-2026");
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["messages"][0]["content"], "Hello");
        assert_eq!(json["stream"], false);
        assert!(json.get("max_tokens").is_none());
        assert!(json.get("tools").is_none());
    }

    #[test]
    fn test_chat_request_serialization_with_options() {
        let request = CohereChatRequest {
            model: "command-a-plus-05-2026".to_string(),
            messages: vec![CohereMessage {
                role: "user".to_string(),
                content: Some(CohereContent::Text("Hello".to_string())),
                tool_calls: None,
                tool_call_id: None,
            }],
            max_tokens: Some(4096),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stop: Some(vec!["END".to_string()]),
            frequency_penalty: None,
            presence_penalty: None,
            tools: None,
            tool_choice: None,
            stream: true,
            response_format: None,
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["max_tokens"], 4096);
        let temp = json["temperature"].as_f64().unwrap();
        assert!(
            (temp - 0.7).abs() < 0.001,
            "temperature should be ~0.7, got {temp}"
        );
        let top_p = json["top_p"].as_f64().unwrap();
        assert!(
            (top_p - 0.9).abs() < 0.001,
            "top_p should be ~0.9, got {top_p}"
        );
        assert_eq!(json["stop"][0], "END");
        assert_eq!(json["stream"], true);
    }

    #[test]
    fn test_chat_request_with_tools() {
        let request = CohereChatRequest {
            model: "command-a-plus-05-2026".to_string(),
            messages: vec![],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            tools: Some(vec![CohereTool {
                tool_type: "function".to_string(),
                function: CohereToolFunction {
                    name: "get_weather".to_string(),
                    description: "Get the weather".to_string(),
                    parameters: serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
                },
            }]),
            tool_choice: None,
            stream: false,
            response_format: None,
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["tools"][0]["type"], "function");
        assert_eq!(json["tools"][0]["function"]["name"], "get_weather");
    }

    // -----------------------------------------------------------------------
    // Response deserialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_chat_response_deserialization() {
        let raw = r#"{
            "id": "resp-123",
            "model": "command-a-plus-05-2026",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello! How can I help?"}]
            },
            "finish_reason": "COMPLETE",
            "usage": {
                "billed_units": {"input_tokens": 10, "output_tokens": 20},
                "tokens": {"input_tokens": 10, "output_tokens": 20}
            }
        }"#;
        let resp: CohereChatResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(resp.id, Some("resp-123".to_string()));
        assert_eq!(resp.model, Some("command-a-plus-05-2026".to_string()));
        assert_eq!(resp.finish_reason, Some("COMPLETE".to_string()));
        let content = resp.message.content.unwrap();
        assert_eq!(content[0].block_type, "text");
        assert_eq!(content[0].text, Some("Hello! How can I help?".to_string()));
    }

    #[test]
    fn test_chat_response_with_tool_calls() {
        let raw = r#"{
            "id": "resp-456",
            "model": "command-a-plus-05-2026",
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\": \"Paris\"}"
                        }
                    }
                ]
            },
            "finish_reason": "TOOL_CALL",
            "usage": {
                "tokens": {"input_tokens": 15, "output_tokens": 25}
            }
        }"#;
        let resp: CohereChatResponse = serde_json::from_str(raw).unwrap();
        let tool_calls = resp.message.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_abc");
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert_eq!(tool_calls[0].function.arguments, "{\"city\": \"Paris\"}");
    }

    #[test]
    fn test_parse_chat_response_to_llm_response() {
        let raw = r#"{
            "id": "resp-789",
            "model": "command-a-plus-05-2026",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Answer"}]
            },
            "finish_reason": "COMPLETE",
            "usage": {
                "tokens": {"input_tokens": 50, "output_tokens": 30}
            }
        }"#;
        let resp: CohereChatResponse = serde_json::from_str(raw).unwrap();
        let llm_resp = CohereProvider::parse_chat_response(resp);
        assert_eq!(llm_resp.content, "Answer");
        assert_eq!(llm_resp.model, "command-a-plus-05-2026");
        assert_eq!(llm_resp.prompt_tokens, 50);
        assert_eq!(llm_resp.completion_tokens, 30);
        assert_eq!(llm_resp.total_tokens, 80);
        assert_eq!(llm_resp.finish_reason, Some("stop".to_string()));
        assert!(!llm_resp.has_tool_calls());
    }

    #[test]
    fn test_parse_chat_response_with_tool_calls() {
        let raw = r#"{
            "id": "resp-tool",
            "model": "command-a-plus-05-2026",
            "message": {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{\"q\":\"rust\"}"}
                    }
                ]
            },
            "finish_reason": "TOOL_CALL",
            "usage": {"tokens": {"input_tokens": 10, "output_tokens": 5}}
        }"#;
        let resp: CohereChatResponse = serde_json::from_str(raw).unwrap();
        let llm_resp = CohereProvider::parse_chat_response(resp);
        assert!(llm_resp.has_tool_calls());
        assert_eq!(llm_resp.tool_calls.len(), 1);
        assert_eq!(llm_resp.tool_calls[0].function.name, "search");
        assert_eq!(llm_resp.finish_reason, Some("tool_calls".to_string()));
    }

    // -----------------------------------------------------------------------
    // Embed request/response tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_embed_request_serialization() {
        let texts = vec!["hello world".to_string(), "foo bar".to_string()];
        let req = CohereEmbedRequest {
            model: "embed-v4.0",
            texts: &texts,
            input_type: "search_document",
            embedding_types: &["float"],
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "embed-v4.0");
        assert_eq!(json["texts"][0], "hello world");
        assert_eq!(json["texts"][1], "foo bar");
        assert_eq!(json["input_type"], "search_document");
        assert_eq!(json["embedding_types"][0], "float");
    }

    #[test]
    fn test_embed_response_deserialization() {
        let raw = r#"{
            "id": "emb-1",
            "embeddings": {
                "float": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            },
            "meta": {"api_version": {"version": "2"}}
        }"#;
        let resp: CohereEmbedResponse = serde_json::from_str(raw).unwrap();
        let floats = resp.embeddings.float.unwrap();
        assert_eq!(floats.len(), 2);
        assert_eq!(floats[0], vec![0.1, 0.2, 0.3]);
        assert_eq!(floats[1], vec![0.4, 0.5, 0.6]);
    }

    // -----------------------------------------------------------------------
    // SSE parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_drain_sse_data_lines_basic() {
        let mut buf = "data: {\"type\":\"content-delta\"}\n\n".to_string();
        let lines = drain_sse_data_lines(&mut buf);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "{\"type\":\"content-delta\"}");
        assert!(buf.is_empty());
    }

    #[test]
    fn test_drain_sse_data_lines_multiple() {
        let mut buf = "data: {\"a\":1}\ndata: {\"b\":2}\n".to_string();
        let lines = drain_sse_data_lines(&mut buf);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "{\"a\":1}");
        assert_eq!(lines[1], "{\"b\":2}");
    }

    #[test]
    fn test_drain_sse_data_lines_skips_done() {
        let mut buf = "data: [DONE]\n".to_string();
        let lines = drain_sse_data_lines(&mut buf);
        assert!(lines.is_empty());
    }

    #[test]
    fn test_drain_sse_data_lines_skips_comments() {
        let mut buf = ": comment\ndata: {\"x\":1}\n".to_string();
        let lines = drain_sse_data_lines(&mut buf);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "{\"x\":1}");
    }

    #[test]
    fn test_drain_sse_data_lines_skips_event_lines() {
        let mut buf = "event: content-delta\ndata: {\"type\":\"content-delta\"}\n".to_string();
        let lines = drain_sse_data_lines(&mut buf);
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn test_drain_sse_data_lines_partial_buffer() {
        let mut buf = "data: {\"incom".to_string();
        let lines = drain_sse_data_lines(&mut buf);
        assert!(lines.is_empty());
        assert_eq!(buf, "data: {\"incom");
    }

    // -----------------------------------------------------------------------
    // Stream event deserialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_stream_event_content_delta() {
        let raw = r#"{"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"Hello"}}}}"#;
        let event: CohereStreamEvent = serde_json::from_str(raw).unwrap();
        if let CohereStreamEvent::ContentDelta { delta, .. } = event {
            let text = delta.message.unwrap().content.unwrap().text.unwrap();
            assert_eq!(text, "Hello");
        } else {
            panic!("Expected ContentDelta");
        }
    }

    #[test]
    fn test_stream_event_message_end() {
        let raw = r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":10,"output_tokens":20}}}}"#;
        let event: CohereStreamEvent = serde_json::from_str(raw).unwrap();
        if let CohereStreamEvent::MessageEnd { delta } = event {
            let d = delta.unwrap();
            assert_eq!(d.finish_reason, Some("COMPLETE".to_string()));
            let usage = d.usage.unwrap();
            assert_eq!(usage.tokens.unwrap().input_tokens, 10);
        } else {
            panic!("Expected MessageEnd");
        }
    }

    #[test]
    fn test_stream_event_tool_call_start() {
        let raw = r#"{"type":"tool-call-start","index":0,"delta":{"message":{"tool_calls":[{"id":"tc_1","function":{"name":"search","arguments":""}}]}}}"#;
        let event: CohereStreamEvent = serde_json::from_str(raw).unwrap();
        if let CohereStreamEvent::ToolCallStart { index, delta } = event {
            assert_eq!(index, Some(0));
            let tc = delta.message.unwrap().tool_calls.unwrap();
            assert_eq!(tc[0].id, "tc_1");
            assert_eq!(tc[0].function.name, "search");
        } else {
            panic!("Expected ToolCallStart");
        }
    }

    // -----------------------------------------------------------------------
    // Message conversion tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_convert_messages_basic() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
        ];
        let converted = CohereProvider::convert_messages(&messages, None);
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].role, "system");
        assert_eq!(converted[1].role, "user");
    }

    #[test]
    fn test_convert_messages_with_system_prompt_option() {
        let messages = vec![ChatMessage::user("Hello")];
        let options = CompletionOptions {
            system_prompt: Some("Be concise".to_string()),
            ..Default::default()
        };
        let converted = CohereProvider::convert_messages(&messages, Some(&options));
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].role, "system");
        if let Some(CohereContent::Text(text)) = &converted[0].content {
            assert_eq!(text, "Be concise");
        } else {
            panic!("Expected text content");
        }
    }

    #[test]
    fn test_convert_messages_tool_result() {
        let msg = ChatMessage::tool_result("call_123", "result data");
        let converted = CohereProvider::convert_messages(&[msg], None);
        assert_eq!(converted[0].role, "tool");
        assert_eq!(converted[0].tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_convert_tools() {
        let tools = vec![ToolDefinition::function(
            "search",
            "Search the web",
            serde_json::json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        )];
        let converted = CohereProvider::convert_tools(&tools);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].tool_type, "function");
        assert_eq!(converted[0].function.name, "search");
        assert_eq!(converted[0].function.description, "Search the web");
    }

    // -----------------------------------------------------------------------
    // Error mapping tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_map_http_error_401() {
        let err = CohereProvider::map_http_error(reqwest::StatusCode::UNAUTHORIZED, "invalid key");
        assert!(matches!(err, LlmError::AuthError(_)));
    }

    #[test]
    fn test_map_http_error_429() {
        let err =
            CohereProvider::map_http_error(reqwest::StatusCode::TOO_MANY_REQUESTS, "slow down");
        assert!(matches!(err, LlmError::RateLimited(_)));
    }

    #[test]
    fn test_map_http_error_400() {
        let err = CohereProvider::map_http_error(reqwest::StatusCode::BAD_REQUEST, "bad params");
        assert!(matches!(err, LlmError::InvalidRequest(_)));
    }

    #[test]
    fn test_map_http_error_500() {
        let err = CohereProvider::map_http_error(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            "server crash",
        );
        assert!(matches!(err, LlmError::ProviderError(_)));
    }

    #[test]
    fn test_map_http_error_404() {
        let err = CohereProvider::map_http_error(reqwest::StatusCode::NOT_FOUND, "not found");
        assert!(matches!(err, LlmError::ModelNotFound(_)));
    }

    // -----------------------------------------------------------------------
    // Finish reason mapping
    // -----------------------------------------------------------------------

    #[test]
    fn test_finish_reason_mapping() {
        let make_response = |reason: &str| -> CohereChatResponse {
            serde_json::from_value(serde_json::json!({
                "id": "x",
                "model": "m",
                "message": {"role": "assistant", "content": [{"type": "text", "text": ""}]},
                "finish_reason": reason,
                "usage": {"tokens": {"input_tokens": 0, "output_tokens": 0}}
            }))
            .unwrap()
        };

        let r = CohereProvider::parse_chat_response(make_response("COMPLETE"));
        assert_eq!(r.finish_reason, Some("stop".to_string()));

        let r = CohereProvider::parse_chat_response(make_response("MAX_TOKENS"));
        assert_eq!(r.finish_reason, Some("length".to_string()));

        let r = CohereProvider::parse_chat_response(make_response("TOOL_CALL"));
        assert_eq!(r.finish_reason, Some("tool_calls".to_string()));
    }
}
