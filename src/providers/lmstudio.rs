//! LM Studio provider implementation.
//!
//! This module provides integration with LM Studio's local OpenAI-compatible API.
//! LM Studio runs local models and exposes them via an OpenAI-compatible HTTP API.
//!
//! # Default Configuration
//!
//! - Base URL: `http://localhost:1234`
//! - Default model: `gemma2-9b-it` (chat), `nomic-embed-text-v1.5` (embeddings, 768 dimensions)
//!
//! # Environment Variables
//!
//! - `LMSTUDIO_HOST`: LM Studio server URL (default: http://localhost:1234)
//! - `LMSTUDIO_MODEL`: Default chat model
//! - `LMSTUDIO_EMBEDDING_MODEL`: Default embedding model
//! - `LMSTUDIO_EMBEDDING_DIM`: Embedding dimension (default: 768)
//!
//! # Example
//!
//! ```rust,ignore
//! use edgequake_llm::LMStudioProvider;
//!
//! // Connect to local LM Studio with defaults
//! let provider = LMStudioProvider::from_env()?;
//!
//! // Or specify custom settings
//! let provider = LMStudioProvider::builder()
//!     .host("http://localhost:1234")
//!     .model("mistral-7b-instruct")
//!     .embedding_model("nomic-embed-text-v1.5")
//!     .build()?;
//! ```

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

use crate::error::{LlmError, Result};
use crate::traits::{
    ChatMessage, ChatRole, CompletionOptions, EmbeddingProvider, ImageData, LLMProvider, LLMResponse,
    StreamChunk, ToolChoice, ToolDefinition,
};

/// Default LM Studio host URL
const DEFAULT_LMSTUDIO_HOST: &str = "http://localhost:1234";

/// Default LM Studio chat model
const DEFAULT_LMSTUDIO_MODEL: &str = "gemma2-9b-it";

/// Default LM Studio embedding model
const DEFAULT_LMSTUDIO_EMBEDDING_MODEL: &str = "nomic-embed-text-v1.5";

/// Default embedding dimension for nomic-embed-text-v1.5
const DEFAULT_LMSTUDIO_EMBEDDING_DIM: usize = 768;

/// LM Studio LLM and embedding provider.
///
/// Provides integration with locally running LM Studio instance.
/// Uses OpenAI-compatible API format.
#[derive(Debug, Clone)]
pub struct LMStudioProvider {
    client: Client,
    host: String,
    model: String,
    embedding_model: String,
    max_context_length: usize,
    embedding_dimension: usize,
    auto_load_models: bool,
}

/// Builder for LMStudioProvider
#[derive(Debug, Clone)]
pub struct LMStudioProviderBuilder {
    host: String,
    model: String,
    embedding_model: String,
    max_context_length: usize,
    embedding_dimension: usize,
    auto_load_models: bool,
}

impl Default for LMStudioProviderBuilder {
    fn default() -> Self {
        Self {
            host: DEFAULT_LMSTUDIO_HOST.to_string(),
            model: DEFAULT_LMSTUDIO_MODEL.to_string(),
            embedding_model: DEFAULT_LMSTUDIO_EMBEDDING_MODEL.to_string(),
            max_context_length: 131072, // OODA-99: Increased to 128K (131072)
            embedding_dimension: DEFAULT_LMSTUDIO_EMBEDDING_DIM,
            auto_load_models: true,
        }
    }
}

impl LMStudioProviderBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the LM Studio host URL
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Set the chat model
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the embedding model
    pub fn embedding_model(mut self, model: impl Into<String>) -> Self {
        self.embedding_model = model.into();
        self
    }

    /// Set the maximum context length
    pub fn max_context_length(mut self, length: usize) -> Self {
        self.max_context_length = length;
        self
    }

    /// Set the embedding dimension
    pub fn embedding_dimension(mut self, dimension: usize) -> Self {
        self.embedding_dimension = dimension;
        self
    }

    /// Enable or disable automatic model loading (default: true)
    pub fn auto_load_models(mut self, enabled: bool) -> Self {
        self.auto_load_models = enabled;
        self
    }

    /// Build the LMStudioProvider
    pub fn build(self) -> Result<LMStudioProvider> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // Longer timeout for local models
            .no_proxy() // CRITICAL: Disable all proxies for localhost connections
            .build()
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        Ok(LMStudioProvider {
            client,
            host: self.host,
            model: self.model,
            embedding_model: self.embedding_model,
            max_context_length: self.max_context_length,
            embedding_dimension: self.embedding_dimension,
            auto_load_models: self.auto_load_models,
        })
    }
}

impl LMStudioProvider {
    /// Create a new LMStudioProvider from environment variables.
    ///
    /// Environment variables:
    /// - `LMSTUDIO_HOST`: Server URL (default: http://localhost:1234)
    /// - `LMSTUDIO_MODEL`: Chat model (default: gemma2-9b-it)
    /// - `LMSTUDIO_EMBEDDING_MODEL`: Embedding model (default: nomic-embed-text-v1.5)
    /// - `LMSTUDIO_EMBEDDING_DIM`: Embedding dimension (default: 768)
    /// - `LMSTUDIO_CONTEXT_LENGTH`: Max context length (default: 32768)
    ///
    /// # OODA-99: Context Length Configuration
    ///
    /// The context length determines how much text can be sent to LMStudio.
    /// If you get "400 Bad Request" errors, increase this value to match
    /// your LMStudio model's configured context window.
    ///
    /// Example: `export LMSTUDIO_CONTEXT_LENGTH=65536`
    pub fn from_env() -> Result<Self> {
        let host =
            std::env::var("LMSTUDIO_HOST").unwrap_or_else(|_| DEFAULT_LMSTUDIO_HOST.to_string());

        let model =
            std::env::var("LMSTUDIO_MODEL").unwrap_or_else(|_| DEFAULT_LMSTUDIO_MODEL.to_string());

        let embedding_model = std::env::var("LMSTUDIO_EMBEDDING_MODEL")
            .unwrap_or_else(|_| DEFAULT_LMSTUDIO_EMBEDDING_MODEL.to_string());

        let embedding_dimension = std::env::var("LMSTUDIO_EMBEDDING_DIM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_LMSTUDIO_EMBEDDING_DIM);

        // OODA-99: Allow context length override
        // Default to 128K (131072) for modern models with large contexts
        // Falls back to 8192 only if parsing fails
        let max_context_length = std::env::var("LMSTUDIO_CONTEXT_LENGTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(131072);

        LMStudioProviderBuilder::new()
            .host(host)
            .model(model)
            .embedding_model(embedding_model)
            .embedding_dimension(embedding_dimension)
            .max_context_length(max_context_length)
            .build()
    }

    /// Create a new builder for LMStudioProvider
    pub fn builder() -> LMStudioProviderBuilder {
        LMStudioProviderBuilder::new()
    }

    /// Create with default settings (localhost:1234)
    pub fn default_local() -> Result<Self> {
        LMStudioProviderBuilder::new().build()
    }

    /// Get the API base URL with /v1 suffix
    fn api_base(&self) -> String {
        if self.host.ends_with("/v1") {
            self.host.clone()
        } else {
            format!("{}/v1", self.host)
        }
    }

    /// Get the REST API base URL for native LMStudio features
    /// OODA-30: Used for reasoning support via /api/v1/chat
    fn rest_api_base(&self) -> String {
        let base = self.host.trim_end_matches("/v1");
        format!("{}/api/v1", base)
    }

    /// Check if LM Studio is running and accessible
    pub async fn health_check(&self) -> Result<()> {
        let url = format!("{}/models", self.api_base());

        let response = self
            .client
            .get(&url)
            .timeout(std::time::Duration::from_secs(5)) // Quick check
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    LlmError::NetworkError(format!(
                        "LM Studio not responding at {}. Is LM Studio running?",
                        self.host
                    ))
                } else if e.is_connect() {
                    LlmError::NetworkError(format!(
                        "Cannot connect to LM Studio at {}. Please start LM Studio and load a model.",
                        self.host
                    ))
                } else {
                    LlmError::NetworkError(format!("LM Studio health check failed: {}", e))
                }
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError(format!(
                "LM Studio returned error status {}: {}. Please check that a model is loaded.",
                status, error_text
            )));
        }

        // Verify models are available
        let models_text = response.text().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to read models response: {}", e))
        })?;

        debug!("LM Studio models response: {}", models_text);

        // Basic check that response contains "data" field
        if !models_text.contains("\"data\"") && !models_text.contains("\"object\":") {
            return Err(LlmError::ApiError(
                "LM Studio /v1/models returned unexpected format. Please ensure LM Studio is properly initialized.".to_string()
            ));
        }

        Ok(())
    }

    /// Check if an error indicates the model is not loaded
    fn is_model_not_loaded_error(error_text: &str) -> bool {
        error_text.contains("not a valid model ID")
            || error_text.contains("model not found")
            || error_text.contains("model not loaded")
            || error_text.contains("No model loaded")
    }

    /// Attempt to load a model using the lms CLI
    async fn load_model_via_cli(&self, model_id: &str) -> Result<()> {
        eprintln!(
            "⏳ Model '{}' not loaded. Loading automatically via lms CLI...",
            model_id
        );
        eprintln!("   This may take 15-60 seconds depending on model size.");

        let start = std::time::Instant::now();

        // Try to load the model using lms CLI
        let output = tokio::process::Command::new("lms")
            .args(["load", model_id, "--gpu", "max", "-y"])
            .output()
            .await
            .map_err(|e| {
                LlmError::ApiError(format!(
                    "Failed to run 'lms load' command: {}\n\n\
                    Troubleshooting:\n\
                    1. Ensure LM Studio is installed\n\
                    2. Make sure 'lms' CLI is in your PATH\n\
                    3. Run 'lms --help' to verify installation\n\
                    4. Alternatively, manually load the model in LM Studio GUI",
                    e
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);

            return Err(LlmError::ApiError(format!(
                "Failed to load model '{}' via lms CLI:\n{}\n{}\n\n\
                Please manually load the model in LM Studio GUI or check:\n\
                1. Model is downloaded locally (run 'lms ls' to check)\n\
                2. Enough RAM/VRAM available\n\
                3. LM Studio is running",
                model_id, stdout, stderr
            )));
        }

        let elapsed = start.elapsed();
        eprintln!(
            "✅ Model '{}' loaded successfully in {:.1}s",
            model_id,
            elapsed.as_secs_f64()
        );

        Ok(())
    }

    /// Execute a chat request with automatic model loading on failure
    async fn chat_with_auto_load(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        // Try the request first
        match self.chat_internal(messages, options).await {
            Ok(response) => Ok(response),
            Err(e) => {
                // Check if error is due to model not loaded and auto-load is enabled
                if self.auto_load_models && Self::is_model_not_loaded_error(&e.to_string()) {
                    debug!(
                        provider = "lmstudio",
                        model = %self.model,
                        "Model not loaded, attempting automatic load"
                    );

                    // Try to load the model
                    self.load_model_via_cli(&self.model).await?;

                    // Retry the request
                    debug!(
                        provider = "lmstudio",
                        model = %self.model,
                        "Retrying request after model load"
                    );
                    self.chat_internal(messages, options).await
                } else {
                    // Not a model-not-loaded error, or auto-load disabled
                    Err(e)
                }
            }
        }
    }

    /// Execute a chat with tools request with automatic model loading on failure
    async fn chat_with_tools_auto_load(
        &self,
        messages: &[ChatMessage],
        tools: &[crate::traits::ToolDefinition],
        tool_choice: Option<crate::traits::ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        // Try the request first
        match self
            .chat_with_tools_internal(messages, tools, tool_choice.clone(), options)
            .await
        {
            Ok(response) => Ok(response),
            Err(e) => {
                // Check if error is due to model not loaded and auto-load is enabled
                if self.auto_load_models && Self::is_model_not_loaded_error(&e.to_string()) {
                    debug!(
                        provider = "lmstudio",
                        model = %self.model,
                        "Model not loaded for tools request, attempting automatic load"
                    );

                    // Try to load the model
                    self.load_model_via_cli(&self.model).await?;

                    // Retry the request
                    debug!(
                        provider = "lmstudio",
                        model = %self.model,
                        "Retrying tools request after model load"
                    );
                    self.chat_with_tools_internal(messages, tools, tool_choice, options)
                        .await
                } else {
                    // Not a model-not-loaded error, or auto-load disabled
                    Err(e)
                }
            }
        }
    }

    /// Internal chat implementation (without auto-load retry logic)
    async fn chat_internal(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        // OODA-30: Route reasoning models to REST API for thinking support
        if is_reasoning_model(&self.model) {
            debug!(
                provider = "lmstudio",
                model = %self.model,
                "Using REST API for reasoning model"
            );
            return self.chat_with_reasoning(messages, options).await;
        }

        let api_messages: Vec<ChatMessageRequest> = messages
            .iter()
            .map(|m| ChatMessageRequest {
                role: map_role(&m.role).to_string(),
                content: build_content(m),
            })
            .collect();

        let opts = options.cloned().unwrap_or_default();
        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: api_messages,
            temperature: opts.temperature,
            max_tokens: opts.max_tokens.map(|t| t as i32),
            stream: false,
            tools: None,
            tool_choice: None,
        };

        let url = format!("{}/chat/completions", self.api_base());

        debug!(
            provider = "lmstudio",
            model = %self.model,
            url = %url,
            message_count = messages.len(),
            "Sending chat completion request"
        );

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(format!("LM Studio request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            // Try to parse as API error
            if let Ok(api_error) = serde_json::from_str::<ApiError>(&error_text) {
                return Err(LlmError::ApiError(format!(
                    "LM Studio API error ({}): {}",
                    status, api_error.error.message
                )));
            }

            return Err(LlmError::ApiError(format!(
                "LM Studio API error ({}): {}",
                status, error_text
            )));
        }

        let completion: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| LlmError::NetworkError(format!("Failed to parse response: {}", e)))?;

        let content = completion
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        let (prompt_tokens, completion_tokens) = completion
            .usage
            .map(|u| (u.prompt_tokens, u.completion_tokens))
            .unwrap_or((0, 0));

        debug!(
            provider = "lmstudio",
            prompt_tokens = prompt_tokens,
            completion_tokens = completion_tokens,
            content_length = content.len(),
            "Received chat completion response"
        );

        Ok(LLMResponse {
            content,
            prompt_tokens,
            completion_tokens,
            model: self.model.clone(),
            total_tokens: prompt_tokens + completion_tokens,
            finish_reason: completion
                .choices
                .first()
                .and_then(|c| c.finish_reason.clone()),
            tool_calls: Vec::new(),
            metadata: HashMap::new(),
            cache_hit_tokens: None,
            // OODA-15: LMStudio doesn't have thinking API
            thinking_tokens: None,
            thinking_content: None,
        })
    }

    /// Internal chat with tools implementation (without auto-load retry logic)
    async fn chat_with_tools_internal(
        &self,
        messages: &[ChatMessage],
        tools: &[crate::traits::ToolDefinition],
        tool_choice: Option<crate::traits::ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let api_messages: Vec<ChatMessageRequest> = messages
            .iter()
            .map(|m| ChatMessageRequest {
                role: map_role(&m.role).to_string(),
                content: build_content(m),
            })
            .collect();

        // Convert tools to OpenAI-compatible format
        let api_tools: Vec<ToolDefinitionRequest> = tools
            .iter()
            .map(|tool| ToolDefinitionRequest {
                type_: "function".to_string(),
                function: FunctionDefinitionRequest {
                    name: tool.function.name.clone(),
                    description: tool.function.description.clone(),
                    parameters: tool.function.parameters.clone(),
                },
            })
            .collect();

        // Convert tool_choice to API format
        // LMStudio only supports: none, auto, required (not specific functions)
        let api_tool_choice = tool_choice.map(|tc| match tc {
            crate::traits::ToolChoice::Auto(_) => "auto".to_string(),
            crate::traits::ToolChoice::Required(_) => "required".to_string(),
            crate::traits::ToolChoice::Function { .. } => {
                // LMStudio doesn't support specific function selection
                // Fall back to required mode to ensure a tool is called
                "required".to_string()
            }
        });

        let opts = options.cloned().unwrap_or_default();
        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: api_messages,
            temperature: opts.temperature,
            max_tokens: opts.max_tokens.map(|t| t as i32),
            stream: false,
            tools: Some(api_tools),
            tool_choice: api_tool_choice,
        };

        let url = format!("{}/chat/completions", self.api_base());

        debug!(
            provider = "lmstudio",
            model = %self.model,
            url = %url,
            message_count = messages.len(),
            tool_count = tools.len(),
            "Sending chat completion request with tools"
        );

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    LlmError::NetworkError(format!(
                        "LM Studio request timed out at {}. The model may be taking too long to respond.",
                        self.host
                    ))
                } else if e.is_connect() {
                    LlmError::NetworkError(format!(
                        "Lost connection to LM Studio at {}. Was LM Studio closed?",
                        self.host
                    ))
                } else {
                    LlmError::NetworkError(format!(
                        "LM Studio request failed: {}\n\nTroubleshooting:\n\
                         1. Check LM Studio is still running\n\
                         2. Verify model '{}' is loaded\n\
                         3. Check LM Studio console for errors",
                        e, self.model
                    ))
                }
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            // Try to parse as API error
            if let Ok(api_error) = serde_json::from_str::<ApiError>(&error_text) {
                return Err(LlmError::ApiError(format!(
                    "LM Studio API error ({}): {}",
                    status, api_error.error.message
                )));
            }

            return Err(LlmError::ApiError(format!(
                "LM Studio API error ({}): {}",
                status, error_text
            )));
        }

        let completion: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| LlmError::NetworkError(format!("Failed to parse response: {}", e)))?;

        let choice = completion
            .choices
            .first()
            .ok_or_else(|| LlmError::ApiError("No choices in response".to_string()))?;

        let content = choice.message.content.clone();
        let api_tool_calls = &choice.message.tool_calls;

        // Convert tool calls to our format
        let tool_calls: Vec<crate::traits::ToolCall> = api_tool_calls
            .iter()
            .map(|tc| crate::traits::ToolCall {
                id: tc.id.clone(),
                call_type: "function".to_string(),
                function: crate::traits::FunctionCall {
                    name: tc.function.name.clone(),
                    arguments: tc.function.arguments.clone(),
                },
            })
            .collect();

        let (prompt_tokens, completion_tokens) = completion
            .usage
            .map(|u| (u.prompt_tokens, u.completion_tokens))
            .unwrap_or((0, 0));

        debug!(
            provider = "lmstudio",
            prompt_tokens = prompt_tokens,
            completion_tokens = completion_tokens,
            content_length = content.len(),
            tool_call_count = tool_calls.len(),
            "Received chat completion response with tools"
        );

        Ok(LLMResponse {
            content,
            prompt_tokens,
            completion_tokens,
            model: self.model.clone(),
            total_tokens: prompt_tokens + completion_tokens,
            finish_reason: choice.finish_reason.clone(),
            tool_calls,
            metadata: HashMap::new(),
            cache_hit_tokens: None,
            thinking_tokens: None,
            thinking_content: None,
        })
    }
}

// =========================================================================
// OODA-30: Helper function to detect reasoning models
// =========================================================================

/// Check if a model supports reasoning (thinking) capabilities
fn is_reasoning_model(model: &str) -> bool {
    let model_lower = model.to_lowercase();
    model_lower.contains("deepseek-r1")
        || model_lower.contains("qwen3")
        || model_lower.contains("qwq")
        || model_lower.contains("phi4-reasoning")
        || model_lower.contains("granite-4")
        || model_lower.contains("reasoning")
        || model_lower.contains("thinking")
}

// =========================================================================
// OODA-30: Native REST API structures for reasoning support
// =========================================================================

/// REST API request for /api/v1/chat
#[derive(Debug, Serialize)]
struct RestChatRequest {
    model: String,
    input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<i32>,
}

/// REST API response from /api/v1/chat
#[derive(Debug, Deserialize)]
struct RestChatResponse {
    #[allow(dead_code)]
    model_instance_id: String,
    output: Vec<RestOutputItem>,
    stats: RestStats,
}

/// Output item in REST API response
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum RestOutputItem {
    #[serde(rename = "reasoning")]
    Reasoning { content: String },
    #[serde(rename = "message")]
    Message { content: String },
}

/// Stats from REST API response
#[derive(Debug, Deserialize)]
struct RestStats {
    input_tokens: usize,
    total_output_tokens: usize,
    #[serde(default)]
    reasoning_output_tokens: usize,
    #[allow(dead_code)]
    tokens_per_second: f64,
    #[allow(dead_code)]
    time_to_first_token_seconds: f64,
}

/// Streaming event from REST API
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum RestStreamEvent {
    #[serde(rename = "chat.start")]
    ChatStart {
        #[allow(dead_code)]
        model_instance_id: Option<String>,
    },
    #[serde(rename = "reasoning.start")]
    ReasoningStart,
    #[serde(rename = "reasoning.delta")]
    ReasoningDelta { content: String },
    #[serde(rename = "reasoning.end")]
    ReasoningEnd,
    #[serde(rename = "message.start")]
    MessageStart,
    #[serde(rename = "message.delta")]
    MessageDelta { content: String },
    #[serde(rename = "message.end")]
    MessageEnd,
    #[serde(rename = "chat.end")]
    ChatEnd { result: RestChatResponse },
    #[serde(rename = "prompt_processing.start")]
    PromptProcessingStart,
    #[serde(rename = "prompt_processing.progress")]
    PromptProcessingProgress {
        #[allow(dead_code)]
        progress: f64,
    },
    #[serde(rename = "prompt_processing.end")]
    PromptProcessingEnd,
    #[serde(rename = "model_load.start")]
    ModelLoadStart {
        #[allow(dead_code)]
        model_instance_id: Option<String>,
    },
    #[serde(rename = "model_load.progress")]
    ModelLoadProgress {
        #[allow(dead_code)]
        progress: f64,
    },
    #[serde(rename = "model_load.end")]
    ModelLoadEnd {
        #[allow(dead_code)]
        load_time_seconds: Option<f64>,
    },
}

// =========================================================================
// OpenAI-compatible API request/response structures
// =========================================================================

/// Build the `content` value for a `ChatMessageRequest`.
///
/// - When `msg.images` is `None` or empty, returns a plain JSON string (text only).
/// - When images are present, returns an OpenAI-compatible content-parts array:
///   `[{"type":"text","text":"…"},{"type":"image_url","image_url":{"url":"data:…"}}]`
///
/// LM Studio exposes an OpenAI-compatible API, so this is the correct format
/// for vision requests (same as the OpenAI provider).
fn build_content(msg: &ChatMessage) -> serde_json::Value {
    match &msg.images {
        Some(imgs) if !imgs.is_empty() => {
            let mut parts: Vec<serde_json::Value> = vec![serde_json::json!({
                "type": "text",
                "text": msg.content
            })];
            for img in imgs {
                parts.push(build_image_part(img));
            }
            serde_json::Value::Array(parts)
        }
        _ => serde_json::Value::String(msg.content.clone()),
    }
}

/// Build a single OpenAI image_url content part from `ImageData`.
fn build_image_part(img: &ImageData) -> serde_json::Value {
    let url = img.to_data_uri();
    let mut image_url = serde_json::json!({ "url": url });
    if let Some(detail) = &img.detail {
        image_url["detail"] = serde_json::Value::String(detail.clone());
    }
    serde_json::json!({ "type": "image_url", "image_url": image_url })
}

/// Map a `ChatRole` to an OpenAI-compatible role string.
fn map_role(role: &ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
        ChatRole::Tool => "tool",
        ChatRole::Function => "function",
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessageRequest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<i32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinitionRequest>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
}

#[derive(Debug, Serialize)]
struct ToolDefinitionRequest {
    #[serde(rename = "type")]
    type_: String,
    function: FunctionDefinitionRequest,
}

#[derive(Debug, Serialize)]
struct FunctionDefinitionRequest {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct ChatMessageRequest {
    role: String,
    /// Either a plain JSON string (`"text"`) or an OpenAI-compatible content-parts
    /// array (`[{"type":"text",…},{"type":"image_url",…}]`) for vision messages.
    content: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
    usage: Option<UsageInfo>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChatMessageResponse {
    role: String,
    content: String,
    #[serde(default)]
    tool_calls: Vec<ToolCallResponse>,
}

#[derive(Debug, Deserialize)]
struct ToolCallResponse {
    id: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    type_: String,
    function: FunctionCallResponse,
}

#[derive(Debug, Deserialize)]
struct FunctionCallResponse {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct UsageInfo {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

// =========================================================================
// OODA-10: Streaming response types for LM Studio
// =========================================================================
//
// LM Studio uses SSE (Server-Sent Events) with OpenAI-compatible format.
// Each chunk starts with "data: " prefix and contains delta content.
// =========================================================================

/// Streaming response chunk from LM Studio
#[derive(Debug, Deserialize)]
struct StreamingChunk {
    #[allow(dead_code)]
    id: Option<String>,
    choices: Vec<StreamingChoice>,
}

/// Individual choice in a streaming chunk
#[derive(Debug, Deserialize)]
struct StreamingChoice {
    delta: StreamingDelta,
    #[allow(dead_code)]
    index: usize,
    finish_reason: Option<String>,
}

/// Delta content in a streaming chunk
#[derive(Debug, Deserialize)]
struct StreamingDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<StreamingToolCall>>,
}

/// Tool call in streaming format
#[derive(Debug, Deserialize)]
struct StreamingToolCall {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<StreamingFunction>,
}

/// Function details in streaming format
#[derive(Debug, Deserialize)]
struct StreamingFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

// API error handling

#[derive(Debug, Deserialize)]
struct ApiError {
    error: ApiErrorDetail,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ApiErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: Option<String>,
}

/// OODA-39: Response structure for LM Studio /v1/models endpoint
#[derive(Debug, Deserialize)]
pub struct LMStudioModelsResponse {
    pub data: Vec<LMStudioModel>,
}

/// OODA-39: Individual model info from LM Studio
#[derive(Debug, Deserialize)]
pub struct LMStudioModel {
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: u64,
    #[serde(default)]
    pub owned_by: String,
}

impl LMStudioProvider {
    /// List locally available LM Studio models.
    ///
    /// # OODA-39: Dynamic Model Discovery
    ///
    /// Fetches the list of models currently loaded in LM Studio
    /// via the OpenAI-compatible GET /v1/models endpoint. This enables
    /// dynamic model selection instead of relying on a static registry.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let provider = LMStudioProvider::default_local()?;
    /// let models = provider.list_models().await?;
    /// for model in models.data {
    ///     println!("Available: {}", model.id);
    /// }
    /// ```
    pub async fn list_models(&self) -> Result<LMStudioModelsResponse> {
        let url = format!("{}/models", self.api_base());

        debug!(url = %url, "Fetching LM Studio models list");

        let response = self.client.get(&url).send().await.map_err(|e| {
            LlmError::NetworkError(format!(
                "Failed to connect to LM Studio at {}: {}. Is LM Studio running?",
                self.host, e
            ))
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "LM Studio /v1/models returned {}: {}",
                status, body
            )));
        }

        response
            .json::<LMStudioModelsResponse>()
            .await
            .map_err(|e| LlmError::ProviderError(format!("Failed to parse models response: {}", e)))
    }

    /// OODA-30: Chat using native REST API with reasoning support
    ///
    /// Uses the LMStudio REST API endpoint /api/v1/chat which provides
    /// reasoning output for thinking models like DeepSeek-R1, Qwen3, etc.
    async fn chat_with_reasoning(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        // Build input from messages (REST API uses a simpler input format)
        let input = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    ChatRole::System => "system",
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                    ChatRole::Tool => "tool",
                    ChatRole::Function => "function",
                };
                format!("[{}]: {}", role, m.content)
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let opts = options.cloned().unwrap_or_default();
        let request = RestChatRequest {
            model: self.model.clone(),
            input,
            reasoning: Some("on".to_string()),
            stream: Some(false),
            temperature: opts.temperature,
            max_output_tokens: opts.max_tokens.map(|t| t as i32),
        };

        let url = format!("{}/chat", self.rest_api_base());

        debug!(
            provider = "lmstudio",
            model = %self.model,
            url = %url,
            message_count = messages.len(),
            "Sending REST API chat request with reasoning"
        );

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                LlmError::NetworkError(format!("LM Studio REST API request failed: {}", e))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            // Try to parse as API error
            if let Ok(api_error) = serde_json::from_str::<ApiError>(&error_text) {
                return Err(LlmError::ApiError(format!(
                    "LM Studio REST API error ({}): {}",
                    status, api_error.error.message
                )));
            }

            return Err(LlmError::ApiError(format!(
                "LM Studio REST API error ({}): {}",
                status, error_text
            )));
        }

        let rest_response: RestChatResponse = response.json().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to parse REST API response: {}", e))
        })?;

        // Extract content and reasoning from output
        let mut content = String::new();
        let mut reasoning_content = String::new();

        for item in &rest_response.output {
            match item {
                RestOutputItem::Reasoning { content: text } => {
                    reasoning_content.push_str(text);
                }
                RestOutputItem::Message { content: text } => {
                    content.push_str(text);
                }
            }
        }

        debug!(
            provider = "lmstudio",
            prompt_tokens = rest_response.stats.input_tokens,
            completion_tokens = rest_response.stats.total_output_tokens,
            reasoning_tokens = rest_response.stats.reasoning_output_tokens,
            content_length = content.len(),
            reasoning_length = reasoning_content.len(),
            "Received REST API response with reasoning"
        );

        let mut response = LLMResponse {
            content,
            prompt_tokens: rest_response.stats.input_tokens,
            completion_tokens: rest_response.stats.total_output_tokens,
            model: self.model.clone(),
            total_tokens: rest_response.stats.input_tokens
                + rest_response.stats.total_output_tokens,
            finish_reason: Some("stop".to_string()),
            tool_calls: Vec::new(),
            metadata: HashMap::new(),
            cache_hit_tokens: None,
            thinking_tokens: None,
            thinking_content: None,
        };

        // Set thinking/reasoning content if present
        if !reasoning_content.is_empty() {
            response = response
                .with_thinking_content(reasoning_content)
                .with_thinking_tokens(rest_response.stats.reasoning_output_tokens);
        }

        Ok(response)
    }

    /// OODA-32: Stream chat using native REST API with reasoning support
    ///
    /// Uses LMStudio's SSE streaming with reasoning events:
    /// - `reasoning.delta` → `StreamChunk::ThinkingContent`
    /// - `message.delta` → `StreamChunk::Content`
    /// - `chat.end` → `StreamChunk::Finished`
    async fn chat_with_reasoning_stream(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        // Build input from messages
        let input = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    ChatRole::System => "system",
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                    ChatRole::Tool => "tool",
                    ChatRole::Function => "function",
                };
                format!("[{}]: {}", role, m.content)
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let opts = options.cloned().unwrap_or_default();
        let request = RestChatRequest {
            model: self.model.clone(),
            input,
            reasoning: Some("on".to_string()),
            stream: Some(true),
            temperature: opts.temperature,
            max_output_tokens: opts.max_tokens.map(|t| t as i32),
        };

        let url = format!("{}/chat", self.rest_api_base());

        debug!(
            provider = "lmstudio",
            model = %self.model,
            url = %url,
            message_count = messages.len(),
            "Starting REST API streaming with reasoning"
        );

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                LlmError::NetworkError(format!("LM Studio REST API request failed: {}", e))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "LM Studio REST API streaming error ({}): {}",
                status, error_text
            )));
        }

        let stream = response.bytes_stream();

        let mapped_stream = stream.map(|chunk_result| {
            match chunk_result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);

                    // Parse SSE lines (event: X\ndata: {JSON})
                    for line in text.lines() {
                        let line = line.trim();
                        if line.is_empty() || line.starts_with("event:") {
                            continue;
                        }

                        if let Some(json_str) = line.strip_prefix("data: ") {
                            // Try to parse as RestStreamEvent
                            if let Ok(event) = serde_json::from_str::<RestStreamEvent>(json_str) {
                                match event {
                                    RestStreamEvent::ReasoningDelta { content } => {
                                        // Estimate tokens (~4 chars per token)
                                        let tokens_used = content.len() / 4;
                                        return Ok(StreamChunk::ThinkingContent {
                                            text: content,
                                            tokens_used: Some(tokens_used),
                                            budget_total: None,
                                        });
                                    }
                                    RestStreamEvent::MessageDelta { content } => {
                                        if !content.is_empty() {
                                            return Ok(StreamChunk::Content(content));
                                        }
                                    }
                                    RestStreamEvent::ChatEnd { result } => {
                                        // OODA-39: Extract native TTFT from REST API stats
                                        let ttft_ms =
                                            Some(result.stats.time_to_first_token_seconds * 1000.0);
                                        return Ok(StreamChunk::Finished {
                                            reason: "stop".to_string(),
                                            ttft_ms,
                                        });
                                    }
                                    // Ignore other events (start, end, progress)
                                    _ => {}
                                }
                            }
                        }
                    }

                    // Default empty content
                    Ok(StreamChunk::Content(String::new()))
                }
                Err(e) => Err(LlmError::NetworkError(e.to_string())),
            }
        });

        Ok(mapped_stream.boxed())
    }
}

#[async_trait]
impl LLMProvider for LMStudioProvider {
    fn name(&self) -> &str {
        "lmstudio"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_context_length(&self) -> usize {
        self.max_context_length
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
        self.chat_with_auto_load(messages, options).await
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[crate::traits::ToolDefinition],
        tool_choice: Option<crate::traits::ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        self.chat_with_tools_auto_load(messages, tools, tool_choice, options)
            .await
    }

    // =========================================================================
    // OODA-10: Streaming methods for LM Studio
    // =========================================================================

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    fn supports_tool_streaming(&self) -> bool {
        true
    }

    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        let api_messages = vec![ChatMessageRequest {
            role: "user".to_string(),
            content: serde_json::Value::String(prompt.to_string()),
        }];

        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: api_messages,
            temperature: None,
            max_tokens: None,
            stream: true,
            tools: None,
            tool_choice: None,
        };

        let url = format!("{}/chat/completions", self.api_base());

        debug!(
            provider = "lmstudio",
            model = %self.model,
            url = %url,
            "Starting streaming request"
        );

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(format!("LM Studio request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "LM Studio streaming error ({}): {}",
                status, error_text
            )));
        }

        let stream = response.bytes_stream();

        let mapped_stream = stream.map(|chunk_result| {
            match chunk_result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    let mut content = String::new();

                    // Parse SSE lines (data: prefix)
                    for line in text.lines() {
                        let line = line.trim();
                        if line.is_empty() || line == "data: [DONE]" {
                            continue;
                        }
                        if let Some(json_str) = line.strip_prefix("data: ") {
                            if let Ok(chunk) = serde_json::from_str::<StreamingChunk>(json_str) {
                                if let Some(choice) = chunk.choices.first() {
                                    if let Some(c) = &choice.delta.content {
                                        content.push_str(c);
                                    }
                                }
                            }
                        }
                    }
                    Ok(content)
                }
                Err(e) => Err(LlmError::NetworkError(e.to_string())),
            }
        });

        Ok(mapped_stream.boxed())
    }

    async fn chat_with_tools_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        // OODA-32: Route reasoning models to REST API streaming
        // Note: REST API doesn't support tool calling during streaming,
        // so we only use it for reasoning without tools
        if is_reasoning_model(&self.model) && tools.is_empty() {
            debug!(
                provider = "lmstudio",
                model = %self.model,
                "Using REST API streaming for reasoning model"
            );
            return self.chat_with_reasoning_stream(messages, options).await;
        }

        let api_messages: Vec<ChatMessageRequest> = messages
            .iter()
            .map(|m| ChatMessageRequest {
                role: map_role(&m.role).to_string(),
                content: build_content(m),
            })
            .collect();

        // Convert tools to OpenAI-compatible format
        let api_tools: Vec<ToolDefinitionRequest> = tools
            .iter()
            .map(|tool| ToolDefinitionRequest {
                type_: "function".to_string(),
                function: FunctionDefinitionRequest {
                    name: tool.function.name.clone(),
                    description: tool.function.description.clone(),
                    parameters: tool.function.parameters.clone(),
                },
            })
            .collect();

        // Convert tool_choice to API format
        // LMStudio only supports: none, auto, required (not specific functions)
        let api_tool_choice = tool_choice.map(|tc| match tc {
            ToolChoice::Auto(_) => "auto".to_string(),
            ToolChoice::Required(_) => "required".to_string(),
            ToolChoice::Function { .. } => {
                // LMStudio doesn't support specific function selection
                // Fall back to required mode to ensure a tool is called
                "required".to_string()
            }
        });

        let opts = options.cloned().unwrap_or_default();
        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: api_messages,
            temperature: opts.temperature,
            max_tokens: opts.max_tokens.map(|t| t as i32),
            stream: true,
            tools: Some(api_tools),
            tool_choice: api_tool_choice,
        };

        let url = format!("{}/chat/completions", self.api_base());

        debug!(
            provider = "lmstudio",
            model = %self.model,
            url = %url,
            tool_count = tools.len(),
            "Starting streaming request with tools"
        );

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(format!("LM Studio request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "LM Studio streaming error ({}): {}",
                status, error_text
            )));
        }

        let stream = response.bytes_stream();

        let mapped_stream = stream.map(|chunk_result| {
            match chunk_result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);

                    // Parse SSE lines
                    for line in text.lines() {
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        if line == "data: [DONE]" {
                            return Ok(StreamChunk::Finished {
                                reason: "stop".to_string(),
                                ttft_ms: None,
                            });
                        }
                        if let Some(json_str) = line.strip_prefix("data: ") {
                            if let Ok(chunk) = serde_json::from_str::<StreamingChunk>(json_str) {
                                if let Some(choice) = chunk.choices.first() {
                                    // Check for finish reason
                                    if let Some(reason) = &choice.finish_reason {
                                        return Ok(StreamChunk::Finished {
                                            reason: reason.clone(),
                                            ttft_ms: None,
                                        });
                                    }

                                    // Check for tool calls
                                    if let Some(tool_calls) = &choice.delta.tool_calls {
                                        if let Some(tc) = tool_calls.first() {
                                            return Ok(StreamChunk::ToolCallDelta {
                                                index: tc.index,
                                                id: tc.id.clone(),
                                                function_name: tc
                                                    .function
                                                    .as_ref()
                                                    .and_then(|f| f.name.clone()),
                                                function_arguments: tc
                                                    .function
                                                    .as_ref()
                                                    .and_then(|f| f.arguments.clone()),
                                            });
                                        }
                                    }

                                    // Check for content
                                    if let Some(content) = &choice.delta.content {
                                        if !content.is_empty() {
                                            return Ok(StreamChunk::Content(content.clone()));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Default empty content
                    Ok(StreamChunk::Content(String::new()))
                }
                Err(e) => Err(LlmError::NetworkError(e.to_string())),
            }
        });

        Ok(mapped_stream.boxed())
    }
}

#[async_trait]
impl EmbeddingProvider for LMStudioProvider {
    fn name(&self) -> &str {
        "lmstudio"
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
        8192 // Default max tokens for embedding models
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let request = EmbeddingRequest {
            model: self.embedding_model.clone(),
            input: texts.to_vec(),
        };

        let url = format!("{}/embeddings", self.api_base());

        debug!(
            provider = "lmstudio",
            model = %self.embedding_model,
            url = %url,
            text_count = texts.len(),
            "Sending embedding request"
        );

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                LlmError::NetworkError(format!("LM Studio embedding request failed: {}", e))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            // Try to parse as API error
            if let Ok(api_error) = serde_json::from_str::<ApiError>(&error_text) {
                return Err(LlmError::ApiError(format!(
                    "LM Studio embedding API error ({}): {}",
                    status, api_error.error.message
                )));
            }

            return Err(LlmError::ApiError(format!(
                "LM Studio embedding API error ({}): {}",
                status, error_text
            )));
        }

        let embedding_response: EmbeddingResponse = response.json().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to parse embedding response: {}", e))
        })?;

        let embeddings: Vec<Vec<f32>> = embedding_response
            .data
            .into_iter()
            .map(|d| d.embedding)
            .collect();

        debug!(
            provider = "lmstudio",
            embedding_count = embeddings.len(),
            dimension = embeddings.first().map(|e: &Vec<f32>| e.len()).unwrap_or(0),
            "Received embeddings"
        );

        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let builder = LMStudioProviderBuilder::new();
        assert_eq!(builder.host, DEFAULT_LMSTUDIO_HOST);
        assert_eq!(builder.model, DEFAULT_LMSTUDIO_MODEL);
        assert_eq!(builder.embedding_model, DEFAULT_LMSTUDIO_EMBEDDING_MODEL);
        assert_eq!(builder.embedding_dimension, DEFAULT_LMSTUDIO_EMBEDDING_DIM);
    }

    #[test]
    fn test_builder_custom() {
        let builder = LMStudioProviderBuilder::new()
            .host("http://custom:8080")
            .model("custom-model")
            .embedding_model("custom-embed")
            .embedding_dimension(1024);

        assert_eq!(builder.host, "http://custom:8080");
        assert_eq!(builder.model, "custom-model");
        assert_eq!(builder.embedding_model, "custom-embed");
        assert_eq!(builder.embedding_dimension, 1024);
    }

    #[test]
    fn test_provider_build() {
        use crate::traits::{EmbeddingProvider, LLMProvider};

        let provider = LMStudioProviderBuilder::new().build().unwrap();
        assert_eq!(LLMProvider::name(&provider), "lmstudio");
        assert_eq!(LLMProvider::model(&provider), DEFAULT_LMSTUDIO_MODEL);
        assert_eq!(
            EmbeddingProvider::dimension(&provider),
            DEFAULT_LMSTUDIO_EMBEDDING_DIM
        );
    }

    #[test]
    fn test_api_base_with_v1() {
        let provider = LMStudioProviderBuilder::new()
            .host("http://localhost:1234/v1")
            .build()
            .unwrap();
        assert_eq!(provider.api_base(), "http://localhost:1234/v1");
    }

    #[test]
    fn test_api_base_without_v1() {
        let provider = LMStudioProviderBuilder::new()
            .host("http://localhost:1234")
            .build()
            .unwrap();
        assert_eq!(provider.api_base(), "http://localhost:1234/v1");
    }

    #[test]
    fn test_from_env_defaults() {
        // Clean environment
        std::env::remove_var("LMSTUDIO_HOST");
        std::env::remove_var("LMSTUDIO_MODEL");
        std::env::remove_var("LMSTUDIO_EMBEDDING_MODEL");
        std::env::remove_var("LMSTUDIO_EMBEDDING_DIM");

        let provider = LMStudioProvider::from_env().unwrap();
        assert_eq!(provider.host, DEFAULT_LMSTUDIO_HOST);
        assert_eq!(provider.model, DEFAULT_LMSTUDIO_MODEL);
    }

    #[test]
    fn test_from_env_custom() {
        std::env::set_var("LMSTUDIO_HOST", "http://custom:9999");
        std::env::set_var("LMSTUDIO_MODEL", "test-model");
        std::env::set_var("LMSTUDIO_EMBEDDING_MODEL", "test-embed");
        std::env::set_var("LMSTUDIO_EMBEDDING_DIM", "512");

        let provider = LMStudioProvider::from_env().unwrap();
        assert_eq!(provider.host, "http://custom:9999");
        assert_eq!(provider.model, "test-model");
        assert_eq!(provider.embedding_model, "test-embed");
        assert_eq!(provider.embedding_dimension, 512);

        // Clean up
        std::env::remove_var("LMSTUDIO_HOST");
        std::env::remove_var("LMSTUDIO_MODEL");
        std::env::remove_var("LMSTUDIO_EMBEDDING_MODEL");
        std::env::remove_var("LMSTUDIO_EMBEDDING_DIM");
    }

    // =========================================================================
    // OODA-30: Tests for reasoning model detection
    // =========================================================================

    #[test]
    fn test_is_reasoning_model_deepseek_r1() {
        assert!(is_reasoning_model("deepseek-r1"));
        assert!(is_reasoning_model("deepseek-r1-7b"));
        assert!(is_reasoning_model("DEEPSEEK-R1-distill"));
        assert!(is_reasoning_model(
            "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF"
        ));
    }

    #[test]
    fn test_is_reasoning_model_qwen3() {
        assert!(is_reasoning_model("qwen3"));
        assert!(is_reasoning_model("qwen3-14b"));
        assert!(is_reasoning_model("Qwen3-Thinking"));
    }

    #[test]
    fn test_is_reasoning_model_others() {
        assert!(is_reasoning_model("qwq"));
        assert!(is_reasoning_model("phi4-reasoning"));
        assert!(is_reasoning_model("granite-4"));
        assert!(is_reasoning_model("model-with-reasoning"));
        assert!(is_reasoning_model("model-thinking"));
    }

    #[test]
    fn test_is_reasoning_model_non_reasoning() {
        assert!(!is_reasoning_model("llama-3"));
        assert!(!is_reasoning_model("gpt-oss-20b"));
        assert!(!is_reasoning_model("mistral-7b"));
        assert!(!is_reasoning_model("gemma2-9b"));
    }

    // =========================================================================
    // OODA-30: Tests for REST API base URL
    // =========================================================================

    #[test]
    fn test_rest_api_base() {
        let provider = LMStudioProviderBuilder::new()
            .host("http://localhost:1234")
            .build()
            .unwrap();
        assert_eq!(provider.rest_api_base(), "http://localhost:1234/api/v1");
    }

    #[test]
    fn test_rest_api_base_with_v1() {
        let provider = LMStudioProviderBuilder::new()
            .host("http://localhost:1234/v1")
            .build()
            .unwrap();
        assert_eq!(provider.rest_api_base(), "http://localhost:1234/api/v1");
    }

    // =========================================================================
    // OODA-30: Tests for REST API response parsing
    // =========================================================================

    #[test]
    fn test_rest_output_item_parsing_reasoning() {
        let json = r#"{"type": "reasoning", "content": "Let me think..."}"#;
        let item: RestOutputItem = serde_json::from_str(json).unwrap();
        match item {
            RestOutputItem::Reasoning { content } => {
                assert_eq!(content, "Let me think...");
            }
            _ => panic!("Expected Reasoning variant"),
        }
    }

    #[test]
    fn test_rest_output_item_parsing_message() {
        let json = r#"{"type": "message", "content": "The answer is 42."}"#;
        let item: RestOutputItem = serde_json::from_str(json).unwrap();
        match item {
            RestOutputItem::Message { content } => {
                assert_eq!(content, "The answer is 42.");
            }
            _ => panic!("Expected Message variant"),
        }
    }

    #[test]
    fn test_rest_stats_parsing() {
        let json = r#"{
            "input_tokens": 100,
            "total_output_tokens": 150,
            "reasoning_output_tokens": 50,
            "tokens_per_second": 43.73,
            "time_to_first_token_seconds": 0.781
        }"#;
        let stats: RestStats = serde_json::from_str(json).unwrap();
        assert_eq!(stats.input_tokens, 100);
        assert_eq!(stats.total_output_tokens, 150);
        assert_eq!(stats.reasoning_output_tokens, 50);
    }

    #[test]
    fn test_rest_chat_request_serialization() {
        let request = RestChatRequest {
            model: "deepseek-r1".to_string(),
            input: "What is 2+2?".to_string(),
            reasoning: Some("on".to_string()),
            stream: Some(false),
            temperature: Some(0.7),
            max_output_tokens: Some(1000),
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"model\":\"deepseek-r1\""));
        assert!(json.contains("\"reasoning\":\"on\""));
        assert!(json.contains("\"stream\":false"));
    }

    #[test]
    fn test_rest_stream_event_parsing_reasoning_delta() {
        let json = r#"{"type": "reasoning.delta", "content": "Step 1..."}"#;
        let event: RestStreamEvent = serde_json::from_str(json).unwrap();
        match event {
            RestStreamEvent::ReasoningDelta { content } => {
                assert_eq!(content, "Step 1...");
            }
            _ => panic!("Expected ReasoningDelta variant"),
        }
    }

    #[test]
    fn test_rest_stream_event_parsing_message_delta() {
        let json = r#"{"type": "message.delta", "content": "Hello"}"#;
        let event: RestStreamEvent = serde_json::from_str(json).unwrap();
        match event {
            RestStreamEvent::MessageDelta { content } => {
                assert_eq!(content, "Hello");
            }
            _ => panic!("Expected MessageDelta variant"),
        }
    }

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_LMSTUDIO_HOST, "http://localhost:1234");
        assert_eq!(DEFAULT_LMSTUDIO_MODEL, "gemma2-9b-it");
        assert_eq!(DEFAULT_LMSTUDIO_EMBEDDING_MODEL, "nomic-embed-text-v1.5");
        assert_eq!(DEFAULT_LMSTUDIO_EMBEDDING_DIM, 768);
    }

    #[test]
    fn test_builder_auto_load_models_default() {
        let builder = LMStudioProviderBuilder::default();
        assert!(builder.auto_load_models);
    }

    #[test]
    fn test_builder_auto_load_models_disabled() {
        let provider = LMStudioProviderBuilder::new()
            .auto_load_models(false)
            .build()
            .unwrap();
        assert!(!provider.auto_load_models);
    }

    #[test]
    fn test_builder_max_context_length() {
        let provider = LMStudioProviderBuilder::new()
            .max_context_length(65536)
            .build()
            .unwrap();
        assert_eq!(provider.max_context_length(), 65536);
    }

    #[test]
    fn test_supports_streaming() {
        let provider = LMStudioProviderBuilder::new().build().unwrap();
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_supports_json_mode() {
        let provider = LMStudioProviderBuilder::new().build().unwrap();
        // LM Studio doesn't override supports_json_mode, so default is false
        assert!(!provider.supports_json_mode());
    }

    #[test]
    fn test_embedding_provider_name() {
        let provider = LMStudioProviderBuilder::new().build().unwrap();
        assert_eq!(EmbeddingProvider::name(&provider), "lmstudio");
    }

    #[test]
    fn test_embedding_provider_model() {
        let provider = LMStudioProviderBuilder::new()
            .embedding_model("custom-embed")
            .build()
            .unwrap();
        assert_eq!(EmbeddingProvider::model(&provider), "custom-embed");
    }

    #[test]
    fn test_embedding_provider_max_tokens() {
        let provider = LMStudioProviderBuilder::new().build().unwrap();
        assert_eq!(provider.max_tokens(), 8192);
    }

    #[tokio::test]
    async fn test_embed_empty_input() {
        let provider = LMStudioProviderBuilder::new().build().unwrap();
        let result = provider.embed(&[]).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_rest_chat_response_parsing() {
        let json = r#"{
            "model_instance_id": "test-instance",
            "output": [
                {"type": "reasoning", "content": "Thinking..."},
                {"type": "message", "content": "Answer"}
            ],
            "stats": {
                "input_tokens": 10,
                "total_output_tokens": 20,
                "reasoning_output_tokens": 5,
                "tokens_per_second": 30.0,
                "time_to_first_token_seconds": 0.2
            }
        }"#;
        let response: RestChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.output.len(), 2);
    }

    // ---- Vision / multimodal message tests ----

    #[test]
    fn test_build_content_text_only_is_string() {
        use crate::traits::ChatMessage;
        let msg = ChatMessage::user("hello world");
        let content = build_content(&msg);
        assert!(
            content.is_string(),
            "Text-only message must serialize as plain JSON string"
        );
        assert_eq!(content.as_str().unwrap(), "hello world");
    }

    #[test]
    fn test_build_content_with_image_is_array() {
        use crate::traits::{ChatMessage, ImageData};
        let img = ImageData::new("abc123", "image/png");
        let msg = ChatMessage::user_with_images("describe this", vec![img]);
        let content = build_content(&msg);
        assert!(
            content.is_array(),
            "Vision message must serialize as content-parts array"
        );
        let parts = content.as_array().unwrap();
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "describe this");
        assert_eq!(parts[1]["type"], "image_url");
        let url = parts[1]["image_url"]["url"].as_str().unwrap();
        assert!(
            url.starts_with("data:image/png;base64,"),
            "Image URL must be data URI, got: {}",
            url
        );
    }

    #[test]
    fn test_build_content_empty_images_is_string() {
        use crate::traits::ChatMessage;
        let mut msg = ChatMessage::user("no images here");
        msg.images = Some(vec![]); // explicitly empty
        let content = build_content(&msg);
        assert!(
            content.is_string(),
            "Empty images vec must also serialize as plain string"
        );
    }
}
