//! Ollama provider implementation.
//!
//! This module provides integration with Ollama's local LLM API.
//! Ollama provides an OpenAI-compatible API, so this provider wraps
//! the functionality with Ollama-specific defaults.
//!
//! # Default Configuration
//!
//! - Base URL: `http://localhost:11434`
//! - Default model: `gemma3:12b` (chat), `embeddinggemma:latest` (embeddings, 768 dimensions)
//!
//! # Environment Variables
//!
//! - `OLLAMA_HOST`: Ollama server URL (default: http://localhost:11434)
//! - `OLLAMA_MODEL`: Default chat model
//! - `OLLAMA_EMBEDDING_MODEL`: Default embedding model
//!
//! # Example
//!
//! ```rust,ignore
//! use edgequake_llm::OllamaProvider;
//!
//! // Connect to local Ollama with defaults
//! let provider = OllamaProvider::from_env()?;
//!
//! // Or specify custom settings
//! let provider = OllamaProvider::builder()
//!     .host("http://localhost:11434")
//!     .model("mistral")
//!     .embedding_model("nomic-embed-text")
//!     .build()?;
//! ```

use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::error::{LlmError, Result};
use crate::traits::{
    ChatMessage, ChatRole, CompletionOptions, EmbeddingProvider, FunctionCall, LLMProvider,
    LLMResponse, StreamChunk as TraitStreamChunk, ToolCall, ToolChoice, ToolDefinition,
};

/// Default Ollama host URL
const DEFAULT_OLLAMA_HOST: &str = "http://localhost:11434";

/// Default Ollama chat model
const DEFAULT_OLLAMA_MODEL: &str = "gemma3:12b";

/// Default Ollama embedding model
const DEFAULT_OLLAMA_EMBEDDING_MODEL: &str = "embeddinggemma:latest";

/// Ollama LLM and embedding provider.
///
/// Provides integration with locally running Ollama instance.
/// Supports both chat completion and text embeddings.
#[derive(Debug, Clone)]
pub struct OllamaProvider {
    client: Client,
    host: String,
    model: String,
    embedding_model: String,
    max_context_length: usize,
    embedding_dimension: usize,
}

/// Builder for OllamaProvider
#[derive(Debug, Clone)]
pub struct OllamaProviderBuilder {
    host: String,
    model: String,
    embedding_model: String,
    max_context_length: usize,
    embedding_dimension: usize,
}

impl Default for OllamaProviderBuilder {
    fn default() -> Self {
        Self {
            host: DEFAULT_OLLAMA_HOST.to_string(),
            model: DEFAULT_OLLAMA_MODEL.to_string(),
            embedding_model: DEFAULT_OLLAMA_EMBEDDING_MODEL.to_string(),
            max_context_length: 131072, // OODA-99: Increased to 128K (131072)
            embedding_dimension: 768, // embeddinggemma:latest default (VERIFIED via Ollama API)
        }
    }
}

impl OllamaProviderBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the Ollama host URL
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

    /// Build the OllamaProvider
    pub fn build(self) -> Result<OllamaProvider> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // Longer timeout for local models
            .no_proxy() // CRITICAL: Disable all proxies for localhost connections
            .build()
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        Ok(OllamaProvider {
            client,
            host: self.host,
            model: self.model,
            embedding_model: self.embedding_model,
            max_context_length: self.max_context_length,
            embedding_dimension: self.embedding_dimension,
        })
    }
}

impl OllamaProvider {
    /// Create a new OllamaProvider from environment variables.
    ///
    /// Environment variables:
    /// - `OLLAMA_HOST`: Server URL (default: http://localhost:11434)
    /// - `OLLAMA_MODEL`: Chat model (default: llama3)
    /// - `OLLAMA_EMBEDDING_MODEL`: Embedding model (default: nomic-embed-text)
    /// - `OLLAMA_CONTEXT_LENGTH`: Max context length (default: 131072 = 128K)
    ///
    /// # OODA-99: Context Length Configuration
    ///
    /// Default is 128K which works for most modern models. Override if needed:
    /// Example: `export OLLAMA_CONTEXT_LENGTH=65536` for 64K
    pub fn from_env() -> Result<Self> {
        let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| DEFAULT_OLLAMA_HOST.to_string());

        let model =
            std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_OLLAMA_MODEL.to_string());

        let embedding_model = std::env::var("OLLAMA_EMBEDDING_MODEL")
            .unwrap_or_else(|_| DEFAULT_OLLAMA_EMBEDDING_MODEL.to_string());

        // OODA-99: Allow context length override, default 128K
        let max_context_length = std::env::var("OLLAMA_CONTEXT_LENGTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(131072);

        OllamaProviderBuilder::new()
            .host(host)
            .model(model)
            .embedding_model(embedding_model)
            .max_context_length(max_context_length)
            .build()
    }

    /// Create a new builder for OllamaProvider
    pub fn builder() -> OllamaProviderBuilder {
        OllamaProviderBuilder::new()
    }

    /// Create with default settings (localhost:11434)
    pub fn default_local() -> Result<Self> {
        OllamaProviderBuilder::new().build()
    }
}

// Ollama API request/response structures

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<ChatOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OllamaTool>>,  // OODA-09: Tool support
    /// OODA-29: Enable thinking for reasoning models (deepseek-r1, qwen3)
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
}

#[derive(Debug, Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    /// Context window size in tokens sent to Ollama (default: provider's max_context_length)
    #[serde(skip_serializing_if = "Option::is_none")]
    num_ctx: Option<usize>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ChatResponse {
    model: String,
    message: ResponseMessage,
    done: bool,
    #[serde(default)]
    total_duration: u64,
    #[serde(default)]
    prompt_eval_count: u32,
    #[serde(default)]
    eval_count: u32,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    #[allow(dead_code)]
    role: String,
    content: String,
    /// OODA-29: Thinking content for reasoning models (deepseek-r1, qwen3)
    #[serde(default)]
    thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,  // OODA-09: Tool calls in response
}

// OODA-09: Renamed to avoid conflict with traits::StreamChunk
#[derive(Debug, Deserialize)]
struct OllamaStreamChunk {
    #[serde(default)]
    message: Option<ResponseMessage>,
    #[allow(dead_code)]
    done: bool,
}

// ============================================================================
// OODA-09: Tool Calling Types
// ============================================================================

/// Tool definition for Ollama (OpenAI-compatible format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaTool {
    pub r#type: String,
    pub function: OllamaFunction,
}

/// Function definition within a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Tool call in response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolCall {
    pub function: OllamaFunctionCall,
}

/// Function call details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaFunctionCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
}

// ============================================================================
// OODA-78: List Models API Types
// ============================================================================

/// Response from GET /api/tags endpoint.
///
/// Lists all locally available Ollama models.
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModelsResponse {
    /// Array of available models.
    pub models: Vec<OllamaModelInfo>,
}

/// Model details from Ollama API.
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModelDetails {
    /// Parent model (if any).
    #[serde(default)]
    pub parent_model: String,
    /// Model format (usually "gguf").
    #[serde(default)]
    pub format: String,
    /// Model family (e.g., "llama", "qwen2").
    #[serde(default)]
    pub family: String,
    /// List of model families.
    #[serde(default)]
    pub families: Vec<String>,
    /// Parameter size (e.g., "7.6B").
    #[serde(default)]
    pub parameter_size: String,
    /// Quantization level (e.g., "Q4_K_M").
    #[serde(default)]
    pub quantization_level: String,
}

/// Individual model info from Ollama API.
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModelInfo {
    /// Model name (e.g., "llama3.2:latest").
    pub name: String,
    /// Model identifier.
    #[serde(default)]
    pub model: String,
    /// Last modified timestamp.
    #[serde(default)]
    pub modified_at: String,
    /// Model size in bytes.
    #[serde(default)]
    pub size: u64,
    /// Model digest.
    #[serde(default)]
    pub digest: String,
    /// Model details.
    #[serde(default)]
    pub details: Option<OllamaModelDetails>,
}

impl OllamaProvider {
    fn convert_role(role: &ChatRole) -> &'static str {
        match role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool | ChatRole::Function => "user", // Ollama doesn't have tool/function role
        }
    }

    fn convert_messages(messages: &[ChatMessage]) -> Vec<OllamaMessage> {
        messages
            .iter()
            .map(|msg| OllamaMessage {
                role: Self::convert_role(&msg.role).to_string(),
                content: msg.content.clone(),
            })
            .collect()
    }
    
    /// Convert tool definitions to Ollama's OpenAI-compatible format.
    ///
    /// # OODA-09: Tool Calling Support
    ///
    /// Ollama uses the OpenAI-compatible tools format:
    /// ```json
    /// {
    ///   "type": "function",
    ///   "function": {
    ///     "name": "tool_name",
    ///     "description": "what it does",
    ///     "parameters": { ... JSON Schema ... }
    ///   }
    /// }
    /// ```
    fn convert_tools(tools: &[ToolDefinition]) -> Vec<OllamaTool> {
        tools
            .iter()
            .map(|tool| OllamaTool {
                r#type: "function".to_string(),
                function: OllamaFunction {
                    name: tool.function.name.clone(),
                    description: tool.function.description.clone(),
                    parameters: tool.function.parameters.clone(),
                },
            })
            .collect()
    }

    /// List locally available Ollama models.
    ///
    /// # OODA-78: Dynamic Model Discovery
    ///
    /// Fetches the list of models currently installed on the Ollama server
    /// via the GET /api/tags endpoint. This enables dynamic model selection
    /// instead of relying on a static registry.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let provider = OllamaProvider::default_local()?;
    /// let models = provider.list_models().await?;
    /// for model in models.models {
    ///     println!("Available: {} ({})", model.name, model.details.unwrap().parameter_size);
    /// }
    /// ```
    pub async fn list_models(&self) -> Result<OllamaModelsResponse> {
        let url = format!("{}/api/tags", self.host);

        debug!(url = %url, "Fetching Ollama models list");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "Ollama API error ({}): {}",
                status, error_text
            )));
        }

        let models: OllamaModelsResponse = response
            .json()
            .await
            .map_err(|e| LlmError::ApiError(format!("Failed to parse models response: {}", e)))?;

        debug!("Ollama returned {} models", models.models.len());

        Ok(models)
    }

    /// Get host URL for this provider.
    pub fn host(&self) -> &str {
        &self.host
    }

    /// OODA-29: Check if model supports thinking/reasoning.
    ///
    /// Returns true for models known to support the `think` parameter:
    /// - DeepSeek R1 models (deepseek-r1)
    /// - Qwen 3 models (qwen3)
    /// - Other thinking-tagged models
    fn is_thinking_model(model: &str) -> bool {
        let model_lower = model.to_lowercase();
        model_lower.contains("deepseek-r1")
            || model_lower.contains("qwen3")
            || model_lower.contains("qwq")
            || model_lower.contains("openthinker")
            || model_lower.contains("phi4-reasoning")
            || model_lower.contains("magistral")
            || model_lower.contains("cogito")
            || model_lower.contains("gpt-oss")  // OpenAI open-weight reasoning
    }
}

#[async_trait]
impl LLMProvider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_context_length(&self) -> usize {
        self.max_context_length
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
        debug!(
            "Ollama chat request: {} messages to model {}",
            messages.len(),
            self.model
        );

        let url = format!("{}/api/chat", self.host);
        let opts = options.cloned().unwrap_or_default();

        let chat_options = ChatOptions {
            temperature: opts.temperature,
            num_predict: opts.max_tokens.map(|t| t as i32),
            stop: opts.stop.clone(),
            num_ctx: Some(self.max_context_length),
        };

        // OODA-29: Enable thinking for reasoning models
        let think = Self::is_thinking_model(&self.model);

        let request = ChatRequest {
            model: self.model.clone(),
            messages: Self::convert_messages(messages),
            stream: false,
            options: Some(chat_options),
            tools: None,  // OODA-09: No tools for basic chat
            think: if think { Some(true) } else { None },
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "Ollama API error ({}): {}",
                status, error_text
            )));
        }

        let response: ChatResponse = response
            .json()
            .await
            .map_err(|e| LlmError::ApiError(format!("Failed to parse response: {}", e)))?;

        // OODA-29: Build response with thinking content if available
        let mut llm_response = LLMResponse::new(response.message.content, response.model)
            .with_usage(
                response.prompt_eval_count as usize,
                response.eval_count as usize,
            );

        // Add thinking content if present
        if let Some(thinking) = &response.message.thinking {
            if !thinking.is_empty() {
                llm_response = llm_response.with_thinking_content(thinking.clone());
                // Estimate thinking tokens (rough: ~4 chars per token)
                let thinking_tokens = thinking.len() / 4;
                llm_response = llm_response.with_thinking_tokens(thinking_tokens);
            }
        }

        Ok(llm_response)
    }

    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        use futures::StreamExt;

        debug!("Ollama stream request: prompt to model {}", self.model);

        let url = format!("{}/api/chat", self.host);

        let chat_options = ChatOptions {
            temperature: None,
            num_predict: None,
            stop: None,
            num_ctx: Some(self.max_context_length),
        };

        // OODA-29: Enable thinking for reasoning models
        let think = Self::is_thinking_model(&self.model);

        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![OllamaMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            stream: true,
            options: Some(chat_options),
            tools: None,  // OODA-09: No tools for basic stream
            think: if think { Some(true) } else { None },
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "Ollama API error ({}): {}",
                status, error_text
            )));
        }

        let stream = response.bytes_stream();

        let mapped_stream = stream.map(|chunk_result| {
            match chunk_result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    // Parse NDJSON - each line is a separate JSON object
                    let mut content = String::new();
                    for line in text.lines() {
                        if line.is_empty() {
                            continue;
                        }
                        // OODA-09: Use renamed OllamaStreamChunk
                        if let Ok(chunk) = serde_json::from_str::<OllamaStreamChunk>(line) {
                            if let Some(msg) = chunk.message {
                                content.push_str(&msg.content);
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

    fn supports_streaming(&self) -> bool {
        true
    }
    
    // =========================================================================
    // OODA-09: Tool Calling Support for Ollama
    // =========================================================================
    //
    // Ollama supports tool calling via the OpenAI-compatible tools parameter.
    // See: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
    //
    // Models with tool support: llama3.2, mistral, qwen2, etc.
    // =========================================================================
    
    fn supports_function_calling(&self) -> bool {
        true // Ollama supports tools for compatible models
    }
    
    fn supports_tool_streaming(&self) -> bool {
        true // Ollama supports streaming with tools
    }
    
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        _tool_choice: Option<ToolChoice>,  // Ollama doesn't support tool_choice
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let url = format!("{}/api/chat", self.host);
        let opts = options.cloned().unwrap_or_default();
        
        let chat_options = ChatOptions {
            temperature: opts.temperature,
            num_predict: opts.max_tokens.map(|t| t as i32),
            stop: opts.stop.clone(),
            num_ctx: Some(self.max_context_length),
        };

        // Convert tools to Ollama format
        let ollama_tools = if !tools.is_empty() {
            Some(Self::convert_tools(tools))
        } else {
            None
        };

        // OODA-29: Enable thinking for reasoning models
        let think = Self::is_thinking_model(&self.model);
        
        let request = ChatRequest {
            model: self.model.clone(),
            messages: Self::convert_messages(messages),
            stream: false,
            options: Some(chat_options),
            tools: ollama_tools,
            think: if think { Some(true) } else { None },
        };
        
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;
        
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "Ollama API error ({}): {}",
                status, error_text
            )));
        }
        
        let response: ChatResponse = response
            .json()
            .await
            .map_err(|e| LlmError::ApiError(format!("Failed to parse response: {}", e)))?;
        
        // Convert tool calls if present
        let tool_calls: Vec<ToolCall> = response.message.tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|tc| ToolCall {
                id: uuid::Uuid::new_v4().to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: tc.function.name,
                    arguments: serde_json::to_string(&tc.function.arguments).unwrap_or_default(),
                },
            })
            .collect();

        // OODA-29: Build response with thinking content if available
        let mut llm_response = LLMResponse::new(response.message.content, response.model)
            .with_usage(response.prompt_eval_count as usize, response.eval_count as usize)
            .with_tool_calls(tool_calls);

        // Add thinking content if present
        if let Some(thinking) = &response.message.thinking {
            if !thinking.is_empty() {
                llm_response = llm_response.with_thinking_content(thinking.clone());
                // Estimate thinking tokens (rough: ~4 chars per token)
                let thinking_tokens = thinking.len() / 4;
                llm_response = llm_response.with_thinking_tokens(thinking_tokens);
            }
        }

        Ok(llm_response)
    }
    
    async fn chat_with_tools_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        _tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<TraitStreamChunk>>> {
        use futures::StreamExt;
        
        let url = format!("{}/api/chat", self.host);
        let opts = options.cloned().unwrap_or_default();
        
        let chat_options = ChatOptions {
            temperature: opts.temperature,
            num_predict: opts.max_tokens.map(|t| t as i32),
            stop: opts.stop.clone(),
            num_ctx: Some(self.max_context_length),
        };

        let ollama_tools = if !tools.is_empty() {
            Some(Self::convert_tools(tools))
        } else {
            None
        };

        // OODA-29: Enable thinking for reasoning models
        let think = Self::is_thinking_model(&self.model);
        
        let request = ChatRequest {
            model: self.model.clone(),
            messages: Self::convert_messages(messages),
            stream: true,
            options: Some(chat_options),
            tools: ollama_tools,
            think: if think { Some(true) } else { None },
        };
        
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;
        
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "Ollama API error ({}): {}",
                status, error_text
            )));
        }
        
        let stream = response.bytes_stream();
        
        let mapped_stream = stream.map(|chunk_result| {
            match chunk_result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    
                    for line in text.lines() {
                        if line.is_empty() {
                            continue;
                        }
                        if let Ok(chunk) = serde_json::from_str::<OllamaStreamChunk>(line) {
                            if let Some(msg) = chunk.message {
                                // OODA-29: Check for thinking content first
                                if let Some(thinking) = &msg.thinking {
                                    if !thinking.is_empty() {
                                        // Estimate thinking tokens (rough: ~4 chars per token)
                                        let tokens_used = thinking.len() / 4;
                                        return Ok(TraitStreamChunk::ThinkingContent {
                                            text: thinking.clone(),
                                            tokens_used: Some(tokens_used),
                                            budget_total: None,
                                        });
                                    }
                                }
                                
                                // Check for tool calls
                                if let Some(tool_calls) = msg.tool_calls {
                                    if let Some(tc) = tool_calls.first() {
                                        return Ok(TraitStreamChunk::ToolCallDelta {
                                            index: 0,
                                            id: Some(uuid::Uuid::new_v4().to_string()),
                                            function_name: Some(tc.function.name.clone()),
                                            function_arguments: serde_json::to_string(&tc.function.arguments).ok(),
                                        });
                                    }
                                }
                                
                                // Return content if not empty
                                if !msg.content.is_empty() {
                                    return Ok(TraitStreamChunk::Content(msg.content));
                                }
                            }
                            
                            // Check for completion
                            if chunk.done {
                                return Ok(TraitStreamChunk::Finished {
                                    reason: "stop".to_string(),
                                    ttft_ms: None,
                                });
                            }
                        }
                    }
                    
                    // Default empty content
                    Ok(TraitStreamChunk::Content(String::new()))
                }
                Err(e) => Err(LlmError::NetworkError(e.to_string())),
            }
        });
        
        Ok(mapped_stream.boxed())
    }
}

#[async_trait]
impl EmbeddingProvider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
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
        8192 // Most Ollama embedding models support this
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        debug!(
            "Ollama embedding request: {} texts with model {}",
            texts.len(),
            self.embedding_model
        );

        let url = format!("{}/api/embed", self.host);

        let request = EmbeddingRequest {
            model: self.embedding_model.clone(),
            input: texts.to_vec(),
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "Ollama API error ({}): {}",
                status, error_text
            )));
        }

        let response: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| LlmError::ApiError(format!("Failed to parse response: {}", e)))?;

        debug!(
            "Ollama embedding response: {} embeddings",
            response.embeddings.len()
        );

        Ok(response.embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let provider = OllamaProviderBuilder::new()
            .host("http://localhost:11434")
            .model("mistral")
            .embedding_model("nomic-embed-text")
            .build()
            .unwrap();

        assert_eq!(LLMProvider::name(&provider), "ollama");
        assert_eq!(LLMProvider::model(&provider), "mistral");
        assert_eq!(EmbeddingProvider::model(&provider), "nomic-embed-text");
    }

    #[test]
    fn test_default_builder() {
        let provider = OllamaProviderBuilder::new().build().unwrap();

        assert_eq!(LLMProvider::name(&provider), "ollama");
        assert_eq!(LLMProvider::model(&provider), "gemma3:12b");
        assert_eq!(EmbeddingProvider::model(&provider), "embeddinggemma:latest");
        assert_eq!(provider.max_context_length(), 131072);
    }

    #[test]
    fn test_message_conversion() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
        ];

        let converted = OllamaProvider::convert_messages(&messages);

        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0].role, "system");
        assert_eq!(converted[1].role, "user");
        assert_eq!(converted[2].role, "assistant");
    }

    #[tokio::test]
    async fn test_embed_empty_input() {
        let provider = OllamaProviderBuilder::new().build().unwrap();
        let result = provider.embed(&[]).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    // OODA-29: Tests for thinking model detection
    #[test]
    fn test_is_thinking_model_deepseek_r1() {
        assert!(OllamaProvider::is_thinking_model("deepseek-r1:8b"));
        assert!(OllamaProvider::is_thinking_model("deepseek-r1:70b"));
        assert!(OllamaProvider::is_thinking_model("DEEPSEEK-R1:latest"));
    }

    #[test]
    fn test_is_thinking_model_qwen3() {
        assert!(OllamaProvider::is_thinking_model("qwen3:8b"));
        assert!(OllamaProvider::is_thinking_model("qwen3:32b"));
        assert!(OllamaProvider::is_thinking_model("QWEN3:latest"));
    }

    #[test]
    fn test_is_thinking_model_others() {
        assert!(OllamaProvider::is_thinking_model("qwq:32b"));
        assert!(OllamaProvider::is_thinking_model("openthinker:7b"));
        assert!(OllamaProvider::is_thinking_model("phi4-reasoning:14b"));
        assert!(OllamaProvider::is_thinking_model("magistral:24b"));
        assert!(OllamaProvider::is_thinking_model("cogito:8b"));
        assert!(OllamaProvider::is_thinking_model("gpt-oss:20b"));
    }

    #[test]
    fn test_is_thinking_model_non_thinking() {
        assert!(!OllamaProvider::is_thinking_model("llama3.2:8b"));
        assert!(!OllamaProvider::is_thinking_model("gemma3:12b"));
        assert!(!OllamaProvider::is_thinking_model("mistral:7b"));
        assert!(!OllamaProvider::is_thinking_model("codellama:34b"));
    }

    #[test]
    fn test_response_with_thinking_parsing() {
        // Test that ResponseMessage can parse thinking field
        let json = r#"{
            "role": "assistant",
            "content": "The answer is 3.",
            "thinking": "Let me count the r's in strawberry: s-t-r-a-w-b-e-r-r-y. That's 3 r's."
        }"#;
        
        let msg: ResponseMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content, "The answer is 3.");
        assert!(msg.thinking.is_some());
        assert!(msg.thinking.unwrap().contains("Let me count"));
    }

    #[test]
    fn test_response_without_thinking_parsing() {
        // Test that ResponseMessage works without thinking field
        let json = r#"{
            "role": "assistant",
            "content": "Hello, how can I help you?"
        }"#;
        
        let msg: ResponseMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content, "Hello, how can I help you?");
        assert!(msg.thinking.is_none());
    }

    #[test]
    fn test_chat_request_with_think_serialization() {
        let request = ChatRequest {
            model: "deepseek-r1:8b".to_string(),
            messages: vec![OllamaMessage {
                role: "user".to_string(),
                content: "How many r's in strawberry?".to_string(),
            }],
            stream: false,
            options: None,
            tools: None,
            think: Some(true),
        };
        
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"think\":true"));
    }

    #[test]
    fn test_chat_request_without_think_serialization() {
        let request = ChatRequest {
            model: "llama3.2:8b".to_string(),
            messages: vec![OllamaMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            stream: false,
            options: None,
            tools: None,
            think: None,
        };
        
        let json = serde_json::to_string(&request).unwrap();
        // think should not be present when None
        assert!(!json.contains("think"));
    }

    #[test]
    fn test_chat_options_num_ctx_serialization() {
        let options = ChatOptions {
            temperature: None,
            num_predict: None,
            stop: None,
            num_ctx: Some(65536),
        };

        let json = serde_json::to_string(&options).unwrap();
        assert!(
            json.contains("\"num_ctx\":65536"),
            "num_ctx should be serialized in options: {}",
            json
        );
    }

    #[test]
    fn test_chat_options_num_ctx_omitted_when_none() {
        let options = ChatOptions {
            temperature: None,
            num_predict: None,
            stop: None,
            num_ctx: None,
        };

        let json = serde_json::to_string(&options).unwrap();
        assert!(
            !json.contains("num_ctx"),
            "num_ctx should be omitted when None: {}",
            json
        );
    }
}
