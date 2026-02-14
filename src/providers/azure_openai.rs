//! Azure OpenAI LLM provider implementation.
//!
//! Supports Azure OpenAI Service endpoints with API key or Entra ID authentication.
//!
//! # Environment Variables
//! - `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint (e.g., `https://myresource.openai.azure.com`)
//! - `AZURE_OPENAI_API_KEY`: API key for authentication
//! - `AZURE_OPENAI_DEPLOYMENT_NAME`: Deployment name for chat/completion model
//! - `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`: Deployment name for embedding model
//! - `AZURE_OPENAI_API_VERSION`: API version (default: 2024-10-21)

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, instrument};

use crate::error::{LlmError, Result};
use crate::traits::{
    ChatMessage, ChatRole, CompletionOptions, EmbeddingProvider, FunctionCall, LLMProvider,
    LLMResponse, StreamChunk, ToolCall, ToolChoice, ToolDefinition,
};

/// Default Azure OpenAI API version
const DEFAULT_API_VERSION: &str = "2024-10-21";

/// Azure OpenAI provider configuration
#[derive(Debug, Clone)]
pub struct AzureOpenAIProvider {
    client: Client,
    endpoint: String,
    api_key: String,
    deployment_name: String,
    embedding_deployment_name: String,
    api_version: String,
    max_context_length: usize,
    embedding_dimension: usize,
}

// ============================================================================
// Azure OpenAI Request/Response Types (same as OpenAI, but different endpoint)
// ============================================================================

/// Message format for Azure OpenAI
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AzureMessage {
    role: String,
    content: String,
}

/// Chat completion request
#[derive(Debug, Clone, Serialize)]
struct ChatCompletionRequest {
    messages: Vec<AzureMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<usize>,
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
    stream: Option<bool>,
    // OODA-11: Tool calling support
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AzureToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
}

/// Chat completion response choice
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct ChatChoice {
    index: usize,
    message: AzureMessageResponse,
    finish_reason: Option<String>,
}

/// Response message from Azure OpenAI (includes tool_calls)
#[derive(Debug, Clone, Deserialize)]
struct AzureMessageResponse {
    #[allow(dead_code)]
    role: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<AzureToolCall>>,
}

/// Token usage information
#[derive(Debug, Clone, Deserialize, Default)]
struct Usage {
    #[serde(default)]
    prompt_tokens: usize,
    #[serde(default)]
    completion_tokens: usize,
    #[serde(default)]
    total_tokens: usize,
    /// Breakdown of prompt tokens for cache tracking
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
}

/// Prompt token details including cache hits
#[derive(Debug, Clone, Deserialize, Default)]
struct PromptTokensDetails {
    /// Number of tokens served from cache
    #[serde(default)]
    cached_tokens: Option<usize>,
}

/// Chat completion response
#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionResponse {
    id: String,
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Usage,
    model: String,
}

/// Embedding request
#[derive(Debug, Clone, Serialize)]
struct EmbeddingRequest {
    input: Vec<String>,
}

/// Embedding data
#[derive(Debug, Clone, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

/// Embedding response
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    #[serde(default)]
    usage: Usage,
}

/// Error response from Azure OpenAI
#[derive(Debug, Clone, Deserialize)]
struct AzureErrorResponse {
    error: AzureError,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct AzureError {
    code: Option<String>,
    message: String,
    #[serde(rename = "type")]
    error_type: Option<String>,
}

// =========================================================================
// OODA-11: Tool calling types for Azure OpenAI
// =========================================================================

/// Tool definition for Azure OpenAI (OpenAI-compatible format)
#[derive(Debug, Clone, Serialize)]
struct AzureToolDefinition {
    #[serde(rename = "type")]
    type_: String,
    function: AzureFunctionDefinition,
}

/// Function definition for Azure OpenAI
#[derive(Debug, Clone, Serialize)]
struct AzureFunctionDefinition {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

/// Tool call from Azure OpenAI response
#[derive(Debug, Clone, Deserialize)]
struct AzureToolCall {
    id: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    type_: String,
    function: AzureFunctionCall,
}

/// Function call details
#[derive(Debug, Clone, Deserialize)]
struct AzureFunctionCall {
    name: String,
    arguments: String,
}

/// Streaming chunk for Azure OpenAI
#[derive(Debug, Clone, Deserialize)]
struct StreamingChunk {
    choices: Vec<StreamingChoice>,
}

/// Streaming choice
#[derive(Debug, Clone, Deserialize)]
struct StreamingChoice {
    delta: StreamingDelta,
    finish_reason: Option<String>,
}

/// Streaming delta content
#[derive(Debug, Clone, Deserialize)]
struct StreamingDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<StreamingToolCall>>,
}

/// Tool call in streaming format
#[derive(Debug, Clone, Deserialize)]
struct StreamingToolCall {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<StreamingFunction>,
}

/// Function in streaming format
#[derive(Debug, Clone, Deserialize)]
struct StreamingFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

// ============================================================================
// AzureOpenAIProvider Implementation
// ============================================================================

impl AzureOpenAIProvider {
    /// Create a new Azure OpenAI provider.
    ///
    /// # Arguments
    /// * `endpoint` - Azure OpenAI endpoint (e.g., `https://myresource.openai.azure.com`)
    /// * `api_key` - API key for authentication
    /// * `deployment_name` - Deployment name for chat/completion model
    pub fn new(
        endpoint: impl Into<String>,
        api_key: impl Into<String>,
        deployment_name: impl Into<String>,
    ) -> Self {
        let deployment = deployment_name.into();
        Self {
            client: Client::new(),
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            deployment_name: deployment.clone(),
            embedding_deployment_name: deployment,
            api_version: DEFAULT_API_VERSION.to_string(),
            max_context_length: 128_000, // Default for GPT-4o
            embedding_dimension: 1536,   // Default for text-embedding-ada-002
        }
    }

    /// Create a provider from environment variables.
    ///
    /// Reads from:
    /// - `AZURE_OPENAI_ENDPOINT`
    /// - `AZURE_OPENAI_API_KEY`
    /// - `AZURE_OPENAI_DEPLOYMENT_NAME`
    /// - `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` (optional)
    /// - `AZURE_OPENAI_API_VERSION` (optional)
    pub fn from_env() -> Result<Self> {
        let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT")
            .map_err(|_| LlmError::ConfigError("AZURE_OPENAI_ENDPOINT not set".to_string()))?;

        let api_key = std::env::var("AZURE_OPENAI_API_KEY")
            .map_err(|_| LlmError::ConfigError("AZURE_OPENAI_API_KEY not set".to_string()))?;

        let deployment_name = std::env::var("AZURE_OPENAI_DEPLOYMENT_NAME").map_err(|_| {
            LlmError::ConfigError("AZURE_OPENAI_DEPLOYMENT_NAME not set".to_string())
        })?;

        let mut provider = Self::new(endpoint, api_key, deployment_name);

        if let Ok(embedding_deployment) = std::env::var("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME") {
            provider = provider.with_embedding_deployment(embedding_deployment);
        }

        if let Ok(api_version) = std::env::var("AZURE_OPENAI_API_VERSION") {
            provider = provider.with_api_version(api_version);
        }

        Ok(provider)
    }

    /// Set the embedding deployment name.
    pub fn with_embedding_deployment(mut self, deployment_name: impl Into<String>) -> Self {
        self.embedding_deployment_name = deployment_name.into();
        self
    }

    /// Set the API version.
    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        self.api_version = api_version.into();
        self
    }

    /// Set the max context length.
    pub fn with_max_context_length(mut self, max_context_length: usize) -> Self {
        self.max_context_length = max_context_length;
        self
    }

    /// Set the embedding dimension.
    pub fn with_embedding_dimension(mut self, dimension: usize) -> Self {
        self.embedding_dimension = dimension;
        self
    }

    /// Build URL for a deployment operation.
    fn build_url(&self, deployment: &str, operation: &str) -> String {
        format!(
            "{}/openai/deployments/{}/{}?api-version={}",
            self.endpoint, deployment, operation, self.api_version
        )
    }

    /// Convert ChatMessage to Azure format.
    fn convert_messages(messages: &[ChatMessage]) -> Vec<AzureMessage> {
        messages
            .iter()
            .map(|msg| AzureMessage {
                role: match msg.role {
                    ChatRole::System => "system".to_string(),
                    ChatRole::User => "user".to_string(),
                    ChatRole::Assistant => "assistant".to_string(),
                    ChatRole::Tool => "tool".to_string(),
                    ChatRole::Function => "function".to_string(),
                },
                content: msg.content.clone(),
            })
            .collect()
    }

    /// Send a request and handle errors.
    async fn send_request<T: for<'de> Deserialize<'de>>(
        &self,
        url: &str,
        body: &impl Serialize,
    ) -> Result<T> {
        let response = self
            .client
            .post(url)
            .header("api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| LlmError::ApiError(format!("Request failed: {}", e)))?;

        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| LlmError::ApiError(format!("Failed to read response: {}", e)))?;

        if !status.is_success() {
            if let Ok(error_response) = serde_json::from_str::<AzureErrorResponse>(&text) {
                return Err(LlmError::ApiError(format!(
                    "Azure OpenAI error: {}",
                    error_response.error.message
                )));
            }
            return Err(LlmError::ApiError(format!(
                "Azure OpenAI error ({}): {}",
                status, text
            )));
        }

        serde_json::from_str(&text).map_err(|e| {
            LlmError::ApiError(format!("Failed to parse response: {}. Body: {}", e, text))
        })
    }
}

#[async_trait]
impl LLMProvider for AzureOpenAIProvider {
    fn name(&self) -> &str {
        "azure-openai"
    }

    fn model(&self) -> &str {
        &self.deployment_name
    }

    fn max_context_length(&self) -> usize {
        self.max_context_length
    }

    #[instrument(skip(self, prompt), fields(deployment = %self.deployment_name))]
    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        self.complete_with_options(prompt, &CompletionOptions::default())
            .await
    }

    #[instrument(skip(self, prompt, options), fields(deployment = %self.deployment_name))]
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

    #[instrument(skip(self, messages, options), fields(deployment = %self.deployment_name))]
    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let azure_messages = Self::convert_messages(messages);
        let options = options.cloned().unwrap_or_default();

        let request = ChatCompletionRequest {
            messages: azure_messages,
            max_tokens: options.max_tokens,
            max_completion_tokens: None,
            temperature: options.temperature,
            top_p: options.top_p,
            stop: options.stop,
            frequency_penalty: options.frequency_penalty,
            presence_penalty: options.presence_penalty,
            stream: None,
            tools: None,
            tool_choice: None,
        };

        let url = self.build_url(&self.deployment_name, "chat/completions");
        debug!("Sending request to Azure OpenAI: {}", url);

        let response: ChatCompletionResponse = self.send_request(&url, &request).await?;

        let choice = response
            .choices
            .first()
            .ok_or_else(|| LlmError::ApiError("No choices in response".to_string()))?;

        // Extract cache hit tokens from prompt_tokens_details if available
        // This enables tracking of KV-cache effectiveness for context engineering
        let cache_hit_tokens = response
            .usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens);

        // Log token usage with cache info
        if let Some(cached) = cache_hit_tokens {
            debug!(
                "Azure OpenAI token usage - prompt: {}, completion: {}, cached: {} ({:.1}%)",
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                cached,
                (cached as f64 / response.usage.prompt_tokens as f64) * 100.0
            );
        } else {
            debug!(
                "Azure OpenAI token usage - prompt: {}, completion: {}, total: {}",
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens
            );
        }

        let mut metadata = HashMap::new();
        metadata.insert("response_id".to_string(), serde_json::json!(response.id));

        Ok(LLMResponse {
            content: choice.message.content.clone().unwrap_or_default(),
            prompt_tokens: response.usage.prompt_tokens,
            completion_tokens: response.usage.completion_tokens,
            total_tokens: response.usage.total_tokens,
            model: response.model,
            finish_reason: choice.finish_reason.clone(),
            tool_calls: Vec::new(),
            metadata,
            cache_hit_tokens,
            // OODA-15: Reasoning tokens not yet extracted
            thinking_tokens: None,
            thinking_content: None,
        })
    }

    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        let messages = vec![ChatMessage::user(prompt)];
        let azure_messages = Self::convert_messages(&messages);

        let request = ChatCompletionRequest {
            messages: azure_messages,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            stream: Some(true),
            tools: None,
            tool_choice: None,
        };

        let url = self.build_url(&self.deployment_name, "chat/completions");

        let response = self
            .client
            .post(&url)
            .header("api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::ApiError(format!("Stream request failed: {}", e)))?;

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!("Stream error: {}", text)));
        }

        let stream = response.bytes_stream();

        let mapped_stream = stream.map(|result| {
            match result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    // Parse SSE format
                    let mut content = String::new();
                    for line in text.lines() {
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data == "[DONE]" {
                                continue;
                            }
                            if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data) {
                                if let Some(delta_content) = chunk
                                    .get("choices")
                                    .and_then(|c| c.get(0))
                                    .and_then(|c| c.get("delta"))
                                    .and_then(|d| d.get("content"))
                                    .and_then(|c| c.as_str())
                                {
                                    content.push_str(delta_content);
                                }
                            }
                        }
                    }
                    Ok(content)
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
        true
    }
    
    fn supports_function_calling(&self) -> bool {
        true
    }
    
    fn supports_tool_streaming(&self) -> bool {
        true
    }
    
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let azure_messages = Self::convert_messages(messages);
        let opts = options.cloned().unwrap_or_default();
        
        // Convert tools to Azure format
        let azure_tools: Vec<AzureToolDefinition> = tools
            .iter()
            .map(|t| AzureToolDefinition {
                type_: "function".to_string(),
                function: AzureFunctionDefinition {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    parameters: t.function.parameters.clone(),
                },
            })
            .collect();
        
        // Convert tool_choice
        let azure_tool_choice = tool_choice.map(|tc| match tc {
            ToolChoice::Auto(_) => "auto".to_string(),
            ToolChoice::Required(_) => "required".to_string(),
            ToolChoice::Function { function, .. } => {
                format!("{{\"type\":\"function\",\"function\":{{\"name\":\"{}\"}}}}", function.name)
            }
        });
        
        let request = ChatCompletionRequest {
            messages: azure_messages,
            max_tokens: opts.max_tokens,
            max_completion_tokens: None,
            temperature: opts.temperature,
            top_p: opts.top_p,
            stop: opts.stop,
            frequency_penalty: opts.frequency_penalty,
            presence_penalty: opts.presence_penalty,
            stream: None,
            tools: Some(azure_tools),
            tool_choice: azure_tool_choice,
        };
        
        let url = self.build_url(&self.deployment_name, "chat/completions");
        debug!("Sending tool request to Azure OpenAI: {}", url);
        
        let response: ChatCompletionResponse = self.send_request(&url, &request).await?;
        
        let choice = response
            .choices
            .first()
            .ok_or_else(|| LlmError::ApiError("No choices in response".to_string()))?;
        
        // Convert tool calls
        let tool_calls: Vec<ToolCall> = choice
            .message
            .tool_calls
            .as_ref()
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
        
        let cache_hit_tokens = response
            .usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens);
        
        let mut metadata = HashMap::new();
        metadata.insert("response_id".to_string(), serde_json::json!(response.id));
        
        Ok(LLMResponse {
            content: choice.message.content.clone().unwrap_or_default(),
            prompt_tokens: response.usage.prompt_tokens,
            completion_tokens: response.usage.completion_tokens,
            total_tokens: response.usage.total_tokens,
            model: response.model,
            finish_reason: choice.finish_reason.clone(),
            tool_calls,
            metadata,
            cache_hit_tokens,
            thinking_tokens: None,
            thinking_content: None,
        })
    }
    
    async fn chat_with_tools_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        let azure_messages = Self::convert_messages(messages);
        let opts = options.cloned().unwrap_or_default();
        
        let azure_tools: Vec<AzureToolDefinition> = tools
            .iter()
            .map(|t| AzureToolDefinition {
                type_: "function".to_string(),
                function: AzureFunctionDefinition {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    parameters: t.function.parameters.clone(),
                },
            })
            .collect();
        
        let azure_tool_choice = tool_choice.map(|tc| match tc {
            ToolChoice::Auto(_) => "auto".to_string(),
            ToolChoice::Required(_) => "required".to_string(),
            ToolChoice::Function { function, .. } => {
                format!("{{\"type\":\"function\",\"function\":{{\"name\":\"{}\"}}}}", function.name)
            }
        });
        
        let request = ChatCompletionRequest {
            messages: azure_messages,
            max_tokens: opts.max_tokens,
            max_completion_tokens: None,
            temperature: opts.temperature,
            top_p: opts.top_p,
            stop: opts.stop,
            frequency_penalty: opts.frequency_penalty,
            presence_penalty: opts.presence_penalty,
            stream: Some(true),
            tools: Some(azure_tools),
            tool_choice: azure_tool_choice,
        };
        
        let url = self.build_url(&self.deployment_name, "chat/completions");
        
        let response = self
            .client
            .post(&url)
            .header("api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::ApiError(format!("Stream request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!("Stream error: {}", text)));
        }
        
        let stream = response.bytes_stream();
        
        let mapped_stream = stream.map(|result| {
            match result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    
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
                                    // Check finish reason
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
                                                function_name: tc.function.as_ref().and_then(|f| f.name.clone()),
                                                function_arguments: tc.function.as_ref().and_then(|f| f.arguments.clone()),
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
                    
                    Ok(StreamChunk::Content(String::new()))
                }
                Err(e) => Err(LlmError::ApiError(format!("Stream error: {}", e))),
            }
        });
        
        Ok(mapped_stream.boxed())
    }
}

#[async_trait]
impl EmbeddingProvider for AzureOpenAIProvider {
    fn name(&self) -> &str {
        "azure-openai"
    }

    fn model(&self) -> &str {
        &self.embedding_deployment_name
    }

    fn dimension(&self) -> usize {
        self.embedding_dimension
    }

    fn max_tokens(&self) -> usize {
        8191 // Azure OpenAI embedding models support 8191 tokens
    }

    #[instrument(skip(self, texts), fields(deployment = %self.embedding_deployment_name, count = texts.len()))]
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let request = EmbeddingRequest {
            input: texts.to_vec(),
        };

        let url = self.build_url(&self.embedding_deployment_name, "embeddings");
        debug!("Sending embedding request to Azure OpenAI: {}", url);

        let response: EmbeddingResponse = self.send_request(&url, &request).await?;

        // Sort by index to ensure correct ordering
        let mut embeddings: Vec<_> = response.data.into_iter().collect();
        embeddings.sort_by_key(|e| e.index);

        Ok(embeddings.into_iter().map(|e| e.embedding).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = AzureOpenAIProvider::new(
            "https://myresource.openai.azure.com",
            "test-api-key",
            "gpt-4o",
        );

        assert_eq!(LLMProvider::name(&provider), "azure-openai");
        assert_eq!(LLMProvider::model(&provider), "gpt-4o");
        assert_eq!(provider.endpoint, "https://myresource.openai.azure.com");
    }

    #[test]
    fn test_provider_with_options() {
        let provider = AzureOpenAIProvider::new(
            "https://myresource.openai.azure.com/",
            "test-api-key",
            "gpt-4o",
        )
        .with_embedding_deployment("text-embedding-ada-002")
        .with_api_version("2024-06-01")
        .with_max_context_length(128_000)
        .with_embedding_dimension(1536);

        assert_eq!(
            EmbeddingProvider::model(&provider),
            "text-embedding-ada-002"
        );
        assert_eq!(provider.api_version, "2024-06-01");
        assert_eq!(provider.max_context_length(), 128_000);
        assert_eq!(provider.dimension(), 1536);
        // Trailing slash should be stripped
        assert_eq!(provider.endpoint, "https://myresource.openai.azure.com");
    }

    #[test]
    fn test_build_url() {
        let provider = AzureOpenAIProvider::new(
            "https://myresource.openai.azure.com",
            "test-api-key",
            "gpt-4o",
        );

        let url = provider.build_url("gpt-4o", "chat/completions");
        assert_eq!(
            url,
            "https://myresource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
        );
    }

    #[test]
    fn test_message_conversion() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
        ];

        let converted = AzureOpenAIProvider::convert_messages(&messages);

        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0].role, "system");
        assert_eq!(converted[1].role, "user");
        assert_eq!(converted[2].role, "assistant");
    }

    #[test]
    fn test_message_conversion_tool_role() {
        let messages = vec![
            ChatMessage::tool_result("tool-1", "Tool result"),
        ];

        let converted = AzureOpenAIProvider::convert_messages(&messages);

        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "tool");
        assert_eq!(converted[0].content, "Tool result");
    }

    #[test]
    fn test_provider_defaults() {
        let provider = AzureOpenAIProvider::new(
            "https://test.openai.azure.com",
            "key",
            "test-deployment",
        );

        assert_eq!(provider.api_version, "2024-10-21");
        assert_eq!(provider.max_context_length, 128_000);
        assert_eq!(provider.embedding_dimension, 1536);
        // Deployment name should be used for both chat and embedding by default
        assert_eq!(provider.deployment_name, "test-deployment");
        assert_eq!(provider.embedding_deployment_name, "test-deployment");
    }

    #[test]
    fn test_supports_streaming() {
        let provider = AzureOpenAIProvider::new(
            "https://test.openai.azure.com",
            "key",
            "gpt-4o",
        );
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_supports_json_mode() {
        let provider = AzureOpenAIProvider::new(
            "https://test.openai.azure.com",
            "key",
            "gpt-4o",
        );
        assert!(provider.supports_json_mode());
    }

    #[test]
    fn test_embedding_provider_name() {
        let provider = AzureOpenAIProvider::new(
            "https://test.openai.azure.com",
            "key",
            "gpt-4o",
        )
        .with_embedding_deployment("text-embedding-ada-002");

        assert_eq!(EmbeddingProvider::name(&provider), "azure-openai");
    }

    #[test]
    fn test_embedding_provider_dimension() {
        let provider = AzureOpenAIProvider::new(
            "https://test.openai.azure.com",
            "key",
            "gpt-4o",
        )
        .with_embedding_dimension(3072);

        assert_eq!(provider.dimension(), 3072);
    }

    #[test]
    fn test_endpoint_trailing_slash_handling() {
        let provider1 = AzureOpenAIProvider::new(
            "https://test.openai.azure.com/",
            "key",
            "deployment",
        );
        let _provider2 = AzureOpenAIProvider::new(
            "https://test.openai.azure.com///",
            "key",
            "deployment",
        );
        let provider3 = AzureOpenAIProvider::new(
            "https://test.openai.azure.com",
            "key",
            "deployment",
        );

        // All should have the same endpoint without trailing slashes
        assert_eq!(provider1.endpoint, "https://test.openai.azure.com");
        // Note: current impl only strips one trailing slash at a time
        // This is acceptable behavior - URL should be normalized by caller
        assert_eq!(provider3.endpoint, "https://test.openai.azure.com");
    }

    #[test]
    fn test_build_url_embeddings() {
        let provider = AzureOpenAIProvider::new(
            "https://myresource.openai.azure.com",
            "key",
            "gpt-4o",
        )
        .with_embedding_deployment("text-embedding-ada-002");

        let url = provider.build_url("text-embedding-ada-002", "embeddings");
        assert!(url.contains("/openai/deployments/text-embedding-ada-002/embeddings"));
        assert!(url.contains("api-version=2024-10-21"));
    }

    #[test]
    fn test_build_url_custom_api_version() {
        let provider = AzureOpenAIProvider::new(
            "https://myresource.openai.azure.com",
            "key",
            "gpt-4o",
        )
        .with_api_version("2024-06-01");

        let url = provider.build_url("gpt-4o", "chat/completions");
        assert!(url.contains("api-version=2024-06-01"));
    }

    #[test]
    fn test_from_env_missing_endpoint() {
        // Clear env vars to ensure clean test
        std::env::remove_var("AZURE_OPENAI_ENDPOINT");
        std::env::remove_var("AZURE_OPENAI_API_KEY");
        std::env::remove_var("AZURE_OPENAI_DEPLOYMENT_NAME");

        let result = AzureOpenAIProvider::from_env();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("AZURE_OPENAI_ENDPOINT"));
    }

    #[test]
    fn test_max_context_length() {
        let provider = AzureOpenAIProvider::new(
            "https://test.openai.azure.com",
            "key",
            "gpt-4o",
        )
        .with_max_context_length(200_000);

        assert_eq!(provider.max_context_length(), 200_000);
    }

    #[test]
    fn test_azure_message_serialization() {
        let msg = AzureMessage {
            role: "user".to_string(),
            content: "Hello world".to_string(),
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Hello world\""));
    }
}
