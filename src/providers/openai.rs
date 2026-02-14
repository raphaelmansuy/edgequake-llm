//! OpenAI provider implementation.
//!
//! Supports OpenAI and OpenAI-compatible APIs (Ollama, LM Studio, etc.)

use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        ChatCompletionTool, ChatCompletionToolChoiceOption, ChatCompletionToolType,
        CreateChatCompletionRequestArgs, CreateEmbeddingRequestArgs, EmbeddingInput,
        FinishReason, FunctionObjectArgs,
    },
    Client,
};
use async_trait::async_trait;
use futures::StreamExt;
use std::collections::HashMap;
use tracing::debug;

use crate::error::{LlmError, Result};
use crate::traits::{
    ChatMessage, ChatRole, CompletionOptions, EmbeddingProvider, LLMProvider, LLMResponse,
    StreamChunk, ToolChoice, ToolDefinition,
};

/// OpenAI provider for text completion and embeddings.
pub struct OpenAIProvider {
    client: Client<OpenAIConfig>,
    model: String,
    embedding_model: String,
    max_context_length: usize,
    embedding_dimension: usize,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        let config = OpenAIConfig::new().with_api_key(api_key);
        Self::with_config(config)
    }

    /// Create a provider with custom configuration.
    /// Defaults to GPT-5-mini for best balance of performance and cost.
    pub fn with_config(config: OpenAIConfig) -> Self {
        Self {
            client: Client::with_config(config),
            model: "gpt-5-mini".to_string(), // Updated default to GPT-5-mini
            embedding_model: "text-embedding-3-small".to_string(),
            max_context_length: 200000, // GPT-5-mini context length
            embedding_dimension: 1536,
        }
    }

    /// Create a provider for an OpenAI-compatible API.
    pub fn compatible(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        let config = OpenAIConfig::new()
            .with_api_key(api_key)
            .with_api_base(base_url);
        Self::with_config(config)
    }

    /// Set the completion model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self.max_context_length = Self::context_length_for_model(&self.model);
        self
    }

    /// Set the embedding model.
    pub fn with_embedding_model(mut self, model: impl Into<String>) -> Self {
        self.embedding_model = model.into();
        self.embedding_dimension = Self::dimension_for_model(&self.embedding_model);
        self
    }

    /// Get the context length for a model.
    fn context_length_for_model(model: &str) -> usize {
        match model {
            // GPT-5 series (2026 models)
            m if m.contains("gpt-5.2") || m.contains("gpt-5.1") => 200000,
            m if m.contains("gpt-5-nano") => 128000,
            m if m.contains("gpt-5-mini") || m.contains("gpt-5") => 200000,

            // GPT-4.1 series
            m if m.contains("gpt-4.1") => 128000,

            // O-series reasoning models
            m if m.contains("o4") || m.contains("o3") => 200000,
            m if m.contains("o1") => 200000,

            // GPT-4 series
            m if m.contains("gpt-4o") => 128000,
            m if m.contains("gpt-4-turbo") => 128000,
            m if m.contains("gpt-4-32k") => 32768,
            m if m.contains("gpt-4") => 8192,

            // GPT-3.5 series
            m if m.contains("gpt-3.5-turbo-16k") => 16384,
            m if m.contains("gpt-3.5") => 4096,

            // Codex models
            m if m.contains("codex") => 200000,

            // Realtime and audio models
            m if m.contains("gpt-realtime") || m.contains("gpt-audio") => 128000,

            _ => 128000, // Updated default for newer models
        }
    }

    /// Get the embedding dimension for a model.
    fn dimension_for_model(model: &str) -> usize {
        match model {
            m if m.contains("text-embedding-3-large") => 3072,
            m if m.contains("text-embedding-3-small") => 1536,
            m if m.contains("text-embedding-ada") => 1536,
            _ => 1536, // Default
        }
    }

    /// Convert chat messages to OpenAI format.
    fn convert_messages(messages: &[ChatMessage]) -> Result<Vec<ChatCompletionRequestMessage>> {
        messages
            .iter()
            .map(|msg| {
                match msg.role {
                    ChatRole::System => ChatCompletionRequestSystemMessageArgs::default()
                        .content(msg.content.as_str())
                        .build()
                        .map(Into::into)
                        .map_err(|e| LlmError::InvalidRequest(e.to_string())),
                    ChatRole::User => ChatCompletionRequestUserMessageArgs::default()
                        .content(msg.content.as_str())
                        .build()
                        .map(Into::into)
                        .map_err(|e| LlmError::InvalidRequest(e.to_string())),
                    ChatRole::Assistant => ChatCompletionRequestAssistantMessageArgs::default()
                        .content(msg.content.as_str())
                        .build()
                        .map(Into::into)
                        .map_err(|e| LlmError::InvalidRequest(e.to_string())),
                    ChatRole::Tool | ChatRole::Function => {
                        // Tool/Function messages are handled as user messages in simplified API
                        ChatCompletionRequestUserMessageArgs::default()
                            .content(msg.content.as_str())
                            .build()
                            .map(Into::into)
                            .map_err(|e| LlmError::InvalidRequest(e.to_string()))
                    }
                }
            })
            .collect()
    }
}

#[async_trait]
impl LLMProvider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
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
        let openai_messages = Self::convert_messages(messages)?;
        let options = options.cloned().unwrap_or_default();

        let mut request_builder = CreateChatCompletionRequestArgs::default();
        request_builder.model(&self.model).messages(openai_messages);

        if let Some(max_tokens) = options.max_tokens {
            request_builder.max_tokens(max_tokens as u32);
        }

        if let Some(temp) = options.temperature {
            request_builder.temperature(temp);
        }

        if let Some(top_p) = options.top_p {
            request_builder.top_p(top_p);
        }

        if let Some(stop) = options.stop {
            request_builder.stop(stop);
        }

        if let Some(freq_penalty) = options.frequency_penalty {
            request_builder.frequency_penalty(freq_penalty);
        }

        if let Some(pres_penalty) = options.presence_penalty {
            request_builder.presence_penalty(pres_penalty);
        }

        let request = request_builder
            .build()
            .map_err(|e| LlmError::InvalidRequest(e.to_string()))?;

        let response = self.client.chat().create(request).await?;

        // Debug logging for token tracking
        debug!(
            "OpenAI response - usage: {:?}, model: {}",
            response.usage, response.model
        );

        let choice = response
            .choices
            .first()
            .ok_or_else(|| LlmError::ApiError("No choices in response".to_string()))?;

        let content = choice.message.content.clone().unwrap_or_default();

        let usage = response
            .usage
            .clone()
            .unwrap_or(async_openai::types::CompletionUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            });

        // Note: Cache hit tracking for OpenAI requires async-openai >= 0.32
        // which adds prompt_tokens_details.cached_tokens field.
        // Current version (0.24) does not support cache hit extraction.
        // TODO: Upgrade async-openai when feasible for full cache tracking.
        let cache_hit_tokens: Option<usize> = None;

        // Log extracted token counts
        debug!(
            "OpenAI token usage - prompt: {}, completion: {}, total: {}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );

        let mut metadata = HashMap::new();
        metadata.insert("response_id".to_string(), serde_json::json!(response.id));

        Ok(LLMResponse {
            content,
            prompt_tokens: usage.prompt_tokens as usize,
            completion_tokens: usage.completion_tokens as usize,
            total_tokens: usage.total_tokens as usize,
            model: response.model,
            finish_reason: choice.finish_reason.map(|r| format!("{:?}", r)),
            tool_calls: Vec::new(),
            metadata,
            cache_hit_tokens,
            // OODA-15: Reasoning tokens not yet extracted (requires async-openai 0.32+)
            // TODO: Extract output_tokens_details.reasoning_tokens when library upgraded
            thinking_tokens: None,
            thinking_content: None,
        })
    }

    async fn stream(
        &self,
        prompt: &str,
    ) -> Result<futures::stream::BoxStream<'static, Result<String>>> {
        use futures::StreamExt;

        let request = ChatCompletionRequestUserMessageArgs::default()
            .content(prompt)
            .build()
            .map(Into::into)
            .map_err(|e| LlmError::InvalidRequest(e.to_string()))?;

        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages(vec![request])
            .stream(true)
            .build()
            .map_err(|e| LlmError::InvalidRequest(e.to_string()))?;

        let stream = self.client.chat().create_stream(request).await?;

        let mapped_stream = stream.map(|res| match res {
            Ok(response) => {
                let content = response
                    .choices
                    .first()
                    .and_then(|c| c.delta.content.clone())
                    .unwrap_or_default();
                Ok(content)
            }
            Err(e) => Err(LlmError::ApiError(e.to_string())),
        });

        Ok(mapped_stream.boxed())
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn chat_with_tools_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamChunk>>> {
        let openai_messages = Self::convert_messages(messages)?;
        let options = options.cloned().unwrap_or_default();

        // Convert tools to OpenAI format
        let openai_tools: Vec<ChatCompletionTool> = tools
            .iter()
            .map(|tool| ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: FunctionObjectArgs::default()
                    .name(&tool.function.name)
                    .description(&tool.function.description)
                    .parameters(tool.function.parameters.clone())
                    .build()
                    .expect("Invalid tool definition"),
            })
            .collect();

        // Build request
        let mut request_builder = CreateChatCompletionRequestArgs::default();
        request_builder
            .model(&self.model)
            .messages(openai_messages)
            .tools(openai_tools)
            .stream(true); // Enable streaming

        // Set tool choice if specified
        if let Some(tc) = tool_choice {
            match tc {
                ToolChoice::Auto(_) => {
                    request_builder.tool_choice(ChatCompletionToolChoiceOption::Auto);
                }
                ToolChoice::Required(_) => {
                    request_builder.tool_choice(ChatCompletionToolChoiceOption::Required);
                }
                _ => {}
            }
        }

        if let Some(temp) = options.temperature {
            request_builder.temperature(temp);
        }

        if let Some(max_tokens) = options.max_tokens {
            request_builder.max_tokens(max_tokens as u32);
        }

        let request = request_builder
            .build()
            .map_err(|e| LlmError::InvalidRequest(e.to_string()))?;

        let stream = self.client.chat().create_stream(request).await?;

        // Map OpenAI stream to our StreamChunk format
        let mapped_stream = stream.map(|result| {
            match result {
                Ok(response) => {
                    let choice = response.choices.first();
                    if let Some(choice) = choice {
                        // Stream content chunks immediately
                        if let Some(content) = &choice.delta.content {
                            return Ok(StreamChunk::Content(content.clone()));
                        }

                        // Return tool call delta (agent will accumulate)
                        if let Some(tool_call_chunks) = &choice.delta.tool_calls {
                            if let Some(chunk) = tool_call_chunks.first() {
                                return Ok(StreamChunk::ToolCallDelta {
                                    index: chunk.index as usize,
                                    id: chunk.id.clone(),
                                    function_name: chunk
                                        .function
                                        .as_ref()
                                        .and_then(|f| f.name.clone()),
                                    function_arguments: chunk
                                        .function
                                        .as_ref()
                                        .and_then(|f| f.arguments.clone()),
                                });
                            }
                        }

                        // Check finish reason
                        if let Some(finish_reason) = &choice.finish_reason {
                            let reason = match finish_reason {
                                FinishReason::Stop => "stop",
                                FinishReason::Length => "length",
                                FinishReason::ToolCalls => "tool_calls",
                                FinishReason::ContentFilter => "content_filter",
                                FinishReason::FunctionCall => "function_call",
                            };
                            return Ok(StreamChunk::Finished {
                                reason: reason.to_string(),
                                ttft_ms: None,
                            });
                        }
                    }
                    // Empty chunk (no content or tool calls)
                    Ok(StreamChunk::Content(String::new()))
                }
                Err(e) => Err(LlmError::ApiError(e.to_string())),
            }
        });

        Ok(mapped_stream.boxed())
    }

    fn supports_tool_streaming(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        self.model.contains("gpt-4") || self.model.contains("gpt-3.5-turbo")
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    /// Returns the embedding model name (not completion model).
    ///
    /// Note: `model` field refers to completion, `embedding_model` is for embeddings.
    #[allow(clippy::misnamed_getters)]
    fn model(&self) -> &str {
        &self.embedding_model
    }

    fn dimension(&self) -> usize {
        self.embedding_dimension
    }

    fn max_tokens(&self) -> usize {
        8191 // OpenAI embedding models support 8191 tokens
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let input = EmbeddingInput::StringArray(texts.to_vec());

        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.embedding_model)
            .input(input)
            .build()
            .map_err(|e| LlmError::InvalidRequest(e.to_string()))?;

        let response = self.client.embeddings().create(request).await?;

        let embeddings: Vec<Vec<f32>> = response.data.into_iter().map(|e| e.embedding).collect();

        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_length_detection() {
        assert_eq!(OpenAIProvider::context_length_for_model("gpt-4o"), 128000);
        assert_eq!(OpenAIProvider::context_length_for_model("gpt-4"), 8192);
        assert_eq!(
            OpenAIProvider::context_length_for_model("gpt-3.5-turbo"),
            4096
        );
    }

    #[test]
    fn test_embedding_dimension_detection() {
        assert_eq!(
            OpenAIProvider::dimension_for_model("text-embedding-3-large"),
            3072
        );
        assert_eq!(
            OpenAIProvider::dimension_for_model("text-embedding-3-small"),
            1536
        );
    }

    #[test]
    fn test_provider_builder() {
        let provider = OpenAIProvider::new("test-key")
            .with_model("gpt-4")
            .with_embedding_model("text-embedding-3-large");

        assert_eq!(LLMProvider::model(&provider), "gpt-4");
        assert_eq!(provider.dimension(), 3072);
    }

    #[test]
    fn test_message_conversion() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
        ];

        let converted = OpenAIProvider::convert_messages(&messages).unwrap();
        assert_eq!(converted.len(), 3);
    }
}
