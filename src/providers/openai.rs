//! OpenAI provider implementation.
//!
//! Supports OpenAI and OpenAI-compatible APIs (Ollama, LM Studio, etc.)

use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
        ChatCompletionTool, ChatCompletionToolChoiceOption, ChatCompletionToolType,
        CreateChatCompletionRequestArgs, CreateEmbeddingRequestArgs, EmbeddingInput, FinishReason,
        FunctionObjectArgs, ImageDetail, ImageUrl,
    },
    Client,
};
use async_trait::async_trait;
use futures::StreamExt;
use std::collections::HashMap;
use tracing::debug;

use crate::error::{LlmError, Result};
use crate::traits::ImageData;
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
                    ChatRole::User => {
                        let content = Self::build_user_content(msg);
                        ChatCompletionRequestUserMessageArgs::default()
                            .content(content)
                            .build()
                            .map(Into::into)
                            .map_err(|e| LlmError::InvalidRequest(e.to_string()))
                    }
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

    /// Build user message content, supporting multimodal (text + images).
    ///
    /// WHY: Vision-capable OpenAI models (gpt-4o, gpt-4-vision-preview, etc.) require
    /// content to be an array of typed parts when images are present. This function
    /// detects image presence and builds the appropriate content representation.
    fn build_user_content(msg: &ChatMessage) -> ChatCompletionRequestUserMessageContent {
        if msg.has_images() {
            let mut parts: Vec<ChatCompletionRequestUserMessageContentPart> = Vec::new();

            // Add text part first (if non-empty)
            if !msg.content.is_empty() {
                parts.push(ChatCompletionRequestUserMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText {
                        text: msg.content.clone(),
                    },
                ));
            }

            // Add image parts
            if let Some(ref images) = msg.images {
                for img in images {
                    let detail = Self::parse_image_detail(img);
                    parts.push(ChatCompletionRequestUserMessageContentPart::ImageUrl(
                        ChatCompletionRequestMessageContentPartImage {
                            image_url: ImageUrl {
                                url: img.to_data_uri(),
                                detail,
                            },
                        },
                    ));
                }
            }

            ChatCompletionRequestUserMessageContent::Array(parts)
        } else {
            ChatCompletionRequestUserMessageContent::Text(msg.content.clone())
        }
    }

    /// Parse image detail level from ImageData.
    fn parse_image_detail(img: &ImageData) -> Option<ImageDetail> {
        match img.detail.as_deref() {
            Some("low") => Some(ImageDetail::Low),
            Some("high") => Some(ImageDetail::High),
            Some("auto") => Some(ImageDetail::Auto),
            _ => None,
        }
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

    // ---- Iteration 27: Additional OpenAI tests ----

    #[test]
    fn test_context_length_gpt5_series() {
        assert_eq!(
            OpenAIProvider::context_length_for_model("gpt-5.2-turbo"),
            200000
        );
        assert_eq!(
            OpenAIProvider::context_length_for_model("gpt-5.1-preview"),
            200000
        );
        assert_eq!(
            OpenAIProvider::context_length_for_model("gpt-5-nano"),
            128000
        );
        assert_eq!(
            OpenAIProvider::context_length_for_model("gpt-5-mini"),
            200000
        );
        assert_eq!(OpenAIProvider::context_length_for_model("gpt-5"), 200000);
    }

    #[test]
    fn test_context_length_o_series() {
        assert_eq!(OpenAIProvider::context_length_for_model("o4-mini"), 200000);
        assert_eq!(
            OpenAIProvider::context_length_for_model("o3-preview"),
            200000
        );
        assert_eq!(
            OpenAIProvider::context_length_for_model("o1-preview"),
            200000
        );
    }

    #[test]
    fn test_context_length_gpt4_variants() {
        assert_eq!(
            OpenAIProvider::context_length_for_model("gpt-4-turbo-preview"),
            128000
        );
        assert_eq!(
            OpenAIProvider::context_length_for_model("gpt-4-32k-0613"),
            32768
        );
        assert_eq!(OpenAIProvider::context_length_for_model("gpt-4-0613"), 8192);
    }

    #[test]
    fn test_context_length_gpt35_variants() {
        assert_eq!(
            OpenAIProvider::context_length_for_model("gpt-3.5-turbo-16k"),
            16384
        );
        assert_eq!(
            OpenAIProvider::context_length_for_model("gpt-3.5-turbo-1106"),
            4096
        );
    }

    #[test]
    fn test_context_length_unknown_defaults_high() {
        // Unknown models default to 128K (newer default)
        assert_eq!(
            OpenAIProvider::context_length_for_model("unknown-future-model"),
            128000
        );
    }

    #[test]
    fn test_dimension_ada_model() {
        assert_eq!(
            OpenAIProvider::dimension_for_model("text-embedding-ada-002"),
            1536
        );
    }

    #[test]
    fn test_dimension_unknown_defaults() {
        assert_eq!(
            OpenAIProvider::dimension_for_model("unknown-embedding"),
            1536
        );
    }

    #[test]
    fn test_provider_name() {
        let provider = OpenAIProvider::new("test-key");
        assert_eq!(LLMProvider::name(&provider), "openai");
    }

    #[test]
    fn test_provider_max_context_length() {
        let provider = OpenAIProvider::new("test-key").with_model("gpt-4");
        assert_eq!(provider.max_context_length(), 8192);
    }

    #[test]
    fn test_provider_dimension() {
        let provider =
            OpenAIProvider::new("test-key").with_embedding_model("text-embedding-3-large");
        assert_eq!(provider.dimension(), 3072);
    }

    #[test]
    fn test_provider_embedding_model() {
        let provider =
            OpenAIProvider::new("test-key").with_embedding_model("text-embedding-3-small");
        assert_eq!(
            EmbeddingProvider::model(&provider),
            "text-embedding-3-small"
        );
    }

    #[test]
    fn test_message_conversion_tool_role() {
        let messages = vec![ChatMessage::tool_result("call_1", "result data")];
        // Tool messages convert to user messages in simplified API
        let converted = OpenAIProvider::convert_messages(&messages).unwrap();
        assert_eq!(converted.len(), 1);
    }

    #[test]
    fn test_supports_streaming() {
        let provider = OpenAIProvider::new("test-key");
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_supports_json_mode_gpt4() {
        let provider = OpenAIProvider::new("test-key").with_model("gpt-4o");
        assert!(provider.supports_json_mode());
    }

    #[test]
    fn test_supports_json_mode_gpt35() {
        let provider = OpenAIProvider::new("test-key").with_model("gpt-3.5-turbo");
        assert!(provider.supports_json_mode());
    }

    #[test]
    fn test_supports_json_mode_default_is_false() {
        // Default model is gpt-5-mini which doesn't have "gpt-4" or "gpt-3.5-turbo" in name
        let provider = OpenAIProvider::new("test-key");
        assert!(!provider.supports_json_mode());
    }

    // ---- Vision / multimodal message tests ----

    #[test]
    fn test_build_user_content_text_only() {
        let msg = ChatMessage::user("Hello");
        let content = OpenAIProvider::build_user_content(&msg);
        match content {
            ChatCompletionRequestUserMessageContent::Text(t) => assert_eq!(t, "Hello"),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn test_build_user_content_with_image() {
        use crate::traits::ImageData;
        let img = ImageData::new("base64data", "image/png");
        let msg = ChatMessage::user_with_images("Describe this", vec![img]);
        let content = OpenAIProvider::build_user_content(&msg);
        match content {
            ChatCompletionRequestUserMessageContent::Array(parts) => {
                assert_eq!(parts.len(), 2, "Should have text + image parts");
                assert!(
                    matches!(
                        parts[0],
                        ChatCompletionRequestUserMessageContentPart::Text(_)
                    ),
                    "First part should be text"
                );
                assert!(
                    matches!(
                        parts[1],
                        ChatCompletionRequestUserMessageContentPart::ImageUrl(_)
                    ),
                    "Second part should be image_url"
                );
            }
            _ => panic!("Expected array content for vision message"),
        }
    }

    #[test]
    fn test_build_user_content_image_data_uri() {
        use crate::traits::ImageData;
        let img = ImageData::new("abc123", "image/jpeg");
        let msg = ChatMessage::user_with_images("What's here?", vec![img]);
        let content = OpenAIProvider::build_user_content(&msg);
        if let ChatCompletionRequestUserMessageContent::Array(parts) = content {
            if let ChatCompletionRequestUserMessageContentPart::ImageUrl(img_part) = &parts[1] {
                assert_eq!(
                    img_part.image_url.url, "data:image/jpeg;base64,abc123",
                    "Data URI should be correct"
                );
            } else {
                panic!("Expected ImageUrl part");
            }
        } else {
            panic!("Expected array content");
        }
    }

    #[test]
    fn test_build_user_content_image_with_detail() {
        use crate::traits::ImageData;
        let img = ImageData::new("data", "image/png").with_detail("high");
        let _msg = ChatMessage::user_with_images("Analyze", vec![img]);
        let detail = OpenAIProvider::parse_image_detail(
            &ImageData::new("x", "image/png").with_detail("high"),
        );
        assert!(matches!(detail, Some(ImageDetail::High)));
    }

    #[test]
    fn test_parse_image_detail_low() {
        use crate::traits::ImageData;
        let img = ImageData::new("x", "image/png").with_detail("low");
        let d = OpenAIProvider::parse_image_detail(&img);
        assert!(matches!(d, Some(ImageDetail::Low)));
    }

    #[test]
    fn test_parse_image_detail_auto() {
        use crate::traits::ImageData;
        let img = ImageData::new("x", "image/png").with_detail("auto");
        let d = OpenAIProvider::parse_image_detail(&img);
        assert!(matches!(d, Some(ImageDetail::Auto)));
    }

    #[test]
    fn test_parse_image_detail_none() {
        use crate::traits::ImageData;
        let img = ImageData::new("x", "image/png");
        let d = OpenAIProvider::parse_image_detail(&img);
        assert!(d.is_none());
    }

    #[test]
    fn test_convert_messages_with_image_produces_array_content() {
        use crate::traits::ImageData;
        let img = ImageData::new("iVBORw0KGgo", "image/png");
        let messages = vec![
            ChatMessage::system("You are a vision assistant"),
            ChatMessage::user_with_images("What is in this image?", vec![img]),
        ];
        let converted = OpenAIProvider::convert_messages(&messages).unwrap();
        assert_eq!(converted.len(), 2);
        // Verify the user message is a ChatCompletionRequestMessage::User with Array content
        // We can validate via JSON serialization
        let json = serde_json::to_value(&converted[1]).unwrap();
        let content = &json["content"];
        assert!(
            content.is_array(),
            "Vision user message content must be a JSON array, got: {:?}",
            content
        );
        let parts = content.as_array().unwrap();
        assert_eq!(parts.len(), 2, "Should have text + image parts");
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image_url");
        assert!(parts[1]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,"));
    }

    #[test]
    fn test_convert_messages_without_image_produces_text_content() {
        let messages = vec![ChatMessage::user("Just text")];
        let converted = OpenAIProvider::convert_messages(&messages).unwrap();
        let json = serde_json::to_value(&converted[0]).unwrap();
        let content = &json["content"];
        assert!(
            content.is_string(),
            "Plain text user message content must be a JSON string"
        );
        assert_eq!(content.as_str().unwrap(), "Just text");
    }
}
