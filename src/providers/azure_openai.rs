//! Azure OpenAI LLM provider implementation.
//!
//! Uses the official `async-openai` crate with `AzureConfig` so all
//! request/response types, streaming, tool-calling and embeddings are
//! handled identically to the OpenAI provider — only the HTTP plumbing
//! (endpoint, api-version query param, `api-key` header) differs.
//!
//! # Constructors
//!
//! | Method | When to use |
//! |--------|-------------|
//! | [`AzureOpenAIProvider::new`] | Programmatic |
//! | [`AzureOpenAIProvider::from_env`] | Standard `AZURE_OPENAI_*` vars |
//! | [`AzureOpenAIProvider::from_env_contentgen`] | Enterprise `AZURE_OPENAI_CONTENTGEN_*` vars |
//! | [`AzureOpenAIProvider::from_env_auto`] | Try CONTENTGEN first, fall back to standard |
//!
//! # Standard Environment Variables
//!
//! | Variable | Required | Default | Description |
//! |----------|----------|---------|-------------|
//! | `AZURE_OPENAI_ENDPOINT` | Yes | — | e.g. `https://myresource.openai.azure.com` |
//! | `AZURE_OPENAI_API_KEY` | Yes | — | Resource API key |
//! | `AZURE_OPENAI_DEPLOYMENT_NAME` | Yes | — | Chat/completion deployment |
//! | `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` | No | *(same as chat)* | Embedding deployment |
//! | `AZURE_OPENAI_API_VERSION` | No | `2024-10-21` | REST API version |
//!
//! # CONTENTGEN Environment Variables
//!
//! | Variable | Required | Description |
//! |----------|----------|-------------|
//! | `AZURE_OPENAI_CONTENTGEN_API_ENDPOINT` | Yes | Endpoint URL |
//! | `AZURE_OPENAI_CONTENTGEN_API_KEY` | Yes | API key |
//! | `AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT` | Yes | Chat deployment |
//! | `AZURE_OPENAI_CONTENTGEN_API_VERSION` | No | API version |

use async_openai::{
    config::{AzureConfig, Config},
    types::chat::{
        ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessageArgs,
        ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImage,
        ChatCompletionRequestMessageContentPartText, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, ChatCompletionRequestUserMessageContent,
        ChatCompletionRequestUserMessageContentPart, ChatCompletionTool,
        ChatCompletionToolChoiceOption, ChatCompletionTools, CompletionUsage,
        CreateChatCompletionRequestArgs, FinishReason, FunctionObjectArgs, ImageDetail, ImageUrl,
        ToolChoiceOptions,
    },
    types::embeddings::{CreateEmbeddingRequestArgs, EmbeddingInput},
    Client,
};
use async_trait::async_trait;
use futures::StreamExt;
use std::collections::HashMap;
use tracing::debug;

use crate::error::{LlmError, Result};
use crate::traits::FunctionCall as TraitFunctionCall;
use crate::traits::ToolCall;
use crate::traits::{
    ChatMessage, ChatRole, CompletionOptions, EmbeddingProvider, ImageData, LLMProvider,
    LLMResponse, StreamChunk, ToolChoice, ToolDefinition,
};

const DEFAULT_API_VERSION: &str = "2024-10-21";

// ============================================================================
// AzureOpenAIProvider
// ============================================================================

/// Azure OpenAI provider backed by `async-openai` `Client<AzureConfig>`.
///
/// Two clients are kept so chat and embeddings can target different deployments
/// (Azure scopes the deployment into the URL path).
pub struct AzureOpenAIProvider {
    chat_client: Client<AzureConfig>,
    embedding_client: Client<AzureConfig>,
    deployment_name: String,
    embedding_deployment_name: String,
    api_version: String,
    max_context_length: usize,
    embedding_dimension: usize,
}

impl std::fmt::Debug for AzureOpenAIProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureOpenAIProvider")
            .field("deployment_name", &self.deployment_name)
            .field("embedding_deployment_name", &self.embedding_deployment_name)
            .field("api_version", &self.api_version)
            .field("max_context_length", &self.max_context_length)
            .field("embedding_dimension", &self.embedding_dimension)
            .finish()
    }
}

impl AzureOpenAIProvider {
    // -----------------------------------------------------------------------
    // Internal client factory
    // -----------------------------------------------------------------------
    fn make_client(
        endpoint: &str,
        api_key: &str,
        deployment_id: &str,
        api_version: &str,
    ) -> Client<AzureConfig> {
        Client::with_config(
            AzureConfig::new()
                .with_api_base(endpoint)
                .with_api_key(api_key)
                .with_deployment_id(deployment_id)
                .with_api_version(api_version),
        )
    }

    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Create a new provider programmatically.
    pub fn new(
        endpoint: impl Into<String>,
        api_key: impl Into<String>,
        deployment_name: impl Into<String>,
    ) -> Self {
        let endpoint = endpoint.into().trim_end_matches('/').to_string();
        let api_key = api_key.into();
        let deployment = deployment_name.into();
        let api_version = DEFAULT_API_VERSION.to_string();

        let chat_client = Self::make_client(&endpoint, &api_key, &deployment, &api_version);
        let embedding_client = Self::make_client(&endpoint, &api_key, &deployment, &api_version);

        Self {
            chat_client,
            embedding_client,
            deployment_name: deployment.clone(),
            embedding_deployment_name: deployment,
            api_version,
            max_context_length: 128_000,
            embedding_dimension: 1536,
        }
    }

    /// Create from standard `AZURE_OPENAI_*` environment variables.
    pub fn from_env() -> Result<Self> {
        let _ = dotenvy::dotenv();
        let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT")
            .map_err(|_| LlmError::ConfigError("AZURE_OPENAI_ENDPOINT not set".into()))?;
        if endpoint.is_empty() {
            return Err(LlmError::ConfigError(
                "AZURE_OPENAI_ENDPOINT is empty".into(),
            ));
        }
        let api_key = std::env::var("AZURE_OPENAI_API_KEY")
            .map_err(|_| LlmError::ConfigError("AZURE_OPENAI_API_KEY not set".into()))?;
        if api_key.is_empty() {
            return Err(LlmError::ConfigError(
                "AZURE_OPENAI_API_KEY is empty".into(),
            ));
        }
        let deployment = std::env::var("AZURE_OPENAI_DEPLOYMENT_NAME")
            .map_err(|_| LlmError::ConfigError("AZURE_OPENAI_DEPLOYMENT_NAME not set".into()))?;
        if deployment.is_empty() {
            return Err(LlmError::ConfigError(
                "AZURE_OPENAI_DEPLOYMENT_NAME is empty".into(),
            ));
        }

        let mut p = Self::new(&endpoint, &api_key, &deployment);
        if let Ok(emb) = std::env::var("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME") {
            p = p.with_embedding_deployment(emb);
        }
        if let Ok(ver) = std::env::var("AZURE_OPENAI_API_VERSION") {
            p = p.with_api_version(ver);
        }
        Ok(p)
    }

    /// Create from enterprise `AZURE_OPENAI_CONTENTGEN_*` environment variables.
    pub fn from_env_contentgen() -> Result<Self> {
        let _ = dotenvy::dotenv();
        let endpoint = std::env::var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT").map_err(|_| {
            LlmError::ConfigError("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT not set".into())
        })?;
        if endpoint.is_empty() {
            return Err(LlmError::ConfigError(
                "AZURE_OPENAI_CONTENTGEN_API_ENDPOINT is empty".into(),
            ));
        }
        let api_key = std::env::var("AZURE_OPENAI_CONTENTGEN_API_KEY")
            .map_err(|_| LlmError::ConfigError("AZURE_OPENAI_CONTENTGEN_API_KEY not set".into()))?;
        if api_key.is_empty() {
            return Err(LlmError::ConfigError(
                "AZURE_OPENAI_CONTENTGEN_API_KEY is empty".into(),
            ));
        }
        let deployment =
            std::env::var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT").map_err(|_| {
                LlmError::ConfigError("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT not set".into())
            })?;
        if deployment.is_empty() {
            return Err(LlmError::ConfigError(
                "AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT is empty".into(),
            ));
        }
        let mut p = Self::new(&endpoint, &api_key, &deployment);
        if let Ok(ver) = std::env::var("AZURE_OPENAI_CONTENTGEN_API_VERSION") {
            p = p.with_api_version(ver);
        }
        Ok(p)
    }

    /// Auto-detect: tries CONTENTGEN first, falls back to standard env vars.
    pub fn from_env_auto() -> Result<Self> {
        let _ = dotenvy::dotenv();
        if std::env::var("AZURE_OPENAI_CONTENTGEN_API_KEY").is_ok() {
            Self::from_env_contentgen()
        } else {
            Self::from_env()
        }
    }

    // -----------------------------------------------------------------------
    // Builder methods
    // -----------------------------------------------------------------------

    /// Override the embedding deployment (rebuilds the embedding client).
    pub fn with_embedding_deployment(mut self, deployment: impl Into<String>) -> Self {
        let deployment = deployment.into();
        let (endpoint, key) = self.read_chat_config();
        self.embedding_client = Self::make_client(&endpoint, &key, &deployment, &self.api_version);
        self.embedding_deployment_name = deployment;
        self
    }

    /// Override the chat deployment name.
    pub fn with_deployment(mut self, deployment: impl Into<String>) -> Self {
        let deployment = deployment.into();
        let (endpoint, key) = self.read_chat_config();
        self.chat_client = Self::make_client(&endpoint, &key, &deployment, &self.api_version);
        self.deployment_name = deployment;
        self
    }

    /// Override the API version on both clients.
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        let version = version.into();
        let (endpoint, key) = self.read_chat_config();
        self.chat_client = Self::make_client(&endpoint, &key, &self.deployment_name, &version);
        self.embedding_client =
            Self::make_client(&endpoint, &key, &self.embedding_deployment_name, &version);
        self.api_version = version;
        self
    }

    /// Set maximum context length (informational).
    pub fn with_max_context_length(mut self, len: usize) -> Self {
        self.max_context_length = len;
        self
    }

    /// Set embedding vector dimension (informational).
    pub fn with_embedding_dimension(mut self, dim: usize) -> Self {
        self.embedding_dimension = dim;
        self
    }

    // Extract endpoint and api_key from the chat client config for rebuilding.
    fn read_chat_config(&self) -> (String, String) {
        use secrecy::ExposeSecret;
        let cfg = self.chat_client.config();
        (
            cfg.api_base().to_string(),
            cfg.api_key().expose_secret().to_string(),
        )
    }

    // -----------------------------------------------------------------------
    // Message helpers
    // -----------------------------------------------------------------------

    fn convert_messages(messages: &[ChatMessage]) -> Result<Vec<ChatCompletionRequestMessage>> {
        messages
            .iter()
            .map(|msg| match msg.role {
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
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(msg.content.as_str())
                        .build()
                        .map(Into::into)
                        .map_err(|e| LlmError::InvalidRequest(e.to_string()))
                }
            })
            .collect()
    }

    fn build_user_content(msg: &ChatMessage) -> ChatCompletionRequestUserMessageContent {
        if msg.has_images() {
            let mut parts: Vec<ChatCompletionRequestUserMessageContentPart> = Vec::new();
            if !msg.content.is_empty() {
                parts.push(ChatCompletionRequestUserMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText {
                        text: msg.content.clone(),
                    },
                ));
            }
            if let Some(ref images) = msg.images {
                for img in images {
                    let detail = Self::parse_image_detail(img);
                    parts.push(ChatCompletionRequestUserMessageContentPart::ImageUrl(
                        ChatCompletionRequestMessageContentPartImage {
                            image_url: ImageUrl {
                                // Pass URL directly for URL images; wrap base64 in data URI otherwise.
                                url: img.to_api_url(),
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

    fn parse_image_detail(img: &ImageData) -> Option<ImageDetail> {
        match img.detail.as_deref() {
            Some("low") => Some(ImageDetail::Low),
            Some("high") => Some(ImageDetail::High),
            Some("auto") => Some(ImageDetail::Auto),
            _ => None,
        }
    }

    fn extract_usage(
        usage: Option<CompletionUsage>,
    ) -> (usize, usize, usize, Option<usize>, Option<usize>) {
        let usage = usage.unwrap_or(CompletionUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        });
        let cache_hit = usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
            .map(|t| t as usize);
        let thinking = usage
            .completion_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens)
            .map(|t| t as usize);
        (
            usage.prompt_tokens as usize,
            usage.completion_tokens as usize,
            usage.total_tokens as usize,
            cache_hit,
            thinking,
        )
    }
}

// ============================================================================
// LLMProvider impl
// ============================================================================

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

    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        self.complete_with_options(prompt, &CompletionOptions::default())
            .await
    }

    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<LLMResponse> {
        let mut msgs = Vec::new();
        if let Some(ref sys) = options.system_prompt {
            msgs.push(ChatMessage::system(sys));
        }
        msgs.push(ChatMessage::user(prompt));
        self.chat(&msgs, Some(options)).await
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let openai_messages = Self::convert_messages(messages)?;
        let opts = options.cloned().unwrap_or_default();

        let mut builder = CreateChatCompletionRequestArgs::default();
        builder
            .model(&self.deployment_name)
            .messages(openai_messages);

        if let Some(max_tokens) = opts.max_tokens {
            builder.max_completion_tokens(max_tokens as u32);
        }
        if let Some(temp) = opts.temperature {
            builder.temperature(temp);
        }
        if let Some(top_p) = opts.top_p {
            builder.top_p(top_p);
        }
        if let Some(stop) = opts.stop {
            builder.stop(stop);
        }
        if let Some(fp) = opts.frequency_penalty {
            builder.frequency_penalty(fp);
        }
        if let Some(pp) = opts.presence_penalty {
            builder.presence_penalty(pp);
        }

        let request = builder
            .build()
            .map_err(|e| LlmError::InvalidRequest(e.to_string()))?;

        let response = self.chat_client.chat().create(request).await?;
        debug!(
            "Azure OpenAI response id={} model={}",
            response.id, response.model
        );

        let choice = response
            .choices
            .first()
            .ok_or_else(|| LlmError::ApiError("No choices in response".into()))?;

        // Guardrail: surface content-filter as an explicit error.
        if let Some(FinishReason::ContentFilter) = choice.finish_reason {
            return Err(LlmError::ApiError(
                "Response blocked by Azure content filter (finish_reason=content_filter)".into(),
            ));
        }

        let content = choice.message.content.clone().unwrap_or_default();
        let (prompt_tokens, completion_tokens, total_tokens, cache_hit, thinking) =
            Self::extract_usage(response.usage.clone());

        let mut metadata = HashMap::new();
        metadata.insert("response_id".to_string(), serde_json::json!(response.id));

        Ok(LLMResponse {
            content,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            model: response.model,
            finish_reason: choice.finish_reason.map(|r| format!("{:?}", r)),
            tool_calls: Vec::new(),
            metadata,
            cache_hit_tokens: cache_hit,
            thinking_tokens: thinking,
            thinking_content: None,
        })
    }

    async fn stream(
        &self,
        prompt: &str,
    ) -> Result<futures::stream::BoxStream<'static, Result<String>>> {
        let user_msg: ChatCompletionRequestMessage =
            ChatCompletionRequestUserMessageArgs::default()
                .content(prompt)
                .build()
                .map(Into::into)
                .map_err(|e| LlmError::InvalidRequest(e.to_string()))?;

        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.deployment_name)
            .messages(vec![user_msg])
            .stream(true)
            .build()
            .map_err(|e| LlmError::InvalidRequest(e.to_string()))?;

        let stream = self.chat_client.chat().create_stream(request).await?;

        let mapped = stream.map(|res| match res {
            Ok(r) => Ok(r
                .choices
                .first()
                .and_then(|c| c.delta.content.clone())
                .unwrap_or_default()),
            Err(e) => Err(LlmError::ApiError(e.to_string())),
        });

        Ok(mapped.boxed())
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    fn supports_tool_streaming(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let openai_messages = Self::convert_messages(messages)?;
        let opts = options.cloned().unwrap_or_default();

        let openai_tools: Vec<ChatCompletionTools> = tools
            .iter()
            .map(|t| {
                ChatCompletionTools::Function(ChatCompletionTool {
                    function: FunctionObjectArgs::default()
                        .name(&t.function.name)
                        .description(&t.function.description)
                        .parameters(t.function.parameters.clone())
                        .build()
                        .expect("Invalid tool definition"),
                })
            })
            .collect();

        let mut builder = CreateChatCompletionRequestArgs::default();
        builder
            .model(&self.deployment_name)
            .messages(openai_messages)
            .tools(openai_tools);

        if let Some(tc) = tool_choice {
            match tc {
                ToolChoice::Auto(_) => {
                    builder.tool_choice(ChatCompletionToolChoiceOption::Mode(
                        ToolChoiceOptions::Auto,
                    ));
                }
                ToolChoice::Required(_) => {
                    builder.tool_choice(ChatCompletionToolChoiceOption::Mode(
                        ToolChoiceOptions::Required,
                    ));
                }
                _ => {}
            }
        }
        if let Some(max_tokens) = opts.max_tokens {
            builder.max_completion_tokens(max_tokens as u32);
        }
        if let Some(temp) = opts.temperature {
            builder.temperature(temp);
        }

        let request = builder
            .build()
            .map_err(|e| LlmError::InvalidRequest(e.to_string()))?;
        let response = self.chat_client.chat().create(request).await?;

        let choice = response
            .choices
            .first()
            .ok_or_else(|| LlmError::ApiError("No choices in response".into()))?;

        if let Some(FinishReason::ContentFilter) = choice.finish_reason {
            return Err(LlmError::ApiError(
                "Response blocked by Azure content filter".into(),
            ));
        }

        let tool_calls: Vec<ToolCall> = choice
            .message
            .tool_calls
            .as_deref()
            .unwrap_or_default()
            .iter()
            .filter_map(|tc| {
                if let ChatCompletionMessageToolCalls::Function(f) = tc {
                    Some(ToolCall {
                        id: f.id.clone(),
                        call_type: "function".to_string(),
                        function: TraitFunctionCall {
                            name: f.function.name.clone(),
                            arguments: f.function.arguments.clone(),
                        },
                    })
                } else {
                    None
                }
            })
            .collect();

        let (prompt_tokens, completion_tokens, total_tokens, cache_hit, thinking) =
            Self::extract_usage(response.usage.clone());

        let mut metadata = HashMap::new();
        metadata.insert("response_id".to_string(), serde_json::json!(response.id));

        Ok(LLMResponse {
            content: choice.message.content.clone().unwrap_or_default(),
            prompt_tokens,
            completion_tokens,
            total_tokens,
            model: response.model,
            finish_reason: choice.finish_reason.map(|r| format!("{:?}", r)),
            tool_calls,
            metadata,
            cache_hit_tokens: cache_hit,
            thinking_tokens: thinking,
            thinking_content: None,
        })
    }

    async fn chat_with_tools_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamChunk>>> {
        let openai_messages = Self::convert_messages(messages)?;
        let opts = options.cloned().unwrap_or_default();

        let openai_tools: Vec<ChatCompletionTools> = tools
            .iter()
            .map(|t| {
                ChatCompletionTools::Function(ChatCompletionTool {
                    function: FunctionObjectArgs::default()
                        .name(&t.function.name)
                        .description(&t.function.description)
                        .parameters(t.function.parameters.clone())
                        .build()
                        .expect("Invalid tool definition"),
                })
            })
            .collect();

        let mut builder = CreateChatCompletionRequestArgs::default();
        builder
            .model(&self.deployment_name)
            .messages(openai_messages)
            .tools(openai_tools)
            .stream(true);

        if let Some(tc) = tool_choice {
            match tc {
                ToolChoice::Auto(_) => {
                    builder.tool_choice(ChatCompletionToolChoiceOption::Mode(
                        ToolChoiceOptions::Auto,
                    ));
                }
                ToolChoice::Required(_) => {
                    builder.tool_choice(ChatCompletionToolChoiceOption::Mode(
                        ToolChoiceOptions::Required,
                    ));
                }
                _ => {}
            }
        }
        if let Some(max_tokens) = opts.max_tokens {
            builder.max_completion_tokens(max_tokens as u32);
        }
        if let Some(temp) = opts.temperature {
            builder.temperature(temp);
        }

        let request = builder
            .build()
            .map_err(|e| LlmError::InvalidRequest(e.to_string()))?;

        let stream = self.chat_client.chat().create_stream(request).await?;

        let mapped = stream.map(|result| match result {
            Ok(response) => {
                if let Some(choice) = response.choices.first() {
                    if let Some(content) = &choice.delta.content {
                        return Ok(StreamChunk::Content(content.clone()));
                    }
                    if let Some(chunks) = &choice.delta.tool_calls {
                        if let Some(chunk) = chunks.first() {
                            return Ok(StreamChunk::ToolCallDelta {
                                index: chunk.index as usize,
                                id: chunk.id.clone(),
                                function_name: chunk.function.as_ref().and_then(|f| f.name.clone()),
                                function_arguments: chunk
                                    .function
                                    .as_ref()
                                    .and_then(|f| f.arguments.clone()),
                            });
                        }
                    }
                    if let Some(reason) = &choice.finish_reason {
                        let r = match reason {
                            FinishReason::Stop => "stop",
                            FinishReason::Length => "length",
                            FinishReason::ToolCalls => "tool_calls",
                            FinishReason::ContentFilter => "content_filter",
                            FinishReason::FunctionCall => "function_call",
                        };
                        return Ok(StreamChunk::Finished {
                            reason: r.to_string(),
                            ttft_ms: None,
                        });
                    }
                }
                Ok(StreamChunk::Content(String::new()))
            }
            Err(e) => Err(LlmError::ApiError(e.to_string())),
        });

        Ok(mapped.boxed())
    }
}

// ============================================================================
// EmbeddingProvider impl
// ============================================================================

#[async_trait]
impl EmbeddingProvider for AzureOpenAIProvider {
    fn name(&self) -> &str {
        "azure-openai"
    }

    #[allow(clippy::misnamed_getters)]
    fn model(&self) -> &str {
        &self.embedding_deployment_name
    }

    fn dimension(&self) -> usize {
        self.embedding_dimension
    }

    fn max_tokens(&self) -> usize {
        8191
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.embedding_deployment_name)
            .input(EmbeddingInput::StringArray(texts.to_vec()))
            .build()
            .map_err(|e| LlmError::InvalidRequest(e.to_string()))?;

        let response = self.embedding_client.embeddings().create(request).await?;
        let mut data = response.data;
        data.sort_by_key(|e| e.index);
        Ok(data.into_iter().map(|e| e.embedding).collect())
    }
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let p = AzureOpenAIProvider::new(
            "https://myresource.openai.azure.com",
            "test-api-key",
            "gpt-4o",
        );
        assert_eq!(LLMProvider::name(&p), "azure-openai");
        assert_eq!(LLMProvider::model(&p), "gpt-4o");
        assert_eq!(p.max_context_length(), 128_000);
    }

    #[test]
    fn test_with_embedding_deployment() {
        let p = AzureOpenAIProvider::new("https://x.openai.azure.com", "key", "chat-dep")
            .with_embedding_deployment("embed-dep");
        assert_eq!(EmbeddingProvider::model(&p), "embed-dep");
        assert_eq!(LLMProvider::model(&p), "chat-dep");
    }

    #[test]
    fn test_builder_chain() {
        let p = AzureOpenAIProvider::new("https://x.openai.azure.com", "key", "chat")
            .with_embedding_deployment("embed")
            .with_max_context_length(64_000)
            .with_embedding_dimension(3072);
        assert_eq!(p.max_context_length(), 64_000);
        assert_eq!(p.dimension(), 3072);
    }

    #[test]
    fn test_supports_flags() {
        let p = AzureOpenAIProvider::new("https://x.openai.azure.com", "key", "dep");
        assert!(p.supports_streaming());
        assert!(p.supports_function_calling());
        assert!(p.supports_tool_streaming());
        assert!(p.supports_json_mode());
    }

    #[test]
    fn test_message_conversion_basic() {
        let msgs = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi!"),
        ];
        let converted = AzureOpenAIProvider::convert_messages(&msgs).unwrap();
        assert_eq!(converted.len(), 3);
    }

    #[test]
    fn test_message_conversion_empty() {
        let r = AzureOpenAIProvider::convert_messages(&[]);
        assert!(r.unwrap().is_empty());
    }

    #[test]
    fn test_extract_usage_empty() {
        let (p, c, t, cache, think) = AzureOpenAIProvider::extract_usage(None);
        assert_eq!((p, c, t), (0, 0, 0));
        assert!(cache.is_none());
        assert!(think.is_none());
    }

    #[test]
    fn test_image_detail_parsing() {
        let mut img = ImageData::new("data", "image/png");
        img.detail = Some("high".to_string());
        assert!(matches!(
            AzureOpenAIProvider::parse_image_detail(&img),
            Some(ImageDetail::High)
        ));

        img.detail = Some("low".to_string());
        assert!(matches!(
            AzureOpenAIProvider::parse_image_detail(&img),
            Some(ImageDetail::Low)
        ));

        img.detail = None;
        assert!(AzureOpenAIProvider::parse_image_detail(&img).is_none());
    }
}
