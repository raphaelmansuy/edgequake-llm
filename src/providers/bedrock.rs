//! AWS Bedrock Runtime provider via the Converse API.
//!
//! This provider implements the `LLMProvider` trait using the AWS Bedrock Runtime
//! Converse API, which provides a model-agnostic interface for chat completions
//! across all Bedrock-hosted models (Anthropic Claude, Amazon Nova, Meta Llama,
//! Mistral, Cohere, etc.).
//!
//! # Feature Gate
//!
//! This module is only available when the `bedrock` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! edgequake-llm = { version = "0.2", features = ["bedrock"] }
//! ```
//!
//! # Environment Variables
//!
//! - `AWS_BEDROCK_MODEL`: Model ID (default: `anthropic.claude-3-5-sonnet-20241022-v2:0`)
//! - `AWS_REGION` / `AWS_DEFAULT_REGION`: AWS region (default: `us-east-1`)
//! - Standard AWS credential chain (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
//!   `AWS_SESSION_TOKEN`, `AWS_PROFILE`, IAM roles, etc.)
//!
//! # Example
//!
//! ```rust,ignore
//! use edgequake_llm::BedrockProvider;
//! use edgequake_llm::traits::{ChatMessage, LLMProvider};
//!
//! let provider = BedrockProvider::from_env().await?;
//! let messages = vec![ChatMessage::user("Hello!")];
//! let response = provider.chat(&messages, None).await?;
//! println!("{}", response.content);
//! ```

use std::collections::HashMap;

use async_trait::async_trait;
use aws_config::SdkConfig;
use aws_sdk_bedrockruntime::types::{
    ContentBlock, ConversationRole, ConverseOutput, InferenceConfiguration, Message, StopReason,
    SystemContentBlock, Tool, ToolChoice as BedrockToolChoice, ToolConfiguration, ToolInputSchema,
    ToolSpecification, ToolUseBlock,
};
use aws_sdk_bedrockruntime::Client;
use aws_smithy_types::Document;
use futures::stream::BoxStream;
use tracing::{debug, instrument};

use crate::error::{LlmError, Result};
use crate::traits::{
    ChatMessage, ChatRole, CompletionOptions, LLMProvider, LLMResponse,
    ToolCall as EdgequakeToolCall, ToolChoice as EdgequakeToolChoice,
    ToolDefinition as EdgequakeToolDefinition,
};

// ============================================================================
// Constants
// ============================================================================

/// Default Bedrock model (Claude 3.5 Sonnet v2)
const DEFAULT_MODEL: &str = "anthropic.claude-3-5-sonnet-20241022-v2:0";

/// Default AWS region for Bedrock
const DEFAULT_REGION: &str = "us-east-1";

/// Default max context length (Claude 3.5 Sonnet = 200k tokens)
const DEFAULT_MAX_CONTEXT: usize = 200_000;

// ============================================================================
// BedrockProvider
// ============================================================================

/// AWS Bedrock Runtime LLM provider using the Converse API.
///
/// Uses the model-agnostic Converse API which works with all Bedrock models
/// without requiring model-specific payload formatting.
#[derive(Debug, Clone)]
pub struct BedrockProvider {
    client: Client,
    model: String,
    max_context_length: usize,
}

impl BedrockProvider {
    /// Create a new Bedrock provider from an existing AWS SDK config.
    ///
    /// # Arguments
    ///
    /// * `sdk_config` - Pre-configured AWS SDK config
    /// * `model` - Bedrock model ID (e.g., `anthropic.claude-3-5-sonnet-20241022-v2:0`)
    pub fn new(sdk_config: &SdkConfig, model: impl Into<String>) -> Self {
        let model = model.into();
        let max_context_length = Self::context_length_for_model(&model);
        Self {
            client: Client::new(sdk_config),
            model,
            max_context_length,
        }
    }

    /// Create a provider from environment variables (async).
    ///
    /// Uses the standard AWS credential chain and reads:
    /// - `AWS_BEDROCK_MODEL` for the model ID
    /// - `AWS_REGION` / `AWS_DEFAULT_REGION` for the region
    pub async fn from_env() -> Result<Self> {
        let region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|_| DEFAULT_REGION.to_string());

        let model =
            std::env::var("AWS_BEDROCK_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

        let sdk_config = aws_config::from_env()
            .region(aws_config::Region::new(region))
            .load()
            .await;

        let max_context_length = Self::context_length_for_model(&model);

        Ok(Self {
            client: Client::new(&sdk_config),
            model,
            max_context_length,
        })
    }

    /// Set a custom model ID.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        let model = model.into();
        self.max_context_length = Self::context_length_for_model(&model);
        self.model = model;
        self
    }

    /// Set a custom max context length.
    pub fn with_max_context_length(mut self, length: usize) -> Self {
        self.max_context_length = length;
        self
    }

    /// Estimate context length from model ID.
    fn context_length_for_model(model: &str) -> usize {
        let model_lower = model.to_lowercase();
        if model_lower.contains("claude-3") || model_lower.contains("claude-4") {
            200_000
        } else if model_lower.contains("claude-2") {
            100_000
        } else if model_lower.contains("nova") {
            300_000
        } else if model_lower.contains("llama") {
            128_000
        } else if model_lower.contains("mistral") {
            32_000
        } else if model_lower.contains("cohere") {
            128_000
        } else {
            DEFAULT_MAX_CONTEXT
        }
    }

    // ========================================================================
    // Document Conversion Helpers
    // ========================================================================

    /// Convert a `serde_json::Value` to an `aws_smithy_types::Document`.
    ///
    /// AWS Smithy `Document` does not implement serde traits, so we need
    /// manual conversion between JSON and Document representations.
    fn json_to_document(value: &serde_json::Value) -> Document {
        match value {
            serde_json::Value::Null => Document::Null,
            serde_json::Value::Bool(b) => Document::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(u) = n.as_u64() {
                    Document::Number(aws_smithy_types::Number::PosInt(u))
                } else if let Some(i) = n.as_i64() {
                    Document::Number(aws_smithy_types::Number::NegInt(i))
                } else if let Some(f) = n.as_f64() {
                    Document::Number(aws_smithy_types::Number::Float(f))
                } else {
                    Document::Null
                }
            }
            serde_json::Value::String(s) => Document::String(s.clone()),
            serde_json::Value::Array(arr) => {
                Document::Array(arr.iter().map(Self::json_to_document).collect())
            }
            serde_json::Value::Object(obj) => Document::Object(
                obj.iter()
                    .map(|(k, v)| (k.clone(), Self::json_to_document(v)))
                    .collect(),
            ),
        }
    }

    /// Convert an `aws_smithy_types::Document` to a `serde_json::Value`.
    fn document_to_json(doc: &Document) -> serde_json::Value {
        match doc {
            Document::Null => serde_json::Value::Null,
            Document::Bool(b) => serde_json::Value::Bool(*b),
            Document::Number(n) => match n {
                aws_smithy_types::Number::PosInt(u) => serde_json::json!(*u),
                aws_smithy_types::Number::NegInt(i) => serde_json::json!(*i),
                aws_smithy_types::Number::Float(f) => serde_json::json!(*f),
            },
            Document::String(s) => serde_json::Value::String(s.clone()),
            Document::Array(arr) => {
                serde_json::Value::Array(arr.iter().map(Self::document_to_json).collect())
            }
            Document::Object(obj) => serde_json::Value::Object(
                obj.iter()
                    .map(|(k, v)| (k.clone(), Self::document_to_json(v)))
                    .collect(),
            ),
        }
    }

    // ========================================================================
    // Message Conversion Helpers
    // ========================================================================

    /// Convert edgequake ChatMessages into Bedrock Messages + optional system blocks.
    fn convert_messages(
        messages: &[ChatMessage],
        system_prompt: Option<&str>,
    ) -> Result<(Vec<Message>, Vec<SystemContentBlock>)> {
        let mut bedrock_messages: Vec<Message> = Vec::new();
        let mut system_blocks: Vec<SystemContentBlock> = Vec::new();

        // Add explicit system prompt if provided via CompletionOptions
        if let Some(sys) = system_prompt {
            if !sys.is_empty() {
                system_blocks.push(SystemContentBlock::Text(sys.to_string()));
            }
        }

        for msg in messages {
            match msg.role {
                ChatRole::System => {
                    // System messages go into the system content blocks
                    system_blocks.push(SystemContentBlock::Text(msg.content.clone()));
                }
                ChatRole::User => {
                    let content = ContentBlock::Text(msg.content.clone());
                    let bedrock_msg = Message::builder()
                        .role(ConversationRole::User)
                        .content(content)
                        .build()
                        .map_err(|e| {
                            LlmError::ProviderError(format!(
                                "Failed to build Bedrock user message: {e}"
                            ))
                        })?;
                    bedrock_messages.push(bedrock_msg);
                }
                ChatRole::Assistant => {
                    // Assistant messages may contain tool use blocks
                    let mut content_blocks = vec![ContentBlock::Text(msg.content.clone())];

                    if let Some(ref tool_calls) = msg.tool_calls {
                        for tc in tool_calls {
                            // Convert JSON arguments string to Document
                            let input_doc =
                                serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                                    .map(|v| Self::json_to_document(&v))
                                    .unwrap_or_else(|_| {
                                        Document::String(tc.function.arguments.clone())
                                    });

                            let tool_use = ToolUseBlock::builder()
                                .tool_use_id(&tc.id)
                                .name(&tc.function.name)
                                .input(input_doc)
                                .build()
                                .map_err(|e| {
                                    LlmError::ProviderError(format!(
                                        "Failed to build tool use block: {e}"
                                    ))
                                })?;
                            content_blocks.push(ContentBlock::ToolUse(tool_use));
                        }
                    }

                    let mut builder = Message::builder().role(ConversationRole::Assistant);
                    for block in content_blocks {
                        builder = builder.content(block);
                    }
                    let bedrock_msg = builder.build().map_err(|e| {
                        LlmError::ProviderError(format!(
                            "Failed to build Bedrock assistant message: {e}"
                        ))
                    })?;
                    bedrock_messages.push(bedrock_msg);
                }
                ChatRole::Tool | ChatRole::Function => {
                    // Tool results go as user messages with ToolResult content blocks
                    let tool_call_id = msg.tool_call_id.as_deref().unwrap_or("unknown").to_string();
                    let result_content =
                        aws_sdk_bedrockruntime::types::ToolResultContentBlock::Text(
                            msg.content.clone(),
                        );
                    let tool_result = aws_sdk_bedrockruntime::types::ToolResultBlock::builder()
                        .tool_use_id(tool_call_id)
                        .content(result_content)
                        .build()
                        .map_err(|e| {
                            LlmError::ProviderError(format!(
                                "Failed to build tool result block: {e}"
                            ))
                        })?;
                    let content = ContentBlock::ToolResult(tool_result);
                    let bedrock_msg = Message::builder()
                        .role(ConversationRole::User)
                        .content(content)
                        .build()
                        .map_err(|e| {
                            LlmError::ProviderError(format!(
                                "Failed to build Bedrock tool result message: {e}"
                            ))
                        })?;
                    bedrock_messages.push(bedrock_msg);
                }
            }
        }

        Ok((bedrock_messages, system_blocks))
    }

    /// Build `InferenceConfiguration` from `CompletionOptions`.
    fn build_inference_config(
        options: Option<&CompletionOptions>,
    ) -> Option<InferenceConfiguration> {
        let opts = options?;
        let mut builder = InferenceConfiguration::builder();
        let mut has_config = false;

        if let Some(max_tokens) = opts.max_tokens {
            builder = builder.max_tokens(max_tokens as i32);
            has_config = true;
        }
        if let Some(temperature) = opts.temperature {
            builder = builder.temperature(temperature);
            has_config = true;
        }
        if let Some(top_p) = opts.top_p {
            builder = builder.top_p(top_p);
            has_config = true;
        }
        if let Some(ref stops) = opts.stop {
            for s in stops {
                builder = builder.stop_sequences(s.clone());
            }
            has_config = true;
        }

        if has_config {
            Some(builder.build())
        } else {
            None
        }
    }

    /// Convert edgequake tool definitions to Bedrock ToolConfiguration.
    fn build_tool_config(
        tools: &[EdgequakeToolDefinition],
        tool_choice: Option<&EdgequakeToolChoice>,
    ) -> Result<Option<ToolConfiguration>> {
        if tools.is_empty() {
            return Ok(None);
        }

        let mut bedrock_tools = Vec::new();
        for tool in tools {
            let schema_doc = Self::json_to_document(&tool.function.parameters);

            let spec = ToolSpecification::builder()
                .name(&tool.function.name)
                .description(&tool.function.description)
                .input_schema(ToolInputSchema::Json(schema_doc))
                .build()
                .map_err(|e| {
                    LlmError::ProviderError(format!("Failed to build tool specification: {e}"))
                })?;
            bedrock_tools.push(Tool::ToolSpec(spec));
        }

        let mut config_builder = ToolConfiguration::builder();
        for tool in bedrock_tools {
            config_builder = config_builder.tools(tool);
        }

        // Map tool_choice
        if let Some(choice) = tool_choice {
            let bedrock_choice = match choice {
                // "none" means disable tool calling — omit tool_config entirely
                EdgequakeToolChoice::Auto(s) if s == "none" => {
                    return Ok(None);
                }
                EdgequakeToolChoice::Auto(_) => BedrockToolChoice::Auto(
                    aws_sdk_bedrockruntime::types::AutoToolChoice::builder().build(),
                ),
                EdgequakeToolChoice::Required(_) => BedrockToolChoice::Any(
                    aws_sdk_bedrockruntime::types::AnyToolChoice::builder().build(),
                ),
                EdgequakeToolChoice::Function { function, .. } => BedrockToolChoice::Tool(
                    aws_sdk_bedrockruntime::types::SpecificToolChoice::builder()
                        .name(&function.name)
                        .build()
                        .map_err(|e| {
                            LlmError::ProviderError(format!(
                                "Failed to build specific tool choice: {e}"
                            ))
                        })?,
                ),
            };
            config_builder = config_builder.tool_choice(bedrock_choice);
        }

        let config = config_builder.build().map_err(|e| {
            LlmError::ProviderError(format!("Failed to build tool configuration: {e}"))
        })?;
        Ok(Some(config))
    }

    /// Map Bedrock StopReason to edgequake finish_reason string.
    fn map_stop_reason(reason: &StopReason) -> String {
        match reason {
            StopReason::EndTurn => "stop".to_string(),
            StopReason::MaxTokens => "length".to_string(),
            StopReason::StopSequence => "stop".to_string(),
            StopReason::ToolUse => "tool_calls".to_string(),
            StopReason::ContentFiltered => "content_filter".to_string(),
            StopReason::GuardrailIntervened => "content_filter".to_string(),
            _ => "stop".to_string(),
        }
    }

    /// Extract text content and tool calls from Bedrock ConverseOutput.
    fn extract_content(output: &ConverseOutput) -> (String, Vec<EdgequakeToolCall>) {
        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();

        if let ConverseOutput::Message(msg) = output {
            for block in msg.content() {
                match block {
                    ContentBlock::Text(text) => {
                        text_parts.push(text.clone());
                    }
                    ContentBlock::ToolUse(tool_use) => {
                        // Convert Document input to JSON string for the tool call arguments
                        let arguments_json = Self::document_to_json(&tool_use.input);
                        let arguments_str =
                            serde_json::to_string(&arguments_json).unwrap_or_default();

                        tool_calls.push(EdgequakeToolCall {
                            id: tool_use.tool_use_id.clone(),
                            call_type: "function".to_string(),
                            function: crate::traits::FunctionCall {
                                name: tool_use.name.clone(),
                                arguments: arguments_str,
                            },
                        });
                    }
                    _ => {}
                }
            }
        }

        (text_parts.join(""), tool_calls)
    }
}

// ============================================================================
// LLMProvider Implementation
// ============================================================================

#[async_trait]
impl LLMProvider for BedrockProvider {
    fn name(&self) -> &str {
        "bedrock"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_context_length(&self) -> usize {
        self.max_context_length
    }

    #[instrument(skip(self, prompt), fields(provider = "bedrock", model = %self.model))]
    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        let messages = vec![ChatMessage::user(prompt)];
        self.chat(&messages, None).await
    }

    #[instrument(skip(self, prompt, options), fields(provider = "bedrock", model = %self.model))]
    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<LLMResponse> {
        let messages = vec![ChatMessage::user(prompt)];
        self.chat(&messages, Some(options)).await
    }

    #[instrument(skip(self, messages, options), fields(provider = "bedrock", model = %self.model))]
    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let system_prompt = options.and_then(|o| o.system_prompt.as_deref());
        let (bedrock_messages, system_blocks) = Self::convert_messages(messages, system_prompt)?;

        let mut request = self.client.converse().model_id(&self.model);

        // Add messages
        for msg in bedrock_messages {
            request = request.messages(msg);
        }

        // Add system blocks
        for block in system_blocks {
            request = request.system(block);
        }

        // Add inference config
        if let Some(config) = Self::build_inference_config(options) {
            request = request.inference_config(config);
        }

        debug!("Sending Bedrock Converse request for model: {}", self.model);

        let response = request
            .send()
            .await
            .map_err(|e| LlmError::ProviderError(format!("Bedrock Converse API error: {e}")))?;

        // Extract content and tool calls
        let (content, tool_calls) = response
            .output()
            .map(Self::extract_content)
            .unwrap_or_default();

        // Extract token usage (i32 → usize)
        let (prompt_tokens, completion_tokens, total_tokens) = response
            .usage()
            .map(|u| {
                let input = u.input_tokens() as usize;
                let output = u.output_tokens() as usize;
                (input, output, input + output)
            })
            .unwrap_or((0, 0, 0));

        let finish_reason = Self::map_stop_reason(&response.stop_reason);

        Ok(LLMResponse {
            content,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            model: self.model.clone(),
            finish_reason: Some(finish_reason),
            tool_calls,
            metadata: HashMap::new(),
            cache_hit_tokens: None,
            thinking_tokens: None,
            thinking_content: None,
        })
    }

    #[instrument(skip(self, messages, tools, tool_choice, options), fields(provider = "bedrock", model = %self.model))]
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[EdgequakeToolDefinition],
        tool_choice: Option<EdgequakeToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let system_prompt = options.and_then(|o| o.system_prompt.as_deref());
        let (bedrock_messages, system_blocks) = Self::convert_messages(messages, system_prompt)?;

        let mut request = self.client.converse().model_id(&self.model);

        for msg in bedrock_messages {
            request = request.messages(msg);
        }
        for block in system_blocks {
            request = request.system(block);
        }
        if let Some(config) = Self::build_inference_config(options) {
            request = request.inference_config(config);
        }

        // Add tool configuration
        if let Some(tool_config) = Self::build_tool_config(tools, tool_choice.as_ref())? {
            request = request.tool_config(tool_config);
        }

        debug!(
            "Sending Bedrock Converse request with {} tools for model: {}",
            tools.len(),
            self.model
        );

        let response = request
            .send()
            .await
            .map_err(|e| LlmError::ProviderError(format!("Bedrock Converse API error: {e}")))?;

        let (content, tool_calls) = response
            .output()
            .map(Self::extract_content)
            .unwrap_or_default();

        let (prompt_tokens, completion_tokens, total_tokens) = response
            .usage()
            .map(|u| {
                let input = u.input_tokens() as usize;
                let output = u.output_tokens() as usize;
                (input, output, input + output)
            })
            .unwrap_or((0, 0, 0));

        let finish_reason = Self::map_stop_reason(&response.stop_reason);

        Ok(LLMResponse {
            content,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            model: self.model.clone(),
            finish_reason: Some(finish_reason),
            tool_calls,
            metadata: HashMap::new(),
            cache_hit_tokens: None,
            thinking_tokens: None,
            thinking_content: None,
        })
    }

    #[instrument(skip(self, prompt), fields(provider = "bedrock", model = %self.model))]
    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        let messages = vec![ChatMessage::user(prompt)];
        let (bedrock_messages, system_blocks) = Self::convert_messages(&messages, None)?;

        let mut request = self.client.converse_stream().model_id(&self.model);

        for msg in bedrock_messages {
            request = request.messages(msg);
        }
        for block in system_blocks {
            request = request.system(block);
        }

        debug!(
            "Sending Bedrock ConverseStream request for model: {}",
            self.model
        );

        let response = request.send().await.map_err(|e| {
            LlmError::ProviderError(format!("Bedrock ConverseStream API error: {e}"))
        })?;

        // Use futures::stream::unfold to convert EventReceiver into a BoxStream<Result<String>>
        use futures::stream;

        let mapped_stream = stream::unfold(response.stream, |mut rx| async move {
            loop {
                match rx.recv().await {
                    Ok(Some(event)) => {
                        use aws_sdk_bedrockruntime::types::ConverseStreamOutput as CSO;
                        match event {
                            CSO::ContentBlockDelta(delta_event) => {
                                if let Some(delta) = delta_event.delta() {
                                    use aws_sdk_bedrockruntime::types::ContentBlockDelta;
                                    if let ContentBlockDelta::Text(text) = delta {
                                        return Some((Ok(text.clone()), rx));
                                    }
                                }
                                // Non-text delta, continue to next event
                            }
                            CSO::MessageStop(_) => {
                                return None; // End of stream
                            }
                            _ => {
                                // MessageStart, ContentBlockStart, ContentBlockStop, Metadata
                                // Skip these and continue
                            }
                        }
                    }
                    Ok(None) => {
                        return None; // Stream ended
                    }
                    Err(e) => {
                        return Some((
                            Err(LlmError::ProviderError(format!(
                                "Bedrock stream error: {e}"
                            ))),
                            rx,
                        ));
                    }
                }
            }
        });

        Ok(Box::pin(mapped_stream))
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_tool_streaming(&self) -> bool {
        false // Tool streaming via Bedrock ConverseStream is complex; deferred
    }

    fn supports_json_mode(&self) -> bool {
        false
    }

    fn supports_function_calling(&self) -> bool {
        // Most Bedrock models support tool use via the Converse API
        true
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_length_claude3() {
        assert_eq!(
            BedrockProvider::context_length_for_model("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            200_000
        );
        assert_eq!(
            BedrockProvider::context_length_for_model("anthropic.claude-4-sonnet-20250514-v1:0"),
            200_000
        );
    }

    #[test]
    fn test_context_length_claude2() {
        assert_eq!(
            BedrockProvider::context_length_for_model("anthropic.claude-2"),
            100_000
        );
    }

    #[test]
    fn test_context_length_nova() {
        assert_eq!(
            BedrockProvider::context_length_for_model("amazon.nova-pro-v1:0"),
            300_000
        );
    }

    #[test]
    fn test_context_length_llama() {
        assert_eq!(
            BedrockProvider::context_length_for_model("meta.llama3-70b-instruct-v1:0"),
            128_000
        );
    }

    #[test]
    fn test_context_length_mistral() {
        assert_eq!(
            BedrockProvider::context_length_for_model("mistral.mistral-large-2407-v1:0"),
            32_000
        );
    }

    #[test]
    fn test_context_length_cohere() {
        assert_eq!(
            BedrockProvider::context_length_for_model("cohere.command-r-plus-v1:0"),
            128_000
        );
    }

    #[test]
    fn test_context_length_default() {
        assert_eq!(
            BedrockProvider::context_length_for_model("some-unknown-model"),
            DEFAULT_MAX_CONTEXT
        );
    }

    #[test]
    fn test_stop_reason_mapping() {
        assert_eq!(
            BedrockProvider::map_stop_reason(&StopReason::EndTurn),
            "stop"
        );
        assert_eq!(
            BedrockProvider::map_stop_reason(&StopReason::MaxTokens),
            "length"
        );
        assert_eq!(
            BedrockProvider::map_stop_reason(&StopReason::StopSequence),
            "stop"
        );
        assert_eq!(
            BedrockProvider::map_stop_reason(&StopReason::ToolUse),
            "tool_calls"
        );
        assert_eq!(
            BedrockProvider::map_stop_reason(&StopReason::ContentFiltered),
            "content_filter"
        );
        assert_eq!(
            BedrockProvider::map_stop_reason(&StopReason::GuardrailIntervened),
            "content_filter"
        );
    }

    #[test]
    fn test_build_inference_config_none() {
        assert!(BedrockProvider::build_inference_config(None).is_none());
    }

    #[test]
    fn test_build_inference_config_with_options() {
        let opts = CompletionOptions {
            max_tokens: Some(1024),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stop: Some(vec!["END".to_string()]),
            ..Default::default()
        };
        let config = BedrockProvider::build_inference_config(Some(&opts));
        assert!(config.is_some());
    }

    #[test]
    fn test_build_inference_config_empty_options() {
        let opts = CompletionOptions::default();
        let config = BedrockProvider::build_inference_config(Some(&opts));
        assert!(config.is_none());
    }

    #[test]
    fn test_convert_messages_system() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
        ];
        let (bedrock_msgs, system_blocks) =
            BedrockProvider::convert_messages(&messages, None).unwrap();
        // System message goes into system_blocks, user message into bedrock_msgs
        assert_eq!(system_blocks.len(), 1);
        assert_eq!(bedrock_msgs.len(), 1);
    }

    #[test]
    fn test_convert_messages_with_system_prompt_option() {
        let messages = vec![ChatMessage::user("Hello")];
        let (bedrock_msgs, system_blocks) =
            BedrockProvider::convert_messages(&messages, Some("Be concise")).unwrap();
        assert_eq!(system_blocks.len(), 1);
        assert_eq!(bedrock_msgs.len(), 1);
    }

    #[test]
    fn test_convert_messages_empty_system_prompt_ignored() {
        let messages = vec![ChatMessage::user("Hello")];
        let (_, system_blocks) = BedrockProvider::convert_messages(&messages, Some("")).unwrap();
        assert_eq!(system_blocks.len(), 0);
    }

    #[test]
    fn test_convert_messages_tool_result() {
        let messages = vec![ChatMessage::tool_result("call_123", "Result data")];
        let (bedrock_msgs, system_blocks) =
            BedrockProvider::convert_messages(&messages, None).unwrap();
        assert_eq!(system_blocks.len(), 0);
        assert_eq!(bedrock_msgs.len(), 1);
        // Tool results are sent as user messages in Bedrock
    }

    #[test]
    fn test_convert_messages_multiple_system_blocks() {
        let messages = vec![
            ChatMessage::system("System 1"),
            ChatMessage::system("System 2"),
            ChatMessage::user("Hello"),
        ];
        let (bedrock_msgs, system_blocks) =
            BedrockProvider::convert_messages(&messages, Some("Prefix system")).unwrap();
        // 1 from options + 2 from messages = 3 system blocks
        assert_eq!(system_blocks.len(), 3);
        assert_eq!(bedrock_msgs.len(), 1);
    }

    #[test]
    fn test_convert_messages_user_and_assistant() {
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
            ChatMessage::user("How are you?"),
        ];
        let (bedrock_msgs, system_blocks) =
            BedrockProvider::convert_messages(&messages, None).unwrap();
        assert_eq!(system_blocks.len(), 0);
        assert_eq!(bedrock_msgs.len(), 3);
    }

    #[test]
    fn test_json_to_document_null() {
        let doc = BedrockProvider::json_to_document(&serde_json::Value::Null);
        assert!(matches!(doc, Document::Null));
    }

    #[test]
    fn test_json_to_document_bool() {
        let doc = BedrockProvider::json_to_document(&serde_json::json!(true));
        assert!(matches!(doc, Document::Bool(true)));
    }

    #[test]
    fn test_json_to_document_string() {
        let doc = BedrockProvider::json_to_document(&serde_json::json!("hello"));
        assert!(matches!(doc, Document::String(s) if s == "hello"));
    }

    #[test]
    fn test_json_to_document_number() {
        let doc = BedrockProvider::json_to_document(&serde_json::json!(42));
        assert!(matches!(
            doc,
            Document::Number(aws_smithy_types::Number::PosInt(42))
        ));
    }

    #[test]
    fn test_json_to_document_negative_number() {
        let doc = BedrockProvider::json_to_document(&serde_json::json!(-5));
        assert!(matches!(
            doc,
            Document::Number(aws_smithy_types::Number::NegInt(-5))
        ));
    }

    #[test]
    fn test_json_to_document_float() {
        let doc = BedrockProvider::json_to_document(&serde_json::json!(1.125));
        if let Document::Number(aws_smithy_types::Number::Float(f)) = doc {
            assert!((f - 1.125).abs() < f64::EPSILON);
        } else {
            panic!("Expected float document");
        }
    }

    #[test]
    fn test_json_to_document_array() {
        let doc = BedrockProvider::json_to_document(&serde_json::json!([1, "two", null]));
        if let Document::Array(arr) = doc {
            assert_eq!(arr.len(), 3);
        } else {
            panic!("Expected array document");
        }
    }

    #[test]
    fn test_json_to_document_object() {
        let doc = BedrockProvider::json_to_document(&serde_json::json!({"key": "value"}));
        if let Document::Object(obj) = doc {
            assert_eq!(obj.len(), 1);
            assert!(obj.contains_key("key"));
        } else {
            panic!("Expected object document");
        }
    }

    #[test]
    fn test_document_to_json_roundtrip() {
        let original = serde_json::json!({
            "name": "test",
            "age": 30,
            "active": true,
            "tags": ["a", "b"],
            "nested": {"x": 1.5}
        });
        let doc = BedrockProvider::json_to_document(&original);
        let recovered = BedrockProvider::document_to_json(&doc);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_extract_content_text_only() {
        let msg = Message::builder()
            .role(ConversationRole::Assistant)
            .content(ContentBlock::Text("Hello world".to_string()))
            .build()
            .unwrap();
        let output = ConverseOutput::Message(msg);
        let (text, tool_calls) = BedrockProvider::extract_content(&output);
        assert_eq!(text, "Hello world");
        assert!(tool_calls.is_empty());
    }

    #[test]
    fn test_extract_content_multiple_text_blocks() {
        let msg = Message::builder()
            .role(ConversationRole::Assistant)
            .content(ContentBlock::Text("Hello ".to_string()))
            .content(ContentBlock::Text("world".to_string()))
            .build()
            .unwrap();
        let output = ConverseOutput::Message(msg);
        let (text, _) = BedrockProvider::extract_content(&output);
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn test_extract_content_with_tool_use() {
        let tool_use = ToolUseBlock::builder()
            .tool_use_id("call_123")
            .name("get_weather")
            .input(Document::Object(
                vec![("city".to_string(), Document::String("Paris".to_string()))]
                    .into_iter()
                    .collect(),
            ))
            .build()
            .unwrap();

        let msg = Message::builder()
            .role(ConversationRole::Assistant)
            .content(ContentBlock::Text("Let me check the weather.".to_string()))
            .content(ContentBlock::ToolUse(tool_use))
            .build()
            .unwrap();

        let output = ConverseOutput::Message(msg);
        let (text, tool_calls) = BedrockProvider::extract_content(&output);
        assert_eq!(text, "Let me check the weather.");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_123");
        assert_eq!(tool_calls[0].call_type, "function");
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert!(tool_calls[0].function.arguments.contains("Paris"));
    }

    #[test]
    fn test_build_tool_config_empty_tools() {
        let result = BedrockProvider::build_tool_config(&[], None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_build_tool_config_auto_none_returns_none() {
        let tools = vec![EdgequakeToolDefinition::function(
            "test_fn",
            "A test function",
            serde_json::json!({"type": "object", "properties": {}}),
        )];
        let choice = EdgequakeToolChoice::none();
        let result = BedrockProvider::build_tool_config(&tools, Some(&choice)).unwrap();
        assert!(
            result.is_none(),
            "tool_choice='none' should omit tool config"
        );
    }

    #[test]
    fn test_build_tool_config_with_tools() {
        let tools = vec![EdgequakeToolDefinition::function(
            "search",
            "Search the web",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }),
        )];
        let result = BedrockProvider::build_tool_config(&tools, None).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_provider_name_and_model() {
        // We can't easily construct a BedrockProvider without AWS config in tests,
        // but we can test the static helper methods which don't require a client.
        assert_eq!(
            BedrockProvider::context_length_for_model("anthropic.claude-3-5-haiku-20241022-v1:0"),
            200_000
        );
    }

    #[test]
    fn test_with_model_updates_context() {
        // Verify that context_length_for_model returns different values for different models
        let claude =
            BedrockProvider::context_length_for_model("anthropic.claude-3-5-sonnet-20241022-v2:0");
        let nova = BedrockProvider::context_length_for_model("amazon.nova-pro-v1:0");
        let llama = BedrockProvider::context_length_for_model("meta.llama3-70b-instruct-v1:0");
        assert_ne!(claude, nova);
        assert_ne!(nova, llama);
    }
}
