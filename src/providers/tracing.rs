//! Tracing Provider Wrapper for OpenTelemetry GenAI Observability
//!
//! OODA-LOG-03: Provides distributed tracing for all LLM provider calls
//! following OpenTelemetry GenAI semantic conventions.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────┐
//! │  Agent / CLI        │
//! └─────────┬───────────┘
//!           │ chat(), complete()
//!           ▼
//! ┌─────────────────────┐
//! │  TracingProvider    │  ← Creates spans with GenAI attributes
//! │  • gen_ai.operation │
//! │  • gen_ai.system    │
//! │  • gen_ai.request.* │
//! │  • gen_ai.usage.*   │
//! └─────────┬───────────┘
//!           │ delegates
//!           ▼
//! ┌─────────────────────┐
//! │  Inner Provider     │
//! │  (OpenAI, Azure,    │
//! │   Gemini, etc.)     │
//! └─────────────────────┘
//! ```
//!
//! # GenAI Semantic Conventions
//!
//! Follows: <https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/>
//!
//! | Attribute | Description |
//! |-----------|-------------|
//! | gen_ai.operation.name | "chat", "embeddings", etc. |
//! | gen_ai.system | Provider name (openai, azure, gemini) |
//! | gen_ai.request.model | Requested model name |
//! | gen_ai.request.max_tokens | Max tokens requested |
//! | gen_ai.request.temperature | Temperature setting |
//! | gen_ai.response.model | Actual model used |
//! | gen_ai.usage.input_tokens | Prompt tokens |
//! | gen_ai.usage.output_tokens | Completion tokens |
//! | gen_ai.response.finish_reasons | Finish reason array |

use async_trait::async_trait;
use futures::stream::BoxStream;
use tracing::{info_span, Instrument};

use crate::application_context::ApplicationContext;
use crate::error::Result;
use crate::providers::genai_events;
use crate::traits::{
    ChatMessage, CompletionOptions, LLMProvider, LLMResponse, StreamChunk, ToolChoice,
    ToolDefinition,
};

/// GenAI semantic convention attribute names.
///
/// From: <https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/>
pub mod genai_attrs {
    /// The name of the operation being performed.
    pub const OPERATION_NAME: &str = "gen_ai.operation.name";
    /// The name of the GenAI system (provider).
    pub const SYSTEM: &str = "gen_ai.system";
    /// The requested model name.
    pub const REQUEST_MODEL: &str = "gen_ai.request.model";
    /// Maximum tokens to generate.
    pub const REQUEST_MAX_TOKENS: &str = "gen_ai.request.max_tokens";
    /// Temperature for generation.
    pub const REQUEST_TEMPERATURE: &str = "gen_ai.request.temperature";
    /// Top-p sampling parameter.
    pub const REQUEST_TOP_P: &str = "gen_ai.request.top_p";
    /// The actual model used in response.
    pub const RESPONSE_MODEL: &str = "gen_ai.response.model";
    /// Input/prompt token count.
    pub const USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
    /// Output/completion token count.
    pub const USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
    /// Finish reasons array.
    pub const RESPONSE_FINISH_REASONS: &str = "gen_ai.response.finish_reasons";

    // OODA-15: Reasoning/Thinking token attributes
    /// Number of reasoning/thinking tokens used by the model.
    ///
    /// Custom extension following OTel GenAI naming conventions.
    /// OpenAI o-series: extracted from output_tokens_details.reasoning_tokens
    /// Anthropic Claude: derived from thinking block token count
    pub const USAGE_REASONING_TOKENS: &str = "gen_ai.usage.reasoning_tokens";

    /// Reasoning/thinking content (opt-in, may be large).
    ///
    /// Contains the model's internal reasoning text if available
    /// and content capture is enabled.
    pub const REASONING_CONTENT: &str = "gen_ai.reasoning.content";

    /// Application identifier for attribution (proposed GenAI convention).
    pub const APPLICATION_ID: &str = "gen_ai.application.id";
    /// Application display name.
    pub const APPLICATION_NAME: &str = "gen_ai.application.name";
    /// Application public URL.
    pub const APPLICATION_URL: &str = "gen_ai.application.url";
    /// Multi-tenant partition (OTEL general convention).
    pub const TENANT_ID: &str = "tenant.id";
}

/// Check if content capture is enabled via environment variable.
///
/// Returns true if EDGECODE_CAPTURE_CONTENT is set to "true" or "1".
/// This is privacy-by-default - content is NOT captured unless explicitly enabled.
fn should_capture_content() -> bool {
    std::env::var("EDGECODE_CAPTURE_CONTENT")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false)
}

/// A wrapper that adds OpenTelemetry tracing to any LLM provider.
///
/// Implements the decorator pattern to add GenAI semantic convention
/// spans around all provider calls without modifying the inner provider.
///
/// # Example
///
/// ```ignore
/// use edgequake_llm::providers::{OpenAIProvider, TracingProvider};
///
/// let provider = OpenAIProvider::new("api-key");
/// let traced = TracingProvider::new(provider);
///
/// // All calls now generate spans with GenAI attributes
/// let response = traced.chat(&messages, None).await?;
/// ```
pub struct TracingProvider<P: LLMProvider> {
    inner: P,
    application_context: Option<ApplicationContext>,
}

impl<P: LLMProvider> TracingProvider<P> {
    /// Create a new tracing wrapper around the given provider.
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            application_context: None,
        }
    }

    /// Attach application attribution fields to emitted spans.
    pub fn with_application_context(mut self, ctx: ApplicationContext) -> Self {
        self.application_context = Some(ctx);
        self
    }

    fn record_application_context(&self, span: &tracing::Span) {
        if let Some(ref ctx) = self.application_context {
            if let Some(ref id) = ctx.app_id {
                span.record(genai_attrs::APPLICATION_ID, id.as_str());
            }
            if let Some(ref name) = ctx.app_name {
                span.record(genai_attrs::APPLICATION_NAME, name.as_str());
            }
            if let Some(ref url) = ctx.app_url {
                span.record(genai_attrs::APPLICATION_URL, url.as_str());
            }
            if let Some(ref tid) = ctx.tenant_id {
                span.record(genai_attrs::TENANT_ID, tid.as_str());
            }
        }
    }

    fn prepare_span(&self, span: &tracing::Span) {
        if let Some(ref ctx) = self.application_context {
            self.record_application_context(span);
            crate::http::attribution::promote_application_context_to_baggage(ctx);
            let kind =
                crate::http::attribution::attribution_kind_from_provider_name(self.inner.name());
            crate::http::attribution::record_attribution_span_events(kind, ctx, self.inner.name());
        }
    }

    /// Get a reference to the inner provider.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Consume the wrapper and return the inner provider.
    pub fn into_inner(self) -> P {
        self.inner
    }
}

#[async_trait]
impl<P: LLMProvider> LLMProvider for TracingProvider<P> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn model(&self) -> &str {
        self.inner.model()
    }

    fn max_context_length(&self) -> usize {
        self.inner.max_context_length()
    }

    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        let span = info_span!(
            "gen_ai.complete",
            { genai_attrs::OPERATION_NAME } = "complete",
            { genai_attrs::SYSTEM } = self.inner.name(),
            { genai_attrs::REQUEST_MODEL } = self.inner.model(),
            prompt_length = prompt.len(),
            "gen_ai.prompt" = tracing::field::Empty,
            "gen_ai.completion.content" = tracing::field::Empty,
            "langfuse.observation.input" = tracing::field::Empty,
            "langfuse.observation.output" = tracing::field::Empty,
        );

        self.prepare_span(&span);

        // OODA-LOG-09: Capture prompt content if enabled
        if should_capture_content() {
            span.record("gen_ai.prompt", prompt);
            span.record("langfuse.observation.input", prompt);
        }

        let response = self.inner.complete(prompt).instrument(span.clone()).await?;

        // Record response attributes
        span.record(genai_attrs::RESPONSE_MODEL, &response.model);
        span.record(
            genai_attrs::USAGE_INPUT_TOKENS,
            response.prompt_tokens as i64,
        );
        span.record(
            genai_attrs::USAGE_OUTPUT_TOKENS,
            response.completion_tokens as i64,
        );
        if let Some(reason) = &response.finish_reason {
            span.record(genai_attrs::RESPONSE_FINISH_REASONS, reason.as_str());
        }

        // OODA-LOG-09: Capture response content if enabled
        if should_capture_content() {
            span.record("gen_ai.completion.content", response.content.as_str());
            span.record("langfuse.observation.output", response.content.as_str());
        }

        Ok(response)
    }

    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<LLMResponse> {
        let span = info_span!(
            "gen_ai.complete",
            { genai_attrs::OPERATION_NAME } = "complete",
            { genai_attrs::SYSTEM } = self.inner.name(),
            { genai_attrs::REQUEST_MODEL } = self.inner.model(),
            { genai_attrs::REQUEST_MAX_TOKENS } = options.max_tokens.map(|t| t as i64),
            { genai_attrs::REQUEST_TEMPERATURE } = options.temperature.map(|t| t as f64),
            { genai_attrs::REQUEST_TOP_P } = options.top_p.map(|t| t as f64),
            prompt_length = prompt.len(),
            "gen_ai.prompt" = tracing::field::Empty,
            "gen_ai.completion.content" = tracing::field::Empty,
            "langfuse.observation.input" = tracing::field::Empty,
            "langfuse.observation.output" = tracing::field::Empty,
        );

        self.prepare_span(&span);

        // OODA-LOG-09: Capture prompt content if enabled
        if should_capture_content() {
            span.record("gen_ai.prompt", prompt);
            span.record("langfuse.observation.input", prompt);
        }

        let response = self
            .inner
            .complete_with_options(prompt, options)
            .instrument(span.clone())
            .await?;

        span.record(genai_attrs::RESPONSE_MODEL, &response.model);
        span.record(
            genai_attrs::USAGE_INPUT_TOKENS,
            response.prompt_tokens as i64,
        );
        span.record(
            genai_attrs::USAGE_OUTPUT_TOKENS,
            response.completion_tokens as i64,
        );
        if let Some(reason) = &response.finish_reason {
            span.record(genai_attrs::RESPONSE_FINISH_REASONS, reason.as_str());
        }

        // OODA-LOG-09: Capture response content if enabled
        if should_capture_content() {
            span.record("gen_ai.completion.content", response.content.as_str());
            span.record("langfuse.observation.output", response.content.as_str());
        }

        // OODA-LOG-11: Emit GenAI event with prompt and completion content
        let input_messages = vec![ChatMessage {
            role: crate::traits::ChatRole::User,
            content: prompt.to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            cache_control: None,
            images: None,
        }];
        let output_messages = vec![ChatMessage {
            role: crate::traits::ChatRole::Assistant,
            content: response.content.clone(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            cache_control: None,
            images: None,
        }];

        // Emit GenAI event - use async block with instrument to ensure it's in span context
        async {
            genai_events::emit_inference_event(
                &input_messages,
                &output_messages,
                &response,
                Some(options),
            );
        }
        .instrument(span)
        .await;

        Ok(response)
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let max_tokens = options.and_then(|o| o.max_tokens).map(|t| t as i64);
        let temperature = options.and_then(|o| o.temperature).map(|t| t as f64);
        let top_p = options.and_then(|o| o.top_p).map(|t| t as f64);

        let span = info_span!(
            "gen_ai.chat",
            { genai_attrs::OPERATION_NAME } = "chat",
            { genai_attrs::SYSTEM } = self.inner.name(),
            { genai_attrs::REQUEST_MODEL } = self.inner.model(),
            { genai_attrs::REQUEST_MAX_TOKENS } = max_tokens,
            { genai_attrs::REQUEST_TEMPERATURE } = temperature,
            { genai_attrs::REQUEST_TOP_P } = top_p,
            message_count = messages.len(),
            // Response fields - to be filled after completion
            { genai_attrs::RESPONSE_MODEL } = tracing::field::Empty,
            { genai_attrs::USAGE_INPUT_TOKENS } = tracing::field::Empty,
            { genai_attrs::USAGE_OUTPUT_TOKENS } = tracing::field::Empty,
            { genai_attrs::USAGE_REASONING_TOKENS } = tracing::field::Empty,
            { genai_attrs::RESPONSE_FINISH_REASONS } = tracing::field::Empty,
            "gen_ai.prompt" = tracing::field::Empty,
            "gen_ai.completion.content" = tracing::field::Empty,
            { genai_attrs::REASONING_CONTENT } = tracing::field::Empty,
            "langfuse.observation.input" = tracing::field::Empty,
            "langfuse.observation.output" = tracing::field::Empty,
        );

        self.prepare_span(&span);

        // OODA-LOG-09: Capture message content if enabled (opt-in for privacy)
        if should_capture_content() {
            // Serialize messages as JSON for easier viewing in Jaeger
            if let Ok(messages_json) = serde_json::to_string(messages) {
                span.record("gen_ai.prompt", messages_json.as_str());
                span.record("langfuse.observation.input", messages_json.as_str());
            }
        }

        let response = self
            .inner
            .chat(messages, options)
            .instrument(span.clone())
            .await?;

        // Record response attributes after completion
        span.record(genai_attrs::RESPONSE_MODEL, &response.model);
        span.record(
            genai_attrs::USAGE_INPUT_TOKENS,
            response.prompt_tokens as i64,
        );
        span.record(
            genai_attrs::USAGE_OUTPUT_TOKENS,
            response.completion_tokens as i64,
        );

        // OODA-15: Record reasoning/thinking tokens if present
        if let Some(thinking_tokens) = response.thinking_tokens {
            span.record(genai_attrs::USAGE_REASONING_TOKENS, thinking_tokens as i64);
        }

        if let Some(reason) = &response.finish_reason {
            span.record(genai_attrs::RESPONSE_FINISH_REASONS, reason.as_str());
        }

        // OODA-LOG-09: Capture response content if enabled
        if should_capture_content() {
            span.record("gen_ai.completion.content", response.content.as_str());
            span.record("langfuse.observation.output", response.content.as_str());

            // OODA-15: Capture thinking content if available and content capture enabled
            if let Some(thinking) = &response.thinking_content {
                // Truncate to 10KB to prevent trace bloat
                let truncated = if thinking.len() > 10240 {
                    // Find valid UTF-8 boundary near 10240 bytes
                    let truncate_at = thinking
                        .char_indices()
                        .take_while(|(idx, _)| *idx < 10240)
                        .last()
                        .map(|(idx, c)| idx + c.len_utf8())
                        .unwrap_or(0);
                    format!("{}...[truncated]", &thinking[..truncate_at])
                } else {
                    thinking.clone()
                };
                span.record(genai_attrs::REASONING_CONTENT, truncated.as_str());
            }
        }

        // OODA-LOG-11: Emit GenAI event with input and output messages
        let output_messages = vec![ChatMessage {
            role: crate::traits::ChatRole::Assistant,
            content: response.content.clone(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            cache_control: None,
            images: None,
        }];

        // Emit GenAI event - use async block with instrument to ensure it's in span context
        async {
            genai_events::emit_inference_event(messages, &output_messages, &response, options);
        }
        .instrument(span)
        .await;

        tracing::info!(
            target: "gen_ai.usage",
            input_tokens = response.prompt_tokens,
            output_tokens = response.completion_tokens,
            model = %response.model,
            "LLM chat completion"
        );

        Ok(response)
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let max_tokens = options.and_then(|o| o.max_tokens).map(|t| t as i64);
        let temperature = options.and_then(|o| o.temperature).map(|t| t as f64);

        let span = info_span!(
            "gen_ai.chat_with_tools",
            { genai_attrs::OPERATION_NAME } = "chat_with_tools",
            { genai_attrs::SYSTEM } = self.inner.name(),
            { genai_attrs::REQUEST_MODEL } = self.inner.model(),
            { genai_attrs::REQUEST_MAX_TOKENS } = max_tokens,
            { genai_attrs::REQUEST_TEMPERATURE } = temperature,
            message_count = messages.len(),
            tool_count = tools.len(),
            { genai_attrs::RESPONSE_MODEL } = tracing::field::Empty,
            { genai_attrs::USAGE_INPUT_TOKENS } = tracing::field::Empty,
            { genai_attrs::USAGE_OUTPUT_TOKENS } = tracing::field::Empty,
            { genai_attrs::USAGE_REASONING_TOKENS } = tracing::field::Empty,
            { genai_attrs::RESPONSE_FINISH_REASONS } = tracing::field::Empty,
            "gen_ai.prompt" = tracing::field::Empty,
            "gen_ai.completion.content" = tracing::field::Empty,
            { genai_attrs::REASONING_CONTENT } = tracing::field::Empty,
            "langfuse.observation.input" = tracing::field::Empty,
            "langfuse.observation.output" = tracing::field::Empty,
            "langfuse.observation.type" = "generation",
        );

        self.prepare_span(&span);

        // OODA-LOG-09: Capture message content if enabled
        if should_capture_content() {
            // Serialize messages as JSON for easier viewing in Jaeger
            if let Ok(messages_json) = serde_json::to_string(messages) {
                span.record("gen_ai.prompt", messages_json.as_str());
                span.record("langfuse.observation.input", messages_json.as_str());
            }
        }

        let response = self
            .inner
            .chat_with_tools(messages, tools, tool_choice, options)
            .instrument(span.clone())
            .await?;

        span.record(genai_attrs::RESPONSE_MODEL, &response.model);
        span.record(
            genai_attrs::USAGE_INPUT_TOKENS,
            response.prompt_tokens as i64,
        );
        span.record(
            genai_attrs::USAGE_OUTPUT_TOKENS,
            response.completion_tokens as i64,
        );

        // OODA-15: Record reasoning/thinking tokens if present
        if let Some(thinking_tokens) = response.thinking_tokens {
            span.record(genai_attrs::USAGE_REASONING_TOKENS, thinking_tokens as i64);
        }

        if let Some(reason) = &response.finish_reason {
            span.record(genai_attrs::RESPONSE_FINISH_REASONS, reason.as_str());
        }

        // OODA-LOG-09: Capture response content if enabled
        if should_capture_content() {
            // OODA-22: For tool call responses, serialize tool calls as output
            let output_content = if !response.tool_calls.is_empty() {
                // When the response contains tool calls, serialize them as JSON output
                serde_json::to_string(&response.tool_calls)
                    .unwrap_or_else(|_| format!("{} tool call(s)", response.tool_calls.len()))
            } else {
                // Normal text content
                response.content.clone()
            };

            span.record("gen_ai.completion.content", output_content.as_str());
            span.record("langfuse.observation.output", output_content.as_str());

            // OODA-15: Capture thinking content if available
            if let Some(thinking) = &response.thinking_content {
                let truncated = if thinking.len() > 10240 {
                    // Find valid UTF-8 boundary near 10240 bytes
                    let truncate_at = thinking
                        .char_indices()
                        .take_while(|(idx, _)| *idx < 10240)
                        .last()
                        .map(|(idx, c)| idx + c.len_utf8())
                        .unwrap_or(0);
                    format!("{}...[truncated]", &thinking[..truncate_at])
                } else {
                    thinking.clone()
                };
                span.record(genai_attrs::REASONING_CONTENT, truncated.as_str());
            }
        }

        tracing::info!(
            target: "gen_ai.usage",
            input_tokens = response.prompt_tokens,
            output_tokens = response.completion_tokens,
            model = %response.model,
            tool_calls = response.tool_calls.len(),
            "LLM chat with tools completion"
        );

        Ok(response)
    }

    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        let span = info_span!(
            "gen_ai.stream",
            { genai_attrs::OPERATION_NAME } = "stream",
            { genai_attrs::SYSTEM } = self.inner.name(),
            { genai_attrs::REQUEST_MODEL } = self.inner.model(),
            prompt_length = prompt.len(),
            "gen_ai.prompt" = tracing::field::Empty,
        );

        self.prepare_span(&span);

        // OODA-LOG-09: Capture prompt content if enabled
        if should_capture_content() {
            span.record("gen_ai.prompt", prompt);
        }

        // Note: Streaming doesn't give us token counts until the end,
        // so we can't record usage in the span for now
        self.inner.stream(prompt).instrument(span).await
    }

    async fn chat_with_tools_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        let max_tokens = options.and_then(|o| o.max_tokens).map(|t| t as i64);
        let temperature = options.and_then(|o| o.temperature).map(|t| t as f64);

        let span = info_span!(
            "gen_ai.stream_with_tools",
            { genai_attrs::OPERATION_NAME } = "stream_with_tools",
            { genai_attrs::SYSTEM } = self.inner.name(),
            { genai_attrs::REQUEST_MODEL } = self.inner.model(),
            { genai_attrs::REQUEST_MAX_TOKENS } = max_tokens,
            { genai_attrs::REQUEST_TEMPERATURE } = temperature,
            message_count = messages.len(),
            tool_count = tools.len(),
            "gen_ai.prompt" = tracing::field::Empty,
            "langfuse.observation.input" = tracing::field::Empty,
            "langfuse.observation.type" = "generation",
        );

        self.prepare_span(&span);

        // OODA-LOG-09: Capture message content if enabled
        if should_capture_content() {
            // Serialize messages as JSON for easier viewing in Jaeger
            if let Ok(messages_json) = serde_json::to_string(messages) {
                span.record("gen_ai.prompt", messages_json.as_str());
                span.record("langfuse.observation.input", messages_json.as_str());
            }
        }

        // DEBUG: Log span creation
        tracing::info!(
            span_id = ?span.id(),
            span_name = "gen_ai.stream_with_tools",
            "Created LLM streaming span"
        );

        // NOTE: VsCodeCopilotProvider does not support tool streaming (supports_tool_streaming returns false)
        // so this path is currently unused by the React agent. The non-streaming chat_with_tools is used instead.
        // Keeping this implementation for future providers that support tool streaming.
        self.inner
            .chat_with_tools_stream(messages, tools, tool_choice, options)
            .instrument(span)
            .await
    }

    fn supports_streaming(&self) -> bool {
        self.inner.supports_streaming()
    }

    fn supports_tool_streaming(&self) -> bool {
        self.inner.supports_tool_streaming()
    }

    fn supports_json_mode(&self) -> bool {
        self.inner.supports_json_mode()
    }

    fn supports_function_calling(&self) -> bool {
        self.inner.supports_function_calling()
    }

    async fn refresh_model_metadata(&self) -> Result<()> {
        self.inner.refresh_model_metadata().await
    }

    fn default_max_output_tokens(&self) -> Option<usize> {
        self.inner.default_max_output_tokens()
    }

    fn model_name(&self) -> Option<String> {
        self.inner.model_name()
    }
}

/// Thin `Arc<dyn LLMProvider>` adapter so [`TracingProvider`] can decorate trait objects.
struct ArcDynProvider(std::sync::Arc<dyn LLMProvider>);

#[async_trait]
impl LLMProvider for ArcDynProvider {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn model(&self) -> &str {
        self.0.model()
    }

    fn max_context_length(&self) -> usize {
        self.0.max_context_length()
    }

    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        self.0.complete(prompt).await
    }

    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<LLMResponse> {
        self.0.complete_with_options(prompt, options).await
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        self.0.chat(messages, options).await
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        self.0
            .chat_with_tools(messages, tools, tool_choice, options)
            .await
    }

    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        self.0.stream(prompt).await
    }

    async fn chat_with_tools_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        self.0
            .chat_with_tools_stream(messages, tools, tool_choice, options)
            .await
    }

    fn supports_streaming(&self) -> bool {
        self.0.supports_streaming()
    }

    fn supports_tool_streaming(&self) -> bool {
        self.0.supports_tool_streaming()
    }

    fn supports_json_mode(&self) -> bool {
        self.0.supports_json_mode()
    }

    fn supports_function_calling(&self) -> bool {
        self.0.supports_function_calling()
    }

    async fn refresh_model_metadata(&self) -> Result<()> {
        self.0.refresh_model_metadata().await
    }

    fn default_max_output_tokens(&self) -> Option<usize> {
        self.0.default_max_output_tokens()
    }

    fn model_name(&self) -> Option<String> {
        self.0.model_name()
    }
}

/// Wrap an `Arc<dyn LLMProvider>` with [`TracingProvider`] for GenAI spans.
pub fn trace_llm_arc(provider: std::sync::Arc<dyn LLMProvider>) -> std::sync::Arc<dyn LLMProvider> {
    std::sync::Arc::new(TracingProvider::new(ArcDynProvider(provider)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::mock::MockProvider;
    use std::sync::Arc;

    #[test]
    fn trace_llm_arc_preserves_identity() {
        let inner: Arc<dyn LLMProvider> = Arc::new(MockProvider::new());
        let wrapped = trace_llm_arc(inner);
        assert_eq!(wrapped.name(), "mock");
        assert_eq!(wrapped.model(), "mock-model");
    }

    #[test]
    fn test_tracing_provider_delegates_name() {
        let mock = MockProvider::new();
        let traced = TracingProvider::new(mock);
        assert_eq!(traced.name(), "mock");
    }

    #[test]
    fn test_tracing_provider_delegates_model() {
        let mock = MockProvider::new();
        let traced = TracingProvider::new(mock);
        assert_eq!(traced.model(), "mock-model");
    }

    #[test]
    fn test_tracing_provider_delegates_max_context_length() {
        let mock = MockProvider::new();
        let traced = TracingProvider::new(mock);
        assert_eq!(traced.max_context_length(), 4096);
    }

    #[test]
    fn test_into_inner() {
        let mock = MockProvider::new();
        let traced = TracingProvider::new(mock);
        let inner = traced.into_inner();
        assert_eq!(inner.name(), "mock");
    }

    #[tokio::test]
    async fn test_complete_creates_span() {
        let mock = MockProvider::new();
        let traced = TracingProvider::new(mock);

        let response = traced.complete("Hello").await.unwrap();
        assert_eq!(response.content, "Mock response");
    }

    #[tokio::test]
    async fn test_chat_creates_span() {
        let mock = MockProvider::new();
        let traced = TracingProvider::new(mock);

        let messages = vec![ChatMessage::user("Hello")];
        let response = traced.chat(&messages, None).await.unwrap();
        assert_eq!(response.content, "Mock response");
    }

    #[tokio::test]
    async fn test_complete_with_options_creates_span() {
        let mock = MockProvider::new();
        let traced = TracingProvider::new(mock);

        let options = CompletionOptions {
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: Some(0.9),
            ..Default::default()
        };

        let response = traced
            .complete_with_options("Prompt", &options)
            .await
            .unwrap();
        assert_eq!(response.content, "Mock response");
    }

    // ---- Iteration 26: Additional tracing provider tests ----

    #[test]
    fn test_inner_accessor() {
        let mock = MockProvider::new();
        let traced = TracingProvider::new(mock);
        assert_eq!(traced.inner().name(), "mock");
    }

    #[test]
    fn test_supports_streaming_delegation() {
        let mock = MockProvider::new();
        let traced = TracingProvider::new(mock);
        // MockProvider default: false
        assert!(!traced.supports_streaming());
    }

    #[test]
    fn test_supports_json_mode_delegation() {
        let mock = MockProvider::new();
        let traced = TracingProvider::new(mock);
        assert!(!traced.supports_json_mode());
    }

    #[test]
    fn test_supports_function_calling_delegation() {
        let mock = MockProvider::new();
        let traced = TracingProvider::new(mock);
        assert!(!traced.supports_function_calling());
    }

    #[test]
    fn test_supports_tool_streaming_delegation() {
        let mock = MockProvider::new();
        let traced = TracingProvider::new(mock);
        assert!(!traced.supports_tool_streaming());
    }

    #[test]
    fn test_genai_attrs_constants() {
        assert_eq!(genai_attrs::OPERATION_NAME, "gen_ai.operation.name");
        assert_eq!(genai_attrs::SYSTEM, "gen_ai.system");
        assert_eq!(genai_attrs::REQUEST_MODEL, "gen_ai.request.model");
        assert_eq!(genai_attrs::REQUEST_MAX_TOKENS, "gen_ai.request.max_tokens");
        assert_eq!(
            genai_attrs::REQUEST_TEMPERATURE,
            "gen_ai.request.temperature"
        );
        assert_eq!(genai_attrs::REQUEST_TOP_P, "gen_ai.request.top_p");
        assert_eq!(genai_attrs::RESPONSE_MODEL, "gen_ai.response.model");
        assert_eq!(genai_attrs::USAGE_INPUT_TOKENS, "gen_ai.usage.input_tokens");
        assert_eq!(
            genai_attrs::USAGE_OUTPUT_TOKENS,
            "gen_ai.usage.output_tokens"
        );
        assert_eq!(
            genai_attrs::RESPONSE_FINISH_REASONS,
            "gen_ai.response.finish_reasons"
        );
        assert_eq!(
            genai_attrs::USAGE_REASONING_TOKENS,
            "gen_ai.usage.reasoning_tokens"
        );
        assert_eq!(genai_attrs::REASONING_CONTENT, "gen_ai.reasoning.content");
    }

    #[test]
    fn test_should_capture_content_default_false() {
        // When env var is not set, should be false
        std::env::remove_var("EDGECODE_CAPTURE_CONTENT");
        assert!(!should_capture_content());
    }

    #[tokio::test]
    async fn test_chat_with_options() {
        let mock = MockProvider::new();
        let traced = TracingProvider::new(mock);

        let messages = vec![ChatMessage::user("Hello")];
        let options = CompletionOptions {
            max_tokens: Some(50),
            temperature: Some(0.5),
            ..Default::default()
        };

        let response = traced.chat(&messages, Some(&options)).await.unwrap();
        assert_eq!(response.content, "Mock response");
    }

    #[tokio::test]
    async fn test_chat_with_tools_delegation() {
        use crate::providers::mock::MockAgentProvider;

        let mock = MockAgentProvider::new();
        mock.add_response("tool response").await;

        let traced = TracingProvider::new(mock);
        let messages = vec![ChatMessage::user("use tools")];
        let response = traced
            .chat_with_tools(&messages, &[], None, None)
            .await
            .unwrap();
        assert_eq!(response.content, "tool response");
    }

    #[tokio::test]
    async fn test_stream_delegation() {
        use futures::StreamExt;

        let mock = MockProvider::new();
        mock.add_response("streamed").await;
        let traced = TracingProvider::new(mock);

        let mut stream = traced.stream("prompt").await.unwrap();
        let chunk = stream.next().await.unwrap().unwrap();
        assert_eq!(chunk, "streamed");
    }

    #[tokio::test]
    async fn test_complete_with_queued_response() {
        let mock = MockProvider::new();
        mock.add_response("custom traced").await;
        let traced = TracingProvider::new(mock);

        let response = traced.complete("Hi").await.unwrap();
        assert_eq!(response.content, "custom traced");
    }
}
