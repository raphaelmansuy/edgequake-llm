//! xAI Grok Provider - Direct access to xAI's Grok models.
//!
//! @implements OODA-71: xAI Grok API Integration
//!
//! # Overview
//!
//! This provider connects directly to xAI's API (api.x.ai) for Grok models.
//! xAI's API is OpenAI-compatible, so we leverage `OpenAICompatibleProvider`
//! internally for maximum code reuse and battle-tested functionality.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    xAI Provider Architecture                            │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │   User Request                                                           │
//! │        │                                                                 │
//! │        ▼                                                                 │
//! │  ┌─────────────┐  ┌──────────────────────────┐  ┌──────────────────┐   │
//! │  │ XAIProvider  │─►│ OpenAICompatibleProvider │─►│ api.x.ai         │   │
//! │  │  (wrapper)   │  │  (implementation)        │  │ /v1/chat/*       │   │
//! │  └─────────────┘  └──────────────────────────┘  └──────────────────┘   │
//! │        │                                                                 │
//! │        └─ strips prohibited params (presence_penalty,                    │
//! │           frequency_penalty, stop) before delegating                     │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Model Consolidation (May 15, 2026)
//!
//! As of May 15, 2026, all previous Grok models have been retired. The **only
//! active model** is `grok-4.3` (1M context). Legacy slugs (`grok-3`, `grok-4`,
//! `grok-4.20`, `grok-4-1-fast`, etc.) still resolve via API redirect but all
//! point to `grok-4.3`.
//!
//! `grok-4.3` is always a reasoning model. Reasoning effort is configurable
//! via the first-class `reasoning.effort` parameter (none / low / medium / high),
//! so `reasoning_effort` is **no longer stripped**.
//!
//! The xAI API still returns HTTP 400 for these parameters on `grok-4.3`:
//!
//! - `presence_penalty`
//! - `frequency_penalty`
//! - `stop` (stop sequences)
//!
//! `XAIProvider` automatically strips these fields before delegating to the
//! inner `OpenAICompatibleProvider`, so callers need not worry about this.
//!
//! # Environment Variables
//!
//! | Variable | Required | Default | Description |
//! |----------|----------|---------|-------------|
//! | `XAI_API_KEY` | ✅ Yes | - | xAI API key from console.x.ai |
//! | `XAI_MODEL` | ❌ No | `grok-4.3` | Default model to use |
//! | `XAI_BASE_URL` | ❌ No | `https://api.x.ai/v1` | API endpoint override |
//!
//! # Available Models (as of July 2026, docs.x.ai)
//!
//! | Model | Context | Features |
//! |-------|---------|----------|
//! | `grok-4.3` | 1M | Flagship (reasoning, vision, tools) |
//! | `grok-build-0.1` | 256K | Fast coding model (early access) |
//!
//! Legacy aliases (`grok-4.3-latest`, `grok-latest`, `grok-3`, `grok-4`,
//! `grok-4-0709`, `grok-4.20`, etc.) all redirect to `grok-4.3` at the API
//! level. They are kept in the model catalog for backward compatibility.
//!
//! Pricing (grok-4.3): $1.25 input / $2.50 output per 1M tokens.
//!
//! # Example
//!
//! ```bash
//! # Set API key
//! export XAI_API_KEY=xai-your-api-key
//!
//! # Use with EdgeCode (auto-detected)
//! edgecode react "Write hello world in Rust"
//!
//! # Explicit provider selection
//! edgecode react --provider xai "Write hello world in Rust"
//!
//! # Use coding model
//! export XAI_MODEL=grok-build-0.1
//! edgecode react "Build a complex app"
//! ```

use async_trait::async_trait;
use futures::stream::BoxStream;
use tracing::{debug, warn};

use crate::error::{LlmError, Result};
use crate::model_config::{
    ModelCapabilities, ModelCard, ModelType, ProviderConfig, ProviderType as ConfigProviderType,
};
use crate::providers::openai_compatible::OpenAICompatibleProvider;
use crate::traits::StreamChunk;
use crate::traits::{
    ChatMessage, CompletionOptions, EmbeddingProvider, LLMProvider, LLMResponse, ToolChoice,
    ToolDefinition,
};

// ============================================================================
// Constants
// ============================================================================

/// Default xAI API base URL (includes /v1 prefix for OpenAI compatibility)
const XAI_BASE_URL: &str = "https://api.x.ai/v1";

/// Default model — Grok 4.3 (the only active model as of May 15, 2026).
///
/// Grok 4.3 is the sole production model. It supports 1M context, function
/// calling, structured outputs, vision, and configurable reasoning effort.
const XAI_DEFAULT_MODEL: &str = "grok-4.3";

/// Provider display name
const XAI_PROVIDER_NAME: &str = "xai";

/// xAI model catalog with context lengths.
///
/// WHY: Pre-defined models ensure users get correct context limits without
/// having to check documentation.  Context lengths and capabilities are
/// sourced from docs.x.ai (July 2026).
///
/// ## Current state (post May 15, 2026 consolidation)
///
/// Only two models are actively served:
///
/// - `grok-4.3`       — sole flagship (1M context, reasoning, vision, tools)
/// - `grok-build-0.1` — fast coding model (256K context, early access)
///
/// All legacy slugs (`grok-3`, `grok-4`, `grok-4.20`, `grok-4-0709`,
/// `grok-4-1-fast`, etc.) still resolve at the API level but redirect to
/// `grok-4.3`. They are kept here so that users with `XAI_MODEL` set to an
/// old slug get correct context length (1M, matching the redirect target).
///
/// ## Reasoning
///
/// `grok-4.3` (and all aliases that redirect to it) is always a reasoning
/// model. Reasoning effort is controlled via the first-class
/// `reasoning.effort` parameter (none/low/medium/high). The API still
/// rejects `presence_penalty`, `frequency_penalty`, and `stop`.
///
/// Last updated: July 2026 (docs.x.ai)
const XAI_MODELS: &[(&str, &str, usize)] = &[
    // ---- Active models (July 2026) -----------------------------------------
    (
        "grok-4.3",
        "Grok 4.3 (Flagship, 1M, reasoning+vision)",
        1_000_000,
    ),
    ("grok-4.3-latest", "Grok 4.3 Latest (1M)", 1_000_000),
    ("grok-latest", "Grok Latest → 4.3 (1M)", 1_000_000),
    (
        "grok-build-0.1",
        "Grok Build 0.1 (Fast Coding, 256K, early access)",
        262_144,
    ),
    // ---- Legacy aliases (all redirect to grok-4.3 since May 15, 2026) ------
    ("grok-4.20", "Grok 4.20 → 4.3 (legacy, 1M)", 1_000_000),
    (
        "grok-4.20-latest",
        "Grok 4.20 Latest → 4.3 (legacy, 1M)",
        1_000_000,
    ),
    ("grok-4", "Grok 4 → 4.3 (legacy, 1M)", 1_000_000),
    ("grok-4-0709", "Grok 4 0709 → 4.3 (legacy, 1M)", 1_000_000),
    (
        "grok-4-latest",
        "Grok 4 Latest → 4.3 (legacy, 1M)",
        1_000_000,
    ),
    (
        "grok-4-1-fast",
        "Grok 4.1 Fast → 4.3 (legacy, 1M)",
        1_000_000,
    ),
    ("grok-3", "Grok 3 → 4.3 (legacy, 1M)", 1_000_000),
    (
        "grok-3-latest",
        "Grok 3 Latest → 4.3 (legacy, 1M)",
        1_000_000,
    ),
    ("grok-3-mini", "Grok 3 Mini → 4.3 (legacy, 1M)", 1_000_000),
    (
        "grok-3-mini-latest",
        "Grok 3 Mini Latest → 4.3 (legacy, 1M)",
        1_000_000,
    ),
];
// ============================================================================
// XAI Provider
// ============================================================================

/// xAI Grok provider for direct API access.
///
/// This is a thin wrapper around `OpenAICompatibleProvider` that provides:
/// - Automatic `XAI_API_KEY` detection
/// - Default configuration for xAI's API
/// - Model catalog with correct context sizes
/// - Automatic parameter stripping for reasoning models (see module docs)
///
/// # Why Wrap OpenAICompatibleProvider?
///
/// xAI's API is 100% OpenAI-compatible, so we get:
/// - Battle-tested HTTP client
/// - Streaming support
/// - Tool/function calling
/// - Vision (image input)
/// - JSON mode
/// - Error handling
/// - Retry logic
///
/// Without code duplication!
#[derive(Debug)]
pub struct XAIProvider {
    /// Inner OpenAI-compatible provider
    inner: OpenAICompatibleProvider,
    /// Current model name
    model: String,
}

impl XAIProvider {
    /// Create provider from environment variables.
    ///
    /// # Environment Variables
    ///
    /// - `XAI_API_KEY`: Required API key
    /// - `XAI_MODEL`: Model name (default: `grok-4.3`)
    /// - `XAI_BASE_URL`: Custom base URL (default: `https://api.x.ai/v1`)
    ///
    /// # Errors
    ///
    /// Returns error if `XAI_API_KEY` is not set.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("XAI_API_KEY").map_err(|_| {
            LlmError::ConfigError(
                "XAI_API_KEY environment variable not set. \
                 Get your API key from https://console.x.ai"
                    .to_string(),
            )
        })?;

        if api_key.is_empty() {
            return Err(LlmError::ConfigError(
                "XAI_API_KEY is empty. Please set a valid API key.".to_string(),
            ));
        }

        let model = std::env::var("XAI_MODEL").unwrap_or_else(|_| XAI_DEFAULT_MODEL.to_string());
        let base_url = std::env::var("XAI_BASE_URL").unwrap_or_else(|_| XAI_BASE_URL.to_string());

        Self::new(api_key, model, Some(base_url))
    }

    /// Create provider with explicit configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - xAI API key
    /// * `model` - Model name (e.g., "grok-4.3")
    /// * `base_url` - Optional custom base URL
    pub fn new(api_key: String, model: String, base_url: Option<String>) -> Result<Self> {
        // Build ProviderConfig for OpenAICompatibleProvider
        let config = Self::build_config(&api_key, &model, base_url.as_deref());

        // Create inner provider
        let inner = OpenAICompatibleProvider::from_config(config)?;

        debug!(
            provider = XAI_PROVIDER_NAME,
            model = %model,
            "Created xAI provider"
        );

        Ok(Self { inner, model })
    }

    /// Create with a different model.
    ///
    /// Returns a new provider instance configured for the specified model.
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self.inner = self.inner.with_model(model);
        self
    }

    /// Attach `ApplicationContext` for upstream attribution headers.
    pub fn with_application_context(
        mut self,
        ctx: crate::application_context::ApplicationContext,
    ) -> Self {
        self.inner = self.inner.with_application_context(ctx);
        self
    }

    /// Build ProviderConfig for OpenAICompatibleProvider.
    ///
    /// WHY: We need to set XAI_API_KEY env var before creating the provider because
    /// OpenAICompatibleProvider reads the API key from the environment variable
    /// specified in api_key_env, not from a config field.
    fn build_config(_api_key: &str, model: &str, base_url: Option<&str>) -> ProviderConfig {
        // Build model cards from XAI_MODELS with proper capabilities
        let models: Vec<ModelCard> = XAI_MODELS
            .iter()
            .map(|(name, display, context)| {
                // Vision: grok-4.3 (and all aliases redirecting to it) supports
                // image input. grok-build-0.1 is text-only.
                let supports_vision = *name != "grok-build-0.1";

                // Reasoning/thinking: grok-4.3 and all legacy redirects are
                // reasoning models. grok-build-0.1 is not.
                let supports_thinking = Self::is_reasoning_model(name);

                ModelCard {
                    name: name.to_string(),
                    display_name: display.to_string(),
                    model_type: ModelType::Llm,
                    capabilities: ModelCapabilities {
                        context_length: *context,
                        supports_function_calling: true,
                        supports_json_mode: true,
                        supports_streaming: true,
                        supports_system_message: true,
                        supports_vision,
                        supports_thinking,
                        ..Default::default()
                    },
                    ..Default::default()
                }
            })
            .collect();

        ProviderConfig {
            name: XAI_PROVIDER_NAME.to_string(),
            display_name: "xAI Grok".to_string(),
            provider_type: ConfigProviderType::OpenAICompatible,
            api_key_env: Some("XAI_API_KEY".to_string()),
            base_url: Some(base_url.unwrap_or(XAI_BASE_URL).to_string()),
            base_url_env: Some("XAI_BASE_URL".to_string()),
            default_llm_model: Some(model.to_string()),
            default_embedding_model: None, // xAI doesn't provide embeddings yet
            models,
            headers: std::collections::HashMap::new(),
            enabled: true,
            // WHY 600s: grok-4.3 is a deep-reasoning model whose extended
            // thinking phase can stream for several minutes before the first
            // token arrives. 600s (10 min) gives enough headroom while still
            // protecting against truly hung connections.
            timeout_seconds: 600,
            ..Default::default()
        }
    }

    /// Get context length for a model.
    ///
    /// Returns the known context window size, or 1M as the fallback for unknown
    /// model names (since all legacy slugs redirect to grok-4.3 which has 1M).
    pub fn context_length(model: &str) -> usize {
        XAI_MODELS
            .iter()
            .find(|(name, _, _)| *name == model)
            .map(|(_, _, ctx)| *ctx)
            .unwrap_or(1_000_000) // Fallback: 1M (unknown slugs redirect to grok-4.3)
    }

    /// List available models.
    pub fn available_models() -> Vec<(&'static str, &'static str, usize)> {
        XAI_MODELS.to_vec()
    }

    // -------------------------------------------------------------------------
    // Reasoning model helpers
    // -------------------------------------------------------------------------

    /// Returns `true` when `model` is a **reasoning model** that rejects
    /// certain OpenAI-compatible parameters (`presence_penalty`,
    /// `frequency_penalty`, `stop`).
    ///
    /// # Post May 2026 Consolidation
    ///
    /// `grok-4.3` is always a reasoning model. Reasoning effort is now a
    /// first-class parameter (`reasoning.effort`), so `reasoning_effort` is
    /// no longer stripped. All legacy slugs (`grok-3`, `grok-4`, `grok-4.20`,
    /// etc.) redirect to `grok-4.3` at the API level, so they are also
    /// treated as reasoning models.
    ///
    /// The only non-reasoning model is `grok-build-0.1`.
    ///
    /// Reference: <https://docs.x.ai> (July 2026)
    ///
    /// # Rule
    ///
    /// A model is considered a reasoning model when it is NOT `grok-build-*`.
    /// In practice this means everything except `grok-build-0.1`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use edgequake_llm::XAIProvider;
    /// assert!(XAIProvider::is_reasoning_model("grok-4.3"));
    /// assert!(XAIProvider::is_reasoning_model("grok-4.3-latest"));
    /// assert!(XAIProvider::is_reasoning_model("grok-latest"));
    /// assert!(XAIProvider::is_reasoning_model("grok-4"));
    /// assert!(XAIProvider::is_reasoning_model("grok-4.20"));
    /// assert!(XAIProvider::is_reasoning_model("grok-3"));
    /// assert!(!XAIProvider::is_reasoning_model("grok-build-0.1"));
    /// ```
    pub fn is_reasoning_model(model: &str) -> bool {
        !model.starts_with("grok-build")
    }

    /// Strip parameters that are prohibited by xAI's grok-4.3 model.
    ///
    /// Returns a new `CompletionOptions` with `presence_penalty`,
    /// `frequency_penalty`, and `stop` set to `None`.
    /// All other fields (including `reasoning_effort`) are preserved.
    ///
    /// This is a no-op if those fields are already `None`.
    ///
    /// # Why
    ///
    /// The xAI API returns HTTP 400 if `presence_penalty`, `frequency_penalty`,
    /// or `stop` are sent to grok-4.3. `reasoning_effort` is now accepted as a
    /// first-class parameter (mapped to `reasoning.effort` levels:
    /// none / low / medium / high), so it is no longer stripped.
    fn filter_for_reasoning(options: &CompletionOptions) -> CompletionOptions {
        if options.presence_penalty.is_none()
            && options.frequency_penalty.is_none()
            && options.stop.is_none()
        {
            return options.clone();
        }

        if options.presence_penalty.is_some()
            || options.frequency_penalty.is_some()
            || options.stop.is_some()
        {
            warn!(
                model = %"xai",
                "Stripping presence_penalty / frequency_penalty / stop \
                 from options — these are not supported by xAI grok-4.3 and would \
                 cause a HTTP 400 error."
            );
        }

        CompletionOptions {
            presence_penalty: None,
            frequency_penalty: None,
            stop: None,
            ..options.clone()
        }
    }

    /// Resolve effective `CompletionOptions` for the current model.
    ///
    /// If the current model is a reasoning model, strips prohibited parameters.
    /// Otherwise returns the options unchanged.
    fn resolve_options<'o>(
        &self,
        options: Option<&'o CompletionOptions>,
    ) -> std::borrow::Cow<'o, CompletionOptions> {
        match options {
            None => std::borrow::Cow::Owned(CompletionOptions::default()),
            Some(opts) if Self::is_reasoning_model(&self.model) => {
                std::borrow::Cow::Owned(Self::filter_for_reasoning(opts))
            }
            Some(opts) => std::borrow::Cow::Borrowed(opts),
        }
    }
}

// ============================================================================
// LLMProvider Implementation (delegates to inner OpenAICompatibleProvider)
// ============================================================================

#[async_trait]
impl LLMProvider for XAIProvider {
    fn name(&self) -> &str {
        XAI_PROVIDER_NAME
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_context_length(&self) -> usize {
        Self::context_length(&self.model)
    }

    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        // Delegate via complete_with_options so reasoning filter is applied.
        self.complete_with_options(prompt, &CompletionOptions::default())
            .await
    }

    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<LLMResponse> {
        let filtered = self.resolve_options(Some(options));
        self.inner
            .complete_with_options(prompt, filtered.as_ref())
            .await
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let filtered = self.resolve_options(options);
        self.inner.chat(messages, Some(filtered.as_ref())).await
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let filtered = self.resolve_options(options);
        self.inner
            .chat_with_tools(messages, tools, tool_choice, Some(filtered.as_ref()))
            .await
    }

    async fn chat_with_tools_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        let filtered = self.resolve_options(options);
        // Convert Cow to an owned value so the 'static lifetime is satisfied.
        let owned = filtered.into_owned();
        self.inner
            .chat_with_tools_stream(messages, tools, tool_choice, Some(&owned))
            .await
    }

    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        // stream() uses default options internally; no filtering needed unless
        // the caller extends this path. Keep as-is (no prohibited params).
        self.inner.stream(prompt).await
    }

    fn supports_function_calling(&self) -> bool {
        self.inner.supports_function_calling()
    }

    fn supports_tool_streaming(&self) -> bool {
        self.inner.supports_tool_streaming()
    }
}

// ============================================================================
// EmbeddingProvider Implementation (not supported - xAI doesn't have embeddings API)
// ============================================================================

#[async_trait]
impl EmbeddingProvider for XAIProvider {
    fn name(&self) -> &str {
        XAI_PROVIDER_NAME
    }

    fn model(&self) -> &str {
        "none"
    }

    fn dimension(&self) -> usize {
        0 // Not supported
    }

    fn max_tokens(&self) -> usize {
        0 // Not supported
    }

    async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // xAI doesn't provide embeddings API yet
        Err(LlmError::ConfigError(
            "xAI does not provide an embeddings API. \
             Use OpenAI or another provider for embeddings."
                .to_string(),
        ))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------

    #[test]
    fn test_provider_name_constant() {
        assert_eq!(XAI_PROVIDER_NAME, "xai");
    }

    #[test]
    fn test_default_model_constant() {
        assert_eq!(XAI_DEFAULT_MODEL, "grok-4.3");
    }

    #[test]
    fn test_default_base_url_constant() {
        assert_eq!(XAI_BASE_URL, "https://api.x.ai/v1");
    }

    // -------------------------------------------------------------------------
    // Context lengths
    // -------------------------------------------------------------------------

    #[test]
    fn test_context_length_active_models() {
        assert_eq!(XAIProvider::context_length("grok-4.3"), 1_000_000);
        assert_eq!(XAIProvider::context_length("grok-4.3-latest"), 1_000_000);
        assert_eq!(XAIProvider::context_length("grok-latest"), 1_000_000);
        assert_eq!(XAIProvider::context_length("grok-build-0.1"), 262_144);
    }

    #[test]
    fn test_context_length_legacy_aliases_return_1m() {
        // All legacy slugs redirect to grok-4.3 (1M) since May 15, 2026
        assert_eq!(XAIProvider::context_length("grok-4.20"), 1_000_000);
        assert_eq!(XAIProvider::context_length("grok-4.20-latest"), 1_000_000);
        assert_eq!(XAIProvider::context_length("grok-4"), 1_000_000);
        assert_eq!(XAIProvider::context_length("grok-4-0709"), 1_000_000);
        assert_eq!(XAIProvider::context_length("grok-4-latest"), 1_000_000);
        assert_eq!(XAIProvider::context_length("grok-4-1-fast"), 1_000_000);
        assert_eq!(XAIProvider::context_length("grok-3"), 1_000_000);
        assert_eq!(XAIProvider::context_length("grok-3-latest"), 1_000_000);
        assert_eq!(XAIProvider::context_length("grok-3-mini"), 1_000_000);
        assert_eq!(XAIProvider::context_length("grok-3-mini-latest"), 1_000_000);
    }

    #[test]
    fn test_context_length_unknown_model_defaults_1m() {
        // Unknown slugs fallback to 1M (they'll redirect to grok-4.3)
        assert_eq!(XAIProvider::context_length("grok-unknown"), 1_000_000);
        assert_eq!(XAIProvider::context_length("custom-model"), 1_000_000);
    }

    // -------------------------------------------------------------------------
    // is_reasoning_model
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_reasoning_model_grok43() {
        assert!(XAIProvider::is_reasoning_model("grok-4.3"));
        assert!(XAIProvider::is_reasoning_model("grok-4.3-latest"));
        assert!(XAIProvider::is_reasoning_model("grok-latest"));
    }

    #[test]
    fn test_is_reasoning_model_legacy_aliases() {
        // All legacy aliases redirect to grok-4.3, so they are reasoning models
        assert!(XAIProvider::is_reasoning_model("grok-4"));
        assert!(XAIProvider::is_reasoning_model("grok-4-0709"));
        assert!(XAIProvider::is_reasoning_model("grok-4-latest"));
        assert!(XAIProvider::is_reasoning_model("grok-4.20"));
        assert!(XAIProvider::is_reasoning_model("grok-4.20-latest"));
        assert!(XAIProvider::is_reasoning_model("grok-4-1-fast"));
        assert!(XAIProvider::is_reasoning_model("grok-3"));
        assert!(XAIProvider::is_reasoning_model("grok-3-latest"));
        assert!(XAIProvider::is_reasoning_model("grok-3-mini"));
        assert!(XAIProvider::is_reasoning_model("grok-3-mini-latest"));
    }

    #[test]
    fn test_is_reasoning_model_grok_build_not_reasoning() {
        assert!(!XAIProvider::is_reasoning_model("grok-build-0.1"));
    }

    // -------------------------------------------------------------------------
    // filter_for_reasoning
    // -------------------------------------------------------------------------

    #[test]
    fn test_filter_for_reasoning_strips_prohibited_fields() {
        let opts = CompletionOptions {
            temperature: Some(0.7),
            max_tokens: Some(1000),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(0.3),
            stop: Some(vec!["END".to_string()]),
            reasoning_effort: Some("high".to_string()),
            ..Default::default()
        };

        let filtered = XAIProvider::filter_for_reasoning(&opts);

        // presence_penalty, frequency_penalty, stop must be stripped
        assert!(filtered.presence_penalty.is_none());
        assert!(filtered.frequency_penalty.is_none());
        assert!(filtered.stop.is_none());

        // reasoning_effort is now preserved (first-class parameter)
        assert_eq!(filtered.reasoning_effort, Some("high".to_string()));

        // Safe fields must be preserved
        assert_eq!(filtered.temperature, Some(0.7));
        assert_eq!(filtered.max_tokens, Some(1000));
    }

    #[test]
    fn test_filter_for_reasoning_noop_when_clean() {
        let opts = CompletionOptions {
            temperature: Some(0.5),
            max_tokens: Some(512),
            ..Default::default()
        };

        let filtered = XAIProvider::filter_for_reasoning(&opts);

        assert_eq!(filtered.temperature, Some(0.5));
        assert_eq!(filtered.max_tokens, Some(512));
        assert!(filtered.presence_penalty.is_none());
        assert!(filtered.frequency_penalty.is_none());
        assert!(filtered.stop.is_none());
        assert!(filtered.reasoning_effort.is_none());
    }

    #[test]
    fn test_filter_for_reasoning_preserves_reasoning_effort() {
        let opts = CompletionOptions {
            reasoning_effort: Some("medium".to_string()),
            ..Default::default()
        };

        let filtered = XAIProvider::filter_for_reasoning(&opts);

        // reasoning_effort is now accepted by grok-4.3, must be preserved
        assert_eq!(filtered.reasoning_effort, Some("medium".to_string()));
    }

    #[test]
    fn test_filter_for_reasoning_preserves_system_prompt_and_format() {
        let opts = CompletionOptions {
            system_prompt: Some("You are helpful.".to_string()),
            response_format: Some("json_object".to_string()),
            temperature: Some(0.0),
            frequency_penalty: Some(1.0), // should be stripped
            ..Default::default()
        };

        let filtered = XAIProvider::filter_for_reasoning(&opts);

        assert_eq!(filtered.system_prompt, Some("You are helpful.".to_string()));
        assert_eq!(filtered.response_format, Some("json_object".to_string()));
        assert_eq!(filtered.temperature, Some(0.0));
        assert!(filtered.frequency_penalty.is_none());
    }

    // -------------------------------------------------------------------------
    // Available models catalog
    // -------------------------------------------------------------------------

    #[test]
    fn test_available_models_contains_active_models() {
        let models = XAIProvider::available_models();
        let names: Vec<&str> = models.iter().map(|(n, _, _)| *n).collect();

        assert!(names.contains(&"grok-4.3"), "missing grok-4.3");
        assert!(
            names.contains(&"grok-4.3-latest"),
            "missing grok-4.3-latest"
        );
        assert!(names.contains(&"grok-latest"), "missing grok-latest");
        assert!(names.contains(&"grok-build-0.1"), "missing grok-build-0.1");
    }

    #[test]
    fn test_available_models_contains_legacy_aliases() {
        let models = XAIProvider::available_models();
        let names: Vec<&str> = models.iter().map(|(n, _, _)| *n).collect();

        assert!(
            names.contains(&"grok-4.20"),
            "missing grok-4.20 legacy alias"
        );
        assert!(names.contains(&"grok-4"), "missing grok-4 legacy alias");
        assert!(
            names.contains(&"grok-4-0709"),
            "missing grok-4-0709 legacy alias"
        );
        assert!(
            names.contains(&"grok-4-latest"),
            "missing grok-4-latest legacy alias"
        );
        assert!(
            names.contains(&"grok-4-1-fast"),
            "missing grok-4-1-fast legacy alias"
        );
        assert!(names.contains(&"grok-3"), "missing grok-3 legacy alias");
        assert!(
            names.contains(&"grok-3-mini"),
            "missing grok-3-mini legacy alias"
        );
    }

    #[test]
    fn test_available_models_all_have_positive_context_length() {
        for (name, _desc, ctx) in XAIProvider::available_models() {
            assert!(ctx > 0, "Model '{}' has zero context length", name);
        }
    }

    // -------------------------------------------------------------------------
    // build_config
    // -------------------------------------------------------------------------

    #[test]
    fn test_build_config_defaults() {
        let config = XAIProvider::build_config("test-key", "grok-4.3", None);
        assert_eq!(config.name, "xai");
        assert_eq!(config.display_name, "xAI Grok");
        assert_eq!(config.base_url, Some("https://api.x.ai/v1".to_string()));
        assert_eq!(config.api_key_env, Some("XAI_API_KEY".to_string()));
        assert_eq!(config.default_llm_model, Some("grok-4.3".to_string()));
        assert!(config.enabled);
        assert_eq!(config.timeout_seconds, 600);
    }

    #[test]
    fn test_build_config_custom_base_url() {
        let config =
            XAIProvider::build_config("test-key", "grok-build-0.1", Some("https://custom.api"));
        assert_eq!(config.base_url, Some("https://custom.api".to_string()));
        assert_eq!(config.default_llm_model, Some("grok-build-0.1".to_string()));
    }

    #[test]
    fn test_build_config_model_cards_not_empty() {
        let config = XAIProvider::build_config("test-key", "grok-4.3", None);
        assert!(!config.models.is_empty());

        let card = config.models.iter().find(|m| m.name == "grok-4.3");
        assert!(card.is_some(), "grok-4.3 model card missing");
        let card = card.unwrap();
        assert!(card.capabilities.supports_function_calling);
        assert!(card.capabilities.supports_json_mode);
        assert!(card.capabilities.supports_streaming);
        assert!(card.capabilities.supports_vision);
        assert!(card.capabilities.supports_thinking); // reasoning model
        assert_eq!(card.capabilities.context_length, 1_000_000);
    }

    #[test]
    fn test_build_config_grok_build_model_card() {
        let config = XAIProvider::build_config("test-key", "grok-build-0.1", None);
        let card = config.models.iter().find(|m| m.name == "grok-build-0.1");
        assert!(card.is_some(), "grok-build-0.1 model card missing");
        let card = card.unwrap();
        assert!(!card.capabilities.supports_thinking); // not a reasoning model
        assert!(!card.capabilities.supports_vision); // text-only coding model
        assert!(card.capabilities.supports_function_calling);
        assert_eq!(card.capabilities.context_length, 262_144);
    }

    #[test]
    fn test_build_config_legacy_alias_card_has_thinking_and_vision() {
        let config = XAIProvider::build_config("test-key", "grok-4.3", None);
        // Legacy grok-3 alias now redirects to grok-4.3, should have reasoning+vision
        let card = config.models.iter().find(|m| m.name == "grok-3");
        assert!(card.is_some());
        let card = card.unwrap();
        assert!(card.capabilities.supports_thinking);
        assert!(card.capabilities.supports_vision);
        assert_eq!(card.capabilities.context_length, 1_000_000);
    }

    // -------------------------------------------------------------------------
    // from_env
    // -------------------------------------------------------------------------

    #[test]
    fn test_from_env_missing_api_key() {
        std::env::remove_var("XAI_API_KEY");
        std::env::remove_var("XAI_MODEL");
        std::env::remove_var("XAI_BASE_URL");

        let result = XAIProvider::from_env();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("XAI_API_KEY"),
            "Error should mention XAI_API_KEY, got: {}",
            err
        );
    }

    #[test]
    fn test_from_env_empty_api_key_rejected() {
        std::env::set_var("XAI_API_KEY", "");
        std::env::remove_var("XAI_MODEL");
        std::env::remove_var("XAI_BASE_URL");

        let result = XAIProvider::from_env();
        std::env::remove_var("XAI_API_KEY");

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("empty"),
            "Error should mention 'empty', got: {}",
            err
        );
    }
}
