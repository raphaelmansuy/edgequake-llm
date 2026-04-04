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
//! │        └─ strips prohibited params for reasoning models before delegate  │
//! │           (presence_penalty, frequency_penalty, stop, reasoning_effort) │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Reasoning Model Restrictions (xAI API v1, April 2026)
//!
//! All `grok-4*` models (except `-non-reasoning` variants) are **reasoning models**.
//! The xAI API returns HTTP 400 if the following parameters are sent to them:
//!
//! - `presence_penalty`
//! - `frequency_penalty`
//! - `stop` (stop sequences)
//! - `reasoning_effort`
//!
//! `XAIProvider` automatically strips these fields before delegating to the
//! inner `OpenAICompatibleProvider`, so callers need not worry about this.
//!
//! # Environment Variables
//!
//! | Variable | Required | Default | Description |
//! |----------|----------|---------|-------------|
//! | `XAI_API_KEY` | ✅ Yes | - | xAI API key from console.x.ai |
//! | `XAI_MODEL` | ❌ No | `grok-4.20` | Default model to use |
//! | `XAI_BASE_URL` | ❌ No | `https://api.x.ai/v1` | API endpoint override |
//!
//! # Available Models (as of April 2026, docs.x.ai/developers/models)
//!
//! | Model | Context | Features |
//! |-------|---------|----------|
//! | `grok-4.20` | 2M | Newest flagship (reasoning) |
//! | `grok-4.20-reasoning` | 2M | Reasoning variant alias |
//! | `grok-4.20-non-reasoning` | 2M | Non-reasoning variant alias |
//! | `grok-4.20-multi-agent` | 2M | Multi-agent orchestration |
//! | `grok-4` | 256K | Previous flagship (reasoning only) |
//! | `grok-4-0709` | 256K | July 2025 dated release |
//! | `grok-4-1-fast` | 2M | Fast agentic, tool calling |
//! | `grok-4-1-fast-non-reasoning` | 2M | Non-reasoning fast variant |
//! | `grok-3` | 128K | Previous generation |
//! | `grok-3-mini` | 128K | Smaller, faster |
//! | `grok-2-vision-1212` | 32K | Image understanding |
//! | `grok-code-fast-1` | 128K | Fast coding assistant |
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
//! # Use specific model
//! export XAI_MODEL=grok-4.20-non-reasoning
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

/// Default model — updated to Grok 4.20 (released March 2026, newest flagship).
///
/// Grok 4.20 is the current recommended default as of April 2026.
/// It supports 2M context, function calling, structured outputs, and vision.
const XAI_DEFAULT_MODEL: &str = "grok-4.20";

/// Provider display name
const XAI_PROVIDER_NAME: &str = "xai";

/// xAI model catalog with context lengths.
///
/// WHY: Pre-defined models ensure users get correct context limits without
/// having to check documentation.  Context lengths and capabilities are
/// sourced from docs.x.ai/developers/models (April 2026).
///
/// ## Naming conventions
///
/// - `grok-4.20-*`  — March 2026 flagship series (2 M context, dots in name)
/// - `grok-4-*`     — July 2025 generation  (256 K context, dashes only)
/// - `grok-4-1-fast-*` — Fast agentic models (2 M context)
/// - `grok-3-*`     — April 2025 generation (128 K context)
///
/// ## Reasoning vs non-reasoning
///
/// Any model whose name starts with `grok-4` and does _not_ end with
/// `-non-reasoning` is a **reasoning model** and does not accept
/// `presence_penalty`, `frequency_penalty`, `stop`, or `reasoning_effort`.
/// See [`XAIProvider::is_reasoning_model`].
///
/// Last updated: April 2026 (docs.x.ai/developers/models)
const XAI_MODELS: &[(&str, &str, usize)] = &[
    // ---- Grok 4.20 series (March 2026 flagship, 2M context) ----------------
    ("grok-4.20", "Grok 4.20 (Latest Flagship, 2M)", 2_000_000),
    ("grok-4.20-latest", "Grok 4.20 Latest (2M)", 2_000_000),
    ("grok-4.20-reasoning", "Grok 4.20 Reasoning (2M)", 2_000_000),
    (
        "grok-4.20-non-reasoning",
        "Grok 4.20 Non-Reasoning (2M)",
        2_000_000,
    ),
    ("grok-4.20-0309", "Grok 4.20 (0309 dated, 2M)", 2_000_000),
    (
        "grok-4.20-0309-reasoning",
        "Grok 4.20 Reasoning (0309 dated, 2M)",
        2_000_000,
    ),
    (
        "grok-4.20-0309-non-reasoning",
        "Grok 4.20 Non-Reasoning (0309 dated, 2M)",
        2_000_000,
    ),
    (
        "grok-4.20-multi-agent",
        "Grok 4.20 Multi-Agent (2M)",
        2_000_000,
    ),
    (
        "grok-4.20-multi-agent-0309",
        "Grok 4.20 Multi-Agent (0309 dated, 2M)",
        2_000_000,
    ),
    // ---- Grok 4 series (July 2025, 256K context) ---------------------------
    ("grok-4", "Grok 4 (256K, reasoning)", 262_144),
    ("grok-4-0709", "Grok 4 (July 2025, 256K)", 262_144),
    ("grok-4-latest", "Grok 4 Latest (256K)", 262_144),
    // ---- Grok 4.1 Fast series (2M context) ---------------------------------
    ("grok-4-1-fast", "Grok 4.1 Fast (2M, reasoning)", 2_000_000),
    (
        "grok-4-1-fast-reasoning",
        "Grok 4.1 Fast Reasoning (2M)",
        2_000_000,
    ),
    (
        "grok-4-1-fast-non-reasoning",
        "Grok 4.1 Fast Non-Reasoning (2M)",
        2_000_000,
    ),
    // ---- Grok 3 series (April 2025, 128K context) --------------------------
    ("grok-3", "Grok 3 (128K)", 131_072),
    ("grok-3-latest", "Grok 3 Latest (128K)", 131_072),
    ("grok-3-mini", "Grok 3 Mini (128K)", 131_072),
    ("grok-3-mini-latest", "Grok 3 Mini Latest (128K)", 131_072),
    // ---- Specialised models ------------------------------------------------
    ("grok-2-vision-1212", "Grok 2 Vision (32K)", 32_768),
    ("grok-code-fast-1", "Grok Code Fast (128K)", 131_072),
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
    /// - `XAI_MODEL`: Model name (default: `grok-4.20`)
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
    /// * `model` - Model name (e.g., "grok-4")
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
                // Vision support: all grok-4 series + grok-2-vision
                // grok-4, grok-4.20, grok-4.20-*, grok-4-0709, grok-4-1-fast-*, grok-2-vision
                let supports_vision = name.starts_with("grok-4") || name.contains("vision");

                // Reasoning/thinking support: all grok-4* models that are NOT
                // explicitly the non-reasoning variant support chain-of-thought.
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
            // WHY 600s: Grok 4 and Grok 4-latest are deep-reasoning models.
            // Their extended thinking phase can stream for several minutes
            // before the first tool-call or text token arrives. The default
            // 120s client timeout fires mid-stream, causes a NetworkError,
            // and triggers 3 retries × 120s ≈ 8 minutes of "Still processing"
            // before the error surfaces. 600s (10 min) gives the model enough
            // headroom while still protecting against truly hung connections.
            timeout_seconds: 600,
            ..Default::default()
        }
    }

    /// Get context length for a model.
    ///
    /// Returns the known context window size, or 256K as a conservative fallback
    /// for unknown model names (most Grok 4 models have at least 256K).
    pub fn context_length(model: &str) -> usize {
        XAI_MODELS
            .iter()
            .find(|(name, _, _)| *name == model)
            .map(|(_, _, ctx)| *ctx)
            .unwrap_or(262_144) // Conservative fallback: 256K
    }

    /// List available models.
    pub fn available_models() -> Vec<(&'static str, &'static str, usize)> {
        XAI_MODELS.to_vec()
    }

    // -------------------------------------------------------------------------
    // Reasoning model helpers
    // -------------------------------------------------------------------------

    /// Returns `true` when `model` is a **reasoning model** that rejects
    /// certain OpenAI-compatible parameters.
    ///
    /// # Why This Exists
    ///
    /// The xAI API returns HTTP 400 "Bad Request" when any of
    /// `presence_penalty`, `frequency_penalty`, `stop`, or `reasoning_effort`
    /// are included in a request to a reasoning model. All `grok-4*` models
    /// are reasoning models **unless** the name ends with `-non-reasoning`.
    ///
    /// Reference: <https://docs.x.ai/developers/models> (April 2026)
    ///
    /// # Rule
    ///
    /// A model is considered a reasoning model when:
    /// - Its name starts with `"grok-4"`, AND
    /// - Its name does NOT end with `"-non-reasoning"`
    ///
    /// # Examples
    ///
    /// ```
    /// # use edgequake_llm::XAIProvider;
    /// assert!(XAIProvider::is_reasoning_model("grok-4"));
    /// assert!(XAIProvider::is_reasoning_model("grok-4.20"));
    /// assert!(XAIProvider::is_reasoning_model("grok-4.20-reasoning"));
    /// assert!(XAIProvider::is_reasoning_model("grok-4-1-fast"));
    /// assert!(!XAIProvider::is_reasoning_model("grok-4.20-non-reasoning"));
    /// assert!(!XAIProvider::is_reasoning_model("grok-4-1-fast-non-reasoning"));
    /// assert!(!XAIProvider::is_reasoning_model("grok-3"));
    /// ```
    pub fn is_reasoning_model(model: &str) -> bool {
        model.starts_with("grok-4") && !model.ends_with("-non-reasoning")
    }

    /// Strip parameters that are prohibited by xAI reasoning models.
    ///
    /// Returns a new `CompletionOptions` with `presence_penalty`,
    /// `frequency_penalty`, `stop`, and `reasoning_effort` set to `None`.
    /// All other fields are preserved unchanged.
    ///
    /// This is a no-op if those fields are already `None`.
    ///
    /// # Why
    ///
    /// The xAI API returns HTTP 400 if these params are sent to a reasoning
    /// model.  By stripping them transparently we avoid breaking the caller.
    ///
    /// Callers that need these parameters should pick a `-non-reasoning`
    /// variant explicitly (e.g. `grok-4.20-non-reasoning`).
    fn filter_for_reasoning(options: &CompletionOptions) -> CompletionOptions {
        if options.presence_penalty.is_none()
            && options.frequency_penalty.is_none()
            && options.stop.is_none()
            && options.reasoning_effort.is_none()
        {
            // Fast path: nothing to strip, avoid clone
            return options.clone();
        }

        if options.presence_penalty.is_some()
            || options.frequency_penalty.is_some()
            || options.stop.is_some()
            || options.reasoning_effort.is_some()
        {
            warn!(
                model = %"xai",
                "Stripping presence_penalty / frequency_penalty / stop / reasoning_effort \
                 from options — these are not supported by xAI reasoning models and would \
                 cause a HTTP 400 error. Use a *-non-reasoning model variant to keep them."
            );
        }

        CompletionOptions {
            presence_penalty: None,
            frequency_penalty: None,
            stop: None,
            reasoning_effort: None,
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
        // Updated to Grok 4.20 (newest flagship as of April 2026)
        assert_eq!(XAI_DEFAULT_MODEL, "grok-4.20");
    }

    #[test]
    fn test_default_base_url_constant() {
        assert_eq!(XAI_BASE_URL, "https://api.x.ai/v1");
    }

    // -------------------------------------------------------------------------
    // Context lengths
    // -------------------------------------------------------------------------

    #[test]
    fn test_context_length_grok420_series() {
        // Grok 4.20 series: 2M context (March 2026)
        assert_eq!(XAIProvider::context_length("grok-4.20"), 2_000_000);
        assert_eq!(XAIProvider::context_length("grok-4.20-latest"), 2_000_000);
        assert_eq!(
            XAIProvider::context_length("grok-4.20-reasoning"),
            2_000_000
        );
        assert_eq!(
            XAIProvider::context_length("grok-4.20-non-reasoning"),
            2_000_000
        );
        assert_eq!(XAIProvider::context_length("grok-4.20-0309"), 2_000_000);
        assert_eq!(
            XAIProvider::context_length("grok-4.20-0309-reasoning"),
            2_000_000
        );
        assert_eq!(
            XAIProvider::context_length("grok-4.20-0309-non-reasoning"),
            2_000_000
        );
        assert_eq!(
            XAIProvider::context_length("grok-4.20-multi-agent"),
            2_000_000
        );
        assert_eq!(
            XAIProvider::context_length("grok-4.20-multi-agent-0309"),
            2_000_000
        );
    }

    #[test]
    fn test_context_length_grok4_series() {
        // Grok 4 (July 2025): 256K context
        assert_eq!(XAIProvider::context_length("grok-4"), 262_144);
        assert_eq!(XAIProvider::context_length("grok-4-0709"), 262_144);
        assert_eq!(XAIProvider::context_length("grok-4-latest"), 262_144);
    }

    #[test]
    fn test_context_length_grok41_fast_series() {
        // Grok 4.1 Fast: 2M context
        assert_eq!(XAIProvider::context_length("grok-4-1-fast"), 2_000_000);
        assert_eq!(
            XAIProvider::context_length("grok-4-1-fast-reasoning"),
            2_000_000
        );
        assert_eq!(
            XAIProvider::context_length("grok-4-1-fast-non-reasoning"),
            2_000_000
        );
    }

    #[test]
    fn test_context_length_grok3_series() {
        // Grok 3 (April 2025): 128K context
        assert_eq!(XAIProvider::context_length("grok-3"), 131_072);
        assert_eq!(XAIProvider::context_length("grok-3-latest"), 131_072);
        assert_eq!(XAIProvider::context_length("grok-3-mini"), 131_072);
        assert_eq!(XAIProvider::context_length("grok-3-mini-latest"), 131_072);
    }

    #[test]
    fn test_context_length_specialized_models() {
        assert_eq!(XAIProvider::context_length("grok-2-vision-1212"), 32_768); // 32K
        assert_eq!(XAIProvider::context_length("grok-code-fast-1"), 131_072); // 128K
    }

    #[test]
    fn test_context_length_unknown_model_defaults_256k() {
        // Conservative fallback: unknown models get 256K
        assert_eq!(XAIProvider::context_length("grok-unknown"), 262_144);
        assert_eq!(XAIProvider::context_length("custom-model"), 262_144);
    }

    // -------------------------------------------------------------------------
    // is_reasoning_model
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_reasoning_model_grok4_base() {
        // All grok-4 base/alias models are reasoning-only
        assert!(XAIProvider::is_reasoning_model("grok-4"));
        assert!(XAIProvider::is_reasoning_model("grok-4-0709"));
        assert!(XAIProvider::is_reasoning_model("grok-4-latest"));
    }

    #[test]
    fn test_is_reasoning_model_grok420_series() {
        // grok-4.20 base/aliases are reasoning (default variant)
        assert!(XAIProvider::is_reasoning_model("grok-4.20"));
        assert!(XAIProvider::is_reasoning_model("grok-4.20-latest"));
        assert!(XAIProvider::is_reasoning_model("grok-4.20-reasoning"));
        assert!(XAIProvider::is_reasoning_model("grok-4.20-0309"));
        assert!(XAIProvider::is_reasoning_model("grok-4.20-0309-reasoning"));
        assert!(XAIProvider::is_reasoning_model("grok-4.20-multi-agent"));
        assert!(XAIProvider::is_reasoning_model(
            "grok-4.20-multi-agent-0309"
        ));
    }

    #[test]
    fn test_is_reasoning_model_grok420_non_reasoning() {
        // Explicit -non-reasoning suffix opts out of reasoning mode
        assert!(!XAIProvider::is_reasoning_model("grok-4.20-non-reasoning"));
        assert!(!XAIProvider::is_reasoning_model(
            "grok-4.20-0309-non-reasoning"
        ));
    }

    #[test]
    fn test_is_reasoning_model_grok41_fast() {
        // grok-4-1-fast alias defaults to reasoning; non-reasoning is explicit
        assert!(XAIProvider::is_reasoning_model("grok-4-1-fast"));
        assert!(XAIProvider::is_reasoning_model("grok-4-1-fast-reasoning"));
        assert!(!XAIProvider::is_reasoning_model(
            "grok-4-1-fast-non-reasoning"
        ));
    }

    #[test]
    fn test_is_reasoning_model_grok3_series_not_reasoning() {
        // Grok 3 and older are NOT reasoning models
        assert!(!XAIProvider::is_reasoning_model("grok-3"));
        assert!(!XAIProvider::is_reasoning_model("grok-3-latest"));
        assert!(!XAIProvider::is_reasoning_model("grok-3-mini"));
        assert!(!XAIProvider::is_reasoning_model("grok-2-vision-1212"));
        assert!(!XAIProvider::is_reasoning_model("grok-code-fast-1"));
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

        // Prohibited fields must be None
        assert!(filtered.presence_penalty.is_none());
        assert!(filtered.frequency_penalty.is_none());
        assert!(filtered.stop.is_none());
        assert!(filtered.reasoning_effort.is_none());

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

        // All fields preserved unchanged
        assert_eq!(filtered.temperature, Some(0.5));
        assert_eq!(filtered.max_tokens, Some(512));
        assert!(filtered.presence_penalty.is_none());
        assert!(filtered.frequency_penalty.is_none());
        assert!(filtered.stop.is_none());
        assert!(filtered.reasoning_effort.is_none());
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
    fn test_available_models_contains_grok420_series() {
        let models = XAIProvider::available_models();
        let names: Vec<&str> = models.iter().map(|(n, _, _)| *n).collect();

        assert!(names.contains(&"grok-4.20"), "missing grok-4.20");
        assert!(
            names.contains(&"grok-4.20-latest"),
            "missing grok-4.20-latest"
        );
        assert!(
            names.contains(&"grok-4.20-reasoning"),
            "missing grok-4.20-reasoning"
        );
        assert!(
            names.contains(&"grok-4.20-non-reasoning"),
            "missing grok-4.20-non-reasoning"
        );
        assert!(names.contains(&"grok-4.20-0309"), "missing grok-4.20-0309");
        assert!(
            names.contains(&"grok-4.20-0309-reasoning"),
            "missing grok-4.20-0309-reasoning"
        );
        assert!(
            names.contains(&"grok-4.20-0309-non-reasoning"),
            "missing grok-4.20-0309-non-reasoning"
        );
        assert!(
            names.contains(&"grok-4.20-multi-agent"),
            "missing grok-4.20-multi-agent"
        );
        assert!(
            names.contains(&"grok-4.20-multi-agent-0309"),
            "missing grok-4.20-multi-agent-0309"
        );
    }

    #[test]
    fn test_available_models_contains_all_legacy_series() {
        let models = XAIProvider::available_models();
        let names: Vec<&str> = models.iter().map(|(n, _, _)| *n).collect();

        // Grok 4 (July 2025) series
        assert!(names.contains(&"grok-4"));
        assert!(names.contains(&"grok-4-0709"));
        assert!(names.contains(&"grok-4-latest"));

        // Grok 4.1 Fast series
        assert!(names.contains(&"grok-4-1-fast"));
        assert!(names.contains(&"grok-4-1-fast-reasoning"));
        assert!(names.contains(&"grok-4-1-fast-non-reasoning"));

        // Grok 3 series
        assert!(names.contains(&"grok-3"));
        assert!(names.contains(&"grok-3-mini"));

        // Specialised
        assert!(names.contains(&"grok-2-vision-1212"));
        assert!(names.contains(&"grok-code-fast-1"));
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
        let config = XAIProvider::build_config("test-key", "grok-4.20", None);
        assert_eq!(config.name, "xai");
        assert_eq!(config.display_name, "xAI Grok");
        assert_eq!(config.base_url, Some("https://api.x.ai/v1".to_string()));
        assert_eq!(config.api_key_env, Some("XAI_API_KEY".to_string()));
        assert_eq!(config.default_llm_model, Some("grok-4.20".to_string()));
        assert!(config.enabled);
        assert_eq!(config.timeout_seconds, 600);
    }

    #[test]
    fn test_build_config_custom_base_url() {
        let config = XAIProvider::build_config("test-key", "grok-3", Some("https://custom.api"));
        assert_eq!(config.base_url, Some("https://custom.api".to_string()));
        assert_eq!(config.default_llm_model, Some("grok-3".to_string()));
    }

    #[test]
    fn test_build_config_model_cards_not_empty() {
        let config = XAIProvider::build_config("test-key", "grok-4.20", None);
        assert!(!config.models.is_empty());

        // Verify grok-4.20 card exists and has correct capabilities
        let card = config.models.iter().find(|m| m.name == "grok-4.20");
        assert!(card.is_some(), "grok-4.20 model card missing");
        let card = card.unwrap();
        assert!(card.capabilities.supports_function_calling);
        assert!(card.capabilities.supports_json_mode);
        assert!(card.capabilities.supports_streaming);
        assert!(card.capabilities.supports_vision);
        assert!(card.capabilities.supports_thinking); // reasoning model
    }

    #[test]
    fn test_build_config_non_reasoning_model_card_no_thinking() {
        let config = XAIProvider::build_config("test-key", "grok-4.20-non-reasoning", None);
        let card = config
            .models
            .iter()
            .find(|m| m.name == "grok-4.20-non-reasoning");
        assert!(card.is_some());
        let card = card.unwrap();
        // non-reasoning variant should NOT have thinking flagged
        assert!(!card.capabilities.supports_thinking);
    }

    #[test]
    fn test_build_config_grok3_model_card_no_thinking() {
        let config = XAIProvider::build_config("test-key", "grok-3", None);
        let card = config.models.iter().find(|m| m.name == "grok-3");
        assert!(card.is_some());
        let card = card.unwrap();
        assert!(!card.capabilities.supports_thinking);
        // grok-3 is text only, not vision
        assert!(!card.capabilities.supports_vision);
    }

    // -------------------------------------------------------------------------
    // from_env
    // -------------------------------------------------------------------------

    #[test]
    fn test_from_env_missing_api_key() {
        // Clear env vars to ensure clean test
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
        // restore
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
