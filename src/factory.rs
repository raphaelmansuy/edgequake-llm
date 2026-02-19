//! LLM provider factory for environment-based selection.
//!
//! @implements SPEC-032: Ollama/LM Studio provider support
//! @implements FEAT0017: Multi-provider LLM support
//!
//! # Environment Variables
//!
//! ## Provider Selection
//!
//! - `EDGEQUAKE_LLM_PROVIDER`: Override provider selection (openai|ollama|lmstudio|mock)
//!
//! ## Provider-Specific Configuration
//!
//! See individual provider documentation for configuration variables:
//! - OpenAI: `OPENAI_API_KEY`, `OPENAI_BASE_URL`
//! - Ollama: `OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_EMBEDDING_MODEL`
//! - LM Studio: `LMSTUDIO_HOST`, `LMSTUDIO_MODEL`, `LMSTUDIO_EMBEDDING_MODEL`
//!
//! # Auto-Detection Priority
//!
//! When `EDGEQUAKE_LLM_PROVIDER` is not set:
//! 1. Check for OLLAMA_HOST or OLLAMA_MODEL → Use Ollama
//! 2. Check for OPENAI_API_KEY → Use OpenAI
//! 3. Fallback → Use Mock provider
//!
//! # Example
//!
//! ```rust,ignore
//! use edgequake_llm::ProviderFactory;
//!
//! // Auto-detect from environment
//! let (llm, embedding) = ProviderFactory::from_env()?;
//!
//! // Explicit provider selection
//! std::env::set_var("EDGEQUAKE_LLM_PROVIDER", "ollama");
//! let (llm, embedding) = ProviderFactory::from_env()?;
//! ```

use std::sync::Arc;

use tracing::warn;

use crate::error::{LlmError, Result};
use crate::model_config::{ProviderConfig, ProviderType as ConfigProviderType};
use crate::providers::anthropic::AnthropicProvider;
use crate::providers::gemini::GeminiProvider;
use crate::providers::huggingface::HuggingFaceProvider;
use crate::providers::lmstudio::LMStudioProvider;
use crate::providers::openai_compatible::OpenAICompatibleProvider;
use crate::providers::openrouter::OpenRouterProvider;
use crate::providers::mistral::MistralProvider;
use crate::providers::xai::XAIProvider;
use crate::traits::{EmbeddingProvider, LLMProvider};
use crate::{MockProvider, OllamaProvider, OpenAIProvider, VsCodeCopilotProvider};

/// Supported provider types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderType {
    /// OpenAI provider (cloud API)
    OpenAI,
    /// Anthropic provider (Claude models)
    Anthropic,
    /// Gemini provider (Google AI / VertexAI)
    Gemini,
    /// OpenRouter provider (200+ models)
    OpenRouter,
    /// xAI provider (Grok models via api.x.ai)
    XAI,
    /// HuggingFace Hub provider (open-source models)
    HuggingFace,
    /// Ollama provider (local models)
    Ollama,
    /// LM Studio provider (OpenAI-compatible local API)
    LMStudio,
    /// VSCode Copilot provider (via proxy)
    VsCodeCopilot,
    /// Mock provider (testing only)
    Mock,
    /// Mistral AI (La Plateforme)
    Mistral,
}

impl ProviderType {
    /// Parse provider type from string (case-insensitive)
    ///
    /// # Examples
    ///
    /// ```
    /// use edgequake_llm::ProviderType;
    ///
    /// assert_eq!(ProviderType::from_str("openai"), Some(ProviderType::OpenAI));
    /// assert_eq!(ProviderType::from_str("OLLAMA"), Some(ProviderType::Ollama));
    /// assert_eq!(ProviderType::from_str("lm-studio"), Some(ProviderType::LMStudio));
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "openai" => Some(Self::OpenAI),
            "anthropic" | "claude" => Some(Self::Anthropic),
            "gemini" | "google" | "vertex" | "vertexai" => Some(Self::Gemini),
            "openrouter" | "open-router" => Some(Self::OpenRouter),
            "xai" | "grok" => Some(Self::XAI),
            "huggingface" | "hf" | "hugging-face" | "hugging_face" => Some(Self::HuggingFace),
            "ollama" => Some(Self::Ollama),
            "lmstudio" | "lm-studio" | "lm_studio" => Some(Self::LMStudio),
            "vscode" | "vscode-copilot" | "copilot" => Some(Self::VsCodeCopilot),
            "mock" => Some(Self::Mock),
            "mistral" | "mistral-ai" | "mistralai" => Some(Self::Mistral),
            _ => None,
        }
    }
}

/// Provider factory for creating LLM and embedding providers.
///
/// Provides environment-based auto-detection and explicit provider selection.
pub struct ProviderFactory;

impl ProviderFactory {
    /// Auto-detect and create providers from environment.
    ///
    /// # Priority
    ///
    /// 1. `EDGEQUAKE_LLM_PROVIDER` environment variable (explicit selection)
    /// 2. Auto-detect: OLLAMA_HOST → LMSTUDIO_HOST → OPENAI_API_KEY → Mock
    ///
    /// # Returns
    ///
    /// Returns a tuple of (LLMProvider, EmbeddingProvider). In most cases,
    /// the same provider implementation is used for both.
    ///
    /// # Errors
    ///
    /// Returns error if required configuration for selected provider is missing.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// std::env::set_var("OLLAMA_HOST", "http://localhost:11434");
    /// let (llm, embedding) = ProviderFactory::from_env()?;
    /// assert_eq!(llm.name(), "ollama");
    /// ```
    pub fn from_env() -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        // Check explicit provider selection
        if let Ok(provider_str) = std::env::var("EDGEQUAKE_LLM_PROVIDER") {
            if let Some(provider_type) = ProviderType::from_str(&provider_str) {
                return Self::create(provider_type);
            }
            return Err(LlmError::ConfigError(format!(
                "Unknown provider type: {}. Valid options: openai, anthropic, ollama, lmstudio, mock",
                provider_str
            )));
        }

        // Auto-detect based on environment
        // Priority: Ollama → LM Studio → Anthropic → Gemini → xAI → OpenRouter → OpenAI → Mock
        if std::env::var("OLLAMA_HOST").is_ok() || std::env::var("OLLAMA_MODEL").is_ok() {
            return Self::create(ProviderType::Ollama);
        }

        // LM Studio detection (checks LMSTUDIO_HOST or LMSTUDIO_MODEL)
        if std::env::var("LMSTUDIO_HOST").is_ok() || std::env::var("LMSTUDIO_MODEL").is_ok() {
            return Self::create(ProviderType::LMStudio);
        }

        // Anthropic detection
        if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
            if !api_key.is_empty() {
                return Self::create(ProviderType::Anthropic);
            }
        }

        // OODA-73: Gemini detection (GEMINI_API_KEY or GOOGLE_API_KEY)
        if let Ok(api_key) =
            std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("GOOGLE_API_KEY"))
        {
            if !api_key.is_empty() {
                return Self::create(ProviderType::Gemini);
            }
        }

        // Mistral detection
        if let Ok(api_key) = std::env::var("MISTRAL_API_KEY") {
            if !api_key.is_empty() {
                return Self::create(ProviderType::Mistral);
            }
        }

        // OODA-71: xAI detection (Grok models via api.x.ai)
        if let Ok(api_key) = std::env::var("XAI_API_KEY") {
            if !api_key.is_empty() {
                return Self::create(ProviderType::XAI);
            }
        }

        // OODA-80: HuggingFace Hub detection (HF_TOKEN or HUGGINGFACE_TOKEN)
        if let Ok(api_key) =
            std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGINGFACE_TOKEN"))
        {
            if !api_key.is_empty() {
                return Self::create(ProviderType::HuggingFace);
            }
        }

        // OODA-73: OpenRouter detection
        if let Ok(api_key) = std::env::var("OPENROUTER_API_KEY") {
            if !api_key.is_empty() {
                return Self::create(ProviderType::OpenRouter);
            }
        }

        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            if !api_key.is_empty() && api_key != "test-key" {
                return Self::create(ProviderType::OpenAI);
            }
        }

        // Fallback to mock
        Ok(Self::create_mock())
    }

    /// Create specific provider type.
    ///
    /// # Arguments
    ///
    /// * `provider_type` - The type of provider to create
    ///
    /// # Returns
    ///
    /// Returns a tuple of (LLMProvider, EmbeddingProvider).
    ///
    /// # Errors
    ///
    /// Returns error if required configuration for the provider is missing.
    pub fn create(
        provider_type: ProviderType,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        match provider_type {
            ProviderType::OpenAI => Self::create_openai(),
            ProviderType::Anthropic => Self::create_anthropic(),
            ProviderType::Gemini => Self::create_gemini(),
            ProviderType::OpenRouter => Self::create_openrouter(),
            ProviderType::XAI => Self::create_xai(),
            ProviderType::HuggingFace => Self::create_huggingface(),
            ProviderType::Ollama => Self::create_ollama(),
            ProviderType::LMStudio => Self::create_lmstudio(),
            ProviderType::VsCodeCopilot => Self::create_vscode_copilot(),
            ProviderType::Mock => Ok(Self::create_mock()),
            ProviderType::Mistral => Self::create_mistral(),
        }
    }

    /// OODA-04: Create specific provider type with a model override.
    ///
    /// Like `create()` but allows specifying a model name instead of using defaults.
    /// Useful for CLI where user specifies `--provider openrouter --model mistral/model`.
    ///
    /// # Arguments
    ///
    /// * `provider_type` - The type of provider to create
    /// * `model` - Optional model name to use instead of provider default
    ///
    /// # Returns
    ///
    /// Returns a tuple of (LLMProvider, EmbeddingProvider).
    ///
    /// # Errors
    ///
    /// Returns error if required configuration for the provider is missing.
    pub fn create_with_model(
        provider_type: ProviderType,
        model: Option<&str>,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        match model {
            Some(m) => match provider_type {
                ProviderType::OpenRouter => Self::create_openrouter_with_model(m),
                ProviderType::Anthropic => Self::create_anthropic_with_model(m),
                ProviderType::Gemini => Self::create_gemini_with_model(m),
                ProviderType::XAI => Self::create_xai_with_model(m),
                ProviderType::OpenAI => Self::create_openai_with_model(m),
                ProviderType::Ollama => Self::create_ollama_with_model(m),
                ProviderType::LMStudio => Self::create_lmstudio_with_model(m),
                // These don't need model override
                ProviderType::HuggingFace => Self::create_huggingface(),
                ProviderType::VsCodeCopilot => Self::create_vscode_copilot(),
                ProviderType::Mock => Ok(Self::create_mock()),
                ProviderType::Mistral => Self::create_mistral_with_model(m),
            },
            None => Self::create(provider_type),
        }
    }

    /// OODA-200: Create provider from TOML configuration.
    ///
    /// This is the primary entry point for custom providers defined in models.toml.
    /// Supports both native providers (OpenAI, Ollama) and OpenAI-compatible APIs.
    ///
    /// # Arguments
    ///
    /// * `config` - Provider configuration from models.toml
    ///
    /// # Returns
    ///
    /// Returns a tuple of (LLMProvider, EmbeddingProvider).
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Required API key environment variable is not set
    /// - Base URL is not configured for OpenAI-compatible providers
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use edgequake_llm::{ModelsConfig, ProviderFactory};
    ///
    /// let config = ModelsConfig::load()?;
    /// let provider_config = config.get_provider("zai").unwrap();
    /// let (llm, embedding) = ProviderFactory::from_config(provider_config)?;
    /// ```
    pub fn from_config(
        config: &ProviderConfig,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        Self::from_config_with_model(config, None)
    }

    /// Create provider from configuration with a specific model override.
    ///
    /// This is used when the user selects a specific model via `/model` command
    /// that differs from the provider's default model.
    ///
    /// # Arguments
    ///
    /// * `config` - Provider configuration from models.toml
    /// * `model_name` - Optional model name to use instead of the default
    ///
    /// # Returns
    ///
    /// Tuple of (LLM provider, Embedding provider)
    pub fn from_config_with_model(
        config: &ProviderConfig,
        model_name: Option<&str>,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        match config.provider_type {
            ConfigProviderType::OpenAI => Self::create_openai(),
            ConfigProviderType::Ollama => Self::create_ollama(),
            ConfigProviderType::LMStudio => Self::create_lmstudio(),
            ConfigProviderType::Mock => Ok(Self::create_mock()),
            ConfigProviderType::OpenAICompatible => {
                Self::create_openai_compatible_with_model(config, model_name)
            }
            ConfigProviderType::Azure => Err(LlmError::ConfigError(
                "Azure OpenAI is not yet supported via from_config. \
                     Use AZURE_OPENAI_* environment variables instead."
                    .to_string(),
            )),
            ConfigProviderType::Anthropic => Self::create_anthropic_from_config(config, model_name),
            ConfigProviderType::OpenRouter => {
                Self::create_openrouter_from_config(config, model_name)
            }
            ConfigProviderType::Mistral => {
                Self::create_mistral_from_config(config, model_name)
            }
        }
    }

    /// Create OpenAI-compatible provider from TOML configuration.
    ///
    /// This creates a generic provider that can work with any OpenAI-compatible API:
    /// - Z.ai (GLM models)
    /// - DeepSeek
    /// - Together AI
    /// - Groq
    /// - Any custom endpoint
    #[allow(dead_code)]
    fn create_openai_compatible(
        config: &ProviderConfig,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        Self::create_openai_compatible_with_model(config, None)
    }

    /// Create OpenAI-compatible provider with optional model override.
    fn create_openai_compatible_with_model(
        config: &ProviderConfig,
        model_name: Option<&str>,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        let mut provider_instance = OpenAICompatibleProvider::from_config(config.clone())?;

        // Apply model override if provided
        if let Some(model) = model_name {
            provider_instance = provider_instance.with_model(model);
        }

        let provider = Arc::new(provider_instance);

        // For embedding, check if this provider has embedding models configured
        let has_embedding = config.default_embedding_model.is_some();

        if has_embedding {
            // Use the same provider for both LLM and embedding
            Ok((provider.clone(), provider))
        } else {
            // Fall back to OpenAI for embeddings if available
            match Self::create_openai() {
                Ok((_, embedding)) => Ok((provider, embedding)),
                Err(_) => {
                    // No OpenAI available, use this provider anyway
                    // (embedding calls will fail but LLM will work)
                    Ok((provider.clone(), provider))
                }
            }
        }
    }

    /// Create OpenAI provider from environment.
    ///
    /// Reads `OPENAI_API_KEY` environment variable.
    fn create_openai() -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            LlmError::ConfigError("OPENAI_API_KEY not set for OpenAI provider".to_string())
        })?;

        if api_key.is_empty() || api_key == "test-key" {
            return Err(LlmError::ConfigError(
                "OPENAI_API_KEY is empty or invalid".to_string(),
            ));
        }

        let provider = Arc::new(OpenAIProvider::new(api_key));
        Ok((provider.clone(), provider))
    }

    /// Create Anthropic provider from environment.
    ///
    /// Reads `ANTHROPIC_API_KEY` environment variable.
    /// Also reads `ANTHROPIC_BASE_URL` and `ANTHROPIC_MODEL` if set.
    ///
    /// Note: Anthropic does not provide embeddings API, so we fall back to
    /// OpenAI for embeddings if available, otherwise use mock.
    ///
    /// # OODA-44: Fixed to use from_env() instead of new()
    ///
    /// WHY: The original implementation only read ANTHROPIC_API_KEY and ANTHROPIC_MODEL,
    ///      but ignored ANTHROPIC_BASE_URL. This prevented using Ollama's Anthropic-
    ///      compatible API. Now uses from_env() which properly handles all env vars:
    ///
    /// ```text
    /// ┌─────────────────────────────────────────────────────────────────────┐
    /// │  Environment Variables → AnthropicProvider                         │
    /// ├─────────────────────────────────────────────────────────────────────┤
    /// │  ANTHROPIC_API_KEY    → Required authentication key                │
    /// │  ANTHROPIC_BASE_URL   → Custom endpoint (e.g., Ollama localhost)   │
    /// │  ANTHROPIC_MODEL      → Model to use (default: claude-sonnet-4)    │
    /// └─────────────────────────────────────────────────────────────────────┘
    /// ```
    fn create_anthropic() -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        // OODA-44: Use from_env() which handles ANTHROPIC_BASE_URL for Ollama support
        let provider = Arc::new(AnthropicProvider::from_env()?);

        // Anthropic doesn't provide embeddings, fall back to OpenAI or mock
        let embedding: Arc<dyn EmbeddingProvider> = match Self::create_openai() {
            Ok((_, embedding)) => embedding,
            Err(_) => Arc::new(MockProvider::new()),
        };

        Ok((provider, embedding))
    }

    /// Create Anthropic provider from TOML configuration.
    ///
    /// Reads API key from environment variable specified in config.
    fn create_anthropic_from_config(
        config: &ProviderConfig,
        model_name: Option<&str>,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        // Get API key from environment
        let api_key_var = config.api_key_env.as_deref().unwrap_or("ANTHROPIC_API_KEY");
        let api_key = std::env::var(api_key_var).map_err(|_| {
            LlmError::ConfigError(format!("{} not set for Anthropic provider", api_key_var))
        })?;

        if api_key.is_empty() {
            return Err(LlmError::ConfigError(format!("{} is empty", api_key_var)));
        }

        // Determine model
        let model = model_name
            .map(|s| s.to_string())
            .or_else(|| config.default_llm_model.clone())
            .unwrap_or_else(|| "claude-sonnet-4-5-20250929".to_string());

        // Create provider with optional base URL
        let mut provider = AnthropicProvider::new(api_key).with_model(model);

        if let Some(base_url) = &config.base_url {
            provider = provider.with_base_url(base_url);
        }

        let llm_provider = Arc::new(provider);

        // Anthropic doesn't provide embeddings, fall back to OpenAI or mock
        let embedding: Arc<dyn EmbeddingProvider> = match Self::create_openai() {
            Ok((_, embedding)) => embedding,
            Err(_) => Arc::new(MockProvider::new()),
        };

        Ok((llm_provider, embedding))
    }

    /// OODA-73: Create Gemini provider from environment.
    ///
    /// Uses GeminiProvider::from_env() which reads:
    /// - GEMINI_API_KEY or GOOGLE_API_KEY (required for Google AI)
    /// - GOOGLE_APPLICATION_CREDENTIALS (for VertexAI)
    /// - GOOGLE_CLOUD_PROJECT (for VertexAI)
    /// - GOOGLE_CLOUD_REGION (optional, default: us-central1)
    /// - GEMINI_MODEL (optional, default: gemini-2.5-flash)
    fn create_gemini() -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        use crate::GeminiProvider;

        let provider = GeminiProvider::from_env()?;
        let llm_provider: Arc<dyn LLMProvider> = Arc::new(provider);

        // Gemini doesn't have native embeddings API, fall back to OpenAI or mock
        let embedding: Arc<dyn EmbeddingProvider> = match Self::create_openai() {
            Ok((_, embedding)) => embedding,
            Err(_) => Arc::new(MockProvider::new()),
        };

        Ok((llm_provider, embedding))
    }

    /// Create OpenRouter provider from environment.
    ///
    /// Uses OpenRouterProvider::from_env() which reads:
    /// - OPENROUTER_API_KEY (required)
    /// - OPENROUTER_MODEL (optional, default: anthropic/claude-3.5-sonnet)
    /// - OPENROUTER_SITE_URL (optional, for dashboard tracking)
    /// - OPENROUTER_SITE_NAME (optional, for dashboard tracking)
    fn create_openrouter() -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        let api_key = std::env::var("OPENROUTER_API_KEY").map_err(|_| {
            LlmError::ConfigError("OPENROUTER_API_KEY not set for OpenRouter provider".to_string())
        })?;

        if api_key.is_empty() {
            return Err(LlmError::ConfigError(
                "OPENROUTER_API_KEY is empty".to_string(),
            ));
        }

        let model = std::env::var("OPENROUTER_MODEL")
            .unwrap_or_else(|_| "anthropic/claude-3.5-sonnet".to_string());

        let mut provider = OpenRouterProvider::new(api_key).with_model(model);

        // Optional site URL and name
        if let Ok(url) = std::env::var("OPENROUTER_SITE_URL") {
            provider = provider.with_site_url(url);
        }
        if let Ok(name) = std::env::var("OPENROUTER_SITE_NAME") {
            provider = provider.with_site_name(name);
        }

        let llm_provider = Arc::new(provider);

        // OpenRouter doesn't provide embeddings, fall back to OpenAI or mock
        let embedding: Arc<dyn EmbeddingProvider> = match Self::create_openai() {
            Ok((_, embedding)) => embedding,
            Err(_) => Arc::new(MockProvider::new()),
        };

        Ok((llm_provider, embedding))
    }

    /// OODA-04: Create OpenRouter provider with specific model.
    ///
    /// Like `create_openrouter()` but uses the provided model instead of env/default.
    ///
    /// # Arguments
    ///
    /// * `model` - Model name (e.g., "mistralai/ministral-14b-2512")
    fn create_openrouter_with_model(
        model: &str,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        let api_key = std::env::var("OPENROUTER_API_KEY").map_err(|_| {
            LlmError::ConfigError("OPENROUTER_API_KEY not set for OpenRouter provider".to_string())
        })?;

        if api_key.is_empty() {
            return Err(LlmError::ConfigError(
                "OPENROUTER_API_KEY is empty".to_string(),
            ));
        }

        let mut provider = OpenRouterProvider::new(api_key).with_model(model);

        // Optional site URL and name
        if let Ok(url) = std::env::var("OPENROUTER_SITE_URL") {
            provider = provider.with_site_url(url);
        }
        if let Ok(name) = std::env::var("OPENROUTER_SITE_NAME") {
            provider = provider.with_site_name(name);
        }

        let llm_provider = Arc::new(provider);

        // OpenRouter doesn't provide embeddings, fall back to OpenAI or mock
        let embedding: Arc<dyn EmbeddingProvider> = match Self::create_openai() {
            Ok((_, embedding)) => embedding,
            Err(_) => Arc::new(MockProvider::new()),
        };

        Ok((llm_provider, embedding))
    }

    /// OODA-04: Create OpenAI provider with specific model.
    fn create_openai_with_model(
        model: &str,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            LlmError::ConfigError("OPENAI_API_KEY not set for OpenAI provider".to_string())
        })?;

        if api_key.is_empty() || api_key == "test-key" {
            return Err(LlmError::ConfigError(
                "OPENAI_API_KEY is empty or invalid".to_string(),
            ));
        }

        let provider = Arc::new(OpenAIProvider::new(api_key).with_model(model));
        Ok((provider.clone(), provider))
    }

    /// OODA-04: Create Anthropic provider with specific model.
    fn create_anthropic_with_model(
        model: &str,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
            LlmError::ConfigError("ANTHROPIC_API_KEY not set for Anthropic provider".to_string())
        })?;

        let mut provider = AnthropicProvider::new(&api_key);

        // Support custom base URL for Ollama-style compatibility
        if let Ok(base_url) = std::env::var("ANTHROPIC_BASE_URL") {
            provider = provider.with_base_url(&base_url);
        }
        provider = provider.with_model(model);

        let llm_provider = Arc::new(provider);

        // Anthropic doesn't provide embeddings, fall back to OpenAI or mock
        let embedding: Arc<dyn EmbeddingProvider> = match Self::create_openai() {
            Ok((_, embedding)) => embedding,
            Err(_) => Arc::new(MockProvider::new()),
        };

        Ok((llm_provider, embedding))
    }

    /// OODA-04: Create Gemini provider with specific model.
    /// OODA-95: Supports vertexai: prefixed models for VertexAI endpoint.
    fn create_gemini_with_model(
        model: &str,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        // OODA-95: Check if this is a VertexAI model (prefixed with "vertexai:")
        if model.starts_with("vertexai:") {
            // Strip the "vertexai:" prefix to get the actual model name
            let actual_model = model.strip_prefix("vertexai:").unwrap_or(model);

            // Use VertexAI endpoint with gcloud auto-auth
            let provider = Arc::new(GeminiProvider::from_env_vertex_ai()?.with_model(actual_model));

            // Gemini doesn't have native embeddings API, fall back to OpenAI or mock
            let embedding: Arc<dyn EmbeddingProvider> = match Self::create_openai() {
                Ok((_, embedding)) => embedding,
                Err(_) => Arc::new(MockProvider::new()),
            };

            return Ok((provider, embedding));
        }

        // Regular GoogleAI endpoint (requires GEMINI_API_KEY or GOOGLE_API_KEY)
        let api_key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_API_KEY"))
            .map_err(|_| {
                LlmError::ConfigError(
                    "GEMINI_API_KEY or GOOGLE_API_KEY not set for Gemini provider".to_string(),
                )
            })?;

        let provider = Arc::new(GeminiProvider::new(&api_key).with_model(model));

        // Gemini doesn't have native embeddings API, fall back to OpenAI or mock
        let embedding: Arc<dyn EmbeddingProvider> = match Self::create_openai() {
            Ok((_, embedding)) => embedding,
            Err(_) => Arc::new(MockProvider::new()),
        };

        Ok((provider, embedding))
    }

    /// OODA-04: Create xAI provider with specific model.
    fn create_xai_with_model(
        model: &str,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        // Verify XAI_API_KEY is set before proceeding
        std::env::var("XAI_API_KEY").map_err(|_| {
            LlmError::ConfigError("XAI_API_KEY not set for xAI provider".to_string())
        })?;

        // Use XAI_MODEL to set model, then call from_env
        std::env::set_var("XAI_MODEL", model);
        let provider = XAIProvider::from_env()?;

        let llm_provider = Arc::new(provider);

        // xAI doesn't provide embeddings, fall back to OpenAI or mock
        let embedding: Arc<dyn EmbeddingProvider> = match Self::create_openai() {
            Ok((_, embedding)) => embedding,
            Err(_) => Arc::new(MockProvider::new()),
        };

        Ok((llm_provider, embedding))
    }

    /// OODA-04: Create Ollama provider with specific model.
    fn create_ollama_with_model(
        model: &str,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        // Use OLLAMA_MODEL to set model, then call from_env
        std::env::set_var("OLLAMA_MODEL", model);
        let provider = Arc::new(OllamaProvider::from_env()?);
        Ok((provider.clone(), provider))
    }

    /// OODA-04: Create LM Studio provider with specific model.
    fn create_lmstudio_with_model(
        model: &str,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        // Use LMSTUDIO_MODEL to set model, then call from_env
        std::env::set_var("LMSTUDIO_MODEL", model);
        let provider = Arc::new(LMStudioProvider::from_env()?);
        Ok((provider.clone(), provider))
    }

    /// Create OpenRouter provider from TOML configuration.
    ///
    /// Reads API key from environment variable specified in config.
    fn create_openrouter_from_config(
        config: &ProviderConfig,
        model_name: Option<&str>,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        // Get API key from environment
        let api_key_var = config
            .api_key_env
            .as_deref()
            .unwrap_or("OPENROUTER_API_KEY");
        let api_key = std::env::var(api_key_var).map_err(|_| {
            LlmError::ConfigError(format!("{} not set for OpenRouter provider", api_key_var))
        })?;

        if api_key.is_empty() {
            return Err(LlmError::ConfigError(format!("{} is empty", api_key_var)));
        }

        // Determine model
        let model = model_name
            .map(|s| s.to_string())
            .or_else(|| config.default_llm_model.clone())
            .unwrap_or_else(|| "anthropic/claude-3.5-sonnet".to_string());

        // Create provider with optional base URL
        let mut provider = OpenRouterProvider::new(api_key).with_model(model);

        if let Some(base_url) = &config.base_url {
            provider = provider.with_base_url(base_url);
        }

        let llm_provider = Arc::new(provider);

        // OpenRouter doesn't provide embeddings, fall back to OpenAI or mock
        let embedding: Arc<dyn EmbeddingProvider> = match Self::create_openai() {
            Ok((_, embedding)) => embedding,
            Err(_) => Arc::new(MockProvider::new()),
        };

        Ok((llm_provider, embedding))
    }

    /// Create Ollama provider from environment.
    ///
    /// Uses OllamaProvider::from_env() which reads:
    /// - OLLAMA_HOST
    /// - OLLAMA_MODEL
    /// - OLLAMA_EMBEDDING_MODEL
    fn create_ollama() -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        let provider = Arc::new(OllamaProvider::from_env()?);
        Ok((provider.clone(), provider))
    }

    /// Create LM Studio provider from environment.
    ///
    /// Uses the dedicated LMStudioProvider which reads:
    /// - `LMSTUDIO_HOST`: LM Studio server URL (default: http://localhost:1234)
    /// - `LMSTUDIO_MODEL`: Chat model name (default: gemma2-9b-it)
    /// - `LMSTUDIO_EMBEDDING_MODEL`: Embedding model (default: nomic-embed-text-v1.5)
    /// - `LMSTUDIO_EMBEDDING_DIM`: Embedding dimension (default: 768)
    fn create_lmstudio() -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        let provider = Arc::new(LMStudioProvider::from_env()?);
        Ok((provider.clone(), provider))
    }

    /// OODA-71: Create xAI provider from environment.
    ///
    /// Reads `XAI_API_KEY` environment variable for authentication.
    /// Optional `XAI_MODEL` for model selection (default: grok-4).
    /// Optional `XAI_BASE_URL` for custom endpoint.
    ///
    /// xAI doesn't provide embeddings API, so we fall back to OpenAI or mock.
    fn create_xai() -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        let provider = Arc::new(XAIProvider::from_env()?);

        // xAI doesn't provide embeddings, fall back to OpenAI or mock
        let embedding: Arc<dyn EmbeddingProvider> = match Self::create_openai() {
            Ok((_, embedding)) => embedding,
            Err(_) => Arc::new(MockProvider::new()),
        };

        Ok((provider, embedding))
    }

    /// OODA-80: Create HuggingFace Hub provider from environment.
    ///
    /// Reads `HF_TOKEN` or `HUGGINGFACE_TOKEN` environment variable for authentication.
    /// Optional `HF_MODEL` for model selection (default: meta-llama/Meta-Llama-3.1-70B-Instruct).
    /// Optional `HF_BASE_URL` for custom endpoint.
    ///
    /// HuggingFace requires a separate endpoint for embeddings (not yet supported),
    /// so we fall back to OpenAI or mock.
    fn create_huggingface() -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        let provider = Arc::new(HuggingFaceProvider::from_env()?);

        // HuggingFace embeddings require a different endpoint, fall back to OpenAI or mock
        let embedding: Arc<dyn EmbeddingProvider> = match Self::create_openai() {
            Ok((_, embedding)) => embedding,
            Err(_) => Arc::new(MockProvider::new()),
        };

        Ok((provider, embedding))
    }

    /// Create VSCode Copilot provider from environment.
    ///
    /// Uses the VsCodeCopilotProvider which reads:
    /// - `VSCODE_COPILOT_DIRECT`: Enable direct API mode (default: true)
    /// - `VSCODE_COPILOT_PROXY_URL`: Proxy URL if not using direct mode
    /// - `VSCODE_COPILOT_MODEL`: Model name (default: gpt-4o-mini)
    /// - `VSCODE_COPILOT_ACCOUNT_TYPE`: Account type (individual/business/enterprise)
    /// - `VSCODE_COPILOT_EMBEDDING_MODEL`: Embedding model (default: text-embedding-3-small)
    ///
    /// Note: Direct mode is now the default. No external proxy required.
    /// For legacy proxy mode, set `VSCODE_COPILOT_DIRECT=false`.
    ///
    /// VsCodeCopilotProvider now implements both LLMProvider and EmbeddingProvider,
    /// using the Copilot API's /embeddings endpoint for text embeddings.
    fn create_vscode_copilot() -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        let model =
            std::env::var("VSCODE_COPILOT_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

        // Builder reads VSCODE_COPILOT_DIRECT, VSCODE_COPILOT_ACCOUNT_TYPE,
        // and VSCODE_COPILOT_EMBEDDING_MODEL automatically
        let builder = VsCodeCopilotProvider::new().model(model);

        let provider = Arc::new(builder.build().map_err(|e| {
            LlmError::ConfigError(format!("Failed to create VSCode Copilot provider: {}", e))
        })?);

        // VsCodeCopilotProvider implements both LLMProvider and EmbeddingProvider
        // Use same provider instance for both (shares HTTP client and token manager)
        Ok((provider.clone(), provider))
    }

    /// Create mock provider for testing.
    ///
    /// Always returns deterministic responses.
    fn create_mock() -> (Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>) {
        let provider = Arc::new(MockProvider::new());
        (provider.clone(), provider)
    }

    /// Create Mistral AI provider from environment.
    ///
    /// Reads `MISTRAL_API_KEY` environment variable for authentication.
    /// Optional `MISTRAL_MODEL` for model selection (default: mistral-small-latest).
    /// Optional `MISTRAL_EMBEDDING_MODEL` (default: mistral-embed).
    /// Optional `MISTRAL_BASE_URL` for custom endpoint.
    ///
    /// MistralProvider supports both LLM and embeddings natively.
    fn create_mistral() -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        let provider = Arc::new(MistralProvider::from_env()?);
        Ok((provider.clone(), provider))
    }

    /// Create Mistral AI provider with a specific model override.
    fn create_mistral_with_model(
        model: &str,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        let provider = Arc::new(MistralProvider::from_env()?.with_model(model));
        Ok((provider.clone(), provider))
    }

    /// Create Mistral AI provider from TOML configuration.
    fn create_mistral_from_config(
        config: &ProviderConfig,
        model_name: Option<&str>,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        let mut provider = MistralProvider::from_config(config)?;
        if let Some(model) = model_name {
            provider = provider.with_model(model);
        }
        let provider = Arc::new(provider);
        Ok((provider.clone(), provider))
    }

    /// Get embedding dimension for current provider configuration.
    ///
    /// Useful for configuring vector storage with the correct dimension.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// std::env::set_var("EDGEQUAKE_LLM_PROVIDER", "ollama");
    /// let dim = ProviderFactory::embedding_dimension()?;
    /// assert_eq!(dim, 768); // embeddinggemma dimension
    /// ```
    pub fn embedding_dimension() -> Result<usize> {
        let (_, embedding_provider) = Self::from_env()?;
        Ok(embedding_provider.dimension())
    }

    /// Create an embedding provider from workspace configuration.
    ///
    /// This is used to create workspace-specific embedding providers for query execution.
    /// The provider is configured with the workspace's embedding model and dimension.
    ///
    /// @implements SPEC-032: Workspace-specific embedding in query process
    ///
    /// # Arguments
    ///
    /// * `provider_name` - Provider type (e.g., "openai", "ollama", "lmstudio", "vscode-copilot", "mock")
    /// * `model` - Embedding model name (e.g., "text-embedding-3-small", "embeddinggemma:latest")
    /// * `dimension` - Embedding dimension (e.g., 1536, 768)
    ///
    /// # Returns
    ///
    /// Returns an `Arc<dyn EmbeddingProvider>` configured for the workspace.
    ///
    /// # Errors
    ///
    /// Returns error if the provider type is unknown or required configuration is missing.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let provider = ProviderFactory::create_embedding_provider(
    ///     "ollama",
    ///     "embeddinggemma:latest",
    ///     768,
    /// )?;
    /// assert_eq!(provider.dimension(), 768);
    /// ```
    pub fn create_embedding_provider(
        provider_name: &str,
        model: &str,
        _dimension: usize,
    ) -> Result<Arc<dyn EmbeddingProvider>> {
        let provider_type = ProviderType::from_str(provider_name).ok_or_else(|| {
            LlmError::ConfigError(format!(
                "Unknown embedding provider: {}. Valid: openai, ollama, lmstudio, vscode-copilot, mock",
                provider_name
            ))
        })?;

        match provider_type {
            ProviderType::OpenAI => {
                let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
                    LlmError::ConfigError(
                        "OPENAI_API_KEY required for OpenAI embedding provider".to_string(),
                    )
                })?;
                // OpenAI provider with specific embedding model
                // Note: OpenAI dimension is auto-detected from model name
                let provider = OpenAIProvider::new(api_key).with_embedding_model(model);
                Ok(Arc::new(provider))
            }
            ProviderType::Anthropic => {
                // Anthropic doesn't have embeddings, fall back to mock
                warn!("Anthropic doesn't support embeddings, using mock provider");
                Ok(Arc::new(MockProvider::new()))
            }
            ProviderType::OpenRouter => {
                // OpenRouter doesn't have embeddings, fall back to mock
                warn!("OpenRouter doesn't support embeddings, using mock provider");
                Ok(Arc::new(MockProvider::new()))
            }
            ProviderType::XAI => {
                // xAI doesn't have embeddings API, fall back to mock
                warn!("xAI doesn't support embeddings, using mock provider");
                Ok(Arc::new(MockProvider::new()))
            }
            ProviderType::HuggingFace => {
                // HuggingFace requires separate endpoint for embeddings, fall back to mock
                warn!("HuggingFace LLM provider doesn't support embeddings, using mock provider");
                Ok(Arc::new(MockProvider::new()))
            }
            ProviderType::Gemini => {
                // OODA-73: Gemini doesn't have embeddings API, fall back to mock
                warn!("Gemini doesn't support embeddings, using mock provider");
                Ok(Arc::new(MockProvider::new()))
            }
            ProviderType::Ollama => {
                // Ollama provider with specific embedding model
                let host = std::env::var("OLLAMA_HOST")
                    .unwrap_or_else(|_| "http://localhost:11434".to_string());
                let provider = OllamaProvider::builder()
                    .host(&host)
                    .embedding_model(model)
                    .build()?;
                Ok(Arc::new(provider))
            }
            ProviderType::LMStudio => {
                // LM Studio provider with specific embedding model
                let host = std::env::var("LMSTUDIO_HOST")
                    .unwrap_or_else(|_| "http://localhost:1234".to_string());
                let provider = LMStudioProvider::builder()
                    .host(&host)
                    .embedding_model(model)
                    .build()?;
                Ok(Arc::new(provider))
            }
            ProviderType::Mock => {
                // Mock provider ignores model/dimension and uses defaults
                Ok(Arc::new(MockProvider::new()))
            }
            ProviderType::VsCodeCopilot => {
                // VsCodeCopilot supports embeddings via Copilot API /embeddings endpoint
                // Uses text-embedding-3-small by default (1536 dimensions)
                let provider = VsCodeCopilotProvider::new()
                    .embedding_model(model)
                    .build()
                    .map_err(|e| LlmError::ApiError(e.to_string()))?;
                Ok(Arc::new(provider))
            }
            ProviderType::Mistral => {
                // Mistral supports embeddings natively via mistral-embed model
                let provider = MistralProvider::from_env()?.with_embedding_model(model);
                Ok(Arc::new(provider))
            }
        }
    }

    /// Create an LLM provider from workspace configuration.
    ///
    /// This is used to create workspace-specific LLM providers for ingestion/extraction.
    /// The provider is configured with the workspace's LLM model.
    ///
    /// @implements SPEC-032: Workspace-specific LLM in ingestion process
    ///
    /// # Arguments
    ///
    /// * `provider_name` - Provider type (e.g., "openai", "ollama", "lmstudio", "mock")
    /// * `model` - LLM model name (e.g., "gpt-4o-mini", "gemma3:12b")
    ///
    /// # Returns
    ///
    /// Returns an `Arc<dyn LLMProvider>` configured for the workspace.
    ///
    /// # Errors
    ///
    /// Returns error if the provider type is unknown or required configuration is missing.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let provider = ProviderFactory::create_llm_provider(
    ///     "ollama",
    ///     "gemma3:12b",
    /// )?;
    /// assert_eq!(provider.model(), "gemma3:12b");
    /// ```
    pub fn create_llm_provider(provider_name: &str, model: &str) -> Result<Arc<dyn LLMProvider>> {
        let provider_type = ProviderType::from_str(provider_name).ok_or_else(|| {
            LlmError::ConfigError(format!(
                "Unknown LLM provider: {}. Valid: openai, ollama, lmstudio, mock",
                provider_name
            ))
        })?;

        match provider_type {
            ProviderType::OpenAI => {
                let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
                    LlmError::ConfigError(
                        "OPENAI_API_KEY required for OpenAI LLM provider".to_string(),
                    )
                })?;
                // OpenAI provider with specific model
                let provider = OpenAIProvider::new(api_key).with_model(model);
                Ok(Arc::new(provider))
            }
            ProviderType::Anthropic => {
                let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
                    LlmError::ConfigError(
                        "ANTHROPIC_API_KEY required for Anthropic LLM provider".to_string(),
                    )
                })?;
                // Anthropic provider with specific model
                let provider = AnthropicProvider::new(api_key).with_model(model);
                Ok(Arc::new(provider))
            }
            ProviderType::OpenRouter => {
                let api_key = std::env::var("OPENROUTER_API_KEY").map_err(|_| {
                    LlmError::ConfigError(
                        "OPENROUTER_API_KEY required for OpenRouter LLM provider".to_string(),
                    )
                })?;
                // OpenRouter provider with specific model
                let provider = OpenRouterProvider::new(api_key).with_model(model);
                Ok(Arc::new(provider))
            }
            ProviderType::XAI => {
                // OODA-71: xAI Grok provider with specific model
                let provider = XAIProvider::from_env()?.with_model(model);
                Ok(Arc::new(provider))
            }
            ProviderType::HuggingFace => {
                // OODA-80: HuggingFace Hub provider with specific model
                let provider = HuggingFaceProvider::from_env()?.with_model(model);
                Ok(Arc::new(provider))
            }
            ProviderType::Gemini => {
                // OODA-73/95: Gemini provider with specific model
                // Handle vertexai: prefix for VertexAI endpoint
                if model.starts_with("vertexai:") {
                    let actual_model = model.strip_prefix("vertexai:").unwrap_or(model);
                    let provider = GeminiProvider::from_env_vertex_ai()?.with_model(actual_model);
                    Ok(Arc::new(provider))
                } else {
                    let provider = GeminiProvider::from_env()?.with_model(model);
                    Ok(Arc::new(provider))
                }
            }
            ProviderType::Ollama => {
                // Ollama provider with specific model
                let host = std::env::var("OLLAMA_HOST")
                    .unwrap_or_else(|_| "http://localhost:11434".to_string());
                let provider = OllamaProvider::builder().host(&host).model(model).build()?;
                Ok(Arc::new(provider))
            }
            ProviderType::LMStudio => {
                // LM Studio provider with specific model
                let host = std::env::var("LMSTUDIO_HOST")
                    .unwrap_or_else(|_| "http://localhost:1234".to_string());
                let provider = LMStudioProvider::builder()
                    .host(&host)
                    .model(model)
                    .build()?;
                Ok(Arc::new(provider))
            }
            ProviderType::Mock => {
                // Mock provider ignores model and uses defaults
                Ok(Arc::new(MockProvider::new()))
            }
            ProviderType::VsCodeCopilot => {
                // VsCodeCopilot provider with specific model
                let proxy_url = std::env::var("VSCODE_COPILOT_PROXY_URL")
                    .unwrap_or_else(|_| "http://localhost:4141".to_string());
                let provider = VsCodeCopilotProvider::new()
                    .proxy_url(&proxy_url)
                    .model(model)
                    .build()?;
                Ok(Arc::new(provider))
            }
            ProviderType::Mistral => {
                // Mistral provider with specific model
                let provider = MistralProvider::from_env()?.with_model(model);
                Ok(Arc::new(provider))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_provider_type_parsing() {
        assert_eq!(ProviderType::from_str("openai"), Some(ProviderType::OpenAI));
        assert_eq!(ProviderType::from_str("OLLAMA"), Some(ProviderType::Ollama));
        assert_eq!(
            ProviderType::from_str("lmstudio"),
            Some(ProviderType::LMStudio)
        );
        assert_eq!(
            ProviderType::from_str("lm-studio"),
            Some(ProviderType::LMStudio)
        );
        assert_eq!(
            ProviderType::from_str("lm_studio"),
            Some(ProviderType::LMStudio)
        );
        assert_eq!(ProviderType::from_str("mock"), Some(ProviderType::Mock));

        // OODA-73: Test Gemini parsing
        assert_eq!(ProviderType::from_str("gemini"), Some(ProviderType::Gemini));
        assert_eq!(ProviderType::from_str("google"), Some(ProviderType::Gemini));
        assert_eq!(ProviderType::from_str("vertex"), Some(ProviderType::Gemini));
        assert_eq!(
            ProviderType::from_str("vertexai"),
            Some(ProviderType::Gemini)
        );

        // Test OpenRouter parsing
        assert_eq!(
            ProviderType::from_str("openrouter"),
            Some(ProviderType::OpenRouter)
        );

        // OODA-71: Test xAI parsing
        assert_eq!(ProviderType::from_str("xai"), Some(ProviderType::XAI));
        assert_eq!(ProviderType::from_str("grok"), Some(ProviderType::XAI));

        // OODA-80: Test HuggingFace parsing
        assert_eq!(
            ProviderType::from_str("huggingface"),
            Some(ProviderType::HuggingFace)
        );
        assert_eq!(
            ProviderType::from_str("hf"),
            Some(ProviderType::HuggingFace)
        );
        assert_eq!(
            ProviderType::from_str("hugging-face"),
            Some(ProviderType::HuggingFace)
        );

        assert_eq!(ProviderType::from_str("invalid"), None);
        assert_eq!(ProviderType::from_str(""), None);
    }

    #[test]
    fn test_mock_creation() {
        let (llm, embedding) = ProviderFactory::create_mock();
        assert_eq!(llm.name(), "mock");
        assert_eq!(embedding.name(), "mock");
        assert_eq!(embedding.dimension(), 1536);
    }

    #[test]
    fn test_explicit_mock_creation() {
        let (llm, embedding) = ProviderFactory::create(ProviderType::Mock).unwrap();
        assert_eq!(llm.name(), "mock");
        assert_eq!(embedding.dimension(), 1536);
    }

    #[test]
    #[serial]
    fn test_from_env_fallback_to_mock() {
        // Clear all provider environment variables
        std::env::remove_var("EDGEQUAKE_LLM_PROVIDER");
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("XAI_API_KEY");
        std::env::remove_var("GOOGLE_API_KEY");
        std::env::remove_var("GEMINI_API_KEY");
        std::env::remove_var("OPENROUTER_API_KEY");
        std::env::remove_var("ANTHROPIC_API_KEY");
        std::env::remove_var("AZURE_OPENAI_API_KEY");
        std::env::remove_var("HUGGINGFACE_API_KEY"); // OODA-04: Added missing env var
        std::env::remove_var("HF_TOKEN"); // OODA-04: Primary HuggingFace token
        std::env::remove_var("HUGGINGFACE_TOKEN"); // OODA-04: Alternative HuggingFace token
        std::env::remove_var("OLLAMA_HOST");
        std::env::remove_var("OLLAMA_MODEL");
        std::env::remove_var("LMSTUDIO_HOST");
        std::env::remove_var("LMSTUDIO_MODEL");
        std::env::remove_var("MISTRAL_API_KEY");

        let (llm, _) = ProviderFactory::from_env().unwrap();
        assert_eq!(llm.name(), "mock");
    }

    #[test]
    #[serial]
    fn test_explicit_provider_env() {
        // Clean up first to avoid interference from other tests
        std::env::remove_var("EDGEQUAKE_LLM_PROVIDER");
        std::env::remove_var("OLLAMA_HOST");
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("XAI_API_KEY");
        std::env::remove_var("GOOGLE_API_KEY");
        std::env::remove_var("GEMINI_API_KEY");
        std::env::remove_var("OPENROUTER_API_KEY");
        std::env::remove_var("ANTHROPIC_API_KEY");
        std::env::remove_var("AZURE_OPENAI_API_KEY");
        std::env::remove_var("LMSTUDIO_HOST");

        std::env::set_var("EDGEQUAKE_LLM_PROVIDER", "mock");
        let (llm, _) = ProviderFactory::from_env().unwrap();
        assert_eq!(llm.name(), "mock");

        // Clean up after
        std::env::remove_var("EDGEQUAKE_LLM_PROVIDER");
    }

    #[test]
    #[serial]
    fn test_lmstudio_auto_detection() {
        // Clean up first to avoid interference from other tests
        std::env::remove_var("EDGEQUAKE_LLM_PROVIDER");
        std::env::remove_var("OLLAMA_HOST");
        std::env::remove_var("OLLAMA_MODEL");
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("XAI_API_KEY");
        std::env::remove_var("GOOGLE_API_KEY");
        std::env::remove_var("GEMINI_API_KEY");
        std::env::remove_var("OPENROUTER_API_KEY");
        std::env::remove_var("ANTHROPIC_API_KEY");
        std::env::remove_var("AZURE_OPENAI_API_KEY");

        // Set LM Studio environment
        std::env::set_var("LMSTUDIO_HOST", "http://localhost:1234");
        let (llm, embedding) = ProviderFactory::from_env().unwrap();
        assert_eq!(llm.name(), "lmstudio");
        assert_eq!(embedding.name(), "lmstudio");

        // Clean up after
        std::env::remove_var("LMSTUDIO_HOST");
    }

    #[test]
    #[serial]
    fn test_lmstudio_model_detection() {
        // Clean up first to avoid interference from other tests
        std::env::remove_var("EDGEQUAKE_LLM_PROVIDER");
        std::env::remove_var("OLLAMA_HOST");
        std::env::remove_var("OLLAMA_MODEL");
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("XAI_API_KEY");
        std::env::remove_var("GOOGLE_API_KEY");
        std::env::remove_var("GEMINI_API_KEY");
        std::env::remove_var("OPENROUTER_API_KEY");
        std::env::remove_var("ANTHROPIC_API_KEY");
        std::env::remove_var("AZURE_OPENAI_API_KEY");
        std::env::remove_var("LMSTUDIO_HOST");

        // Set LM Studio model only
        std::env::set_var("LMSTUDIO_MODEL", "mistral-7b");
        let (llm, _) = ProviderFactory::from_env().unwrap();
        assert_eq!(llm.name(), "lmstudio");

        // Clean up after
        std::env::remove_var("LMSTUDIO_MODEL");
    }

    #[test]
    fn test_explicit_lmstudio_creation() {
        let (llm, embedding) = ProviderFactory::create(ProviderType::LMStudio).unwrap();
        assert_eq!(llm.name(), "lmstudio");
        assert_eq!(embedding.name(), "lmstudio");
        // Default LM Studio embedding dimension is 768 (nomic-embed-text-v1.5)
        assert_eq!(embedding.dimension(), 768);
    }

    #[test]
    #[serial]
    fn test_invalid_provider_env() {
        // Clean up first to avoid interference from other tests
        std::env::remove_var("EDGEQUAKE_LLM_PROVIDER");
        std::env::remove_var("OLLAMA_HOST");
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("XAI_API_KEY");
        std::env::remove_var("GOOGLE_API_KEY");
        std::env::remove_var("GEMINI_API_KEY");
        std::env::remove_var("OPENROUTER_API_KEY");
        std::env::remove_var("ANTHROPIC_API_KEY");
        std::env::remove_var("AZURE_OPENAI_API_KEY");
        std::env::remove_var("LMSTUDIO_HOST");

        std::env::set_var("EDGEQUAKE_LLM_PROVIDER", "invalid_provider");
        let result = ProviderFactory::from_env();
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Unknown provider type"));
        }

        // Clean up after
        std::env::remove_var("EDGEQUAKE_LLM_PROVIDER");
    }

    #[test]
    #[serial]
    fn test_openai_creation_requires_api_key() {
        // Clean up first to avoid interference from other tests
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("EDGEQUAKE_LLM_PROVIDER");
        std::env::remove_var("OLLAMA_HOST");
        std::env::remove_var("LMSTUDIO_HOST");

        let result = ProviderFactory::create(ProviderType::OpenAI);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("OPENAI_API_KEY not set"));
        }
    }

    #[test]
    #[serial]
    fn test_embedding_dimension_detection() {
        // Clean up first to avoid interference from other tests
        std::env::remove_var("EDGEQUAKE_LLM_PROVIDER");
        std::env::remove_var("OLLAMA_HOST");
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("LMSTUDIO_HOST");

        std::env::set_var("EDGEQUAKE_LLM_PROVIDER", "mock");
        let dim = ProviderFactory::embedding_dimension().unwrap();
        assert_eq!(dim, 1536);

        // Clean up after
        std::env::remove_var("EDGEQUAKE_LLM_PROVIDER");
    }

    #[test]
    #[serial]
    fn test_provider_priority_ollama_over_lmstudio() {
        // Clean up first
        std::env::remove_var("EDGEQUAKE_LLM_PROVIDER");
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("LMSTUDIO_HOST");
        std::env::remove_var("LMSTUDIO_MODEL");

        // Set both Ollama and LM Studio - Ollama should win
        std::env::set_var("OLLAMA_HOST", "http://localhost:11434");
        std::env::set_var("LMSTUDIO_HOST", "http://localhost:1234");

        let (llm, _) = ProviderFactory::from_env().unwrap();
        assert_eq!(llm.name(), "ollama");

        // Clean up after
        std::env::remove_var("OLLAMA_HOST");
        std::env::remove_var("LMSTUDIO_HOST");
    }

    #[test]
    fn test_create_with_model_mock_none() {
        // create_with_model with Mock and None model → delegates to create
        let (llm, emb) = ProviderFactory::create_with_model(ProviderType::Mock, None).unwrap();
        assert_eq!(llm.name(), "mock");
        assert_eq!(emb.name(), "mock");
    }

    #[test]
    fn test_create_with_model_mock_some() {
        // create_with_model with Mock and Some model → still creates mock
        let (llm, _) =
            ProviderFactory::create_with_model(ProviderType::Mock, Some("any-model")).unwrap();
        assert_eq!(llm.name(), "mock");
    }

    #[test]
    fn test_create_embedding_provider_mock() {
        let provider =
            ProviderFactory::create_embedding_provider("mock", "mock-model", 1536).unwrap();
        assert_eq!(provider.name(), "mock");
        assert_eq!(provider.dimension(), 1536);
    }

    #[test]
    fn test_create_embedding_provider_unknown() {
        let result = ProviderFactory::create_embedding_provider("unknown", "model", 1536);
        match result {
            Err(e) => assert!(e.to_string().contains("Unknown embedding provider")),
            Ok(_) => panic!("Expected error for unknown provider"),
        }
    }

    #[test]
    fn test_create_llm_provider_mock() {
        let provider = ProviderFactory::create_llm_provider("mock", "mock-model").unwrap();
        assert_eq!(provider.name(), "mock");
    }

    #[test]
    fn test_create_llm_provider_unknown() {
        let result = ProviderFactory::create_llm_provider("unknown", "model");
        match result {
            Err(e) => assert!(e.to_string().contains("Unknown LLM provider")),
            Ok(_) => panic!("Expected error for unknown provider"),
        }
    }

    #[test]
    fn test_provider_type_debug() {
        // Verify Debug trait is derived
        let pt = ProviderType::Mock;
        let debug = format!("{:?}", pt);
        assert_eq!(debug, "Mock");
    }

    #[test]
    fn test_provider_type_clone_eq() {
        let pt1 = ProviderType::OpenAI;
        let pt2 = pt1;
        assert_eq!(pt1, pt2);
        assert_ne!(pt1, ProviderType::Ollama);
    }

    #[test]
    fn test_from_config_azure_error() {
        use crate::model_config::{ProviderConfig, ProviderType as ConfigProviderType};
        let config = ProviderConfig {
            provider_type: ConfigProviderType::Azure,
            ..ProviderConfig::default()
        };
        let result = ProviderFactory::from_config(&config);
        match result {
            Err(e) => assert!(e.to_string().contains("Azure OpenAI")),
            Ok(_) => panic!("Expected error for Azure config"),
        }
    }

    #[test]
    fn test_from_config_mock() {
        use crate::model_config::{ProviderConfig, ProviderType as ConfigProviderType};
        let config = ProviderConfig {
            provider_type: ConfigProviderType::Mock,
            ..ProviderConfig::default()
        };
        let (llm, emb) = ProviderFactory::from_config(&config).unwrap();
        assert_eq!(llm.name(), "mock");
        assert_eq!(emb.name(), "mock");
    }

    #[test]
    fn test_vscode_copilot_parsing() {
        assert_eq!(
            ProviderType::from_str("vscode"),
            Some(ProviderType::VsCodeCopilot)
        );
        assert_eq!(
            ProviderType::from_str("copilot"),
            Some(ProviderType::VsCodeCopilot)
        );
        assert_eq!(
            ProviderType::from_str("vscode-copilot"),
            Some(ProviderType::VsCodeCopilot)
        );
    }

    // OODA-41: Test create_embedding_provider mock fallback paths
    #[test]
    fn test_create_embedding_provider_anthropic_fallback() {
        // Anthropic doesn't support embeddings, should fall back to mock
        let provider =
            ProviderFactory::create_embedding_provider("anthropic", "any-model", 1536).unwrap();
        assert_eq!(provider.name(), "mock");
    }

    #[test]
    fn test_create_embedding_provider_openrouter_fallback() {
        // OpenRouter doesn't support embeddings, should fall back to mock
        let provider =
            ProviderFactory::create_embedding_provider("openrouter", "any-model", 1536).unwrap();
        assert_eq!(provider.name(), "mock");
    }

    #[test]
    fn test_create_embedding_provider_xai_fallback() {
        // xAI doesn't support embeddings, should fall back to mock
        let provider =
            ProviderFactory::create_embedding_provider("xai", "any-model", 1536).unwrap();
        assert_eq!(provider.name(), "mock");
    }

    #[test]
    fn test_create_embedding_provider_huggingface_fallback() {
        // HuggingFace LLM provider doesn't support embeddings, should fall back to mock
        let provider =
            ProviderFactory::create_embedding_provider("huggingface", "any-model", 768).unwrap();
        assert_eq!(provider.name(), "mock");
    }

    #[test]
    fn test_create_embedding_provider_gemini_fallback() {
        // Gemini doesn't support embeddings, should fall back to mock
        let provider =
            ProviderFactory::create_embedding_provider("gemini", "any-model", 768).unwrap();
        assert_eq!(provider.name(), "mock");
    }

    #[test]
    fn test_create_embedding_provider_ollama() {
        // Ollama creates a real provider (works without connectivity for creation)
        let provider =
            ProviderFactory::create_embedding_provider("ollama", "nomic-embed-text", 768).unwrap();
        assert_eq!(provider.name(), "ollama");
    }

    #[test]
    fn test_create_embedding_provider_lmstudio() {
        // LMStudio creates a real provider (works without connectivity for creation)
        let provider =
            ProviderFactory::create_embedding_provider("lmstudio", "nomic-embed-text-v1.5", 768)
                .unwrap();
        assert_eq!(provider.name(), "lmstudio");
    }

    #[test]
    fn test_create_embedding_provider_vscode_copilot() {
        // VsCodeCopilot creates a real provider (works without connectivity for creation)
        let provider = ProviderFactory::create_embedding_provider(
            "vscode-copilot",
            "text-embedding-3-small",
            1536,
        )
        .unwrap();
        assert_eq!(provider.name(), "vscode-copilot");
    }

    // OODA-41: Test from_config with different provider types
    #[test]
    fn test_from_config_ollama() {
        use crate::model_config::{ProviderConfig, ProviderType as ConfigProviderType};
        let config = ProviderConfig {
            provider_type: ConfigProviderType::Ollama,
            ..ProviderConfig::default()
        };
        let (llm, emb) = ProviderFactory::from_config(&config).unwrap();
        assert_eq!(llm.name(), "ollama");
        assert_eq!(emb.name(), "ollama");
    }

    #[test]
    fn test_from_config_lmstudio() {
        use crate::model_config::{ProviderConfig, ProviderType as ConfigProviderType};
        let config = ProviderConfig {
            provider_type: ConfigProviderType::LMStudio,
            ..ProviderConfig::default()
        };
        let (llm, emb) = ProviderFactory::from_config(&config).unwrap();
        assert_eq!(llm.name(), "lmstudio");
        assert_eq!(emb.name(), "lmstudio");
    }

    #[test]
    #[serial]
    fn test_from_config_openai_requires_api_key() {
        use crate::model_config::{ProviderConfig, ProviderType as ConfigProviderType};
        std::env::remove_var("OPENAI_API_KEY");
        let config = ProviderConfig {
            provider_type: ConfigProviderType::OpenAI,
            ..ProviderConfig::default()
        };
        let result = ProviderFactory::from_config(&config);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("OPENAI_API_KEY"));
        }
    }

    // OODA-41: Test create_with_model for local providers
    #[test]
    fn test_create_with_model_ollama() {
        let (llm, _) =
            ProviderFactory::create_with_model(ProviderType::Ollama, Some("llama3:8b")).unwrap();
        assert_eq!(llm.name(), "ollama");
        assert_eq!(llm.model(), "llama3:8b");
    }

    #[test]
    fn test_create_with_model_lmstudio() {
        let (llm, _) =
            ProviderFactory::create_with_model(ProviderType::LMStudio, Some("mistral-7b")).unwrap();
        assert_eq!(llm.name(), "lmstudio");
        assert_eq!(llm.model(), "mistral-7b");
    }
}
