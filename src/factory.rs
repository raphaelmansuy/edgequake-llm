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
use crate::providers::azure_openai::AzureOpenAIProvider;
use crate::providers::gemini::GeminiProvider;
use crate::providers::huggingface::HuggingFaceProvider;
use crate::providers::jina::JinaProvider;
use crate::providers::lmstudio::LMStudioProvider;
use crate::providers::mistral::MistralProvider;
use crate::providers::openai_compatible::OpenAICompatibleProvider;
use crate::providers::openrouter::OpenRouterProvider;
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
    /// Google AI Studio Gemini provider (uses GEMINI_API_KEY / GOOGLE_API_KEY).
    ///
    /// Endpoint: generativelanguage.googleapis.com (free tier + pay-as-you-go).
    /// WHY separate from VertexAI: these are completely different services with
    /// different auth (API key vs ADC/service-account), different quota systems
    /// (free 20 RPM vs Vertex paid), and different billing. Collapsing them into
    /// one variant leads to silent endpoint mis-routing when both credentials are
    /// present in the environment (GEMINI_API_KEY wins, Vertex AI is ignored).
    Gemini,
    /// Google Cloud Vertex AI Gemini provider (uses ADC / service-account / gcloud).
    ///
    /// Endpoint: {region}-aiplatform.googleapis.com (paid, high-quota).
    /// Requires: GOOGLE_CLOUD_PROJECT (and optionally GOOGLE_CLOUD_REGION).
    /// Auth: GOOGLE_ACCESS_TOKEN or gcloud auth application-default login.
    ///
    /// WHY separate from Gemini: see Gemini variant doc above.
    VertexAI,
    /// OpenRouter provider (200+ models)
    OpenRouter,
    /// xAI provider (Grok models via api.x.ai)
    XAI,
    /// HuggingFace Hub provider (open-source models)
    HuggingFace,
    /// Generic OpenAI-compatible provider (Groq, Together, DeepSeek, custom gateways)
    OpenAICompatible,
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
    /// Azure OpenAI Service (enterprise deployments)
    AzureOpenAI,
    /// AWS Bedrock Runtime (Converse API)
    #[cfg(feature = "bedrock")]
    Bedrock,
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
    /// assert_eq!(ProviderType::from_str("azure"), Some(ProviderType::AzureOpenAI));
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "openai" => Some(Self::OpenAI),
            "anthropic" | "claude" => Some(Self::Anthropic),
            "gemini" | "google" => Some(Self::Gemini),
            // WHY separate: Vertex AI uses different auth, quotas, billing and
            // endpoint (aiplatform.googleapis.com) vs Google AI Studio
            // (generativelanguage.googleapis.com).  Mapping both to Gemini caused
            // GEMINI_API_KEY to win silently when both credentials were present.
            "vertex" | "vertexai" => Some(Self::VertexAI),
            "openrouter" | "open-router" => Some(Self::OpenRouter),
            "xai" | "grok" => Some(Self::XAI),
            "huggingface" | "hf" | "hugging-face" | "hugging_face" => Some(Self::HuggingFace),
            "openai-compatible" | "openai_compatible" | "openaicompatible" | "compatible" => {
                Some(Self::OpenAICompatible)
            }
            "ollama" => Some(Self::Ollama),
            "lmstudio" | "lm-studio" | "lm_studio" => Some(Self::LMStudio),
            "vscode" | "vscode-copilot" | "copilot" => Some(Self::VsCodeCopilot),
            "mock" => Some(Self::Mock),
            "mistral" | "mistral-ai" | "mistralai" => Some(Self::Mistral),
            "azure" | "azure-openai" | "azure_openai" | "azureopenai" => Some(Self::AzureOpenAI),
            #[cfg(feature = "bedrock")]
            "bedrock" | "aws-bedrock" | "aws_bedrock" => Some(Self::Bedrock),
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
                "Unknown provider type: {}. Valid options: openai, anthropic, gemini, vertexai, openrouter, xai, huggingface, openai-compatible, ollama, lmstudio, vscode-copilot, mistral, azure, bedrock, mock",
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
        if AnthropicProvider::resolve_api_key_from_env().is_ok() {
            return Self::create(ProviderType::Anthropic);
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

        // Azure OpenAI detection (CONTENTGEN variant or standard variant)
        let azure_key = std::env::var("AZURE_OPENAI_CONTENTGEN_API_KEY")
            .or_else(|_| std::env::var("AZURE_OPENAI_API_KEY"));
        if let Ok(api_key) = azure_key {
            if !api_key.is_empty() {
                // Also verify endpoint is set to avoid false positives
                let azure_endpoint = std::env::var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT")
                    .or_else(|_| std::env::var("AZURE_OPENAI_ENDPOINT"));
                if azure_endpoint.is_ok() {
                    return Self::create(ProviderType::AzureOpenAI);
                }
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
            ProviderType::VertexAI => Self::create_vertex_ai(),
            ProviderType::OpenRouter => Self::create_openrouter(),
            ProviderType::XAI => Self::create_xai(),
            ProviderType::HuggingFace => Self::create_huggingface(),
            ProviderType::OpenAICompatible => Self::create_openai_compatible_from_env(),
            ProviderType::Ollama => Self::create_ollama(),
            ProviderType::LMStudio => Self::create_lmstudio(),
            ProviderType::VsCodeCopilot => Self::create_vscode_copilot(),
            ProviderType::Mock => Ok(Self::create_mock()),
            ProviderType::Mistral => Self::create_mistral(),
            ProviderType::AzureOpenAI => Self::create_azure_openai(),
            #[cfg(feature = "bedrock")]
            ProviderType::Bedrock => Self::create_bedrock(),
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
                ProviderType::VertexAI => Self::create_vertex_ai_with_model(m),
                ProviderType::XAI => Self::create_xai_with_model(m),
                ProviderType::OpenAI => Self::create_openai_with_model(m),
                ProviderType::OpenAICompatible => {
                    Self::create_openai_compatible_from_env_with_model(m)
                }
                ProviderType::Ollama => Self::create_ollama_with_model(m),
                ProviderType::LMStudio => Self::create_lmstudio_with_model(m),
                // These don't need model override
                ProviderType::HuggingFace => Self::create_huggingface(),
                ProviderType::VsCodeCopilot => Self::create_vscode_copilot(),
                ProviderType::Mock => Ok(Self::create_mock()),
                ProviderType::Mistral => Self::create_mistral_with_model(m),
                ProviderType::AzureOpenAI => Self::create_azure_openai_with_deployment(m),
                #[cfg(feature = "bedrock")]
                ProviderType::Bedrock => Self::create_bedrock_with_model(m),
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
            ConfigProviderType::Azure => {
                // Azure OpenAI: use the env-auto factory (CONTENTGEN or standard vars)
                Self::create_azure_openai()
            }
            ConfigProviderType::Anthropic => Self::create_anthropic_from_config(config, model_name),
            ConfigProviderType::OpenRouter => {
                Self::create_openrouter_from_config(config, model_name)
            }
            ConfigProviderType::Mistral => Self::create_mistral_from_config(config, model_name),
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

    /// Create a generic OpenAI-compatible provider from environment variables.
    ///
    /// Expected environment variables:
    /// - `OPENAI_COMPATIBLE_BASE_URL` (required)
    /// - `OPENAI_COMPATIBLE_MODEL` (optional, default: `default`)
    /// - `OPENAI_COMPATIBLE_EMBEDDING_MODEL` (optional)
    /// - `OPENAI_COMPATIBLE_API_KEY` (optional; omitted for local gateways)
    fn create_openai_compatible_from_env(
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        Self::create_openai_compatible_from_env_with_model(
            &std::env::var("OPENAI_COMPATIBLE_MODEL").unwrap_or_else(|_| "default".to_string()),
        )
    }

    /// Create a generic OpenAI-compatible provider from environment with an explicit model.
    fn create_openai_compatible_from_env_with_model(
        model: &str,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        let base_url = std::env::var("OPENAI_COMPATIBLE_BASE_URL").map_err(|_| {
            LlmError::ConfigError(
                "OPENAI_COMPATIBLE_BASE_URL not set for OpenAI-compatible provider".to_string(),
            )
        })?;

        let mut config = ProviderConfig {
            name: "openai-compatible".to_string(),
            display_name: "OpenAI Compatible".to_string(),
            provider_type: ConfigProviderType::OpenAICompatible,
            base_url: Some(base_url),
            default_llm_model: Some(model.to_string()),
            default_embedding_model: std::env::var("OPENAI_COMPATIBLE_EMBEDDING_MODEL").ok(),
            ..Default::default()
        };

        if let Ok(api_key) = std::env::var("OPENAI_COMPATIBLE_API_KEY") {
            if !api_key.is_empty() {
                config.api_key = Some(api_key);
            }
        }

        Self::create_openai_compatible_with_model(&config, Some(model))
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
        // Get API key from environment.
        // WHY: ANTHROPIC_AUTH_TOKEN is a legitimate fallback for Anthropic-
        // compatible endpoints and must still work when ANTHROPIC_API_KEY is
        // present but intentionally empty.
        let api_key_var = config.api_key_env.as_deref().unwrap_or("ANTHROPIC_API_KEY");
        let api_key = if matches!(api_key_var, "ANTHROPIC_API_KEY" | "ANTHROPIC_AUTH_TOKEN") {
            AnthropicProvider::resolve_api_key_from_env()?
        } else {
            let value = std::env::var(api_key_var).map_err(|_| {
                LlmError::ConfigError(format!("{} not set for Anthropic provider", api_key_var))
            })?;
            let trimmed = value.trim();
            if trimmed.is_empty() {
                return Err(LlmError::ConfigError(format!("{} is empty", api_key_var)));
            }
            trimmed.to_string()
        };

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

    /// Create Google AI Studio Gemini provider from environment.
    ///
    /// Uses GeminiProvider::from_env() which reads:
    /// - GEMINI_API_KEY or GOOGLE_API_KEY (required)
    /// - GEMINI_MODEL (optional, default: gemini-2.5-flash)
    ///
    /// Endpoint: generativelanguage.googleapis.com (free tier / pay-as-you-go).
    /// For Vertex AI (aiplatform.googleapis.com) use create_vertex_ai() instead.
    fn create_gemini() -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        use crate::GeminiProvider;

        let provider = GeminiProvider::from_env()?;
        let llm_provider: Arc<dyn LLMProvider> = Arc::new(provider);
        let embedding: Arc<dyn EmbeddingProvider> = Arc::new(GeminiProvider::from_env()?);
        Ok((llm_provider, embedding))
    }

    /// Create Google Cloud Vertex AI Gemini provider from environment.
    ///
    /// Uses GeminiProvider::from_env_vertex_ai() which reads:
    /// - GOOGLE_CLOUD_PROJECT (required)
    /// - GOOGLE_CLOUD_REGION (optional, default: us-central1)
    /// - GOOGLE_ACCESS_TOKEN or gcloud auth application-default login
    ///
    /// Endpoint: {region}-aiplatform.googleapis.com (paid, high-quota).
    /// WHY separate from create_gemini(): these are distinct services with
    /// different auth, billing, quota systems and endpoints. Using GEMINI_API_KEY
    /// when the caller requested Vertex AI is a silent mis-routing bug.
    fn create_vertex_ai() -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        use crate::GeminiProvider;

        let provider = GeminiProvider::from_env_vertex_ai()?;
        let llm_provider: Arc<dyn LLMProvider> = Arc::new(provider);
        let embedding: Arc<dyn EmbeddingProvider> = Arc::new(GeminiProvider::from_env_vertex_ai()?);
        Ok((llm_provider, embedding))
    }

    /// Create Vertex AI Gemini provider with a specific model.
    fn create_vertex_ai_with_model(
        model: &str,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        use crate::GeminiProvider;

        // Strip vertexai: prefix if the caller still passes it (defensive)
        let actual = model.strip_prefix("vertexai:").unwrap_or(model);
        let provider = Arc::new(GeminiProvider::from_env_vertex_ai()?.with_model(actual));
        let embedding: Arc<dyn EmbeddingProvider> = Arc::new(GeminiProvider::from_env_vertex_ai()?);
        Ok((provider, embedding))
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
        let api_key = AnthropicProvider::resolve_api_key_from_env()?;

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

            // GeminiProvider implements EmbeddingProvider natively
            let embedding: Arc<dyn EmbeddingProvider> =
                Arc::new(GeminiProvider::from_env_vertex_ai()?);

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

        // GeminiProvider implements EmbeddingProvider natively
        let embedding: Arc<dyn EmbeddingProvider> = Arc::new(GeminiProvider::new(&api_key));

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
    /// - `VSCODE_COPILOT_MODEL`: Model name (default: auto; also accepts `copilot/gpt-4.1` and `copilot/auto` aliases)
    /// - `VSCODE_COPILOT_ACCOUNT_TYPE`: Account type (individual/business/enterprise)
    /// - `VSCODE_COPILOT_EMBEDDING_MODEL`: Embedding model (default: text-embedding-3-small)
    ///
    /// Note: Direct mode is now the default. No external proxy required.
    /// For legacy proxy mode, set `VSCODE_COPILOT_DIRECT=false`.
    ///
    /// VsCodeCopilotProvider now implements both LLMProvider and EmbeddingProvider,
    /// using the Copilot API's /embeddings endpoint for text embeddings.
    fn create_vscode_copilot() -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        let model = std::env::var("VSCODE_COPILOT_MODEL").unwrap_or_else(|_| "auto".to_string());

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

    // -----------------------------------------------------------------------
    // Azure OpenAI provider factory methods
    // -----------------------------------------------------------------------

    /// Create Azure OpenAI provider from environment variables.
    ///
    /// Tries `AZURE_OPENAI_CONTENTGEN_*` first (common enterprise naming),
    /// then falls back to `AZURE_OPENAI_*` standard variables.
    ///
    /// Environment variables (CONTENTGEN variant):
    /// - `AZURE_OPENAI_CONTENTGEN_API_ENDPOINT`     → endpoint URL
    /// - `AZURE_OPENAI_CONTENTGEN_API_KEY`          → API key
    /// - `AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT` → deployment name
    /// - `AZURE_OPENAI_CONTENTGEN_API_VERSION`      → (optional)
    ///
    /// Environment variables (standard variant):
    /// - `AZURE_OPENAI_ENDPOINT`                    → endpoint URL
    /// - `AZURE_OPENAI_API_KEY`                     → API key
    /// - `AZURE_OPENAI_DEPLOYMENT_NAME`             → deployment name
    /// - `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`   → (optional)
    /// - `AZURE_OPENAI_API_VERSION`                 → (optional)
    pub fn create_azure_openai() -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        let provider = Arc::new(AzureOpenAIProvider::from_env_auto()?);
        Ok((provider.clone(), provider))
    }

    /// Create Azure OpenAI provider with a specific deployment name (model) override.
    fn create_azure_openai_with_deployment(
        deployment: &str,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        // Build from env then override the deployment
        let provider = Arc::new(AzureOpenAIProvider::from_env_auto()?.with_deployment(deployment));
        Ok((provider.clone(), provider))
    }

    /// Create AWS Bedrock provider from environment.
    ///
    /// Uses the standard AWS credential chain (env vars, profiles, IAM roles).
    /// The async `from_env()` is executed via the current Tokio runtime.
    ///
    /// # Environment Variables
    ///
    /// - `AWS_BEDROCK_MODEL`: Model ID (default: `anthropic.claude-3-5-sonnet-20241022-v2:0`)
    /// - `AWS_REGION` / `AWS_DEFAULT_REGION`: AWS region (default: `us-east-1`)
    /// - Standard AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, etc.)
    #[cfg(feature = "bedrock")]
    fn create_bedrock() -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        use crate::providers::bedrock::BedrockProvider;

        let rt = tokio::runtime::Handle::try_current().map_err(|_| {
            LlmError::ConfigError(
                "Bedrock provider requires a Tokio runtime (use #[tokio::main] or Runtime::new())"
                    .to_string(),
            )
        })?;
        let provider = Arc::new(tokio::task::block_in_place(|| {
            rt.block_on(BedrockProvider::from_env())
        })?);

        // Bedrock supports embeddings natively via invoke_model API
        let embedding: Arc<dyn EmbeddingProvider> = provider.clone();

        Ok((provider, embedding))
    }

    /// Create Bedrock provider with a specific model override.
    #[cfg(feature = "bedrock")]
    fn create_bedrock_with_model(
        model: &str,
    ) -> Result<(Arc<dyn LLMProvider>, Arc<dyn EmbeddingProvider>)> {
        use crate::providers::bedrock::BedrockProvider;

        let rt = tokio::runtime::Handle::try_current().map_err(|_| {
            LlmError::ConfigError("Bedrock provider requires a Tokio runtime".to_string())
        })?;
        let provider = Arc::new(
            tokio::task::block_in_place(|| rt.block_on(BedrockProvider::from_env()))?
                .with_model(model),
        );

        // Bedrock supports embeddings natively via invoke_model API
        let embedding: Arc<dyn EmbeddingProvider> = provider.clone();

        Ok((provider, embedding))
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
        if provider_name.eq_ignore_ascii_case("jina") {
            let api_key = std::env::var("JINA_API_KEY").map_err(|_| {
                LlmError::ConfigError("JINA_API_KEY is required for Jina embeddings".to_string())
            })?;
            let base_url = std::env::var("JINA_BASE_URL")
                .unwrap_or_else(|_| "https://api.jina.ai".to_string());
            let provider = JinaProvider::builder()
                .api_key(api_key)
                .base_url(base_url)
                .embedding_model(model)
                .build()?;
            return Ok(Arc::new(provider));
        }

        let provider_type = ProviderType::from_str(provider_name).ok_or_else(|| {
            LlmError::ConfigError(format!(
                "Unknown embedding provider: {}. Valid: openai, anthropic, gemini, vertexai, openrouter, xai, huggingface, openai-compatible, ollama, lmstudio, vscode-copilot, mistral, azure, bedrock, jina, mock",
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
            ProviderType::OpenAICompatible => {
                let (_, embedding) = Self::create_openai_compatible_from_env_with_model(model)?;
                Ok(embedding)
            }
            ProviderType::HuggingFace => {
                // HuggingFace requires separate endpoint for embeddings, fall back to mock
                warn!("HuggingFace LLM provider doesn't support embeddings, using mock provider");
                Ok(Arc::new(MockProvider::new()))
            }
            ProviderType::Gemini => {
                // Google AI Studio embedding (generativelanguage.googleapis.com).
                // If credentials are unavailable fall back to mock.
                match GeminiProvider::from_env() {
                    Ok(provider) => Ok(Arc::new(provider.with_embedding_model(model))),
                    Err(e) => {
                        warn!(
                            "Gemini credentials unavailable ({}), falling back to mock embedding provider",
                            e
                        );
                        Ok(Arc::new(MockProvider::new()))
                    }
                }
            }
            ProviderType::VertexAI => {
                // Vertex AI embedding (aiplatform.googleapis.com).
                match GeminiProvider::from_env_vertex_ai() {
                    Ok(provider) => Ok(Arc::new(provider.with_embedding_model(model))),
                    Err(e) => {
                        warn!(
                            "Vertex AI credentials unavailable ({}), falling back to mock embedding provider",
                            e
                        );
                        Ok(Arc::new(MockProvider::new()))
                    }
                }
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
            ProviderType::AzureOpenAI => {
                // Azure OpenAI supports embeddings via a separate deployment
                let provider =
                    AzureOpenAIProvider::from_env_auto()?.with_embedding_deployment(model);
                Ok(Arc::new(provider))
            }
            #[cfg(feature = "bedrock")]
            ProviderType::Bedrock => {
                let (_, embedding) = Self::create_bedrock_with_model(model)?;
                Ok(embedding)
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
                "Unknown LLM provider: {}. Valid: openai, anthropic, gemini, vertexai, openrouter, xai, huggingface, openai-compatible, ollama, lmstudio, vscode-copilot, mistral, azure, bedrock, mock",
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
                // Reuse the shared constructor so the explicit provider/model path
                // stays behaviorally identical to from_env() and config-based setup.
                // WHY: Anthropic-compatible gateways such as POE require BOTH the
                // blank-value credential fallback and ANTHROPIC_BASE_URL support.
                let (provider, _) = Self::create_anthropic_with_model(model)?;
                Ok(provider)
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
            ProviderType::OpenAICompatible => {
                let (provider, _) = Self::create_openai_compatible_from_env_with_model(model)?;
                Ok(provider)
            }
            ProviderType::HuggingFace => {
                // OODA-80: HuggingFace Hub provider with specific model
                let provider = HuggingFaceProvider::from_env()?.with_model(model);
                Ok(Arc::new(provider))
            }
            ProviderType::Gemini => {
                // OODA-73/95: Google AI Studio provider.
                // vertexai: prefix kept for backward compat — callers using
                // create_llm_provider("gemini", "vertexai:model") still work,
                // but the preferred approach is ProviderType::VertexAI.
                if model.starts_with("vertexai:") {
                    let actual_model = model.strip_prefix("vertexai:").unwrap_or(model);
                    let provider = GeminiProvider::from_env_vertex_ai()?.with_model(actual_model);
                    Ok(Arc::new(provider))
                } else {
                    let provider = GeminiProvider::from_env()?.with_model(model);
                    Ok(Arc::new(provider))
                }
            }
            ProviderType::VertexAI => {
                // Vertex AI (aiplatform.googleapis.com) — always uses ADC/gcloud,
                // never picks up GEMINI_API_KEY.
                let actual_model = model.strip_prefix("vertexai:").unwrap_or(model);
                let provider = GeminiProvider::from_env_vertex_ai()?.with_model(actual_model);
                Ok(Arc::new(provider))
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
            ProviderType::AzureOpenAI => {
                // Azure OpenAI provider with specific deployment (model)
                let provider = AzureOpenAIProvider::from_env_auto()?.with_deployment(model);
                Ok(Arc::new(provider))
            }
            #[cfg(feature = "bedrock")]
            ProviderType::Bedrock => {
                // Bedrock provider with specific model
                use crate::providers::bedrock::BedrockProvider;
                let handle = tokio::runtime::Handle::try_current().map_err(|_| {
                    LlmError::ConfigError("Bedrock provider requires a Tokio runtime".to_string())
                })?;
                let provider =
                    tokio::task::block_in_place(|| handle.block_on(BedrockProvider::from_env()))
                        .map_err(|e| {
                            LlmError::ConfigError(format!(
                                "Failed to initialize Bedrock provider: {e}"
                            ))
                        })?;
                let provider = provider.with_model(model);
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

        // Google AI Studio (GEMINI_API_KEY)
        assert_eq!(ProviderType::from_str("gemini"), Some(ProviderType::Gemini));
        assert_eq!(ProviderType::from_str("google"), Some(ProviderType::Gemini));
        // Vertex AI (ADC / aiplatform.googleapis.com) — now a distinct variant
        assert_eq!(
            ProviderType::from_str("vertex"),
            Some(ProviderType::VertexAI)
        );
        assert_eq!(
            ProviderType::from_str("vertexai"),
            Some(ProviderType::VertexAI)
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

        // Azure OpenAI parsing
        assert_eq!(
            ProviderType::from_str("azure"),
            Some(ProviderType::AzureOpenAI)
        );
        assert_eq!(
            ProviderType::from_str("azure-openai"),
            Some(ProviderType::AzureOpenAI)
        );
        assert_eq!(
            ProviderType::from_str("azure_openai"),
            Some(ProviderType::AzureOpenAI)
        );
        assert_eq!(
            ProviderType::from_str("AZURE"),
            Some(ProviderType::AzureOpenAI)
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
        // Clear all provider environment variables (including Azure CONTENTGEN variants
        // that other serial tests may have set — dotenvy reloads them if remove_var'd
        // by a prior test and that test subsequently called a provider from_env()).
        std::env::remove_var("EDGEQUAKE_LLM_PROVIDER");
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("XAI_API_KEY");
        std::env::remove_var("GOOGLE_API_KEY");
        std::env::remove_var("GEMINI_API_KEY");
        std::env::remove_var("OPENROUTER_API_KEY");
        std::env::remove_var("ANTHROPIC_API_KEY");
        // Azure standard vars
        std::env::remove_var("AZURE_OPENAI_API_KEY");
        std::env::remove_var("AZURE_OPENAI_ENDPOINT");
        std::env::remove_var("AZURE_OPENAI_DEPLOYMENT_NAME");
        // Azure CONTENTGEN vars (checked first in factory::from_env)
        std::env::remove_var("AZURE_OPENAI_CONTENTGEN_API_KEY");
        std::env::remove_var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT");
        std::env::remove_var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT");
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
    fn test_anthropic_auto_detection_uses_auth_token_when_api_key_empty() {
        std::env::remove_var("EDGEQUAKE_LLM_PROVIDER");
        std::env::remove_var("OLLAMA_HOST");
        std::env::remove_var("OLLAMA_MODEL");
        std::env::remove_var("LMSTUDIO_HOST");
        std::env::remove_var("LMSTUDIO_MODEL");
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("XAI_API_KEY");
        std::env::remove_var("GOOGLE_API_KEY");
        std::env::remove_var("GEMINI_API_KEY");
        std::env::remove_var("OPENROUTER_API_KEY");
        std::env::remove_var("AZURE_OPENAI_API_KEY");

        std::env::set_var("ANTHROPIC_API_KEY", "");
        std::env::set_var("ANTHROPIC_AUTH_TOKEN", "poe-token");

        let (llm, _) = ProviderFactory::from_env().unwrap();
        assert_eq!(llm.name(), "anthropic");

        std::env::remove_var("ANTHROPIC_API_KEY");
        std::env::remove_var("ANTHROPIC_AUTH_TOKEN");
    }

    #[test]
    #[serial]
    fn test_create_anthropic_from_config_uses_auth_token_fallback() {
        std::env::remove_var("ANTHROPIC_API_KEY");
        std::env::remove_var("ANTHROPIC_AUTH_TOKEN");
        std::env::set_var("ANTHROPIC_API_KEY", "");
        std::env::set_var("ANTHROPIC_AUTH_TOKEN", "poe-token");

        let config = ProviderConfig {
            provider_type: ConfigProviderType::Anthropic,
            api_key_env: Some("ANTHROPIC_API_KEY".to_string()),
            base_url: Some("https://api.poe.com".to_string()),
            default_llm_model: Some("claude-sonnet-4-6".to_string()),
            ..Default::default()
        };

        let (llm, _) = ProviderFactory::from_config(&config).unwrap();
        assert_eq!(llm.name(), "anthropic");
        assert_eq!(llm.model(), "claude-sonnet-4-6");

        std::env::remove_var("ANTHROPIC_API_KEY");
        std::env::remove_var("ANTHROPIC_AUTH_TOKEN");
    }

    #[test]
    #[serial]
    fn test_create_llm_provider_anthropic_uses_auth_token_fallback() {
        std::env::remove_var("ANTHROPIC_API_KEY");
        std::env::remove_var("ANTHROPIC_AUTH_TOKEN");
        std::env::set_var("ANTHROPIC_API_KEY", "");
        std::env::set_var("ANTHROPIC_AUTH_TOKEN", "poe-token");

        let provider = ProviderFactory::create_llm_provider("anthropic", "claude-haiku-4-5")
            .expect("anthropic LLM provider should use auth token fallback");
        assert_eq!(provider.name(), "anthropic");
        assert_eq!(provider.model(), "claude-haiku-4-5");

        std::env::remove_var("ANTHROPIC_API_KEY");
        std::env::remove_var("ANTHROPIC_AUTH_TOKEN");
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
    #[serial]
    fn test_from_config_azure_no_creds() {
        use crate::model_config::{ProviderConfig, ProviderType as ConfigProviderType};
        // Set Azure env vars to empty strings so dotenvy won't override them
        // (dotenvy only sets vars that are not already present in the environment)
        std::env::set_var("AZURE_OPENAI_CONTENTGEN_API_KEY", "");
        std::env::set_var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT", "");
        std::env::set_var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT", "");
        std::env::set_var("AZURE_OPENAI_API_KEY", "");
        std::env::set_var("AZURE_OPENAI_ENDPOINT", "");
        std::env::set_var("AZURE_OPENAI_DEPLOYMENT_NAME", "");
        let config = ProviderConfig {
            provider_type: ConfigProviderType::Azure,
            ..ProviderConfig::default()
        };
        let result = ProviderFactory::from_config(&config);
        // Restore env vars
        std::env::remove_var("AZURE_OPENAI_CONTENTGEN_API_KEY");
        std::env::remove_var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT");
        std::env::remove_var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT");
        std::env::remove_var("AZURE_OPENAI_API_KEY");
        std::env::remove_var("AZURE_OPENAI_ENDPOINT");
        std::env::remove_var("AZURE_OPENAI_DEPLOYMENT_NAME");
        // Should fail with a config error (empty credentials)
        assert!(
            result.is_err(),
            "Expected error when Azure credentials are not set"
        );
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

    #[test]
    fn test_vscode_copilot_default_model_is_auto() {
        std::env::remove_var("VSCODE_COPILOT_MODEL");
        let (llm, embedding) = ProviderFactory::create(ProviderType::VsCodeCopilot).unwrap();
        assert_eq!(llm.model(), "auto");
        assert_eq!(embedding.name(), "vscode-copilot");
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
        // GeminiProvider supports embeddings natively when GEMINI_API_KEY is set.
        // Without GEMINI_API_KEY (and no VertexAI creds), it falls back to mock.
        let provider =
            ProviderFactory::create_embedding_provider("gemini", "any-model", 768).unwrap();
        let name = provider.name();
        assert!(
            name == "gemini" || name == "mock",
            "Expected 'gemini' (with API key) or 'mock' (without), got '{}'",
            name
        );
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

    // ── Azure OpenAI factory unit tests ──────────────────────────────────────

    #[test]
    #[serial]
    fn test_provider_type_parsing_azure() {
        assert_eq!(
            ProviderType::from_str("azure"),
            Some(ProviderType::AzureOpenAI)
        );
        assert_eq!(
            ProviderType::from_str("azure-openai"),
            Some(ProviderType::AzureOpenAI)
        );
        assert_eq!(
            ProviderType::from_str("AZURE"),
            Some(ProviderType::AzureOpenAI)
        );
    }

    #[test]
    #[serial]
    fn test_create_azure_openai_fails_without_env() {
        // Set Azure env vars to empty strings so dotenvy won't override them
        // (dotenvy only sets vars that are not already present in the environment)
        std::env::set_var("AZURE_OPENAI_CONTENTGEN_API_KEY", "");
        std::env::set_var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT", "");
        std::env::set_var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT", "");
        std::env::set_var("AZURE_OPENAI_API_KEY", "");
        std::env::set_var("AZURE_OPENAI_ENDPOINT", "");
        std::env::set_var("AZURE_OPENAI_DEPLOYMENT_NAME", "");

        let result = ProviderFactory::create(ProviderType::AzureOpenAI);
        // Restore env vars
        std::env::remove_var("AZURE_OPENAI_CONTENTGEN_API_KEY");
        std::env::remove_var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT");
        std::env::remove_var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT");
        std::env::remove_var("AZURE_OPENAI_API_KEY");
        std::env::remove_var("AZURE_OPENAI_ENDPOINT");
        std::env::remove_var("AZURE_OPENAI_DEPLOYMENT_NAME");
        assert!(
            result.is_err(),
            "Azure provider should fail when env vars are empty"
        );
    }

    #[test]
    #[serial]
    fn test_from_env_auto_detects_azure_with_contentgen_vars() {
        // Remove all other providers to isolate Azure detection
        std::env::remove_var("EDGEQUAKE_LLM_PROVIDER");
        std::env::remove_var("OLLAMA_HOST");
        std::env::remove_var("OLLAMA_MODEL");
        std::env::remove_var("LMSTUDIO_HOST");
        std::env::remove_var("LMSTUDIO_MODEL");
        std::env::remove_var("ANTHROPIC_API_KEY");
        std::env::remove_var("GEMINI_API_KEY");
        std::env::remove_var("GOOGLE_API_KEY");
        std::env::remove_var("MISTRAL_API_KEY");

        // Set Azure CONTENTGEN vars
        std::env::set_var("AZURE_OPENAI_CONTENTGEN_API_KEY", "test-azure-key");
        std::env::set_var(
            "AZURE_OPENAI_CONTENTGEN_API_ENDPOINT",
            "https://test.openai.azure.com",
        );
        std::env::set_var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT", "gpt-4o");

        let result = ProviderFactory::from_env();

        // Cleanup before asserting so env is always cleaned
        std::env::remove_var("AZURE_OPENAI_CONTENTGEN_API_KEY");
        std::env::remove_var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT");
        std::env::remove_var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT");

        let (llm, _) = result.expect("Should detect Azure from CONTENTGEN vars");
        assert_eq!(llm.name(), "azure-openai");
    }

    #[test]
    #[serial]
    fn test_from_env_auto_detects_azure_with_standard_vars() {
        // Remove all other providers to isolate Azure detection
        std::env::remove_var("EDGEQUAKE_LLM_PROVIDER");
        std::env::remove_var("OLLAMA_HOST");
        std::env::remove_var("OLLAMA_MODEL");
        std::env::remove_var("LMSTUDIO_HOST");
        std::env::remove_var("LMSTUDIO_MODEL");
        std::env::remove_var("ANTHROPIC_API_KEY");
        std::env::remove_var("GEMINI_API_KEY");
        std::env::remove_var("GOOGLE_API_KEY");
        std::env::remove_var("MISTRAL_API_KEY");
        std::env::remove_var("AZURE_OPENAI_CONTENTGEN_API_KEY");

        // Set Azure standard vars
        std::env::set_var("AZURE_OPENAI_API_KEY", "test-azure-key");
        std::env::set_var("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com");
        std::env::set_var("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o");

        let result = ProviderFactory::from_env();

        // Cleanup
        std::env::remove_var("AZURE_OPENAI_API_KEY");
        std::env::remove_var("AZURE_OPENAI_ENDPOINT");
        std::env::remove_var("AZURE_OPENAI_DEPLOYMENT_NAME");

        let (llm, _) = result.expect("Should detect Azure from standard vars");
        assert_eq!(llm.name(), "azure-openai");
    }

    #[test]
    #[serial]
    fn test_explicit_azure_provider_selection() {
        // Remove all other providers
        std::env::remove_var("EDGEQUAKE_LLM_PROVIDER");
        std::env::remove_var("OLLAMA_HOST");
        std::env::remove_var("LMSTUDIO_HOST");
        std::env::remove_var("OPENAI_API_KEY");

        // Set Azure CONTENTGEN vars
        std::env::set_var("AZURE_OPENAI_CONTENTGEN_API_KEY", "test-key");
        std::env::set_var(
            "AZURE_OPENAI_CONTENTGEN_API_ENDPOINT",
            "https://test.openai.azure.com",
        );
        std::env::set_var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT", "gpt-4.1-mini");

        std::env::set_var("EDGEQUAKE_LLM_PROVIDER", "azure");
        let result = ProviderFactory::from_env();

        // Cleanup
        std::env::remove_var("EDGEQUAKE_LLM_PROVIDER");
        std::env::remove_var("AZURE_OPENAI_CONTENTGEN_API_KEY");
        std::env::remove_var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT");
        std::env::remove_var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT");

        let (llm, _) = result.expect("Explicit azure provider selection should succeed");
        assert_eq!(llm.name(), "azure-openai");
        assert_eq!(llm.model(), "gpt-4.1-mini");
    }

    #[test]
    #[serial]
    fn test_create_with_model_azure() {
        std::env::set_var("AZURE_OPENAI_CONTENTGEN_API_KEY", "test-key");
        std::env::set_var(
            "AZURE_OPENAI_CONTENTGEN_API_ENDPOINT",
            "https://test.openai.azure.com",
        );
        std::env::set_var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT", "gpt-4o");

        let result = ProviderFactory::create_with_model(
            ProviderType::AzureOpenAI,
            Some("my-custom-deployment"),
        );

        // Cleanup
        std::env::remove_var("AZURE_OPENAI_CONTENTGEN_API_KEY");
        std::env::remove_var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT");
        std::env::remove_var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT");

        let (llm, _) = result.expect("Azure create_with_model should succeed");
        assert_eq!(llm.name(), "azure-openai");
        assert_eq!(llm.model(), "my-custom-deployment");
    }

    // ── VertexAI routing discriminating tests ────────────────────────────────
    //
    // WHY: prove that create_llm_provider("vertexai", ...) / ("vertex", ...)
    // routes through from_env_vertex_ai() which checks GOOGLE_CLOUD_PROJECT,
    // NOT through from_env() which checks GEMINI_API_KEY.
    // This is the behavioral guarantee of the ProviderType::VertexAI fix.

    #[test]
    fn test_vertex_provider_type_is_distinct_from_gemini() {
        // "vertex" and "vertexai" must map to VertexAI, NOT Gemini
        assert_eq!(
            ProviderType::from_str("vertex"),
            Some(ProviderType::VertexAI)
        );
        assert_eq!(
            ProviderType::from_str("vertexai"),
            Some(ProviderType::VertexAI)
        );
        assert_eq!(
            ProviderType::from_str("VERTEXAI"),
            Some(ProviderType::VertexAI)
        );
        // Gemini variants must still map to Gemini
        assert_eq!(ProviderType::from_str("gemini"), Some(ProviderType::Gemini));
        assert_eq!(ProviderType::from_str("google"), Some(ProviderType::Gemini));
        // The two variants are distinct
        assert_ne!(ProviderType::VertexAI, ProviderType::Gemini);
    }

    #[test]
    #[serial]
    fn test_create_llm_provider_vertexai_uses_vertex_auth() {
        // Without GOOGLE_CLOUD_PROJECT the error must mention GOOGLE_CLOUD_PROJECT,
        // NOT GEMINI_API_KEY — proving the VertexAI arm was taken.
        std::env::remove_var("GOOGLE_CLOUD_PROJECT");
        std::env::remove_var("GOOGLE_ACCESS_TOKEN");

        let result = ProviderFactory::create_llm_provider("vertexai", "gemini-2.5-flash");
        let msg = result
            .err()
            .expect("Expected error without Vertex AI credentials")
            .to_string();
        assert!(
            msg.contains("GOOGLE_CLOUD_PROJECT"),
            "VertexAI error must mention GOOGLE_CLOUD_PROJECT, got: {msg}"
        );
    }

    #[test]
    #[serial]
    fn test_create_llm_provider_vertex_alias_same_as_vertexai() {
        // "vertex" is an alias for "vertexai" — both must hit the same VertexAI arm
        std::env::remove_var("GOOGLE_CLOUD_PROJECT");
        std::env::remove_var("GOOGLE_ACCESS_TOKEN");

        let result = ProviderFactory::create_llm_provider("vertex", "gemini-2.5-flash");
        let msg = result
            .err()
            .expect("Expected error from vertex alias")
            .to_string();
        assert!(
            msg.contains("GOOGLE_CLOUD_PROJECT"),
            "\"vertex\" alias must route to VertexAI arm: {msg}"
        );
    }

    #[test]
    #[serial]
    fn test_vertexai_prefix_stripped_on_model_name() {
        // "vertexai:gemini-2.5-flash" passed as model must have prefix stripped;
        // auth check must still mention GOOGLE_CLOUD_PROJECT (not a parsing error).
        std::env::remove_var("GOOGLE_CLOUD_PROJECT");
        std::env::remove_var("GOOGLE_ACCESS_TOKEN");

        let result = ProviderFactory::create_llm_provider("vertexai", "vertexai:gemini-2.5-flash");
        let msg = result
            .err()
            .expect("Expected error for prefixed model")
            .to_string();
        assert!(
            msg.contains("GOOGLE_CLOUD_PROJECT"),
            "vertexai: prefix must be stripped before auth check: {msg}"
        );
    }

    #[test]
    #[serial]
    fn test_vertexai_ignores_gemini_api_key() {
        // This is the key discriminating test: VertexAI must NOT be satisfied by
        // GEMINI_API_KEY.  Even when GEMINI_API_KEY is present and valid-looking,
        // the VertexAI arm routes through from_env_vertex_ai() which checks
        // GOOGLE_CLOUD_PROJECT — not the Gemini AI Studio API key.
        std::env::set_var("GEMINI_API_KEY", "fake-key-should-not-satisfy-vertexai");
        std::env::remove_var("GOOGLE_CLOUD_PROJECT");
        std::env::remove_var("GOOGLE_ACCESS_TOKEN");

        let result = ProviderFactory::create_llm_provider("vertexai", "gemini-2.5-flash");
        std::env::remove_var("GEMINI_API_KEY");

        let msg = result
            .err()
            .expect("VertexAI must fail: GEMINI_API_KEY alone is not sufficient")
            .to_string();
        assert!(
            msg.contains("GOOGLE_CLOUD_PROJECT"),
            "VertexAI must require GOOGLE_CLOUD_PROJECT even when GEMINI_API_KEY is set: {msg}"
        );
    }
}
