//! Model Configuration Module
//!
//! This module provides TOML-based configuration for LLM and embedding models,
//! including model cards with capabilities (vision, context length, costs, etc.).
//!
//! @implements SPEC-032: Ollama/LM Studio provider support - Model cards configuration
//! @iteration OODA Loop #51-55 - TOML Config Schema Design
//!
//! # Overview
//!
//! The configuration file (`models.toml`) defines:
//! - Available LLM providers and models
//! - Embedding providers and models  
//! - Model capabilities (vision, max tokens, context length)
//! - Cost information (per 1K tokens)
//! - Default selections for LLM and embedding
//!
//! # Configuration File Location
//!
//! The config file is loaded from (in order of priority):
//! 1. `EDGEQUAKE_MODELS_CONFIG` environment variable
//! 2. `./models.toml` (current working directory)
//! 3. `~/.edgequake/models.toml` (user config)
//! 4. Built-in default configuration
//!
//! # Example Configuration
//!
//! ```toml
//! [defaults]
//! llm_provider = "openai"
//! llm_model = "gpt-4o-mini"
//! embedding_provider = "openai"
//! embedding_model = "text-embedding-3-small"
//!
//! [[providers]]
//! name = "openai"
//! display_name = "OpenAI"
//! type = "openai"
//! api_key_env = "OPENAI_API_KEY"
//!
//! [[providers.models]]
//! name = "gpt-4o"
//! display_name = "GPT-4 Omni"
//! type = "llm"
//! context_length = 128000
//! max_output_tokens = 16384
//! supports_vision = true
//! supports_function_calling = true
//! cost_per_1k_input = 0.0025
//! cost_per_1k_output = 0.01
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during model configuration loading.
#[derive(Error, Debug)]
pub enum ModelConfigError {
    /// Failed to read configuration file.
    #[error("Failed to read config file: {0}")]
    IoError(#[from] std::io::Error),

    /// Failed to parse TOML configuration.
    #[error("Failed to parse TOML config: {0}")]
    ParseError(String),

    /// Invalid configuration (missing required fields, invalid values).
    #[error("Invalid configuration: {0}")]
    ValidationError(String),

    /// Provider not found in configuration.
    #[error("Provider not found: {0}")]
    ProviderNotFound(String),

    /// Model not found in configuration.
    #[error("Model not found: {0}")]
    ModelNotFound(String),
}

// ============================================================================
// Model Types
// ============================================================================

/// Type of model (LLM for chat/completion, Embedding for vectors).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    /// Language model for chat/completion.
    #[default]
    Llm,
    /// Embedding model for vector generation.
    Embedding,
    /// Multi-modal model supporting both.
    Multimodal,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Llm => write!(f, "llm"),
            ModelType::Embedding => write!(f, "embedding"),
            ModelType::Multimodal => write!(f, "multimodal"),
        }
    }
}

/// Provider type for API compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    /// OpenAI API.
    #[default]
    OpenAI,
    /// Ollama local server.
    Ollama,
    /// LM Studio local server.
    LMStudio,
    /// Azure OpenAI.
    Azure,
    /// Anthropic Claude.
    Anthropic,
    /// OpenRouter (200+ models).
    OpenRouter,
    /// Generic OpenAI-compatible API.
    OpenAICompatible,
    /// Mock provider for testing.
    Mock,
    /// Mistral AI (La Plateforme).
    Mistral,
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderType::OpenAI => write!(f, "openai"),
            ProviderType::Ollama => write!(f, "ollama"),
            ProviderType::LMStudio => write!(f, "lmstudio"),
            ProviderType::Azure => write!(f, "azure"),
            ProviderType::Anthropic => write!(f, "anthropic"),
            ProviderType::OpenRouter => write!(f, "openrouter"),
            ProviderType::OpenAICompatible => write!(f, "openai_compatible"),
            ProviderType::Mock => write!(f, "mock"),
            ProviderType::Mistral => write!(f, "mistral"),
        }
    }
}

// ============================================================================
// Model Capabilities
// ============================================================================

/// Capabilities of a specific model.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelCapabilities {
    /// Maximum context length (input + output tokens).
    #[serde(default)]
    pub context_length: usize,

    /// Maximum output tokens the model can generate.
    #[serde(default)]
    pub max_output_tokens: usize,

    /// Whether the model supports vision/image input.
    #[serde(default)]
    pub supports_vision: bool,

    /// Whether the model supports function/tool calling.
    #[serde(default)]
    pub supports_function_calling: bool,

    /// Whether the model supports structured JSON output.
    #[serde(default)]
    pub supports_json_mode: bool,

    /// Whether the model supports streaming responses.
    #[serde(default = "default_true")]
    pub supports_streaming: bool,

    /// Whether the model supports system messages.
    #[serde(default = "default_true")]
    pub supports_system_message: bool,

    /// Embedding dimension (only for embedding models).
    #[serde(default)]
    pub embedding_dimension: usize,

    /// Maximum tokens for embedding input.
    #[serde(default)]
    pub max_embedding_tokens: usize,

    /// OODA-200: Whether the model supports thinking/chain-of-thought mode.
    #[serde(default)]
    pub supports_thinking: bool,

    /// OODA-200: Whether the model supports web search tool.
    #[serde(default)]
    pub supports_web_search: bool,

    /// OODA-200: Recommended temperature for this model (0.0-1.0).
    #[serde(default = "default_temperature")]
    pub default_temperature: f32,
}

fn default_temperature() -> f32 {
    1.0
}

fn default_true() -> bool {
    true
}

// ============================================================================
// Cost Information
// ============================================================================

/// Cost information for a model (per 1000 tokens).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelCost {
    /// Cost per 1000 input tokens (USD).
    #[serde(default)]
    pub input_per_1k: f64,

    /// Cost per 1000 output tokens (USD).
    #[serde(default)]
    pub output_per_1k: f64,

    /// Cost per 1000 embedding tokens (USD, for embedding models).
    #[serde(default)]
    pub embedding_per_1k: f64,

    /// Cost per image processed (USD, for vision models).
    #[serde(default)]
    pub image_per_unit: f64,

    /// Currency code (default: USD).
    #[serde(default = "default_currency")]
    pub currency: String,
}

fn default_currency() -> String {
    "USD".to_string()
}

// ============================================================================
// Model Card
// ============================================================================

/// Complete model card with all metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    /// Unique model identifier (e.g., "gpt-4o", "nomic-embed-text").
    pub name: String,

    /// Human-readable display name.
    pub display_name: String,

    /// Model type (LLM, Embedding, Multimodal).
    #[serde(default)]
    pub model_type: ModelType,

    /// Model capabilities.
    #[serde(default)]
    pub capabilities: ModelCapabilities,

    /// Cost information.
    #[serde(default)]
    pub cost: ModelCost,

    /// Optional description of the model.
    #[serde(default)]
    pub description: String,

    /// Release date or version.
    #[serde(default)]
    pub version: String,

    /// Whether the model is deprecated.
    #[serde(default)]
    pub deprecated: bool,

    /// Recommended replacement if deprecated.
    #[serde(default)]
    pub replacement: Option<String>,

    /// Tags for categorization (e.g., "recommended", "fast", "vision").
    #[serde(default)]
    pub tags: Vec<String>,

    /// Additional metadata as key-value pairs.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl Default for ModelCard {
    fn default() -> Self {
        Self {
            name: "unknown".to_string(),
            display_name: "Unknown Model".to_string(),
            model_type: ModelType::Llm,
            capabilities: ModelCapabilities::default(),
            cost: ModelCost::default(),
            description: String::new(),
            version: String::new(),
            deprecated: false,
            replacement: None,
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

// ============================================================================
// Provider Configuration
// ============================================================================

/// Configuration for a provider (OpenAI, Ollama, LM Studio, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Unique provider identifier (e.g., "openai", "ollama").
    pub name: String,

    /// Human-readable display name.
    pub display_name: String,

    /// Provider type for API compatibility.
    #[serde(rename = "type")]
    pub provider_type: ProviderType,

    /// Environment variable name for API key (if required).
    #[serde(default)]
    pub api_key_env: Option<String>,

    /// Base URL for the provider API.
    #[serde(default)]
    pub base_url: Option<String>,

    /// Environment variable for base URL override.
    #[serde(default)]
    pub base_url_env: Option<String>,

    /// Default model for LLM operations.
    #[serde(default)]
    pub default_llm_model: Option<String>,

    /// Default model for embedding operations.
    #[serde(default)]
    pub default_embedding_model: Option<String>,

    /// List of available models for this provider.
    #[serde(default)]
    pub models: Vec<ModelCard>,

    /// Whether this provider is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Priority for auto-selection (lower = higher priority).
    #[serde(default = "default_priority")]
    pub priority: u32,

    /// Description of the provider.
    #[serde(default)]
    pub description: String,

    /// Additional provider-specific settings.
    #[serde(default)]
    pub settings: HashMap<String, String>,

    /// OODA-200: Custom HTTP headers for API requests.
    /// Useful for providers that require additional headers like Accept-Language.
    #[serde(default)]
    pub headers: HashMap<String, String>,

    /// OODA-200: Request timeout in seconds (default: 120).
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,

    /// OODA-200: Whether this provider supports thinking/reasoning mode (e.g., Z.ai GLM-4.5).
    #[serde(default)]
    pub supports_thinking: bool,
}

fn default_timeout() -> u64 {
    120
}

fn default_priority() -> u32 {
    100
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            name: "unknown".to_string(),
            display_name: "Unknown Provider".to_string(),
            provider_type: ProviderType::OpenAI,
            api_key_env: None,
            base_url: None,
            base_url_env: None,
            default_llm_model: None,
            default_embedding_model: None,
            models: Vec::new(),
            enabled: true,
            priority: 100,
            description: String::new(),
            settings: HashMap::new(),
            headers: HashMap::new(),
            timeout_seconds: default_timeout(),
            supports_thinking: false,
        }
    }
}

// ============================================================================
// Default Configuration
// ============================================================================

/// Default provider and model selections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultsConfig {
    /// Default LLM provider name.
    #[serde(default = "default_llm_provider")]
    pub llm_provider: String,

    /// Default LLM model name.
    #[serde(default = "default_llm_model")]
    pub llm_model: String,

    /// Default embedding provider name.
    #[serde(default = "default_embedding_provider")]
    pub embedding_provider: String,

    /// Default embedding model name.
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,
}

fn default_llm_provider() -> String {
    "openai".to_string()
}

fn default_llm_model() -> String {
    "gpt-4o-mini".to_string()
}

fn default_embedding_provider() -> String {
    "openai".to_string()
}

fn default_embedding_model() -> String {
    "text-embedding-3-small".to_string()
}

impl Default for DefaultsConfig {
    fn default() -> Self {
        Self {
            llm_provider: default_llm_provider(),
            llm_model: default_llm_model(),
            embedding_provider: default_embedding_provider(),
            embedding_model: default_embedding_model(),
        }
    }
}

// ============================================================================
// Root Configuration
// ============================================================================

/// Root configuration structure for models.toml.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelsConfig {
    /// Default selections.
    #[serde(default)]
    pub defaults: DefaultsConfig,

    /// List of configured providers.
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
}

impl ModelsConfig {
    /// Load configuration from the default location.
    ///
    /// Searches in order:
    /// 1. `EDGEQUAKE_MODELS_CONFIG` environment variable
    /// 2. `./models.toml`
    /// 3. `~/.edgequake/models.toml`
    /// 4. Built-in defaults
    pub fn load() -> Result<Self, ModelConfigError> {
        // Check environment variable first
        if let Ok(path) = std::env::var("EDGEQUAKE_MODELS_CONFIG") {
            if Path::new(&path).exists() {
                return Self::from_file(&path);
            }
        }

        // Check current directory
        let local_path = Path::new("models.toml");
        if local_path.exists() {
            return Self::from_file(local_path);
        }

        // Check user config directory
        if let Some(home) = dirs::home_dir() {
            let user_path = home.join(".edgequake").join("models.toml");
            if user_path.exists() {
                return Self::from_file(&user_path);
            }
        }

        // Fall back to built-in defaults
        Ok(Self::builtin_defaults())
    }

    /// Load configuration from a specific file path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ModelConfigError> {
        let content = std::fs::read_to_string(path.as_ref())?;
        Self::from_toml(&content)
    }

    /// Parse configuration from TOML string.
    pub fn from_toml(toml_str: &str) -> Result<Self, ModelConfigError> {
        toml::from_str(toml_str).map_err(|e| ModelConfigError::ParseError(e.to_string()))
    }

    /// Serialize configuration to TOML string.
    pub fn to_toml(&self) -> Result<String, ModelConfigError> {
        toml::to_string_pretty(self).map_err(|e| ModelConfigError::ParseError(e.to_string()))
    }

    /// Save configuration to a file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ModelConfigError> {
        let toml_str = self.to_toml()?;
        std::fs::write(path.as_ref(), toml_str)?;
        Ok(())
    }

    /// Get built-in default configuration with common providers.
    pub fn builtin_defaults() -> Self {
        Self {
            defaults: DefaultsConfig::default(),
            providers: vec![
                // OpenAI provider
                ProviderConfig {
                    name: "openai".to_string(),
                    display_name: "OpenAI".to_string(),
                    provider_type: ProviderType::OpenAI,
                    api_key_env: Some("OPENAI_API_KEY".to_string()),
                    base_url: Some("https://api.openai.com/v1".to_string()),
                    base_url_env: Some("OPENAI_API_BASE".to_string()),
                    default_llm_model: Some("gpt-4o-mini".to_string()),
                    default_embedding_model: Some("text-embedding-3-small".to_string()),
                    priority: 10,
                    models: vec![
                        ModelCard {
                            name: "gpt-4o".to_string(),
                            display_name: "GPT-4 Omni".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 128000,
                                max_output_tokens: 16384,
                                supports_vision: true,
                                supports_function_calling: true,
                                supports_json_mode: true,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost {
                                input_per_1k: 0.0025,
                                output_per_1k: 0.01,
                                ..Default::default()
                            },
                            description: "Most capable GPT-4 model with vision support".to_string(),
                            ..Default::default()
                        },
                        ModelCard {
                            name: "gpt-4o-mini".to_string(),
                            display_name: "GPT-4 Omni Mini".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 128000,
                                max_output_tokens: 16384,
                                supports_vision: true,
                                supports_function_calling: true,
                                supports_json_mode: true,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost {
                                input_per_1k: 0.00015,
                                output_per_1k: 0.0006,
                                ..Default::default()
                            },
                            description: "Cost-effective GPT-4 variant".to_string(),
                            ..Default::default()
                        },
                        ModelCard {
                            name: "text-embedding-3-small".to_string(),
                            display_name: "Embedding 3 Small".to_string(),
                            model_type: ModelType::Embedding,
                            capabilities: ModelCapabilities {
                                embedding_dimension: 1536,
                                max_embedding_tokens: 8191,
                                ..Default::default()
                            },
                            cost: ModelCost {
                                embedding_per_1k: 0.00002,
                                ..Default::default()
                            },
                            description: "Efficient embedding model".to_string(),
                            ..Default::default()
                        },
                        ModelCard {
                            name: "text-embedding-3-large".to_string(),
                            display_name: "Embedding 3 Large".to_string(),
                            model_type: ModelType::Embedding,
                            capabilities: ModelCapabilities {
                                embedding_dimension: 3072,
                                max_embedding_tokens: 8191,
                                ..Default::default()
                            },
                            cost: ModelCost {
                                embedding_per_1k: 0.00013,
                                ..Default::default()
                            },
                            description: "High-quality embedding model".to_string(),
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                // OODA-32: Anthropic provider (Claude models)
                // WHY: Direct Anthropic API access for Claude models
                // Supports: claude-sonnet-4.5, claude-3.5-sonnet, claude-3.5-haiku
                ProviderConfig {
                    name: "anthropic".to_string(),
                    display_name: "Anthropic (Claude)".to_string(),
                    provider_type: ProviderType::Anthropic,
                    api_key_env: Some("ANTHROPIC_API_KEY".to_string()),
                    base_url: Some("https://api.anthropic.com".to_string()),
                    base_url_env: Some("ANTHROPIC_API_BASE".to_string()),
                    default_llm_model: Some("claude-sonnet-4-5-20250929".to_string()),
                    default_embedding_model: None, // Anthropic doesn't support embeddings
                    priority: 15, // Higher than OpenAI (10), prefer Claude when available
                    models: vec![
                        ModelCard {
                            name: "claude-sonnet-4-5-20250929".to_string(),
                            display_name: "Claude Sonnet 4.5".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 200000,
                                max_output_tokens: 8192,
                                supports_vision: true,
                                supports_function_calling: true,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost {
                                input_per_1k: 0.003,
                                output_per_1k: 0.015,
                                ..Default::default()
                            },
                            description: "Anthropic's most capable model with excellent coding".to_string(),
                            ..Default::default()
                        },
                        ModelCard {
                            name: "claude-3-5-sonnet-20241022".to_string(),
                            display_name: "Claude 3.5 Sonnet".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 200000,
                                max_output_tokens: 8192,
                                supports_vision: true,
                                supports_function_calling: true,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost {
                                input_per_1k: 0.003,
                                output_per_1k: 0.015,
                                ..Default::default()
                            },
                            description: "Previous generation Sonnet, stable and reliable".to_string(),
                            ..Default::default()
                        },
                        ModelCard {
                            name: "claude-3-5-haiku-20241022".to_string(),
                            display_name: "Claude 3.5 Haiku".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 200000,
                                max_output_tokens: 8192,
                                supports_vision: true,
                                supports_function_calling: true,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost {
                                input_per_1k: 0.0008,
                                output_per_1k: 0.004,
                                ..Default::default()
                            },
                            description: "Fast and cost-effective Claude model".to_string(),
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                // Ollama provider
                ProviderConfig {
                    name: "ollama".to_string(),
                    display_name: "Ollama (Local)".to_string(),
                    provider_type: ProviderType::Ollama,
                    base_url: Some("http://localhost:11434".to_string()),
                    base_url_env: Some("OLLAMA_HOST".to_string()),
                    default_llm_model: Some("gemma3:12b".to_string()),
                    default_embedding_model: Some("nomic-embed-text".to_string()),
                    priority: 20,
                    models: vec![
                        ModelCard {
                            name: "gemma3:12b".to_string(),
                            display_name: "Gemma 3 12B".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 8192,
                                max_output_tokens: 4096,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(), // Free for local
                            description: "Google's Gemma 3 12B parameter model".to_string(),
                            ..Default::default()
                        },
                        ModelCard {
                            name: "llama3.3:70b".to_string(),
                            display_name: "Llama 3.3 70B".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 131072,
                                max_output_tokens: 8192,
                                supports_function_calling: true,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "Meta's Llama 3.3 70B with extended context".to_string(),
                            ..Default::default()
                        },
                        // OODA-32: Add qwen3-coder and gpt-oss:20b for coding tasks
                        ModelCard {
                            name: "qwen3-coder".to_string(),
                            display_name: "Qwen3 Coder".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 32768,
                                max_output_tokens: 8192,
                                supports_function_calling: true,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "Qwen3 optimized for coding tasks".to_string(),
                            ..Default::default()
                        },
                        ModelCard {
                            name: "gpt-oss:20b".to_string(),
                            display_name: "GPT-OSS 20B".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 32768,
                                max_output_tokens: 8192,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "Open-source GPT model, 20B parameters".to_string(),
                            ..Default::default()
                        },
                        ModelCard {
                            name: "nomic-embed-text".to_string(),
                            display_name: "Nomic Embed Text".to_string(),
                            model_type: ModelType::Embedding,
                            capabilities: ModelCapabilities {
                                embedding_dimension: 768,
                                max_embedding_tokens: 8192,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "High-quality local embedding model".to_string(),
                            ..Default::default()
                        },
                        ModelCard {
                            name: "mxbai-embed-large".to_string(),
                            display_name: "MxBai Embed Large".to_string(),
                            model_type: ModelType::Embedding,
                            capabilities: ModelCapabilities {
                                embedding_dimension: 1024,
                                max_embedding_tokens: 512,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "Large embedding model with 1024 dimensions".to_string(),
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                // LM Studio provider
                ProviderConfig {
                    name: "lmstudio".to_string(),
                    display_name: "LM Studio (Local)".to_string(),
                    provider_type: ProviderType::LMStudio,
                    base_url: Some("http://localhost:1234/v1".to_string()),
                    base_url_env: Some("LMSTUDIO_HOST".to_string()),
                    default_llm_model: Some("local-model".to_string()),
                    default_embedding_model: Some("nomic-embed-text-v1.5".to_string()),
                    priority: 30,
                    models: vec![
                        ModelCard {
                            name: "local-model".to_string(),
                            display_name: "Local LM Studio Model".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 4096,
                                max_output_tokens: 2048,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "Currently loaded model in LM Studio".to_string(),
                            ..Default::default()
                        },
                        ModelCard {
                            name: "nomic-embed-text-v1.5".to_string(),
                            display_name: "Nomic Embed Text v1.5".to_string(),
                            model_type: ModelType::Embedding,
                            capabilities: ModelCapabilities {
                                embedding_dimension: 768,
                                max_embedding_tokens: 8192,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "Nomic embedding model for LM Studio".to_string(),
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                // Z.ai provider (OpenAI-compatible)
                // OODA-200: Configurable OpenAI-compatible providers
                ProviderConfig {
                    name: "zai".to_string(),
                    display_name: "Z.AI Platform".to_string(),
                    provider_type: ProviderType::OpenAICompatible,
                    api_key_env: Some("ZAI_API_KEY".to_string()),
                    base_url: Some("https://api.z.ai/api/paas/v4".to_string()),
                    default_llm_model: Some("glm-4.7-flash".to_string()),
                    priority: 15,
                    headers: {
                        let mut h = std::collections::HashMap::new();
                        h.insert("Accept-Language".to_string(), "en-US,en".to_string());
                        h
                    },
                    supports_thinking: true,
                    models: vec![
                        ModelCard {
                            name: "glm-4.7".to_string(),
                            display_name: "GLM-4.7 (Premium)".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 128000,
                                max_output_tokens: 16384,
                                supports_vision: true,
                                supports_function_calling: true,
                                supports_json_mode: true,
                                supports_streaming: true,
                                supports_thinking: true,
                                ..Default::default()
                            },
                            cost: ModelCost {
                                input_per_1k: 0.2,
                                output_per_1k: 1.1,
                                ..Default::default()
                            },
                            description: "Z.ai's flagship model with thinking mode".to_string(),
                            tags: vec!["reasoning".to_string(), "coding".to_string(), "agent".to_string()],
                            ..Default::default()
                        },
                        ModelCard {
                            name: "glm-4.7-flash".to_string(),
                            display_name: "GLM-4.7 Flash (Fast)".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 128000,
                                max_output_tokens: 8192,
                                supports_function_calling: true,
                                supports_json_mode: true,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost {
                                input_per_1k: 0.0,
                                output_per_1k: 0.0,
                                ..Default::default()
                            },
                            description: "Free, fast Z.ai model".to_string(),
                            tags: vec!["fast".to_string(), "free".to_string()],
                            ..Default::default()
                        },
                        ModelCard {
                            name: "glm-4.5".to_string(),
                            display_name: "GLM-4.5 (Reasoning)".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 128000,
                                max_output_tokens: 96000,
                                supports_vision: true,
                                supports_function_calling: true,
                                supports_streaming: true,
                                supports_thinking: true,
                                ..Default::default()
                            },
                            cost: ModelCost {
                                input_per_1k: 0.2,
                                output_per_1k: 1.1,
                                ..Default::default()
                            },
                            description: "Z.ai reasoning model for complex tasks".to_string(),
                            tags: vec!["reasoning".to_string(), "coding".to_string()],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                // POE provider (OpenAI-compatible)
                // OODA-200: Configurable OpenAI-compatible providers
                // Updated 2026-01-24: Use correct POE API model names (PascalCase)
                // Reference: https://creator.poe.com/api-reference/listModels
                ProviderConfig {
                    name: "poe".to_string(),
                    display_name: "POE Platform".to_string(),
                    provider_type: ProviderType::OpenAICompatible,
                    api_key_env: Some("POE_API_KEY".to_string()),
                    base_url: Some("https://api.poe.com/v1".to_string()),
                    default_llm_model: Some("Claude-Haiku-4.5".to_string()),
                    priority: 16,
                    models: vec![
                        // Claude models via POE (Anthropic's latest)
                        ModelCard {
                            name: "Claude-Sonnet-4.5".to_string(),
                            display_name: "Claude Sonnet 4.5 (POE)".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 200000,
                                max_output_tokens: 16384,
                                supports_vision: true,
                                supports_function_calling: true,
                                supports_streaming: true,
                                supports_thinking: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "Claude Sonnet 4.5 - Anthropic's most advanced model via POE".to_string(),
                            tags: vec!["reasoning".to_string(), "coding".to_string()],
                            ..Default::default()
                        },
                        ModelCard {
                            name: "Claude-Haiku-4.5".to_string(),
                            display_name: "Claude Haiku 4.5 (POE)".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 200000,
                                max_output_tokens: 8192,
                                supports_vision: true,
                                supports_function_calling: true,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "Claude Haiku 4.5 - Fast and efficient with frontier intelligence via POE".to_string(),
                            tags: vec!["fast".to_string()],
                            ..Default::default()
                        },
                        ModelCard {
                            name: "Claude-Opus-4.1".to_string(),
                            display_name: "Claude Opus 4.1 (POE)".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 200000,
                                max_output_tokens: 16384,
                                supports_vision: true,
                                supports_function_calling: true,
                                supports_streaming: true,
                                supports_thinking: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "Claude Opus 4.1 - Anthropic's premium model for complex tasks via POE".to_string(),
                            tags: vec!["reasoning".to_string(), "pro".to_string()],
                            ..Default::default()
                        },
                        // GPT models via POE (OpenAI's latest)
                        ModelCard {
                            name: "GPT-5-Pro".to_string(),
                            display_name: "GPT-5 Pro (POE)".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 128000,
                                max_output_tokens: 32768,
                                supports_vision: true,
                                supports_function_calling: true,
                                supports_streaming: true,
                                supports_thinking: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "GPT-5 Pro - OpenAI's flagship model with extended reasoning via POE".to_string(),
                            tags: vec!["reasoning".to_string(), "pro".to_string()],
                            ..Default::default()
                        },
                        ModelCard {
                            name: "GPT-5".to_string(),
                            display_name: "GPT-5 (POE)".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 128000,
                                max_output_tokens: 16384,
                                supports_vision: true,
                                supports_function_calling: true,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "GPT-5 - OpenAI's next-generation model via POE".to_string(),
                            ..Default::default()
                        },
                        ModelCard {
                            name: "GPT-5-Codex".to_string(),
                            display_name: "GPT-5 Codex (POE)".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 128000,
                                max_output_tokens: 16384,
                                supports_function_calling: true,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "GPT-5 Codex - Specialized for software engineering tasks via POE".to_string(),
                            tags: vec!["coding".to_string()],
                            ..Default::default()
                        },
                        // Grok models via POE (xAI)
                        ModelCard {
                            name: "Grok-4".to_string(),
                            display_name: "Grok-4 (POE)".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 131072,
                                max_output_tokens: 32768,
                                supports_function_calling: true,
                                supports_streaming: true,
                                supports_thinking: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "Grok-4 - xAI's most intelligent language model via POE".to_string(),
                            tags: vec!["reasoning".to_string(), "coding".to_string()],
                            ..Default::default()
                        },
                        // DeepSeek models via POE
                        ModelCard {
                            name: "DeepSeek-R1".to_string(),
                            display_name: "DeepSeek R1 (POE)".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 128000,
                                max_output_tokens: 16384,
                                supports_function_calling: true,
                                supports_streaming: true,
                                supports_thinking: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "DeepSeek R1 - Top open-source reasoning model via POE".to_string(),
                            tags: vec!["reasoning".to_string(), "open-source".to_string()],
                            ..Default::default()
                        },
                        ModelCard {
                            name: "DeepSeek-V3".to_string(),
                            display_name: "DeepSeek V3 (POE)".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 128000,
                                max_output_tokens: 16384,
                                supports_function_calling: true,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "DeepSeek V3 - Advanced open-source model via POE".to_string(),
                            tags: vec!["open-source".to_string()],
                            ..Default::default()
                        },
                        // Gemini models via POE (Google)
                        ModelCard {
                            name: "Gemini-2.5-Pro".to_string(),
                            display_name: "Gemini 2.5 Pro (POE)".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 1000000,
                                max_output_tokens: 65536,
                                supports_vision: true,
                                supports_function_calling: true,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "Gemini 2.5 Pro - Google's advanced model with web search via POE".to_string(),
                            tags: vec!["reasoning".to_string(), "web-search".to_string()],
                            ..Default::default()
                        },
                        ModelCard {
                            name: "Gemini-2.5-Flash".to_string(),
                            display_name: "Gemini 2.5 Flash (POE)".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 1000000,
                                max_output_tokens: 65536,
                                supports_vision: true,
                                supports_function_calling: true,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "Gemini 2.5 Flash - Fast variant with large context via POE".to_string(),
                            tags: vec!["fast".to_string()],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                // Mistral AI provider
                ProviderConfig {
                    name: "mistral".to_string(),
                    display_name: "Mistral AI".to_string(),
                    provider_type: ProviderType::Mistral,
                    api_key_env: Some("MISTRAL_API_KEY".to_string()),
                    base_url: Some("https://api.mistral.ai/v1".to_string()),
                    default_llm_model: Some("mistral-small-latest".to_string()),
                    default_embedding_model: Some("mistral-embed".to_string()),
                    priority: 50,
                    models: vec![
                        ModelCard {
                            name: "mistral-small-latest".to_string(),
                            display_name: "Mistral Small (Latest)".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 32768,
                                max_output_tokens: 4096,
                                supports_vision: false,
                                supports_function_calling: true,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "Mistral Small — efficient and cost-effective model".to_string(),
                            tags: vec!["fast".to_string(), "affordable".to_string()],
                            ..Default::default()
                        },
                        ModelCard {
                            name: "mistral-large-latest".to_string(),
                            display_name: "Mistral Large (Latest)".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 131072,
                                max_output_tokens: 4096,
                                supports_vision: false,
                                supports_function_calling: true,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "Mistral Large — flagship reasoning model".to_string(),
                            tags: vec!["powerful".to_string()],
                            ..Default::default()
                        },
                        ModelCard {
                            name: "mistral-medium-latest".to_string(),
                            display_name: "Mistral Medium (Latest)".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 131072,
                                max_output_tokens: 4096,
                                supports_vision: false,
                                supports_function_calling: true,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "Mistral Medium — balanced performance model".to_string(),
                            tags: vec!["balanced".to_string()],
                            ..Default::default()
                        },
                        ModelCard {
                            name: "codestral-latest".to_string(),
                            display_name: "Codestral (Latest)".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 32768,
                                max_output_tokens: 4096,
                                supports_vision: false,
                                supports_function_calling: true,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "Codestral — specialized code generation model".to_string(),
                            tags: vec!["code".to_string()],
                            ..Default::default()
                        },
                        ModelCard {
                            name: "mistral-embed".to_string(),
                            display_name: "Mistral Embed".to_string(),
                            model_type: ModelType::Embedding,
                            capabilities: ModelCapabilities {
                                embedding_dimension: 1024,
                                max_embedding_tokens: 8192,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "Mistral embedding model — 1024-dimensional dense embeddings".to_string(),
                            tags: vec!["embedding".to_string()],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                // Mock provider for testing
                ProviderConfig {
                    name: "mock".to_string(),
                    display_name: "Mock (Testing)".to_string(),
                    provider_type: ProviderType::Mock,
                    default_llm_model: Some("mock-model".to_string()),
                    default_embedding_model: Some("mock-embedding".to_string()),
                    priority: 1000,
                    models: vec![
                        ModelCard {
                            name: "mock-model".to_string(),
                            display_name: "Mock LLM".to_string(),
                            model_type: ModelType::Llm,
                            capabilities: ModelCapabilities {
                                context_length: 4096,
                                max_output_tokens: 2048,
                                supports_streaming: true,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "Mock model for testing".to_string(),
                            ..Default::default()
                        },
                        ModelCard {
                            name: "mock-embedding".to_string(),
                            display_name: "Mock Embedding".to_string(),
                            model_type: ModelType::Embedding,
                            capabilities: ModelCapabilities {
                                embedding_dimension: 1536,
                                max_embedding_tokens: 512,
                                ..Default::default()
                            },
                            cost: ModelCost::default(),
                            description: "Mock embedding for testing".to_string(),
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
            ],
        }
    }

    /// Get a provider by name.
    pub fn get_provider(&self, name: &str) -> Option<&ProviderConfig> {
        self.providers.iter().find(|p| p.name == name)
    }

    /// Get a model by provider and model name.
    pub fn get_model(&self, provider: &str, model: &str) -> Option<&ModelCard> {
        self.get_provider(provider)
            .and_then(|p| p.models.iter().find(|m| m.name == model))
    }

    /// OODA-200: Find a provider by model name.
    ///
    /// Searches all enabled providers for a model with the given name.
    /// Returns the provider config if found, None otherwise.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The model identifier to search for (e.g., "glm-4.7")
    ///
    /// # Returns
    ///
    /// The provider configuration containing this model, or None.
    pub fn find_provider_for_model(&self, model_name: &str) -> Option<&ProviderConfig> {
        self.providers
            .iter()
            .find(|p| p.enabled && p.models.iter().any(|m| m.name == model_name))
    }

    /// OODA-200: Find a provider and model by model name.
    ///
    /// Searches all enabled providers for a model with the given name.
    /// Returns both the provider config and model card if found.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The model identifier to search for (e.g., "glm-4.7")
    ///
    /// # Returns
    ///
    /// A tuple of (ProviderConfig, ModelCard) if found, None otherwise.
    pub fn find_provider_and_model(
        &self,
        model_name: &str,
    ) -> Option<(&ProviderConfig, &ModelCard)> {
        for provider in &self.providers {
            if !provider.enabled {
                continue;
            }
            for model in &provider.models {
                if model.name == model_name {
                    return Some((provider, model));
                }
            }
        }
        None
    }

    /// Get all LLM models across all providers.
    pub fn all_llm_models(&self) -> Vec<(&ProviderConfig, &ModelCard)> {
        self.providers
            .iter()
            .filter(|p| p.enabled)
            .flat_map(|p| {
                p.models
                    .iter()
                    .filter(|m| matches!(m.model_type, ModelType::Llm | ModelType::Multimodal))
                    .map(move |m| (p, m))
            })
            .collect()
    }

    /// Get all embedding models across all providers.
    pub fn all_embedding_models(&self) -> Vec<(&ProviderConfig, &ModelCard)> {
        self.providers
            .iter()
            .filter(|p| p.enabled)
            .flat_map(|p| {
                p.models
                    .iter()
                    .filter(|m| {
                        matches!(m.model_type, ModelType::Embedding | ModelType::Multimodal)
                    })
                    .map(move |m| (p, m))
            })
            .collect()
    }

    /// Get the default LLM provider and model.
    pub fn default_llm(&self) -> Option<(&ProviderConfig, &ModelCard)> {
        self.get_model(&self.defaults.llm_provider, &self.defaults.llm_model)
            .and_then(|m| {
                self.get_provider(&self.defaults.llm_provider)
                    .map(|p| (p, m))
            })
    }

    /// Get the default embedding provider and model.
    pub fn default_embedding(&self) -> Option<(&ProviderConfig, &ModelCard)> {
        self.get_model(
            &self.defaults.embedding_provider,
            &self.defaults.embedding_model,
        )
        .and_then(|m| {
            self.get_provider(&self.defaults.embedding_provider)
                .map(|p| (p, m))
        })
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), ModelConfigError> {
        // Check that default providers exist
        if self.get_provider(&self.defaults.llm_provider).is_none() {
            return Err(ModelConfigError::ValidationError(format!(
                "Default LLM provider '{}' not found in providers list",
                self.defaults.llm_provider
            )));
        }

        if self
            .get_provider(&self.defaults.embedding_provider)
            .is_none()
        {
            return Err(ModelConfigError::ValidationError(format!(
                "Default embedding provider '{}' not found in providers list",
                self.defaults.embedding_provider
            )));
        }

        // Check that default models exist
        if self
            .get_model(&self.defaults.llm_provider, &self.defaults.llm_model)
            .is_none()
        {
            return Err(ModelConfigError::ValidationError(format!(
                "Default LLM model '{}' not found in provider '{}'",
                self.defaults.llm_model, self.defaults.llm_provider
            )));
        }

        if self
            .get_model(
                &self.defaults.embedding_provider,
                &self.defaults.embedding_model,
            )
            .is_none()
        {
            return Err(ModelConfigError::ValidationError(format!(
                "Default embedding model '{}' not found in provider '{}'",
                self.defaults.embedding_model, self.defaults.embedding_provider
            )));
        }

        // Check for duplicate provider names
        let mut seen_providers = std::collections::HashSet::new();
        for provider in &self.providers {
            if !seen_providers.insert(&provider.name) {
                return Err(ModelConfigError::ValidationError(format!(
                    "Duplicate provider name: '{}'",
                    provider.name
                )));
            }

            // Check for duplicate model names within a provider
            let mut seen_models = std::collections::HashSet::new();
            for model in &provider.models {
                if !seen_models.insert(&model.name) {
                    return Err(ModelConfigError::ValidationError(format!(
                        "Duplicate model name '{}' in provider '{}'",
                        model.name, provider.name
                    )));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_defaults() {
        let config = ModelsConfig::builtin_defaults();
        assert!(config.validate().is_ok());
        assert!(!config.providers.is_empty());
    }

    #[test]
    fn test_get_provider() {
        let config = ModelsConfig::builtin_defaults();
        assert!(config.get_provider("openai").is_some());
        assert!(config.get_provider("ollama").is_some());
        assert!(config.get_provider("nonexistent").is_none());
    }

    #[test]
    fn test_get_model() {
        let config = ModelsConfig::builtin_defaults();
        assert!(config.get_model("openai", "gpt-4o").is_some());
        assert!(config.get_model("ollama", "nomic-embed-text").is_some());
        assert!(config.get_model("openai", "nonexistent").is_none());
    }

    #[test]
    fn test_all_llm_models() {
        let config = ModelsConfig::builtin_defaults();
        let llm_models = config.all_llm_models();
        assert!(!llm_models.is_empty());
        assert!(llm_models.iter().any(|(_, m)| m.name == "gpt-4o"));
    }

    #[test]
    fn test_all_embedding_models() {
        let config = ModelsConfig::builtin_defaults();
        let embedding_models = config.all_embedding_models();
        assert!(!embedding_models.is_empty());
        assert!(embedding_models
            .iter()
            .any(|(_, m)| m.name == "text-embedding-3-small"));
    }

    #[test]
    fn test_toml_roundtrip() {
        let config = ModelsConfig::builtin_defaults();
        let toml_str = config.to_toml().expect("Failed to serialize");
        let parsed: ModelsConfig = ModelsConfig::from_toml(&toml_str).expect("Failed to parse");
        assert_eq!(config.providers.len(), parsed.providers.len());
    }

    #[test]
    fn test_model_capabilities() {
        let config = ModelsConfig::builtin_defaults();
        let gpt4o = config
            .get_model("openai", "gpt-4o")
            .expect("gpt-4o should exist");
        assert!(gpt4o.capabilities.supports_vision);
        assert!(gpt4o.capabilities.supports_function_calling);
        assert_eq!(gpt4o.capabilities.context_length, 128000);
    }

    #[test]
    fn test_embedding_dimensions() {
        let config = ModelsConfig::builtin_defaults();

        let openai_embed = config
            .get_model("openai", "text-embedding-3-small")
            .unwrap();
        assert_eq!(openai_embed.capabilities.embedding_dimension, 1536);

        let ollama_embed = config.get_model("ollama", "nomic-embed-text").unwrap();
        assert_eq!(ollama_embed.capabilities.embedding_dimension, 768);
    }

    #[test]
    fn test_validation_duplicate_provider() {
        let mut config = ModelsConfig::builtin_defaults();
        config.providers.push(config.providers[0].clone());
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_parse_models_toml_file() {
        // Read the actual models.toml file from the project root
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let toml_path = std::path::Path::new(&manifest_dir)
            .parent() // crates/
            .unwrap()
            .parent() // edgequake/
            .unwrap()
            .join("models.toml");

        if toml_path.exists() {
            let content = std::fs::read_to_string(&toml_path).expect("Failed to read models.toml");
            let config = ModelsConfig::from_toml(&content).expect("Failed to parse models.toml");

            // Validate the parsed config
            assert!(config.validate().is_ok(), "models.toml failed validation");

            // Check we have expected providers
            assert!(
                config.get_provider("openai").is_some(),
                "OpenAI provider should exist"
            );
            assert!(
                config.get_provider("ollama").is_some(),
                "Ollama provider should exist"
            );
            assert!(
                config.get_provider("lmstudio").is_some(),
                "LM Studio provider should exist"
            );
            assert!(
                config.get_provider("mock").is_some(),
                "Mock provider should exist"
            );

            // Check default selections are set
            assert!(!config.defaults.llm_provider.is_empty());
            assert!(!config.defaults.llm_model.is_empty());
            assert!(!config.defaults.embedding_provider.is_empty());
            assert!(!config.defaults.embedding_model.is_empty());

            // Check we have LLM and embedding models
            let llm_models = config.all_llm_models();
            let embedding_models = config.all_embedding_models();
            assert!(!llm_models.is_empty(), "Should have LLM models");
            assert!(!embedding_models.is_empty(), "Should have embedding models");
        }
    }

    #[test]
    fn test_provider_priorities() {
        let config = ModelsConfig::builtin_defaults();
        let mut priorities: Vec<(String, u32)> = config
            .providers
            .iter()
            .map(|p| (p.name.clone(), p.priority))
            .collect();
        priorities.sort_by_key(|(_, p)| *p);

        // Lower priority means higher preference
        // OpenAI should have lower priority number than Mock
        let openai_prio = config.get_provider("openai").unwrap().priority;
        let mock_prio = config.get_provider("mock").unwrap().priority;
        assert!(
            openai_prio < mock_prio,
            "OpenAI should have higher priority than mock"
        );
    }

    // ====================================================================
    // Display impl tests
    // ====================================================================

    #[test]
    fn test_model_type_display() {
        assert_eq!(ModelType::Llm.to_string(), "llm");
        assert_eq!(ModelType::Embedding.to_string(), "embedding");
        assert_eq!(ModelType::Multimodal.to_string(), "multimodal");
    }

    #[test]
    fn test_provider_type_display() {
        assert_eq!(ProviderType::OpenAI.to_string(), "openai");
        assert_eq!(ProviderType::Ollama.to_string(), "ollama");
        assert_eq!(ProviderType::LMStudio.to_string(), "lmstudio");
        assert_eq!(ProviderType::Azure.to_string(), "azure");
        assert_eq!(ProviderType::Anthropic.to_string(), "anthropic");
        assert_eq!(ProviderType::OpenRouter.to_string(), "openrouter");
        assert_eq!(
            ProviderType::OpenAICompatible.to_string(),
            "openai_compatible"
        );
        assert_eq!(ProviderType::Mock.to_string(), "mock");
    }

    // ====================================================================
    // Default impl tests
    // ====================================================================

    #[test]
    fn test_model_type_default() {
        assert_eq!(ModelType::default(), ModelType::Llm);
    }

    #[test]
    fn test_provider_type_default() {
        assert_eq!(ProviderType::default(), ProviderType::OpenAI);
    }

    #[test]
    fn test_model_card_default() {
        let card = ModelCard::default();
        assert_eq!(card.name, "unknown");
        assert_eq!(card.display_name, "Unknown Model");
        assert_eq!(card.model_type, ModelType::Llm);
        assert!(!card.deprecated);
        assert!(card.replacement.is_none());
        assert!(card.tags.is_empty());
    }

    #[test]
    fn test_provider_config_default() {
        let config = ProviderConfig::default();
        assert_eq!(config.name, "unknown");
        assert!(config.enabled);
        assert_eq!(config.priority, 100);
        assert_eq!(config.timeout_seconds, 120);
        assert!(config.api_key_env.is_none());
    }

    #[test]
    fn test_defaults_config_default() {
        let defaults = DefaultsConfig::default();
        assert_eq!(defaults.llm_provider, "openai");
        assert_eq!(defaults.llm_model, "gpt-4o-mini");
        assert_eq!(defaults.embedding_provider, "openai");
        assert_eq!(defaults.embedding_model, "text-embedding-3-small");
    }

    #[test]
    fn test_model_capabilities_default() {
        let caps = ModelCapabilities::default();
        assert_eq!(caps.context_length, 0);
        assert!(!caps.supports_vision);
        assert!(!caps.supports_function_calling);
    }

    #[test]
    fn test_model_cost_default() {
        let cost = ModelCost::default();
        assert_eq!(cost.input_per_1k, 0.0);
        assert_eq!(cost.output_per_1k, 0.0);
    }

    // ====================================================================
    // Find methods
    // ====================================================================

    #[test]
    fn test_find_provider_for_model() {
        let config = ModelsConfig::builtin_defaults();
        let provider = config.find_provider_for_model("gpt-4o");
        assert!(provider.is_some());
        assert_eq!(provider.unwrap().name, "openai");
    }

    #[test]
    fn test_find_provider_for_model_not_found() {
        let config = ModelsConfig::builtin_defaults();
        assert!(config
            .find_provider_for_model("nonexistent-model-xyz")
            .is_none());
    }

    #[test]
    fn test_find_provider_and_model() {
        let config = ModelsConfig::builtin_defaults();
        let result = config.find_provider_and_model("gpt-4o");
        assert!(result.is_some());
        let (provider, model) = result.unwrap();
        assert_eq!(provider.name, "openai");
        assert_eq!(model.name, "gpt-4o");
    }

    #[test]
    fn test_find_provider_and_model_not_found() {
        let config = ModelsConfig::builtin_defaults();
        assert!(config.find_provider_and_model("nonexistent-xyz").is_none());
    }

    // ====================================================================
    // Default model selection
    // ====================================================================

    #[test]
    fn test_default_llm() {
        let config = ModelsConfig::builtin_defaults();
        let result = config.default_llm();
        assert!(result.is_some());
        let (provider, model) = result.unwrap();
        assert_eq!(provider.name, "openai");
        assert_eq!(model.name, "gpt-4o-mini");
    }

    #[test]
    fn test_default_embedding() {
        let config = ModelsConfig::builtin_defaults();
        let result = config.default_embedding();
        assert!(result.is_some());
        let (provider, model) = result.unwrap();
        assert_eq!(provider.name, "openai");
        assert_eq!(model.name, "text-embedding-3-small");
    }

    // ====================================================================
    // Error type tests
    // ====================================================================

    #[test]
    fn test_model_config_error_display() {
        let err = ModelConfigError::ProviderNotFound("test".to_string());
        assert!(err.to_string().contains("test"));

        let err = ModelConfigError::ModelNotFound("gpt-5".to_string());
        assert!(err.to_string().contains("gpt-5"));

        let err = ModelConfigError::ValidationError("missing field".to_string());
        assert!(err.to_string().contains("missing field"));

        let err = ModelConfigError::ParseError("bad toml".to_string());
        assert!(err.to_string().contains("bad toml"));
    }

    // ====================================================================
    // TOML parsing edge cases
    // ====================================================================

    #[test]
    fn test_from_toml_invalid() {
        let result = ModelsConfig::from_toml("this is not valid toml {{{");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_toml_empty() {
        let config = ModelsConfig::from_toml("").unwrap();
        assert!(config.providers.is_empty());
    }

    #[test]
    fn test_models_config_default() {
        let config = ModelsConfig::default();
        assert!(config.providers.is_empty());
    }

    #[test]
    fn test_validation_empty_config() {
        let config = ModelsConfig::default();
        // Empty config should fail validation because default providers are missing
        assert!(config.validate().is_err());
    }
}
