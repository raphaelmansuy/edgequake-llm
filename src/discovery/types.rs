//! Core types for the model discovery system.
//!
//! Every type in this module follows the Anti-Heuristic Manifesto:
//! capabilities come from API responses, verified documentation, or
//! explicit "unknown" — never from name-pattern inference.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::model_config::{ModelCapabilities, ModelType};

/// A discovered model with normalized capabilities.
///
/// Single source of truth for model metadata regardless of whether
/// it came from a dynamic API call or a static registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredModel {
    /// Provider-specific model ID (e.g., "gpt-4.1", "claude-opus-4-8").
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// Provider that owns this model (string, not enum — avoids coupling to ProviderType).
    pub provider: String,
    /// Maximum input context window in tokens.
    pub context_length: usize,
    /// Maximum output tokens the model can generate.
    pub max_output_tokens: usize,
    /// Normalized capabilities.
    pub capabilities: ModelCapabilities,
    /// How this model was discovered.
    pub source: DiscoverySource,
    /// When this information was last verified.
    pub discovered_at: DateTime<Utc>,
    /// Whether the model is currently available/loaded.
    pub available: bool,
    /// Cost per million input tokens (USD).
    pub cost_per_m_input: Option<f64>,
    /// Cost per million output tokens (USD).
    pub cost_per_m_output: Option<f64>,
    /// Model type (LLM, Embedding, Multimodal).
    pub model_type: ModelType,
    /// Tags for filtering (e.g., "reasoning", "coding", "fast").
    pub tags: Vec<String>,
    /// Whether the model is deprecated.
    pub deprecated: bool,
}

impl Default for DiscoveredModel {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            provider: String::new(),
            context_length: 0,
            max_output_tokens: 0,
            capabilities: ModelCapabilities::default(),
            source: DiscoverySource::Unknown,
            discovered_at: Utc::now(),
            available: true,
            cost_per_m_input: None,
            cost_per_m_output: None,
            model_type: ModelType::Llm,
            tags: Vec::new(),
            deprecated: false,
        }
    }
}

/// How a model was discovered.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum DiscoverySource {
    /// Discovered via live API call (highest freshness).
    DynamicApi,
    /// From built-in static registry (manually verified, may be stale).
    StaticRegistry,
    /// Dynamic API enriched with static metadata.
    Hybrid,
    /// User-provided configuration (models.toml override).
    UserConfig,
    /// Model not found in any source — all capabilities are unknown.
    /// Caller MUST handle this explicitly (never guess).
    #[default]
    Unknown,
}

/// How a provider discovers its models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryStrategy {
    /// Models discovered via live API call (Ollama, LM Studio, OpenRouter, Anthropic, Gemini, Mistral).
    Dynamic,
    /// Models from a built-in static registry (xAI, HuggingFace).
    Static,
    /// API call enriched with static metadata (OpenAI, NVIDIA, Bedrock).
    Hybrid,
}

/// Filter for discovering models by capability.
///
/// All fields use AND logic — a model must satisfy every non-None constraint.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapabilityFilter {
    /// Minimum input context window (tokens).
    pub min_context_length: Option<usize>,
    /// Maximum input context window (tokens).
    pub max_context_length: Option<usize>,
    /// Minimum max output tokens the model can generate.
    pub min_output_tokens: Option<usize>,
    /// Maximum max output tokens the model can generate.
    pub max_output_tokens: Option<usize>,
    pub requires_vision: Option<bool>,
    pub requires_tools: Option<bool>,
    pub requires_thinking: Option<bool>,
    pub requires_streaming: Option<bool>,
    pub requires_json_mode: Option<bool>,
    pub model_type: Option<ModelType>,
    pub provider: Option<String>,
    pub tags: Option<Vec<String>>,
    pub max_cost_per_m_input: Option<f64>,
    pub exclude_deprecated: Option<bool>,
}

/// Typed model capability for ergonomic filter construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelCapability {
    Vision,
    Tools,
    Thinking,
    Streaming,
    JsonMode,
}

impl CapabilityFilter {
    /// Require a specific capability (sets the corresponding `requires_*` field to `true`).
    pub fn requiring(mut self, capability: ModelCapability) -> Self {
        match capability {
            ModelCapability::Vision => self.requires_vision = Some(true),
            ModelCapability::Tools => self.requires_tools = Some(true),
            ModelCapability::Thinking => self.requires_thinking = Some(true),
            ModelCapability::Streaming => self.requires_streaming = Some(true),
            ModelCapability::JsonMode => self.requires_json_mode = Some(true),
        }
        self
    }

    /// Require multiple capabilities (AND logic).
    pub fn requiring_all(
        mut self,
        capabilities: impl IntoIterator<Item = ModelCapability>,
    ) -> Self {
        for capability in capabilities {
            self = self.requiring(capability);
        }
        self
    }

    pub fn with_min_context_length(mut self, tokens: usize) -> Self {
        self.min_context_length = Some(tokens);
        self
    }

    pub fn with_max_context_length(mut self, tokens: usize) -> Self {
        self.max_context_length = Some(tokens);
        self
    }

    pub fn with_min_output_tokens(mut self, tokens: usize) -> Self {
        self.min_output_tokens = Some(tokens);
        self
    }

    pub fn with_max_output_tokens(mut self, tokens: usize) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    pub fn with_max_cost_per_m_input(mut self, usd: f64) -> Self {
        self.max_cost_per_m_input = Some(usd);
        self
    }

    pub fn excluding_deprecated(mut self) -> Self {
        self.exclude_deprecated = Some(true);
        self
    }

    pub fn with_model_type(mut self, model_type: ModelType) -> Self {
        self.model_type = Some(model_type);
        self
    }

    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags = Some(tags.into_iter().map(Into::into).collect());
        self
    }

    /// Test whether a model matches all filter constraints.
    pub fn matches(&self, model: &DiscoveredModel) -> bool {
        if let Some(min) = self.min_context_length {
            if model.context_length < min {
                return false;
            }
        }
        if let Some(max) = self.max_context_length {
            if model.context_length > max {
                return false;
            }
        }
        if let Some(min) = self.min_output_tokens {
            if model.max_output_tokens < min {
                return false;
            }
        }
        if let Some(max) = self.max_output_tokens {
            if model.max_output_tokens > max {
                return false;
            }
        }
        if let Some(true) = self.requires_vision {
            if !model.capabilities.supports_vision {
                return false;
            }
        }
        if let Some(true) = self.requires_tools {
            if !model.capabilities.supports_function_calling {
                return false;
            }
        }
        if let Some(true) = self.requires_thinking {
            if !model.capabilities.supports_thinking {
                return false;
            }
        }
        if let Some(true) = self.requires_streaming {
            if !model.capabilities.supports_streaming {
                return false;
            }
        }
        if let Some(true) = self.requires_json_mode {
            if !model.capabilities.supports_json_mode {
                return false;
            }
        }
        if let Some(ref mt) = self.model_type {
            if &model.model_type != mt {
                return false;
            }
        }
        if let Some(ref p) = self.provider {
            if &model.provider != p {
                return false;
            }
        }
        if let Some(ref tags) = self.tags {
            if !tags.iter().any(|t| model.tags.contains(t)) {
                return false;
            }
        }
        if let Some(max) = self.max_cost_per_m_input {
            match model.cost_per_m_input {
                Some(cost) if cost > max => return false,
                None => return false,
                _ => {}
            }
        }
        if let Some(true) = self.exclude_deprecated {
            if model.deprecated {
                return false;
            }
        }
        true
    }
}

/// Errors specific to model discovery.
#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    #[error("Provider '{provider}' unreachable: {source}")]
    ProviderUnreachable {
        provider: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Failed to parse discovery response from '{provider}': {details}")]
    ParseError { provider: String, details: String },

    #[error("Authentication required for '{provider}' model discovery")]
    AuthRequired { provider: String },

    #[error("Discovery timed out for '{provider}' after {timeout_ms}ms")]
    Timeout { provider: String, timeout_ms: u64 },

    #[error("Provider '{provider}' not registered")]
    ProviderNotRegistered { provider: String },

    #[error("{0}")]
    Other(String),
}

impl From<DiscoveryError> for crate::error::LlmError {
    fn from(e: DiscoveryError) -> Self {
        crate::error::LlmError::Unknown(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model(
        id: &str,
        provider: &str,
        ctx: usize,
        vision: bool,
        tools: bool,
        thinking: bool,
    ) -> DiscoveredModel {
        DiscoveredModel {
            id: id.into(),
            name: id.into(),
            provider: provider.into(),
            context_length: ctx,
            max_output_tokens: 4096,
            capabilities: ModelCapabilities {
                supports_vision: vision,
                supports_function_calling: tools,
                supports_thinking: thinking,
                supports_streaming: true,
                supports_json_mode: true,
                ..Default::default()
            },
            cost_per_m_input: Some(2.0),
            ..Default::default()
        }
    }

    #[test]
    fn test_filter_matches_all() {
        let model = make_model("gpt-4.1", "openai", 1_000_000, true, true, false);
        let filter = CapabilityFilter::default();
        assert!(filter.matches(&model));
    }

    #[test]
    fn test_filter_max_context_length() {
        let model = make_model("big", "openai", 1_000_000, true, true, false);
        let filter = CapabilityFilter {
            max_context_length: Some(500_000),
            ..Default::default()
        };
        assert!(!filter.matches(&model));
    }

    #[test]
    fn test_filter_max_output_tokens() {
        let mut model = make_model("small-out", "openai", 128_000, true, true, false);
        model.max_output_tokens = 4096;
        let filter = CapabilityFilter {
            max_output_tokens: Some(2048),
            ..Default::default()
        };
        assert!(!filter.matches(&model));
    }

    #[test]
    fn test_filter_min_context() {
        let model = make_model("small", "openai", 8_000, false, false, false);
        let filter = CapabilityFilter {
            min_context_length: Some(100_000),
            ..Default::default()
        };
        assert!(!filter.matches(&model));
    }

    #[test]
    fn test_filter_requires_vision() {
        let model = make_model("no-vision", "openai", 128_000, false, true, false);
        let filter = CapabilityFilter {
            requires_vision: Some(true),
            ..Default::default()
        };
        assert!(!filter.matches(&model));
    }

    #[test]
    fn test_filter_requires_tools() {
        let model = make_model("no-tools", "openai", 128_000, true, false, false);
        let filter = CapabilityFilter {
            requires_tools: Some(true),
            ..Default::default()
        };
        assert!(!filter.matches(&model));
    }

    #[test]
    fn test_filter_requires_thinking() {
        let model = make_model("thinker", "anthropic", 200_000, true, true, true);
        let filter = CapabilityFilter {
            requires_thinking: Some(true),
            ..Default::default()
        };
        assert!(filter.matches(&model));
    }

    #[test]
    fn test_filter_provider() {
        let model = make_model("claude", "anthropic", 200_000, true, true, true);
        let filter = CapabilityFilter {
            provider: Some("openai".into()),
            ..Default::default()
        };
        assert!(!filter.matches(&model));
    }

    #[test]
    fn test_filter_max_cost() {
        let model = make_model("expensive", "openai", 128_000, true, true, false);
        let filter = CapabilityFilter {
            max_cost_per_m_input: Some(1.0),
            ..Default::default()
        };
        assert!(!filter.matches(&model));
    }

    #[test]
    fn test_filter_exclude_deprecated() {
        let mut model = make_model("old", "openai", 128_000, true, true, false);
        model.deprecated = true;
        let filter = CapabilityFilter {
            exclude_deprecated: Some(true),
            ..Default::default()
        };
        assert!(!filter.matches(&model));
    }

    #[test]
    fn test_filter_builder_requires_capabilities() {
        let filter = CapabilityFilter::default()
            .requiring_all([ModelCapability::Vision, ModelCapability::Tools]);
        let model = make_model("vision-tools", "openai", 128_000, true, true, false);
        assert!(filter.matches(&model));
        let no_tools = make_model("vision-only", "openai", 128_000, true, false, false);
        assert!(!filter.matches(&no_tools));
    }

    #[test]
    fn test_filter_combined() {
        let model = make_model("claude-opus-4-8", "anthropic", 1_000_000, true, true, true);
        let filter = CapabilityFilter {
            min_context_length: Some(500_000),
            requires_vision: Some(true),
            requires_tools: Some(true),
            requires_thinking: Some(true),
            provider: Some("anthropic".into()),
            ..Default::default()
        };
        assert!(filter.matches(&model));
    }

    #[test]
    fn test_discovery_source_default_is_unknown() {
        assert_eq!(DiscoverySource::default(), DiscoverySource::Unknown);
    }

    #[test]
    fn test_unknown_model_has_zero_capabilities() {
        let model = DiscoveredModel::default();
        assert_eq!(model.context_length, 0);
        assert_eq!(model.max_output_tokens, 0);
        assert!(!model.capabilities.supports_vision);
        assert!(!model.capabilities.supports_function_calling);
        assert!(!model.capabilities.supports_thinking);
        assert_eq!(model.source, DiscoverySource::Unknown);
    }
}
