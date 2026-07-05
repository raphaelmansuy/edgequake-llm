//! Discovery provider trait — the core abstraction for model discovery.
//!
//! Separate from `LLMProvider` by design (Interface Segregation):
//! discovery doesn't require a model to be selected or a chat session.

use async_trait::async_trait;

use super::types::{DiscoveredModel, DiscoveryStrategy};

/// Trait for providers that can discover available models.
///
/// Implementors return `Vec<DiscoveredModel>` — a normalized, provider-agnostic
/// representation of model capabilities. Three strategies exist:
///
/// - **Dynamic**: live API call (Ollama, OpenRouter, Anthropic, Gemini, Mistral, LM Studio)
/// - **Static**: built-in registry with cited documentation (xAI, HuggingFace)
/// - **Hybrid**: API call for availability + static data for capabilities (OpenAI, NVIDIA, Bedrock)
#[async_trait]
pub trait ModelDiscoveryProvider: Send + Sync {
    /// Provider identifier string (e.g., "openai", "anthropic").
    fn provider_id(&self) -> &str;

    /// How this provider discovers models.
    fn discovery_strategy(&self) -> DiscoveryStrategy;

    /// Discover all available models from this provider.
    ///
    /// On network failure, implementations MUST return `Ok(Vec::new())`
    /// and log a warning — never propagate connection errors as hard failures.
    async fn discover_models(&self) -> crate::error::Result<Vec<DiscoveredModel>>;

    /// Get capabilities for a specific model by ID.
    ///
    /// Default implementation searches `discover_models()` results.
    async fn get_model_capabilities(
        &self,
        model_id: &str,
    ) -> crate::error::Result<Option<DiscoveredModel>> {
        let models = self.discover_models().await?;
        Ok(models.into_iter().find(|m| m.id == model_id))
    }
}
