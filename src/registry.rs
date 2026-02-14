//! LLM Provider Registry - Pluggable Provider Management
//!
//! # Purpose
//!
//! Enables dynamic registration and lookup of LLM and Embedding providers,
//! supporting extensibility without modifying factory code.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │              ProviderRegistry                               │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                             │
//! │  LLM Providers:                                             │
//! │  ┌─ openai: Arc<dyn LLMProvider>                           │
//! │  ├─ ollama: Arc<dyn LLMProvider>                           │
//! │  ├─ lmstudio: Arc<dyn LLMProvider>                         │
//! │  ├─ vscode-copilot: Arc<dyn LLMProvider>                   │
//! │  ├─ gemini: Arc<dyn LLMProvider>                           │
//! │  ├─ jina: Arc<dyn LLMProvider>                             │
//! │  ├─ azure-openai: Arc<dyn LLMProvider>                     │
//! │  └─ mock: Arc<dyn LLMProvider>                             │
//! │                                                             │
//! │  Embedding Providers:                                       │
//! │  ┌─ openai: Arc<dyn EmbeddingProvider>                     │
//! │  ├─ ollama: Arc<dyn EmbeddingProvider>                     │
//! │  ├─ lmstudio: Arc<dyn EmbeddingProvider>                   │
//! │  ├─ vscode-copilot: Arc<dyn EmbeddingProvider>             │
//! │  └─ mock: Arc<dyn EmbeddingProvider>                       │
//! │                                                             │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Providers by Category
//!
//! ## LLM Providers (Chat/Completion)
//! - **openai**: OpenAI GPT models (gpt-4o, gpt-3.5-turbo)
//! - **azure-openai**: Azure OpenAI deployment
//! - **ollama**: Local Ollama models (Llama, Mistral, etc.)
//! - **lmstudio**: OpenAI-compatible local API
//! - **vscode-copilot**: VSCode Copilot proxy
//! - **gemini**: Google Gemini API
//! - **jina**: Jina.ai API
//! - **mock**: Testing provider
//!
//! ## Embedding Providers
//! - **openai**: OpenAI text-embedding-3 family
//! - **ollama**: Local Ollama embedding models
//! - **lmstudio**: OpenAI-compatible embeddings
//! - **vscode-copilot**: Copilot API embeddings
//! - **mock**: Testing provider
//!
//! # Why Registry Pattern?
//!
//! Instead of hardcoded factory with switch statements:
//! - **Extensibility**: New providers don't require factory modification
//! - **Pluggability**: External code can register custom providers
//! - **Decoupling**: Registry doesn't depend on specific provider implementations
//! - **Testing**: Easy to mock or substitute providers
//! - **Discovery**: Can enumerate available providers
//!
//! # Example
//!
//! ```ignore
//! use edgequake_llm::ProviderRegistry;
//!
//! // Create registry with built-in providers
//! let registry = ProviderRegistry::new()?;
//!
//! // Get a provider
//! if let Some(provider) = registry.get_llm("openai") {
//!     let response = provider.complete(&request).await?;
//! }
//!
//! // List available providers
//! let providers = registry.list_llm();
//! println!("Available: {:?}", providers);
//!
//! // Register a custom provider
//! let mut registry = ProviderRegistry::new()?;
//! let custom = Arc::new(MyCustomProvider::new());
//! registry.register_llm("my_custom", custom);
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Result;
use crate::traits::{EmbeddingProvider, LLMProvider};
use crate::{GeminiProvider, JinaProvider, MockProvider, ProviderFactory, ProviderType};

/// Provider registry for managing LLM and Embedding providers
///
/// Stores instantiated providers keyed by name for O(1) lookup.
/// Supports both built-in and custom provider registration.
///
/// # Built-in Providers
///
/// The registry is initialized with all built-in providers:
/// - LLM: openai, azure-openai, ollama, lmstudio, vscode-copilot, gemini, jina, mock
/// - Embeddings: openai, ollama, lmstudio, vscode-copilot, mock
///
/// # Custom Providers
///
/// Applications can register additional providers:
/// ```ignore
/// let mut registry = ProviderRegistry::new()?;
/// registry.register_llm("custom", Arc::new(MyProvider));
/// ```
#[derive(Default)]
pub struct ProviderRegistry {
    /// Map of LLM providers by name
    llm_providers: HashMap<String, Arc<dyn LLMProvider>>,

    /// Map of Embedding providers by name
    embedding_providers: HashMap<String, Arc<dyn EmbeddingProvider>>,
}

impl ProviderRegistry {
    /// Create a new registry with all built-in providers
    ///
    /// # Errors
    ///
    /// Returns error if any built-in provider fails to initialize
    /// (e.g., API key not set for OpenAI)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let registry = ProviderRegistry::new()?;
    /// assert!(!registry.list_llm().is_empty());
    /// ```
    pub fn new() -> Result<Self> {
        let mut registry = Self::default();

        // Register LLM providers
        // Note: Some providers may fail initialization if required config is missing.
        // We only register those that can be initialized without errors.

        // OpenAI - requires OPENAI_API_KEY
        if let Ok((llm, embed)) = ProviderFactory::create(ProviderType::OpenAI) {
            registry.register_llm("openai", llm);
            registry.register_embedding("openai", embed);
        }

        // Ollama - requires OLLAMA_HOST or OLLAMA_MODEL
        if let Ok((llm, embed)) = ProviderFactory::create(ProviderType::Ollama) {
            registry.register_llm("ollama", llm);
            registry.register_embedding("ollama", embed);
        }

        // LM Studio - requires LMSTUDIO_HOST or LMSTUDIO_MODEL
        if let Ok((llm, embed)) = ProviderFactory::create(ProviderType::LMStudio) {
            registry.register_llm("lmstudio", llm);
            registry.register_embedding("lmstudio", embed);
        }

        // VSCode Copilot - usually available
        if let Ok((llm, embed)) = ProviderFactory::create(ProviderType::VsCodeCopilot) {
            registry.register_llm("vscode-copilot", llm);
            registry.register_embedding("vscode-copilot", embed);
        }

        // Gemini - requires GEMINI_API_KEY (LLM only)
        if let Ok(provider) = GeminiProvider::from_env() {
            registry.register_llm("gemini", Arc::new(provider));
        }

        // Jina - requires JINA_API_KEY (Embedding only)
        if let Ok(provider) = JinaProvider::from_env() {
            registry.register_embedding("jina", Arc::new(provider));
        }

        // Mock - always available (for testing)
        let mock = Arc::new(MockProvider::new());
        registry.register_llm("mock", mock.clone());
        registry.register_embedding("mock", mock);

        Ok(registry)
    }

    /// Register an LLM provider
    ///
    /// If a provider with the same name exists, it will be replaced.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique identifier for the provider
    /// * `provider` - The provider instance
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut registry = ProviderRegistry::new()?;
    /// registry.register_llm("custom", Arc::new(MyProvider));
    /// ```
    pub fn register_llm(&mut self, name: impl Into<String>, provider: Arc<dyn LLMProvider>) {
        self.llm_providers.insert(name.into(), provider);
    }

    /// Register an Embedding provider
    ///
    /// If a provider with the same name exists, it will be replaced.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique identifier for the provider
    /// * `provider` - The provider instance
    pub fn register_embedding(
        &mut self,
        name: impl Into<String>,
        provider: Arc<dyn EmbeddingProvider>,
    ) {
        self.embedding_providers.insert(name.into(), provider);
    }

    /// Get an LLM provider by name
    ///
    /// # Arguments
    ///
    /// * `name` - Provider name (e.g., "openai", "ollama")
    ///
    /// # Returns
    ///
    /// `Some(provider)` if found, `None` otherwise
    ///
    /// # Example
    ///
    /// ```ignore
    /// let registry = ProviderRegistry::new()?;
    /// if let Some(provider) = registry.get_llm("openai") {
    ///     let response = provider.complete(&request).await?;
    /// }
    /// ```
    pub fn get_llm(&self, name: &str) -> Option<Arc<dyn LLMProvider>> {
        self.llm_providers.get(name).cloned()
    }

    /// Get an Embedding provider by name
    ///
    /// # Arguments
    ///
    /// * `name` - Provider name (e.g., "openai", "ollama")
    ///
    /// # Returns
    ///
    /// `Some(provider)` if found, `None` otherwise
    pub fn get_embedding(&self, name: &str) -> Option<Arc<dyn EmbeddingProvider>> {
        self.embedding_providers.get(name).cloned()
    }

    /// List all registered LLM provider names
    ///
    /// # Returns
    ///
    /// Vector of provider names in arbitrary order
    ///
    /// # Example
    ///
    /// ```ignore
    /// let registry = ProviderRegistry::new()?;
    /// let providers = registry.list_llm();
    /// println!("Available LLM providers: {:?}", providers);
    /// ```
    pub fn list_llm(&self) -> Vec<String> {
        self.llm_providers.keys().cloned().collect()
    }

    /// List all registered Embedding provider names
    ///
    /// # Returns
    ///
    /// Vector of provider names in arbitrary order
    pub fn list_embedding(&self) -> Vec<String> {
        self.embedding_providers.keys().cloned().collect()
    }

    /// Check if an LLM provider is registered
    ///
    /// # Arguments
    ///
    /// * `name` - Provider name
    ///
    /// # Returns
    ///
    /// `true` if provider exists, `false` otherwise
    pub fn has_llm(&self, name: &str) -> bool {
        self.llm_providers.contains_key(name)
    }

    /// Check if an Embedding provider is registered
    ///
    /// # Arguments
    ///
    /// * `name` - Provider name
    ///
    /// # Returns
    ///
    /// `true` if provider exists, `false` otherwise
    pub fn has_embedding(&self, name: &str) -> bool {
        self.embedding_providers.contains_key(name)
    }

    /// Remove an LLM provider from the registry
    ///
    /// # Arguments
    ///
    /// * `name` - Provider name
    ///
    /// # Returns
    ///
    /// The removed provider, or `None` if not found
    pub fn remove_llm(&mut self, name: &str) -> Option<Arc<dyn LLMProvider>> {
        self.llm_providers.remove(name)
    }

    /// Remove an Embedding provider from the registry
    ///
    /// # Arguments
    ///
    /// * `name` - Provider name
    ///
    /// # Returns
    ///
    /// The removed provider, or `None` if not found
    pub fn remove_embedding(&mut self, name: &str) -> Option<Arc<dyn EmbeddingProvider>> {
        self.embedding_providers.remove(name)
    }

    /// Get the count of registered LLM providers
    pub fn llm_count(&self) -> usize {
        self.llm_providers.len()
    }

    /// Get the count of registered Embedding providers
    pub fn embedding_count(&self) -> usize {
        self.embedding_providers.len()
    }

    /// Clear all registered LLM providers
    pub fn clear_llm(&mut self) {
        self.llm_providers.clear();
    }

    /// Clear all registered Embedding providers
    pub fn clear_embedding(&mut self) {
        self.embedding_providers.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_registry_new() {
        let registry = ProviderRegistry::new();
        assert!(registry.is_ok());

        let registry = registry.unwrap();
        // Mock should always be available
        assert!(registry.has_llm("mock"));
        assert!(registry.has_embedding("mock"));
    }

    #[tokio::test]
    async fn test_get_mock_provider() {
        let registry = ProviderRegistry::new().unwrap();
        let mock = registry.get_llm("mock");
        assert!(mock.is_some());

        let embed_mock = registry.get_embedding("mock");
        assert!(embed_mock.is_some());
    }

    #[tokio::test]
    async fn test_get_nonexistent_provider() {
        let registry = ProviderRegistry::new().unwrap();
        let unknown = registry.get_llm("nonexistent");
        assert!(unknown.is_none());

        let unknown_embed = registry.get_embedding("nonexistent");
        assert!(unknown_embed.is_none());
    }

    #[tokio::test]
    async fn test_register_custom_llm_provider() {
        let mut registry = ProviderRegistry::new().unwrap();
        let mock = Arc::new(MockProvider::new());
        registry.register_llm("custom", mock);

        assert!(registry.has_llm("custom"));
        assert!(registry.get_llm("custom").is_some());
    }

    #[tokio::test]
    async fn test_register_custom_embedding_provider() {
        let mut registry = ProviderRegistry::new().unwrap();
        let mock = Arc::new(MockProvider::new());
        registry.register_embedding("custom_embed", mock);

        assert!(registry.has_embedding("custom_embed"));
        assert!(registry.get_embedding("custom_embed").is_some());
    }

    #[tokio::test]
    async fn test_list_providers() {
        let registry = ProviderRegistry::new().unwrap();

        let llm_names = registry.list_llm();
        assert!(!llm_names.is_empty());
        assert!(llm_names.contains(&"mock".to_string()));

        let embed_names = registry.list_embedding();
        assert!(!embed_names.is_empty());
        assert!(embed_names.contains(&"mock".to_string()));
    }

    #[tokio::test]
    async fn test_remove_provider() {
        let mut registry = ProviderRegistry::new().unwrap();
        let mock = Arc::new(MockProvider::new());
        registry.register_llm("to_remove", mock);

        assert!(registry.has_llm("to_remove"));
        let removed = registry.remove_llm("to_remove");
        assert!(removed.is_some());
        assert!(!registry.has_llm("to_remove"));
    }

    #[tokio::test]
    async fn test_provider_count() {
        let registry = ProviderRegistry::new().unwrap();
        assert!(registry.llm_count() > 0);
        assert!(registry.embedding_count() > 0);
    }

    #[tokio::test]
    async fn test_overwrite_provider() {
        let mut registry = ProviderRegistry::new().unwrap();
        let _original = registry.get_llm("mock").unwrap();
        let new_mock = Arc::new(MockProvider::new());
        registry.register_llm("mock", new_mock);

        let _updated = registry.get_llm("mock").unwrap();
        // Both should exist, providers are different instances
        assert!(registry.get_llm("mock").is_some());
    }

    // ---- Iteration 23: Additional registry tests ----

    #[test]
    fn test_registry_default_is_empty() {
        let registry = ProviderRegistry::default();
        assert_eq!(registry.llm_count(), 0);
        assert_eq!(registry.embedding_count(), 0);
        assert!(registry.list_llm().is_empty());
        assert!(registry.list_embedding().is_empty());
    }

    #[test]
    fn test_has_on_empty_registry() {
        let registry = ProviderRegistry::default();
        assert!(!registry.has_llm("anything"));
        assert!(!registry.has_embedding("anything"));
    }

    #[test]
    fn test_get_on_empty_registry() {
        let registry = ProviderRegistry::default();
        assert!(registry.get_llm("mock").is_none());
        assert!(registry.get_embedding("mock").is_none());
    }

    #[test]
    fn test_remove_nonexistent_llm() {
        let mut registry = ProviderRegistry::default();
        let removed = registry.remove_llm("nonexistent");
        assert!(removed.is_none());
    }

    #[test]
    fn test_remove_nonexistent_embedding() {
        let mut registry = ProviderRegistry::default();
        let removed = registry.remove_embedding("nonexistent");
        assert!(removed.is_none());
    }

    #[test]
    fn test_remove_embedding_provider() {
        let mut registry = ProviderRegistry::default();
        let mock = Arc::new(MockProvider::new());
        registry.register_embedding("emb1", mock);
        assert!(registry.has_embedding("emb1"));

        let removed = registry.remove_embedding("emb1");
        assert!(removed.is_some());
        assert!(!registry.has_embedding("emb1"));
    }

    #[test]
    fn test_clear_llm() {
        let mut registry = ProviderRegistry::default();
        let mock = Arc::new(MockProvider::new());
        registry.register_llm("a", mock.clone());
        registry.register_llm("b", mock);
        assert_eq!(registry.llm_count(), 2);

        registry.clear_llm();
        assert_eq!(registry.llm_count(), 0);
        assert!(registry.list_llm().is_empty());
    }

    #[test]
    fn test_clear_embedding() {
        let mut registry = ProviderRegistry::default();
        let mock = Arc::new(MockProvider::new());
        registry.register_embedding("x", mock.clone());
        registry.register_embedding("y", mock);
        assert_eq!(registry.embedding_count(), 2);

        registry.clear_embedding();
        assert_eq!(registry.embedding_count(), 0);
        assert!(registry.list_embedding().is_empty());
    }

    #[test]
    fn test_register_multiple_custom_providers() {
        let mut registry = ProviderRegistry::default();
        for i in 0..5 {
            let mock = Arc::new(MockProvider::new());
            registry.register_llm(format!("custom_{}", i), mock);
        }
        assert_eq!(registry.llm_count(), 5);
        for i in 0..5 {
            assert!(registry.has_llm(&format!("custom_{}", i)));
        }
    }

    #[test]
    fn test_clear_llm_does_not_affect_embedding() {
        let mut registry = ProviderRegistry::default();
        let mock = Arc::new(MockProvider::new());
        registry.register_llm("shared", mock.clone());
        registry.register_embedding("shared", mock);
        assert_eq!(registry.llm_count(), 1);
        assert_eq!(registry.embedding_count(), 1);

        registry.clear_llm();
        assert_eq!(registry.llm_count(), 0);
        assert_eq!(registry.embedding_count(), 1); // Not affected
    }

    #[test]
    fn test_clear_embedding_does_not_affect_llm() {
        let mut registry = ProviderRegistry::default();
        let mock = Arc::new(MockProvider::new());
        registry.register_llm("shared", mock.clone());
        registry.register_embedding("shared", mock);

        registry.clear_embedding();
        assert_eq!(registry.llm_count(), 1); // Not affected
        assert_eq!(registry.embedding_count(), 0);
    }

    #[test]
    fn test_get_returns_cloned_arc() {
        let mut registry = ProviderRegistry::default();
        let mock = Arc::new(MockProvider::new());
        registry.register_llm("test", mock);

        let p1 = registry.get_llm("test").unwrap();
        let p2 = registry.get_llm("test").unwrap();
        // Both Arcs point to the same allocation
        assert!(Arc::ptr_eq(&p1, &p2));
    }

    #[test]
    fn test_llm_count_after_removal() {
        let mut registry = ProviderRegistry::default();
        let mock = Arc::new(MockProvider::new());
        registry.register_llm("a", mock.clone());
        registry.register_llm("b", mock);
        assert_eq!(registry.llm_count(), 2);

        registry.remove_llm("a");
        assert_eq!(registry.llm_count(), 1);
    }
}
