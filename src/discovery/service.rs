//! `ModelDiscoveryService` — the main entry point for agent harnesses.
//!
//! Aggregates discovery across all registered providers with:
//! - Parallel discovery via `join_all`
//! - Per-provider caching with configurable TTL
//! - Graceful degradation (no panics, stale cache on error)
//! - `CapabilityFilter` query interface

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use super::cache::DiscoveryCache;
use super::search::ModelSearchQuery;
use super::traits::ModelDiscoveryProvider;
use super::types::{CapabilityFilter, DiscoveredModel};

/// Service that aggregates discovery across all providers.
///
/// # Usage
///
/// ```rust,ignore
/// let service = ModelDiscoveryService::builder().build();
/// let all = service.discover_all().await?;
///
/// let filter = CapabilityFilter {
///     requires_vision: Some(true),
///     requires_tools: Some(true),
///     min_context_length: Some(100_000),
///     ..Default::default()
/// };
/// let matches = service.find_models(&filter).await?;
/// ```
pub struct ModelDiscoveryService {
    providers: Vec<Box<dyn ModelDiscoveryProvider>>,
    cache: Arc<RwLock<DiscoveryCache>>,
}

impl ModelDiscoveryService {
    /// Create a builder for customizing the service.
    pub fn builder() -> ModelDiscoveryServiceBuilder {
        ModelDiscoveryServiceBuilder::new()
    }

    /// Create with default providers and 1-hour cache TTL.
    pub fn new() -> Self {
        Self::builder().build()
    }

    /// Register an additional discovery provider.
    pub fn register(&mut self, provider: Box<dyn ModelDiscoveryProvider>) {
        self.providers.push(provider);
    }

    /// Discover all models across all registered providers.
    /// Results are cached per-provider.
    pub async fn discover_all(&self) -> crate::error::Result<Vec<DiscoveredModel>> {
        let futures: Vec<_> = self
            .providers
            .iter()
            .map(|p| self.discover_cached(p.as_ref()))
            .collect();

        let results = futures::future::join_all(futures).await;

        let mut all = Vec::new();
        for result in results {
            match result {
                Ok(models) => all.extend(models),
                Err(e) => tracing::warn!("Discovery failed: {}", e),
            }
        }
        Ok(all)
    }

    /// Discover models for a specific provider.
    pub async fn discover_for_provider(
        &self,
        provider_id: &str,
    ) -> crate::error::Result<Vec<DiscoveredModel>> {
        let provider = self
            .providers
            .iter()
            .find(|p| p.provider_id() == provider_id);

        match provider {
            Some(p) => self.discover_cached(p.as_ref()).await,
            None => {
                tracing::warn!("Provider '{}' not registered", provider_id);
                Ok(Vec::new())
            }
        }
    }

    /// Find models matching a capability filter across all providers.
    pub async fn find_models(
        &self,
        filter: &CapabilityFilter,
    ) -> crate::error::Result<Vec<DiscoveredModel>> {
        let all = self.discover_all().await?;
        Ok(all.into_iter().filter(|m| filter.matches(m)).collect())
    }

    /// Find models for one provider that match a capability filter.
    pub async fn find_models_for_provider(
        &self,
        provider_id: &str,
        filter: &CapabilityFilter,
    ) -> crate::error::Result<Vec<DiscoveredModel>> {
        let models = self.discover_for_provider(provider_id).await?;
        Ok(models.into_iter().filter(|m| filter.matches(m)).collect())
    }

    /// Search the built-in static registry without network I/O.
    pub fn find_models_static(&self, filter: &CapabilityFilter) -> Vec<DiscoveredModel> {
        let _ = self;
        super::registry::find_static_models(filter)
    }

    /// Search models by name or ID across all providers (live discovery).
    pub async fn search_models(
        &self,
        query: &ModelSearchQuery,
    ) -> crate::error::Result<Vec<super::search::ModelSearchMatch>> {
        let all = self.discover_all().await?;
        Ok(super::search::search_models(all, query))
    }

    /// Search models by name or ID for one provider (live discovery).
    pub async fn search_models_for_provider(
        &self,
        provider_id: &str,
        query: &ModelSearchQuery,
    ) -> crate::error::Result<Vec<super::search::ModelSearchMatch>> {
        let models = self.discover_for_provider(provider_id).await?;
        Ok(super::search::search_models(models, query))
    }

    /// Search the static registry by name or ID (no network I/O).
    pub fn search_models_static(
        &self,
        query: &ModelSearchQuery,
    ) -> Vec<super::search::ModelSearchMatch> {
        let _ = self;
        super::search::search_static_models(query)
    }

    /// Resolve a model by exact ID or display name within a provider (static registry).
    pub fn lookup_model_by_name(
        &self,
        provider_id: &str,
        name_or_id: &str,
    ) -> Option<DiscoveredModel> {
        let _ = self;
        super::search::static_lookup_by_name(provider_id, name_or_id)
    }

    /// Get detailed info for a specific model.
    pub async fn get_model(
        &self,
        provider_id: &str,
        model_id: &str,
    ) -> crate::error::Result<Option<DiscoveredModel>> {
        let models = self.discover_for_provider(provider_id).await?;
        Ok(models.into_iter().find(|m| m.id == model_id))
    }

    /// Invalidate cache for a provider (force re-discovery on next call).
    pub async fn invalidate_cache(&self, provider_id: &str) {
        let mut cache = self.cache.write().await;
        cache.invalidate(provider_id);
    }

    /// Invalidate all cached data.
    pub async fn invalidate_all_caches(&self) {
        let mut cache = self.cache.write().await;
        cache.invalidate_all();
    }

    /// Get all registered provider IDs.
    pub fn provider_ids(&self) -> Vec<&str> {
        self.providers.iter().map(|p| p.provider_id()).collect()
    }

    async fn discover_cached(
        &self,
        provider: &dyn ModelDiscoveryProvider,
    ) -> crate::error::Result<Vec<DiscoveredModel>> {
        let pid = provider.provider_id();

        // Read lock: check cache
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(pid) {
                return Ok(cached.to_vec());
            }
        }

        // Cache miss: fetch from provider
        let result = provider.discover_models().await;

        match result {
            Ok(models) => {
                let mut cache = self.cache.write().await;
                cache.put(pid, models.clone());
                Ok(models)
            }
            Err(e) => {
                // Try stale cache before giving up
                let cache = self.cache.read().await;
                if let Some(stale) = cache.get_stale(pid) {
                    tracing::warn!("Discovery failed for '{}', serving stale cache: {}", pid, e);
                    Ok(stale.to_vec())
                } else {
                    Err(e)
                }
            }
        }
    }
}

impl Default for ModelDiscoveryService {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for `ModelDiscoveryService`.
pub struct ModelDiscoveryServiceBuilder {
    providers: Vec<Box<dyn ModelDiscoveryProvider>>,
    cache_ttl: Duration,
    include_defaults: bool,
}

impl ModelDiscoveryServiceBuilder {
    fn new() -> Self {
        Self {
            providers: Vec::new(),
            cache_ttl: Duration::from_secs(3600),
            include_defaults: true,
        }
    }

    /// Set default cache TTL.
    pub fn cache_ttl(mut self, ttl: Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// Disable default providers (start with empty list).
    pub fn no_defaults(mut self) -> Self {
        self.include_defaults = false;
        self
    }

    /// Add a custom discovery provider.
    pub fn provider(mut self, provider: Box<dyn ModelDiscoveryProvider>) -> Self {
        self.providers.push(provider);
        self
    }

    /// Build the service.
    pub fn build(mut self) -> ModelDiscoveryService {
        if self.include_defaults {
            let mut defaults = default_providers();
            defaults.append(&mut self.providers);
            self.providers = defaults;
        }

        let mut cache = DiscoveryCache::new(self.cache_ttl);
        // Local providers get shorter TTL (models change frequently)
        cache.set_provider_ttl("ollama", Duration::from_secs(60));
        cache.set_provider_ttl("lmstudio", Duration::from_secs(60));
        // Static providers get longer TTL
        cache.set_provider_ttl("xai", Duration::from_secs(86400));

        ModelDiscoveryService {
            providers: self.providers,
            cache: Arc::new(RwLock::new(cache)),
        }
    }
}

/// Create default discovery providers based on available env vars.
fn default_providers() -> Vec<Box<dyn ModelDiscoveryProvider>> {
    use super::providers::*;

    #[allow(unused_mut)]
    let mut providers: Vec<Box<dyn ModelDiscoveryProvider>> = vec![
        Box::new(openai::OpenAIDiscovery::new()),
        Box::new(anthropic::AnthropicDiscovery::new()),
        Box::new(gemini::GeminiDiscovery::new()),
        Box::new(ollama::OllamaDiscovery::new()),
        Box::new(lmstudio::LMStudioDiscovery::new()),
        Box::new(openrouter::OpenRouterDiscovery::new()),
        Box::new(mistral::MistralDiscovery::new()),
        Box::new(nvidia::NvidiaDiscovery::new()),
        Box::new(xai::XAIDiscovery::new()),
    ];

    #[cfg(feature = "bedrock")]
    providers.push(Box::new(bedrock::BedrockDiscovery::new()));

    providers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::types::{DiscoverySource, DiscoveryStrategy};

    struct MockProvider {
        id: &'static str,
        models: Vec<DiscoveredModel>,
    }

    #[async_trait::async_trait]
    impl ModelDiscoveryProvider for MockProvider {
        fn provider_id(&self) -> &str {
            self.id
        }
        fn discovery_strategy(&self) -> DiscoveryStrategy {
            DiscoveryStrategy::Static
        }
        async fn discover_models(&self) -> crate::error::Result<Vec<DiscoveredModel>> {
            Ok(self.models.clone())
        }
    }

    fn make_test_model(id: &str, provider: &str) -> DiscoveredModel {
        DiscoveredModel {
            id: id.into(),
            name: id.into(),
            provider: provider.into(),
            context_length: 128_000,
            max_output_tokens: 4096,
            source: DiscoverySource::StaticRegistry,
            available: true,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_service_no_defaults() {
        let svc = ModelDiscoveryService::builder()
            .no_defaults()
            .provider(Box::new(MockProvider {
                id: "test",
                models: vec![make_test_model("model-1", "test")],
            }))
            .build();

        let models = svc.discover_all().await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "model-1");
    }

    #[tokio::test]
    async fn test_service_discover_for_provider() {
        let svc = ModelDiscoveryService::builder()
            .no_defaults()
            .provider(Box::new(MockProvider {
                id: "alpha",
                models: vec![make_test_model("a1", "alpha")],
            }))
            .provider(Box::new(MockProvider {
                id: "beta",
                models: vec![make_test_model("b1", "beta")],
            }))
            .build();

        let alpha = svc.discover_for_provider("alpha").await.unwrap();
        assert_eq!(alpha.len(), 1);
        assert_eq!(alpha[0].id, "a1");

        let unknown = svc.discover_for_provider("unknown").await.unwrap();
        assert!(unknown.is_empty());
    }

    #[tokio::test]
    async fn test_service_find_models_with_filter() {
        let svc = ModelDiscoveryService::builder()
            .no_defaults()
            .provider(Box::new(MockProvider {
                id: "test",
                models: vec![
                    {
                        let mut m = make_test_model("big", "test");
                        m.context_length = 1_000_000;
                        m
                    },
                    make_test_model("small", "test"),
                ],
            }))
            .build();

        let filter = CapabilityFilter {
            min_context_length: Some(500_000),
            ..Default::default()
        };
        let results = svc.find_models(&filter).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "big");
    }

    #[tokio::test]
    async fn test_service_get_model() {
        let svc = ModelDiscoveryService::builder()
            .no_defaults()
            .provider(Box::new(MockProvider {
                id: "test",
                models: vec![make_test_model("m1", "test")],
            }))
            .build();

        assert!(svc.get_model("test", "m1").await.unwrap().is_some());
        assert!(svc.get_model("test", "m2").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_service_caching() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

        struct CountingProvider;

        #[async_trait::async_trait]
        impl ModelDiscoveryProvider for CountingProvider {
            fn provider_id(&self) -> &str {
                "counter"
            }
            fn discovery_strategy(&self) -> DiscoveryStrategy {
                DiscoveryStrategy::Static
            }
            async fn discover_models(&self) -> crate::error::Result<Vec<DiscoveredModel>> {
                CALL_COUNT.fetch_add(1, Ordering::SeqCst);
                Ok(vec![make_test_model("c1", "counter")])
            }
        }

        CALL_COUNT.store(0, Ordering::SeqCst);
        let svc = ModelDiscoveryService::builder()
            .no_defaults()
            .provider(Box::new(CountingProvider))
            .build();

        let _ = svc.discover_all().await.unwrap();
        let _ = svc.discover_all().await.unwrap();
        assert_eq!(
            CALL_COUNT.load(Ordering::SeqCst),
            1,
            "Second call should use cache"
        );
    }

    #[tokio::test]
    async fn test_service_invalidate_cache() {
        let svc = ModelDiscoveryService::builder()
            .no_defaults()
            .provider(Box::new(MockProvider {
                id: "test",
                models: vec![make_test_model("m1", "test")],
            }))
            .build();

        let _ = svc.discover_all().await.unwrap();
        svc.invalidate_cache("test").await;
        // Next call should refetch
        let models = svc.discover_all().await.unwrap();
        assert_eq!(models.len(), 1);
    }

    #[tokio::test]
    async fn test_service_provider_ids() {
        let svc = ModelDiscoveryService::builder()
            .no_defaults()
            .provider(Box::new(MockProvider {
                id: "a",
                models: vec![],
            }))
            .provider(Box::new(MockProvider {
                id: "b",
                models: vec![],
            }))
            .build();

        let ids = svc.provider_ids();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_service_concurrent_access() {
        use std::sync::Arc;

        let svc = Arc::new(
            ModelDiscoveryService::builder()
                .no_defaults()
                .provider(Box::new(MockProvider {
                    id: "concurrent",
                    models: vec![
                        make_test_model("m1", "concurrent"),
                        make_test_model("m2", "concurrent"),
                    ],
                }))
                .build(),
        );

        let mut handles = Vec::new();
        for _ in 0..10 {
            let svc_clone = Arc::clone(&svc);
            handles.push(tokio::spawn(async move {
                svc_clone.discover_all().await.unwrap()
            }));
        }

        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        for models in &results {
            assert_eq!(models.len(), 2);
            assert_eq!(models[0].id, "m1");
        }
    }
}
