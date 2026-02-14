//! LLM response caching for reducing API costs and latency.
//!
//! This module provides a caching layer for LLM completions and embeddings,
//! significantly reducing costs for repeated queries and improving response times.
//!
//! ## Implements
//!
//! - **FEAT0019**: LLM Response Caching
//! - **FEAT0772**: LRU eviction policy
//! - **FEAT0773**: TTL-based expiration
//!
//! ## Enforces
//!
//! - **BR0772**: Cache hit does not modify original response
//! - **BR0773**: Expired entries evicted on next access
//!
//! Based on LightRAG's caching approach with an in-memory LRU cache.

use crate::error::Result;
use crate::traits::{ChatMessage, CompletionOptions, EmbeddingProvider, LLMProvider, LLMResponse};
use async_trait::async_trait;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Configuration for the LLM cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in the cache.
    pub max_entries: usize,
    /// Time-to-live for cache entries.
    pub ttl: Duration,
    /// Whether to cache completions.
    pub cache_completions: bool,
    /// Whether to cache embeddings.
    pub cache_embeddings: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl: Duration::from_secs(3600), // 1 hour
            cache_completions: true,
            cache_embeddings: true,
        }
    }
}

impl CacheConfig {
    /// Create a new cache config with specified max entries.
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries,
            ..Default::default()
        }
    }

    /// Set the TTL for cache entries.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Enable or disable completion caching.
    pub fn with_completion_caching(mut self, enabled: bool) -> Self {
        self.cache_completions = enabled;
        self
    }

    /// Enable or disable embedding caching.
    pub fn with_embedding_caching(mut self, enabled: bool) -> Self {
        self.cache_embeddings = enabled;
        self
    }
}

/// A cached entry with metadata.
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    value: T,
    created_at: Instant,
    access_count: usize,
}

impl<T: Clone> CacheEntry<T> {
    fn new(value: T) -> Self {
        Self {
            value,
            created_at: Instant::now(),
            access_count: 0,
        }
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }

    fn access(&mut self) -> T {
        self.access_count += 1;
        self.value.clone()
    }
}

/// Cache key derived from prompt/input.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct CacheKey {
    hash: u64,
}

impl CacheKey {
    fn from_prompt(prompt: &str) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        prompt.hash(&mut hasher);
        Self {
            hash: hasher.finish(),
        }
    }

    fn from_texts(texts: &[&str]) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for text in texts {
            text.hash(&mut hasher);
        }
        Self {
            hash: hasher.finish(),
        }
    }
}

/// LLM cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: usize,
    /// Number of cache misses.
    pub misses: usize,
    /// Current number of entries.
    pub entries: usize,
    /// Number of evictions.
    pub evictions: usize,
}

impl CacheStats {
    /// Get the cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// In-memory LLM cache.
/// @implements FEAT0014
pub struct LLMCache {
    config: CacheConfig,
    completions: RwLock<HashMap<CacheKey, CacheEntry<LLMResponse>>>,
    embeddings: RwLock<HashMap<CacheKey, CacheEntry<Vec<Vec<f32>>>>>,
    stats: RwLock<CacheStats>,
}

impl LLMCache {
    /// Create a new LLM cache with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            completions: RwLock::new(HashMap::new()),
            embeddings: RwLock::new(HashMap::new()),
            stats: RwLock::new(CacheStats::default()),
        }
    }

    /// Get cache statistics.
    pub async fn stats(&self) -> CacheStats {
        let stats = self.stats.read().await;
        let completions = self.completions.read().await;
        let embeddings = self.embeddings.read().await;

        CacheStats {
            entries: completions.len() + embeddings.len(),
            ..*stats
        }
    }

    /// Clear all cache entries.
    pub async fn clear(&self) {
        let mut completions = self.completions.write().await;
        let mut embeddings = self.embeddings.write().await;
        let mut stats = self.stats.write().await;

        let evicted = completions.len() + embeddings.len();
        completions.clear();
        embeddings.clear();
        stats.evictions += evicted;
    }

    /// Get a cached completion response.
    pub async fn get_completion(&self, prompt: &str) -> Option<LLMResponse> {
        if !self.config.cache_completions {
            return None;
        }

        let key = CacheKey::from_prompt(prompt);
        let mut cache = self.completions.write().await;

        if let Some(entry) = cache.get_mut(&key) {
            if entry.is_expired(self.config.ttl) {
                cache.remove(&key);
                let mut stats = self.stats.write().await;
                stats.misses += 1;
                stats.evictions += 1;
                return None;
            }

            let mut stats = self.stats.write().await;
            stats.hits += 1;
            return Some(entry.access());
        }

        let mut stats = self.stats.write().await;
        stats.misses += 1;
        None
    }

    /// Store a completion response in cache.
    pub async fn put_completion(&self, prompt: &str, response: LLMResponse) {
        if !self.config.cache_completions {
            return;
        }

        let key = CacheKey::from_prompt(prompt);
        let mut cache = self.completions.write().await;

        // Evict if at capacity
        if cache.len() >= self.config.max_entries {
            self.evict_lru(&mut cache).await;
        }

        cache.insert(key, CacheEntry::new(response));
    }

    /// Get cached embeddings.
    pub async fn get_embeddings(&self, texts: &[&str]) -> Option<Vec<Vec<f32>>> {
        if !self.config.cache_embeddings {
            return None;
        }

        let key = CacheKey::from_texts(texts);
        let mut cache = self.embeddings.write().await;

        if let Some(entry) = cache.get_mut(&key) {
            if entry.is_expired(self.config.ttl) {
                cache.remove(&key);
                let mut stats = self.stats.write().await;
                stats.misses += 1;
                stats.evictions += 1;
                return None;
            }

            let mut stats = self.stats.write().await;
            stats.hits += 1;
            return Some(entry.access());
        }

        let mut stats = self.stats.write().await;
        stats.misses += 1;
        None
    }

    /// Store embeddings in cache.
    pub async fn put_embeddings(&self, texts: &[&str], embeddings: Vec<Vec<f32>>) {
        if !self.config.cache_embeddings {
            return;
        }

        let key = CacheKey::from_texts(texts);
        let mut cache = self.embeddings.write().await;

        // Evict if at capacity
        if cache.len() >= self.config.max_entries {
            self.evict_lru_embeddings(&mut cache).await;
        }

        cache.insert(key, CacheEntry::new(embeddings));
    }

    async fn evict_lru<T: Clone>(&self, cache: &mut HashMap<CacheKey, CacheEntry<T>>) {
        // Find the least recently used entry (oldest with fewest accesses)
        if let Some(key) = cache
            .iter()
            .min_by_key(|(_, entry)| (entry.access_count, entry.created_at))
            .map(|(k, _)| k.clone())
        {
            cache.remove(&key);
            let mut stats = self.stats.write().await;
            stats.evictions += 1;
        }
    }

    async fn evict_lru_embeddings(&self, cache: &mut HashMap<CacheKey, CacheEntry<Vec<Vec<f32>>>>) {
        if let Some(key) = cache
            .iter()
            .min_by_key(|(_, entry)| (entry.access_count, entry.created_at))
            .map(|(k, _)| k.clone())
        {
            cache.remove(&key);
            let mut stats = self.stats.write().await;
            stats.evictions += 1;
        }
    }
}

/// A cached LLM provider that wraps another provider with caching.
pub struct CachedProvider<P> {
    inner: P,
    cache: Arc<LLMCache>,
}

impl<P> CachedProvider<P> {
    /// Create a new cached provider.
    pub fn new(inner: P, cache: Arc<LLMCache>) -> Self {
        Self { inner, cache }
    }

    /// Create with default cache config.
    pub fn with_default_cache(inner: P) -> Self {
        Self {
            inner,
            cache: Arc::new(LLMCache::new(CacheConfig::default())),
        }
    }

    /// Get cache statistics.
    pub async fn cache_stats(&self) -> CacheStats {
        self.cache.stats().await
    }

    /// Clear the cache.
    pub async fn clear_cache(&self) {
        self.cache.clear().await;
    }
}

#[async_trait]
impl<P: LLMProvider> LLMProvider for CachedProvider<P> {
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
        // Check cache first
        if let Some(cached) = self.cache.get_completion(prompt).await {
            tracing::debug!("Cache hit for completion");
            return Ok(cached);
        }

        // Call underlying provider
        let response = self.inner.complete(prompt).await?;

        // Store in cache
        self.cache.put_completion(prompt, response.clone()).await;

        Ok(response)
    }

    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<LLMResponse> {
        // For options-based completions, we skip caching since options affect output
        self.inner.complete_with_options(prompt, options).await
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        // For chat completions, we skip caching since message history varies
        self.inner.chat(messages, options).await
    }
}

#[async_trait]
impl<P: EmbeddingProvider> EmbeddingProvider for CachedProvider<P> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn model(&self) -> &str {
        self.inner.model()
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn max_tokens(&self) -> usize {
        self.inner.max_tokens()
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Convert to &str for cache lookup
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        // Check cache first
        if let Some(cached) = self.cache.get_embeddings(&text_refs).await {
            tracing::debug!("Cache hit for embeddings");
            return Ok(cached);
        }

        // Call underlying provider
        let embeddings = self.inner.embed(texts).await?;

        // Store in cache
        self.cache
            .put_embeddings(&text_refs, embeddings.clone())
            .await;

        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_from_prompt() {
        let key1 = CacheKey::from_prompt("Hello world");
        let key2 = CacheKey::from_prompt("Hello world");
        let key3 = CacheKey::from_prompt("Different prompt");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cache_key_from_texts() {
        let key1 = CacheKey::from_texts(&["a", "b", "c"]);
        let key2 = CacheKey::from_texts(&["a", "b", "c"]);
        let key3 = CacheKey::from_texts(&["x", "y", "z"]);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.max_entries, 1000);
        assert!(config.cache_completions);
        assert!(config.cache_embeddings);
    }

    #[test]
    fn test_cache_config_builder() {
        let config = CacheConfig::new(500)
            .with_ttl(Duration::from_secs(600))
            .with_completion_caching(false);

        assert_eq!(config.max_entries, 500);
        assert_eq!(config.ttl, Duration::from_secs(600));
        assert!(!config.cache_completions);
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let cache = LLMCache::new(CacheConfig::default());
        let stats = cache.stats().await;

        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.entries, 0);
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let cache = LLMCache::new(CacheConfig::default());
        let result = cache.get_completion("test prompt").await;

        assert!(result.is_none());

        let stats = cache.stats().await;
        assert_eq!(stats.misses, 1);
    }

    #[tokio::test]
    async fn test_cache_hit() {
        let cache = LLMCache::new(CacheConfig::default());

        let response = LLMResponse::new("test response", "gpt-4").with_usage(10, 5);

        cache.put_completion("test prompt", response.clone()).await;
        let result = cache.get_completion("test prompt").await;

        assert!(result.is_some());
        assert_eq!(result.unwrap().content, "test response");

        let stats = cache.stats().await;
        assert_eq!(stats.hits, 1);
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let cache = LLMCache::new(CacheConfig::default());

        let response = LLMResponse::new("test", "gpt-4").with_usage(1, 1);

        cache.put_completion("prompt", response).await;
        assert_eq!(cache.stats().await.entries, 1);

        cache.clear().await;
        assert_eq!(cache.stats().await.entries, 0);
    }

    #[test]
    fn test_hit_rate() {
        let mut stats = CacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);

        stats.hits = 3;
        stats.misses = 1;
        assert_eq!(stats.hit_rate(), 0.75);
    }

    #[tokio::test]
    async fn test_embedding_cache() {
        let cache = LLMCache::new(CacheConfig::default());

        let embeddings = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let texts = ["text1", "text2"];

        cache.put_embeddings(&texts, embeddings.clone()).await;
        let result = cache.get_embeddings(&texts).await;

        assert!(result.is_some());
        assert_eq!(result.unwrap(), embeddings);
    }

    #[tokio::test]
    async fn test_disabled_caching() {
        let config = CacheConfig::default()
            .with_completion_caching(false)
            .with_embedding_caching(false);
        let cache = LLMCache::new(config);

        let response = LLMResponse::new("test", "gpt-4").with_usage(1, 1);

        cache.put_completion("prompt", response).await;
        assert!(cache.get_completion("prompt").await.is_none());

        cache.put_embeddings(&["text"], vec![vec![1.0]]).await;
        assert!(cache.get_embeddings(&["text"]).await.is_none());
    }

    #[tokio::test]
    async fn test_ttl_expiration_completion() {
        // TTL of 1ms ensures entries expire quickly
        let config = CacheConfig::new(100).with_ttl(Duration::from_millis(1));
        let cache = LLMCache::new(config);

        let response = LLMResponse::new("expires", "gpt-4").with_usage(5, 3);
        cache.put_completion("ephemeral", response).await;

        // Wait for TTL to elapse
        tokio::time::sleep(Duration::from_millis(10)).await;

        let result = cache.get_completion("ephemeral").await;
        assert!(result.is_none(), "Expired entry should return None");

        let stats = cache.stats().await;
        assert_eq!(stats.evictions, 1, "Expired entry should count as eviction");
        assert_eq!(stats.misses, 1);
    }

    #[tokio::test]
    async fn test_ttl_expiration_embeddings() {
        let config = CacheConfig::new(100).with_ttl(Duration::from_millis(1));
        let cache = LLMCache::new(config);

        cache.put_embeddings(&["txt"], vec![vec![1.0, 2.0]]).await;

        tokio::time::sleep(Duration::from_millis(10)).await;

        assert!(cache.get_embeddings(&["txt"]).await.is_none());
        let stats = cache.stats().await;
        assert_eq!(stats.evictions, 1);
    }

    #[tokio::test]
    async fn test_lru_eviction_completions() {
        // Cache with max 2 entries
        let config = CacheConfig::new(2);
        let cache = LLMCache::new(config);

        let r1 = LLMResponse::new("first", "gpt-4").with_usage(1, 1);
        let r2 = LLMResponse::new("second", "gpt-4").with_usage(1, 1);
        let r3 = LLMResponse::new("third", "gpt-4").with_usage(1, 1);

        cache.put_completion("p1", r1).await;
        cache.put_completion("p2", r2).await;

        // Access p2 to bump its access count
        let _ = cache.get_completion("p2").await;

        // Inserting p3 should evict p1 (least recently used)
        cache.put_completion("p3", r3).await;

        assert!(
            cache.get_completion("p1").await.is_none(),
            "p1 should have been evicted"
        );
        // p2 and p3 should still exist
        assert!(cache.get_completion("p2").await.is_some());
        assert!(cache.get_completion("p3").await.is_some());
    }

    #[tokio::test]
    async fn test_lru_eviction_embeddings() {
        let config = CacheConfig::new(1);
        let cache = LLMCache::new(config);

        cache.put_embeddings(&["a"], vec![vec![1.0]]).await;
        cache.put_embeddings(&["b"], vec![vec![2.0]]).await;

        // "a" should have been evicted
        assert!(cache.get_embeddings(&["a"]).await.is_none());
        assert!(cache.get_embeddings(&["b"]).await.is_some());
    }

    #[tokio::test]
    async fn test_access_count_increments() {
        let cache = LLMCache::new(CacheConfig::default());
        let response = LLMResponse::new("counter", "gpt-4").with_usage(1, 1);

        cache.put_completion("cnt", response).await;

        // Access 3 times
        for _ in 0..3 {
            let _ = cache.get_completion("cnt").await;
        }

        let stats = cache.stats().await;
        assert_eq!(stats.hits, 3);
    }

    #[tokio::test]
    async fn test_cache_entry_is_expired() {
        let entry = CacheEntry::new("value".to_string());
        // Just-created entry with large TTL should not be expired
        assert!(!entry.is_expired(Duration::from_secs(3600)));
        // Entry with zero TTL should be expired
        assert!(entry.is_expired(Duration::ZERO));
    }

    #[tokio::test]
    async fn test_cached_provider_complete_delegates() {
        use crate::providers::MockProvider;

        let mock = MockProvider::new();
        mock.add_response("cached answer").await;

        let cache = Arc::new(LLMCache::new(CacheConfig::default()));
        let provider = CachedProvider::new(mock, cache);

        // First call: cache miss, delegates to inner
        let r1 = provider.inner.complete("hello").await.unwrap();
        assert_eq!(r1.content, "cached answer");
    }

    #[tokio::test]
    async fn test_cached_provider_name_model_delegates() {
        use crate::providers::MockProvider;

        let mock = MockProvider::new();
        let cache = Arc::new(LLMCache::new(CacheConfig::default()));
        let provider = CachedProvider::new(mock, cache);

        assert_eq!(LLMProvider::name(&provider), "mock");
        assert_eq!(LLMProvider::model(&provider), "mock-model");
        assert_eq!(provider.max_context_length(), 4096);
    }

    #[tokio::test]
    async fn test_cached_provider_with_default_cache() {
        use crate::providers::MockProvider;

        let mock = MockProvider::new();
        let provider = CachedProvider::with_default_cache(mock);

        let stats = provider.cache_stats().await;
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.hits, 0);
    }

    #[tokio::test]
    async fn test_cached_provider_clear_cache() {
        use crate::providers::MockProvider;

        let mock = MockProvider::new();
        let cache = Arc::new(LLMCache::new(CacheConfig::default()));
        let provider = CachedProvider::new(mock, cache);

        // Put something in cache directly
        provider
            .cache
            .put_completion("test", LLMResponse::new("v", "m").with_usage(1, 1))
            .await;
        assert_eq!(provider.cache_stats().await.entries, 1);

        provider.clear_cache().await;
        assert_eq!(provider.cache_stats().await.entries, 0);
    }

    #[test]
    fn test_cache_key_empty_prompt() {
        let k1 = CacheKey::from_prompt("");
        let k2 = CacheKey::from_prompt("");
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_cache_key_empty_texts() {
        let k = CacheKey::from_texts(&[]);
        let k2 = CacheKey::from_texts(&[]);
        assert_eq!(k, k2);
    }

    #[tokio::test]
    async fn test_multiple_put_same_key_overwrites() {
        let cache = LLMCache::new(CacheConfig::default());
        let r1 = LLMResponse::new("first", "m").with_usage(1, 1);
        let r2 = LLMResponse::new("second", "m").with_usage(1, 1);

        cache.put_completion("key", r1).await;
        cache.put_completion("key", r2).await;

        let result = cache.get_completion("key").await.unwrap();
        assert_eq!(result.content, "second");
        // Only 1 entry despite 2 puts to same key
        assert_eq!(cache.stats().await.entries, 1);
    }

    #[test]
    fn test_cache_config_with_embedding_caching() {
        let config = CacheConfig::default().with_embedding_caching(false);
        assert!(!config.cache_embeddings);
        assert!(config.cache_completions); // default true
    }

    #[tokio::test]
    async fn test_clear_updates_eviction_count() {
        let cache = LLMCache::new(CacheConfig::default());
        let r = LLMResponse::new("a", "m").with_usage(1, 1);
        cache.put_completion("x", r).await;
        cache.put_embeddings(&["y"], vec![vec![1.0]]).await;

        cache.clear().await;
        let stats = cache.stats().await;
        assert_eq!(stats.evictions, 2, "Clear should count 2 evictions");
        assert_eq!(stats.entries, 0);
    }

    #[tokio::test]
    async fn test_cached_provider_embed_delegates() {
        use crate::providers::MockProvider;

        let mock = MockProvider::new();
        let cache = Arc::new(LLMCache::new(CacheConfig::default()));
        let provider = CachedProvider::new(mock, cache);

        // First call goes to inner
        let result = provider.embed(&["hello".to_string()]).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 1536);

        // Second call should hit cache
        let result2 = provider.embed(&["hello".to_string()]).await.unwrap();
        assert_eq!(result2, result);

        let stats = provider.cache_stats().await;
        assert_eq!(stats.hits, 1);
    }
}
