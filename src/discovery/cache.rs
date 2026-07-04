//! Per-provider TTL cache for discovery results.
//!
//! Thread-safe via `RwLock`. Stale data is served on network errors
//! (graceful degradation over hard failure).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::types::DiscoveredModel;

/// Cached discovery result for a single provider.
#[derive(Debug, Clone)]
struct CacheEntry {
    models: Vec<DiscoveredModel>,
    fetched_at: Instant,
}

/// Per-provider discovery cache with configurable TTL.
#[derive(Debug)]
pub struct DiscoveryCache {
    entries: HashMap<String, CacheEntry>,
    default_ttl: Duration,
    provider_ttls: HashMap<String, Duration>,
}

impl DiscoveryCache {
    pub fn new(default_ttl: Duration) -> Self {
        Self {
            entries: HashMap::new(),
            default_ttl,
            provider_ttls: HashMap::new(),
        }
    }

    /// Set a custom TTL for a specific provider.
    pub fn set_provider_ttl(&mut self, provider: &str, ttl: Duration) {
        self.provider_ttls.insert(provider.to_string(), ttl);
    }

    fn ttl_for(&self, provider: &str) -> Duration {
        self.provider_ttls
            .get(provider)
            .copied()
            .unwrap_or(self.default_ttl)
    }

    /// Get cached models if the entry exists and is within TTL.
    pub fn get(&self, provider: &str) -> Option<&[DiscoveredModel]> {
        let entry = self.entries.get(provider)?;
        let ttl = self.ttl_for(provider);
        if entry.fetched_at.elapsed() < ttl {
            Some(&entry.models)
        } else {
            None
        }
    }

    /// Get cached models even if stale (for fallback on network errors).
    pub fn get_stale(&self, provider: &str) -> Option<&[DiscoveredModel]> {
        self.entries.get(provider).map(|e| e.models.as_slice())
    }

    /// Store discovery results for a provider.
    pub fn put(&mut self, provider: &str, models: Vec<DiscoveredModel>) {
        self.entries.insert(
            provider.to_string(),
            CacheEntry {
                models,
                fetched_at: Instant::now(),
            },
        );
    }

    /// Invalidate cache for a specific provider.
    pub fn invalidate(&mut self, provider: &str) {
        self.entries.remove(provider);
    }

    /// Invalidate all cached data.
    pub fn invalidate_all(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_get_within_ttl() {
        let mut cache = DiscoveryCache::new(Duration::from_secs(3600));
        cache.put("openai", vec![]);
        assert!(cache.get("openai").is_some());
    }

    #[test]
    fn test_cache_miss_unknown_provider() {
        let cache = DiscoveryCache::new(Duration::from_secs(3600));
        assert!(cache.get("unknown").is_none());
    }

    #[test]
    fn test_cache_invalidate() {
        let mut cache = DiscoveryCache::new(Duration::from_secs(3600));
        cache.put("openai", vec![]);
        cache.invalidate("openai");
        assert!(cache.get("openai").is_none());
    }

    #[test]
    fn test_cache_stale_returns_data() {
        let mut cache = DiscoveryCache::new(Duration::from_nanos(1));
        cache.put("openai", vec![]);
        std::thread::sleep(Duration::from_millis(1));
        assert!(cache.get("openai").is_none());
        assert!(cache.get_stale("openai").is_some());
    }

    #[test]
    fn test_cache_per_provider_ttl() {
        let mut cache = DiscoveryCache::new(Duration::from_secs(3600));
        cache.set_provider_ttl("ollama", Duration::from_nanos(1));
        cache.put("ollama", vec![]);
        std::thread::sleep(Duration::from_millis(1));
        assert!(cache.get("ollama").is_none());
    }

    #[test]
    fn test_cache_invalidate_all() {
        let mut cache = DiscoveryCache::new(Duration::from_secs(3600));
        cache.put("openai", vec![]);
        cache.put("anthropic", vec![]);
        cache.invalidate_all();
        assert!(cache.get("openai").is_none());
        assert!(cache.get("anthropic").is_none());
    }
}
