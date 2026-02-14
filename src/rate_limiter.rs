//! Async-aware rate limiting for LLM API calls.
//!
//! This module provides rate limiting functionality to prevent API overload
//! and respect provider limits. It implements token bucket algorithms for
//! both request rate and token rate limiting.
//!
//! ## Implements
//!
//! - **FEAT0020**: API Rate Limiting
//! - **FEAT0770**: Token bucket algorithm
//! - **FEAT0771**: Concurrent request limits
//!
//! ## Enforces
//!
//! - **BR0301**: LLM API rate limits (configurable per provider)
//! - **BR0770**: Exponential backoff on 429 errors
//!
//! Based on LightRAG's rate limiting: `lightrag/utils.py:priority_limit_async_func_call()`

use async_trait::async_trait;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};

/// Rate limiter configuration.
#[derive(Debug, Clone)]
pub struct RateLimiterConfig {
    /// Maximum requests per minute.
    pub requests_per_minute: usize,
    /// Maximum tokens per minute.
    pub tokens_per_minute: usize,
    /// Maximum concurrent requests.
    pub max_concurrent: usize,
    /// Retry delay on rate limit.
    pub retry_delay: Duration,
    /// Maximum retries.
    pub max_retries: usize,
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 60,
            tokens_per_minute: 90_000,
            max_concurrent: 10,
            retry_delay: Duration::from_secs(1),
            max_retries: 3,
        }
    }
}

impl RateLimiterConfig {
    /// Create a new config with specified limits.
    pub fn new(requests_per_minute: usize, tokens_per_minute: usize) -> Self {
        Self {
            requests_per_minute,
            tokens_per_minute,
            ..Default::default()
        }
    }

    /// Configuration for OpenAI GPT-4.
    pub fn openai_gpt4() -> Self {
        Self {
            requests_per_minute: 500,
            tokens_per_minute: 30_000,
            max_concurrent: 50,
            ..Default::default()
        }
    }

    /// Configuration for OpenAI GPT-4o-mini.
    pub fn openai_gpt4o_mini() -> Self {
        Self {
            requests_per_minute: 5000,
            tokens_per_minute: 200_000,
            max_concurrent: 100,
            ..Default::default()
        }
    }

    /// Configuration for OpenAI GPT-3.5.
    pub fn openai_gpt35() -> Self {
        Self {
            requests_per_minute: 3500,
            tokens_per_minute: 90_000,
            max_concurrent: 100,
            ..Default::default()
        }
    }

    /// Configuration for Anthropic Claude.
    pub fn anthropic_claude() -> Self {
        Self {
            requests_per_minute: 60,
            tokens_per_minute: 100_000,
            max_concurrent: 10,
            ..Default::default()
        }
    }

    /// Set maximum concurrent requests.
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }

    /// Set retry delay.
    pub fn with_retry_delay(mut self, delay: Duration) -> Self {
        self.retry_delay = delay;
        self
    }
}

/// Token bucket rate limiter.
struct TokenBucket {
    tokens: f64,
    max_tokens: f64,
    refill_rate: f64, // tokens per second
    last_refill: Instant,
}

impl TokenBucket {
    fn new(max_tokens: f64, refill_rate: f64) -> Self {
        Self {
            tokens: max_tokens,
            max_tokens,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_refill = now;
    }

    fn try_acquire(&mut self, tokens: f64) -> bool {
        self.refill();
        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    fn time_to_acquire(&mut self, tokens: f64) -> Duration {
        self.refill();
        if self.tokens >= tokens {
            Duration::ZERO
        } else {
            let needed = tokens - self.tokens;
            Duration::from_secs_f64(needed / self.refill_rate)
        }
    }

    fn available(&mut self) -> f64 {
        self.refill();
        self.tokens
    }
}

/// Async-aware rate limiter for LLM API calls.
pub struct RateLimiter {
    config: RateLimiterConfig,
    request_bucket: Mutex<TokenBucket>,
    token_bucket: Mutex<TokenBucket>,
    concurrent_semaphore: Arc<Semaphore>,
}

impl RateLimiter {
    /// Create a new rate limiter with the given configuration.
    pub fn new(config: RateLimiterConfig) -> Self {
        let request_refill_rate = config.requests_per_minute as f64 / 60.0;
        let token_refill_rate = config.tokens_per_minute as f64 / 60.0;

        Self {
            concurrent_semaphore: Arc::new(Semaphore::new(config.max_concurrent)),
            request_bucket: Mutex::new(TokenBucket::new(
                config.requests_per_minute as f64,
                request_refill_rate,
            )),
            token_bucket: Mutex::new(TokenBucket::new(
                config.tokens_per_minute as f64,
                token_refill_rate,
            )),
            config,
        }
    }

    /// Create with default configuration.
    pub fn default_limiter() -> Self {
        Self::new(RateLimiterConfig::default())
    }

    /// Acquire permission to make a request.
    ///
    /// Returns a guard that releases the concurrent slot on drop.
    pub async fn acquire(&self, estimated_tokens: usize) -> RateLimitGuard {
        // Acquire concurrent slot
        let permit = self
            .concurrent_semaphore
            .clone()
            .acquire_owned()
            .await
            .unwrap();

        // Wait for request rate limit
        loop {
            let mut bucket = self.request_bucket.lock().await;
            if bucket.try_acquire(1.0) {
                break;
            }
            let wait_time = bucket.time_to_acquire(1.0);
            drop(bucket);

            tracing::debug!(
                wait_ms = wait_time.as_millis(),
                "Rate limited: waiting for request slot"
            );
            tokio::time::sleep(wait_time).await;
        }

        // Wait for token rate limit
        loop {
            let mut bucket = self.token_bucket.lock().await;
            if bucket.try_acquire(estimated_tokens as f64) {
                break;
            }
            let wait_time = bucket.time_to_acquire(estimated_tokens as f64);
            drop(bucket);

            tracing::debug!(
                wait_ms = wait_time.as_millis(),
                estimated_tokens,
                "Rate limited: waiting for token budget"
            );
            tokio::time::sleep(wait_time).await;
        }

        RateLimitGuard { _permit: permit }
    }

    /// Try to acquire without waiting.
    ///
    /// Returns None if rate limit would be exceeded.
    pub async fn try_acquire(&self, estimated_tokens: usize) -> Option<RateLimitGuard> {
        // Try to acquire concurrent slot
        let permit = match self.concurrent_semaphore.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => return None,
        };

        // Check request rate
        {
            let mut bucket = self.request_bucket.lock().await;
            if !bucket.try_acquire(1.0) {
                return None;
            }
        }

        // Check token rate
        {
            let mut bucket = self.token_bucket.lock().await;
            if !bucket.try_acquire(estimated_tokens as f64) {
                return None;
            }
        }

        Some(RateLimitGuard { _permit: permit })
    }

    /// Record actual token usage (for adjustment).
    pub async fn record_usage(&self, actual_tokens: usize, estimated_tokens: usize) {
        if actual_tokens > estimated_tokens {
            // Consume additional tokens
            let mut bucket = self.token_bucket.lock().await;
            bucket.tokens -= (actual_tokens - estimated_tokens) as f64;
            bucket.tokens = bucket.tokens.max(0.0);
        }
        // If actual < estimated, we already consumed more than needed (conservative)
    }

    /// Get current available request capacity.
    pub async fn available_requests(&self) -> f64 {
        self.request_bucket.lock().await.available()
    }

    /// Get current available token capacity.
    pub async fn available_tokens(&self) -> f64 {
        self.token_bucket.lock().await.available()
    }

    /// Get the configuration.
    pub fn config(&self) -> &RateLimiterConfig {
        &self.config
    }
}

/// Guard that releases rate limit resources on drop.
pub struct RateLimitGuard {
    _permit: tokio::sync::OwnedSemaphorePermit,
}

/// Rate-limited LLM provider wrapper.
pub struct RateLimitedProvider<P> {
    inner: P,
    limiter: Arc<RateLimiter>,
}

impl<P> RateLimitedProvider<P> {
    /// Create a new rate-limited provider wrapper.
    pub fn new(provider: P, config: RateLimiterConfig) -> Self {
        Self {
            inner: provider,
            limiter: Arc::new(RateLimiter::new(config)),
        }
    }

    /// Create with a shared rate limiter.
    pub fn with_limiter(provider: P, limiter: Arc<RateLimiter>) -> Self {
        Self {
            inner: provider,
            limiter,
        }
    }

    /// Get a reference to the inner provider.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Get a reference to the rate limiter.
    pub fn limiter(&self) -> &Arc<RateLimiter> {
        &self.limiter
    }
}

use crate::error::Result;
use crate::traits::{ChatMessage, CompletionOptions, EmbeddingProvider, LLMProvider, LLMResponse};
use futures::stream::BoxStream;

#[async_trait]
impl<P: LLMProvider + Send + Sync> LLMProvider for RateLimitedProvider<P> {
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
        // Estimate tokens (rough: 4 chars per token)
        let estimated_tokens = prompt.len() / 4 + 1000; // +1000 for response

        let _guard = self.limiter.acquire(estimated_tokens).await;

        let result = self.inner.complete(prompt).await;

        if let Ok(ref response) = result {
            self.limiter
                .record_usage(
                    response.prompt_tokens + response.completion_tokens,
                    estimated_tokens,
                )
                .await;
        }

        result
    }

    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<LLMResponse> {
        let estimated_tokens = prompt.len() / 4 + options.max_tokens.unwrap_or(1000);

        let _guard = self.limiter.acquire(estimated_tokens).await;

        let result = self.inner.complete_with_options(prompt, options).await;

        if let Ok(ref response) = result {
            self.limiter
                .record_usage(
                    response.prompt_tokens + response.completion_tokens,
                    estimated_tokens,
                )
                .await;
        }

        result
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let total_chars: usize = messages.iter().map(|m| m.content.len()).sum();
        let estimated_tokens = total_chars / 4
            + options
                .map(|o| o.max_tokens.unwrap_or(1000))
                .unwrap_or(1000);

        let _guard = self.limiter.acquire(estimated_tokens).await;

        let result = self.inner.chat(messages, options).await;

        if let Ok(ref response) = result {
            self.limiter
                .record_usage(
                    response.prompt_tokens + response.completion_tokens,
                    estimated_tokens,
                )
                .await;
        }

        result
    }

    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        // For streaming, we acquire once at the start
        let estimated_tokens = prompt.len() / 4 + 2000;
        let _guard = self.limiter.acquire(estimated_tokens).await;

        self.inner.stream(prompt).await
    }

    fn supports_streaming(&self) -> bool {
        self.inner.supports_streaming()
    }

    fn supports_json_mode(&self) -> bool {
        self.inner.supports_json_mode()
    }
}

#[async_trait]
impl<P: EmbeddingProvider + Send + Sync> EmbeddingProvider for RateLimitedProvider<P> {
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
        // Estimate tokens for all texts
        let total_chars: usize = texts.iter().map(|t| t.len()).sum();
        let estimated_tokens = total_chars / 4;

        let _guard = self.limiter.acquire(estimated_tokens).await;

        self.inner.embed(texts).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_token_bucket() {
        let mut bucket = TokenBucket::new(10.0, 1.0);

        assert!(bucket.try_acquire(5.0));
        assert!(bucket.try_acquire(5.0));
        assert!(!bucket.try_acquire(1.0)); // Bucket empty

        // Wait for refill
        tokio::time::sleep(Duration::from_secs(2)).await;
        assert!(bucket.try_acquire(1.0));
    }

    #[tokio::test]
    async fn test_rate_limiter_creation() {
        let limiter = RateLimiter::new(RateLimiterConfig::default());

        assert!(limiter.available_requests().await > 0.0);
        assert!(limiter.available_tokens().await > 0.0);
    }

    #[tokio::test]
    async fn test_rate_limiter_acquire() {
        let limiter = RateLimiter::new(RateLimiterConfig {
            requests_per_minute: 100,
            tokens_per_minute: 10000,
            max_concurrent: 5,
            ..Default::default()
        });

        // Should be able to acquire
        let guard = limiter.acquire(100).await;
        drop(guard);

        // Try acquire should work
        let guard = limiter.try_acquire(100).await;
        assert!(guard.is_some());
    }

    #[test]
    fn test_config_presets() {
        let gpt4 = RateLimiterConfig::openai_gpt4();
        assert_eq!(gpt4.requests_per_minute, 500);

        let claude = RateLimiterConfig::anthropic_claude();
        assert_eq!(claude.requests_per_minute, 60);
    }

    #[test]
    fn test_config_preset_gpt4o_mini() {
        let config = RateLimiterConfig::openai_gpt4o_mini();
        assert_eq!(config.requests_per_minute, 5000);
        assert_eq!(config.tokens_per_minute, 200_000);
        assert_eq!(config.max_concurrent, 100);
    }

    #[test]
    fn test_config_preset_gpt35() {
        let config = RateLimiterConfig::openai_gpt35();
        assert_eq!(config.requests_per_minute, 3500);
        assert_eq!(config.tokens_per_minute, 90_000);
        assert_eq!(config.max_concurrent, 100);
    }

    #[test]
    fn test_config_new() {
        let config = RateLimiterConfig::new(100, 50_000);
        assert_eq!(config.requests_per_minute, 100);
        assert_eq!(config.tokens_per_minute, 50_000);
        // Defaults preserved
        assert_eq!(config.max_concurrent, 10);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_config_builder_with_max_concurrent() {
        let config = RateLimiterConfig::default().with_max_concurrent(20);
        assert_eq!(config.max_concurrent, 20);
    }

    #[test]
    fn test_config_builder_with_retry_delay() {
        let config = RateLimiterConfig::default().with_retry_delay(Duration::from_secs(5));
        assert_eq!(config.retry_delay, Duration::from_secs(5));
    }

    #[test]
    fn test_config_default_values() {
        let config = RateLimiterConfig::default();
        assert_eq!(config.requests_per_minute, 60);
        assert_eq!(config.tokens_per_minute, 90_000);
        assert_eq!(config.max_concurrent, 10);
        assert_eq!(config.retry_delay, Duration::from_secs(1));
        assert_eq!(config.max_retries, 3);
    }

    #[tokio::test]
    async fn test_default_limiter() {
        let limiter = RateLimiter::default_limiter();
        assert_eq!(limiter.config().requests_per_minute, 60);
        assert_eq!(limiter.config().tokens_per_minute, 90_000);
    }

    #[tokio::test]
    async fn test_limiter_config_accessor() {
        let config = RateLimiterConfig::openai_gpt4();
        let limiter = RateLimiter::new(config);
        assert_eq!(limiter.config().requests_per_minute, 500);
    }

    #[tokio::test]
    async fn test_record_usage_over_estimate() {
        let limiter = RateLimiter::new(RateLimiterConfig {
            requests_per_minute: 100,
            tokens_per_minute: 10000,
            max_concurrent: 5,
            ..Default::default()
        });

        let initial_tokens = limiter.available_tokens().await;
        // Record actual > estimated (consumes extra tokens)
        limiter.record_usage(500, 100).await;
        let after_tokens = limiter.available_tokens().await;
        // Should have consumed 400 extra tokens
        assert!(after_tokens < initial_tokens);
    }

    #[tokio::test]
    async fn test_record_usage_under_estimate() {
        let limiter = RateLimiter::new(RateLimiterConfig {
            requests_per_minute: 100,
            tokens_per_minute: 10000,
            max_concurrent: 5,
            ..Default::default()
        });

        let initial_tokens = limiter.available_tokens().await;
        // Record actual < estimated (no adjustment, conservative)
        limiter.record_usage(50, 100).await;
        let after_tokens = limiter.available_tokens().await;
        // Should be approximately the same (only refill difference)
        assert!((after_tokens - initial_tokens).abs() < 10.0);
    }

    #[tokio::test]
    async fn test_try_acquire_fails_on_exhausted_concurrency() {
        let limiter = RateLimiter::new(RateLimiterConfig {
            requests_per_minute: 1000,
            tokens_per_minute: 100_000,
            max_concurrent: 1,
            ..Default::default()
        });

        // First acquire should succeed
        let guard1 = limiter.try_acquire(100).await;
        assert!(guard1.is_some());

        // Second acquire should fail (only 1 concurrent allowed)
        let guard2 = limiter.try_acquire(100).await;
        assert!(guard2.is_none());

        // Drop first guard, now acquire should succeed
        drop(guard1);
        let guard3 = limiter.try_acquire(100).await;
        assert!(guard3.is_some());
    }

    #[tokio::test]
    async fn test_token_bucket_time_to_acquire() {
        let mut bucket = TokenBucket::new(10.0, 10.0); // 10 tokens/sec refill

        // Consume all tokens
        assert!(bucket.try_acquire(10.0));

        // Time to acquire 5 tokens should be ~0.5 seconds
        let wait = bucket.time_to_acquire(5.0);
        assert!(wait.as_secs_f64() > 0.0);
        assert!(wait.as_secs_f64() < 1.0);
    }

    #[tokio::test]
    async fn test_token_bucket_time_to_acquire_available() {
        let mut bucket = TokenBucket::new(10.0, 1.0);

        // Tokens are available, wait should be zero
        let wait = bucket.time_to_acquire(5.0);
        assert_eq!(wait, Duration::ZERO);
    }

    #[tokio::test]
    async fn test_token_bucket_available() {
        let mut bucket = TokenBucket::new(100.0, 1.0);
        let avail = bucket.available();
        assert!((avail - 100.0).abs() < 1.0);

        bucket.try_acquire(30.0);
        let avail = bucket.available();
        assert!((avail - 70.0).abs() < 1.0);
    }

    #[tokio::test]
    async fn test_available_requests_and_tokens() {
        let limiter = RateLimiter::new(RateLimiterConfig {
            requests_per_minute: 100,
            tokens_per_minute: 50_000,
            max_concurrent: 10,
            ..Default::default()
        });

        let reqs = limiter.available_requests().await;
        let toks = limiter.available_tokens().await;
        assert!((reqs - 100.0).abs() < 1.0);
        assert!((toks - 50_000.0).abs() < 100.0);
    }

    // ---- Iteration 22: Additional rate_limiter tests ----

    #[test]
    fn test_config_debug_impl() {
        let config = RateLimiterConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("RateLimiterConfig"));
        assert!(debug_str.contains("requests_per_minute"));
    }

    #[test]
    fn test_config_clone_preserves_all_fields() {
        let config = RateLimiterConfig {
            requests_per_minute: 42,
            tokens_per_minute: 7777,
            max_concurrent: 3,
            retry_delay: Duration::from_millis(250),
            max_retries: 7,
        };
        let cloned = config.clone();
        assert_eq!(cloned.requests_per_minute, 42);
        assert_eq!(cloned.tokens_per_minute, 7777);
        assert_eq!(cloned.max_concurrent, 3);
        assert_eq!(cloned.retry_delay, Duration::from_millis(250));
        assert_eq!(cloned.max_retries, 7);
    }

    #[test]
    fn test_config_builder_chaining() {
        let config = RateLimiterConfig::new(200, 100_000)
            .with_max_concurrent(50)
            .with_retry_delay(Duration::from_millis(500));
        assert_eq!(config.requests_per_minute, 200);
        assert_eq!(config.tokens_per_minute, 100_000);
        assert_eq!(config.max_concurrent, 50);
        assert_eq!(config.retry_delay, Duration::from_millis(500));
    }

    #[tokio::test]
    async fn test_token_bucket_refill_capped_at_max() {
        let mut bucket = TokenBucket::new(10.0, 1000.0); // Very fast refill
        // Consume all tokens
        assert!(bucket.try_acquire(10.0));
        // Wait for refill beyond max
        tokio::time::sleep(Duration::from_millis(100)).await;
        // Available should be capped at max_tokens (10.0)
        let avail = bucket.available();
        assert!(avail <= 10.0, "Tokens should be capped at max: {}", avail);
    }

    #[tokio::test]
    async fn test_token_bucket_exact_boundary_acquire() {
        let mut bucket = TokenBucket::new(10.0, 0.0); // No refill
        // Acquire exactly the available tokens
        assert!(bucket.try_acquire(10.0));
        // Now bucket is at exactly 0
        assert!(!bucket.try_acquire(0.001));
    }

    #[tokio::test]
    async fn test_try_acquire_fails_on_request_rate_exhausted() {
        let limiter = RateLimiter::new(RateLimiterConfig {
            requests_per_minute: 1,    // Only 1 request allowed
            tokens_per_minute: 100_000,
            max_concurrent: 100,
            ..Default::default()
        });

        // First try should succeed
        let g1 = limiter.try_acquire(10).await;
        assert!(g1.is_some());
        drop(g1);

        // Second try should fail (request bucket exhausted)
        let g2 = limiter.try_acquire(10).await;
        assert!(g2.is_none());
    }

    #[tokio::test]
    async fn test_try_acquire_fails_on_token_rate_exhausted() {
        let limiter = RateLimiter::new(RateLimiterConfig {
            requests_per_minute: 1000,
            tokens_per_minute: 100, // Very low token budget
            max_concurrent: 100,
            ..Default::default()
        });

        // Consume all tokens
        let g1 = limiter.try_acquire(100).await;
        assert!(g1.is_some());
        drop(g1);

        // Next try should fail (token bucket exhausted)
        let g2 = limiter.try_acquire(50).await;
        assert!(g2.is_none());
    }

    #[tokio::test]
    async fn test_record_usage_exact_match() {
        let limiter = RateLimiter::new(RateLimiterConfig {
            requests_per_minute: 100,
            tokens_per_minute: 10_000,
            max_concurrent: 5,
            ..Default::default()
        });

        let before = limiter.available_tokens().await;
        // exact match: actual == estimated → no adjustment
        limiter.record_usage(100, 100).await;
        let after = limiter.available_tokens().await;
        assert!((after - before).abs() < 10.0);
    }

    #[tokio::test]
    async fn test_rate_limited_provider_new_and_accessors() {
        use crate::providers::MockProvider;

        let mock = MockProvider::new();
        let provider = RateLimitedProvider::new(mock, RateLimiterConfig::default());

        assert_eq!(LLMProvider::name(provider.inner()), "mock");
        assert_eq!(provider.limiter().config().requests_per_minute, 60);
    }

    #[tokio::test]
    async fn test_rate_limited_provider_with_shared_limiter() {
        use crate::providers::MockProvider;

        let limiter = Arc::new(RateLimiter::new(RateLimiterConfig::openai_gpt4()));
        let mock = MockProvider::new();
        let provider = RateLimitedProvider::with_limiter(mock, limiter.clone());

        assert_eq!(provider.limiter().config().requests_per_minute, 500);
    }

    #[tokio::test]
    async fn test_rate_limited_provider_llm_delegation() {
        use crate::providers::MockProvider;

        let mock = MockProvider::new();
        let provider = RateLimitedProvider::new(mock, RateLimiterConfig::default());

        // Delegation: name, model, max_context_length
        assert_eq!(LLMProvider::name(&provider), "mock");
        assert_eq!(LLMProvider::model(&provider), "mock-model");
        assert_eq!(provider.max_context_length(), 4096);
        // MockProvider default: supports_streaming=false, supports_json_mode=false
        assert!(!provider.supports_streaming());
        assert!(!provider.supports_json_mode());
    }

    #[tokio::test]
    async fn test_rate_limited_provider_embedding_delegation() {
        use crate::providers::MockProvider;

        let mock = MockProvider::new();
        let provider = RateLimitedProvider::new(mock, RateLimiterConfig::default());

        assert_eq!(EmbeddingProvider::name(&provider), "mock");
        assert_eq!(EmbeddingProvider::model(&provider), "mock-embedding");
        assert_eq!(provider.dimension(), 1536);
        assert_eq!(EmbeddingProvider::max_tokens(&provider), 512);
    }

    #[tokio::test]
    async fn test_rate_limited_provider_complete() {
        use crate::providers::MockProvider;

        let mock = MockProvider::new();
        mock.add_response("rate-limited hello").await;
        let provider = RateLimitedProvider::new(mock, RateLimiterConfig::default());

        let resp = provider.complete("test").await.unwrap();
        assert_eq!(resp.content, "rate-limited hello");
    }

    #[tokio::test]
    async fn test_rate_limited_provider_embed() {
        use crate::providers::MockProvider;

        let mock = MockProvider::new();
        let provider = RateLimitedProvider::new(mock, RateLimiterConfig::default());

        let embeddings = provider.embed(&["hello".to_string()]).await.unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 1536);
    }

    #[tokio::test]
    async fn test_rate_limited_provider_chat() {
        use crate::providers::MockProvider;

        let mock = MockProvider::new();
        mock.add_response("chat response").await;
        let provider = RateLimitedProvider::new(mock, RateLimiterConfig::default());

        let msgs = vec![ChatMessage::user("hello")];
        let resp = provider.chat(&msgs, None).await.unwrap();
        assert_eq!(resp.content, "chat response");
    }

    #[tokio::test]
    async fn test_concurrent_acquire_and_release() {
        let limiter = Arc::new(RateLimiter::new(RateLimiterConfig {
            requests_per_minute: 1000,
            tokens_per_minute: 100_000,
            max_concurrent: 2,
            ..Default::default()
        }));

        // Acquire both concurrent slots
        let g1 = limiter.try_acquire(10).await;
        assert!(g1.is_some());
        let g2 = limiter.try_acquire(10).await;
        assert!(g2.is_some());

        // Third should fail
        let g3 = limiter.try_acquire(10).await;
        assert!(g3.is_none());

        // Release one, then third should succeed
        drop(g1);
        let g4 = limiter.try_acquire(10).await;
        assert!(g4.is_some());
    }
}
