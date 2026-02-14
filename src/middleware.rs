//! LLM Provider Middleware System (OODA-125)
//!
//! Provides middleware for LLM provider calls, enabling cross-cutting
//! concerns like logging, metrics, cost tracking, and auditing.
//!
//! # Architecture
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    LLM Request Pipeline                          │
//! │                                                                  │
//! │   Messages  ──►  [Middleware 1]  ──►  [Middleware 2]  ──►  ...  │
//! │                       │                    │                     │
//! │                    before()             before()                 │
//! │                       │                    │                     │
//! │                       └────────┬───────────┘                     │
//! │                                ▼                                 │
//! │                   ┌────────────────────────┐                     │
//! │                   │  LLMProvider.chat()    │                     │
//! │                   └────────────────────────┘                     │
//! │                                │                                 │
//! │                       ┌────────┴───────────┐                     │
//! │                       │                    │                     │
//! │                    after()              after()                  │
//! │                       │                    │                     │
//! │   LLMResponse    ◄──  [Middleware 1]  ◄──  [Middleware 2]  ◄──  │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//! ```ignore
//! use edgequake_llm::middleware::{LLMMiddleware, LoggingLLMMiddleware, MetricsLLMMiddleware};
//!
//! let mut stack = LLMMiddlewareStack::new();
//! stack.add(Arc::new(LoggingLLMMiddleware::new()));
//! stack.add(Arc::new(MetricsLLMMiddleware::new()));
//!
//! // Execute with middleware
//! stack.before(&request).await?;
//! let response = provider.chat(&messages, options).await?;
//! stack.after(&request, &response, duration_ms).await?;
//! ```

use async_trait::async_trait;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{debug, info, trace};

use crate::error::Result;
use crate::model_config::ModelCost;
use crate::traits::{ChatMessage, CompletionOptions, LLMResponse, ToolDefinition};

// ============================================================================
// Request Type
// ============================================================================

/// Wrapper for LLM request data passed to middleware.
#[derive(Debug, Clone)]
pub struct LLMRequest {
    /// Chat messages for the request.
    pub messages: Vec<ChatMessage>,

    /// Optional tools for function calling.
    pub tools: Option<Vec<ToolDefinition>>,

    /// Optional completion options.
    pub options: Option<CompletionOptions>,

    /// Provider name.
    pub provider: String,

    /// Model name.
    pub model: String,
}

impl LLMRequest {
    /// Create a new LLM request.
    pub fn new(
        messages: Vec<ChatMessage>,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            messages,
            tools: None,
            options: None,
            provider: provider.into(),
            model: model.into(),
        }
    }

    /// Add tools to the request.
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Add options to the request.
    pub fn with_options(mut self, options: CompletionOptions) -> Self {
        self.options = Some(options);
        self
    }

    /// Get the number of messages.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Get the number of tools.
    pub fn tool_count(&self) -> usize {
        self.tools.as_ref().map(|t| t.len()).unwrap_or(0)
    }
}

// ============================================================================
// Middleware Trait
// ============================================================================

/// Middleware for intercepting LLM provider calls.
///
/// Implement this trait to add cross-cutting concerns to all LLM calls.
/// Middlewares are executed in registration order for `before()` and
/// reverse order for `after()`.
#[async_trait]
pub trait LLMMiddleware: Send + Sync {
    /// Middleware name for debugging and logging.
    fn name(&self) -> &str;

    /// Called before LLM request is sent.
    ///
    /// Use for:
    /// - Logging the request
    /// - Validating input
    /// - Recording start time
    ///
    /// Return `Ok(())` to continue, or `Err` to abort the request.
    async fn before(&self, request: &LLMRequest) -> Result<()> {
        let _ = request;
        Ok(())
    }

    /// Called after LLM response is received.
    ///
    /// Use for:
    /// - Logging the response
    /// - Recording metrics
    /// - Cost tracking
    async fn after(
        &self,
        request: &LLMRequest,
        response: &LLMResponse,
        duration_ms: u64,
    ) -> Result<()> {
        let _ = (request, response, duration_ms);
        Ok(())
    }
}

// ============================================================================
// Middleware Stack
// ============================================================================

/// Stack of middlewares to execute in order.
#[derive(Default)]
pub struct LLMMiddlewareStack {
    middlewares: Vec<Arc<dyn LLMMiddleware>>,
}

impl LLMMiddlewareStack {
    /// Create a new empty middleware stack.
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
        }
    }

    /// Add a middleware to the stack.
    pub fn add(&mut self, middleware: Arc<dyn LLMMiddleware>) {
        self.middlewares.push(middleware);
    }

    /// Get the number of middlewares.
    pub fn len(&self) -> usize {
        self.middlewares.len()
    }

    /// Check if the stack is empty.
    pub fn is_empty(&self) -> bool {
        self.middlewares.is_empty()
    }

    /// Execute all before hooks in registration order.
    pub async fn before(&self, request: &LLMRequest) -> Result<()> {
        for middleware in &self.middlewares {
            middleware.before(request).await?;
        }
        Ok(())
    }

    /// Execute all after hooks in reverse order.
    pub async fn after(
        &self,
        request: &LLMRequest,
        response: &LLMResponse,
        duration_ms: u64,
    ) -> Result<()> {
        for middleware in self.middlewares.iter().rev() {
            middleware.after(request, response, duration_ms).await?;
        }
        Ok(())
    }
}

// ============================================================================
// Built-in Middleware Implementations
// ============================================================================

/// Logging middleware that logs LLM requests and responses.
pub struct LoggingLLMMiddleware {
    /// Log level for before hooks.
    log_level: LogLevel,
}

/// Log level for logging middleware.
#[derive(Debug, Clone, Copy, Default)]
pub enum LogLevel {
    /// Minimal logging (request/response summary).
    #[default]
    Info,
    /// Detailed logging (includes message previews).
    Debug,
    /// Full logging (complete messages and responses).
    Trace,
}

impl LoggingLLMMiddleware {
    /// Create a new logging middleware with default settings.
    pub fn new() -> Self {
        Self {
            log_level: LogLevel::Info,
        }
    }

    /// Create a logging middleware with specified log level.
    pub fn with_level(level: LogLevel) -> Self {
        Self { log_level: level }
    }
}

impl Default for LoggingLLMMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMMiddleware for LoggingLLMMiddleware {
    fn name(&self) -> &str {
        "logging"
    }

    async fn before(&self, request: &LLMRequest) -> Result<()> {
        match self.log_level {
            LogLevel::Info => {
                info!(
                    provider = %request.provider,
                    model = %request.model,
                    messages = request.message_count(),
                    tools = request.tool_count(),
                    "[LLM] Request"
                );
            }
            LogLevel::Debug => {
                let last_msg = request.messages.last().map(|m| {
                    let preview = if m.content.chars().count() > 100 {
                        let truncated: String = m.content.chars().take(97).collect();
                        format!("{}...", truncated)
                    } else {
                        m.content.clone()
                    };
                    format!("[{:?}] {}", m.role, preview)
                });
                debug!(
                    provider = %request.provider,
                    model = %request.model,
                    messages = request.message_count(),
                    tools = request.tool_count(),
                    last_message = ?last_msg,
                    "[LLM] Request"
                );
            }
            LogLevel::Trace => {
                trace!(
                    provider = %request.provider,
                    model = %request.model,
                    messages = ?request.messages,
                    "[LLM] Full request"
                );
            }
        }
        Ok(())
    }

    async fn after(
        &self,
        request: &LLMRequest,
        response: &LLMResponse,
        duration_ms: u64,
    ) -> Result<()> {
        match self.log_level {
            LogLevel::Info => {
                info!(
                    model = %request.model,
                    tokens = response.total_tokens,
                    duration_ms = duration_ms,
                    finish_reason = ?response.finish_reason,
                    "[LLM] Response"
                );
            }
            LogLevel::Debug => {
                let preview = if response.content.chars().count() > 200 {
                    let truncated: String = response.content.chars().take(197).collect();
                    format!("{}...", truncated)
                } else {
                    response.content.clone()
                };
                debug!(
                    model = %request.model,
                    tokens = response.total_tokens,
                    duration_ms = duration_ms,
                    tool_calls = response.tool_calls.len(),
                    content_preview = %preview,
                    "[LLM] Response"
                );
            }
            LogLevel::Trace => {
                trace!(
                    model = %request.model,
                    response = ?response,
                    "[LLM] Full response"
                );
            }
        }
        Ok(())
    }
}

/// Metrics middleware that tracks LLM usage statistics.
///
/// OODA-22: Extended with cache metrics for context engineering visibility.
/// Tracks cache hit tokens to measure effectiveness of append-only context
/// and deterministic serialization patterns.
pub struct MetricsLLMMiddleware {
    /// Total requests made.
    pub total_requests: AtomicU64,
    /// Total tokens used (prompt + completion).
    pub total_tokens: AtomicU64,
    /// Total prompt tokens.
    pub prompt_tokens: AtomicU64,
    /// Total completion tokens.
    pub completion_tokens: AtomicU64,
    /// Total time spent in milliseconds.
    pub total_time_ms: AtomicU64,
    /// Number of requests with tool calls.
    pub tool_call_requests: AtomicU64,
    /// Total cache hit tokens across all requests (OODA-22).
    /// WHY: Track effectiveness of context engineering patterns.
    /// High values indicate good KV-cache utilization (10x cost savings).
    pub cache_hit_tokens: AtomicU64,
    /// Number of requests that had cache hits (OODA-22).
    /// WHY: Distinguish between "no cache support" vs "cache miss".
    pub requests_with_cache: AtomicU64,
}

impl MetricsLLMMiddleware {
    /// Create a new metrics middleware.
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            total_tokens: AtomicU64::new(0),
            prompt_tokens: AtomicU64::new(0),
            completion_tokens: AtomicU64::new(0),
            total_time_ms: AtomicU64::new(0),
            tool_call_requests: AtomicU64::new(0),
            cache_hit_tokens: AtomicU64::new(0),
            requests_with_cache: AtomicU64::new(0),
        }
    }

    /// Get the total number of requests.
    pub fn get_total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    /// Get the total tokens used.
    pub fn get_total_tokens(&self) -> u64 {
        self.total_tokens.load(Ordering::Relaxed)
    }

    /// Get the average latency in milliseconds.
    pub fn get_average_latency_ms(&self) -> f64 {
        let requests = self.total_requests.load(Ordering::Relaxed);
        if requests == 0 {
            0.0
        } else {
            self.total_time_ms.load(Ordering::Relaxed) as f64 / requests as f64
        }
    }

    /// Get the total cache hit tokens (OODA-22).
    pub fn get_cache_hit_tokens(&self) -> u64 {
        self.cache_hit_tokens.load(Ordering::Relaxed)
    }

    /// Get the cache hit rate as a percentage (OODA-22).
    ///
    /// # Returns
    /// Percentage (0.0 - 100.0) of prompt tokens that were cache hits.
    /// Returns 0.0 if no prompt tokens were used.
    pub fn get_cache_hit_rate(&self) -> f64 {
        let prompt = self.prompt_tokens.load(Ordering::Relaxed);
        if prompt == 0 {
            0.0
        } else {
            (self.cache_hit_tokens.load(Ordering::Relaxed) as f64 / prompt as f64) * 100.0
        }
    }

    /// Get all metrics as a summary.
    pub fn get_summary(&self) -> MetricsSummary {
        MetricsSummary {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            total_tokens: self.total_tokens.load(Ordering::Relaxed),
            prompt_tokens: self.prompt_tokens.load(Ordering::Relaxed),
            completion_tokens: self.completion_tokens.load(Ordering::Relaxed),
            total_time_ms: self.total_time_ms.load(Ordering::Relaxed),
            tool_call_requests: self.tool_call_requests.load(Ordering::Relaxed),
            cache_hit_tokens: self.cache_hit_tokens.load(Ordering::Relaxed),
            requests_with_cache: self.requests_with_cache.load(Ordering::Relaxed),
        }
    }
}

impl Default for MetricsLLMMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of LLM metrics.
///
/// OODA-22: Extended with cache metrics for context engineering visibility.
#[derive(Debug, Clone, Default)]
pub struct MetricsSummary {
    /// Total requests made.
    pub total_requests: u64,
    /// Total tokens used.
    pub total_tokens: u64,
    /// Total prompt tokens.
    pub prompt_tokens: u64,
    /// Total completion tokens.
    pub completion_tokens: u64,
    /// Total time in milliseconds.
    pub total_time_ms: u64,
    /// Requests with tool calls.
    pub tool_call_requests: u64,
    /// Total cache hit tokens (OODA-22).
    pub cache_hit_tokens: u64,
    /// Requests with cache hits (OODA-22).
    pub requests_with_cache: u64,
}

/// Builder for MetricsSummary (OODA-36).
///
/// Provides fluent API for constructing summaries with sensible defaults.
/// Particularly useful for tests where only specific fields need values.
///
/// # Example
/// ```ignore
/// let summary = MetricsSummaryBuilder::new()
///     .requests(10)
///     .prompt_tokens(4000)
///     .completion_tokens(1000)
///     .cache_hit_tokens(3200)
///     .time_ms(1500)
///     .build();
/// assert_eq!(summary.cache_hit_rate(), 80.0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MetricsSummaryBuilder {
    inner: MetricsSummary,
}

impl MetricsSummaryBuilder {
    /// Create a new builder with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set total requests.
    pub fn requests(mut self, n: u64) -> Self {
        self.inner.total_requests = n;
        self
    }

    /// Set prompt (input) tokens.
    pub fn prompt_tokens(mut self, n: u64) -> Self {
        self.inner.prompt_tokens = n;
        self.inner.total_tokens = self.inner.prompt_tokens + self.inner.completion_tokens;
        self
    }

    /// Set completion (output) tokens.
    pub fn completion_tokens(mut self, n: u64) -> Self {
        self.inner.completion_tokens = n;
        self.inner.total_tokens = self.inner.prompt_tokens + self.inner.completion_tokens;
        self
    }

    /// Set cache hit tokens.
    pub fn cache_hit_tokens(mut self, n: u64) -> Self {
        self.inner.cache_hit_tokens = n;
        self.inner.requests_with_cache = if n > 0 { 1 } else { 0 };
        self
    }

    /// Set total time in milliseconds.
    pub fn time_ms(mut self, ms: u64) -> Self {
        self.inner.total_time_ms = ms;
        self
    }

    /// Set tool call requests count.
    pub fn tool_calls(mut self, n: u64) -> Self {
        self.inner.tool_call_requests = n;
        self
    }

    /// Set requests with cache hits.
    pub fn requests_with_cache(mut self, n: u64) -> Self {
        self.inner.requests_with_cache = n;
        self
    }

    /// Build the MetricsSummary.
    pub fn build(self) -> MetricsSummary {
        self.inner
    }
}

impl MetricsSummary {
    /// Get average latency in milliseconds.
    pub fn average_latency_ms(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.total_time_ms as f64 / self.total_requests as f64
        }
    }

    /// Get average tokens per request.
    pub fn average_tokens_per_request(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.total_tokens as f64 / self.total_requests as f64
        }
    }

    /// Total tokens processed per second (OODA-34).
    ///
    /// Calculates overall throughput including both input and output tokens.
    /// Useful for comparing total processing capacity across sessions.
    ///
    /// WHY: Standard metric for LLM throughput monitoring and benchmarking.
    ///
    /// # Returns
    /// Tokens per second (0.0 if no time elapsed).
    pub fn tokens_per_second(&self) -> f64 {
        if self.total_time_ms == 0 {
            return 0.0;
        }
        (self.total_tokens as f64) / (self.total_time_ms as f64 / 1000.0)
    }

    /// Output tokens generated per second (OODA-34).
    ///
    /// Calculates completion throughput, which is the key metric for
    /// generation speed comparisons between models and providers.
    ///
    /// WHY: Output tokens/sec is the industry standard for LLM speed benchmarks.
    ///
    /// # Returns
    /// Completion tokens per second (0.0 if no time elapsed).
    pub fn output_tokens_per_second(&self) -> f64 {
        if self.total_time_ms == 0 {
            return 0.0;
        }
        (self.completion_tokens as f64) / (self.total_time_ms as f64 / 1000.0)
    }

    /// Token efficiency ratio (OODA-35).
    ///
    /// Measures productive output relative to input context.
    /// Higher values indicate more efficient prompting.
    ///
    /// WHY: Low efficiency may indicate context bloat or
    /// overly verbose system prompts.
    ///
    /// # Returns
    /// Percentage: (completion_tokens / prompt_tokens) * 100
    ///
    /// # Interpretation
    /// - <5%: Heavy context, minimal output
    /// - 5-15%: Normal for complex coding tasks
    /// - 15-30%: Efficient prompting
    /// - >30%: Very efficient (simple tasks)
    pub fn token_efficiency(&self) -> f64 {
        if self.prompt_tokens == 0 {
            return 0.0;
        }
        (self.completion_tokens as f64 / self.prompt_tokens as f64) * 100.0
    }

    /// Cache hit rate as percentage of prompt tokens (OODA-22).
    ///
    /// # Returns
    /// Percentage (0.0 - 100.0) of prompt tokens that were cache hits.
    /// Returns 0.0 if no prompt tokens were used.
    ///
    /// # Example
    /// ```ignore
    /// let summary = metrics.get_summary();
    /// println!("Cache hit rate: {:.1}%", summary.cache_hit_rate());
    /// ```
    pub fn cache_hit_rate(&self) -> f64 {
        if self.prompt_tokens == 0 {
            0.0
        } else {
            (self.cache_hit_tokens as f64 / self.prompt_tokens as f64) * 100.0
        }
    }

    /// Percentage of requests that utilized cache (OODA-22).
    ///
    /// # Returns
    /// Percentage (0.0 - 100.0) of requests that had cache hits.
    pub fn cache_utilization(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.requests_with_cache as f64 / self.total_requests as f64) * 100.0
        }
    }

    /// Estimated cost savings from cache hits (OODA-30).
    ///
    /// Calculates the savings based on the difference between cached and
    /// uncached token costs. Anthropic Claude cached tokens cost ~0.1x of
    /// uncached tokens, so savings = cache_hit_tokens * 0.9 * cost_per_1k / 1000.
    ///
    /// # Arguments
    /// * `cost_per_1k_prompt` - Cost per 1000 uncached prompt tokens (e.g., 0.003 for $0.003/1k)
    ///
    /// # Returns
    /// Estimated savings in the same currency as cost_per_1k_prompt.
    ///
    /// # Example
    /// ```ignore
    /// let summary = metrics.get_summary();
    /// let savings = summary.estimated_savings(0.003); // $0.003/1k tokens
    /// println!("Estimated savings: ${:.4}", savings);
    /// ```
    pub fn estimated_savings(&self, cost_per_1k_prompt: f64) -> f64 {
        // Cached tokens cost ~10% of uncached tokens (90% savings)
        let savings_rate = 0.9;
        (self.cache_hit_tokens as f64 / 1000.0) * cost_per_1k_prompt * savings_rate
    }

    /// Estimated total cost (OODA-30).
    ///
    /// Calculates total cost based on cached and uncached token usage.
    ///
    /// # Arguments
    /// * `cost_per_1k_prompt` - Cost per 1000 prompt tokens
    /// * `cost_per_1k_completion` - Cost per 1000 completion tokens
    ///
    /// # Returns
    /// Estimated total cost.
    pub fn estimated_cost(&self, cost_per_1k_prompt: f64, cost_per_1k_completion: f64) -> f64 {
        // Uncached prompt tokens = prompt_tokens - cache_hit_tokens
        let uncached_prompt = self.prompt_tokens.saturating_sub(self.cache_hit_tokens);

        // Cached tokens cost 10% of uncached
        let cached_cost = (self.cache_hit_tokens as f64 / 1000.0) * cost_per_1k_prompt * 0.1;
        let uncached_cost = (uncached_prompt as f64 / 1000.0) * cost_per_1k_prompt;
        let completion_cost = (self.completion_tokens as f64 / 1000.0) * cost_per_1k_completion;

        cached_cost + uncached_cost + completion_cost
    }

    /// Estimated savings using ModelCost pricing (OODA-31).
    ///
    /// Convenience method that extracts pricing from a ModelCost struct.
    /// WHY: Single source of truth for model pricing improves accuracy.
    ///
    /// # Arguments
    /// * `cost` - ModelCost containing input_per_1k pricing
    ///
    /// # Returns
    /// Estimated savings from cache hits in USD.
    pub fn estimated_savings_for_model(&self, cost: &ModelCost) -> f64 {
        self.estimated_savings(cost.input_per_1k)
    }

    /// Estimated cost using ModelCost pricing (OODA-31).
    ///
    /// Convenience method that extracts pricing from a ModelCost struct.
    /// WHY: Single source of truth for model pricing improves accuracy.
    ///
    /// # Arguments
    /// * `cost` - ModelCost containing input_per_1k and output_per_1k pricing
    ///
    /// # Returns
    /// Estimated total cost in USD.
    pub fn estimated_cost_for_model(&self, cost: &ModelCost) -> f64 {
        self.estimated_cost(cost.input_per_1k, cost.output_per_1k)
    }
}

/// Display implementation for MetricsSummary (OODA-33).
///
/// Provides compact single-line format for logging:
/// `reqs=10 tokens=5000/1000/4000 cache=80.0% latency=150ms tps=66.7`
///
/// WHY: Standard Display trait enables easy integration with
/// logging frameworks, debugging output, and tracing.
impl fmt::Display for MetricsSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "reqs={} tokens={}/{}/{} cache={:.1}% latency={:.0}ms tps={:.1}",
            self.total_requests,
            self.prompt_tokens,
            self.completion_tokens,
            self.cache_hit_tokens,
            self.cache_hit_rate(),
            self.average_latency_ms(),
            self.output_tokens_per_second()
        )
    }
}

#[async_trait]
impl LLMMiddleware for MetricsLLMMiddleware {
    fn name(&self) -> &str {
        "metrics"
    }

    async fn before(&self, _request: &LLMRequest) -> Result<()> {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    async fn after(
        &self,
        _request: &LLMRequest,
        response: &LLMResponse,
        duration_ms: u64,
    ) -> Result<()> {
        self.total_tokens
            .fetch_add(response.total_tokens as u64, Ordering::Relaxed);
        self.prompt_tokens
            .fetch_add(response.prompt_tokens as u64, Ordering::Relaxed);
        self.completion_tokens
            .fetch_add(response.completion_tokens as u64, Ordering::Relaxed);
        self.total_time_ms.fetch_add(duration_ms, Ordering::Relaxed);

        if !response.tool_calls.is_empty() {
            self.tool_call_requests.fetch_add(1, Ordering::Relaxed);
        }

        // OODA-22: Track cache hit tokens for context engineering visibility
        // WHY: Enables measurement of KV-cache effectiveness.
        // High cache hit rates (>80%) indicate successful context engineering.
        if let Some(cache_hits) = response.cache_hit_tokens {
            self.cache_hit_tokens
                .fetch_add(cache_hits as u64, Ordering::Relaxed);
            if cache_hits > 0 {
                self.requests_with_cache.fetch_add(1, Ordering::Relaxed);
            }
        }

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_request() -> LLMRequest {
        LLMRequest::new(
            vec![ChatMessage::user("Hello")],
            "test-provider",
            "test-model",
        )
    }

    fn create_test_response() -> LLMResponse {
        LLMResponse::new("Hello back!", "test-model").with_usage(10, 5)
    }

    #[tokio::test]
    async fn test_empty_middleware_stack() {
        let stack = LLMMiddlewareStack::new();
        assert!(stack.is_empty());

        let request = create_test_request();
        let response = create_test_response();

        // Should not fail with empty stack
        stack.before(&request).await.unwrap();
        stack.after(&request, &response, 100).await.unwrap();
    }

    #[tokio::test]
    async fn test_middleware_stack_with_logging() {
        let mut stack = LLMMiddlewareStack::new();
        stack.add(Arc::new(LoggingLLMMiddleware::new()));

        assert_eq!(stack.len(), 1);

        let request = create_test_request();
        let response = create_test_response();

        stack.before(&request).await.unwrap();
        stack.after(&request, &response, 100).await.unwrap();
    }

    #[tokio::test]
    async fn test_metrics_middleware() {
        let metrics = Arc::new(MetricsLLMMiddleware::new());
        let mut stack = LLMMiddlewareStack::new();
        stack.add(metrics.clone());

        let request = create_test_request();
        let response = create_test_response();

        stack.before(&request).await.unwrap();
        stack.after(&request, &response, 150).await.unwrap();

        assert_eq!(metrics.get_total_requests(), 1);
        assert_eq!(metrics.get_total_tokens(), 15);
        assert_eq!(metrics.get_average_latency_ms(), 150.0);
    }

    #[tokio::test]
    async fn test_multiple_middlewares() {
        let metrics = Arc::new(MetricsLLMMiddleware::new());
        let mut stack = LLMMiddlewareStack::new();
        stack.add(Arc::new(LoggingLLMMiddleware::new()));
        stack.add(metrics.clone());

        assert_eq!(stack.len(), 2);

        let request = create_test_request();
        let response = create_test_response();

        stack.before(&request).await.unwrap();
        stack.after(&request, &response, 200).await.unwrap();

        assert_eq!(metrics.get_total_requests(), 1);
    }

    #[tokio::test]
    async fn test_metrics_summary() {
        let metrics = MetricsLLMMiddleware::new();

        let request = create_test_request();
        let response = create_test_response();

        metrics.before(&request).await.unwrap();
        metrics.after(&request, &response, 100).await.unwrap();

        metrics.before(&request).await.unwrap();
        metrics.after(&request, &response, 200).await.unwrap();

        let summary = metrics.get_summary();
        assert_eq!(summary.total_requests, 2);
        assert_eq!(summary.total_tokens, 30);
        assert_eq!(summary.average_latency_ms(), 150.0);
        assert_eq!(summary.average_tokens_per_request(), 15.0);
    }

    #[test]
    fn test_llm_request_builder() {
        let request = LLMRequest::new(vec![ChatMessage::user("Test")], "openai", "gpt-4")
            .with_options(CompletionOptions::with_temperature(0.7));

        assert_eq!(request.provider, "openai");
        assert_eq!(request.model, "gpt-4");
        assert_eq!(request.message_count(), 1);
        assert_eq!(request.tool_count(), 0);
        assert!(request.options.is_some());
    }

    // ========================================================================
    // OODA-22: Cache Metrics Tests
    // ========================================================================

    #[tokio::test]
    async fn test_cache_metrics_tracking() {
        let metrics = Arc::new(MetricsLLMMiddleware::new());
        let mut stack = LLMMiddlewareStack::new();
        stack.add(metrics.clone());

        let request = create_test_request();

        // Response with cache hits
        let response = LLMResponse::new("Hello", "test-model")
            .with_usage(100, 20)
            .with_cache_hit_tokens(80);

        stack.before(&request).await.unwrap();
        stack.after(&request, &response, 100).await.unwrap();

        assert_eq!(metrics.get_cache_hit_tokens(), 80);
        assert_eq!(metrics.get_summary().requests_with_cache, 1);

        // Verify cache hit rate: 80/100 = 80%
        let rate = metrics.get_cache_hit_rate();
        assert!((rate - 80.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_cache_metrics_none() {
        let metrics = Arc::new(MetricsLLMMiddleware::new());
        let mut stack = LLMMiddlewareStack::new();
        stack.add(metrics.clone());

        let request = create_test_request();

        // Response without cache hits (None)
        let response = LLMResponse::new("Hello", "test-model").with_usage(100, 20);

        stack.before(&request).await.unwrap();
        stack.after(&request, &response, 100).await.unwrap();

        assert_eq!(metrics.get_cache_hit_tokens(), 0);
        assert_eq!(metrics.get_summary().requests_with_cache, 0);
        assert_eq!(metrics.get_cache_hit_rate(), 0.0);
    }

    #[tokio::test]
    async fn test_cache_metrics_zero_hits() {
        let metrics = Arc::new(MetricsLLMMiddleware::new());
        let mut stack = LLMMiddlewareStack::new();
        stack.add(metrics.clone());

        let request = create_test_request();

        // Response with explicit 0 cache hits
        let response = LLMResponse::new("Hello", "test-model")
            .with_usage(100, 20)
            .with_cache_hit_tokens(0);

        stack.before(&request).await.unwrap();
        stack.after(&request, &response, 100).await.unwrap();

        assert_eq!(metrics.get_cache_hit_tokens(), 0);
        // Zero cache hits should not count as "with cache"
        assert_eq!(metrics.get_summary().requests_with_cache, 0);
    }

    #[tokio::test]
    async fn test_cache_metrics_aggregation() {
        let metrics = Arc::new(MetricsLLMMiddleware::new());
        let mut stack = LLMMiddlewareStack::new();
        stack.add(metrics.clone());

        let request = create_test_request();

        // First request: 80 cache hits / 100 prompt tokens
        let response1 = LLMResponse::new("Hello", "test-model")
            .with_usage(100, 20)
            .with_cache_hit_tokens(80);

        stack.before(&request).await.unwrap();
        stack.after(&request, &response1, 100).await.unwrap();

        // Second request: 150 cache hits / 200 prompt tokens
        let response2 = LLMResponse::new("World", "test-model")
            .with_usage(200, 50)
            .with_cache_hit_tokens(150);

        stack.before(&request).await.unwrap();
        stack.after(&request, &response2, 100).await.unwrap();

        // Third request: no cache support
        let response3 = LLMResponse::new("Test", "test-model").with_usage(100, 30);

        stack.before(&request).await.unwrap();
        stack.after(&request, &response3, 100).await.unwrap();

        let summary = metrics.get_summary();
        assert_eq!(summary.total_requests, 3);
        assert_eq!(summary.prompt_tokens, 400); // 100 + 200 + 100
        assert_eq!(summary.cache_hit_tokens, 230); // 80 + 150
        assert_eq!(summary.requests_with_cache, 2);

        // Cache hit rate: 230/400 = 57.5%
        assert!((summary.cache_hit_rate() - 57.5).abs() < 0.01);

        // Cache utilization: 2/3 = 66.67%
        assert!((summary.cache_utilization() - 66.67).abs() < 0.1);
    }

    #[test]
    fn test_cache_hit_rate_calculation() {
        let summary = MetricsSummary {
            total_requests: 10,
            total_tokens: 1500,
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_time_ms: 5000,
            tool_call_requests: 5,
            cache_hit_tokens: 800,
            requests_with_cache: 8,
        };

        // 800 / 1000 = 80%
        assert!((summary.cache_hit_rate() - 80.0).abs() < 0.01);

        // 8 / 10 = 80%
        assert!((summary.cache_utilization() - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_cache_hit_rate_zero_prompts() {
        let summary = MetricsSummary {
            total_requests: 0,
            total_tokens: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_time_ms: 0,
            tool_call_requests: 0,
            cache_hit_tokens: 0,
            requests_with_cache: 0,
        };

        // Should return 0.0 instead of NaN/infinity
        assert_eq!(summary.cache_hit_rate(), 0.0);
        assert_eq!(summary.cache_utilization(), 0.0);
    }

    // ========================================================================
    // OODA-30: Cost Estimation Tests
    // ========================================================================

    #[test]
    fn test_estimated_savings_calculation() {
        let summary = MetricsSummary {
            total_requests: 5,
            total_tokens: 1500,
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_time_ms: 5000,
            tool_call_requests: 2,
            cache_hit_tokens: 800,
            requests_with_cache: 4,
        };

        // Cost per 1k tokens = $0.003
        // Savings = (800 / 1000) * 0.003 * 0.9 = 0.00216
        let savings = summary.estimated_savings(0.003);
        assert!((savings - 0.00216).abs() < 0.00001);
    }

    #[test]
    fn test_estimated_savings_zero_cache() {
        let summary = MetricsSummary {
            total_requests: 5,
            total_tokens: 1500,
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_time_ms: 5000,
            tool_call_requests: 2,
            cache_hit_tokens: 0,
            requests_with_cache: 0,
        };

        // No cache hits = no savings
        assert_eq!(summary.estimated_savings(0.003), 0.0);
    }

    #[test]
    fn test_estimated_cost_all_cached() {
        let summary = MetricsSummary {
            total_requests: 1,
            total_tokens: 2000,
            prompt_tokens: 1000,
            completion_tokens: 1000,
            total_time_ms: 100,
            tool_call_requests: 0,
            cache_hit_tokens: 1000, // All prompt tokens cached
            requests_with_cache: 1,
        };

        // Prompt: $0.003/1k, Completion: $0.015/1k
        // All 1000 prompt tokens cached = (1000/1000) * 0.003 * 0.1 = 0.0003
        // Uncached prompt = 0
        // Completion = (1000/1000) * 0.015 = 0.015
        // Total = 0.0003 + 0 + 0.015 = 0.0153
        let cost = summary.estimated_cost(0.003, 0.015);
        assert!((cost - 0.0153).abs() < 0.00001);
    }

    #[test]
    fn test_estimated_cost_no_cache() {
        let summary = MetricsSummary {
            total_requests: 1,
            total_tokens: 2000,
            prompt_tokens: 1000,
            completion_tokens: 1000,
            total_time_ms: 100,
            tool_call_requests: 0,
            cache_hit_tokens: 0,
            requests_with_cache: 0,
        };

        // Prompt: $0.003/1k, Completion: $0.015/1k
        // Cached = 0
        // Uncached prompt = (1000/1000) * 0.003 = 0.003
        // Completion = (1000/1000) * 0.015 = 0.015
        // Total = 0 + 0.003 + 0.015 = 0.018
        let cost = summary.estimated_cost(0.003, 0.015);
        assert!((cost - 0.018).abs() < 0.00001);
    }

    #[test]
    fn test_estimated_cost_partial_cache() {
        let summary = MetricsSummary {
            total_requests: 1,
            total_tokens: 2000,
            prompt_tokens: 1000,
            completion_tokens: 1000,
            total_time_ms: 100,
            tool_call_requests: 0,
            cache_hit_tokens: 500, // 50% cached
            requests_with_cache: 1,
        };

        // Prompt: $0.003/1k, Completion: $0.015/1k
        // Cached = (500/1000) * 0.003 * 0.1 = 0.00015
        // Uncached prompt = (500/1000) * 0.003 = 0.0015
        // Completion = (1000/1000) * 0.015 = 0.015
        // Total = 0.00015 + 0.0015 + 0.015 = 0.01665
        let cost = summary.estimated_cost(0.003, 0.015);
        assert!((cost - 0.01665).abs() < 0.00001);
    }

    // ========================================================================
    // OODA-31: ModelCost Integration Tests
    // ========================================================================

    #[test]
    fn test_estimated_savings_for_model() {
        use crate::model_config::ModelCost;

        let summary = MetricsSummary {
            total_requests: 5,
            total_tokens: 1500,
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_time_ms: 5000,
            tool_call_requests: 2,
            cache_hit_tokens: 800,
            requests_with_cache: 4,
        };

        let model_cost = ModelCost {
            input_per_1k: 0.003,
            output_per_1k: 0.015,
            embedding_per_1k: 0.0,
            image_per_unit: 0.0,
            currency: "USD".to_string(),
        };

        // Should delegate to estimated_savings correctly
        let savings_direct = summary.estimated_savings(0.003);
        let savings_model = summary.estimated_savings_for_model(&model_cost);
        assert!((savings_direct - savings_model).abs() < 0.00001);
    }

    #[test]
    fn test_estimated_cost_for_model() {
        use crate::model_config::ModelCost;

        let summary = MetricsSummary {
            total_requests: 1,
            total_tokens: 2000,
            prompt_tokens: 1000,
            completion_tokens: 1000,
            total_time_ms: 100,
            tool_call_requests: 0,
            cache_hit_tokens: 500,
            requests_with_cache: 1,
        };

        let model_cost = ModelCost {
            input_per_1k: 0.003,
            output_per_1k: 0.015,
            embedding_per_1k: 0.0,
            image_per_unit: 0.0,
            currency: "USD".to_string(),
        };

        // Should delegate to estimated_cost correctly
        let cost_direct = summary.estimated_cost(0.003, 0.015);
        let cost_model = summary.estimated_cost_for_model(&model_cost);
        assert!((cost_direct - cost_model).abs() < 0.00001);
    }

    #[test]
    fn test_model_cost_gpt4_pricing() {
        use crate::model_config::ModelCost;

        let summary = MetricsSummary {
            total_requests: 10,
            total_tokens: 50000,
            prompt_tokens: 40000,
            completion_tokens: 10000,
            total_time_ms: 30000,
            tool_call_requests: 5,
            cache_hit_tokens: 35000, // 87.5% cache hit
            requests_with_cache: 9,
        };

        // GPT-4o pricing (approximate as of 2025)
        let gpt4_cost = ModelCost {
            input_per_1k: 0.0025, // $2.50/1M input
            output_per_1k: 0.01,  // $10/1M output
            embedding_per_1k: 0.0,
            image_per_unit: 0.0,
            currency: "USD".to_string(),
        };

        // Savings from cache: 35K * 0.9 * $0.0025/1k = $0.07875
        let savings = summary.estimated_savings_for_model(&gpt4_cost);
        assert!((savings - 0.07875).abs() < 0.0001);

        // Cost calculation:
        // Cached: 35K * 0.1 * $0.0025/1k = $0.00875
        // Uncached prompt: 5K * $0.0025/1k = $0.0125
        // Completion: 10K * $0.01/1k = $0.10
        // Total: $0.00875 + $0.0125 + $0.10 = $0.12125
        let cost = summary.estimated_cost_for_model(&gpt4_cost);
        assert!((cost - 0.12125).abs() < 0.0001);
    }

    // ========================================================================
    // OODA-33: Display Trait Tests
    // ========================================================================

    #[test]
    fn test_metrics_summary_display() {
        let summary = MetricsSummary {
            total_requests: 10,
            total_tokens: 5000,
            prompt_tokens: 4000,
            completion_tokens: 1000,
            total_time_ms: 1500,
            tool_call_requests: 3,
            cache_hit_tokens: 3200,
            requests_with_cache: 8,
        };

        let display = format!("{}", summary);

        // Verify format: reqs=10 tokens=4000/1000/3200 cache=80.0% latency=150ms tps=666.7
        assert!(display.contains("reqs=10"));
        assert!(display.contains("tokens=4000/1000/3200"));
        assert!(display.contains("cache=80.0%")); // 3200/4000 = 80%
        assert!(display.contains("latency=150ms")); // 1500ms / 10 reqs = 150ms
        assert!(display.contains("tps=")); // Output tps: 1000 / 1.5s = 666.7
    }

    #[test]
    fn test_metrics_summary_display_zero_values() {
        let summary = MetricsSummary {
            total_requests: 0,
            total_tokens: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_time_ms: 0,
            tool_call_requests: 0,
            cache_hit_tokens: 0,
            requests_with_cache: 0,
        };

        let display = format!("{}", summary);

        // Should handle zero values gracefully
        assert!(display.contains("reqs=0"));
        assert!(display.contains("cache=0.0%"));
        assert!(display.contains("tps=0.0"));
    }

    // ========================================================================
    // OODA-34: Throughput Metrics Tests
    // ========================================================================

    #[test]
    fn test_tokens_per_second() {
        let summary = MetricsSummary {
            total_requests: 5,
            total_tokens: 10000, // 10K total tokens
            prompt_tokens: 8000,
            completion_tokens: 2000,
            total_time_ms: 2000, // 2 seconds
            tool_call_requests: 0,
            cache_hit_tokens: 0,
            requests_with_cache: 0,
        };

        // 10000 tokens / 2 seconds = 5000 tokens/sec
        assert!((summary.tokens_per_second() - 5000.0).abs() < 0.1);
    }

    #[test]
    fn test_output_tokens_per_second() {
        let summary = MetricsSummary {
            total_requests: 5,
            total_tokens: 10000,
            prompt_tokens: 8000,
            completion_tokens: 2000, // 2K output tokens
            total_time_ms: 2000,     // 2 seconds
            tool_call_requests: 0,
            cache_hit_tokens: 0,
            requests_with_cache: 0,
        };

        // 2000 completion tokens / 2 seconds = 1000 tokens/sec
        assert!((summary.output_tokens_per_second() - 1000.0).abs() < 0.1);
    }

    #[test]
    fn test_throughput_zero_time() {
        let summary = MetricsSummary {
            total_requests: 1,
            total_tokens: 1000,
            prompt_tokens: 800,
            completion_tokens: 200,
            total_time_ms: 0, // Zero time
            tool_call_requests: 0,
            cache_hit_tokens: 0,
            requests_with_cache: 0,
        };

        // Should return 0.0 for zero time (not NaN or infinity)
        assert_eq!(summary.tokens_per_second(), 0.0);
        assert_eq!(summary.output_tokens_per_second(), 0.0);
    }

    #[test]
    fn test_throughput_realistic_session() {
        // Simulate a realistic 30-second agent session
        let summary = MetricsSummary {
            total_requests: 20,
            total_tokens: 50000,      // 50K total
            prompt_tokens: 40000,     // 40K prompt
            completion_tokens: 10000, // 10K completion
            total_time_ms: 30000,     // 30 seconds
            tool_call_requests: 15,
            cache_hit_tokens: 35000, // 87.5% cache
            requests_with_cache: 18,
        };

        // Total: 50K / 30s = 1666.7 tps
        assert!((summary.tokens_per_second() - 1666.67).abs() < 1.0);

        // Output: 10K / 30s = 333.3 tps
        assert!((summary.output_tokens_per_second() - 333.33).abs() < 1.0);
    }

    // ========================================================================
    // OODA-35: Token Efficiency Tests
    // ========================================================================

    #[test]
    fn test_token_efficiency() {
        let summary = MetricsSummary {
            total_requests: 5,
            total_tokens: 5000,
            prompt_tokens: 4000,
            completion_tokens: 1000, // 25% efficiency
            total_time_ms: 5000,
            tool_call_requests: 2,
            cache_hit_tokens: 0,
            requests_with_cache: 0,
        };

        // 1000 / 4000 * 100 = 25%
        assert!((summary.token_efficiency() - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_token_efficiency_low() {
        let summary = MetricsSummary {
            total_requests: 1,
            total_tokens: 10100,
            prompt_tokens: 10000,   // 10K input
            completion_tokens: 100, // Only 100 output = 1%
            total_time_ms: 1000,
            tool_call_requests: 0,
            cache_hit_tokens: 0,
            requests_with_cache: 0,
        };

        // Low efficiency: 100 / 10000 * 100 = 1%
        assert!((summary.token_efficiency() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_token_efficiency_zero_prompt() {
        let summary = MetricsSummary {
            total_requests: 0,
            total_tokens: 0,
            prompt_tokens: 0, // Edge case
            completion_tokens: 0,
            total_time_ms: 0,
            tool_call_requests: 0,
            cache_hit_tokens: 0,
            requests_with_cache: 0,
        };

        // Should return 0.0, not NaN
        assert_eq!(summary.token_efficiency(), 0.0);
    }

    // ========================================================================
    // OODA-36: Builder Pattern Tests
    // ========================================================================

    #[test]
    fn test_builder_basic() {
        let summary = MetricsSummaryBuilder::new()
            .requests(10)
            .prompt_tokens(4000)
            .completion_tokens(1000)
            .time_ms(1500)
            .build();

        assert_eq!(summary.total_requests, 10);
        assert_eq!(summary.prompt_tokens, 4000);
        assert_eq!(summary.completion_tokens, 1000);
        assert_eq!(summary.total_tokens, 5000); // Auto-calculated
        assert_eq!(summary.total_time_ms, 1500);
    }

    #[test]
    fn test_builder_with_cache() {
        let summary = MetricsSummaryBuilder::new()
            .requests(5)
            .prompt_tokens(10000)
            .completion_tokens(2000)
            .cache_hit_tokens(8000)
            .build();

        // Cache hit rate should be 80%
        assert!((summary.cache_hit_rate() - 80.0).abs() < 0.01);
        // Auto-set requests_with_cache
        assert_eq!(summary.requests_with_cache, 1);
    }

    #[test]
    fn test_builder_default() {
        let summary = MetricsSummaryBuilder::new().build();

        assert_eq!(summary.total_requests, 0);
        assert_eq!(summary.prompt_tokens, 0);
        assert_eq!(summary.completion_tokens, 0);
        assert_eq!(summary.cache_hit_tokens, 0);
    }

    #[test]
    fn test_builder_metrics_calculation() {
        let summary = MetricsSummaryBuilder::new()
            .requests(10)
            .prompt_tokens(5000)
            .completion_tokens(1000)
            .cache_hit_tokens(4000)
            .time_ms(2000)
            .build();

        // Latency: 2000ms / 10 = 200ms
        assert!((summary.average_latency_ms() - 200.0).abs() < 0.01);

        // TPS: 1000 / 2s = 500
        assert!((summary.output_tokens_per_second() - 500.0).abs() < 0.01);

        // Efficiency: 1000 / 5000 * 100 = 20%
        assert!((summary.token_efficiency() - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_llm_request_with_tools() {
        let tools = vec![ToolDefinition::function(
            "get_weather",
            "Get weather data",
            serde_json::json!({}),
        )];
        let request = LLMRequest::new(vec![ChatMessage::user("Hi")], "p", "m")
            .with_tools(tools);
        assert_eq!(request.tool_count(), 1);
        assert!(request.tools.is_some());
    }

    #[tokio::test]
    async fn test_metrics_tool_call_tracking() {
        let metrics = Arc::new(MetricsLLMMiddleware::new());
        let mut stack = LLMMiddlewareStack::new();
        stack.add(metrics.clone());

        let request = create_test_request();

        // Response with tool calls
        let mut response = LLMResponse::new("", "m").with_usage(10, 5);
        response.tool_calls = vec![crate::traits::ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: crate::traits::FunctionCall {
                name: "test".to_string(),
                arguments: "{}".to_string(),
            },
        }];

        stack.before(&request).await.unwrap();
        stack.after(&request, &response, 100).await.unwrap();

        let summary = metrics.get_summary();
        assert_eq!(summary.tool_call_requests, 1);
    }

    #[tokio::test]
    async fn test_logging_middleware_debug_level() {
        let logging = LoggingLLMMiddleware::with_level(LogLevel::Debug);
        assert_eq!(logging.name(), "logging");

        let request = create_test_request();
        let response = create_test_response();

        // Should not panic at Debug level
        logging.before(&request).await.unwrap();
        logging.after(&request, &response, 100).await.unwrap();
    }

    #[tokio::test]
    async fn test_logging_middleware_trace_level() {
        let logging = LoggingLLMMiddleware::with_level(LogLevel::Trace);
        let request = create_test_request();
        let response = create_test_response();

        logging.before(&request).await.unwrap();
        logging.after(&request, &response, 100).await.unwrap();
    }

    #[test]
    fn test_logging_middleware_default() {
        let logging = LoggingLLMMiddleware::default();
        assert_eq!(logging.name(), "logging");
    }

    #[test]
    fn test_metrics_middleware_default() {
        let metrics = MetricsLLMMiddleware::default();
        assert_eq!(metrics.get_total_requests(), 0);
        assert_eq!(metrics.get_total_tokens(), 0);
    }

    #[test]
    fn test_middleware_stack_default() {
        let stack = LLMMiddlewareStack::default();
        assert!(stack.is_empty());
        assert_eq!(stack.len(), 0);
    }

    #[test]
    fn test_builder_tool_calls() {
        let summary = MetricsSummaryBuilder::new()
            .requests(5)
            .tool_calls(3)
            .build();
        assert_eq!(summary.tool_call_requests, 3);
    }

    #[test]
    fn test_builder_requests_with_cache_override() {
        let summary = MetricsSummaryBuilder::new()
            .requests(10)
            .requests_with_cache(7)
            .build();
        assert_eq!(summary.requests_with_cache, 7);
    }

    #[tokio::test]
    async fn test_logging_debug_long_message() {
        let logging = LoggingLLMMiddleware::with_level(LogLevel::Debug);
        // Message longer than 100 chars
        let long_msg = "x".repeat(200);
        let request = LLMRequest::new(
            vec![ChatMessage::user(&long_msg)],
            "p",
            "m",
        );
        // Should truncate without panic
        logging.before(&request).await.unwrap();
    }

    #[tokio::test]
    async fn test_logging_debug_long_response() {
        let logging = LoggingLLMMiddleware::with_level(LogLevel::Debug);
        let request = create_test_request();
        let long_content = "y".repeat(300);
        let response = LLMResponse::new(&long_content, "m").with_usage(10, 5);
        // Should truncate without panic
        logging.after(&request, &response, 100).await.unwrap();
    }

    #[test]
    fn test_metrics_cache_hit_rate_no_prompts() {
        let metrics = MetricsLLMMiddleware::new();
        // No requests made
        assert_eq!(metrics.get_cache_hit_rate(), 0.0);
    }
}
