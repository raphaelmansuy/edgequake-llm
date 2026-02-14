//! Inference Metrics for Real-Time Streaming Display
//!
//! OODA-33: Unified metrics collection for LLM streaming operations.
//!
//! ## Purpose
//!
//! Provides a single source of truth for all metrics collected during
//! a single LLM inference stream. Used by the display layer to show:
//! - Time to first token (TTFT)
//! - Token generation rate (tokens/second)
//! - Thinking/reasoning progress
//! - Total tokens generated
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    InferenceMetrics Flow                                │
//! │                                                                         │
//! │  Provider Stream ──► InferenceMetrics ──► Display Layer                 │
//! │        │                    │                   │                       │
//! │        ▼                    ▼                   ▼                       │
//! │  - StreamChunk        - record_first_token()  - ttft_ms()              │
//! │  - ThinkingContent    - add_output_tokens()   - tokens_per_second()    │
//! │  - Finished           - add_thinking_tokens() - thinking_tokens()      │
//! │                       - set_provider_ttft()                            │
//! │                                                                         │
//! │  Provider Preference:                                                   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ ttft_ms() returns:                                              │   │
//! │  │   1. provider_ttft_ms if set (native, most accurate)           │   │
//! │  │   2. measured TTFT from first_token_time (client-side)         │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use edgequake_llm::InferenceMetrics;
//!
//! let mut metrics = InferenceMetrics::new();
//!
//! // On first token received
//! metrics.record_first_token();
//!
//! // On each content chunk
//! metrics.add_output_tokens(5);
//!
//! // On thinking chunk
//! metrics.add_thinking_tokens(10);
//!
//! // Display metrics
//! println!("TTFT: {:?}ms", metrics.ttft_ms());
//! println!("Rate: {:.1} t/s", metrics.tokens_per_second());
//! ```

use std::time::{Duration, Instant};

/// Characters per token estimation (average across models).
const CHARS_PER_TOKEN: usize = 4;

/// Real-time metrics for a single LLM inference stream.
///
/// OODA-33: Unified metrics collection for streaming display.
///
/// Captures timing, token counts, and thinking metrics in one place.
/// Provides both measured and provider-reported values where available.
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    /// When the request was sent
    request_start: Instant,

    /// When the first token was received (for TTFT calculation)
    first_token_time: Option<Instant>,

    /// Total output tokens generated
    output_tokens: usize,

    /// Thinking/reasoning tokens (from ThinkingContent)
    thinking_tokens: usize,

    /// Input tokens (from request or provider response)
    input_tokens: Option<usize>,

    /// Provider-reported TTFT in milliseconds (if available)
    provider_ttft_ms: Option<f64>,

    /// Thinking budget (if applicable, for budget display)
    thinking_budget: Option<usize>,

    /// Total characters received (for token estimation)
    chars_received: usize,

    /// Last token time for rate calculation
    last_token_time: Option<Instant>,
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceMetrics {
    /// Create new metrics instance with current time as start.
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            request_start: now,
            first_token_time: None,
            output_tokens: 0,
            thinking_tokens: 0,
            input_tokens: None,
            provider_ttft_ms: None,
            thinking_budget: None,
            chars_received: 0,
            last_token_time: None,
        }
    }

    /// Create metrics with a specific start time (for testing).
    pub fn with_start_time(start: Instant) -> Self {
        Self {
            request_start: start,
            first_token_time: None,
            output_tokens: 0,
            thinking_tokens: 0,
            input_tokens: None,
            provider_ttft_ms: None,
            thinking_budget: None,
            chars_received: 0,
            last_token_time: None,
        }
    }

    /// Record arrival of first token (for TTFT calculation).
    ///
    /// Should be called once when the first content/thinking token arrives.
    /// If called multiple times, only the first call is recorded.
    pub fn record_first_token(&mut self) {
        if self.first_token_time.is_none() {
            let now = Instant::now();
            self.first_token_time = Some(now);
            self.last_token_time = Some(now);
        }
    }

    /// Add output tokens to the count.
    ///
    /// # Arguments
    /// * `count` - Number of tokens to add (can be 0)
    pub fn add_output_tokens(&mut self, count: usize) {
        self.output_tokens += count;
        self.last_token_time = Some(Instant::now());
    }

    /// Add thinking/reasoning tokens to the count.
    ///
    /// # Arguments
    /// * `count` - Number of thinking tokens to add
    pub fn add_thinking_tokens(&mut self, count: usize) {
        self.thinking_tokens += count;
        self.last_token_time = Some(Instant::now());
    }

    /// Add characters received (for token estimation).
    ///
    /// # Arguments
    /// * `count` - Number of characters to add
    pub fn add_chars(&mut self, count: usize) {
        self.chars_received += count;
    }

    /// Set provider-reported TTFT in milliseconds.
    ///
    /// When available, this takes precedence over measured TTFT.
    ///
    /// # Arguments
    /// * `ms` - Time to first token in milliseconds
    pub fn set_provider_ttft(&mut self, ms: f64) {
        self.provider_ttft_ms = Some(ms);
    }

    /// Set input token count from provider response.
    ///
    /// # Arguments
    /// * `count` - Number of input/prompt tokens
    pub fn set_input_tokens(&mut self, count: usize) {
        self.input_tokens = Some(count);
    }

    /// Set thinking budget (for budget display like "1.2k/10k").
    ///
    /// # Arguments
    /// * `budget` - Total thinking token budget
    pub fn set_thinking_budget(&mut self, budget: usize) {
        self.thinking_budget = Some(budget);
    }

    /// Get time to first token in milliseconds.
    ///
    /// Prefers provider-reported TTFT if available, otherwise
    /// calculates from measured first_token_time.
    ///
    /// # Returns
    /// TTFT in milliseconds, or None if no token received yet
    pub fn ttft_ms(&self) -> Option<f64> {
        // Prefer provider-reported TTFT
        if let Some(ttft) = self.provider_ttft_ms {
            return Some(ttft);
        }

        // Fall back to measured TTFT
        self.first_token_time
            .map(|ft| ft.duration_since(self.request_start).as_secs_f64() * 1000.0)
    }

    /// Get output tokens per second.
    ///
    /// Calculates generation rate based on elapsed time since first token.
    /// Returns 0.0 if no tokens generated or no time elapsed.
    ///
    /// # Returns
    /// Tokens per second as f64
    pub fn tokens_per_second(&self) -> f64 {
        let Some(first) = self.first_token_time else {
            return 0.0;
        };

        let elapsed = first.elapsed().as_secs_f64();
        if elapsed <= 0.0 || self.output_tokens == 0 {
            return 0.0;
        }

        self.output_tokens as f64 / elapsed
    }

    /// Get total tokens per second (output + thinking).
    ///
    /// # Returns
    /// Total tokens per second as f64
    pub fn total_tokens_per_second(&self) -> f64 {
        let Some(first) = self.first_token_time else {
            return 0.0;
        };

        let elapsed = first.elapsed().as_secs_f64();
        if elapsed <= 0.0 {
            return 0.0;
        }

        (self.output_tokens + self.thinking_tokens) as f64 / elapsed
    }

    /// Get elapsed time since request start.
    ///
    /// # Returns
    /// Duration since metrics were created
    pub fn elapsed(&self) -> Duration {
        self.request_start.elapsed()
    }

    /// Get time since first token.
    ///
    /// # Returns
    /// Duration since first token, or Duration::ZERO if no token yet
    pub fn time_since_first_token(&self) -> Duration {
        self.first_token_time
            .map(|ft| ft.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    /// Get output token count.
    pub fn output_tokens(&self) -> usize {
        self.output_tokens
    }

    /// Get thinking token count.
    pub fn thinking_tokens(&self) -> usize {
        self.thinking_tokens
    }

    /// Get total tokens (output + thinking).
    pub fn total_tokens(&self) -> usize {
        self.output_tokens + self.thinking_tokens
    }

    /// Get input token count (if set).
    pub fn input_tokens(&self) -> Option<usize> {
        self.input_tokens
    }

    /// Get thinking budget (if set).
    pub fn thinking_budget(&self) -> Option<usize> {
        self.thinking_budget
    }

    /// Estimate tokens from accumulated characters.
    ///
    /// Uses the standard heuristic of ~4 characters per token.
    ///
    /// # Returns
    /// Estimated token count (minimum 1 for non-zero chars)
    pub fn estimated_tokens(&self) -> usize {
        if self.chars_received == 0 {
            return 0;
        }
        std::cmp::max(1, self.chars_received / CHARS_PER_TOKEN)
    }

    /// Get characters received count.
    pub fn chars_received(&self) -> usize {
        self.chars_received
    }

    /// Check if first token has been received.
    pub fn has_first_token(&self) -> bool {
        self.first_token_time.is_some()
    }

    /// Format thinking progress (e.g., "1.2k/10k").
    ///
    /// # Returns
    /// Formatted string, or None if no thinking tokens
    pub fn format_thinking_progress(&self) -> Option<String> {
        if self.thinking_tokens == 0 {
            return None;
        }

        let tokens = format_tokens(self.thinking_tokens);
        match self.thinking_budget {
            Some(budget) => Some(format!("{}/{}", tokens, format_tokens(budget))),
            None => Some(tokens),
        }
    }

    /// Format TTFT for display (e.g., "1.2s" or "850ms").
    ///
    /// # Returns
    /// Formatted string, or None if no TTFT available
    pub fn format_ttft(&self) -> Option<String> {
        self.ttft_ms().map(|ms| {
            if ms >= 1000.0 {
                format!("{:.1}s", ms / 1000.0)
            } else {
                format!("{:.0}ms", ms)
            }
        })
    }

    /// Format token rate for display (e.g., "42 t/s").
    ///
    /// # Returns
    /// Formatted string (always returns, shows "0 t/s" if no rate)
    pub fn format_rate(&self) -> String {
        let rate = self.tokens_per_second();
        if rate >= 10.0 {
            format!("{:.0} t/s", rate)
        } else {
            format!("{:.1} t/s", rate)
        }
    }
}

/// Format token count for display (e.g., "1.2k", "12", "1.5M").
fn format_tokens(count: usize) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}k", count as f64 / 1_000.0)
    } else {
        format!("{}", count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_new_initializes_correctly() {
        let metrics = InferenceMetrics::new();
        assert_eq!(metrics.output_tokens(), 0);
        assert_eq!(metrics.thinking_tokens(), 0);
        assert!(!metrics.has_first_token());
        assert!(metrics.ttft_ms().is_none());
    }

    #[test]
    fn test_first_token_ttft() {
        let start = Instant::now();
        let mut metrics = InferenceMetrics::with_start_time(start);

        // Small delay to simulate TTFT
        sleep(Duration::from_millis(10));
        metrics.record_first_token();

        let ttft = metrics.ttft_ms();
        assert!(ttft.is_some());
        assert!(ttft.unwrap() >= 10.0);
    }

    #[test]
    fn test_tokens_per_second() {
        let mut metrics = InferenceMetrics::new();
        metrics.record_first_token();

        // Add tokens with small delay
        sleep(Duration::from_millis(50));
        metrics.add_output_tokens(50);

        let rate = metrics.tokens_per_second();
        // Should be approximately 50 tokens / 0.05 seconds = 1000 t/s
        // But timing can vary, so just check it's reasonable
        assert!(rate > 0.0);
    }

    #[test]
    fn test_provider_ttft_takes_precedence() {
        let mut metrics = InferenceMetrics::new();
        metrics.record_first_token();

        // Set provider TTFT (should override measured)
        metrics.set_provider_ttft(123.45);

        let ttft = metrics.ttft_ms();
        assert!(ttft.is_some());
        assert!((ttft.unwrap() - 123.45).abs() < 0.001);
    }

    #[test]
    fn test_thinking_tokens_tracked() {
        let mut metrics = InferenceMetrics::new();
        metrics.add_thinking_tokens(100);
        metrics.add_thinking_tokens(50);

        assert_eq!(metrics.thinking_tokens(), 150);
        assert_eq!(metrics.total_tokens(), 150);
    }

    #[test]
    fn test_total_tokens() {
        let mut metrics = InferenceMetrics::new();
        metrics.add_output_tokens(100);
        metrics.add_thinking_tokens(50);

        assert_eq!(metrics.output_tokens(), 100);
        assert_eq!(metrics.thinking_tokens(), 50);
        assert_eq!(metrics.total_tokens(), 150);
    }

    #[test]
    fn test_estimated_tokens() {
        let mut metrics = InferenceMetrics::new();
        metrics.add_chars(100);

        assert_eq!(metrics.estimated_tokens(), 25); // 100 / 4
    }

    #[test]
    fn test_estimated_tokens_minimum() {
        let mut metrics = InferenceMetrics::new();
        metrics.add_chars(3);

        // 3 / 4 = 0, but minimum is 1
        assert_eq!(metrics.estimated_tokens(), 1);
    }

    #[test]
    fn test_format_tokens() {
        assert_eq!(format_tokens(0), "0");
        assert_eq!(format_tokens(123), "123");
        assert_eq!(format_tokens(1234), "1.2k");
        assert_eq!(format_tokens(12345), "12.3k");
        assert_eq!(format_tokens(1234567), "1.2M");
    }

    #[test]
    fn test_format_thinking_progress() {
        let mut metrics = InferenceMetrics::new();
        
        // No thinking tokens
        assert!(metrics.format_thinking_progress().is_none());

        // With thinking tokens, no budget
        metrics.add_thinking_tokens(1500);
        assert_eq!(metrics.format_thinking_progress(), Some("1.5k".to_string()));

        // With budget
        metrics.set_thinking_budget(10000);
        assert_eq!(
            metrics.format_thinking_progress(),
            Some("1.5k/10.0k".to_string())
        );
    }

    #[test]
    fn test_format_ttft() {
        let mut metrics = InferenceMetrics::new();

        // No TTFT yet
        assert!(metrics.format_ttft().is_none());

        // Milliseconds
        metrics.set_provider_ttft(850.0);
        assert_eq!(metrics.format_ttft(), Some("850ms".to_string()));

        // Seconds
        metrics.set_provider_ttft(1250.0);
        assert_eq!(metrics.format_ttft(), Some("1.2s".to_string()));
    }

    #[test]
    fn test_first_token_only_recorded_once() {
        let start = Instant::now();
        let mut metrics = InferenceMetrics::with_start_time(start);

        sleep(Duration::from_millis(10));
        metrics.record_first_token();
        let ttft1 = metrics.ttft_ms().unwrap();

        sleep(Duration::from_millis(10));
        metrics.record_first_token(); // Should not update
        let ttft2 = metrics.ttft_ms().unwrap();

        // TTFTs should be the same (only first call matters)
        assert!((ttft1 - ttft2).abs() < 1.0);
    }

    #[test]
    fn test_default_impl() {
        let metrics = InferenceMetrics::default();
        assert_eq!(metrics.output_tokens(), 0);
        assert_eq!(metrics.thinking_tokens(), 0);
        assert!(!metrics.has_first_token());
    }

    #[test]
    fn test_total_tokens_per_second_no_first_token() {
        let metrics = InferenceMetrics::new();
        assert_eq!(metrics.total_tokens_per_second(), 0.0);
    }

    #[test]
    fn test_total_tokens_per_second_with_tokens() {
        let mut metrics = InferenceMetrics::new();
        metrics.record_first_token();
        sleep(Duration::from_millis(50));
        metrics.add_output_tokens(30);
        metrics.add_thinking_tokens(20);

        let rate = metrics.total_tokens_per_second();
        assert!(rate > 0.0);
    }

    #[test]
    fn test_tokens_per_second_no_first_token() {
        let metrics = InferenceMetrics::new();
        assert_eq!(metrics.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_tokens_per_second_zero_output_tokens() {
        let mut metrics = InferenceMetrics::new();
        metrics.record_first_token();
        sleep(Duration::from_millis(10));
        // No output tokens added
        assert_eq!(metrics.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_elapsed() {
        let metrics = InferenceMetrics::new();
        sleep(Duration::from_millis(10));
        let elapsed = metrics.elapsed();
        assert!(elapsed >= Duration::from_millis(10));
    }

    #[test]
    fn test_time_since_first_token_none() {
        let metrics = InferenceMetrics::new();
        assert_eq!(metrics.time_since_first_token(), Duration::ZERO);
    }

    #[test]
    fn test_time_since_first_token_some() {
        let mut metrics = InferenceMetrics::new();
        metrics.record_first_token();
        sleep(Duration::from_millis(10));
        let since = metrics.time_since_first_token();
        assert!(since >= Duration::from_millis(10));
    }

    #[test]
    fn test_input_tokens() {
        let mut metrics = InferenceMetrics::new();
        assert!(metrics.input_tokens().is_none());
        metrics.set_input_tokens(500);
        assert_eq!(metrics.input_tokens(), Some(500));
    }

    #[test]
    fn test_thinking_budget() {
        let mut metrics = InferenceMetrics::new();
        assert!(metrics.thinking_budget().is_none());
        metrics.set_thinking_budget(10000);
        assert_eq!(metrics.thinking_budget(), Some(10000));
    }

    #[test]
    fn test_chars_received() {
        let mut metrics = InferenceMetrics::new();
        assert_eq!(metrics.chars_received(), 0);
        metrics.add_chars(100);
        assert_eq!(metrics.chars_received(), 100);
        metrics.add_chars(50);
        assert_eq!(metrics.chars_received(), 150);
    }

    #[test]
    fn test_estimated_tokens_zero_chars() {
        let metrics = InferenceMetrics::new();
        assert_eq!(metrics.estimated_tokens(), 0);
    }

    #[test]
    fn test_format_rate_high() {
        let mut metrics = InferenceMetrics::new();
        metrics.record_first_token();
        sleep(Duration::from_millis(10));
        metrics.add_output_tokens(500);
        let rate = metrics.format_rate();
        // High rate, should use {:.0} format
        assert!(rate.contains("t/s"));
    }

    #[test]
    fn test_format_rate_low() {
        let mut metrics = InferenceMetrics::new();
        // No first token - rate = 0
        let rate = metrics.format_rate();
        assert_eq!(rate, "0.0 t/s");
    }

    #[test]
    fn test_has_first_token() {
        let mut metrics = InferenceMetrics::new();
        assert!(!metrics.has_first_token());
        metrics.record_first_token();
        assert!(metrics.has_first_token());
    }

    #[test]
    fn test_debug_impl() {
        let metrics = InferenceMetrics::new();
        let debug = format!("{:?}", metrics);
        assert!(debug.contains("InferenceMetrics"));
    }
}
