//! Session Cost Tracker
//!
//! # OODA-21: Cost Tracking
//!
//! This module provides session-level cost tracking and aggregation for LLM API calls.
//!
//! # Overview
//!
//! The `SessionCostTracker` aggregates costs across multiple LLM calls within a session,
//! providing breakdowns by model, provider, and operation type.
//!
//! # Usage
//!
//! ```rust
//! use edgequake_llm::cost_tracker::{SessionCostTracker, ModelPricing};
//!
//! let mut tracker = SessionCostTracker::new();
//!
//! // Set pricing for a model
//! tracker.set_pricing("claude-3-opus-20240229", ModelPricing::new(15.0, 75.0));
//!
//! // Record a cost entry
//! tracker.record_usage(
//!     "claude-3-opus-20240229",
//!     "anthropic",
//!     1000,  // input tokens
//!     500,   // output tokens
//! );
//!
//! // Get summary
//! let summary = tracker.summary();
//! println!("Total cost: ${:.4}", summary.total_cost);
//! ```
//!
//! # Built-in pricing
//!
//! Models without a [`SessionCostTracker::set_pricing`] entry are priced from the
//! [`llm-prices`](https://docs.rs/llm-prices) table: dated, source-cited list prices for
//! Anthropic, OpenAI, Gemini, DeepSeek, Mistral, xAI, Groq, Together, Fireworks and Cohere,
//! including cached-input rates. Snapshot ids (`gpt-4o-mini-2024-07-18`), `-latest` aliases,
//! Bedrock and Vertex ids and router prefixes (`openrouter/...`) resolve to their row, and
//! [`SessionCostTracker::record_usage`] honors the provider that served the call, since hosts
//! sell the same open-weight model at different prices.
//!
//! ```rust
//! use edgequake_llm::cost_tracker::SessionCostTracker;
//!
//! let mut tracker = SessionCostTracker::new();
//! // Claude Opus 4.5 lists at $5 / $25 per million tokens.
//! let cost = tracker.record_usage("claude-opus-4-5", "anthropic", 1_000_000, 1_000_000);
//! assert!((cost - 30.0).abs() < 1e-9);
//! ```
//!
//! # See Also
//!
//! - `middleware.rs` for per-request cost estimation
//! - `cache_prompt.rs` for cache-aware cost calculations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, SystemTime};

/// Pricing information for a model.
///
/// Costs are specified in dollars per million tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    /// Cost per million input tokens (USD).
    pub input_cost_per_million: f64,

    /// Cost per million output tokens (USD).
    pub output_cost_per_million: f64,

    /// Optional: Cost per million cached input tokens (USD).
    /// If None, caching is not supported or priced the same as regular input.
    pub cached_input_cost_per_million: Option<f64>,
}

impl ModelPricing {
    /// Create new pricing with input and output costs.
    pub fn new(input_cost_per_million: f64, output_cost_per_million: f64) -> Self {
        Self {
            input_cost_per_million,
            output_cost_per_million,
            cached_input_cost_per_million: None,
        }
    }

    /// Create pricing with cache support.
    pub fn with_cache(
        input_cost_per_million: f64,
        output_cost_per_million: f64,
        cached_cost_per_million: f64,
    ) -> Self {
        Self {
            input_cost_per_million,
            output_cost_per_million,
            cached_input_cost_per_million: Some(cached_cost_per_million),
        }
    }

    /// Calculate cost for given token counts.
    pub fn calculate_cost(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        let input_cost = (input_tokens as f64 / 1_000_000.0) * self.input_cost_per_million;
        let output_cost = (output_tokens as f64 / 1_000_000.0) * self.output_cost_per_million;
        input_cost + output_cost
    }

    /// Calculate cost with cached tokens.
    pub fn calculate_cost_with_cache(
        &self,
        input_tokens: u64,
        cached_tokens: u64,
        output_tokens: u64,
    ) -> f64 {
        let uncached = input_tokens.saturating_sub(cached_tokens);
        let input_cost = (uncached as f64 / 1_000_000.0) * self.input_cost_per_million;

        let cached_cost = if let Some(cache_price) = self.cached_input_cost_per_million {
            (cached_tokens as f64 / 1_000_000.0) * cache_price
        } else {
            (cached_tokens as f64 / 1_000_000.0) * self.input_cost_per_million
        };

        let output_cost = (output_tokens as f64 / 1_000_000.0) * self.output_cost_per_million;

        input_cost + cached_cost + output_cost
    }

    /// Pricing from an `llm-prices` row. `None` when the row has no input or output price.
    fn from_price(row: &llm_prices::Price) -> Option<Self> {
        Some(Self {
            input_cost_per_million: row.input?,
            output_cost_per_million: row.output?,
            cached_input_cost_per_million: row.cache_read,
        })
    }
}

impl Default for ModelPricing {
    fn default() -> Self {
        // Default to GPT-4o pricing as a reasonable default
        Self::new(2.5, 10.0)
    }
}

/// Family patterns for ids the `llm-prices` table does not list itself, such as
/// `o1-preview`, `chatgpt-4o-latest` or `gemini-2.0-flash-lite`. Matched after the table,
/// longest pattern first. Prices come from the table where it has the family's row; the
/// hand-written ones are retired models it no longer lists.
fn fallback_patterns() -> &'static [(&'static str, ModelPricing)] {
    static PATTERNS: OnceLock<Vec<(&'static str, ModelPricing)>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        let from_table = [
            "claude-sonnet-4",
            "claude-3-5-haiku",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "o1",
        ]
        .into_iter()
        .filter_map(|id| {
            let pricing = table_pricing_for(llm_prices::lookup(id)?)?;
            Some((id, pricing.clone()))
        });
        let retired = [
            ("claude-3-opus", ModelPricing::with_cache(15.0, 75.0, 1.5)),
            (
                "claude-3-5-sonnet",
                ModelPricing::with_cache(3.0, 15.0, 0.3),
            ),
            ("o1-mini", ModelPricing::new(3.0, 12.0)),
            ("gemini-2.0-flash", ModelPricing::new(0.075, 0.3)),
            ("gemini-1.5-pro", ModelPricing::new(1.25, 5.0)),
        ];
        from_table.chain(retired).collect()
    })
}

/// Every `llm-prices` row converted to [`ModelPricing`] once, in table order.
fn table_pricing() -> &'static [Option<ModelPricing>] {
    static TABLE: OnceLock<Vec<Option<ModelPricing>>> = OnceLock::new();
    TABLE.get_or_init(|| {
        llm_prices::all()
            .iter()
            .map(ModelPricing::from_price)
            .collect()
    })
}

/// The `llm-prices` row for `model`, as sold by `provider` when the table knows that provider.
fn table_row(provider: Option<&str>, model: &str) -> Option<&'static llm_prices::Price> {
    provider
        .and_then(|p| llm_prices::lookup_provider(p, model))
        .or_else(|| llm_prices::lookup(model))
}

/// The converted pricing for a row of the `llm-prices` table.
fn table_pricing_for(row: &'static llm_prices::Price) -> Option<&'static ModelPricing> {
    let index = llm_prices::all()
        .iter()
        .position(|r| std::ptr::eq(r, row))?;
    table_pricing().get(index)?.as_ref()
}

/// The pattern `model` matches most specifically: the longest pattern it contains,
/// preferring a prefix match on equal length, then the alphabetically first pattern.
/// The result never depends on `HashMap` iteration order.
fn best_pattern_match<'a>(
    model: &str,
    candidates: impl Iterator<Item = (&'a str, &'a ModelPricing)>,
) -> Option<&'a ModelPricing> {
    candidates
        .filter(|(pattern, _)| model.contains(pattern))
        .max_by(|(a, _), (b, _)| {
            a.len()
                .cmp(&b.len())
                .then_with(|| model.starts_with(a).cmp(&model.starts_with(b)))
                .then_with(|| b.cmp(a))
        })
        .map(|(_, pricing)| pricing)
}

/// A single cost entry for one LLM call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEntry {
    /// Model name.
    pub model: String,

    /// Provider name.
    pub provider: String,

    /// Input tokens used.
    pub input_tokens: u64,

    /// Output tokens generated.
    pub output_tokens: u64,

    /// Cached input tokens (if applicable).
    pub cached_tokens: u64,

    /// Calculated cost in USD.
    pub cost: f64,

    /// When the call was made.
    pub timestamp: SystemTime,

    /// Duration of the call.
    pub duration: Option<Duration>,

    /// Operation type (e.g., "chat", "completion", "embedding").
    pub operation: String,
}

impl CostEntry {
    /// Create a new cost entry.
    pub fn new(
        model: impl Into<String>,
        provider: impl Into<String>,
        input_tokens: u64,
        output_tokens: u64,
        cost: f64,
    ) -> Self {
        Self {
            model: model.into(),
            provider: provider.into(),
            input_tokens,
            output_tokens,
            cached_tokens: 0,
            cost,
            timestamp: SystemTime::now(),
            duration: None,
            operation: "chat".to_string(),
        }
    }

    /// Set cached tokens.
    pub fn with_cached_tokens(mut self, cached_tokens: u64) -> Self {
        self.cached_tokens = cached_tokens;
        self
    }

    /// Set duration.
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Set operation type.
    pub fn with_operation(mut self, operation: impl Into<String>) -> Self {
        self.operation = operation.into();
        self
    }
}

/// Summary of costs across a session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostSummary {
    /// Total cost in USD.
    pub total_cost: f64,

    /// Total input tokens.
    pub total_input_tokens: u64,

    /// Total output tokens.
    pub total_output_tokens: u64,

    /// Total cached tokens.
    pub total_cached_tokens: u64,

    /// Number of API calls.
    pub call_count: usize,

    /// Average cost per call.
    pub avg_cost_per_call: f64,

    /// Breakdown by model.
    pub by_model: HashMap<String, f64>,

    /// Breakdown by provider.
    pub by_provider: HashMap<String, f64>,

    /// Breakdown by operation.
    pub by_operation: HashMap<String, f64>,
}

impl CostSummary {
    /// Calculate cache hit rate.
    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_input_tokens == 0 {
            0.0
        } else {
            self.total_cached_tokens as f64 / self.total_input_tokens as f64
        }
    }

    /// Estimate savings from caching.
    pub fn cache_savings(&self, normal_price_per_million: f64) -> f64 {
        let would_have_cost =
            (self.total_cached_tokens as f64 / 1_000_000.0) * normal_price_per_million;
        let actual_cost =
            (self.total_cached_tokens as f64 / 1_000_000.0) * normal_price_per_million * 0.1;
        would_have_cost - actual_cost
    }
}

/// Session-level cost tracker.
///
/// Tracks and aggregates costs across multiple LLM calls within a session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionCostTracker {
    /// All cost entries.
    entries: Vec<CostEntry>,

    /// Pricing set with [`SessionCostTracker::set_pricing`], by model id or pattern.
    pricing: HashMap<String, ModelPricing>,

    /// Optional budget limit in USD.
    budget_limit: Option<f64>,

    /// Warning threshold (percentage of budget).
    warning_threshold: f64,
}

impl SessionCostTracker {
    /// Create a new session cost tracker.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            pricing: HashMap::new(),
            budget_limit: None,
            warning_threshold: 0.8,
        }
    }

    /// Create with a budget limit.
    pub fn with_budget(limit: f64) -> Self {
        let mut tracker = Self::new();
        tracker.budget_limit = Some(limit);
        tracker
    }

    /// Set budget limit.
    pub fn set_budget(&mut self, limit: f64) {
        self.budget_limit = Some(limit);
    }

    /// Set warning threshold (0.0 to 1.0).
    pub fn set_warning_threshold(&mut self, threshold: f64) {
        self.warning_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Set pricing for a model.
    pub fn set_pricing(&mut self, model: impl Into<String>, pricing: ModelPricing) {
        self.pricing.insert(model.into(), pricing);
    }

    /// Get pricing for a model.
    ///
    /// Same resolution as [`SessionCostTracker::get_pricing_for_provider`] without a
    /// provider hint.
    pub fn get_pricing(&self, model: &str) -> Option<&ModelPricing> {
        self.resolve_pricing(None, model)
    }

    /// Get pricing for a model as sold by `provider` (`"openai"`, `"anthropic"`,
    /// `"gemini"`, `"vertexai"`, `"bedrock"`, `"azure-openai"`, `"openrouter"`, ...).
    ///
    /// Resolution order:
    ///
    /// 1. an exact [`SessionCostTracker::set_pricing`] entry;
    /// 2. the `llm-prices` row for the model, under `provider` when the table knows it.
    ///    A `set_pricing` entry for the same row wins, so pricing set for `gpt-4o` also
    ///    covers `gpt-4o-2024-08-06` but not `gpt-4o-mini`;
    /// 3. the most specific `set_pricing` pattern the model contains (longest first);
    /// 4. built-in family patterns (`gpt-4o`, `o1`, `claude-3-opus`, ...) for ids the
    ///    table does not list, such as previews and retired models.
    pub fn get_pricing_for_provider(&self, provider: &str, model: &str) -> Option<&ModelPricing> {
        self.resolve_pricing(Some(provider), model)
    }

    fn resolve_pricing(&self, provider: Option<&str>, model: &str) -> Option<&ModelPricing> {
        if let Some(p) = self.pricing.get(model) {
            return Some(p);
        }

        if let Some(row) = table_row(provider, model) {
            if let Some(table) = table_pricing_for(row) {
                // A custom price for this same model under another id (an alias, or the
                // undated id of a snapshot) overrides the list price.
                let custom = best_pattern_match(
                    model,
                    self.pricing
                        .iter()
                        .filter(|(id, _)| {
                            table_row(provider, id).is_some_and(|r| std::ptr::eq(r, row))
                        })
                        .map(|(id, p)| (id.as_str(), p)),
                );
                return Some(custom.unwrap_or(table));
            }
        }

        best_pattern_match(model, self.pricing.iter().map(|(id, p)| (id.as_str(), p))).or_else(
            || best_pattern_match(model, fallback_patterns().iter().map(|(id, p)| (*id, p))),
        )
    }

    /// Record usage and calculate cost.
    pub fn record_usage(
        &mut self,
        model: &str,
        provider: &str,
        input_tokens: u64,
        output_tokens: u64,
    ) -> f64 {
        let cost = self
            .get_pricing_for_provider(provider, model)
            .map(|p| p.calculate_cost(input_tokens, output_tokens))
            .unwrap_or_else(|| {
                // Use default pricing
                ModelPricing::default().calculate_cost(input_tokens, output_tokens)
            });

        let entry = CostEntry::new(model, provider, input_tokens, output_tokens, cost);
        self.entries.push(entry);

        cost
    }

    /// Record usage with cached tokens.
    pub fn record_usage_with_cache(
        &mut self,
        model: &str,
        provider: &str,
        input_tokens: u64,
        cached_tokens: u64,
        output_tokens: u64,
    ) -> f64 {
        let cost = self
            .get_pricing_for_provider(provider, model)
            .map(|p| p.calculate_cost_with_cache(input_tokens, cached_tokens, output_tokens))
            .unwrap_or_else(|| {
                ModelPricing::default().calculate_cost_with_cache(
                    input_tokens,
                    cached_tokens,
                    output_tokens,
                )
            });

        let entry = CostEntry::new(model, provider, input_tokens, output_tokens, cost)
            .with_cached_tokens(cached_tokens);
        self.entries.push(entry);

        cost
    }

    /// Add a pre-calculated cost entry.
    pub fn add_entry(&mut self, entry: CostEntry) {
        self.entries.push(entry);
    }

    /// Get total cost.
    pub fn total_cost(&self) -> f64 {
        self.entries.iter().map(|e| e.cost).sum()
    }

    /// Get number of API calls.
    pub fn call_count(&self) -> usize {
        self.entries.len()
    }

    /// Check if budget is exceeded.
    pub fn is_over_budget(&self) -> bool {
        self.budget_limit
            .map(|b| self.total_cost() >= b)
            .unwrap_or(false)
    }

    /// Check if approaching budget limit.
    pub fn is_near_budget(&self) -> bool {
        self.budget_limit
            .map(|b| self.total_cost() >= b * self.warning_threshold)
            .unwrap_or(false)
    }

    /// Get remaining budget.
    pub fn remaining_budget(&self) -> Option<f64> {
        self.budget_limit.map(|b| (b - self.total_cost()).max(0.0))
    }

    /// Get budget usage percentage.
    pub fn budget_usage_percent(&self) -> Option<f64> {
        self.budget_limit
            .map(|b| (self.total_cost() / b * 100.0).min(100.0))
    }

    /// Get summary statistics.
    pub fn summary(&self) -> CostSummary {
        let mut summary = CostSummary::default();

        for entry in &self.entries {
            summary.total_cost += entry.cost;
            summary.total_input_tokens += entry.input_tokens;
            summary.total_output_tokens += entry.output_tokens;
            summary.total_cached_tokens += entry.cached_tokens;
            summary.call_count += 1;

            *summary.by_model.entry(entry.model.clone()).or_default() += entry.cost;
            *summary
                .by_provider
                .entry(entry.provider.clone())
                .or_default() += entry.cost;
            *summary
                .by_operation
                .entry(entry.operation.clone())
                .or_default() += entry.cost;
        }

        if summary.call_count > 0 {
            summary.avg_cost_per_call = summary.total_cost / summary.call_count as f64;
        }

        summary
    }

    /// Get all entries.
    pub fn entries(&self) -> &[CostEntry] {
        &self.entries
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get entries since a timestamp.
    pub fn entries_since(&self, since: SystemTime) -> Vec<&CostEntry> {
        self.entries
            .iter()
            .filter(|e| e.timestamp >= since)
            .collect()
    }
}

/// Format cost in a human-readable way.
pub fn format_cost(cost: f64) -> String {
    if cost < 0.01 {
        format!("${:.4}", cost)
    } else if cost < 1.0 {
        format!("${:.3}", cost)
    } else {
        format!("${:.2}", cost)
    }
}

/// Format token count with commas.
pub fn format_tokens(tokens: u64) -> String {
    if tokens >= 1_000_000 {
        format!("{:.1}M", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1_000 {
        format!("{:.1}K", tokens as f64 / 1_000.0)
    } else {
        format!("{}", tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_pricing_new() {
        let pricing = ModelPricing::new(15.0, 75.0);
        assert_eq!(pricing.input_cost_per_million, 15.0);
        assert_eq!(pricing.output_cost_per_million, 75.0);
        assert!(pricing.cached_input_cost_per_million.is_none());
    }

    #[test]
    fn test_model_pricing_with_cache() {
        let pricing = ModelPricing::with_cache(15.0, 75.0, 1.5);
        assert_eq!(pricing.cached_input_cost_per_million, Some(1.5));
    }

    #[test]
    fn test_calculate_cost() {
        let pricing = ModelPricing::new(15.0, 75.0);
        let cost = pricing.calculate_cost(1_000_000, 1_000_000);
        assert!((cost - 90.0).abs() < 0.001);
    }

    #[test]
    fn test_calculate_cost_small() {
        let pricing = ModelPricing::new(3.0, 15.0); // claude-3.5-sonnet
        let cost = pricing.calculate_cost(1000, 500);
        // 1000/1M * 3.0 + 500/1M * 15.0 = 0.003 + 0.0075 = 0.0105
        assert!((cost - 0.0105).abs() < 0.0001);
    }

    #[test]
    fn test_calculate_cost_with_cache() {
        let pricing = ModelPricing::with_cache(15.0, 75.0, 1.5);
        let cost = pricing.calculate_cost_with_cache(1_000_000, 800_000, 100_000);
        // Uncached: 200K * 15/1M = 3.0
        // Cached: 800K * 1.5/1M = 1.2
        // Output: 100K * 75/1M = 7.5
        // Total = 11.7
        assert!((cost - 11.7).abs() < 0.001);
    }

    #[test]
    fn test_cost_entry_new() {
        let entry = CostEntry::new("gpt-4o", "openai", 1000, 500, 0.05);
        assert_eq!(entry.model, "gpt-4o");
        assert_eq!(entry.provider, "openai");
        assert_eq!(entry.input_tokens, 1000);
        assert_eq!(entry.output_tokens, 500);
        assert_eq!(entry.cost, 0.05);
    }

    #[test]
    fn test_cost_entry_with_cache() {
        let entry = CostEntry::new("claude-3-opus", "anthropic", 10000, 1000, 0.10)
            .with_cached_tokens(8000);
        assert_eq!(entry.cached_tokens, 8000);
    }

    #[test]
    fn test_session_tracker_new() {
        let tracker = SessionCostTracker::new();
        assert_eq!(tracker.call_count(), 0);
        assert_eq!(tracker.total_cost(), 0.0);
    }

    #[test]
    fn test_session_tracker_with_budget() {
        let tracker = SessionCostTracker::with_budget(10.0);
        assert_eq!(tracker.remaining_budget(), Some(10.0));
    }

    #[test]
    fn test_record_usage() {
        let mut tracker = SessionCostTracker::new();
        let cost = tracker.record_usage("gpt-4o", "openai", 1000, 500);
        assert!(cost > 0.0);
        assert_eq!(tracker.call_count(), 1);
    }

    #[test]
    fn test_record_usage_with_cache() {
        let mut tracker = SessionCostTracker::new();
        let cost = tracker.record_usage_with_cache("claude-3-opus", "anthropic", 10000, 8000, 1000);
        assert!(cost > 0.0);
        assert_eq!(tracker.entries()[0].cached_tokens, 8000);
    }

    #[test]
    fn test_budget_tracking() {
        let mut tracker = SessionCostTracker::with_budget(0.1);
        tracker.record_usage("gpt-4o", "openai", 1000000, 500000);
        assert!(tracker.is_over_budget());
    }

    #[test]
    fn test_near_budget() {
        let mut tracker = SessionCostTracker::with_budget(1.0);
        tracker.set_warning_threshold(0.5);
        // Add enough to be at 50%+
        tracker.record_usage("gpt-4o", "openai", 100000, 50000);
        // Check if near budget
        assert!(tracker.total_cost() > 0.0);
    }

    #[test]
    fn test_summary() {
        let mut tracker = SessionCostTracker::new();
        tracker.record_usage("gpt-4o", "openai", 1000, 500);
        tracker.record_usage("claude-3-opus", "anthropic", 2000, 1000);

        let summary = tracker.summary();
        assert_eq!(summary.call_count, 2);
        assert_eq!(summary.total_input_tokens, 3000);
        assert_eq!(summary.total_output_tokens, 1500);
        assert!(summary.by_model.contains_key("gpt-4o"));
        assert!(summary.by_model.contains_key("claude-3-opus"));
    }

    #[test]
    fn test_summary_by_provider() {
        let mut tracker = SessionCostTracker::new();
        tracker.record_usage("gpt-4o", "openai", 1000, 500);
        tracker.record_usage("gpt-4o-mini", "openai", 2000, 1000);
        tracker.record_usage("claude-3-opus", "anthropic", 1000, 500);

        let summary = tracker.summary();
        assert!(summary.by_provider.contains_key("openai"));
        assert!(summary.by_provider.contains_key("anthropic"));
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut tracker = SessionCostTracker::new();
        tracker.record_usage_with_cache("claude-3-opus", "anthropic", 10000, 8000, 1000);

        let summary = tracker.summary();
        assert!((summary.cache_hit_rate() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_clear() {
        let mut tracker = SessionCostTracker::new();
        tracker.record_usage("gpt-4o", "openai", 1000, 500);
        assert_eq!(tracker.call_count(), 1);

        tracker.clear();
        assert_eq!(tracker.call_count(), 0);
    }

    #[test]
    fn test_format_cost() {
        assert_eq!(format_cost(0.001), "$0.0010");
        assert_eq!(format_cost(0.05), "$0.050");
        assert_eq!(format_cost(1.50), "$1.50");
        assert_eq!(format_cost(10.00), "$10.00");
    }

    #[test]
    fn test_format_tokens() {
        assert_eq!(format_tokens(500), "500");
        assert_eq!(format_tokens(1500), "1.5K");
        assert_eq!(format_tokens(1_500_000), "1.5M");
    }

    #[test]
    fn test_default_pricing() {
        let tracker = SessionCostTracker::new();
        assert!(tracker.get_pricing("claude-3-opus").is_some());
        assert!(tracker.get_pricing("gpt-4o").is_some());
        assert!(tracker.get_pricing("gemini-2.0-flash").is_some());
    }

    #[test]
    fn test_pricing_prefix_match() {
        let tracker = SessionCostTracker::new();
        // Should match "claude-3-opus" pattern
        assert!(tracker.get_pricing("claude-3-opus-20240229").is_some());
    }

    #[test]
    fn test_serialization() {
        let mut tracker = SessionCostTracker::with_budget(10.0);
        tracker.record_usage("gpt-4o", "openai", 1000, 500);

        let json = serde_json::to_string(&tracker).unwrap();
        let restored: SessionCostTracker = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.call_count(), 1);
        assert!((restored.total_cost() - tracker.total_cost()).abs() < 0.0001);
    }

    #[test]
    fn test_budget_usage_percent() {
        let mut tracker = SessionCostTracker::with_budget(10.0);
        // Manually add an entry with known cost
        let entry = CostEntry::new("test", "test", 0, 0, 5.0);
        tracker.add_entry(entry);

        assert_eq!(tracker.budget_usage_percent(), Some(50.0));
    }

    #[test]
    fn test_cost_entry_with_duration() {
        let entry =
            CostEntry::new("m", "p", 100, 50, 0.01).with_duration(Duration::from_millis(500));
        assert_eq!(entry.duration, Some(Duration::from_millis(500)));
    }

    #[test]
    fn test_cost_entry_with_operation() {
        let entry = CostEntry::new("m", "p", 100, 50, 0.01).with_operation("embedding");
        assert_eq!(entry.operation, "embedding");
    }

    #[test]
    fn test_model_pricing_default() {
        let pricing = ModelPricing::default();
        assert_eq!(pricing.input_cost_per_million, 2.5);
        assert_eq!(pricing.output_cost_per_million, 10.0);
        assert!(pricing.cached_input_cost_per_million.is_none());
    }

    #[test]
    fn test_cost_summary_cache_savings() {
        let summary = CostSummary {
            total_cached_tokens: 1_000_000,
            ..Default::default()
        };
        // Savings should be positive for cached tokens
        let savings = summary.cache_savings(15.0);
        // 1M * 15/1M - 1M * 15 * 0.1/1M = 15.0 - 1.5 = 13.5
        assert!((savings - 13.5).abs() < 0.001);
    }

    #[test]
    fn test_cost_summary_cache_savings_zero() {
        let summary = CostSummary::default();
        assert_eq!(summary.cache_savings(15.0), 0.0);
    }

    #[test]
    fn test_cost_summary_cache_hit_rate_zero_input() {
        let summary = CostSummary::default();
        assert_eq!(summary.cache_hit_rate(), 0.0);
    }

    #[test]
    fn test_set_budget() {
        let mut tracker = SessionCostTracker::new();
        assert!(tracker.remaining_budget().is_none());
        tracker.set_budget(5.0);
        assert_eq!(tracker.remaining_budget(), Some(5.0));
    }

    #[test]
    fn test_set_pricing_and_use() {
        let mut tracker = SessionCostTracker::new();
        tracker.set_pricing("custom-model", ModelPricing::new(1.0, 2.0));
        let cost = tracker.record_usage("custom-model", "custom", 1_000_000, 1_000_000);
        // 1M * 1/1M + 1M * 2/1M = 3.0
        assert!((cost - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_entries_since() {
        let mut tracker = SessionCostTracker::new();
        let before = SystemTime::now();
        tracker.record_usage("gpt-4o", "openai", 100, 50);

        let entries = tracker.entries_since(before);
        assert_eq!(entries.len(), 1);

        // Future time should return nothing
        let future = SystemTime::now() + Duration::from_secs(3600);
        let entries_future = tracker.entries_since(future);
        assert!(entries_future.is_empty());
    }

    #[test]
    fn test_is_near_budget_no_budget() {
        let tracker = SessionCostTracker::new();
        assert!(!tracker.is_near_budget());
    }

    #[test]
    fn test_is_over_budget_no_budget() {
        let tracker = SessionCostTracker::new();
        assert!(!tracker.is_over_budget());
    }

    #[test]
    fn test_budget_usage_percent_no_budget() {
        let tracker = SessionCostTracker::new();
        assert!(tracker.budget_usage_percent().is_none());
    }

    #[test]
    fn test_budget_usage_percent_capped_at_100() {
        let mut tracker = SessionCostTracker::with_budget(0.001);
        tracker.add_entry(CostEntry::new("m", "p", 0, 0, 1.0));
        assert_eq!(tracker.budget_usage_percent(), Some(100.0));
    }

    #[test]
    fn test_remaining_budget_capped_at_zero() {
        let mut tracker = SessionCostTracker::with_budget(0.001);
        tracker.add_entry(CostEntry::new("m", "p", 0, 0, 1.0));
        assert_eq!(tracker.remaining_budget(), Some(0.0));
    }

    #[test]
    fn test_set_warning_threshold_clamped() {
        let mut tracker = SessionCostTracker::new();
        tracker.set_warning_threshold(2.0);
        // Should clamp to 1.0
        tracker.set_budget(10.0);
        tracker.add_entry(CostEntry::new("m", "p", 0, 0, 10.0));
        assert!(tracker.is_near_budget());
    }

    #[test]
    fn test_summary_avg_cost_per_call() {
        let mut tracker = SessionCostTracker::new();
        tracker.add_entry(CostEntry::new("m", "p", 0, 0, 2.0));
        tracker.add_entry(CostEntry::new("m", "p", 0, 0, 4.0));
        let summary = tracker.summary();
        assert!((summary.avg_cost_per_call - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_summary_by_operation() {
        let mut tracker = SessionCostTracker::new();
        tracker.add_entry(CostEntry::new("m", "p", 100, 50, 1.0).with_operation("embedding"));
        tracker.add_entry(CostEntry::new("m", "p", 100, 50, 2.0).with_operation("chat"));
        let summary = tracker.summary();
        assert!(summary.by_operation.contains_key("embedding"));
        assert!(summary.by_operation.contains_key("chat"));
    }

    #[test]
    fn test_record_usage_unknown_model_uses_default() {
        let mut tracker = SessionCostTracker::new();
        let cost = tracker.record_usage("unknown-model", "unknown", 1_000_000, 1_000_000);
        // Default pricing: 2.5 + 10.0 = 12.5
        assert!((cost - 12.5).abs() < 0.001);
    }

    #[test]
    fn test_calculate_cost_with_cache_no_cache_price() {
        let pricing = ModelPricing::new(15.0, 75.0); // No cached price
        let cost = pricing.calculate_cost_with_cache(1_000_000, 800_000, 100_000);
        // Uncached: 200K * 15/1M = 3.0
        // Cached at regular price: 800K * 15/1M = 12.0
        // Output: 100K * 75/1M = 7.5
        // Total: 22.5
        assert!((cost - 22.5).abs() < 0.001);
    }

    // ---- Built-in pricing from llm-prices -------------------------------

    /// Cost of 1M input + 1M output tokens.
    fn million_each(tracker: &SessionCostTracker, provider: &str, model: &str) -> f64 {
        tracker
            .get_pricing_for_provider(provider, model)
            .map(|p| p.calculate_cost(1_000_000, 1_000_000))
            .unwrap()
    }

    #[test]
    fn test_current_models_priced_from_table_not_gpt4o_default() {
        let mut tracker = SessionCostTracker::new();
        // Claude Opus 4.5: $5 / $25. Previously billed at the $2.50 / $10 default.
        let opus = tracker.record_usage("claude-opus-4-5", "anthropic", 1_000_000, 1_000_000);
        assert!((opus - 30.0).abs() < 1e-9, "opus = {opus}");
        // GPT-5.5: $5 / $30.
        let gpt = tracker.record_usage("gpt-5.5", "openai", 1_000_000, 1_000_000);
        assert!((gpt - 35.0).abs() < 1e-9, "gpt-5.5 = {gpt}");
        // Gemini 2.5 Pro on Vertex AI is billed at Google's list price ($1.25 / $10).
        let gemini = tracker.record_usage("gemini-2.5-pro", "vertexai", 1_000_000, 1_000_000);
        assert!((gemini - 11.25).abs() < 1e-9, "gemini = {gemini}");
    }

    #[test]
    fn test_snapshot_ids_resolve_to_their_own_model() {
        // HashMap iteration order used to decide whether this hit gpt-4o or gpt-4o-mini.
        // Fresh trackers get fresh hash seeds, so check many of them.
        for _ in 0..64 {
            let tracker = SessionCostTracker::new();
            let p = tracker.get_pricing("gpt-4o-mini-2024-07-18").unwrap();
            assert_eq!(p.input_cost_per_million, 0.15);
            assert_eq!(p.output_cost_per_million, 0.6);
        }
    }

    #[test]
    fn test_snapshot_with_its_own_price() {
        // gpt-4o-2024-05-13 lists at $5 / $15, not the $2.50 / $10 of the gpt-4o alias.
        let tracker = SessionCostTracker::new();
        assert!((million_each(&tracker, "openai", "gpt-4o-2024-05-13") - 20.0).abs() < 1e-9);
        assert!((million_each(&tracker, "openai", "gpt-4o") - 12.5).abs() < 1e-9);
    }

    #[test]
    fn test_table_cached_input_rate_is_used() {
        let mut tracker = SessionCostTracker::new();
        // GPT-4o: $2.50 input, $1.25 cached input, $10 output.
        let cost = tracker.record_usage_with_cache("gpt-4o", "openai", 1_000_000, 1_000_000, 0);
        assert!((cost - 1.25).abs() < 1e-9, "cost = {cost}");
    }

    #[test]
    fn test_router_and_bedrock_ids_resolve() {
        let tracker = SessionCostTracker::new();
        let direct = million_each(&tracker, "anthropic", "claude-sonnet-4-20250514");
        let routed = million_each(&tracker, "openrouter", "anthropic/claude-sonnet-4");
        let bedrock = million_each(
            &tracker,
            "bedrock",
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
        );
        assert!((direct - 18.0).abs() < 1e-9);
        assert_eq!(direct, routed);
        assert_eq!(direct, bedrock);
    }

    #[test]
    fn test_provider_aware_pricing_for_open_weight_models() {
        let tracker = SessionCostTracker::new();
        let groq = tracker
            .get_pricing_for_provider("groq", "openai/gpt-oss-120b")
            .unwrap();
        let fireworks = tracker
            .get_pricing_for_provider("fireworks", "accounts/fireworks/models/gpt-oss-120b")
            .unwrap();
        // Groq lists no cached-input price; Fireworks does.
        assert_eq!(groq.cached_input_cost_per_million, None);
        assert_eq!(fireworks.cached_input_cost_per_million, Some(0.015));
    }

    #[test]
    fn test_custom_pricing_overrides_table_for_same_model() {
        let mut tracker = SessionCostTracker::new();
        tracker.set_pricing("gpt-4o", ModelPricing::new(1.0, 2.0));
        // A snapshot of the same model uses the negotiated price...
        assert!((million_each(&tracker, "openai", "gpt-4o-2024-08-06") - 3.0).abs() < 1e-9);
        // ...but gpt-4o-mini is a different model and keeps its own list price.
        assert!((million_each(&tracker, "openai", "gpt-4o-mini") - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_custom_pattern_longest_match_is_deterministic() {
        for _ in 0..64 {
            let mut tracker = SessionCostTracker::new();
            tracker.set_pricing("acme", ModelPricing::new(1.0, 1.0));
            tracker.set_pricing("acme-large", ModelPricing::new(4.0, 4.0));
            tracker.set_pricing("acme-large-v2", ModelPricing::new(9.0, 9.0));
            let p = tracker.get_pricing("acme-large-v2-0611").unwrap();
            assert_eq!(p.input_cost_per_million, 9.0);
            let p = tracker.get_pricing("acme-large-ft").unwrap();
            assert_eq!(p.input_cost_per_million, 4.0);
        }
    }

    #[test]
    fn test_retired_models_keep_builtin_prices() {
        let tracker = SessionCostTracker::new();
        let opus3 = tracker.get_pricing("claude-3-opus-20240229").unwrap();
        assert_eq!(opus3.input_cost_per_million, 15.0);
        assert_eq!(opus3.output_cost_per_million, 75.0);
        assert!(tracker.get_pricing("claude-3-5-sonnet-20241022").is_some());
        assert!(tracker.get_pricing("o1-mini-2024-09-12").is_some());
        assert!(tracker.get_pricing("gemini-1.5-pro-002").is_some());
    }

    #[test]
    fn test_unlisted_ids_still_match_their_family() {
        let tracker = SessionCostTracker::new();
        // Not in the table, so the gpt-4o / o1 / gemini-2.0-flash family patterns apply.
        assert_eq!(
            tracker
                .get_pricing("chatgpt-4o-latest")
                .unwrap()
                .input_cost_per_million,
            2.5
        );
        assert_eq!(
            tracker
                .get_pricing("o1-preview")
                .unwrap()
                .input_cost_per_million,
            15.0
        );
        assert_eq!(
            tracker
                .get_pricing("gemini-2.0-flash-lite")
                .unwrap()
                .input_cost_per_million,
            0.075
        );
        // o1-pro has its own row ($150 / $600) instead of matching the o1 pattern.
        assert_eq!(
            tracker
                .get_pricing("o1-pro")
                .unwrap()
                .input_cost_per_million,
            150.0
        );
    }

    #[test]
    fn test_deserialized_tracker_with_old_seed_table() {
        // Trackers serialized before this change carry the old hand table in `pricing`.
        let json = r#"{"entries":[],"pricing":{
            "gpt-4o":{"input_cost_per_million":2.5,"output_cost_per_million":10.0,"cached_input_cost_per_million":null},
            "gpt-4o-mini":{"input_cost_per_million":0.15,"output_cost_per_million":0.6,"cached_input_cost_per_million":null}
        },"budget_limit":null,"warning_threshold":0.8}"#;
        let tracker: SessionCostTracker = serde_json::from_str(json).unwrap();
        assert!((million_each(&tracker, "openai", "gpt-4o-2024-05-13") - 20.0).abs() < 1e-9);
        assert!((million_each(&tracker, "anthropic", "claude-opus-4-5") - 30.0).abs() < 1e-9);
    }
}
