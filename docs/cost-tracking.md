# Cost Tracking

EdgeQuake LLM provides session-level cost tracking and aggregation
for monitoring and controlling LLM API spending.

## Overview

```text
  API Call 1             API Call 2             API Call N
  (GPT-4o, 1K in/500 out) (Claude, 2K in/1K out) ...
       |                      |                    |
       v                      v                    v
  +----------------------------------------------------------------+
  |                   SessionCostTracker                            |
  |                                                                |
  |   Pricing DB   -->  Cost Calculation  -->  CostEntry Log       |
  |   (per model)       (input + output)       (timestamped)       |
  |                                                                |
  |   Budget Limit  -->  Threshold Check  -->  Warning/Stop        |
  |   (optional)        (80% default)                              |
  +----------------------------------------------------------------+
       |
       v
  +-------------------+
  |   CostSummary     |
  |   - total_cost    |
  |   - by_model      |
  |   - by_provider   |
  |   - by_operation  |
  +-------------------+
```

## Setting Up Pricing

Model pricing is specified in USD per million tokens:

```rust,ignore
use edgequake_llm::cost_tracker::{SessionCostTracker, ModelPricing};

let mut tracker = SessionCostTracker::new();

// Set custom pricing
tracker.set_pricing("gpt-4o", ModelPricing::new(2.5, 10.0));
tracker.set_pricing("claude-sonnet-4-5-20250929", ModelPricing::new(3.0, 15.0));

// With cache discount pricing
tracker.set_pricing(
    "claude-3-opus-20240229",
    ModelPricing::with_cache(15.0, 75.0, 1.5),  // cached input at 90% discount
);
```

### Default Pricing

The tracker includes default pricing for common models:

| Model | Input (per 1M) | Output (per 1M) | Cached Input |
|-------|---------------|-----------------|--------------|
| GPT-4o | $2.50 | $10.00 | - |
| GPT-4o-mini | $0.15 | $0.60 | - |
| Claude Sonnet 4.5 | $3.00 | $15.00 | $0.30 |
| Claude Opus 4.5 | $15.00 | $75.00 | $1.50 |
| Gemini 2.5 Flash | $0.15 | $0.60 | - |
| Gemini 2.5 Pro | $1.25 | $5.00 | - |

## Recording Usage

```rust,ignore
// Basic usage recording
let cost = tracker.record_usage(
    "gpt-4o",      // model
    "openai",      // provider
    1000,          // input tokens
    500,           // output tokens
);
println!("This call cost: ${:.4}", cost);

// With cached tokens
let cost = tracker.record_usage_with_cache(
    "claude-sonnet-4-5-20250929",
    "anthropic",
    10000,     // total input tokens
    8000,      // cached tokens (90% discount)
    1000,      // output tokens
);
```

## Budget Management

```rust,ignore
// Create tracker with budget limit
let mut tracker = SessionCostTracker::with_budget(5.00); // $5 budget

// Set warning threshold (default: 80%)
tracker.set_warning_threshold(0.75); // Warn at 75%

// Record usage
tracker.record_usage("gpt-4o", "openai", 1000, 500);

// Check budget status
if tracker.is_over_budget() {
    eprintln!("Budget exceeded!");
} else if tracker.is_near_budget() {
    eprintln!("Warning: approaching budget limit");
}

// Check remaining budget
if let Some(remaining) = tracker.remaining_budget() {
    println!("Remaining: ${:.2}", remaining);
}

// Budget usage percentage
if let Some(pct) = tracker.budget_usage_percent() {
    println!("Budget used: {:.1}%", pct);
}
```

## Getting a Summary

```rust,ignore
let summary = tracker.summary();

println!("Total cost:     ${:.4}", summary.total_cost);
println!("Total calls:    {}", summary.call_count);
println!("Avg per call:   ${:.4}", summary.avg_cost_per_call);
println!("Input tokens:   {}", summary.total_input_tokens);
println!("Output tokens:  {}", summary.total_output_tokens);
println!("Cached tokens:  {}", summary.total_cached_tokens);
println!("Cache hit rate: {:.1}%", summary.cache_hit_rate() * 100.0);

// Breakdown by model
for (model, cost) in &summary.by_model {
    println!("  {}: ${:.4}", model, cost);
}

// Breakdown by provider
for (provider, cost) in &summary.by_provider {
    println!("  {}: ${:.4}", provider, cost);
}
```

## Formatting Helpers

```rust,ignore
use edgequake_llm::cost_tracker::{format_cost, format_tokens};

println!("{}", format_cost(0.0042));    // "$0.0042"
println!("{}", format_tokens(1_500));   // "1.5K"
println!("{}", format_tokens(2_300_000)); // "2.3M"
```

## Integration with Middleware

The `MetricsLLMMiddleware` automatically records cost data
when used with the middleware pipeline:

```rust,ignore
use edgequake_llm::middleware::{LLMMiddlewareStack, MetricsLLMMiddleware};
use std::sync::Arc;

let mut stack = LLMMiddlewareStack::new();
let metrics = Arc::new(MetricsLLMMiddleware::new());
stack.add(metrics.clone());

// After requests, get cost summary
let summary = metrics.summary();
println!("Session cost: ${:.4}", summary.total_cost);
```

## Source Files

| File | Purpose |
|------|---------|
| `src/cost_tracker.rs` | Session cost tracking and aggregation |
| `src/middleware.rs` | MetricsLLMMiddleware (automatic cost recording) |
| `src/cache_prompt.rs` | Cache-aware cost calculations |
