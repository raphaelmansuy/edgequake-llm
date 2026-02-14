//! Cost Tracking example
//!
//! Demonstrates session-level cost tracking and budget management.
//!
//! Run with: cargo run --example cost_tracking
//!
//! This example shows:
//! - Setting up a cost tracker with budget limits
//! - Recording API usage and calculating costs
//! - Getting summaries by model, provider, and operation
//! - Budget alerts and warnings
//! - Cache savings estimation

use edgequake_llm::{
    format_cost, format_tokens, CostEntry, CostSummary, ModelPricing, SessionCostTracker,
};
use std::time::Duration;

fn main() {
    println!("💰 EdgeQuake LLM - Cost Tracking Example\n");
    println!("{}", "─".repeat(60));

    // Create a session tracker with a budget limit
    let mut tracker = SessionCostTracker::with_budget(10.0); // $10 budget

    // Set custom pricing for models we'll use
    tracker.set_pricing(
        "gpt-4o",
        ModelPricing::with_cache(2.50, 10.00, 0.25), // $2.50/M input, $10/M output, $0.25/M cached
    );
    tracker.set_pricing(
        "claude-sonnet-4-5-20250929",
        ModelPricing::with_cache(3.00, 15.00, 0.30), // $3/M input, $15/M output, $0.30/M cached
    );
    tracker.set_pricing(
        "gemini-2.5-flash",
        ModelPricing::new(0.075, 0.30), // $0.075/M input, $0.30/M output
    );

    // Simulate some API calls
    println!("\n📊 Simulating API Usage...\n");

    // Call 1: OpenAI GPT-4o
    let cost1 = tracker.record_usage("gpt-4o", "openai", 1000, 500);
    println!(
        "Call 1: GPT-4o - 1000 input, 500 output = ${:.6}",
        cost1
    );

    // Call 2: Claude with caching
    let cost2 = tracker.record_usage_with_cache(
        "claude-sonnet-4-5-20250929",
        "anthropic",
        5000,  // input tokens
        3000,  // cached tokens (60% cache hit)
        1000,  // output tokens
    );
    println!(
        "Call 2: Claude - 5000 input (3000 cached), 1000 output = ${:.6}",
        cost2
    );

    // Call 3: Gemini (cheap!)
    let cost3 = tracker.record_usage("gemini-2.5-flash", "google", 10000, 2000);
    println!(
        "Call 3: Gemini - 10000 input, 2000 output = ${:.6}",
        cost3
    );

    // Call 4: More GPT-4o
    let cost4 = tracker.record_usage("gpt-4o", "openai", 2000, 1000);
    println!(
        "Call 4: GPT-4o - 2000 input, 1000 output = ${:.6}",
        cost4
    );

    // Get summary
    let summary = tracker.summary();
    print_summary(&summary);

    // Check budget status
    println!("\n💼 Budget Status:");
    if let Some(remaining) = tracker.remaining_budget() {
        println!("   Remaining: ${:.4}", remaining);
        println!(
            "   Used: {:.1}% of $10.00 budget",
            (1.0 - remaining / 10.0) * 100.0
        );
    }

    if tracker.is_over_budget() {
        println!("   ⚠️  OVER BUDGET!");
    } else if tracker.is_near_budget() {
        println!("   ⚠️  Warning: Approaching budget limit!");
    } else {
        println!("   ✅ Within budget");
    }

    // Demonstrate cost formatting helpers
    println!("\n📝 Formatting Helpers:");
    println!("   {} = {}", format_tokens(1_500_000), format_cost(1.5));
    println!("   {} = {}", format_tokens(50_000), format_cost(0.0005));

    // Demonstrate operations tracking using CostEntry directly
    println!("\n🔧 Recording Different Operations:");
    let mut op_tracker = SessionCostTracker::new();
    
    // Record different operation types using CostEntry builder
    let entry1 = CostEntry::new("gpt-4o", "openai", 500, 200, 0.0035)
        .with_operation("chat");
    op_tracker.add_entry(entry1);
    
    let entry2 = CostEntry::new("text-embedding-3-small", "openai", 1000, 0, 0.0001)
        .with_operation("embedding");
    op_tracker.add_entry(entry2);
    
    let entry3 = CostEntry::new("gpt-4o", "openai", 300, 100, 0.0018)
        .with_operation("completion")
        .with_duration(Duration::from_millis(450));
    op_tracker.add_entry(entry3);
    
    let op_summary = op_tracker.summary();
    println!("\n   Operations breakdown:");
    for (op, cost) in &op_summary.by_operation {
        println!("   - {}: ${:.6}", op, cost);
    }

    println!("\n{}", "─".repeat(60));
    println!("✨ Cost tracking helps optimize LLM spend and stay within budget!");
}

fn print_summary(summary: &CostSummary) {
    println!("\n📈 Cost Summary:");
    println!("   Total Cost: ${:.6}", summary.total_cost);
    println!(
        "   Total Tokens: {} input + {} output",
        format_tokens(summary.total_input_tokens),
        format_tokens(summary.total_output_tokens)
    );
    println!("   API Calls: {}", summary.call_count);
    println!("   Avg Cost/Call: ${:.6}", summary.avg_cost_per_call);

    if summary.total_cached_tokens > 0 {
        println!(
            "\n   Cache Stats: {} cached ({:.1}% hit rate)",
            format_tokens(summary.total_cached_tokens),
            summary.cache_hit_rate() * 100.0
        );
    }

    println!("\n   By Model:");
    for (model, cost) in &summary.by_model {
        println!("   - {}: ${:.6}", model, cost);
    }

    println!("\n   By Provider:");
    for (provider, cost) in &summary.by_provider {
        println!("   - {}: ${:.6}", provider, cost);
    }
}
