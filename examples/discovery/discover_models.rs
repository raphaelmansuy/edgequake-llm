//! End-to-end discovery example.
//!
//! Demonstrates the full model discovery pipeline:
//! 1. Discover all available models across providers
//! 2. Filter by capabilities (vision + tools + 100k context)
//! 3. Look up a specific model
//!
//! Run with: cargo run --example discover_models
//!
//! Set API keys to enable dynamic discovery:
//! - OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, etc.
//!
//! Without API keys, the service falls back to static registry data.

use edgequake_llm::{CapabilityFilter, ModelDiscoveryService};

#[tokio::main]
async fn main() {
    let service = ModelDiscoveryService::new();

    println!("=== Registered providers ===");
    for id in service.provider_ids() {
        println!("  - {}", id);
    }
    println!();

    // Discover all models
    println!("=== Discovering all models... ===");
    let all_models = service.discover_all().await.unwrap_or_default();
    println!("Found {} models across all providers\n", all_models.len());

    for model in &all_models {
        println!(
            "  [{:>12}] {:<40} ctx={:>10} out={:>8} vision={} tools={} thinking={} | {:?}",
            model.provider,
            model.id,
            model.context_length,
            model.max_output_tokens,
            if model.capabilities.supports_vision {
                "Y"
            } else {
                "N"
            },
            if model.capabilities.supports_function_calling {
                "Y"
            } else {
                "N"
            },
            if model.capabilities.supports_thinking {
                "Y"
            } else {
                "N"
            },
            model.source,
        );
    }
    println!();

    // Filter: vision + tools + large context
    let filter = CapabilityFilter {
        requires_vision: Some(true),
        requires_tools: Some(true),
        min_context_length: Some(100_000),
        exclude_deprecated: Some(true),
        ..Default::default()
    };

    println!("=== Models with vision + tools + 100k+ context ===");
    let filtered = service.find_models(&filter).await.unwrap_or_default();
    println!("Found {} matching models\n", filtered.len());

    for model in &filtered {
        println!(
            "  [{:>12}] {:<40} ctx={:>10}",
            model.provider, model.id, model.context_length
        );
    }
    println!();

    // Filter: reasoning models
    let reasoning_filter = CapabilityFilter {
        requires_thinking: Some(true),
        ..Default::default()
    };

    println!("=== Reasoning models (thinking support) ===");
    let reasoning = service
        .find_models(&reasoning_filter)
        .await
        .unwrap_or_default();
    println!("Found {} reasoning models\n", reasoning.len());

    for model in &reasoning {
        println!(
            "  [{:>12}] {:<40} ctx={:>10}",
            model.provider, model.id, model.context_length
        );
    }
    println!();

    // Look up specific model
    println!("=== Specific model lookup ===");
    if let Ok(Some(model)) = service.get_model("openai", "gpt-4.1").await {
        println!("  Found: {} ({})", model.name, model.id);
        println!("    Context: {} tokens", model.context_length);
        println!("    Max output: {} tokens", model.max_output_tokens);
        println!("    Vision: {}", model.capabilities.supports_vision);
        println!(
            "    Tools: {}",
            model.capabilities.supports_function_calling
        );
        if let Some(cost) = model.cost_per_m_input {
            println!("    Cost: ${}/M input tokens", cost);
        }
    } else {
        println!("  gpt-4.1 not found (OPENAI_API_KEY may not be set)");
    }
}
