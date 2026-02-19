//! Mistral AI provider example
//!
//! Demonstrates chat completions, streaming, embeddings, and model listing
//! using the Mistral AI provider.
//!
//! # Setup
//!
//! ```bash
//! export MISTRAL_API_KEY=your-api-key
//! cargo run --example mistral_chat
//! ```
//!
//! # What this example covers
//!
//! 1. Basic chat completion (non-streaming)
//! 2. Streaming chat response
//! 3. Text embeddings using mistral-embed
//! 4. Listing available models
//! 5. Function / tool calling

use edgequake_llm::traits::{
    ChatMessage, EmbeddingProvider, LLMProvider, ToolChoice, ToolDefinition,
};
use edgequake_llm::MistralProvider;
use futures::StreamExt;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌟 EdgeQuake LLM — Mistral AI Example\n");

    // -----------------------------------------------------------------------
    // 1. Create provider from environment
    // -----------------------------------------------------------------------
    let provider = MistralProvider::from_env().map_err(|e| {
        eprintln!("❌ Failed to create Mistral provider: {}", e);
        eprintln!("   Make sure MISTRAL_API_KEY is set.");
        e
    })?;

    println!(
        "✅ Provider: {} | Model: {}",
        LLMProvider::name(&provider),
        LLMProvider::model(&provider),
    );
    println!();

    // -----------------------------------------------------------------------
    // 2. Basic chat completion
    // -----------------------------------------------------------------------
    println!("💬 Basic chat completion");
    println!("{}", "─".repeat(50));

    let messages = vec![
        ChatMessage::system("You are a concise and helpful assistant."),
        ChatMessage::user("Explain Rust's ownership model in two sentences."),
    ];

    let response = provider.chat(&messages, None).await?;
    println!("Response: {}", response.content);
    println!(
        "Tokens: {} prompt + {} completion = {} total",
        response.prompt_tokens,
        response.completion_tokens,
        response.prompt_tokens + response.completion_tokens
    );
    println!();

    // -----------------------------------------------------------------------
    // 3. Streaming chat
    // -----------------------------------------------------------------------
    println!("🌊 Streaming chat");
    println!("{}", "─".repeat(50));

    let mut stream = provider.stream("Write a haiku about asynchronous programming.").await?;
    let mut full_response = String::new();

    while let Some(result) = stream.next().await {
        match result {
            Ok(content) => {
                print!("{}", content);
                io::stdout().flush()?;
                full_response.push_str(&content);
            }
            Err(e) => {
                eprintln!("\n❌ Stream error: {}", e);
                break;
            }
        }
    }
    println!("\n");

    // -----------------------------------------------------------------------
    // 4. Embeddings
    // -----------------------------------------------------------------------
    println!("🔢 Embeddings (mistral-embed)");
    println!("{}", "─".repeat(50));

    let texts = vec![
        "Rust is a systems programming language focused on safety.".to_string(),
        "Python is great for data science and machine learning.".to_string(),
        "Go is simple, efficient, and designed for cloud services.".to_string(),
    ];

    let embeddings = provider.embed(&texts).await?;

    println!(
        "Generated {} embeddings, dimension: {}",
        embeddings.len(),
        embeddings[0].len()
    );

    // Compute cosine similarity between first two embeddings
    let sim_01 = cosine_similarity(&embeddings[0], &embeddings[1]);
    let sim_02 = cosine_similarity(&embeddings[0], &embeddings[2]);
    println!("Cosine similarity (Rust, Python): {:.4}", sim_01);
    println!("Cosine similarity (Rust, Go):     {:.4}", sim_02);
    println!();

    // -----------------------------------------------------------------------
    // 5. List available models
    // -----------------------------------------------------------------------
    println!("📋 Available Mistral models");
    println!("{}", "─".repeat(50));

    let models_response = provider.list_models().await?;
    for model in &models_response.data {
        let ctx = model
            .max_context_length
            .map(|c| format!(" | ctx: {}K", c / 1024))
            .unwrap_or_default();
        println!("  • {}{}", model.id, ctx);
    }
    println!();

    // -----------------------------------------------------------------------
    // 6. Tool / function calling
    // -----------------------------------------------------------------------
    println!("🔧 Tool calling");
    println!("{}", "─".repeat(50));

    let tools = vec![ToolDefinition::function(
        "get_weather",
        "Retrieve weather information for a given city",
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["city"]
        }),
    )];

    let tool_messages = vec![ChatMessage::user(
        "What is the weather in Paris and London today?",
    )];

    let tool_response = provider
        .chat_with_tools(&tool_messages, &tools, Some(ToolChoice::auto()), None)
        .await?;

    if tool_response.tool_calls.is_empty() {
        println!("Response (no tool call): {}", tool_response.content);
    } else {
        for tc in &tool_response.tool_calls {
            println!("Tool call: {} — args: {}", tc.name(), tc.arguments());
        }
    }
    println!();

    println!("✅ All examples completed successfully!");
    Ok(())
}

/// Compute cosine similarity between two float vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}
