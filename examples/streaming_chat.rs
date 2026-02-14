//! Streaming completion example
//!
//! Demonstrates async streaming responses with real-time token output.
//!
//! Run with: cargo run --example streaming_chat
//! Requires: OPENAI_API_KEY environment variable
//!
//! This example shows:
//! - Setting up a streaming completion
//! - Processing chunks as they arrive
//! - Proper cleanup and error handling

use edgequake_llm::{LLMProvider, OpenAIProvider};
use futures::StreamExt;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    // Initialize provider
    let provider = OpenAIProvider::new(&api_key).with_model("gpt-4o-mini");

    println!("🌊 EdgeQuake LLM - Streaming Example\n");

    // Check if provider supports streaming
    if !provider.supports_streaming() {
        eprintln!("Provider does not support streaming!");
        return Ok(());
    }

    let prompt = "Write a haiku about Rust programming. Think step by step.";

    println!("📤 Prompt: {}\n", prompt);
    println!("{}", "─".repeat(50));

    // Get streaming response using the stream() method
    let mut stream = provider.stream(prompt).await?;

    // Process chunks as they arrive
    let mut full_response = String::new();
    let mut chunk_count = 0;

    while let Some(result) = stream.next().await {
        match result {
            Ok(content) => {
                // Print each chunk without newline for smooth streaming effect
                print!("{}", content);
                io::stdout().flush()?;

                full_response.push_str(&content);
                chunk_count += 1;
            }
            Err(e) => {
                eprintln!("\n❌ Stream error: {}", e);
                break;
            }
        }
    }

    println!("\n");
    println!("{}", "─".repeat(50));
    println!("📊 Statistics:");
    println!("   Chunks received: {}", chunk_count);
    println!("   Total characters: {}", full_response.len());

    Ok(())
}

