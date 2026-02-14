//! Basic LLM completion example
//!
//! Run with: cargo run --example basic_completion
//! Requires: OPENAI_API_KEY environment variable

use edgequake_llm::{ChatMessage, LLMProvider, OpenAIProvider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    // Initialize provider
    let provider = OpenAIProvider::new(&api_key).with_model("gpt-4");

    println!("🤖 EdgeQuake LLM - Basic Completion Example\n");

    // Create message
    let messages = vec![ChatMessage::user("What is the capital of France? Answer in one word.")];

    println!("Sending request to OpenAI...");

    // Get completion
    let response = provider.chat(&messages, None).await?;

    println!("\n✨ Response: {}", response.content);
    println!(
        "📊 Tokens: {} prompt + {} completion = {} total",
        response.prompt_tokens,
        response.completion_tokens,
        response.total_tokens
    );

    Ok(())
}
