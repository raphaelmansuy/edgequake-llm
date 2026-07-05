//! Ollama Cloud example
//!
//! Demonstrates calling Ollama Cloud via `https://ollama.com` with Bearer auth.
//!
//! Run with:
//! ```bash
//! export OLLAMA_API_KEY=your_key_from_ollama.com
//! export OLLAMA_MODEL=gpt-oss:120b   # or any cloud model you have access to
//! cargo run --example ollama_cloud
//! ```
//!
//! See: https://docs.ollama.com/cloud

use edgequake_llm::{ChatMessage, CompletionOptions, LLMProvider, OllamaProvider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("☁️  EdgeQuake LLM - Ollama Cloud Example\n");

    if std::env::var("OLLAMA_API_KEY").is_err() {
        eprintln!("❌ Set OLLAMA_API_KEY (create one at https://ollama.com)");
        return Ok(());
    }

    let provider = OllamaProvider::from_env_cloud()?;

    println!("Provider: {}", provider.name());
    println!("Host: {}", provider.host());
    println!("Model: {}", provider.model());
    println!("Cloud mode: {}\n", provider.is_cloud());

    let messages = vec![ChatMessage::user(
        "Reply with exactly one word: hello",
    )];

    let options = CompletionOptions {
        max_tokens: Some(16),
        temperature: Some(0.0),
        ..Default::default()
    };

    let response = provider.chat(&messages, Some(&options)).await?;

    println!("Response: {}", response.content.trim());
    println!(
        "Tokens: {} prompt + {} completion",
        response.prompt_tokens, response.completion_tokens
    );

    Ok(())
}
