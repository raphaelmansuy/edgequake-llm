//! Multi-provider example showing provider abstraction
//!
//! Run with: cargo run --example multi_provider
//! Requires: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY

use edgequake_llm::{
    AnthropicProvider, ChatMessage, GeminiProvider, LLMProvider, OpenAIProvider,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 EdgeQuake LLM - Multi-Provider Example\n");

    let message = ChatMessage::user("Explain async/await in Rust in 2 sentences.");

    // Create providers (skip if API key not available)
    let mut providers: Vec<Box<dyn LLMProvider>> = vec![];

    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        providers.push(Box::new(OpenAIProvider::new(api_key)));
    }

    if let Ok(provider) = AnthropicProvider::from_env() {
        providers.push(Box::new(provider));
    }

    if let Ok(provider) = GeminiProvider::from_env() {
        providers.push(Box::new(provider));
    }

    if providers.is_empty() {
        eprintln!("❌ No API keys found. Please set at least one:");
        eprintln!("   - OPENAI_API_KEY");
        eprintln!("   - ANTHROPIC_API_KEY");
        eprintln!("   - GOOGLE_API_KEY");
        return Ok(());
    }

    // Try each provider
    for provider in providers {
        println!("🔄 Testing: {}", provider.name());
        println!("   Model: {}", provider.model());

        match provider.chat(&[message.clone()], None).await {
            Ok(response) => {
                println!("   ✅ Response: {}", response.content.lines().next().unwrap_or(""));
                println!("   📊 Tokens: {}\n", response.total_tokens);
            }
            Err(e) => {
                println!("   ❌ Error: {}\n", e);
            }
        }
    }

    Ok(())
}
