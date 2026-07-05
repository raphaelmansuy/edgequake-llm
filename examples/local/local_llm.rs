//! Local LLM example
//!
//! Demonstrates using local LLM providers (Ollama and LM Studio) and Ollama Cloud.
//!
//! Run with: cargo run --example local_llm
//! Requires: Either Ollama or LM Studio running locally, or OLLAMA_API_KEY for cloud
//!
//! # Setup
//!
//! ## Ollama local (recommended)
//! ```bash
//! # Install Ollama: https://ollama.ai
//! ollama pull llama3.2
//! ollama serve
//! ```
//!
//! ## Ollama Cloud
//! ```bash
//! export OLLAMA_API_KEY=your_key_from_ollama.com
//! export OLLAMA_MODEL=gpt-oss:120b
//! cargo run --example local_llm
//! ```
//!
//! ## LM Studio
//! - Download from https://lmstudio.ai
//! - Load a model (e.g., "gemma-2-2b-it")
//! - Start local server (default port 1234)
//!
//! This example shows:
//! - Creating local LLM providers
//! - Checking server availability
//! - Unified interface across local providers
//! - No cloud API keys required

use edgequake_llm::{
    ChatMessage, CompletionOptions, LLMProvider, LMStudioProvider, OllamaProvider,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏠 EdgeQuake LLM - Local LLM Example\n");
    println!("{}", "─".repeat(60));

    // Try Ollama Cloud when API key is set
    if std::env::var("OLLAMA_API_KEY").is_ok() {
        println!("\n☁️  Checking Ollama Cloud (https://ollama.com)...\n");
        match try_ollama_cloud().await {
            Ok(()) => println!("✅ Ollama Cloud test successful!"),
            Err(e) => println!("⚠️  Ollama Cloud not available: {}", e),
        }
        println!("{}", "─".repeat(60));
    }

    // Try Ollama local
    println!("\n📦 Checking Ollama (http://localhost:11434)...\n");
    match try_ollama().await {
        Ok(()) => println!("✅ Ollama test successful!"),
        Err(e) => println!("⚠️  Ollama not available: {}", e),
    }

    println!("{}", "─".repeat(60));

    // Try LM Studio
    println!("\n📦 Checking LM Studio (http://localhost:1234)...\n");
    match try_lmstudio().await {
        Ok(()) => println!("✅ LM Studio test successful!"),
        Err(e) => println!("⚠️  LM Studio not available: {}", e),
    }

    println!("\n{}", "─".repeat(60));
    println!("💡 Tip: Install Ollama (https://ollama.ai) for easiest local setup");
    println!("   Run: ollama pull llama3.2 && ollama serve");

    Ok(())
}

/// Test Ollama Cloud provider.
async fn try_ollama_cloud() -> Result<(), Box<dyn std::error::Error>> {
    let provider = OllamaProvider::from_env_cloud()?;

    println!("Provider: {}", provider.name());
    println!("Host: {}", provider.host());
    println!("Cloud mode: {}", provider.is_cloud());

    let messages = vec![ChatMessage::user("Say hello in exactly 3 words.")];
    let options = CompletionOptions {
        max_tokens: Some(50),
        temperature: Some(0.7),
        ..Default::default()
    };

    let response = provider.chat(&messages, Some(&options)).await?;

    println!("\nResponse: {}", response.content);
    println!(
        "Tokens: {} prompt + {} completion",
        response.prompt_tokens, response.completion_tokens
    );

    Ok(())
}

/// Test Ollama provider.
async fn try_ollama() -> Result<(), Box<dyn std::error::Error>> {
    // Create provider with defaults
    let provider = OllamaProvider::builder()
        .host("http://localhost:11434")
        .model("llama3.2") // Small, fast model
        .build()?;

    println!("Provider: {}", provider.name());
    println!("Model: {}", provider.model());

    // Create a simple chat message
    let messages = vec![ChatMessage::user("Say hello in exactly 3 words.")];

    // Generate response with options
    let options = CompletionOptions {
        max_tokens: Some(50),
        temperature: Some(0.7),
        ..Default::default()
    };

    let response = provider.chat(&messages, Some(&options)).await?;

    println!("\nResponse: {}", response.content);
    println!(
        "Tokens: {} prompt + {} completion",
        response.prompt_tokens, response.completion_tokens
    );

    Ok(())
}

/// Test LM Studio provider.
async fn try_lmstudio() -> Result<(), Box<dyn std::error::Error>> {
    // Create provider with defaults
    let provider = LMStudioProvider::builder()
        .host("http://localhost:1234")
        .model("gemma-2-2b-it") // Common LM Studio model
        .build()?;

    println!("Provider: {}", provider.name());
    println!("Model: {}", provider.model());

    // Create a simple chat message
    let messages = vec![ChatMessage::user("Say hello in exactly 3 words.")];

    // Generate response with options
    let options = CompletionOptions {
        max_tokens: Some(50),
        temperature: Some(0.7),
        ..Default::default()
    };

    let response = provider.chat(&messages, Some(&options)).await?;

    println!("\nResponse: {}", response.content);
    println!(
        "Tokens: {} prompt + {} completion",
        response.prompt_tokens, response.completion_tokens
    );

    Ok(())
}
