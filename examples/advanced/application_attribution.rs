//! Application attribution example
//!
//! Shows how to propagate caller identity (app ID, request ID) to upstream LLM APIs.
//!
//! Run with any configured provider, e.g.:
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! cargo run --example application_attribution
//! ```
//!
//! Or with Ollama Cloud:
//! ```bash
//! export OLLAMA_API_KEY=...
//! export EDGEQUAKE_LLM_PROVIDER=ollama
//! cargo run --example application_attribution
//! ```

use edgequake_llm::application_context::ApplicationContextBuilder;
use edgequake_llm::factory::ProviderFactory;
use edgequake_llm::traits::ChatMessage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏷️  EdgeQuake LLM - Application Attribution Example\n");

    let ctx = ApplicationContextBuilder::new()
        .app_id("edgequake-demo")
        .app_name("EdgeQuake Demo")
        .request_id(format!("demo-{}", uuid::Uuid::new_v4()))
        .build()?;

    let provider = ProviderFactory::create_llm_provider_with_context("openai", "gpt-4o-mini", ctx.clone())
        .or_else(|_| {
            ProviderFactory::create_llm_provider_with_context("ollama", "gemma4:latest", ctx)
        })?;

    println!("Provider: {}", provider.name());
    println!("Model: {}\n", provider.model());

    let messages = vec![ChatMessage::user("Say hi in 3 words.")];
    let response = provider.chat(&messages, None).await?;

    println!("Response: {}", response.content.trim());

    Ok(())
}
