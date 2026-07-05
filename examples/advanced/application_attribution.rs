//! Application attribution example
//!
//! Shows how to propagate caller identity (app ID, request ID) to upstream LLM APIs.
//! Provider HTTP headers/body fields come from `create_llm_provider_with_context`.
//!
//! For OTEL span attributes (`gen_ai.application.*`), wrap a **concrete** provider type:
//!
//! ```ignore
//! use edgequake_llm::providers::{OllamaProvider, TracingProvider};
//! let inner = OllamaProvider::from_env()?.with_application_context(ctx.clone())?;
//! let provider = TracingProvider::new(inner).with_application_context(ctx);
//! ```
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
use edgequake_llm::provider_catalog::ProviderCatalog;
use edgequake_llm::traits::ChatMessage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏷️  EdgeQuake LLM - Application Attribution Example\n");

    let ctx = ApplicationContextBuilder::new()
        .app_id("edgequake-demo")
        .app_name("EdgeQuake Demo")
        .request_id(format!("demo-{}", uuid::Uuid::new_v4()))
        .build()?;

    let provider_name = std::env::var("EDGEQUAKE_LLM_PROVIDER").unwrap_or_else(|_| "openai".into());
    let model = match provider_name.as_str() {
        "ollama" => "gemma4:latest",
        _ => "gpt-4o-mini",
    };

    if let Some(descriptor) = ProviderCatalog::get(&provider_name) {
        println!(
            "Attribution support for {provider_name}: {:?}\n",
            descriptor.attribution_support()
        );
    }

    let provider =
        ProviderFactory::create_llm_provider_with_context(&provider_name, model, ctx.clone())
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
