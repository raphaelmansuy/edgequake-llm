//! NVIDIA NIM provider example
//!
//! Demonstrates chat, streaming, live model listing, and free-model filtering.
//!
//! Setup:
//!   export NVIDIA_API_KEY=nvapi-...
//!   cargo run --example nvidia_chat

use edgequake_llm::traits::{ChatMessage, CompletionOptions, LLMProvider};
use edgequake_llm::NvidiaProvider;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("NVIDIA NIM example\n");

    let provider = NvidiaProvider::from_env()?;
    println!(
        "provider={} model={}",
        LLMProvider::name(&provider),
        LLMProvider::model(&provider)
    );

    // 1) Basic chat
    let messages = vec![
        ChatMessage::system("You are concise and practical."),
        ChatMessage::user("Explain Rust lifetimes in two short sentences."),
    ];
    let resp = provider.chat(&messages, None).await?;
    println!("\n[chat]\n{}", resp.content);

    // 2) Streaming
    let mut stream = provider
        .stream("Write a 3-line poem about GPUs and inference.")
        .await?;
    println!("\n[stream]");
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(text) => print!("{}", text),
            Err(e) => {
                eprintln!("\nstream error: {}", e);
                break;
            }
        }
    }
    println!();

    // 3) Model listing and free-tier filtering
    let models = provider.list_models().await?;
    println!("\n[models] total={}", models.data.len());

    let free_count = models.data.iter().filter(|m| m.is_free).count();
    println!("[models] free={}", free_count);

    for m in models.data.iter().filter(|m| m.is_free).take(5) {
        println!("  free: {} (owner={})", m.id, m.owned_by);
    }

    // 4) Thinking example with reasoning_effort
    let thinking_provider = provider.with_model("deepseek-ai/deepseek-v4-flash");
    let opts = CompletionOptions {
        reasoning_effort: Some("high".to_string()),
        max_tokens: Some(400),
        ..Default::default()
    };
    let think_resp = thinking_provider
        .chat(
            &[ChatMessage::user(
                "Solve 17*19 and explain each arithmetic step briefly.",
            )],
            Some(&opts),
        )
        .await?;
    println!("\n[thinking]\n{}", think_resp.content);

    Ok(())
}
