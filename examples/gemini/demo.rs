//! Gemini provider demo — full API coverage walkthrough.
//!
//! Run: cargo run --example gemini_demo
//! Requires: GEMINI_API_KEY
//!
//! Covers:
//!   1. Provider construction (from env)
//!   2. Simple completion
//!   3. Multi-turn chat with system prompt
//!   4. Streaming
//!   5. Tool calling (function)
//!   6. JSON mode / structured output
//!   7. Vision (multimodal base64 image)
//!   8. Embeddings (single + batch + similarity)
//!   9. Model listing (dynamic discovery)
//!  10. Model selection (Gemini 2.5 Flash, Pro, 3 Flash, 3 Pro)

use edgequake_llm::providers::gemini::GeminiProvider;
use edgequake_llm::traits::{
    ChatMessage, CompletionOptions, EmbeddingProvider, FunctionDefinition, LLMProvider, ToolChoice,
    ToolDefinition,
};
use futures::StreamExt;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ------------------------------------------------------------------ 1 --
    // Build from env var GEMINI_API_KEY (loads .env automatically).
    // ------------------------------------------------------------------ 1 --
    let provider = GeminiProvider::from_env()?;
    println!("Provider  : {}", LLMProvider::name(&provider));
    println!("Model     : {}", LLMProvider::model(&provider));
    println!("CtxLength : {}", provider.max_context_length());
    println!();

    // ------------------------------------------------------------------ 2 --
    // Simple one-shot completion
    // ------------------------------------------------------------------ 2 --
    println!("=== 1. Simple completion ===");
    let resp = provider
        .complete("Write a short haiku about Rust programming.")
        .await?;
    println!("{}", resp.content);
    println!(
        "(tokens: {}p + {}c = {}t)",
        resp.prompt_tokens, resp.completion_tokens, resp.total_tokens
    );
    if let Some(thinking) = &resp.thinking_tokens {
        println!("(thinking tokens: {})", thinking);
    }
    println!();

    // ------------------------------------------------------------------ 3 --
    // Multi-turn chat with system prompt
    // ------------------------------------------------------------------ 3 --
    println!("=== 2. Multi-turn chat ===");
    let messages = vec![
        ChatMessage::system("You are a helpful assistant who replies concisely in 1-2 sentences."),
        ChatMessage::user("What is the capital of France?"),
    ];
    let opts = CompletionOptions {
        max_tokens: Some(100),
        ..Default::default()
    };
    let resp = provider.chat(&messages, Some(&opts)).await?;
    println!("Assistant: {}", resp.content);
    println!();

    // Continue the conversation
    let messages2 = vec![
        ChatMessage::system("You are a helpful assistant who replies concisely."),
        ChatMessage::user("What is the capital of France?"),
        ChatMessage::assistant(&resp.content),
        ChatMessage::user("What is its population?"),
    ];
    let resp2 = provider.chat(&messages2, Some(&opts)).await?;
    println!("Follow-up: {}", resp2.content);
    println!();

    // ------------------------------------------------------------------ 4 --
    // Streaming
    // ------------------------------------------------------------------ 4 --
    println!("=== 3. Streaming ===");
    print!("Streaming: ");
    let mut stream = provider
        .stream("Count from 1 to 5, one number per word.")
        .await?;
    let mut chunk_count = 0;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(text) => {
                print!("{}", text);
                io::stdout().flush()?;
                chunk_count += 1;
            }
            Err(e) => eprintln!("\nStream error: {e}"),
        }
    }
    println!("\n(chunks received: {})\n", chunk_count);

    // ------------------------------------------------------------------ 5 --
    // Tool calling
    // ------------------------------------------------------------------ 5 --
    println!("=== 4. Tool calling ===");
    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_weather".to_string(),
            description: "Get current weather for a city".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }),
            strict: None,
        },
    }];

    let tool_messages = vec![ChatMessage::user("What's the weather in Tokyo?")];

    let tool_resp = provider
        .chat_with_tools(&tool_messages, &tools, Some(ToolChoice::auto()), None)
        .await?;

    if tool_resp.has_tool_calls() {
        for tc in &tool_resp.tool_calls {
            println!("Tool call: {}({})", tc.name(), tc.arguments());
        }
    } else {
        println!("Response (no tool call): {}", tool_resp.content);
    }
    println!();

    // ------------------------------------------------------------------ 6 --
    // JSON mode / structured output
    // ------------------------------------------------------------------ 6 --
    println!("=== 5. JSON mode ===");
    let json_messages = vec![ChatMessage::user(
        "Return a JSON object with fields 'name' (string) and 'age' (number) for a person named Alice who is 30.",
    )];
    let json_opts = CompletionOptions::json_mode();
    let json_resp = provider.chat(&json_messages, Some(&json_opts)).await?;
    println!("JSON: {}", json_resp.content);
    // Validate it's parseable
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json_resp.content) {
        println!("Parsed: name={}, age={}", parsed["name"], parsed["age"]);
    }
    println!();

    // ------------------------------------------------------------------ 7 --
    // Vision (multimodal base64 image)
    // ------------------------------------------------------------------ 7 --
    println!("=== 6. Vision (base64 image) ===");
    // Minimal 1x1 red pixel PNG
    let tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
    let image = edgequake_llm::ImageData::new(tiny_png, "image/png");
    let vision_messages = vec![ChatMessage::user_with_images(
        "What color is this single-pixel image? Just say the color.",
        vec![image],
    )];
    let vision_resp = provider.chat(&vision_messages, None).await?;
    println!("Vision: {}", vision_resp.content);
    println!();

    // ------------------------------------------------------------------ 8 --
    // Embeddings
    // ------------------------------------------------------------------ 8 --
    println!("=== 7. Embeddings ===");
    let emb_provider = GeminiProvider::from_env()?;
    println!(
        "Embedding model: {} (dim={})",
        EmbeddingProvider::model(&emb_provider),
        emb_provider.dimension()
    );

    // Single embedding
    let embedding = emb_provider.embed_one("Hello, world!").await?;
    println!(
        "Single embedding: {} dims, first 5: {:?}",
        embedding.len(),
        &embedding[..5.min(embedding.len())]
    );

    // Batch embeddings
    let texts = vec![
        "Rust is a systems programming language.".to_string(),
        "Python is great for data science.".to_string(),
        "JavaScript powers the web.".to_string(),
    ];
    let embeddings = emb_provider.embed(&texts).await?;
    println!("Batch: {} embeddings generated", embeddings.len());

    // Cosine similarity
    let sim = cosine_similarity(&embeddings[0], &embeddings[1]);
    println!("Similarity(Rust, Python) = {:.4}", sim);
    let sim2 = cosine_similarity(&embeddings[0], &embeddings[2]);
    println!("Similarity(Rust, JavaScript) = {:.4}", sim2);
    println!();

    // ------------------------------------------------------------------ 9 --
    // Model listing
    // ------------------------------------------------------------------ 9 --
    println!("=== 8. Model listing ===");
    let models = provider.list_models().await?;
    println!("Available models ({} total):", models.models.len());
    for model in models.models.iter().take(10) {
        println!("  {} — {}", model.name, model.display_name);
    }
    if models.models.len() > 10 {
        println!("  ... and {} more", models.models.len() - 10);
    }
    println!();

    // ------------------------------------------------------------------ 10 --
    // Temperature control
    // ------------------------------------------------------------------ 10 --
    println!("=== 9. Temperature ===");
    let low_temp = CompletionOptions::with_temperature(0.0);
    let resp_low = provider
        .complete_with_options("Say just 'hello'", &low_temp)
        .await?;
    println!("Low temp (0.0): {}", resp_low.content);

    let high_temp = CompletionOptions::with_temperature(1.5);
    let resp_high = provider
        .complete_with_options("Say just 'hello'", &high_temp)
        .await?;
    println!("High temp (1.5): {}", resp_high.content);
    println!();

    println!("=== Done! All Gemini features demonstrated. ===");
    Ok(())
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}
