//! Vertex AI provider demo — full API coverage walkthrough.
//!
//! Run: cargo run --example vertexai_demo
//! Requires:
//!   - GOOGLE_CLOUD_PROJECT (your GCP project ID)
//!   - Authenticated via `gcloud auth login` (or GOOGLE_ACCESS_TOKEN)
//!   - GOOGLE_CLOUD_REGION (optional, default: us-central1)
//!
//! Covers:
//!   1. VertexAI provider construction (auto gcloud token)
//!   2. Simple completion
//!   3. Multi-turn chat with system prompt
//!   4. Streaming
//!   5. Tool calling (function)
//!   6. JSON mode / structured output
//!   7. Vision (multimodal base64 image)
//!   8. Embeddings via :predict endpoint
//!   9. Thinking support (Gemini 2.5+/3.x)

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
    // Build from VertexAI credentials
    // Uses GOOGLE_CLOUD_PROJECT + gcloud auth print-access-token
    // ------------------------------------------------------------------ 1 --
    println!("=== VertexAI Provider Demo ===\n");

    let provider = GeminiProvider::from_env_vertex_ai().unwrap_or_else(|e| {
        eprintln!("❌  Auth error: {}", e);
        eprintln!();
        eprintln!("To authenticate, run one of:");
        eprintln!("  gcloud auth login");
        eprintln!("  gcloud auth application-default login");
        eprintln!();
        eprintln!("Then set your project:");
        eprintln!("  export GOOGLE_CLOUD_PROJECT=saas-app-001");
        std::process::exit(1);
    });
    println!("Provider  : {}", LLMProvider::name(&provider));
    println!("Model     : {}", LLMProvider::model(&provider));
    println!("CtxLength : {}", provider.max_context_length());
    println!(
        "Thinking  : {}",
        if provider.supports_thinking() {
            "yes"
        } else {
            "no"
        }
    );
    println!(
        "Streaming : {}",
        if provider.supports_streaming() {
            "yes"
        } else {
            "no"
        }
    );
    println!();

    // ------------------------------------------------------------------ 2 --
    // Simple one-shot completion
    // ------------------------------------------------------------------ 2 --
    println!("=== 1. Simple completion ===");
    let resp = provider
        .complete("What is the capital of Japan? Reply in one sentence.")
        .await?;
    println!("{}", resp.content);
    println!(
        "(tokens: {}p + {}c = {}t)",
        resp.prompt_tokens, resp.completion_tokens, resp.total_tokens
    );
    if let Some(thinking) = &resp.thinking_tokens {
        println!("(thinking tokens: {})", thinking);
    }
    if let Some(thinking_text) = &resp.thinking_content {
        let preview = if thinking_text.len() > 200 {
            format!("{}...", &thinking_text[..200])
        } else {
            thinking_text.clone()
        };
        println!("(thinking: {})", preview);
    }
    println!();

    // ------------------------------------------------------------------ 3 --
    // Multi-turn chat
    // ------------------------------------------------------------------ 3 --
    println!("=== 2. Multi-turn chat ===");
    let messages = vec![
        ChatMessage::system("You are a helpful assistant who replies concisely in 1-2 sentences."),
        ChatMessage::user("Who painted the Mona Lisa?"),
    ];
    let opts = CompletionOptions {
        max_tokens: Some(100),
        ..Default::default()
    };
    let resp = provider.chat(&messages, Some(&opts)).await?;
    println!("Q: Who painted the Mona Lisa?");
    println!("A: {}", resp.content);

    // Follow-up
    let messages2 = vec![
        ChatMessage::system("You are a helpful assistant who replies concisely."),
        ChatMessage::user("Who painted the Mona Lisa?"),
        ChatMessage::assistant(&resp.content),
        ChatMessage::user("When was it painted?"),
    ];
    let resp2 = provider.chat(&messages2, Some(&opts)).await?;
    println!("Q: When was it painted?");
    println!("A: {}", resp2.content);
    println!();

    // ------------------------------------------------------------------ 4 --
    // Streaming
    // ------------------------------------------------------------------ 4 --
    println!("=== 3. Streaming ===");
    print!("Streaming: ");
    let mut stream = provider
        .stream("List the planets in our solar system, one per line.")
        .await?;
    let mut chunks = 0;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(text) => {
                print!("{}", text);
                io::stdout().flush()?;
                chunks += 1;
            }
            Err(e) => eprintln!("\nStream error: {e}"),
        }
    }
    println!("\n(chunks: {})\n", chunks);

    // ------------------------------------------------------------------ 5 --
    // Tool calling
    // ------------------------------------------------------------------ 5 --
    println!("=== 4. Tool calling ===");
    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_population".to_string(),
            description: "Get the population of a city or country".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or country name"
                    }
                },
                "required": ["location"]
            }),
            strict: None,
        },
    }];

    let tool_messages = vec![ChatMessage::user("What is the population of Germany?")];
    let tool_resp = provider
        .chat_with_tools(&tool_messages, &tools, Some(ToolChoice::auto()), None)
        .await?;

    if tool_resp.has_tool_calls() {
        for tc in &tool_resp.tool_calls {
            println!("Tool: {}({})", tc.name(), tc.arguments());
        }
    } else {
        println!("Response: {}", tool_resp.content);
    }
    println!();

    // ------------------------------------------------------------------ 6 --
    // JSON mode
    // ------------------------------------------------------------------ 6 --
    println!("=== 5. JSON mode ===");
    let json_opts = CompletionOptions::json_mode();
    let json_messages = vec![ChatMessage::user(
        "Create a JSON object: { \"language\": string, \"year_created\": number, \"paradigm\": string } for Rust.",
    )];
    let json_resp = provider.chat(&json_messages, Some(&json_opts)).await?;
    println!("JSON: {}", json_resp.content);
    println!();

    // ------------------------------------------------------------------ 7 --
    // Vision
    // ------------------------------------------------------------------ 7 --
    println!("=== 6. Vision ===");
    // Minimal 1x1 PNG
    let tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
    let image = edgequake_llm::ImageData::new(tiny_png, "image/png");
    let vision_messages = vec![ChatMessage::user_with_images(
        "Describe this image in one word.",
        vec![image],
    )];
    let vision_resp = provider.chat(&vision_messages, None).await?;
    println!("Vision: {}", vision_resp.content);
    println!();

    // ------------------------------------------------------------------ 8 --
    // Embeddings via VertexAI :predict endpoint
    // ------------------------------------------------------------------ 8 --
    println!("=== 7. VertexAI embeddings ===");

    // Create a fresh VertexAI provider for embeddings
    let emb_provider = GeminiProvider::from_env_vertex_ai()?;
    println!(
        "Embedding model: {} (dim={})",
        EmbeddingProvider::model(&emb_provider),
        emb_provider.dimension()
    );

    let embedding = emb_provider.embed_one("Hello from Vertex AI!").await?;
    println!("Embedding: {} dims", embedding.len());
    println!("First 5 values: {:?}", &embedding[..5.min(embedding.len())]);

    // Batch
    let texts = vec![
        "Cloud computing is powerful.".to_string(),
        "Machine learning drives innovation.".to_string(),
    ];
    let batch = emb_provider.embed(&texts).await?;
    println!("Batch: {} embeddings", batch.len());

    // Similarity
    let sim = cosine_similarity(&batch[0], &batch[1]);
    println!("Similarity: {:.4}", sim);
    println!();

    // ------------------------------------------------------------------ 9 --
    // Thinking support (Gemini 2.5+ models)
    // ------------------------------------------------------------------ 9 --
    println!("=== 8. Thinking support ===");
    if provider.supports_thinking() {
        let messages = vec![ChatMessage::user(
            "What is 15% of 240? Show your reasoning step by step.",
        )];
        let resp = provider
            .chat(
                &messages,
                Some(&CompletionOptions {
                    max_tokens: Some(200),
                    ..Default::default()
                }),
            )
            .await?;
        println!("Answer: {}", resp.content);
        if let Some(tokens) = resp.thinking_tokens {
            println!("Thinking tokens used: {}", tokens);
        }
        if let Some(thinking) = &resp.thinking_content {
            let preview = if thinking.len() > 300 {
                format!("{}...", &thinking[..300])
            } else {
                thinking.clone()
            };
            println!("Thinking: {}", preview);
        }
    } else {
        println!("(Current model does not support thinking)");
    }
    println!();

    println!("=== Done! All VertexAI features demonstrated. ===");
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
