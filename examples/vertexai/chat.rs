//! VertexAI chat examples — multi-turn conversations, system prompts, personas.
//!
//! Run: cargo run --example vertexai_chat
//! Requires:
//!   - GOOGLE_CLOUD_PROJECT
//!   - Authenticated via `gcloud auth login` (or GOOGLE_ACCESS_TOKEN)

use edgequake_llm::providers::gemini::GeminiProvider;
use edgequake_llm::traits::{ChatMessage, CompletionOptions, LLMProvider};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Build VertexAI-based provider
    let provider = GeminiProvider::from_env_vertex_ai()?;
    println!(
        "VertexAI Chat Examples — model: {}\n",
        LLMProvider::model(&provider)
    );

    // ────────────────────────────────────────────────────────────── 1 ──
    // Basic Q&A
    // ────────────────────────────────────────────────────────────── 1 ──
    println!("=== 1. Basic question-answer ===");
    let messages = vec![ChatMessage::user(
        "What programming language is the Linux kernel mainly written in?",
    )];
    let resp = provider
        .chat(
            &messages,
            Some(&CompletionOptions {
                max_tokens: Some(100),
                ..Default::default()
            }),
        )
        .await?;
    println!("A: {}\n", resp.content);

    // ────────────────────────────────────────────────────────────── 2 ──
    // System prompt — concise teacher persona
    // ────────────────────────────────────────────────────────────── 2 ──
    println!("=== 2. System prompt — teacher persona ===");
    let messages = vec![
        ChatMessage::system(
            "You are a patient teacher who explains concepts clearly in 2-3 sentences, \
             using analogies when helpful.",
        ),
        ChatMessage::user("What is recursion in programming?"),
    ];
    let resp = provider
        .chat(
            &messages,
            Some(&CompletionOptions {
                max_tokens: Some(200),
                ..Default::default()
            }),
        )
        .await?;
    println!("Teacher: {}\n", resp.content);

    // ────────────────────────────────────────────────────────────── 3 ──
    // System prompt — pirate translator
    // ────────────────────────────────────────────────────────────── 3 ──
    println!("=== 3. System prompt — pirate translator ===");
    let messages = vec![
        ChatMessage::system(
            "You are a pirate translator. Translate anything the user says into pirate speak. \
             Keep the same meaning but use pirate vocabulary, arrr!",
        ),
        ChatMessage::user("I would like to order a latte and a croissant, please."),
    ];
    let resp = provider.chat(&messages, None).await?;
    println!("Pirate: {}\n", resp.content);

    // ────────────────────────────────────────────────────────────── 4 ──
    // Multi-turn conversation
    // ────────────────────────────────────────────────────────────── 4 ──
    println!("=== 4. Multi-turn conversation ===");
    let system = "You are a knowledgeable history assistant. Keep answers to 2-3 sentences.";

    // Turn 1
    let messages = vec![
        ChatMessage::system(system),
        ChatMessage::user("When was the Great Wall of China built?"),
    ];
    let resp1 = provider
        .chat(
            &messages,
            Some(&CompletionOptions {
                max_tokens: Some(150),
                ..Default::default()
            }),
        )
        .await?;
    println!("Q: When was the Great Wall of China built?");
    println!("A: {}", resp1.content);

    // Turn 2
    let messages = vec![
        ChatMessage::system(system),
        ChatMessage::user("When was the Great Wall of China built?"),
        ChatMessage::assistant(&resp1.content),
        ChatMessage::user("How long is it?"),
    ];
    let resp2 = provider
        .chat(
            &messages,
            Some(&CompletionOptions {
                max_tokens: Some(150),
                ..Default::default()
            }),
        )
        .await?;
    println!("Q: How long is it?");
    println!("A: {}", resp2.content);

    // Turn 3
    let messages = vec![
        ChatMessage::system(system),
        ChatMessage::user("When was the Great Wall of China built?"),
        ChatMessage::assistant(&resp1.content),
        ChatMessage::user("How long is it?"),
        ChatMessage::assistant(&resp2.content),
        ChatMessage::user("Can you see it from space?"),
    ];
    let resp3 = provider
        .chat(
            &messages,
            Some(&CompletionOptions {
                max_tokens: Some(150),
                ..Default::default()
            }),
        )
        .await?;
    println!("Q: Can you see it from space?");
    println!("A: {}\n", resp3.content);

    // ────────────────────────────────────────────────────────────── 5 ──
    // Temperature comparison
    // ────────────────────────────────────────────────────────────── 5 ──
    println!("=== 5. Temperature comparison ===");
    let prompt = "Give me a creative name for a coffee shop.";

    for temp in [0.0, 0.5, 1.0, 1.5] {
        let messages = vec![ChatMessage::user(prompt)];
        let opts = CompletionOptions {
            max_tokens: Some(30),
            temperature: Some(temp),
            ..Default::default()
        };
        let resp = provider.chat(&messages, Some(&opts)).await?;
        println!("  temp={:.1}: {}", temp, resp.content.trim());
    }
    println!();

    // ────────────────────────────────────────────────────────────── 6 ──
    // JSON mode
    // ────────────────────────────────────────────────────────────── 6 ──
    println!("=== 6. JSON mode ===");
    let json_opts = CompletionOptions::json_mode();
    let messages = vec![ChatMessage::user(
        "Return a JSON object with keys: name, population, continent for the city of Tokyo.",
    )];
    let resp = provider.chat(&messages, Some(&json_opts)).await?;
    println!("JSON: {}\n", resp.content);

    println!("=== Done! ===");
    Ok(())
}
