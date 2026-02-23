//! Gemini chat example — focused on conversation patterns.
//!
//! Run: cargo run --example gemini_chat
//! Requires: GEMINI_API_KEY
//!
//! Shows:
//!   1. Simple question/answer
//!   2. System prompt (pirate, translator, etc.)
//!   3. Multi-turn conversation with context
//!   4. Temperature and max_tokens control
//!   5. JSON mode structured output

use edgequake_llm::providers::gemini::GeminiProvider;
use edgequake_llm::traits::{ChatMessage, CompletionOptions, LLMProvider};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = GeminiProvider::from_env()?;
    println!(
        "Provider: {} | Model: {}",
        LLMProvider::name(&provider),
        LLMProvider::model(&provider),
    );
    println!("{}", "-".repeat(60));

    // ── 1. Simple Q&A ────────────────────────────────────────────────────
    println!("\n=== 1. Simple Q&A ===");
    let resp = provider
        .complete("What is 2 + 2? Reply with just the number.")
        .await?;
    println!("Answer: {}", resp.content);

    // ── 2. System prompt: pirate speak ─────────────────────────────────
    println!("\n=== 2. System prompt (pirate) ===");
    let messages = vec![
        ChatMessage::system(
            "You are a pirate. Always respond in pirate speak. Keep replies short.",
        ),
        ChatMessage::user("How are you today?"),
    ];
    let resp = provider
        .chat(
            &messages,
            Some(&CompletionOptions {
                max_tokens: Some(80),
                ..Default::default()
            }),
        )
        .await?;
    println!("Pirate: {}", resp.content);

    // ── 3. System prompt: translator ──────────────────────────────────────
    println!("\n=== 3. System prompt (translator) ===");
    let messages = vec![
        ChatMessage::system("You are a French translator. Translate the user's text to French. Only output the translation."),
        ChatMessage::user("The weather is beautiful today."),
    ];
    let resp = provider.chat(&messages, None).await?;
    println!("French: {}", resp.content);

    // ── 4. Multi-turn conversation ──────────────────────────────────────
    println!("\n=== 4. Multi-turn conversation ===");
    let mut history = vec![ChatMessage::system(
        "You are a helpful math tutor. Explain concepts simply.",
    )];

    // Turn 1
    history.push(ChatMessage::user("What is a prime number?"));
    let resp = provider
        .chat(
            &history,
            Some(&CompletionOptions {
                max_tokens: Some(100),
                ..Default::default()
            }),
        )
        .await?;
    println!("Turn 1 Q: What is a prime number?");
    println!("Turn 1 A: {}\n", resp.content);
    history.push(ChatMessage::assistant(&resp.content));

    // Turn 2
    history.push(ChatMessage::user("Give me the first 5 prime numbers."));
    let resp = provider
        .chat(
            &history,
            Some(&CompletionOptions {
                max_tokens: Some(100),
                ..Default::default()
            }),
        )
        .await?;
    println!("Turn 2 Q: Give me the first 5 prime numbers.");
    println!("Turn 2 A: {}\n", resp.content);
    history.push(ChatMessage::assistant(&resp.content));

    // Turn 3 (testing context retention)
    history.push(ChatMessage::user("Is the sum of those numbers also prime?"));
    let resp = provider
        .chat(
            &history,
            Some(&CompletionOptions {
                max_tokens: Some(100),
                ..Default::default()
            }),
        )
        .await?;
    println!("Turn 3 Q: Is the sum of those numbers also prime?");
    println!("Turn 3 A: {}", resp.content);

    // ── 5. Temperature comparison ────────────────────────────────────────
    println!("\n=== 5. Temperature comparison ===");
    let prompt = "Generate a creative name for a cat.";

    let low = CompletionOptions {
        temperature: Some(0.0),
        max_tokens: Some(30),
        ..Default::default()
    };
    let high = CompletionOptions {
        temperature: Some(1.5),
        max_tokens: Some(30),
        ..Default::default()
    };

    let resp_low = provider.complete_with_options(prompt, &low).await?;
    let resp_high = provider.complete_with_options(prompt, &high).await?;
    println!("Low temp (0.0): {}", resp_low.content);
    println!("High temp (1.5): {}", resp_high.content);

    // ── 6. JSON mode ─────────────────────────────────────────────────────
    println!("\n=== 6. JSON structured output ===");
    let json_opts = CompletionOptions::json_mode();
    let messages = vec![
        ChatMessage::system("You output JSON only. No markdown, no explanation."),
        ChatMessage::user(
            "Create a JSON object describing a book: { \"title\": string, \"author\": string, \"year\": number, \"genres\": [string] }. Use 'The Hobbit'."
        ),
    ];
    let resp = provider.chat(&messages, Some(&json_opts)).await?;
    println!("JSON: {}", resp.content);

    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&resp.content) {
        println!("  Title: {}", v["title"]);
        println!("  Author: {}", v["author"]);
        println!("  Year: {}", v["year"]);
    }

    println!("\n=== Done! Chat features demonstrated. ===");
    Ok(())
}
