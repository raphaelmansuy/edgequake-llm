//! OpenAI provider demo — full API coverage walkthrough.
//!
//! Run: cargo run --example openai_demo
//! Requires: OPENAI_API_KEY
//!
//! Covers:
//!   1. Provider construction (from env)
//!   2. Simple completion
//!   3. Multi-turn chat
//!   4. Streaming
//!   5. Tool calling (function)
//!   6. JSON mode / structured output
//!   7. Vision (multimodal URL image)
//!   8. Model families & capabilities reference

use edgequake_llm::providers::openai::OpenAIProvider;
use edgequake_llm::traits::{
    ChatMessage, CompletionOptions, FunctionDefinition, LLMProvider, ToolDefinition,
};
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ------------------------------------------------------------------ 1 --
    // Build from env var OPENAI_API_KEY (loads .env automatically).
    // ------------------------------------------------------------------ 1 --
    let provider = OpenAIProvider::from_env()?;
    println!("Provider  : {}", LLMProvider::name(&provider));
    println!("Model     : {}", LLMProvider::model(&provider));
    println!("CtxLength : {}", provider.max_context_length());
    println!();

    // ------------------------------------------------------------------ 2 --
    // Simple one-shot completion
    // ------------------------------------------------------------------ 2 --
    println!("=== Simple completion ===");
    let poem = provider
        .complete("Write a short haiku about Rust programming.")
        .await?;
    println!("{}", poem.content);
    println!(
        "(tokens: {}p + {}c = {})",
        poem.prompt_tokens, poem.completion_tokens, poem.total_tokens
    );
    println!();

    // ------------------------------------------------------------------ 3 --
    // Multi-turn chat
    // ------------------------------------------------------------------ 3 --
    println!("=== Multi-turn chat ===");
    let messages = vec![
        ChatMessage::system("You are a helpful assistant who replies concisely."),
        ChatMessage::user("What is the capital of France?"),
    ];
    // Note: some models (gpt-5-mini, o1, o3, o4) only accept the default
    // temperature (1.0). Omit temperature when model-agnostic behaviour is needed.
    let opts = CompletionOptions {
        max_tokens: Some(50),
        ..Default::default()
    };
    let resp = provider.chat(&messages, Some(&opts)).await?;
    println!("Assistant: {}", resp.content);
    println!();

    // ------------------------------------------------------------------ 4 --
    // Streaming
    // ------------------------------------------------------------------ 4 --
    println!("=== Streaming ===");
    print!("Streaming: ");
    let mut stream = provider
        .stream("Count from 1 to 5, one number per word.")
        .await?;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(text) => print!("{}", text),
            Err(e) => eprintln!("\nStream error: {e}"),
        }
    }
    println!("\n");

    // ------------------------------------------------------------------ 5 --
    // Tool calling
    // ------------------------------------------------------------------ 5 --
    println!("=== Tool calling ===");
    use edgequake_llm::traits::ToolChoice;

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
            strict: Some(true),
        },
    }];

    let messages = vec![ChatMessage::user("What's the weather like in Paris?")];
    let tool_resp = provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), None)
        .await?;

    if tool_resp.tool_calls.is_empty() {
        println!("Content: {}", tool_resp.content);
    } else {
        for tc in &tool_resp.tool_calls {
            println!(
                "Tool call: {} args={}",
                tc.function.name, tc.function.arguments
            );
        }
    }
    println!();

    // ------------------------------------------------------------------ 6 --
    // JSON mode — force the model to output valid JSON.
    // Requires: model that supports response_format=json_object
    //   (gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, gpt-5-mini, …)
    // ------------------------------------------------------------------ 6 --
    println!("=== JSON mode ===");
    if provider.supports_json_mode() {
        let json_opts = CompletionOptions::json_mode();
        let json_messages = vec![
            ChatMessage::system("You are a helpful assistant that always replies with JSON."),
            ChatMessage::user(
                "Return the three largest countries by area as JSON array: \
                 [{\"name\":\"...\",\"area_km2\":...}]",
            ),
        ];
        let resp = provider.chat(&json_messages, Some(&json_opts)).await?;
        println!("JSON: {}", resp.content);
    } else {
        println!("(model {} does not support JSON mode)", provider.model());
    }
    println!();

    // ------------------------------------------------------------------ 7 --
    // Vision — multimodal message with a URL image.
    // Switch model to gpt-4o-mini which has vision capability.
    // ------------------------------------------------------------------ 7 --
    println!("=== Vision (URL image) ===");
    {
        use edgequake_llm::traits::ImageData;
        let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        let vision_provider = OpenAIProvider::new(&api_key).with_model("gpt-4o-mini");
        let img = ImageData::from_url(
            "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files\
/master/ComputerVision/Images/landmark.jpg",
        )
        .with_detail("low");
        let resp = vision_provider
            .chat(
                &[
                    ChatMessage::system("You are a concise image analyst. One sentence max."),
                    ChatMessage::user_with_images("What landmark is shown?", vec![img]),
                ],
                Some(&CompletionOptions {
                    max_tokens: Some(80),
                    ..Default::default()
                }),
            )
            .await?;
        println!("Vision: {}", resp.content);
    }
    println!();

    // ------------------------------------------------------------------ 8 --
    // Model families reference
    // ------------------------------------------------------------------ 8 --
    println!("=== OpenAI model families ===");
    println!("  GPT-5 family  : gpt-5, gpt-5-mini            — latest, temp=1.0 only");
    println!("  GPT-4o family : gpt-4o, gpt-4o-mini         — vision, tools, JSON, full temp");
    println!("  GPT-4.1 family: gpt-4.1, gpt-4.1-mini       — vision, tools, JSON, 1M ctx");
    println!("  o-series      : o1, o3, o4-mini              — reasoning, no streaming, temp=1");
    println!("  Embedding     : text-embedding-3-small/large — embeddings only, no chat");
    println!();

    println!("All examples completed successfully.");
    Ok(())
}
