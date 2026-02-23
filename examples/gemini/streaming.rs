//! Gemini streaming example — real-time token streaming with thinking support.
//!
//! Run: cargo run --example gemini_streaming
//! Requires: GEMINI_API_KEY
//!
//! Shows:
//!   1. Basic text streaming (stream method)
//!   2. Chat streaming with tools (chat_with_tools_stream)
//!   3. Thinking content streaming (Gemini 2.5+/3.x)

use edgequake_llm::providers::gemini::GeminiProvider;
use edgequake_llm::traits::{
    ChatMessage, CompletionOptions, FunctionDefinition, LLMProvider, StreamChunk, ToolDefinition,
};
use futures::StreamExt;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = GeminiProvider::from_env()?;
    println!(
        "Provider: {} | Model: {} | Thinking: {}",
        LLMProvider::name(&provider),
        LLMProvider::model(&provider),
        if provider.supports_thinking() {
            "yes"
        } else {
            "no"
        }
    );
    println!("{}", "-".repeat(60));

    // ── 1. Basic text streaming ──────────────────────────────────────────
    println!("\n=== 1. Basic streaming ===");
    print!("Streaming: ");
    let mut stream = provider
        .stream("Write a short poem about the ocean.")
        .await?;
    let mut full_response = String::new();
    let mut chunks = 0;

    while let Some(result) = stream.next().await {
        match result {
            Ok(text) => {
                print!("{}", text);
                io::stdout().flush()?;
                full_response.push_str(&text);
                chunks += 1;
            }
            Err(e) => {
                eprintln!("\nStream error: {}", e);
                break;
            }
        }
    }
    println!(
        "\n\n(chunks: {}, total chars: {})",
        chunks,
        full_response.len()
    );

    // ── 2. Chat streaming with tools ──────────────────────────────────────
    println!("\n=== 2. Chat streaming with tools ===");

    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_temperature".to_string(),
            description: "Get the current temperature for a location".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'San Francisco'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }),
            strict: None,
        },
    }];

    let messages = vec![
        ChatMessage::system("You are a helpful weather assistant."),
        ChatMessage::user("What is the temperature in Paris right now?"),
    ];

    let mut stream = provider
        .chat_with_tools_stream(&messages, &tools, None, None)
        .await?;

    print!("Streaming with tools: ");
    while let Some(result) = stream.next().await {
        match result {
            Ok(chunk) => match chunk {
                StreamChunk::Content(text) => {
                    print!("{}", text);
                    io::stdout().flush()?;
                }
                StreamChunk::ThinkingContent {
                    text, tokens_used, ..
                } => {
                    print!("[thinking: {}]", &text[..text.len().min(60)]);
                    if let Some(tokens) = tokens_used {
                        print!(" ({} tokens)", tokens);
                    }
                    io::stdout().flush()?;
                }
                StreamChunk::ToolCallDelta {
                    function_name,
                    function_arguments,
                    ..
                } => {
                    if let Some(name) = function_name {
                        print!("\n  -> Tool: {}", name);
                    }
                    if let Some(args) = function_arguments {
                        print!("({})", args);
                    }
                    io::stdout().flush()?;
                }
                StreamChunk::Finished { reason, .. } => {
                    println!("\n  [finished: {}]", reason);
                }
            },
            Err(e) => {
                eprintln!("\nStream error: {}", e);
                break;
            }
        }
    }
    println!();

    // ── 3. Streaming with options ────────────────────────────────────────
    println!("\n=== 3. Streaming with options (low temp) ===");
    let messages = vec![
        ChatMessage::system("You are a concise assistant. Reply in one sentence."),
        ChatMessage::user("Explain quantum computing."),
    ];
    let opts = CompletionOptions {
        temperature: Some(0.2),
        max_tokens: Some(100),
        ..Default::default()
    };

    let mut stream = provider
        .chat_with_tools_stream(&messages, &[], None, Some(&opts))
        .await?;

    print!("Streaming: ");
    while let Some(result) = stream.next().await {
        match result {
            Ok(StreamChunk::Content(text)) => {
                print!("{}", text);
                io::stdout().flush()?;
            }
            Ok(StreamChunk::ThinkingContent { text, .. }) => {
                // Show a summary of thinking
                let preview = if text.len() > 80 {
                    format!("{}...", &text[..80])
                } else {
                    text
                };
                print!("\n  [thought: {}]", preview);
                io::stdout().flush()?;
            }
            Ok(StreamChunk::Finished { reason, .. }) => {
                println!("\n  [done: {}]", reason);
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("\nError: {}", e);
                break;
            }
        }
    }
    println!();

    println!("=== Done! Streaming features demonstrated. ===");
    Ok(())
}
