//! VertexAI streaming examples — real-time token output, thinking, tool deltas.
//!
//! Run: cargo run --example vertexai_streaming
//! Requires:
//!   - GOOGLE_CLOUD_PROJECT
//!   - Authenticated via `gcloud auth login` (or GOOGLE_ACCESS_TOKEN)

use edgequake_llm::providers::gemini::GeminiProvider;
use edgequake_llm::traits::{
    ChatMessage, CompletionOptions, FunctionDefinition, LLMProvider, StreamChunk, ToolChoice,
    ToolDefinition,
};
use futures::StreamExt;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = GeminiProvider::from_env_vertex_ai()?;
    println!(
        "VertexAI Streaming Examples — model: {}\n",
        LLMProvider::model(&provider)
    );

    // ────────────────────────────────────────────────────────────── 1 ──
    // Basic text streaming
    // ────────────────────────────────────────────────────────────── 1 ──
    println!("=== 1. Basic text streaming ===");
    print!(">>> ");
    let mut stream = provider
        .stream("Write a haiku about cloud computing.")
        .await?;
    let mut chunk_count = 0;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(text) => {
                print!("{}", text);
                io::stdout().flush()?;
                chunk_count += 1;
            }
            Err(e) => eprintln!("\nError: {e}"),
        }
    }
    println!("\n(received {} chunks)\n", chunk_count);

    // ────────────────────────────────────────────────────────────── 2 ──
    // Streaming with chat messages (multi-turn)
    // ────────────────────────────────────────────────────────────── 2 ──
    println!("=== 2. Chat streaming ===");
    let messages = vec![
        ChatMessage::system("You are a concise assistant. Keep answers under 3 sentences."),
        ChatMessage::user("Explain the difference between TCP and UDP."),
    ];
    let opts = CompletionOptions {
        max_tokens: Some(200),
        ..Default::default()
    };
    print!(">>> ");
    let mut stream = provider
        .chat_with_tools_stream(&messages, &[], None, Some(&opts))
        .await?;
    let mut thinking_seen = false;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(StreamChunk::Content(text)) => {
                print!("{}", text);
                io::stdout().flush()?;
            }
            Ok(StreamChunk::ThinkingContent { text, .. }) => {
                if !thinking_seen {
                    print!("[thinking] ");
                    thinking_seen = true;
                }
                print!("{}", text);
                io::stdout().flush()?;
            }
            Ok(StreamChunk::Finished { .. }) => {}
            Ok(StreamChunk::ToolCallDelta { .. }) => {}
            Err(e) => eprintln!("\nError: {e}"),
        }
    }
    println!("\n");

    // ────────────────────────────────────────────────────────────── 3 ──
    // Streaming with thinking content (Gemini 2.5+)
    // ────────────────────────────────────────────────────────────── 3 ──
    println!("=== 3. Streaming with thinking content ===");
    if provider.supports_thinking() {
        let messages = vec![ChatMessage::user(
            "What is the probability of rolling at least one 6 in four rolls of a die?",
        )];

        let mut stream = provider
            .chat_with_tools_stream(&messages, &[], None, None)
            .await?;

        let mut in_thinking = false;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(StreamChunk::Content(text)) => {
                    if in_thinking {
                        println!("\n--- End thinking ---\n");
                        in_thinking = false;
                    }
                    print!("{}", text);
                    io::stdout().flush()?;
                }
                Ok(StreamChunk::ThinkingContent { text, .. }) => {
                    if !in_thinking {
                        println!("--- Thinking ---");
                        in_thinking = true;
                    }
                    print!("{}", text);
                    io::stdout().flush()?;
                }
                Ok(StreamChunk::ToolCallDelta { .. }) => {}
                Ok(StreamChunk::Finished { reason, .. }) => {
                    println!();
                    println!("(finished: {})", reason);
                }
                Err(e) => eprintln!("\nError: {e}"),
            }
        }
    } else {
        println!("(Current model does not support thinking — try gemini-2.5-flash)");
    }
    println!();

    // ────────────────────────────────────────────────────────────── 4 ──
    // Streaming with tool call deltas
    // ────────────────────────────────────────────────────────────── 4 ──
    println!("=== 4. Streaming with tool calls ===");
    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_weather".to_string(),
            description: "Get current weather for a location".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string", "description": "City name" },
                    "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
                },
                "required": ["location"]
            }),
            strict: None,
        },
    }];

    let messages = vec![ChatMessage::user(
        "What's the weather like in Paris right now?",
    )];
    let mut stream = provider
        .chat_with_tools_stream(&messages, &tools, Some(ToolChoice::auto()), None)
        .await?;

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(StreamChunk::Content(text)) => {
                print!("{}", text);
                io::stdout().flush()?;
            }
            Ok(StreamChunk::ToolCallDelta {
                id,
                function_name,
                function_arguments,
                ..
            }) => {
                if let Some(n) = &function_name {
                    println!("Tool call: {}", n);
                }
                if let Some(tid) = &id {
                    println!("  id: {}", tid);
                }
                if let Some(args) = &function_arguments {
                    print!("{}", args);
                    io::stdout().flush()?;
                }
            }
            Ok(StreamChunk::Finished { .. }) => {
                println!();
            }
            _ => {}
        }
    }
    println!();

    // ────────────────────────────────────────────────────────────── 5 ──
    // Timed streaming — measure time-to-first-token
    // ────────────────────────────────────────────────────────────── 5 ──
    println!("=== 5. Time-to-first-token measurement ===");
    let start = std::time::Instant::now();
    let mut stream = provider
        .stream("Say hello in 5 different languages.")
        .await?;
    let mut first_token_time = None;
    let mut total_tokens = 0;
    while let Some(chunk) = stream.next().await {
        if let Ok(text) = chunk {
            if first_token_time.is_none() {
                first_token_time = Some(start.elapsed());
            }
            print!("{}", text);
            io::stdout().flush()?;
            total_tokens += 1;
        }
    }
    let total_time = start.elapsed();
    println!();
    if let Some(ttft) = first_token_time {
        println!("Time-to-first-token: {:.0?}", ttft);
    }
    println!(
        "Total time: {:.0?} ({} chunks, {:.1} chunks/sec)",
        total_time,
        total_tokens,
        total_tokens as f64 / total_time.as_secs_f64()
    );
    println!();

    println!("=== Done! ===");
    Ok(())
}
