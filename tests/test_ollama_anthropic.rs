//! E2E test for AnthropicProvider with Ollama backend.
//!
//! Run with: cargo test --test test_ollama_anthropic -- --ignored --nocapture
//! Requires: Ollama running at localhost:11434 with gpt-oss:20b model

use edgequake_llm::providers::anthropic::AnthropicProvider;
use edgequake_llm::traits::{ChatMessage, FunctionDefinition, LLMProvider, ToolDefinition};
use futures::StreamExt;

#[tokio::test]
#[ignore] // Requires Ollama to be running
async fn test_anthropic_provider_with_ollama() {
    let provider = AnthropicProvider::new("ollama")
        .with_base_url("http://localhost:11434")
        .with_model("gpt-oss:20b");

    println!("Provider: {}, Model: {}", provider.name(), provider.model());

    let messages = vec![ChatMessage::user("Say hello in one word")];

    println!("Sending request...");
    let result = provider.chat(&messages, None).await;

    match &result {
        Ok(response) => {
            println!("Success: {}", response.content);
            assert!(!response.content.is_empty());
        }
        Err(e) => {
            println!("Error: {:?}", e);
            panic!("Request failed: {:?}", e);
        }
    }
}

#[tokio::test]
#[ignore] // Requires Ollama to be running
async fn test_anthropic_provider_with_tools() {
    // Test with tools - this is what the React agent uses
    let provider = AnthropicProvider::new("ollama")
        .with_base_url("http://localhost:11434")
        .with_model("gpt-oss:20b");

    println!("Provider: {}, Model: {}", provider.name(), provider.model());

    let messages = vec![ChatMessage::user("Create a file called test.txt")];

    // Define a simple tool
    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "write_file".to_string(),
            description: "Write content to a file".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path" },
                    "content": { "type": "string", "description": "File content" }
                },
                "required": ["path", "content"]
            }),
            strict: None,
        },
    }];

    println!("Sending request with tools...");
    let result = provider
        .chat_with_tools(&messages, &tools, None, None)
        .await;

    match &result {
        Ok(response) => {
            println!("Success: content={}", response.content);
            println!("Tool calls: {:?}", response.tool_calls);
            // The model should call the write_file tool
        }
        Err(e) => {
            println!("Error: {:?}", e);
            panic!("Request failed: {:?}", e);
        }
    }
}

#[tokio::test]
#[ignore] // Requires Ollama to be running
async fn test_anthropic_provider_with_tools_stream() {
    // Test with streaming tools - this is exactly what the React agent uses
    let provider = AnthropicProvider::new("ollama")
        .with_base_url("http://localhost:11434")
        .with_model("gpt-oss:20b");

    println!("Provider: {}, Model: {}", provider.name(), provider.model());

    let messages = vec![ChatMessage::user(
        "Create a file called test.txt with 'hello world'",
    )];

    // Define a simple tool
    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "write_file".to_string(),
            description: "Write content to a file".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path" },
                    "content": { "type": "string", "description": "File content" }
                },
                "required": ["path", "content"]
            }),
            strict: None,
        },
    }];

    println!("Sending streaming request with tools (same as React agent)...");
    let result = provider
        .chat_with_tools_stream(&messages, &tools, None, None)
        .await;

    match result {
        Ok(mut stream) => {
            println!("Stream started successfully!");
            let mut chunk_count = 0;
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        chunk_count += 1;
                        println!("  Chunk {}: {:?}", chunk_count, chunk);
                    }
                    Err(e) => {
                        println!("  Chunk error: {:?}", e);
                    }
                }
            }
            println!("Stream completed with {} chunks", chunk_count);
        }
        Err(e) => {
            println!("Error: {:?}", e);
            panic!("Streaming request failed: {:?}", e);
        }
    }
}
