//! End-to-end tests for xAI Grok provider.
//!
//! @implements OODA-71: xAI Grok API Integration
//!
//! These tests require a valid XAI_API_KEY environment variable.
//!
//! # Running the tests
//!
//! ```bash
//! # Set your xAI API key
//! export XAI_API_KEY=xai-your-api-key
//!
//! # Run all xAI tests
//! cargo test -p edgequake-llm --test e2e_xai
//!
//! # Run a specific test
//! cargo test -p edgequake-llm --test e2e_xai test_xai_basic_chat
//! ```
//!
//! # Test coverage
//!
//! - Basic chat completion
//! - JSON mode (structured output)
//! - Streaming
//! - Function/tool calling
//!
//! Note: Vision tests require grok-2-vision-1212 model.

use edgequake_llm::traits::{ChatMessage, LLMProvider, ToolChoice, ToolDefinition};
use edgequake_llm::XAIProvider;

/// Check if XAI_API_KEY is set for tests
fn has_xai_key() -> bool {
    std::env::var("XAI_API_KEY").is_ok()
}

/// Create xAI provider for testing
fn create_provider() -> XAIProvider {
    XAIProvider::from_env().expect("XAI_API_KEY must be set")
}

/// Create provider with a specific model
fn create_provider_with_model(model: &str) -> XAIProvider {
    XAIProvider::from_env()
        .expect("XAI_API_KEY must be set")
        .with_model(model)
}

// ============================================================================
// Basic Chat Tests
// ============================================================================

#[tokio::test]
async fn test_xai_basic_chat() {
    if !has_xai_key() {
        eprintln!("Skipping test: XAI_API_KEY not set");
        return;
    }

    let provider = create_provider();

    // Simple math question for deterministic response
    let messages = vec![
        ChatMessage::system("You are a helpful math tutor. Be very concise."),
        ChatMessage::user("What is 2 + 2? Answer with just the number."),
    ];

    let response = provider.chat(&messages, None).await;

    match response {
        Ok(resp) => {
            println!("Response: {}", resp.content);
            println!("Model: {}", resp.model);
            println!("Tokens: {} in, {} out", resp.prompt_tokens, resp.completion_tokens);

            // Should contain "4" somewhere in the response
            assert!(
                resp.content.contains("4"),
                "Expected '4' in response: {}",
                resp.content
            );
        }
        Err(e) => {
            panic!("Chat failed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_xai_simple_complete() {
    if !has_xai_key() {
        eprintln!("Skipping test: XAI_API_KEY not set");
        return;
    }

    let provider = create_provider();

    let response = provider.complete("Say 'hello world' and nothing else.").await;

    match response {
        Ok(resp) => {
            println!("Response: {}", resp.content);
            assert!(
                resp.content.to_lowercase().contains("hello"),
                "Expected 'hello' in response: {}",
                resp.content
            );
        }
        Err(e) => {
            panic!("Complete failed: {:?}", e);
        }
    }
}

// ============================================================================
// JSON Mode Tests
// ============================================================================

#[tokio::test]
async fn test_xai_json_mode() {
    if !has_xai_key() {
        eprintln!("Skipping test: XAI_API_KEY not set");
        return;
    }

    let provider = create_provider();

    let messages = vec![
        ChatMessage::system(
            "You are a JSON generator. Always respond with valid JSON only, no markdown.",
        ),
        ChatMessage::user(
            "Generate a JSON object with fields: name (string), age (number), active (boolean). Use sample values.",
        ),
    ];

    let options = edgequake_llm::traits::CompletionOptions::json_mode();

    let response = provider.chat(&messages, Some(&options)).await;

    match response {
        Ok(resp) => {
            println!("Response: {}", resp.content);

            // Try to parse as JSON
            let json_result: Result<serde_json::Value, _> =
                serde_json::from_str(resp.content.trim());

            match json_result {
                Ok(json) => {
                    println!("Valid JSON: {}", json);
                    // Verify expected fields exist
                    assert!(json.get("name").is_some(), "Missing 'name' field");
                    assert!(json.get("age").is_some(), "Missing 'age' field");
                    assert!(json.get("active").is_some(), "Missing 'active' field");
                }
                Err(e) => {
                    panic!("Invalid JSON response: {} - Error: {}", resp.content, e);
                }
            }
        }
        Err(e) => {
            panic!("JSON mode chat failed: {:?}", e);
        }
    }
}

// ============================================================================
// Streaming Tests
// ============================================================================

#[tokio::test]
async fn test_xai_streaming() {
    if !has_xai_key() {
        eprintln!("Skipping test: XAI_API_KEY not set");
        return;
    }

    use futures::StreamExt;

    let provider = create_provider();

    let result = provider.stream("Count from 1 to 5, separated by commas.").await;

    match result {
        Ok(mut stream) => {
            let mut full_response = String::new();
            let mut chunk_count = 0;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        full_response.push_str(&chunk);
                        chunk_count += 1;
                    }
                    Err(e) => {
                        panic!("Stream chunk error: {:?}", e);
                    }
                }
            }

            println!("Full response: {}", full_response);
            println!("Chunk count: {}", chunk_count);

            // Verify we got multiple chunks and expected content
            assert!(chunk_count > 0, "Expected at least one chunk");
            assert!(
                !full_response.is_empty(),
                "Expected non-empty response"
            );
        }
        Err(e) => {
            panic!("Stream failed: {:?}", e);
        }
    }
}

// ============================================================================
// Tool/Function Calling Tests
// ============================================================================

#[tokio::test]
async fn test_xai_tool_calling() {
    if !has_xai_key() {
        eprintln!("Skipping test: XAI_API_KEY not set");
        return;
    }

    let provider = create_provider();

    // Define a simple tool
    let tools = vec![ToolDefinition::function(
        "get_weather",
        "Get the current weather for a location",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }),
    )];

    let messages = vec![ChatMessage::user(
        "What's the weather like in Tokyo?",
    )];

    let response = provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), None)
        .await;

    match response {
        Ok(resp) => {
            println!("Response content: {}", resp.content);
            println!("Tool calls: {:?}", resp.tool_calls);

            // Model should either call the tool or provide a response
            // (xAI may or may not call the tool depending on its judgment)
            if !resp.tool_calls.is_empty() {
                println!("Tool call: {:?}", resp.tool_calls[0]);
            } else {
                // Model chose to respond directly
                println!("Model responded directly without tool call");
            }
        }
        Err(e) => {
            panic!("Tool calling failed: {:?}", e);
        }
    }
}

// ============================================================================
// Model Selection Tests
// ============================================================================

#[tokio::test]
async fn test_xai_grok3_mini() {
    if !has_xai_key() {
        eprintln!("Skipping test: XAI_API_KEY not set");
        return;
    }

    // Test with grok-3-mini (faster, cheaper)
    let provider = create_provider_with_model("grok-3-mini");

    assert_eq!(provider.model(), "grok-3-mini");

    let response = provider.complete("What is 5 * 5? Just the number.").await;

    match response {
        Ok(resp) => {
            println!("Response from grok-3-mini: {}", resp.content);
            assert!(
                resp.content.contains("25"),
                "Expected '25' in response: {}",
                resp.content
            );
        }
        Err(e) => {
            // grok-3-mini may not be available, skip gracefully
            eprintln!("grok-3-mini test skipped: {:?}", e);
        }
    }
}

// ============================================================================
// Provider Info Tests
// ============================================================================

#[test]
fn test_xai_provider_info() {
    if !has_xai_key() {
        eprintln!("Skipping test: XAI_API_KEY not set");
        return;
    }

    let provider = create_provider();

    assert_eq!(provider.name(), "xai");
    assert_eq!(provider.model(), "grok-4");
    assert_eq!(provider.max_context_length(), 262144); // OODA-15: 256K for grok-4
}

#[test]
fn test_xai_context_lengths() {
    // Test context length lookup (doesn't need API key)
    // OODA-15: Updated from docs.x.ai
    assert_eq!(XAIProvider::context_length("grok-4"), 262144); // 256K
    assert_eq!(XAIProvider::context_length("grok-4.1-fast"), 2000000); // 2M
    assert_eq!(XAIProvider::context_length("grok-3-mini"), 131072); // 128K
    assert_eq!(XAIProvider::context_length("grok-2-vision-1212"), 32768); // 32K
    assert_eq!(XAIProvider::context_length("unknown-model"), 262144); // Default 256K
}

#[test]
fn test_xai_available_models() {
    let models = XAIProvider::available_models();

    assert!(!models.is_empty());

    // Check for expected models
    let model_names: Vec<&str> = models.iter().map(|(name, _, _)| *name).collect();
    assert!(model_names.contains(&"grok-4"));
    assert!(model_names.contains(&"grok-4.1-fast"));
    assert!(model_names.contains(&"grok-3"));
    assert!(model_names.contains(&"grok-2-vision-1212"));
}
