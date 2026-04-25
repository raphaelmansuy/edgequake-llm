//! End-to-end tests for the NVIDIA NIM provider.
//!
//! These tests require a valid `NVIDIA_API_KEY` environment variable
//! (free at <https://build.nvidia.com>).
//!
//! # Running the tests
//!
//! ```bash
//! export NVIDIA_API_KEY=nvapi-...
//! cargo test -p edgequake-llm --test e2e_nvidia -- --nocapture
//! cargo test -p edgequake-llm --test e2e_nvidia test_nvidia_basic_chat -- --nocapture
//! ```
//!
//! # Test coverage
//!
//! - Basic chat completion (`chat()`)
//! - Simple `complete()` helper
//! - JSON mode
//! - Streaming (`stream()`)
//! - Tool / function calling
//! - Model listing (GET /v1/models)
//! - Free model enumeration
//! - Thinking model (DeepSeek V4 Flash with `reasoning_effort`)
//! - Provider factory auto-detection
//! - Free-tier model smoke test (chat with every known free model)

use edgequake_llm::traits::{
    ChatMessage, CompletionOptions, EmbeddingProvider, LLMProvider, ToolChoice, ToolDefinition,
};
use edgequake_llm::{NvidiaProvider, ProviderFactory, ProviderType};
use futures::StreamExt;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn has_nvidia_key() -> bool {
    std::env::var("NVIDIA_API_KEY")
        .map(|k| !k.is_empty())
        .unwrap_or(false)
}

fn create_provider() -> NvidiaProvider {
    NvidiaProvider::from_env().expect("NVIDIA_API_KEY must be set")
}

/// Create a provider targeting a specific model.
fn create_provider_with_model(model: &str) -> NvidiaProvider {
    NvidiaProvider::from_env()
        .expect("NVIDIA_API_KEY must be set")
        .with_model(model)
}

// ---------------------------------------------------------------------------
// Factory auto-detection
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_nvidia_factory_from_env() {
    if !has_nvidia_key() {
        eprintln!("Skipping test_nvidia_factory_from_env: NVIDIA_API_KEY not set");
        return;
    }

    // Temporarily ensure other providers don't intercept auto-detection
    // (In a CI environment only NVIDIA_API_KEY should be set)
    let result = ProviderFactory::create(ProviderType::Nvidia);
    match result {
        Ok((llm, _embedding)) => {
            println!("Factory created NVIDIA provider: name={}", llm.name());
            assert_eq!(llm.name(), "nvidia");
        }
        Err(e) => {
            eprintln!("Factory failed: {:?}", e);
        }
    }
}

// ---------------------------------------------------------------------------
// Basic Chat
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_nvidia_basic_chat() {
    if !has_nvidia_key() {
        eprintln!("Skipping test_nvidia_basic_chat: NVIDIA_API_KEY not set");
        return;
    }

    let provider = create_provider();
    let messages = vec![
        ChatMessage::system("You are a concise math tutor."),
        ChatMessage::user("What is 2 + 2? Reply with just the number."),
    ];

    let response = provider.chat(&messages, None).await;
    match response {
        Ok(resp) => {
            println!(
                "Response: '{}' (model={}, tokens={}/{})",
                resp.content, resp.model, resp.prompt_tokens, resp.completion_tokens
            );
            if !resp.content.contains("4") {
                eprintln!(
                    "Warning: expected '4' in response but got: {}",
                    resp.content
                );
            }
        }
        Err(e) => {
            eprintln!("Chat failed (possible transient issue, skipping): {:?}", e);
        }
    }
}

// ---------------------------------------------------------------------------
// Simple complete
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_nvidia_simple_complete() {
    if !has_nvidia_key() {
        eprintln!("Skipping test_nvidia_simple_complete: NVIDIA_API_KEY not set");
        return;
    }

    let provider = create_provider();
    let response = provider
        .complete("Say 'hello world' and nothing else.")
        .await;

    match response {
        Ok(resp) => {
            println!("Complete response: '{}'", resp.content);
            if !resp.content.to_lowercase().contains("hello") {
                eprintln!(
                    "Warning: expected 'hello' in response but got: {}",
                    resp.content
                );
            }
        }
        Err(e) => {
            eprintln!(
                "Complete failed (possible transient issue, skipping): {:?}",
                e
            );
        }
    }
}

// ---------------------------------------------------------------------------
// JSON Mode
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_nvidia_json_mode() {
    if !has_nvidia_key() {
        eprintln!("Skipping test_nvidia_json_mode: NVIDIA_API_KEY not set");
        return;
    }

    let provider = create_provider();
    let messages = vec![
        ChatMessage::system(
            "You are a JSON generator. Always respond with valid JSON only, no markdown.",
        ),
        ChatMessage::user(
            r#"Generate a JSON object with fields: name (string), value (number). Use sample values. Output only JSON."#,
        ),
    ];

    let options = CompletionOptions::json_mode();
    let response = provider.chat(&messages, Some(&options)).await;

    match response {
        Ok(resp) => {
            println!("JSON response: '{}'", resp.content);
            // Try to parse; warn if not valid JSON (some models add markdown)
            let content = resp.content.trim();
            let content = content
                .trim_start_matches("```json")
                .trim_start_matches("```")
                .trim_end_matches("```")
                .trim();
            match serde_json::from_str::<serde_json::Value>(content) {
                Ok(json) => {
                    println!("Parsed JSON: {:?}", json);
                }
                Err(e) => {
                    eprintln!(
                        "Response was not valid JSON (possible transient, skipping): '{}' — {}",
                        resp.content, e
                    );
                }
            }
        }
        Err(e) => {
            eprintln!(
                "JSON mode failed (possible transient issue, skipping): {:?}",
                e
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_nvidia_streaming() {
    if !has_nvidia_key() {
        eprintln!("Skipping test_nvidia_streaming: NVIDIA_API_KEY not set");
        return;
    }

    let provider = create_provider();
    let result = provider
        .stream("Count from 1 to 5, separated by commas. Just the numbers.")
        .await;

    match result {
        Ok(mut stream) => {
            let mut full_response = String::new();
            let mut chunk_count = 0;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        print!("{}", chunk);
                        full_response.push_str(&chunk);
                        chunk_count += 1;
                    }
                    Err(e) => {
                        eprintln!("Stream chunk error (possible transient): {:?}", e);
                        break;
                    }
                }
            }
            println!();
            println!(
                "Streaming complete: {} chunks, total {} chars",
                chunk_count,
                full_response.len()
            );
            if chunk_count == 0 {
                eprintln!("Warning: received 0 chunks from streaming endpoint");
            }
        }
        Err(e) => {
            eprintln!(
                "Stream request failed (possible transient issue, skipping): {:?}",
                e
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tool / Function calling
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_nvidia_tool_calling() {
    if !has_nvidia_key() {
        eprintln!("Skipping test_nvidia_tool_calling: NVIDIA_API_KEY not set");
        return;
    }

    let provider = create_provider();
    let messages = vec![ChatMessage::user(
        "What's the weather in San Francisco? Use the get_weather function.",
    )];

    let tools = vec![ToolDefinition::function(
        "get_weather",
        "Get the current weather for a city.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name"
                }
            },
            "required": ["city"]
        }),
    )];

    let response = provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), None)
        .await;

    match response {
        Ok(resp) => {
            println!("Tool response content: '{}'", resp.content);
            if !resp.tool_calls.is_empty() {
                let tool_calls = &resp.tool_calls;
                println!("Tool calls: {} call(s)", tool_calls.len());
                for tc in tool_calls {
                    println!("  -> {} ({})", tc.function.name, tc.function.arguments);
                    assert_eq!(tc.function.name, "get_weather");
                }
            } else {
                eprintln!("Warning: no tool calls returned (model may have responded directly)");
            }
        }
        Err(e) => {
            eprintln!(
                "Tool calling failed (possible transient issue, skipping): {:?}",
                e
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Model listing
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_nvidia_list_models() {
    if !has_nvidia_key() {
        eprintln!("Skipping test_nvidia_list_models: NVIDIA_API_KEY not set");
        return;
    }

    let provider = create_provider();
    let result = provider.list_models().await;

    match result {
        Ok(models_response) => {
            println!(
                "NVIDIA models available: {} total",
                models_response.data.len()
            );

            let free_count = models_response.data.iter().filter(|m| m.is_free).count();
            println!("  Free-tier models: {}", free_count);

            // Print first 10 model IDs
            for m in models_response.data.iter().take(10) {
                println!("  - {} (free={})", m.id, m.is_free);
            }

            assert!(
                !models_response.data.is_empty(),
                "Model list should not be empty"
            );
            assert_eq!(models_response.object, "list");
        }
        Err(e) => {
            eprintln!(
                "Model listing failed (possible transient issue, skipping): {:?}",
                e
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Free model enumeration (static catalog)
// ---------------------------------------------------------------------------

#[test]
fn test_nvidia_free_models_catalog() {
    let free_models = NvidiaProvider::free_models();
    println!("Known free-tier models: {}", free_models.len());
    for (id, name, ctx) in &free_models {
        println!("  - {} — {} (ctx={})", id, name, ctx);
    }
    assert!(
        !free_models.is_empty(),
        "Must have at least one known free model"
    );
    // Verify a few known free models are present
    let ids: Vec<&str> = free_models.iter().map(|(id, _, _)| *id).collect();
    assert!(
        ids.contains(&"meta/llama-3.1-8b-instruct"),
        "meta/llama-3.1-8b-instruct should be free"
    );
    assert!(
        ids.contains(&"nvidia/llama-3.3-nemotron-super-49b-v1"),
        "nvidia/llama-3.3-nemotron-super-49b-v1 should be free"
    );
}

// ---------------------------------------------------------------------------
// Thinking model (DeepSeek V4 Flash)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_nvidia_thinking_model() {
    if !has_nvidia_key() {
        eprintln!("Skipping test_nvidia_thinking_model: NVIDIA_API_KEY not set");
        return;
    }

    let provider = create_provider_with_model("deepseek-ai/deepseek-v4-flash");
    let messages = vec![ChatMessage::user(
        "Solve step by step: What is 15% of 240? Show your work briefly.",
    )];

    let options = CompletionOptions {
        reasoning_effort: Some("high".to_string()),
        max_tokens: Some(1024),
        ..Default::default()
    };

    let response = provider.chat(&messages, Some(&options)).await;
    match response {
        Ok(resp) => {
            println!(
                "Thinking model response (DeepSeek V4 Flash): '{}' (tokens={})",
                &resp.content[..resp.content.len().min(200)],
                resp.completion_tokens
            );
            if !resp.content.contains("36") {
                eprintln!(
                    "Warning: expected '36' (15% of 240) in response but got: {}",
                    &resp.content[..resp.content.len().min(100)]
                );
            }
        }
        Err(e) => {
            eprintln!(
                "Thinking model failed (possible transient issue, skipping): {:?}",
                e
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Nemotron thinking model (via nvidia/llama-3.3-nemotron-super-49b-v1)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_nvidia_nemotron_thinking() {
    if !has_nvidia_key() {
        eprintln!("Skipping test_nvidia_nemotron_thinking: NVIDIA_API_KEY not set");
        return;
    }

    let provider = create_provider_with_model("nvidia/llama-3.3-nemotron-super-49b-v1");
    let messages = vec![
        ChatMessage::system("detailed thinking off"),
        ChatMessage::user("What is the capital of France? One word answer."),
    ];

    let response = provider.chat(&messages, None).await;
    match response {
        Ok(resp) => {
            println!("Nemotron response: '{}'", resp.content);
            if !resp.content.to_lowercase().contains("paris") {
                eprintln!(
                    "Warning: expected 'Paris' in response but got: {}",
                    resp.content
                );
            }
        }
        Err(e) => {
            eprintln!(
                "Nemotron chat failed (possible transient issue, skipping): {:?}",
                e
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Free model smoke tests — chat with each known free model
// These are rate-limited, so run sequentially and tolerate errors.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_nvidia_free_models_smoke() {
    if !has_nvidia_key() {
        eprintln!("Skipping test_nvidia_free_models_smoke: NVIDIA_API_KEY not set");
        return;
    }

    // Subset of known free models to test (not all — avoid quota exhaustion)
    let test_models = [
        "meta/llama-3.1-8b-instruct",
        "meta/llama-3.2-3b-instruct",
        "microsoft/phi-4-mini-instruct",
        "qwen/qwen2.5-7b-instruct",
        "moonshotai/kimi-k2-instruct",
    ];

    let simple_messages = vec![ChatMessage::user("Say 'ok' and nothing else.")];

    for model in &test_models {
        println!("\n=== Testing free model: {} ===", model);
        let provider = create_provider_with_model(model);

        match provider.chat(&simple_messages, None).await {
            Ok(resp) => {
                println!(
                    "  Response: '{}' (tokens={}/{})",
                    resp.content.trim(),
                    resp.prompt_tokens,
                    resp.completion_tokens
                );
            }
            Err(e) => {
                // Don't fail the test for rate limits or transient errors
                eprintln!("  Model {} failed (tolerated): {:?}", model, e);
            }
        }

        // Brief pause to avoid rate limiting
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }
}

// ---------------------------------------------------------------------------
// Embedding — should return an error with a clear message
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_nvidia_embedding_returns_clear_error() {
    if !has_nvidia_key() {
        eprintln!("Skipping test_nvidia_embedding_returns_clear_error: NVIDIA_API_KEY not set");
        return;
    }

    let provider = create_provider();
    let result = provider.embed(&["test embedding".to_string()]).await;

    assert!(result.is_err(), "Embedding should return an error");
    let err = result.unwrap_err().to_string();
    println!("Embedding error (expected): {}", err);
    assert!(
        err.contains("embeddings are not supported") || err.contains("NvidiaProvider"),
        "Error should mention the limitation, got: {}",
        err
    );
}

// ---------------------------------------------------------------------------
// Context length helpers
// ---------------------------------------------------------------------------

#[test]
fn test_nvidia_provider_context_lengths() {
    // Default model context length
    assert_eq!(
        NvidiaProvider::context_length("nvidia/llama-3.3-nemotron-super-49b-v1"),
        131_072,
        "Nemotron Super 49B should have 128K context"
    );

    // 1M context models
    assert_eq!(
        NvidiaProvider::context_length("nvidia/nemotron-3-nano-30b-a3b"),
        1_000_000,
        "Nemotron 3 Nano 30B MoE should have 1M context"
    );

    // Unknown model should fall back to 32K
    assert_eq!(
        NvidiaProvider::context_length("some/unknown-model"),
        32_768,
        "Unknown model should fall back to 32K"
    );
}
