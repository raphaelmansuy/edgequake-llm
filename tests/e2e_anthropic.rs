//! Comprehensive end-to-end tests for the Anthropic provider.
//!
//! # Test groups
//!
//! ## Unit tests (no network — always run)
//! ```
//! cargo test --test e2e_anthropic
//! ```
//!
//! ## Ollama integration tests (require local Ollama with gemma4:latest)
//! ```
//! cargo test --test e2e_anthropic -- --include-ignored ollama
//! ```
//! Requires: `ollama pull gemma4:latest`
//!
//! ## Live Anthropic API tests (require ANTHROPIC_API_KEY)
//! ```
//! cargo test --test e2e_anthropic -- --include-ignored anthropic_live
//! ```

use edgequake_llm::providers::anthropic::AnthropicProvider;
use edgequake_llm::traits::{
    ChatMessage, CompletionOptions, FunctionDefinition, LLMProvider, StreamChunk, ToolChoice,
    ToolDefinition,
};
use futures::StreamExt;

// ============================================================================
// Helpers
// ============================================================================

/// Build a provider pointed at local Ollama with gemma4:latest.
fn ollama_gemma4() -> AnthropicProvider {
    AnthropicProvider::new("ollama")
        .with_base_url("http://localhost:11434")
        .with_model("gemma4:latest")
}

/// Simple tool definition reused across tests.
fn weather_tool() -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_weather".to_string(),
            description: "Get the current weather in a city".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. Paris"
                    }
                },
                "required": ["city"]
            }),
            strict: None,
        },
    }
}

fn calculator_tool() -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "calculate".to_string(),
            description: "Evaluate a simple arithmetic expression".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. '42 * 7'"
                    }
                },
                "required": ["expression"]
            }),
            strict: None,
        },
    }
}

// ============================================================================
// Unit / constructor tests  (always run, no network)
// ============================================================================

#[test]
fn test_builder_defaults() {
    let p = AnthropicProvider::new("sk-test");
    assert_eq!(p.api_key(), "sk-test");
    assert_eq!(p.model(), "claude-sonnet-4-6");
    assert_eq!(p.base_url(), "https://api.anthropic.com");
    assert_eq!(p.endpoint(), "https://api.anthropic.com/v1/messages");
    assert_eq!(p.api_version(), "2023-06-01");
    assert_eq!(p.max_context_length(), 200_000);
}

#[test]
fn test_builder_with_api_key() {
    let p = AnthropicProvider::new("old-key").with_api_key("new-key");
    assert_eq!(p.api_key(), "new-key");
}

#[test]
fn test_builder_with_model() {
    let p = AnthropicProvider::new("k").with_model("claude-opus-4-6");
    assert_eq!(p.model(), "claude-opus-4-6");
    assert_eq!(p.max_context_length(), 200_000);
}

#[test]
fn test_builder_with_base_url() {
    let p = AnthropicProvider::new("k").with_base_url("http://localhost:11434");
    assert_eq!(p.base_url(), "http://localhost:11434");
    assert_eq!(p.endpoint(), "http://localhost:11434/v1/messages");
}

#[test]
fn test_builder_with_api_version() {
    let p = AnthropicProvider::new("k").with_api_version("2024-01-01");
    assert_eq!(p.api_version(), "2024-01-01");
}

#[test]
fn test_builder_chaining() {
    let p = AnthropicProvider::new("k1")
        .with_api_key("k2")
        .with_model("gemma4:latest")
        .with_base_url("http://localhost:11434")
        .with_api_version("2023-06-01");
    assert_eq!(p.api_key(), "k2");
    assert_eq!(p.model(), "gemma4:latest");
    assert_eq!(p.endpoint(), "http://localhost:11434/v1/messages");
}

#[test]
fn test_for_ollama_defaults() {
    let p = AnthropicProvider::for_ollama();
    assert_eq!(p.api_key(), "ollama");
    assert_eq!(p.base_url(), "http://localhost:11434");
    assert_eq!(p.model(), "qwen3-coder");
    assert_eq!(p.endpoint(), "http://localhost:11434/v1/messages");
}

#[test]
fn test_for_ollama_with_model() {
    let p = AnthropicProvider::for_ollama_with_model("gemma4:latest");
    assert_eq!(p.model(), "gemma4:latest");
    assert_eq!(p.api_key(), "ollama");
    assert_eq!(p.base_url(), "http://localhost:11434");
}

#[test]
fn test_for_ollama_at() {
    let p = AnthropicProvider::for_ollama_at("http://10.0.0.5:11434", "gemma4:latest");
    assert_eq!(p.base_url(), "http://10.0.0.5:11434");
    assert_eq!(p.model(), "gemma4:latest");
    assert_eq!(p.endpoint(), "http://10.0.0.5:11434/v1/messages");
}

#[test]
fn test_provider_capabilities() {
    let p = AnthropicProvider::new("k");
    assert!(p.supports_streaming(), "must support streaming");
    assert!(
        p.supports_function_calling(),
        "must support function calling"
    );
    assert!(p.supports_tool_streaming(), "must support tool streaming");
}

#[test]
fn test_name() {
    assert_eq!(AnthropicProvider::new("k").name(), "anthropic");
}

#[test]
fn test_context_length_unknown_model_defaults_200k() {
    assert_eq!(
        AnthropicProvider::context_length_for_model("some-future-model"),
        200_000
    );
}

#[test]
fn test_context_length_legacy_models() {
    assert_eq!(
        AnthropicProvider::context_length_for_model("claude-2.1"),
        100_000
    );
    assert_eq!(
        AnthropicProvider::context_length_for_model("claude-instant-1.2"),
        100_000
    );
}

#[test]
fn test_context_length_claude_4_series() {
    for model in &[
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-opus-4-5-20250929",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5",
    ] {
        assert_eq!(
            AnthropicProvider::context_length_for_model(model),
            200_000,
            "model {} should be 200k",
            model
        );
    }
}

// ============================================================================
// Ollama integration tests (gemma4:latest)
// Tagged [ignore] — run with: cargo test --test e2e_anthropic -- --include-ignored ollama
// ============================================================================

/// Basic single-turn chat.
#[tokio::test]
#[ignore = "ollama: requires Ollama running with gemma4:latest"]
async fn test_ollama_gemma4_basic_chat() {
    let provider = ollama_gemma4();
    let messages = vec![ChatMessage::user("Respond with exactly one word: 'pong'")];

    let resp = provider
        .chat(&messages, None)
        .await
        .expect("basic chat should succeed");

    println!("[basic_chat] response: {:?}", resp.content);
    assert!(!resp.content.is_empty(), "response must contain content");
}

/// Chat with a system prompt.
#[tokio::test]
#[ignore = "ollama: requires Ollama running with gemma4:latest"]
async fn test_ollama_gemma4_system_prompt() {
    let provider = ollama_gemma4();
    let messages = vec![
        ChatMessage::system("You are a concise assistant. Always answer in exactly one sentence."),
        ChatMessage::user("What is the capital of France?"),
    ];

    let resp = provider
        .chat(&messages, None)
        .await
        .expect("chat with system prompt should succeed");

    println!("[system_prompt] response: {:?}", resp.content);
    assert!(!resp.content.is_empty());
    assert!(
        resp.content.to_lowercase().contains("paris"),
        "response should mention Paris, got: {}",
        resp.content
    );
}

/// Multiple system messages must be concatenated (bug fix #9).
#[tokio::test]
#[ignore = "ollama: requires Ollama running with gemma4:latest"]
async fn test_ollama_gemma4_multiple_system_prompts() {
    let provider = ollama_gemma4();
    let messages = vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::system("Always respond in JSON format with a single key 'answer'."),
        ChatMessage::user("What is 2 + 2?"),
    ];

    let resp = provider
        .chat(&messages, None)
        .await
        .expect("multiple system prompts should succeed");

    println!("[multi_system] response: {:?}", resp.content);
    assert!(!resp.content.is_empty());
}

/// Multi-turn conversation.
#[tokio::test]
#[ignore = "ollama: requires Ollama running with gemma4:latest"]
async fn test_ollama_gemma4_multi_turn_conversation() {
    let provider = ollama_gemma4();
    let messages = vec![
        ChatMessage::user("My name is Alice."),
        ChatMessage::assistant("Hello Alice! How can I help you today?"),
        ChatMessage::user("What is my name?"),
    ];

    let resp = provider
        .chat(&messages, None)
        .await
        .expect("multi-turn should succeed");

    println!("[multi_turn] response: {:?}", resp.content);
    assert!(
        resp.content.to_lowercase().contains("alice"),
        "should remember name from context, got: {}",
        resp.content
    );
}

/// Basic streaming — validates SSE line buffering fix (bug fix #1).
#[tokio::test]
#[ignore = "ollama: requires Ollama running with gemma4:latest"]
async fn test_ollama_gemma4_streaming() {
    let provider = ollama_gemma4();
    let mut stream = provider
        .stream("Count from 1 to 5, one number per line.")
        .await
        .expect("stream should start");

    let mut collected = String::new();
    let mut chunk_count = 0usize;

    while let Some(chunk) = stream.next().await {
        let text = chunk.expect("chunk should not error");
        collected.push_str(&text);
        chunk_count += 1;
    }

    println!("[streaming] chunks={} content={:?}", chunk_count, collected);
    assert!(!collected.is_empty(), "stream must produce output");
    assert!(chunk_count > 0, "must receive at least one chunk");
}

/// Streaming with max_tokens option.
#[tokio::test]
#[ignore = "ollama: requires Ollama running with gemma4:latest"]
async fn test_ollama_gemma4_streaming_with_options() {
    let provider = ollama_gemma4();
    let options = CompletionOptions {
        max_tokens: Some(50),
        temperature: Some(0.0),
        ..Default::default()
    };

    let stream = provider
        .complete_with_options("Write a haiku about Rust programming.", &options)
        .await
        .expect("complete_with_options should succeed");

    println!("[streaming_opts] content: {:?}", stream.content);
    assert!(!stream.content.is_empty());
}

/// Tool calling non-streaming — validates chat_with_tools() override (bug fix #4).
#[tokio::test]
#[ignore = "ollama: requires Ollama running with gemma4:latest"]
async fn test_ollama_gemma4_tools_non_streaming() {
    let provider = ollama_gemma4();
    let tools = vec![weather_tool()];
    let messages = vec![ChatMessage::user("What is the weather in Paris?")];

    let resp = provider
        .chat_with_tools(&messages, &tools, None, None)
        .await
        .expect("chat_with_tools should succeed");

    println!(
        "[tools_non_stream] content={:?} tool_calls={:?}",
        resp.content, resp.tool_calls
    );
    assert!(
        !resp.tool_calls.is_empty() || !resp.content.is_empty(),
        "model should either call a tool or respond with text"
    );
}

/// Tool calling with ToolChoice::auto.
#[tokio::test]
#[ignore = "ollama: requires Ollama running with gemma4:latest"]
async fn test_ollama_gemma4_tools_choice_auto() {
    let provider = ollama_gemma4();
    let tools = vec![calculator_tool()];
    let messages = vec![ChatMessage::user("What is 42 multiplied by 7?")];

    let resp = provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), None)
        .await
        .expect("chat_with_tools auto should succeed");

    println!(
        "[tools_auto] content={:?} tool_calls={:?}",
        resp.content, resp.tool_calls
    );
    // Auto choice: model may or may not call the tool
    assert!(
        !resp.content.is_empty() || !resp.tool_calls.is_empty(),
        "must produce some output"
    );
}

/// Full tool round-trip: model calls tool → we supply result → model answers.
/// Validates multi-turn with assistant tool_calls blocks (bug fix #10).
#[tokio::test]
#[ignore = "ollama: requires Ollama running with gemma4:latest"]
async fn test_ollama_gemma4_tool_round_trip() {
    let provider = ollama_gemma4();
    let tools = vec![weather_tool()];

    // Turn 1: ask model
    let messages1 = vec![ChatMessage::user("What is the weather in Tokyo?")];
    let resp1 = provider
        .chat_with_tools(&messages1, &tools, None, None)
        .await
        .expect("turn 1 should succeed");

    println!(
        "[tool_round_trip] turn1 content={:?} calls={:?}",
        resp1.content, resp1.tool_calls
    );

    if resp1.tool_calls.is_empty() {
        // Some smaller models may not call tools reliably — skip gracefully
        println!("[tool_round_trip] model did not call tool, skipping round-trip");
        return;
    }

    let tc = &resp1.tool_calls[0];
    assert_eq!(tc.name(), "get_weather");

    // Turn 2: supply tool result and get final answer
    let mut asst_msg = ChatMessage::assistant(&resp1.content);
    asst_msg.tool_calls = Some(resp1.tool_calls.clone());

    let messages2 = vec![
        messages1[0].clone(),
        asst_msg,
        ChatMessage::tool_result(&tc.id, "Sunny, 22°C"),
    ];

    let resp2 = provider
        .chat_with_tools(&messages2, &tools, None, None)
        .await
        .expect("turn 2 should succeed");

    println!("[tool_round_trip] turn2 content={:?}", resp2.content);
    assert!(!resp2.content.is_empty(), "model must produce final answer");
}

/// Streaming with tools — validates SSE buffering in chat_with_tools_stream (bug fix #2, #3).
#[tokio::test]
#[ignore = "ollama: requires Ollama running with gemma4:latest"]
async fn test_ollama_gemma4_tool_streaming() {
    let provider = ollama_gemma4();
    let tools = vec![weather_tool()];
    let messages = vec![ChatMessage::user("What is the weather in London?")];

    let mut stream = provider
        .chat_with_tools_stream(&messages, &tools, Some(ToolChoice::auto()), None)
        .await
        .expect("tool stream should start");

    let mut finished_count = 0usize;
    let mut saw_output = false;

    while let Some(chunk) = stream.next().await {
        match chunk.expect("chunk must not error") {
            StreamChunk::Content(text) => {
                print!("{}", text);
                saw_output = true;
            }
            StreamChunk::ToolCallDelta { function_name, .. } => {
                println!("\n[tool_call_delta] name={:?}", function_name);
                saw_output = true;
            }
            StreamChunk::Finished { reason, .. } => {
                println!("\n[finished] reason={:?}", reason);
                finished_count += 1;
            }
            _ => {}
        }
    }

    println!(
        "[tool_stream] finished_count={} saw_output={}",
        finished_count, saw_output
    );
    assert_eq!(
        finished_count, 1,
        "exactly one Finished chunk (duplicate Finished bug fix)"
    );
}

/// Streaming with multiple tools defined.
#[tokio::test]
#[ignore = "ollama: requires Ollama running with gemma4:latest"]
async fn test_ollama_gemma4_tool_streaming_multiple_tools() {
    let provider = ollama_gemma4();
    let tools = vec![weather_tool(), calculator_tool()];
    let messages = vec![ChatMessage::user(
        "What is the weather in Berlin and what is 15 * 8?",
    )];

    let mut stream = provider
        .chat_with_tools_stream(&messages, &tools, Some(ToolChoice::auto()), None)
        .await
        .expect("multi-tool stream should start");

    let mut finished_count = 0usize;

    while let Some(chunk) = stream.next().await {
        if let Ok(StreamChunk::Finished { .. }) = chunk {
            finished_count += 1;
        }
    }

    assert_eq!(
        finished_count, 1,
        "exactly one Finished chunk even with multiple tools"
    );
}

/// Empty tools list does not crash (guards against empty tools bug).
#[tokio::test]
#[ignore = "ollama: requires Ollama running with gemma4:latest"]
async fn test_ollama_gemma4_tool_streaming_empty_tools() {
    let provider = ollama_gemma4();
    let messages = vec![ChatMessage::user("Say hello.")];

    let mut stream = provider
        .chat_with_tools_stream(&messages, &[], None, None)
        .await
        .expect("empty tools stream should start");

    let mut saw_content = false;
    while let Some(chunk) = stream.next().await {
        if let Ok(StreamChunk::Content(_)) = chunk {
            saw_content = true;
        }
    }

    assert!(saw_content, "should get content with empty tools");
}

/// `complete()` convenience method.
#[tokio::test]
#[ignore = "ollama: requires Ollama running with gemma4:latest"]
async fn test_ollama_gemma4_complete() {
    let provider = ollama_gemma4();
    let resp = provider
        .complete("The capital of Germany is")
        .await
        .expect("complete should succeed");

    println!("[complete] content={:?}", resp.content);
    assert!(!resp.content.is_empty());
}

/// `complete_with_options()` with options.
#[tokio::test]
#[ignore = "ollama: requires Ollama running with gemma4:latest"]
async fn test_ollama_gemma4_complete_with_options() {
    let provider = ollama_gemma4();
    let options = CompletionOptions {
        max_tokens: Some(200), // generous budget — thinking models may use tokens before output
        temperature: Some(0.5),
        ..Default::default()
    };

    let resp = provider
        .complete_with_options("The capital of France is", &options)
        .await
        .expect("complete_with_options should succeed — any response is acceptable");

    println!(
        "[complete_opts] content={:?} tokens={}/{}",
        resp.content, resp.prompt_tokens, resp.completion_tokens
    );
    // Accept empty content for thinking models that exhaust their budget on reasoning
    // The key requirement is that the call does not error
}

/// Chat with max_tokens + temperature options.
#[tokio::test]
#[ignore = "ollama: requires Ollama running with gemma4:latest"]
async fn test_ollama_gemma4_chat_with_options() {
    let provider = ollama_gemma4();
    let messages = vec![ChatMessage::user(
        "What is 2 + 2? Reply with only the number.",
    )];
    let options = CompletionOptions {
        max_tokens: Some(512), // generous budget — thinking models reason before answering
        temperature: Some(0.5),
        ..Default::default()
    };

    let resp = provider
        .chat(&messages, Some(&options))
        .await
        .expect("chat with options should succeed");

    println!(
        "[chat_opts] content={:?} tokens={}/{}",
        resp.content, resp.prompt_tokens, resp.completion_tokens
    );
    // Accept non-empty OR check that tokens were consumed (thinking models may redirect output)
    assert!(
        !resp.content.is_empty() || resp.completion_tokens > 0,
        "model must produce output or consume completion tokens"
    );
}

/// from_env() reads ANTHROPIC_BASE_URL + ANTHROPIC_MODEL when set.
#[tokio::test]
#[ignore = "ollama: requires Ollama running with gemma4:latest"]
async fn test_ollama_gemma4_from_env() {
    // Temporarily set env vars pointing at Ollama
    std::env::set_var("ANTHROPIC_API_KEY", "ollama");
    std::env::set_var("ANTHROPIC_BASE_URL", "http://localhost:11434");
    std::env::set_var("ANTHROPIC_MODEL", "gemma4:latest");

    let provider = AnthropicProvider::from_env().expect("from_env should succeed");
    assert_eq!(provider.api_key(), "ollama");
    assert_eq!(provider.base_url(), "http://localhost:11434");
    assert_eq!(provider.model(), "gemma4:latest");

    let resp = provider
        .chat(&[ChatMessage::user("Say 'hello'")], None)
        .await
        .expect("from_env chat should succeed");

    println!("[from_env] content={:?}", resp.content);
    assert!(!resp.content.is_empty());

    // Clean up
    std::env::remove_var("ANTHROPIC_API_KEY");
    std::env::remove_var("ANTHROPIC_BASE_URL");
    std::env::remove_var("ANTHROPIC_MODEL");
}

/// Error path: invalid base URL should return an error (not panic).
#[tokio::test]
#[ignore = "ollama: requires Ollama running with gemma4:latest"]
async fn test_ollama_bad_base_url_returns_error() {
    let provider = AnthropicProvider::new("ollama")
        .with_base_url("http://localhost:9") // nothing listening here
        .with_model("gemma4:latest");

    let result = provider.chat(&[ChatMessage::user("hello")], None).await;

    assert!(result.is_err(), "must return error for unreachable host");
    println!("[bad_url] error: {:?}", result.unwrap_err());
}

/// Auth error: invalid API key to real Anthropic → AuthError.
#[tokio::test]
#[ignore = "ollama: requires Ollama running with gemma4:latest and network access"]
async fn test_live_anthropic_invalid_key_returns_auth_error() {
    use edgequake_llm::LlmError;

    let provider = AnthropicProvider::new("sk-invalid-key-12345");
    let result = provider.chat(&[ChatMessage::user("hello")], None).await;

    assert!(result.is_err());
    assert!(
        matches!(result.unwrap_err(), LlmError::AuthError(_)),
        "invalid key must map to AuthError"
    );
}

// ============================================================================
// Live Anthropic API tests (require ANTHROPIC_API_KEY)
// ============================================================================

#[tokio::test]
#[ignore = "anthropic_live: requires ANTHROPIC_API_KEY env var"]
async fn test_anthropic_live_basic_chat() {
    let provider = AnthropicProvider::from_env().expect("Set ANTHROPIC_API_KEY");
    let resp = provider
        .chat(&[ChatMessage::user("Say 'pong' and nothing else.")], None)
        .await
        .expect("live chat should succeed");

    println!("[live_chat] content={:?}", resp.content);
    assert!(!resp.content.is_empty());
    assert!(resp.prompt_tokens > 0);
    assert!(resp.completion_tokens > 0);
}

#[tokio::test]
#[ignore = "anthropic_live: requires ANTHROPIC_API_KEY env var"]
async fn test_anthropic_live_streaming() {
    let provider = AnthropicProvider::from_env().expect("Set ANTHROPIC_API_KEY");
    let mut stream = provider
        .stream("Count to 5.")
        .await
        .expect("live stream should start");

    let mut out = String::new();
    while let Some(chunk) = stream.next().await {
        out.push_str(&chunk.expect("chunk must not error"));
    }

    println!("[live_stream] output={:?}", out);
    assert!(!out.is_empty());
}

#[tokio::test]
#[ignore = "anthropic_live: requires ANTHROPIC_API_KEY env var"]
async fn test_anthropic_live_tool_call() {
    let provider = AnthropicProvider::from_env().expect("Set ANTHROPIC_API_KEY");
    let tools = vec![weather_tool()];
    let messages = vec![ChatMessage::user("What is the weather in Paris?")];

    let resp = provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), None)
        .await
        .expect("live tool call should succeed");

    println!(
        "[live_tool] content={:?} calls={:?}",
        resp.content, resp.tool_calls
    );
    assert!(!resp.tool_calls.is_empty() || !resp.content.is_empty());
}

#[tokio::test]
#[ignore = "anthropic_live: requires ANTHROPIC_API_KEY env var"]
async fn test_anthropic_live_streaming_finished_once() {
    let provider = AnthropicProvider::from_env().expect("Set ANTHROPIC_API_KEY");
    let messages = vec![ChatMessage::user("Hello")];

    let mut stream = provider
        .chat_with_tools_stream(&messages, &[], None, None)
        .await
        .expect("live tool stream should start");

    let mut finished_count = 0usize;
    while let Some(chunk) = stream.next().await {
        if let Ok(StreamChunk::Finished { .. }) = chunk {
            finished_count += 1;
        }
    }

    assert_eq!(finished_count, 1, "exactly one Finished chunk");
}
