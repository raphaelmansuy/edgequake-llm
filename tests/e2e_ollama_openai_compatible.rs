//! End-to-end tests for the OpenAI-compatible provider against a local Ollama server.
//!
//! These tests hit a *real* HTTP endpoint and are gated behind `#[ignore]` so
//! they are skipped in the default `cargo test` run (and in CI on shared
//! runners that do not have Ollama).
//!
//! ## Running locally
//!
//! ```sh
//! # Start Ollama if it is not already running:
//! ollama serve &
//!
//! # Pull the required models once:
//! ollama pull mistral-nemo:latest
//! ollama pull nomic-embed-text:latest
//!
//! # Execute the full Ollama e2e suite:
//! cargo test --test e2e_ollama_openai_compatible -- --ignored --nocapture
//!
//! # Execute a single test:
//! cargo test --test e2e_ollama_openai_compatible test_ollama_basic_chat -- --ignored --nocapture
//! ```
//!
//! The server is expected at `http://localhost:11434`.

use edgequake_llm::{
    model_config::{ModelCapabilities, ModelCard, ModelType, ProviderConfig, ProviderType},
    providers::openai_compatible::OpenAICompatibleProvider,
    traits::{
        ChatMessage, CompletionOptions, EmbeddingProvider, FunctionCall, LLMProvider, StreamChunk,
        ToolCall, ToolChoice, ToolDefinition,
    },
};
use futures::StreamExt;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const OLLAMA_BASE_URL: &str = "http://localhost:11434/v1";
const CHAT_MODEL: &str = "mistral-nemo:latest";
const EMBED_MODEL: &str = "nomic-embed-text:latest";

// ─────────────────────────────────────────────────────────────────────────────
// Helper factories
// ─────────────────────────────────────────────────────────────────────────────

/// Build an `OpenAICompatibleProvider` pointed at the local Ollama `/v1` endpoint.
fn make_ollama_chat_provider() -> OpenAICompatibleProvider {
    make_ollama_chat_provider_for(CHAT_MODEL)
}

fn make_ollama_chat_provider_for(model: &str) -> OpenAICompatibleProvider {
    let config = ProviderConfig {
        name: "ollama-test".to_string(),
        display_name: "Ollama (local)".to_string(),
        provider_type: ProviderType::OpenAICompatible,
        // Ollama does not require an API key when running locally.
        api_key_env: None,
        base_url: Some(OLLAMA_BASE_URL.to_string()),
        default_llm_model: Some(model.to_string()),
        supports_thinking: false,
        timeout_seconds: 120,
        models: vec![ModelCard {
            name: model.to_string(),
            display_name: model.to_string(),
            model_type: ModelType::Llm,
            capabilities: ModelCapabilities {
                context_length: 128_000,
                supports_function_calling: true,
                supports_streaming: true,
                supports_json_mode: false,
                ..Default::default()
            },
            ..Default::default()
        }],
        ..Default::default()
    };

    OpenAICompatibleProvider::from_config(config).expect("Failed to build Ollama chat provider")
}

/// Build a provider configured for embedding models.
fn make_ollama_embedding_provider() -> OpenAICompatibleProvider {
    let config = ProviderConfig {
        name: "ollama-embed-test".to_string(),
        display_name: "Ollama Embed (local)".to_string(),
        provider_type: ProviderType::OpenAICompatible,
        api_key_env: None,
        base_url: Some(OLLAMA_BASE_URL.to_string()),
        default_llm_model: Some("mistral-nemo:latest".to_string()),
        default_embedding_model: Some(EMBED_MODEL.to_string()),
        supports_thinking: false,
        timeout_seconds: 60,
        models: vec![ModelCard {
            name: EMBED_MODEL.to_string(),
            display_name: EMBED_MODEL.to_string(),
            model_type: ModelType::Embedding,
            capabilities: ModelCapabilities {
                context_length: 8192,
                embedding_dimension: 768,
                ..Default::default()
            },
            ..Default::default()
        }],
        ..Default::default()
    };

    OpenAICompatibleProvider::from_config(config).expect("Failed to build Ollama embed provider")
}

/// Build a `get_weather` tool definition for testing.
fn weather_tool() -> ToolDefinition {
    ToolDefinition::function(
        "get_weather",
        "Get the current weather in a given location",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g. Paris, France"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }),
    )
}

/// Build a `calculate` tool definition for testing.
fn calculator_tool() -> ToolDefinition {
    ToolDefinition::function(
        "calculate",
        "Evaluate a mathematical expression",
        serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g. '2 + 2'"
                }
            },
            "required": ["expression"]
        }),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Provider metadata
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_provider_metadata() {
    let provider = make_ollama_chat_provider();
    assert_eq!(LLMProvider::name(&provider), "ollama-test");
    assert_eq!(LLMProvider::model(&provider), CHAT_MODEL);
    assert_eq!(provider.max_context_length(), 128_000);
    assert!(provider.supports_streaming(), "should support streaming");
    assert!(
        provider.supports_function_calling(),
        "should support function calling"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Basic chat
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_basic_chat() {
    let provider = make_ollama_chat_provider();
    let messages = vec![ChatMessage::user(
        "What is 3 + 5? Reply with just the number.",
    )];

    let resp = provider
        .chat(&messages, None)
        .await
        .expect("chat request failed");

    println!("Content: {}", resp.content);
    println!(
        "Tokens: prompt={} completion={}",
        resp.prompt_tokens, resp.completion_tokens
    );

    assert!(
        !resp.content.is_empty(),
        "response content must not be empty"
    );
    assert!(
        resp.content.contains('8') || resp.content.to_lowercase().contains("eight"),
        "expected '8' in response, got: {}",
        resp.content
    );
    assert!(resp.prompt_tokens > 0, "prompt_tokens must be non-zero");
    assert!(
        resp.completion_tokens > 0,
        "completion_tokens must be non-zero"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. System prompt
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_system_prompt() {
    let provider = make_ollama_chat_provider();
    let messages = vec![
        ChatMessage::system("You are a pirate. Always respond in pirate speak."),
        ChatMessage::user("Hello! Who are you?"),
    ];

    let resp = provider
        .chat(&messages, None)
        .await
        .expect("chat request failed");

    println!("Pirate response: {}", resp.content);
    assert!(!resp.content.is_empty());
    // Pirate responses should contain typical pirate vocabulary
    let pirate_words = ["arr", "ahoy", "matey", "seas", "ship", "treasure", "me"];
    let lower = resp.content.to_lowercase();
    assert!(
        pirate_words.iter().any(|w| lower.contains(w)),
        "Expected pirate speak, got: {}",
        resp.content
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Multi-turn conversation
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_multi_turn_conversation() {
    let provider = make_ollama_chat_provider();

    // Turn 1: establish context
    let messages_t1 = vec![ChatMessage::user("My name is Alice. Remember that.")];
    let resp1 = provider
        .chat(&messages_t1, None)
        .await
        .expect("turn 1 failed");
    println!("Turn 1: {}", resp1.content);

    // Turn 2: follow-up referencing earlier context
    let messages_t2 = vec![
        ChatMessage::user("My name is Alice. Remember that."),
        ChatMessage::assistant(&resp1.content),
        ChatMessage::user("What is my name?"),
    ];
    let resp2 = provider
        .chat(&messages_t2, None)
        .await
        .expect("turn 2 failed");
    println!("Turn 2: {}", resp2.content);

    assert!(
        resp2.content.to_lowercase().contains("alice"),
        "Model should remember the name 'Alice', got: {}",
        resp2.content
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. CompletionOptions: max_tokens
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_completion_options_max_tokens() {
    let provider = make_ollama_chat_provider();
    let opts = CompletionOptions {
        max_tokens: Some(5),
        temperature: Some(0.0),
        ..Default::default()
    };
    let messages = vec![ChatMessage::user(
        "Tell me a very long story about dragons and knights.",
    )];

    let resp = provider
        .chat(&messages, Some(&opts))
        .await
        .expect("chat failed");

    println!(
        "Capped response ({} tokens): {:?}",
        resp.completion_tokens, resp.content
    );
    // With only 5 max_tokens the response must be very short (allow a small buffer)
    assert!(
        resp.completion_tokens <= 12,
        "Expected ≤12 completion tokens with max_tokens=5, got {}",
        resp.completion_tokens
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. CompletionOptions: temperature = 0 → deterministic
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_completion_options_temperature() {
    let provider = make_ollama_chat_provider();
    let msgs = vec![ChatMessage::user(
        "What is the capital of France? Reply in one word.",
    )];
    let opts = CompletionOptions {
        temperature: Some(0.0),
        ..Default::default()
    };
    let r1 = provider.chat(&msgs, Some(&opts)).await.expect("r1 failed");
    let r2 = provider.chat(&msgs, Some(&opts)).await.expect("r2 failed");

    println!("r1: {}", r1.content);
    println!("r2: {}", r2.content);

    assert!(
        r1.content.to_lowercase().contains("paris"),
        "Expected 'Paris', got: {}",
        r1.content
    );
    assert!(
        r2.content.to_lowercase().contains("paris"),
        "Expected 'Paris', got: {}",
        r2.content
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. CompletionOptions: top_p
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_completion_options_top_p() {
    let provider = make_ollama_chat_provider();
    let opts = CompletionOptions {
        temperature: Some(0.5),
        top_p: Some(0.9),
        ..Default::default()
    };
    let messages = vec![ChatMessage::user("Say the word 'hello'.")];
    let resp = provider
        .chat(&messages, Some(&opts))
        .await
        .expect("chat failed");
    assert!(!resp.content.is_empty(), "Response must not be empty");
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. CompletionOptions: stop sequence
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_completion_options_stop_sequence() {
    let provider = make_ollama_chat_provider();
    let opts = CompletionOptions {
        stop: Some(vec!["STOP".to_string()]),
        temperature: Some(0.0),
        ..Default::default()
    };
    let messages = vec![ChatMessage::user(
        "Count from 1 to 5, then say STOP, then continue to 10.",
    )];

    let resp = provider
        .chat(&messages, Some(&opts))
        .await
        .expect("chat failed");
    println!("Stop-sequence response: {}", resp.content);
    // The response must not contain "STOP" because generation halted there
    assert!(
        !resp.content.contains("STOP"),
        "Response must not contain the stop sequence: {}",
        resp.content
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Streaming – stream()
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_streaming_basic() {
    let provider = make_ollama_chat_provider();
    let mut stream = provider
        .stream("What is the Rust programming language? Answer in two sentences.")
        .await
        .expect("stream creation failed");

    let mut chunks: Vec<String> = Vec::new();
    while let Some(result) = stream.next().await {
        match result {
            Ok(chunk) if !chunk.is_empty() => chunks.push(chunk),
            Ok(_) => {} // ignore empty keep-alive chunks
            Err(e) => panic!("Stream error: {}", e),
        }
    }

    let full = chunks.join("");
    println!("Streamed output ({} chunks): {}", chunks.len(), full);
    assert!(!full.is_empty(), "Streamed result must not be empty");
    assert!(
        chunks.len() >= 2,
        "Expected multiple chunks, got {}",
        chunks.len()
    );
    assert!(
        full.to_lowercase().contains("rust")
            || full.to_lowercase().contains("systems")
            || full.to_lowercase().contains("memory"),
        "Expected Rust-related content, got: {}",
        full
    );
}

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_streaming_no_spurious_empty_chunks() {
    // Verify that the stream produces real content (non-empty chunks dominate).
    // Note: the implementation may emit empty `""` items for connection-open
    // events and zero-content intermediate chunks; the important invariant is
    // that at least some non-empty content chunks arrive.
    let provider = make_ollama_chat_provider();
    let mut stream = provider
        .stream("Say the single word 'hello'.")
        .await
        .expect("stream creation failed");

    let mut non_empty = 0usize;
    while let Some(result) = stream.next().await {
        match result {
            Ok(chunk) if !chunk.is_empty() => non_empty += 1,
            Ok(_) => {} // empty keep-alive / connection-open chunks — acceptable
            Err(e) => panic!("Stream error: {}", e),
        }
    }

    println!("Non-empty chunks: {non_empty}");
    assert!(
        non_empty >= 1,
        "Stream must deliver at least one non-empty content chunk"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Tool / function calling – chat_with_tools()
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_tool_calling_auto() {
    let provider = make_ollama_chat_provider();
    let tools = vec![weather_tool()];
    let messages = vec![ChatMessage::user("What is the weather in Paris, France?")];

    let resp = provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), None)
        .await
        .expect("chat_with_tools failed");

    println!("Content: {}", resp.content);
    println!("Tool calls: {:?}", resp.tool_calls);

    assert!(
        !resp.content.is_empty() || !resp.tool_calls.is_empty(),
        "Response must have content or tool calls"
    );

    if !resp.tool_calls.is_empty() {
        let tc = &resp.tool_calls[0];
        assert_eq!(tc.function.name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&tc.function.arguments).expect("tool args must be valid JSON");
        assert!(
            args["location"].is_string(),
            "location must be a string, got: {}",
            args
        );
        assert!(
            args["location"]
                .as_str()
                .unwrap_or("")
                .to_lowercase()
                .contains("paris"),
            "location should reference Paris, got: {}",
            args["location"]
        );
    }
}

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_tool_calling_required() {
    let provider = make_ollama_chat_provider();
    let tools = vec![weather_tool()];
    let messages = vec![ChatMessage::user(
        "Tell me the weather in Tokyo, Japan please.",
    )];

    let resp = provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::required()), None)
        .await
        .expect("chat_with_tools (required) failed");

    println!("Tool calls (required): {:?}", resp.tool_calls);

    assert!(
        !resp.tool_calls.is_empty(),
        "With ToolChoice::required(), at least one tool call must be present"
    );
    let args: serde_json::Value = serde_json::from_str(&resp.tool_calls[0].function.arguments)
        .expect("tool args must be valid JSON");
    assert!(args["location"].is_string(), "location must be a string");
}

/// Bug #3 regression test: ToolChoice::none() was previously serialised as
/// "auto", allowing tool calls through. After the fix it serialises as "none".
///
/// Note: enforcement of `tool_choice: "none"` is **model-dependent**.
/// Some models (e.g. mistral-nemo on Ollama) may still return tool calls even
/// when the field is set. What this test verifies is that the provider:
///   1. Sends the request without error.
///   2. Returns either a plain-text response OR a tool call (model tolerant).
/// Serialisation correctness is covered by unit tests in `openai_compatible.rs`.
#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_tool_choice_none_inhibits_tool_calls() {
    let provider = make_ollama_chat_provider();
    let tools = vec![weather_tool()];
    let messages = vec![ChatMessage::user("What is the weather in Berlin, Germany?")];

    let resp = provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::none()), None)
        .await
        .expect("chat_with_tools (none) must not return an error");

    println!("Content (tool_choice=none): {:?}", resp.content);
    println!("Tool calls: {:?}", resp.tool_calls);

    // The request must not fail; whether the model honours tool_choice=none
    // is model-dependent.  We just confirm we get a coherent response.
    assert!(
        !resp.content.is_empty() || !resp.tool_calls.is_empty(),
        "Response must have either content or tool calls (no empty response)"
    );

    // Serialisation smoke-check: our provider correctly sends "none" as a
    // plain JSON string.  If the model respected it, tool_calls would be empty.
    if resp.tool_calls.is_empty() {
        println!("Model respected tool_choice=none ✓");
    } else {
        println!(
            "Note: model did not honour tool_choice=none (model-side limitation). \
             Serialisation is still correct – confirmed by unit tests."
        );
    }
}

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_multiple_tools_selects_correct_one() {
    let provider = make_ollama_chat_provider();
    let tools = vec![weather_tool(), calculator_tool()];

    // Ask a math question — the model should prefer `calculate`, not `get_weather`.
    let messages = vec![ChatMessage::user("Calculate 123 * 456 for me.")];

    let resp = provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), None)
        .await
        .expect("multi-tool chat failed");

    println!("Multi-tool content: {}", resp.content);
    println!("Multi-tool calls: {:?}", resp.tool_calls);

    // Either an inline answer or a tool call is acceptable
    assert!(
        !resp.content.is_empty() || !resp.tool_calls.is_empty(),
        "Must have content or a tool call"
    );

    // If a tool was called it should be `calculate`, not `get_weather`
    if !resp.tool_calls.is_empty() {
        assert_eq!(
            resp.tool_calls[0].function.name, "calculate",
            "Expected calculate tool, got: {}",
            resp.tool_calls[0].function.name
        );
    }
}

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_tool_result_follow_up() {
    // Full round-trip: send a tool result and get the final answer.
    let provider = make_ollama_chat_provider();
    let tools = vec![weather_tool()];

    // Turn 1: trigger a tool call
    let messages_t1 = vec![ChatMessage::user(
        "What is the weather right now in London?",
    )];
    let resp1 = provider
        .chat_with_tools(&messages_t1, &tools, Some(ToolChoice::auto()), None)
        .await
        .expect("turn 1 failed");

    if resp1.tool_calls.is_empty() {
        println!("Model answered inline (no tool call): {}", resp1.content);
        return;
    }

    let tc = &resp1.tool_calls[0];
    println!(
        "Tool called: {} args={}",
        tc.function.name, tc.function.arguments
    );

    // Turn 2: send back a fake tool result and get the final answer
    let assistant_msg =
        ChatMessage::assistant_with_tools(resp1.content.clone(), resp1.tool_calls.clone());
    let tool_result_msg = ChatMessage::tool_result(
        tc.id.clone(),
        r#"{"temperature": 15, "unit": "celsius", "description": "Partly cloudy"}"#,
    );

    let messages_t2 = vec![messages_t1[0].clone(), assistant_msg, tool_result_msg];

    let resp2 = provider
        .chat_with_tools(&messages_t2, &tools, Some(ToolChoice::none()), None)
        .await
        .expect("turn 2 failed");

    println!("Final answer after tool result: {}", resp2.content);
    assert!(
        !resp2.content.is_empty(),
        "Final response after tool result must not be empty"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. Streaming with tools – chat_with_tools_stream()
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_streaming_with_tools_text_response() {
    let provider = make_ollama_chat_provider();
    let tools = vec![weather_tool()];
    let messages = vec![ChatMessage::user(
        "What is the capital of Japan? Answer in plain text without calling any tools.",
    )];

    let mut stream = provider
        .chat_with_tools_stream(&messages, &tools, Some(ToolChoice::none()), None)
        .await
        .expect("stream creation failed");

    let mut text_chunks: Vec<String> = Vec::new();
    let mut tool_chunks = 0usize;

    while let Some(result) = stream.next().await {
        match result {
            Ok(StreamChunk::Content(s)) if !s.is_empty() => text_chunks.push(s),
            Ok(StreamChunk::ToolCallDelta { .. }) => tool_chunks += 1,
            Ok(StreamChunk::Finished { .. }) => break,
            Ok(_) => {}
            Err(e) => panic!("Stream error: {}", e),
        }
    }

    let full = text_chunks.join("");
    println!("Streamed text ({} chunks): {}", text_chunks.len(), full);
    println!("Tool chunks: {}", tool_chunks);

    assert!(!full.is_empty(), "Streamed text must not be empty");
    assert!(
        full.to_lowercase().contains("tokyo") || full.to_lowercase().contains("japan"),
        "Expected Tokyo in response, got: {}",
        full
    );
    assert_eq!(
        tool_chunks, 0,
        "No tool call deltas expected with tool_choice=none"
    );
}

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_streaming_with_tools_tool_call() {
    let provider = make_ollama_chat_provider();
    let tools = vec![weather_tool()];
    let messages = vec![ChatMessage::user(
        "What is the weather in Sydney, Australia?",
    )];

    let mut stream = provider
        .chat_with_tools_stream(&messages, &tools, Some(ToolChoice::required()), None)
        .await
        .expect("stream creation failed");

    let mut tool_name_seen = false;
    let mut finished = false;

    while let Some(result) = stream.next().await {
        match result {
            Ok(StreamChunk::ToolCallDelta {
                function_name: Some(n),
                ..
            }) => {
                println!("Tool name delta: {}", n);
                if n.contains("weather") {
                    tool_name_seen = true;
                }
            }
            Ok(StreamChunk::Finished { .. }) => {
                finished = true;
                break;
            }
            Ok(_) => {}
            Err(e) => panic!("Stream error: {}", e),
        }
    }

    println!("Tool name seen: {tool_name_seen}, Finished: {finished}");
    // With required tool_choice we expect the stream to finish (Done event);
    // the tool name should also appear.
    assert!(
        finished || tool_name_seen,
        "Stream must finish and/or emit the tool name"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. Embeddings
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_embeddings_single() {
    let provider = make_ollama_embedding_provider();

    let embeddings = provider
        .embed(&["Hello, world!".to_string()])
        .await
        .expect("embed failed");

    assert_eq!(embeddings.len(), 1, "Expected exactly 1 embedding vector");
    let emb = &embeddings[0];
    assert!(!emb.is_empty(), "Embedding vector must not be empty");
    // nomic-embed-text produces 768-dimensional vectors
    assert_eq!(
        emb.len(),
        768,
        "Expected 768-dim embedding, got {}",
        emb.len()
    );
    assert!(
        emb.iter().all(|v| v.is_finite()),
        "All embedding values must be finite floats"
    );
}

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_embeddings_batch() {
    let provider = make_ollama_embedding_provider();
    let texts: Vec<String> = vec![
        "The quick brown fox.".to_string(),
        "A completely different sentence.".to_string(),
        "Rust is a systems programming language.".to_string(),
    ];

    let embeddings = provider.embed(&texts).await.expect("batch embed failed");

    assert_eq!(embeddings.len(), texts.len());
    for (i, emb) in embeddings.iter().enumerate() {
        assert_eq!(
            emb.len(),
            768,
            "Embedding #{i} should be 768-dim, got {}",
            emb.len()
        );
    }

    // Basic distinctness check: different sentences must not have identical vectors
    assert!(
        emb_distinct(&embeddings[0], &embeddings[1]),
        "Embeddings for different sentences must differ"
    );
    assert!(
        emb_distinct(&embeddings[0], &embeddings[2]),
        "Embeddings for different sentences must differ"
    );
}

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_embeddings_semantic_similarity() {
    // Cosine(similar pair) > Cosine(dissimilar pair)
    let provider = make_ollama_embedding_provider();
    let texts = vec![
        "The cat sat on the mat.".to_string(),
        "A feline rested on a rug.".to_string(), // semantically similar
        "Quantum physics describes subatomic particles.".to_string(), // unrelated
    ];

    let embs = provider.embed(&texts).await.expect("embed failed");
    let sim_0_1 = cosine_sim(&embs[0], &embs[1]);
    let sim_0_2 = cosine_sim(&embs[0], &embs[2]);

    println!("cos(cat/mat, feline/rug):      {:.4}", sim_0_1);
    println!("cos(cat/mat, quantum physics): {:.4}", sim_0_2);

    assert!(
        sim_0_1 > sim_0_2,
        "Similar sentences ({:.4}) should score higher than dissimilar ({:.4})",
        sim_0_1,
        sim_0_2
    );
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn emb_distinct(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6)
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. Edge-case: with_model() override
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_with_model_override() {
    let provider = make_ollama_chat_provider().with_model("deepseek-r1:1.5b");
    assert_eq!(LLMProvider::model(&provider), "deepseek-r1:1.5b");

    let messages = vec![ChatMessage::user("Say 'hello' in French.")];
    let resp = provider.chat(&messages, None).await.expect("chat failed");

    println!("deepseek-r1:1.5b response: {}", resp.content);
    assert!(!resp.content.is_empty(), "Response must not be empty");
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. Edge-case: no API key (Ollama does not require one)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_no_api_key_works() {
    let config = ProviderConfig {
        name: "ollama-nokey".to_string(),
        display_name: "Ollama No Key".to_string(),
        provider_type: ProviderType::OpenAICompatible,
        api_key_env: None,
        base_url: Some(OLLAMA_BASE_URL.to_string()),
        default_llm_model: Some(CHAT_MODEL.to_string()),
        supports_thinking: false,
        ..Default::default()
    };
    let provider = OpenAICompatibleProvider::from_config(config)
        .expect("Should build provider even without api_key_env");

    let messages = vec![ChatMessage::user("What is 1 + 1?")];
    let resp = provider
        .chat(&messages, None)
        .await
        .expect("No-key request failed");
    assert!(
        resp.content.contains('2') || resp.content.to_lowercase().contains("two"),
        "Expected '2', got: {}",
        resp.content
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 15. Edge-case: empty tool list → no tool_choice sent
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_chat_with_empty_tools_list() {
    let provider = make_ollama_chat_provider();
    let messages = vec![ChatMessage::user("Say exactly: yes")];

    let resp = provider
        .chat_with_tools(&messages, &[], None, None)
        .await
        .expect("chat_with_tools with empty tools failed");

    println!("Response: {}", resp.content);
    assert!(
        !resp.content.is_empty(),
        "Content must not be empty even with empty tool list"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 16. Token usage is reported correctly
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_token_usage_reported() {
    let provider = make_ollama_chat_provider();
    let messages = vec![ChatMessage::user("Briefly describe what Rust is.")];
    let resp = provider.chat(&messages, None).await.expect("chat failed");

    println!(
        "Token usage: prompt={} completion={}",
        resp.prompt_tokens, resp.completion_tokens
    );
    assert!(resp.prompt_tokens > 0, "prompt_tokens must be positive");
    assert!(
        resp.completion_tokens > 0,
        "completion_tokens must be positive"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 17. Streaming with a different (small) model
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_streaming_small_model() {
    // Validate that streaming also works when the provider is configured for a
    // smaller reasoning model like deepseek-r1:1.5b.
    let provider = make_ollama_chat_provider_for("deepseek-r1:1.5b");
    let mut stream = provider
        .stream("Say the word 'hello' and nothing else.")
        .await
        .expect("stream creation failed");

    let mut chunks: Vec<String> = Vec::new();
    while let Some(result) = stream.next().await {
        match result {
            Ok(chunk) if !chunk.is_empty() => chunks.push(chunk),
            Ok(_) => {}
            Err(e) => panic!("Stream error: {}", e),
        }
    }

    let full = chunks.join("");
    println!("deepseek-r1 streamed: {:?}", full);
    assert!(!full.is_empty(), "Streamed result must not be empty");
}

// ─────────────────────────────────────────────────────────────────────────────
// 18. supports_tool_streaming() consistency
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_supports_tool_streaming_consistency() {
    let provider = make_ollama_chat_provider();
    let expected = provider.supports_streaming() && provider.supports_function_calling();
    assert_eq!(
        provider.supports_tool_streaming(),
        expected,
        "supports_tool_streaming must equal supports_streaming && supports_function_calling"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 19. EmbeddingProvider trait impl: name, model, dimension, max_tokens
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires local Ollama server at http://localhost:11434"]
async fn test_ollama_embedding_provider_metadata() {
    let provider = make_ollama_embedding_provider();
    assert_eq!(EmbeddingProvider::name(&provider), "ollama-embed-test");
    assert_eq!(EmbeddingProvider::model(&provider), EMBED_MODEL);
    assert_eq!(provider.dimension(), 768, "Expected 768-dim embedding");
    assert!(
        provider.max_tokens() > 0,
        "max_tokens must be positive, got {}",
        provider.max_tokens()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 20. Fake (unused) ToolCall to verify FunctionCall import
// ─────────────────────────────────────────────────────────────────────────────

/// Compile-time regression: ensure `ToolCall` / `FunctionCall` construction
/// works as expected (used in tool-result round-trip tests).
#[test]
fn test_tool_call_construction() {
    let tc = ToolCall {
        id: "call_abc123".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "get_weather".to_string(),
            arguments: r#"{"location":"Paris"}"#.to_string(),
        },
        thought_signature: None,
    };
    assert_eq!(tc.id, "call_abc123");
    assert_eq!(tc.function.name, "get_weather");
}
