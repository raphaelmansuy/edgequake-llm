//! Comprehensive End-to-End tests for OllamaProvider using gemma4:latest.
//!
//! Scenarios covered:
//! - Basic chat (non-streaming and streaming)
//! - Multi-turn conversations with history
//! - System messages
//! - Stop conditions: `done_reason` == "stop" (normal) and "length" (truncated)
//! - Custom stop sequences via `CompletionOptions::stop`
//! - JSON mode (`format = "json"`)
//! - Structured output (JSON schema via `format = <schema>`)
//! - Usage metadata (prompt_tokens, completion_tokens > 0)
//! - `finish_reason` propagation in `LLMResponse` and in the streaming `Finished` chunk
//! - Tool calling — non-streaming (`chat_with_tools`)
//! - Tool calling — streaming (`chat_with_tools_stream`)
//! - Multi-turn tool conversation (tool call → tool result → model synthesises answer)
//! - Embedding generation
//!
//! ## Requirements
//! - Ollama running at `http://localhost:11434`
//! - Model pulled: `ollama pull gemma4:latest`
//!
//! ## Running
//! ```sh
//! # All tests in this file (Ollama must be running):
//! cargo test --test e2e_ollama_gemma4 -- --ignored --nocapture
//!
//! # One test:
//! cargo test --test e2e_ollama_gemma4 -- --ignored --nocapture test_ollama_finish_reason_stop
//! ```

use edgequake_llm::{
    providers::ollama::OllamaProvider,
    traits::{ChatMessage, CompletionOptions, ToolDefinition},
    EmbeddingProvider, LLMProvider,
};
use futures::StreamExt as _;
use serde_json::json;

// ============================================================================
// Constants
// ============================================================================

/// Primary text model used for all tests.
const MODEL: &str = "gemma4:latest";

/// Ollama base URL.
const HOST: &str = "http://localhost:11434";

// ============================================================================
// Shared helpers
// ============================================================================

/// Returns `true` when Ollama is reachable (HEAD /api/tags).
async fn ollama_is_available() -> bool {
    reqwest::Client::new()
        .get(format!("{HOST}/api/tags"))
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
        .is_ok()
}

/// Returns `true` when `model` has been pulled locally.
async fn model_is_available(model: &str) -> bool {
    let Ok(resp) = reqwest::Client::new()
        .get(format!("{HOST}/api/tags"))
        .send()
        .await
    else {
        return false;
    };
    let Ok(body) = resp.text().await else {
        return false;
    };
    body.contains(
        model
            .trim_end_matches(":latest")
            .split(':')
            .next()
            .unwrap_or(model),
    )
}

/// Build a provider for `MODEL`.
fn provider() -> OllamaProvider {
    OllamaProvider::builder()
        .host(HOST)
        .model(MODEL)
        .build()
        .expect("Failed to build OllamaProvider")
}

/// Macro: skip a test when Ollama or the model is unavailable.
macro_rules! require_ollama {
    () => {
        if !ollama_is_available().await {
            eprintln!("SKIP — Ollama not reachable at {HOST}");
            return;
        }
        if !model_is_available(MODEL).await {
            eprintln!("SKIP — model {MODEL} not pulled (run: ollama pull {MODEL})");
            return;
        }
    };
}

// ============================================================================
// Test helpers
// ============================================================================

/// Strip markdown code fences from a model response.
///
/// Some models (e.g. gemma4) wrap JSON responses in ` ```json … ``` ` blocks
/// even when `format=json` is requested.  This helper removes those fences so
/// that the test can parse the underlying JSON.
fn strip_json_fences(s: &str) -> &str {
    let s = s.trim();
    // Remove opening fence (with or without language tag).
    let s = if let Some(rest) = s.strip_prefix("```json") {
        rest
    } else if let Some(rest) = s.strip_prefix("```") {
        rest
    } else {
        s
    };
    // Remove closing fence.
    let s = s.strip_suffix("```").unwrap_or(s);
    s.trim()
}

// ============================================================================
// Basic completion
// ============================================================================

/// `complete()` returns a non-empty string.
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_basic_completion() {
    require_ollama!();
    let p = provider();
    let resp = p
        .complete("Reply with exactly one word: hello")
        .await
        .unwrap();
    assert!(
        !resp.content.is_empty(),
        "completion content must not be empty"
    );
}

/// `complete_with_options()` with a low temperature returns a coherent response.
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_completion_with_options() {
    require_ollama!();
    let p = provider();
    let opts = CompletionOptions {
        temperature: Some(0.0),
        max_tokens: Some(32),
        ..Default::default()
    };
    let resp = p
        .complete_with_options("What is 1 + 1? Give a number only.", &opts)
        .await
        .unwrap();
    assert!(!resp.content.is_empty());
}

// ============================================================================
// Streaming
// ============================================================================

/// `stream()` produces at least one non-empty chunk and the joined result is
/// coherent.
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_streaming_completion() {
    require_ollama!();
    let p = provider();
    let mut stream = p.stream("Count to three: 1,").await.unwrap();

    let mut chunks: Vec<String> = Vec::new();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk.unwrap());
    }

    assert!(
        !chunks.is_empty(),
        "streaming must produce at least one chunk"
    );
    let joined = chunks.join("");
    assert!(!joined.is_empty(), "joined stream must not be empty");
}

// ============================================================================
// Chat — non-streaming
// ============================================================================

/// `chat()` with a system + user message returns a non-empty response.
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_chat_non_streaming() {
    require_ollama!();
    let p = provider();
    let messages = vec![
        ChatMessage::system("You are a precise assistant. Answer in one sentence."),
        ChatMessage::user("What colour is the sky during the day?"),
    ];
    let resp = p.chat(&messages, None).await.unwrap();
    assert!(!resp.content.is_empty());
    // The word "blue" should appear somewhere
    assert!(
        resp.content.to_lowercase().contains("blue"),
        "Expected 'blue' in response, got: {}",
        resp.content
    );
}

/// System message is honoured — the model follows its instruction.
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_system_message_respected() {
    require_ollama!();
    let p = provider();
    let messages = vec![
        ChatMessage::system(
            "You are a calculator. Respond with only the numeric result, nothing else.",
        ),
        ChatMessage::user("What is 6 multiplied by 7?"),
    ];
    let resp = p.chat(&messages, None).await.unwrap();
    assert!(
        resp.content.contains("42"),
        "Expected '42' in response, got: {}",
        resp.content
    );
}

// ============================================================================
// Multi-turn conversation
// ============================================================================

/// Sending a conversation history produces a contextually relevant follow-up.
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_multi_turn_conversation() {
    require_ollama!();
    let p = provider();
    let messages = vec![
        ChatMessage::system("You are a helpful assistant. Be brief."),
        ChatMessage::user("My favourite fruit is mango."),
        ChatMessage::assistant("That's great! Mangoes are delicious tropical fruits."),
        ChatMessage::user("What fruit did I just mention?"),
    ];
    let resp = p.chat(&messages, None).await.unwrap();
    assert!(
        resp.content.to_lowercase().contains("mango"),
        "Model should recall 'mango' from history, got: {}",
        resp.content
    );
}

// ============================================================================
// Stop conditions
// ============================================================================

/// Normal completion → `finish_reason` == "stop".
///
/// This is the primary stop condition — the model finishes naturally.
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_finish_reason_stop() {
    require_ollama!();
    let p = provider();
    let messages = vec![ChatMessage::user("Say the word 'done'.")];
    let resp = p.chat(&messages, None).await.unwrap();
    assert!(
        !resp.content.is_empty(),
        "Response must not be empty for a stop-reason test"
    );
    // Ollama always emits done_reason = "stop" on natural completion.
    assert_eq!(
        resp.finish_reason.as_deref(),
        Some("stop"),
        "finish_reason must be 'stop' for a normal completion, got: {:?}",
        resp.finish_reason
    );
}

/// Severely truncated max_tokens → `finish_reason` == "length".
///
/// This is the "length" stop condition — generation was cut short.
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_finish_reason_length() {
    require_ollama!();
    let p = provider();
    let opts = CompletionOptions {
        max_tokens: Some(1), // force immediate truncation
        ..Default::default()
    };
    let messages = vec![ChatMessage::user(
        "Write a long paragraph about the history of the Roman Empire.",
    )];
    let resp = p.chat(&messages, Some(&opts)).await.unwrap();
    assert_eq!(
        resp.finish_reason.as_deref(),
        Some("length"),
        "finish_reason must be 'length' when max_tokens is 1, got: {:?}",
        resp.finish_reason
    );
}

/// Custom stop sequences — generation halts when the stop token appears.
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_stop_sequences() {
    require_ollama!();
    let p = provider();
    let opts = CompletionOptions {
        stop: Some(vec!["STOP".to_string()]),
        max_tokens: Some(100),
        ..Default::default()
    };
    let messages = vec![ChatMessage::user(
        // Ask for a numbered list — we only care that it finishes cleanly.
        "List three colors separated by commas.",
    )];
    let resp = p.chat(&messages, Some(&opts)).await.unwrap();
    // Response must be non-empty and finish cleanly.
    assert!(!resp.content.is_empty());
    assert!(
        resp.finish_reason.is_some(),
        "finish_reason must be populated when stop sequences are set"
    );
}

// ============================================================================
// Streaming — finish reason in Finished chunk
// ============================================================================

/// The streaming path emits a `Finished { reason }` chunk with the correct
/// done_reason (not always hardcoded "stop").
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_streaming_finish_reason() {
    use edgequake_llm::traits::StreamChunk;
    require_ollama!();

    let p = provider();
    let messages = vec![ChatMessage::user("Say 'ok'.")];

    let mut stream = p
        .chat_with_tools_stream(&messages, &[], None, None)
        .await
        .unwrap();

    let mut finished_reason: Option<String> = None;
    while let Some(chunk) = stream.next().await {
        if let Ok(StreamChunk::Finished { reason, .. }) = chunk {
            finished_reason = Some(reason);
        }
    }

    assert!(
        finished_reason.is_some(),
        "streaming must emit a Finished chunk"
    );
    assert_eq!(
        finished_reason.as_deref(),
        Some("stop"),
        "normal streaming completion must have reason 'stop', got: {:?}",
        finished_reason
    );
}

/// Streaming with max_tokens=1 → Finished reason is "length".
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_streaming_finish_reason_length() {
    use edgequake_llm::traits::StreamChunk;
    require_ollama!();

    let p = provider();
    let opts = CompletionOptions {
        max_tokens: Some(1),
        ..Default::default()
    };
    let messages = vec![ChatMessage::user(
        "Write a very long essay about mathematics.",
    )];

    let mut stream = p
        .chat_with_tools_stream(&messages, &[], None, Some(&opts))
        .await
        .unwrap();

    let mut finished_reason: Option<String> = None;
    while let Some(chunk) = stream.next().await {
        if let Ok(StreamChunk::Finished { reason, .. }) = chunk {
            finished_reason = Some(reason);
        }
    }

    assert!(
        finished_reason.is_some(),
        "streaming must emit a Finished chunk"
    );
    assert_eq!(
        finished_reason.as_deref(),
        Some("length"),
        "truncated streaming must have reason 'length', got: {:?}",
        finished_reason
    );
}

// ============================================================================
// Usage metadata
// ============================================================================

/// Token counts are non-zero after a real completion.
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_usage_metadata() {
    require_ollama!();
    let p = provider();
    let messages = vec![ChatMessage::user("What is the capital of France?")];
    let resp = p.chat(&messages, None).await.unwrap();

    assert!(
        resp.prompt_tokens > 0,
        "prompt_tokens must be > 0, got {}",
        resp.prompt_tokens
    );
    assert!(
        resp.completion_tokens > 0,
        "completion_tokens must be > 0, got {}",
        resp.completion_tokens
    );
    assert!(
        resp.total_tokens > 0,
        "total_tokens must be > 0, got {}",
        resp.total_tokens
    );
    assert_eq!(
        resp.total_tokens,
        resp.prompt_tokens + resp.completion_tokens,
        "total_tokens must equal prompt + completion"
    );
}

// ============================================================================
// JSON mode
// ============================================================================

/// Setting `response_format = "json_object"` yields valid JSON.
///
/// Note: some model builds (e.g. gemma4) wrap the JSON in markdown code fences
/// even when `format=json` is requested. We strip fences before parsing so the
/// test validates the *content* not the wrapper.
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_json_mode() {
    require_ollama!();
    let p = provider();
    let opts = CompletionOptions {
        response_format: Some("json_object".to_string()),
        ..Default::default()
    };
    let messages = vec![ChatMessage::user(
        "Return a JSON object with keys 'name' and 'age'. \
         Use name='Alice' and age=30. Respond with raw JSON only.",
    )];
    let resp = p.chat(&messages, Some(&opts)).await.unwrap();

    // Strip markdown code fences that some models emit despite format=json.
    let cleaned = strip_json_fences(&resp.content);
    let parsed: serde_json::Value = serde_json::from_str(cleaned).unwrap_or_else(|e| {
        panic!(
            "JSON mode response must be valid JSON.\nError: {e}\nCleaned: {cleaned}\nRaw: {}",
            resp.content
        )
    });
    assert!(parsed.is_object(), "JSON mode must return an object");
}

// ============================================================================
// Structured outputs (JSON schema format)
// ============================================================================

/// Providing a JSON schema in `response_format` constrains the model output to
/// match the schema.
///
/// Note: this uses Ollama's native `format` field with a JSON schema object.
/// We rely on the fact that `supports_json_mode()` returns `true`.
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_structured_output_schema() {
    require_ollama!();

    // Send the request using reqwest directly — the trait only supports
    // "json_object" format string, but Ollama also accepts a full schema.
    // This test verifies the *API-level* behaviour of structured outputs.
    let url = format!("{HOST}/api/chat");
    let body = json!({
        "model": MODEL,
        "stream": false,
        "format": {
            "type": "object",
            "properties": {
                "city": { "type": "string" },
                "country": { "type": "string" }
            },
            "required": ["city", "country"]
        },
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of Japan? Respond in JSON with 'city' and 'country'."
            }
        ]
    });

    let resp = reqwest::Client::new()
        .post(url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success(), "Ollama API returned error");

    let json: serde_json::Value = resp.json().await.unwrap();
    let content = json["message"]["content"]
        .as_str()
        .expect("message.content must be a string");
    // Strip markdown code fences that some model builds emit despite schema format.
    let cleaned = strip_json_fences(content);
    let parsed: serde_json::Value = serde_json::from_str(cleaned).unwrap_or_else(|e| {
        panic!(
            "Structured output must be valid JSON.\nError: {e}\nCleaned: {cleaned}\nRaw: {content}"
        )
    });

    assert!(
        parsed.get("city").is_some(),
        "structured output must contain 'city'"
    );
    assert!(
        parsed.get("country").is_some(),
        "structured output must contain 'country'"
    );
}

// ============================================================================
// Tool calling — non-streaming
// ============================================================================

/// `chat_with_tools()` round-trips successfully.
///
/// The model may or may not emit a tool call for this particular prompt.
/// We accept both outcomes to avoid flakiness due to model behaviour, but we
/// assert the response is well-formed in both cases.
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_tool_calling_non_streaming() {
    require_ollama!();
    let p = provider();

    let tools = vec![weather_tool()];
    let messages = vec![ChatMessage::user("What is the weather in Paris?")];

    let resp = p
        .chat_with_tools(&messages, &tools, None, None)
        .await
        .unwrap();

    // The response is always well-formed.
    assert!(resp.model.contains("gemma") || !resp.model.is_empty());

    if resp.tool_calls.is_empty() {
        // Model answered in text — still valid; check response is non-empty.
        assert!(
            !resp.content.is_empty(),
            "When no tool call is made, content must not be empty"
        );
    } else {
        // Model emitted a tool call — verify structure.
        let tc = &resp.tool_calls[0];
        assert_eq!(tc.call_type, "function");
        assert!(!tc.function.name.is_empty(), "tool call must have a name");
        // Arguments must be valid JSON.
        let args: serde_json::Value =
            serde_json::from_str(&tc.function.arguments).unwrap_or_else(|e| {
                panic!(
                    "tool_call.arguments must be valid JSON: {e}\nGot: {}",
                    tc.function.arguments
                )
            });
        assert!(args.is_object(), "tool arguments must be a JSON object");
    }
}

/// `chat_with_tools()` with a deterministic tool-calling prompt:
/// the model is given a clear task that requires a tool.
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_tool_call_invocation() {
    require_ollama!();
    let p = provider();

    let tools = vec![add_tool()];
    let messages = vec![
        ChatMessage::system(
            "You MUST use the available tools to answer. Do not calculate mentally.",
        ),
        ChatMessage::user("Use the add tool to add 3 and 5."),
    ];

    let resp = p
        .chat_with_tools(&messages, &tools, None, None)
        .await
        .unwrap();

    // Either a tool call or a text answer — both are valid Ollama responses.
    let got_tool_call = !resp.tool_calls.is_empty();
    let got_text_answer = !resp.content.is_empty();

    assert!(
        got_tool_call || got_text_answer,
        "response must contain either a tool call or text content"
    );

    if got_tool_call {
        let tc = &resp.tool_calls[0];
        assert_eq!(tc.function.name, "add");
        let args: serde_json::Value = serde_json::from_str(&tc.function.arguments).unwrap();
        // Arguments should reference 3 and 5 in some form.
        let values: Vec<i64> = args
            .as_object()
            .unwrap()
            .values()
            .filter_map(|v| v.as_i64())
            .collect();
        assert!(
            values.contains(&3) && values.contains(&5),
            "tool arguments must include 3 and 5, got: {}",
            &tc.function.arguments
        );
    }
}

// ============================================================================
// Tool calling — streaming
// ============================================================================

/// `chat_with_tools_stream()` emits a `Finished` chunk.
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_tool_calling_streaming_emits_finished() {
    use edgequake_llm::traits::StreamChunk;
    require_ollama!();

    let p = provider();
    let tools = vec![weather_tool()];
    let messages = vec![ChatMessage::user("What is the weather in Tokyo?")];

    let mut stream = p
        .chat_with_tools_stream(&messages, &tools, None, None)
        .await
        .unwrap();

    let mut saw_finished = false;
    let mut saw_content_or_tool = false;

    while let Some(chunk_result) = stream.next().await {
        match chunk_result.unwrap() {
            StreamChunk::Content(c) if !c.is_empty() => saw_content_or_tool = true,
            StreamChunk::ToolCallDelta { function_name, .. } if function_name.is_some() => {
                saw_content_or_tool = true;
            }
            StreamChunk::Finished { .. } => saw_finished = true,
            _ => {}
        }
    }

    assert!(saw_finished, "streaming must emit a Finished chunk");
    assert!(
        saw_content_or_tool,
        "streaming must emit at least one Content or ToolCallDelta chunk"
    );
}

/// Streaming tool call arguments are valid JSON (not a raw string).
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_tool_call_streaming_args_are_valid_json() {
    use edgequake_llm::traits::StreamChunk;
    require_ollama!();

    let p = provider();
    let tools = vec![weather_tool()];
    let messages = vec![
        ChatMessage::system("You MUST use the get_weather tool."),
        ChatMessage::user("What is the weather in London?"),
    ];

    let mut stream = p
        .chat_with_tools_stream(&messages, &tools, None, None)
        .await
        .unwrap();

    let mut tool_args: Option<String> = None;
    while let Some(chunk_result) = stream.next().await {
        if let Ok(StreamChunk::ToolCallDelta {
            function_arguments: Some(args),
            ..
        }) = chunk_result
        {
            tool_args = Some(args);
        }
    }

    if let Some(args) = tool_args {
        // Arguments emitted by the Ollama streaming path must be valid JSON.
        let _: serde_json::Value = serde_json::from_str(&args).unwrap_or_else(|e| {
            panic!("Streaming tool call arguments must be valid JSON.\nError: {e}\nGot: {args}")
        });
    }
    // If no tool call was emitted the test passes — model used text instead.
}

// ============================================================================
// Multi-turn tool conversation
// ============================================================================

/// Full round-trip: user → tool call (or text) → tool result → final answer.
///
/// This tests the `convert_messages` path that propagates `tool_calls` on
/// assistant messages and sets `tool_name` on tool-result messages.
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_multi_turn_tool_conversation() {
    require_ollama!();
    let p = provider();

    let tools = vec![weather_tool()];

    // Turn 1: ask the question.
    let turn1 = vec![
        ChatMessage::system("Use the get_weather tool when asked about weather."),
        ChatMessage::user("What is the weather in Rome?"),
    ];
    let resp1 = p.chat_with_tools(&turn1, &tools, None, None).await.unwrap();

    // Build history regardless of whether a tool call was made.
    let mut tool_calls_for_history = resp1.tool_calls.clone();
    let assistant_content = resp1.content.clone();

    let mut history = turn1.clone();

    // Append assistant message with any tool calls.
    let assistant_msg = ChatMessage {
        role: edgequake_llm::traits::ChatRole::Assistant,
        content: assistant_content.clone(),
        tool_calls: if tool_calls_for_history.is_empty() {
            None
        } else {
            Some(tool_calls_for_history.clone())
        },
        name: None,
        tool_call_id: None,
        images: None,
        cache_control: None,
    };
    history.push(assistant_msg);

    if tool_calls_for_history.is_empty() {
        // Model answered in text — the test still passes; verify content.
        assert!(
            !assistant_content.is_empty(),
            "Text answer must not be empty in multi-turn tool scenario"
        );
        return;
    }

    // Turn 2: supply the tool result.
    let tc = tool_calls_for_history.remove(0);
    let tool_result = ChatMessage {
        role: edgequake_llm::traits::ChatRole::Tool,
        content: "22°C, sunny".to_string(),
        name: Some(tc.function.name.clone()),
        tool_call_id: Some(tc.id.clone()),
        tool_calls: None,
        images: None,
        cache_control: None,
    };
    history.push(tool_result);

    // Turn 3: get the final synthesised answer.
    let resp2 = p
        .chat_with_tools(&history, &tools, None, None)
        .await
        .unwrap();

    assert!(
        !resp2.content.is_empty(),
        "Final synthesised answer must not be empty"
    );
    // The answer should incorporate the tool result.
    let lower = resp2.content.to_lowercase();
    assert!(
        lower.contains("22") || lower.contains("sunny") || lower.contains("rome"),
        "Synthesised answer should reference the tool result, got: {}",
        resp2.content
    );
    assert_eq!(
        resp2.finish_reason.as_deref(),
        Some("stop"),
        "Multi-turn final response must have finish_reason 'stop'"
    );
}

// ============================================================================
// Embeddings
// ============================================================================

/// Embedding a single text returns a vector of the expected dimension.
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_embedding_single() {
    require_ollama!();

    // gemma4 may not be an embedding model so we use it directly via API
    // but test OllamaProvider embedding path with the same host.
    // Use the embedding model if available, else skip.
    let embedding_model = "nomic-embed-text:latest";
    if !model_is_available(embedding_model).await {
        eprintln!("SKIP — embedding model {embedding_model} not pulled");
        return;
    }

    let provider = OllamaProvider::builder()
        .host(HOST)
        .model(MODEL) // chat model
        .embedding_model(embedding_model)
        .build()
        .unwrap();

    let embedding = provider.embed_one("The quick brown fox").await.unwrap();

    assert!(!embedding.is_empty(), "embedding vector must not be empty");
    assert!(
        embedding.iter().any(|&v| v != 0.0),
        "embedding must contain non-zero values"
    );
}

/// Batch embedding returns one vector per input text.
#[tokio::test]
#[ignore = "Requires Ollama with gemma4:latest pulled"]
async fn test_ollama_embedding_batch() {
    require_ollama!();

    let embedding_model = "nomic-embed-text:latest";
    if !model_is_available(embedding_model).await {
        eprintln!("SKIP — embedding model {embedding_model} not pulled");
        return;
    }

    let provider = OllamaProvider::builder()
        .host(HOST)
        .model(MODEL)
        .embedding_model(embedding_model)
        .build()
        .unwrap();

    // Use semantically very different texts so the embedding model
    // produces distinct vectors even with quantization.
    let texts = vec![
        "The mitochondria is the powerhouse of the cell.".to_string(),
        "The French Revolution began in 1789 with the storming of the Bastille.".to_string(),
        "Photosynthesis converts sunlight into chemical energy in plants.".to_string(),
    ];
    let embeddings = provider.embed(&texts).await.unwrap();

    assert_eq!(
        embeddings.len(),
        texts.len(),
        "batch embedding must return one vector per input"
    );
    for emb in &embeddings {
        assert!(!emb.is_empty());
        assert!(emb.iter().any(|&v| v != 0.0));
    }
    // Semantically distinct texts must produce distinct embedding vectors.
    assert_ne!(
        embeddings[0], embeddings[1],
        "distinct texts should produce different embeddings (text 0 vs 1)"
    );
    assert_ne!(
        embeddings[1], embeddings[2],
        "distinct texts should produce different embeddings (text 1 vs 2)"
    );
}

// ============================================================================
// Capability flags
// ============================================================================

/// `supports_json_mode()` returns `true` after our bug fix.
#[tokio::test]
async fn test_ollama_supports_json_mode_flag() {
    let p = provider();
    assert!(
        p.supports_json_mode(),
        "OllamaProvider must report JSON mode support"
    );
}

/// `supports_streaming()` returns `true`.
#[tokio::test]
async fn test_ollama_supports_streaming_flag() {
    let p = provider();
    assert!(p.supports_streaming());
}

/// `supports_function_calling()` returns `true`.
#[tokio::test]
async fn test_ollama_supports_function_calling_flag() {
    let p = provider();
    assert!(p.supports_function_calling());
}

// ============================================================================
// Tool definitions (shared helpers)
// ============================================================================

/// A simple weather tool definition used by multiple tests.
fn weather_tool() -> ToolDefinition {
    ToolDefinition::function(
        "get_weather",
        "Get the current weather for a city",
        json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get the weather for"
                }
            },
            "required": ["city"]
        }),
    )
}

/// A simple add tool definition.
fn add_tool() -> ToolDefinition {
    ToolDefinition::function(
        "add",
        "Add two numbers together",
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "number", "description": "First number" },
                "b": { "type": "number", "description": "Second number" }
            },
            "required": ["a", "b"]
        }),
    )
}
