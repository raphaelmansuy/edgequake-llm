//! E2E regression tests for async-openai 0.33 upgrade.
//!
//! These tests verify that all functionality works correctly after upgrading
//! from async-openai 0.24 to 0.33, including:
//! - Native max_completion_tokens support (fix for issue #13)
//! - Chat completions with legacy and new model families
//! - Streaming with SSE
//! - Embeddings
//! - Tool/function calling
//! - Vision/multimodal messages
//!
//! Requires OPENAI_API_KEY environment variable.
//! Run with: cargo test --test e2e_openai_033 -- --ignored

use edgequake_llm::providers::openai::OpenAIProvider;
use edgequake_llm::traits::{
    ChatMessage, CompletionOptions, EmbeddingProvider, ImageData, LLMProvider, StreamChunk,
    ToolChoice, ToolDefinition,
};
use futures::StreamExt;

/// Helper: create OpenAI provider (skips test if OPENAI_API_KEY not set).
fn make_provider() -> Option<OpenAIProvider> {
    let api_key = std::env::var("OPENAI_API_KEY").ok()?;
    Some(OpenAIProvider::new(api_key))
}

// ── Basic Chat Completions ─────────────────────────────────────────────────

/// Test basic chat with gpt-4o (legacy model using max_completion_tokens).
/// Verifies the upgrade doesn't break existing models.
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_e2e_basic_chat_gpt4o() {
    let provider = match make_provider() {
        Some(p) => p.with_model("gpt-4o-mini"),
        None => {
            eprintln!("OPENAI_API_KEY not set, skipping test");
            return;
        }
    };

    let messages = vec![ChatMessage::user("What is 2+2? Reply with just the number.")];
    let response = provider.chat(&messages, None).await.expect("Chat failed");

    assert!(!response.content.is_empty(), "Response content should not be empty");
    assert!(response.content.contains('4'), "Response should contain '4'");
    assert!(response.prompt_tokens > 0, "Should have prompt tokens");
    assert!(response.completion_tokens > 0, "Should have completion tokens");
    assert_eq!(response.total_tokens, response.prompt_tokens + response.completion_tokens);

    println!(
        "gpt-4o-mini response: '{}', tokens: {}/{}/{}, cache_hit: {:?}",
        response.content,
        response.prompt_tokens,
        response.completion_tokens,
        response.total_tokens,
        response.cache_hit_tokens
    );
}

/// Test chat with gpt-4.1-nano (issue #13 model family).
/// Verifies max_completion_tokens is sent natively and the model accepts it.
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_e2e_basic_chat_gpt41_nano() {
    let provider = match make_provider() {
        Some(p) => p.with_model("gpt-4.1-nano"),
        None => {
            eprintln!("OPENAI_API_KEY not set, skipping test");
            return;
        }
    };

    let messages = vec![ChatMessage::user("What is the capital of France? One word answer.")];
    let options = CompletionOptions {
        max_tokens: Some(50),
        ..Default::default()
    };

    let response = provider.chat(&messages, Some(&options)).await;

    match response {
        Ok(resp) => {
            assert!(!resp.content.is_empty(), "Response should not be empty");
            println!(
                "gpt-4.1-nano response: '{}', tokens: {}/{}/{}",
                resp.content, resp.prompt_tokens, resp.completion_tokens, resp.total_tokens
            );
        }
        Err(e) => {
            // If the model is not available, skip gracefully
            let err_str = e.to_string();
            if err_str.contains("model") && err_str.contains("not found") {
                eprintln!("gpt-4.1-nano not available in this account, skipping: {}", e);
            } else {
                panic!("Unexpected error: {}", e);
            }
        }
    }
}

/// Test chat with system prompt and max_tokens option.
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_e2e_chat_with_system_prompt() {
    let provider = match make_provider() {
        Some(p) => p.with_model("gpt-4o-mini"),
        None => return,
    };

    let messages = vec![
        ChatMessage::system("You are a concise math tutor. Only answer with numbers."),
        ChatMessage::user("What is 7 times 8?"),
    ];
    let options = CompletionOptions {
        max_tokens: Some(20),
        temperature: Some(0.0),
        ..Default::default()
    };

    let response = provider
        .chat(&messages, Some(&options))
        .await
        .expect("Chat with system prompt failed");

    assert!(!response.content.is_empty());
    assert!(response.content.contains("56"), "7*8=56 should be in response");
    println!("System prompt response: '{}'", response.content);
}

// ── Streaming ─────────────────────────────────────────────────────────────

/// Test streaming chat completion.
/// Verifies that the SSE stream works and accumulates content correctly.
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_e2e_streaming_chat() {
    let provider = match make_provider() {
        Some(p) => p.with_model("gpt-4o-mini"),
        None => return,
    };

    let stream = provider
        .stream("Count from 1 to 5, one number per line.")
        .await
        .expect("Failed to start stream");

    let chunks: Vec<_> = stream.collect().await;
    let full_content: String = chunks
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();

    assert!(!full_content.is_empty(), "Stream should produce content");
    println!("Streamed content: '{}'", full_content);

    // Should contain at least some numbers
    assert!(
        full_content.contains('1') && full_content.contains('5'),
        "Stream should contain numbers 1-5"
    );
}

// ── Token Limits ───────────────────────────────────────────────────────────

/// Test that max_completion_tokens is respected (response stays short).
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_e2e_max_completion_tokens_respected() {
    let provider = match make_provider() {
        Some(p) => p.with_model("gpt-4o-mini"),
        None => return,
    };

    let options = CompletionOptions {
        max_tokens: Some(5), // Very small limit
        ..Default::default()
    };

    let messages = vec![ChatMessage::user("Tell me everything you know about the universe.")];
    let response = provider
        .chat(&messages, Some(&options))
        .await
        .expect("Chat failed");

    // Response should be very short due to token limit
    assert!(
        response.completion_tokens <= 10,
        "Completion tokens ({}) should be close to the 5-token limit",
        response.completion_tokens
    );
    println!(
        "Token-limited response: '{}' (completion_tokens={})",
        response.content, response.completion_tokens
    );
}

// ── Embeddings ────────────────────────────────────────────────────────────

/// Test embedding generation.
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_e2e_embeddings() {
    let provider = match make_provider() {
        Some(p) => p.with_embedding_model("text-embedding-3-small"),
        None => return,
    };

    let texts = vec![
        "The quick brown fox jumps over the lazy dog".to_string(),
        "Hello, world!".to_string(),
    ];

    let embeddings = provider.embed(&texts).await.expect("Embedding failed");

    assert_eq!(embeddings.len(), 2, "Should get 2 embedding vectors");
    assert_eq!(embeddings[0].len(), 1536, "text-embedding-3-small should be 1536-dim");
    assert_eq!(embeddings[1].len(), 1536);

    // Embeddings should be normalized (magnitude close to 1.0)
    let mag: f32 = embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (mag - 1.0).abs() < 0.01,
        "Embedding should be unit-normalized, got magnitude {}",
        mag
    );

    println!(
        "Generated {} embeddings, dim={}, first 3 values: {:?}",
        embeddings.len(),
        embeddings[0].len(),
        &embeddings[0][..3]
    );
}

/// Test that empty input returns empty embeddings (no API call).
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_e2e_embeddings_empty_input() {
    let provider = match make_provider() {
        Some(p) => p,
        None => return,
    };

    let result = provider.embed(&[]).await.expect("Empty embed failed");
    assert!(result.is_empty(), "Empty input should return empty result");
}

// ── Tool Calling ──────────────────────────────────────────────────────────

/// Test tool calling with streaming (validates ChatCompletionTools enum fix).
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_e2e_tool_calling_stream() {
    let provider = match make_provider() {
        Some(p) => p.with_model("gpt-4o-mini"),
        None => return,
    };

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

    let messages = vec![ChatMessage::user("What's the weather in Paris?")];
    let options = CompletionOptions {
        max_tokens: Some(200),
        ..Default::default()
    };

    let stream = provider
        .chat_with_tools_stream(&messages, &tools, Some(ToolChoice::auto()), Some(&options))
        .await
        .expect("Tool stream failed");

    let chunks: Vec<_> = stream.collect().await;
    assert!(!chunks.is_empty(), "Stream should produce chunks");

    // At least one chunk should be ToolCallDelta or Content
    let has_tool_or_content = chunks.iter().any(|c| {
        matches!(
            c,
            Ok(StreamChunk::ToolCallDelta { .. }) | Ok(StreamChunk::Content(_))
        )
    });
    assert!(
        has_tool_or_content,
        "Stream should contain tool call delta or content"
    );

    println!("Tool calling stream: {} chunks", chunks.len());
    for chunk in chunks.iter().take(5) {
        println!("  chunk: {:?}", chunk);
    }
}

// ── Vision / Multimodal ───────────────────────────────────────────────────

/// Test multimodal ChatMessage construction (unit-level, no API call needed).
/// Verifies that ImageData and user_with_images compile and produce the expected struct.
#[test]
fn test_multimodal_message_construction_unit() {
    // 1×1 transparent PNG, base64-encoded.
    let tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    let img = ImageData::new(tiny_png, "image/png").with_detail("high");

    let msg = ChatMessage::user_with_images("Describe this image", vec![img.clone()]);
    assert!(msg.images.is_some());
    assert_eq!(msg.images.as_ref().unwrap().len(), 1);
    assert_eq!(msg.images.as_ref().unwrap()[0].data, tiny_png);
    assert_eq!(msg.images.as_ref().unwrap()[0].mime_type, "image/png");
    assert_eq!(msg.images.as_ref().unwrap()[0].detail.as_deref(), Some("high"));

    // Verify clone and Debug work.
    let _cloned = msg.images.unwrap()[0].clone();
    println!("Multimodal message construction unit test passed (no API call)");
}

// ── Cache Hit and Reasoning Tokens ─────────────────────────────────────────

/// Test that cache hit tokens are extracted when available.
/// Requires a second call to same prompt to trigger caching.
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_e2e_cache_hit_token_extraction() {
    let provider = match make_provider() {
        Some(p) => p.with_model("gpt-4o-mini"),
        None => return,
    };

    // Make two identical requests to trigger prompt caching
    let messages = vec![
        ChatMessage::system("You are a helpful assistant. Always respond in exactly 3 words."),
        ChatMessage::user("What is 1+1?"),
    ];

    // First call to prime the cache
    let _ = provider.chat(&messages, None).await.expect("First call failed");

    // Second call should have cache hit tokens (if OpenAI prompt caching kicks in)
    let response = provider.chat(&messages, None).await.expect("Second call failed");

    println!(
        "Response: '{}', cache_hit_tokens: {:?}, thinking_tokens: {:?}",
        response.content, response.cache_hit_tokens, response.thinking_tokens
    );

    // We can't guarantee cache hits in tests, but we verify extraction works without panic
    assert!(
        response.cache_hit_tokens.is_none() || response.cache_hit_tokens.unwrap() > 0,
        "Cache hit tokens should be None or positive"
    );
}

/// Test that complete_with_options works end-to-end.
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_e2e_complete_with_options() {
    let provider = match make_provider() {
        Some(p) => p.with_model("gpt-4o-mini"),
        None => return,
    };

    let options = CompletionOptions {
        system_prompt: Some("You are a helpful assistant. Be very brief.".to_string()),
        max_tokens: Some(50),
        temperature: Some(0.1),
        ..Default::default()
    };

    let response = provider
        .complete_with_options("Say 'hello world'", &options)
        .await
        .expect("complete_with_options failed");

    assert!(!response.content.is_empty());
    println!(
        "complete_with_options: '{}' ({} tokens)",
        response.content, response.total_tokens
    );
}
