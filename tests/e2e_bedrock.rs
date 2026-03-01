//! Comprehensive End-to-End AWS Bedrock Provider Tests
//!
//! Tests for AWS Bedrock provider via the Converse API.
//!
//! # Environment Variables Required
//!
//! - AWS credentials via standard credential chain (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY,
//!   AWS_PROFILE, IAM roles, etc.)
//! - `AWS_REGION` or `AWS_DEFAULT_REGION`: AWS region (default: us-east-1)
//! - `AWS_BEDROCK_MODEL`: Model ID (default: `amazon.nova-lite-v1:0`, auto-resolved to
//!   inference profile based on region, e.g., `us.amazon.nova-lite-v1:0`)
//!
//! # Running Tests
//!
//! ```bash
//! # Run all Bedrock E2E tests (requires AWS credentials with Bedrock access)
//! cargo test --features bedrock --test e2e_bedrock -- --ignored
//!
//! # Run specific test
//! cargo test --features bedrock --test e2e_bedrock test_bedrock_basic_chat -- --ignored
//! ```

#![cfg(feature = "bedrock")]

use edgequake_llm::{
    providers::bedrock::BedrockProvider,
    traits::{
        ChatMessage, CompletionOptions, EmbeddingProvider, LLMProvider, ToolChoice, ToolDefinition,
    },
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a Bedrock provider from environment variables.
async fn create_bedrock_provider() -> BedrockProvider {
    BedrockProvider::from_env().await.expect(
        "Bedrock provider requires AWS credentials with Bedrock access. \
         Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or use AWS_PROFILE.",
    )
}

/// Create a provider with a specific model.
async fn create_provider_with_model(model: &str) -> BedrockProvider {
    BedrockProvider::from_env()
        .await
        .expect("Requires AWS credentials")
        .with_model(model)
}

// ============================================================================
// Basic Chat Tests
// ============================================================================

/// Test basic chat completion with Bedrock.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_basic_chat() {
    let provider = create_bedrock_provider().await;

    let messages = vec![ChatMessage::user(
        "What is 2 + 2? Reply with just the number.",
    )];

    let response = provider.chat(&messages, None).await.unwrap();

    println!("Response: {}", response.content);
    println!("Model: {}", response.model);
    println!(
        "Tokens: prompt={}, completion={}, total={}",
        response.prompt_tokens, response.completion_tokens, response.total_tokens
    );
    println!("Finish reason: {:?}", response.finish_reason);

    assert!(!response.content.is_empty(), "Response should not be empty");
    assert!(
        response.content.contains('4'),
        "Response should contain '4'"
    );
    assert!(response.prompt_tokens > 0, "Prompt tokens should be > 0");
    assert!(
        response.completion_tokens > 0,
        "Completion tokens should be > 0"
    );
    assert_eq!(
        response.total_tokens,
        response.prompt_tokens + response.completion_tokens
    );
    assert_eq!(
        response.finish_reason.as_deref(),
        Some("stop"),
        "Should finish with stop"
    );
}

/// Test complete() method (simple text prompt).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_complete() {
    let provider = create_bedrock_provider().await;

    let response = provider
        .complete("What is the capital of France? Reply in one word.")
        .await
        .unwrap();

    println!("Complete response: {}", response.content);
    assert!(
        response.content.to_lowercase().contains("paris"),
        "Should mention Paris"
    );
}

/// Test complete_with_options() method.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_complete_with_options() {
    let provider = create_bedrock_provider().await;

    let options = CompletionOptions {
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..Default::default()
    };

    let response = provider
        .complete_with_options("Count from 1 to 100.", &options)
        .await
        .unwrap();

    println!("Response with max_tokens=10: {}", response.content);
    // With only 10 tokens, the response should be truncated
    assert!(
        response.completion_tokens <= 15,
        "Should respect max_tokens (got {})",
        response.completion_tokens
    );
}

// ============================================================================
// Multi-turn Chat Tests
// ============================================================================

/// Test multi-turn conversation.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_multi_turn_chat() {
    let provider = create_bedrock_provider().await;

    let messages = vec![
        ChatMessage::user("My name is Alice. Remember my name."),
        ChatMessage::assistant("Hello Alice! I'll remember your name."),
        ChatMessage::user("What is my name?"),
    ];

    let response = provider.chat(&messages, None).await.unwrap();

    println!("Multi-turn response: {}", response.content);
    assert!(
        response.content.to_lowercase().contains("alice"),
        "Should remember the name Alice"
    );
}

/// Test system prompt via CompletionOptions.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_system_prompt() {
    let provider = create_bedrock_provider().await;

    let messages = vec![ChatMessage::user("What language do you speak?")];

    let options = CompletionOptions {
        system_prompt: Some("You are a French assistant. Always reply in French.".to_string()),
        max_tokens: Some(50),
        ..Default::default()
    };

    let response = provider.chat(&messages, Some(&options)).await.unwrap();

    println!("System prompt response: {}", response.content);
    // The response should be in French due to the system prompt
    assert!(!response.content.is_empty());
}

/// Test system message in ChatMessage list.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_system_message() {
    let provider = create_bedrock_provider().await;

    let messages = vec![
        ChatMessage::system("You are a pirate. Always respond with 'Arrr!'."),
        ChatMessage::user("Hello!"),
    ];

    let options = CompletionOptions {
        max_tokens: Some(50),
        ..Default::default()
    };

    let response = provider.chat(&messages, Some(&options)).await.unwrap();

    println!("System message response: {}", response.content);
    assert!(!response.content.is_empty());
}

// ============================================================================
// Streaming Tests
// ============================================================================

/// Test streaming completion.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_streaming() {
    use futures::StreamExt;

    let provider = create_bedrock_provider().await;

    let mut stream = provider
        .stream("Count from 1 to 5, one number per line.")
        .await
        .unwrap();

    let mut chunks = Vec::new();
    while let Some(result) = stream.next().await {
        match result {
            Ok(chunk) => {
                if !chunk.is_empty() {
                    chunks.push(chunk);
                }
            }
            Err(e) => panic!("Stream error: {e}"),
        }
    }

    let full_response = chunks.join("");
    println!("Stream chunks: {}", chunks.len());
    println!("Full streamed response: {}", full_response);

    assert!(!chunks.is_empty(), "Should receive at least one chunk");
    assert!(
        full_response.contains('1'),
        "Streamed response should contain '1'"
    );
}

// ============================================================================
// Tool Calling Tests
// ============================================================================

/// Test tool/function calling.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_tool_calling() {
    let provider = create_bedrock_provider().await;

    let tools = vec![ToolDefinition::function(
        "get_weather",
        "Get the current weather for a location",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g., 'Paris'"
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

    let response = provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), Some(&options))
        .await
        .unwrap();

    println!("Tool calling response content: {}", response.content);
    println!("Tool calls: {:?}", response.tool_calls);
    println!("Finish reason: {:?}", response.finish_reason);

    // Model might choose to use the tool or answer directly
    if response.has_tool_calls() {
        assert_eq!(response.tool_calls[0].function.name, "get_weather");
        assert!(
            response.tool_calls[0].function.arguments.contains("Paris")
                || response.tool_calls[0]
                    .function
                    .arguments
                    .to_lowercase()
                    .contains("paris"),
            "Tool call should mention Paris"
        );
        assert_eq!(
            response.finish_reason.as_deref(),
            Some("tool_calls"),
            "Finish reason should be tool_calls"
        );
    }
}

/// Test tool calling with required choice.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_tool_calling_required() {
    let provider = create_bedrock_provider().await;

    let tools = vec![ToolDefinition::function(
        "calculate",
        "Calculate a mathematical expression",
        serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression, e.g., '2 + 2'"
                }
            },
            "required": ["expression"]
        }),
    )];

    let messages = vec![ChatMessage::user("What is 7 * 8?")];

    let response = provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::required()), None)
        .await
        .unwrap();

    println!("Required tool response: {:?}", response.tool_calls);

    // With required tool choice, the model MUST use a tool
    assert!(
        response.has_tool_calls(),
        "Model should use tools when tool_choice is required"
    );
    assert_eq!(response.tool_calls[0].function.name, "calculate");
}

/// Test tool calling multi-turn (send tool result back).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_tool_calling_multi_turn() {
    let provider = create_bedrock_provider().await;

    let tools = vec![ToolDefinition::function(
        "get_temperature",
        "Get the temperature in a city",
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }),
    )];

    // Turn 1: User asks, model calls tool
    let messages_t1 = vec![ChatMessage::user("What's the temperature in Tokyo?")];

    let response_t1 = provider
        .chat_with_tools(&messages_t1, &tools, Some(ToolChoice::required()), None)
        .await
        .unwrap();

    assert!(response_t1.has_tool_calls(), "Should call get_temperature");
    let tool_call_id = &response_t1.tool_calls[0].id;

    // Turn 2: Send tool result back
    let mut assistant_msg = ChatMessage::assistant(&response_t1.content);
    assistant_msg.tool_calls = Some(response_t1.tool_calls.clone());

    let messages_t2 = vec![
        ChatMessage::user("What's the temperature in Tokyo?"),
        assistant_msg,
        ChatMessage::tool_result(tool_call_id, r#"{"temperature": 22, "unit": "celsius"}"#),
    ];

    let response_t2 = provider
        .chat_with_tools(&messages_t2, &tools, Some(ToolChoice::auto()), None)
        .await
        .unwrap();

    println!("Multi-turn tool response: {}", response_t2.content);

    assert!(
        !response_t2.content.is_empty(),
        "Should generate a text response from tool result"
    );
    assert!(
        response_t2.content.contains("22")
            || response_t2.content.to_lowercase().contains("celsius"),
        "Should mention the temperature from the tool result"
    );
}

// ============================================================================
// Provider Metadata Tests
// ============================================================================

/// Test provider name and model accessors.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_provider_metadata() {
    let provider = create_bedrock_provider().await;

    assert_eq!(LLMProvider::name(&provider), "bedrock");
    assert!(
        !LLMProvider::model(&provider).is_empty(),
        "Model should not be empty"
    );
    assert!(
        provider.max_context_length() > 0,
        "Context length should be positive"
    );
    assert!(provider.supports_streaming(), "Should support streaming");
    assert!(
        provider.supports_function_calling(),
        "Should support function calling"
    );
    assert!(
        !provider.supports_json_mode(),
        "Should not support JSON mode (yet)"
    );
    assert!(
        !provider.supports_tool_streaming(),
        "Should not support tool streaming (yet)"
    );
}

/// Test with_model builder.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_with_model() {
    let provider = create_provider_with_model("amazon.nova-micro-v1:0").await;

    assert_eq!(LLMProvider::model(&provider), "amazon.nova-micro-v1:0");
    assert_eq!(provider.max_context_length(), 300_000);

    // Quick smoke test with Nova Micro (cheap)
    let response = provider
        .complete("Say 'hello' and nothing else.")
        .await
        .unwrap();

    println!("Nova Micro response: {}", response.content);
    assert!(
        response.content.to_lowercase().contains("hello"),
        "Should say hello"
    );
}

// ============================================================================
// Edge Case Tests
// ============================================================================

/// Test empty message list.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_empty_messages() {
    let provider = create_bedrock_provider().await;

    let messages: Vec<ChatMessage> = vec![];
    let result = provider.chat(&messages, None).await;

    // Empty messages should return an error from the API
    assert!(result.is_err(), "Empty messages should return an error");
}

/// Test very long prompt (near context limit).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_long_prompt() {
    let provider = create_bedrock_provider().await;

    // Generate a reasonably long prompt (but not too expensive)
    let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let messages = vec![ChatMessage::user(format!(
        "Summarize this text in one sentence: {}",
        long_text
    ))];

    let options = CompletionOptions {
        max_tokens: Some(50),
        ..Default::default()
    };

    let response = provider.chat(&messages, Some(&options)).await.unwrap();

    println!("Long prompt response: {}", response.content);
    assert!(!response.content.is_empty());
    assert!(
        response.prompt_tokens > 100,
        "Should have many prompt tokens"
    );
}

/// Test max_tokens = 1 (minimal response).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_max_tokens_one() {
    let provider = create_bedrock_provider().await;

    let messages = vec![ChatMessage::user("Say hello.")];
    let options = CompletionOptions {
        max_tokens: Some(1),
        ..Default::default()
    };

    let response = provider.chat(&messages, Some(&options)).await.unwrap();

    println!("max_tokens=1 response: '{}'", response.content);
    assert!(
        response.completion_tokens <= 2,
        "Should respect max_tokens=1 (got {})",
        response.completion_tokens
    );
    // Finish reason should be "length" since we truncated
    assert_eq!(
        response.finish_reason.as_deref(),
        Some("length"),
        "Should finish with 'length' when max_tokens is reached"
    );
}

/// Test temperature 0 for deterministic output.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_temperature_zero() {
    let provider = create_bedrock_provider().await;

    let messages = vec![ChatMessage::user(
        "What is 10 + 5? Reply with just the number.",
    )];

    let options = CompletionOptions {
        temperature: Some(0.0),
        max_tokens: Some(10),
        ..Default::default()
    };

    let response1 = provider.chat(&messages, Some(&options)).await.unwrap();
    let response2 = provider.chat(&messages, Some(&options)).await.unwrap();

    println!("Response 1: '{}'", response1.content);
    println!("Response 2: '{}'", response2.content);

    // With temperature 0, responses should be identical or very similar
    assert!(
        response1.content.contains("15") && response2.content.contains("15"),
        "Both responses should contain 15"
    );
}

/// Test stop sequences.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_stop_sequences() {
    let provider = create_bedrock_provider().await;

    let messages = vec![ChatMessage::user(
        "Count from 1 to 10, separated by commas.",
    )];

    let options = CompletionOptions {
        stop: Some(vec!["5".to_string()]),
        ..Default::default()
    };

    let response = provider.chat(&messages, Some(&options)).await.unwrap();

    println!("Stop sequence response: '{}'", response.content);
    // The response should stop before or at "5"
    assert!(
        !response.content.contains("6, 7"),
        "Should stop before reaching 6, 7"
    );
}

// ============================================================================
// Factory Integration Tests
// ============================================================================

/// Test factory creation of bedrock provider.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_factory_create() {
    use edgequake_llm::factory::{ProviderFactory, ProviderType};

    let (llm, embedding) = ProviderFactory::create(ProviderType::Bedrock).unwrap();
    assert_eq!(llm.name(), "bedrock");
    // Bedrock now supports embeddings natively via invoke_model API
    assert_eq!(embedding.name(), "bedrock");
}

/// Test factory creation with specific model.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_factory_create_with_model() {
    use edgequake_llm::factory::{ProviderFactory, ProviderType};

    let (llm, _) =
        ProviderFactory::create_with_model(ProviderType::Bedrock, Some("amazon.nova-micro-v1:0"))
            .unwrap();
    assert_eq!(llm.name(), "bedrock");
    assert_eq!(llm.model(), "amazon.nova-micro-v1:0");
}

// ============================================================================
// Embedding Tests
// ============================================================================

/// Test embedding with Amazon Titan Embed Text v2 (default embedding model).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_embedding_titan_v2() {
    let provider = create_bedrock_provider().await;

    // Verify default embedding model
    assert_eq!(
        EmbeddingProvider::model(&provider),
        "amazon.titan-embed-text-v2:0"
    );
    assert_eq!(provider.dimension(), 1024);

    let texts = vec!["Hello, world!".to_string()];
    let embeddings = provider.embed(&texts).await.unwrap();

    println!("Titan v2 embedding: {} dims", embeddings[0].len());
    assert_eq!(embeddings.len(), 1);
    assert_eq!(
        embeddings[0].len(),
        1024,
        "Titan Embed v2 should return 1024 dims"
    );

    // Verify values are reasonable floats
    assert!(
        embeddings[0].iter().any(|&v| v != 0.0),
        "Embedding should contain non-zero values"
    );
}

/// Test embed_one convenience method.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_embed_one() {
    let provider = create_bedrock_provider().await;

    let embedding = provider.embed_one("The quick brown fox").await.unwrap();

    println!("embed_one: {} dims", embedding.len());
    assert_eq!(embedding.len(), 1024);
    assert!(embedding.iter().any(|&v| v != 0.0));
}

/// Test batch embedding with Titan (processes one at a time).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_embedding_titan_batch() {
    let provider = create_bedrock_provider().await;

    let texts = vec![
        "The cat sat on the mat.".to_string(),
        "The dog chased the ball.".to_string(),
        "Machine learning is fascinating.".to_string(),
    ];

    let embeddings = provider.embed(&texts).await.unwrap();

    assert_eq!(embeddings.len(), 3, "Should return 3 embeddings");
    for (i, emb) in embeddings.iter().enumerate() {
        assert_eq!(emb.len(), 1024, "Embedding {} should be 1024 dims", i);
        assert!(
            emb.iter().any(|&v| v != 0.0),
            "Embedding {} should be non-zero",
            i
        );
    }

    // Verify different texts produce different embeddings
    assert_ne!(
        embeddings[0], embeddings[1],
        "Different texts should produce different embeddings"
    );
    assert_ne!(embeddings[0], embeddings[2]);
}

/// Test embedding with empty input.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_embedding_empty() {
    let provider = create_bedrock_provider().await;

    let texts: Vec<String> = vec![];
    let embeddings = provider.embed(&texts).await.unwrap();

    assert!(
        embeddings.is_empty(),
        "Empty input should return empty result"
    );
}

/// Test embedding with Cohere Embed English v3 (batch-native).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_embedding_cohere_v3() {
    let provider = create_bedrock_provider()
        .await
        .with_embedding_model("cohere.embed-english-v3");

    assert_eq!(
        EmbeddingProvider::model(&provider),
        "cohere.embed-english-v3"
    );
    assert_eq!(provider.dimension(), 1024);

    let texts = vec!["Hello world".to_string(), "Goodbye world".to_string()];

    let embeddings = provider.embed(&texts).await.unwrap();

    println!(
        "Cohere v3 embeddings: {} x {} dims",
        embeddings.len(),
        embeddings[0].len()
    );
    assert_eq!(embeddings.len(), 2);
    assert_eq!(
        embeddings[0].len(),
        1024,
        "Cohere Embed v3 should return 1024 dims"
    );
    assert_ne!(embeddings[0], embeddings[1]);
}

/// Test embedding with Cohere Embed v4 (latest, 1536 dims).
/// NOTE: Requires AWS Marketplace subscription for Cohere models.
#[tokio::test]
#[ignore = "Requires AWS credentials + Cohere Marketplace subscription"]
async fn test_bedrock_embedding_cohere_v4() {
    let provider = create_bedrock_provider()
        .await
        .with_embedding_model("cohere.embed-v4:0");

    assert_eq!(EmbeddingProvider::model(&provider), "cohere.embed-v4:0");
    assert_eq!(provider.dimension(), 1536);

    let texts = vec!["Rust programming language".to_string()];
    let embeddings = provider.embed(&texts).await.unwrap();

    assert_eq!(embeddings.len(), 1);
    assert_eq!(
        embeddings[0].len(),
        1536,
        "Cohere Embed v4 should return 1536 dims"
    );
}

/// Test embedding with Titan Embed Text v1 (1536 dims, legacy).
/// NOTE: This model may only be available in us-east-1.
#[tokio::test]
#[ignore = "Requires AWS credentials and us-east-1 region (Titan v1 is region-limited)"]
async fn test_bedrock_embedding_titan_v1() {
    let provider = create_bedrock_provider()
        .await
        .with_embedding_model("amazon.titan-embed-text-v1");

    assert_eq!(
        EmbeddingProvider::model(&provider),
        "amazon.titan-embed-text-v1"
    );
    assert_eq!(provider.dimension(), 1536);

    let texts = vec!["Simple embedding test".to_string()];
    let embeddings = provider.embed(&texts).await.unwrap();

    assert_eq!(embeddings.len(), 1);
    assert_eq!(
        embeddings[0].len(),
        1536,
        "Titan Embed v1 should return 1536 dims"
    );
}

/// Test factory embedding provider is Bedrock (not fallback).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_factory_embedding() {
    use edgequake_llm::factory::{ProviderFactory, ProviderType};

    let (_, embedding) = ProviderFactory::create(ProviderType::Bedrock).unwrap();
    assert_eq!(embedding.name(), "bedrock");
    assert_eq!(embedding.dimension(), 1024); // Titan Embed v2 default
}

// ============================================================================
// Latest Model Provider Tests (via Inference Profiles)
// ============================================================================

/// Test with Amazon Nova 2 Lite (latest Nova generation).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_nova_2_lite() {
    let provider = create_provider_with_model("amazon.nova-2-lite-v1:0").await;

    let response = provider
        .complete("What is 3 + 7? Reply with just the number.")
        .await
        .unwrap();

    println!("Nova 2 Lite response: {}", response.content);
    assert!(response.content.contains("10"), "Should answer 10");
}

/// Test with Amazon Nova 2 Pro.
/// NOTE: Only available in select regions (us-east-1, us-west-2).
#[tokio::test]
#[ignore = "Requires AWS credentials + us-east-1/us-west-2 region"]
async fn test_bedrock_nova_2_pro() {
    let provider = create_provider_with_model("amazon.nova-2-pro-v1:0").await;

    let response = provider
        .complete("What is the largest planet? Reply in one word.")
        .await
        .unwrap();

    println!("Nova 2 Pro response: {}", response.content);
    assert!(
        response.content.to_lowercase().contains("jupiter"),
        "Should mention Jupiter"
    );
}

/// Test with DeepSeek R1 (reasoning model via inference profile).
/// NOTE: Only available in us-east-1, us-west-2.
#[tokio::test]
#[ignore = "Requires AWS credentials + us-east-1/us-west-2 region"]
async fn test_bedrock_deepseek_r1() {
    let provider = create_provider_with_model("deepseek.r1-v1:0").await;

    let options = CompletionOptions {
        max_tokens: Some(500),
        ..Default::default()
    };

    let messages = vec![ChatMessage::user("What is 15 * 13? Think step by step.")];

    let response = provider.chat(&messages, Some(&options)).await.unwrap();

    println!("DeepSeek R1 response: {}", response.content);
    assert!(response.content.contains("195"), "Should compute 195");
}

/// Test with Meta Llama 4 Scout (latest Llama via inference profile).
/// NOTE: Only available in us-east-1, us-west-2.
#[tokio::test]
#[ignore = "Requires AWS credentials + us-east-1/us-west-2 region"]
async fn test_bedrock_llama_4_scout() {
    let provider = create_provider_with_model("meta.llama4-scout-17b-instruct-v1:0").await;

    let options = CompletionOptions {
        max_tokens: Some(100),
        ..Default::default()
    };

    let response = provider
        .complete_with_options("What color is the sky? Reply briefly.", &options)
        .await
        .unwrap();

    println!("Llama 4 Scout response: {}", response.content);
    assert!(
        response.content.to_lowercase().contains("blue"),
        "Should mention blue"
    );
}

/// Test with Meta Llama 4 Maverick (larger Llama 4).
/// NOTE: Only available in us-east-1, us-west-2.
#[tokio::test]
#[ignore = "Requires AWS credentials + us-east-1/us-west-2 region"]
async fn test_bedrock_llama_4_maverick() {
    let provider = create_provider_with_model("meta.llama4-maverick-17b-instruct-v1:0").await;

    let options = CompletionOptions {
        max_tokens: Some(100),
        ..Default::default()
    };

    let response = provider
        .complete_with_options(
            "What is the square root of 144? Reply with just the number.",
            &options,
        )
        .await
        .unwrap();

    println!("Llama 4 Maverick response: {}", response.content);
    assert!(response.content.contains("12"), "Should answer 12");
}

/// Test with Mistral Large (via inference profile).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_mistral_large() {
    let provider = create_provider_with_model("mistral.mistral-large-2402-v1:0").await;

    let options = CompletionOptions {
        max_tokens: Some(50),
        ..Default::default()
    };

    let response = provider
        .complete_with_options("What is 8 + 9? Reply with just the number.", &options)
        .await
        .unwrap();

    println!("Mistral Large response: {}", response.content);
    assert!(response.content.contains("17"), "Should answer 17");
}

/// Test with Cohere Command R+ (latest Cohere model).
/// NOTE: Only available in us-east-1, us-west-2. Requires Marketplace subscription.
#[tokio::test]
#[ignore = "Requires AWS credentials + us-east-1/us-west-2 + Cohere subscription"]
async fn test_bedrock_cohere_command_r_plus() {
    let provider = create_provider_with_model("cohere.command-r-plus-v1:0").await;

    let options = CompletionOptions {
        max_tokens: Some(50),
        ..Default::default()
    };

    let response = provider
        .complete_with_options(
            "What is the chemical symbol for water? Reply briefly.",
            &options,
        )
        .await
        .unwrap();

    println!("Cohere Command R+ response: {}", response.content);
    assert!(
        response.content.contains("H2O") || response.content.contains("H₂O"),
        "Should mention H2O"
    );
}

/// Test with Writer Palmyra X4 (Writer model via inference profile).
/// NOTE: Only available in us-east-1.
#[tokio::test]
#[ignore = "Requires AWS credentials + us-east-1 region"]
async fn test_bedrock_writer_palmyra() {
    let provider = create_provider_with_model("writer.palmyra-x4-v1:0").await;

    let options = CompletionOptions {
        max_tokens: Some(100),
        ..Default::default()
    };

    let response = provider
        .complete_with_options("Write a one-sentence summary of photosynthesis.", &options)
        .await
        .unwrap();

    println!("Writer Palmyra response: {}", response.content);
    assert!(!response.content.is_empty());
}

/// Test inference profile resolution with explicit prefix.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_explicit_inference_profile() {
    // Use explicitly prefixed model ID (should bypass auto-resolution)
    let provider = create_provider_with_model("eu.amazon.nova-lite-v1:0").await;

    let response = provider
        .complete("Say 'ok' and nothing else.")
        .await
        .unwrap();

    println!(
        "Explicit profile response: {} (model: {})",
        response.content, response.model
    );
    assert!(!response.content.is_empty());
    assert_eq!(response.model, "eu.amazon.nova-lite-v1:0");
}

/// Test embedding dimension override.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_embedding_dimension_override() {
    let provider = create_bedrock_provider()
        .await
        .with_embedding_dimension(512);

    // Dimension should be overridden
    assert_eq!(provider.dimension(), 512);
    // But actual embedding will still be 1024 (model determines it)
    // This just tests the builder method works
}

/// Test with_embedding_model builder preserves LLM model.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_embedding_model_builder() {
    let provider = create_provider_with_model("amazon.nova-micro-v1:0")
        .await
        .with_embedding_model("cohere.embed-english-v3");

    // LLM model should still be Nova Micro
    assert_eq!(LLMProvider::model(&provider), "amazon.nova-micro-v1:0");
    // Embedding model should be Cohere
    assert_eq!(
        EmbeddingProvider::model(&provider),
        "cohere.embed-english-v3"
    );
    assert_eq!(provider.dimension(), 1024);
}

// ============================================================================
// Multi-Provider Tests (models available in eu-west-1)
// ============================================================================

/// Test with Google Gemma 3 27B (available in eu-west-1).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_google_gemma_3() {
    let provider = create_provider_with_model("google.gemma-3-27b-it").await;

    let options = CompletionOptions {
        max_tokens: Some(50),
        ..Default::default()
    };

    let response = provider
        .complete_with_options("What is 6 * 7? Reply with just the number.", &options)
        .await
        .unwrap();

    println!("Google Gemma 3 response: {}", response.content);
    assert!(response.content.contains("42"), "Should answer 42");
}

/// Test with Qwen 3 32B (available in eu-west-1).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_qwen3() {
    let provider = create_provider_with_model("qwen.qwen3-32b-v1:0").await;

    let options = CompletionOptions {
        max_tokens: Some(200),
        ..Default::default()
    };

    let response = provider
        .complete_with_options("What is the capital of Japan? Reply briefly.", &options)
        .await
        .unwrap();

    println!("Qwen 3 response: {}", response.content);
    assert!(
        response.content.to_lowercase().contains("tokyo"),
        "Should mention Tokyo"
    );
}

/// Test with Nvidia Nemotron Nano (available in eu-west-1).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_nvidia_nemotron() {
    let provider = create_provider_with_model("nvidia.nemotron-nano-12b-v2").await;

    let options = CompletionOptions {
        max_tokens: Some(50),
        ..Default::default()
    };

    let response = provider
        .complete_with_options("What is 10 + 5? Reply with just the number.", &options)
        .await
        .unwrap();

    println!("Nvidia Nemotron response: {}", response.content);
    assert!(response.content.contains("15"), "Should answer 15");
}

/// Test with MiniMax M2 (available in eu-west-1).
/// NOTE: MiniMax M2 is a reasoning model that uses "reasoningContent" blocks
/// (chain-of-thought). It needs more max_tokens to produce text output after
/// its internal reasoning phase.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_minimax_m2() {
    let provider = create_provider_with_model("minimax.minimax-m2").await;

    let options = CompletionOptions {
        max_tokens: Some(500),
        ..Default::default()
    };

    let response = provider
        .complete_with_options("What is 2 + 2? Reply with just the number.", &options)
        .await
        .unwrap();

    println!("MiniMax M2 response: '{}'", response.content);
    // MiniMax M2 uses reasoning tokens before answering, so the response may
    // include leading whitespace/newlines after the internal CoT.
    let trimmed = response.content.trim();
    assert!(trimmed.contains('4'), "Should answer 4, got: '{trimmed}'");
}

/// Test with Mistral Pixtral Large (available in eu-west-1).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_mistral_pixtral() {
    let provider = create_provider_with_model("mistral.pixtral-large-2502-v1:0").await;

    let options = CompletionOptions {
        max_tokens: Some(50),
        ..Default::default()
    };

    let response = provider
        .complete_with_options(
            "What planet is closest to the Sun? Reply in one word.",
            &options,
        )
        .await
        .unwrap();

    println!("Mistral Pixtral response: {}", response.content);
    assert!(
        response.content.to_lowercase().contains("mercury"),
        "Should mention Mercury"
    );
}

/// Test with ZAI GLM 4.7 Flash (available in eu-west-1).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_zai_glm() {
    let provider = create_provider_with_model("zai.glm-4.7-flash").await;

    let options = CompletionOptions {
        max_tokens: Some(50),
        ..Default::default()
    };

    let response = provider
        .complete_with_options("What is 3 * 9? Reply with just the number.", &options)
        .await
        .unwrap();

    println!("ZAI GLM response: {}", response.content);
    assert!(response.content.contains("27"), "Should answer 27");
}

// ============================================================================
// Additional Latest Model Tests (2025 Models)
// ============================================================================

/// Test with MiniMax M2.1 (latest MiniMax, available in eu-west-1).
/// This is a reasoning model that uses chain-of-thought before answering.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_minimax_m2_1() {
    let provider = create_provider_with_model("minimax.minimax-m2.1").await;

    let options = CompletionOptions {
        max_tokens: Some(500),
        ..Default::default()
    };

    let response = provider
        .complete_with_options("What is 5 + 3? Reply with just the number.", &options)
        .await
        .unwrap();

    println!("MiniMax M2.1 response: '{}'", response.content);
    let trimmed = response.content.trim();
    assert!(trimmed.contains('8'), "Should answer 8, got: '{trimmed}'");
}

/// Test with Mistral Magistral Small 2509 (available in eu-west-1).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_mistral_magistral_small() {
    let provider = create_provider_with_model("mistral.magistral-small-2509").await;

    let options = CompletionOptions {
        max_tokens: Some(200),
        ..Default::default()
    };

    let response = provider
        .complete_with_options(
            "What is the chemical symbol for gold? Reply with just the symbol.",
            &options,
        )
        .await
        .unwrap();

    println!("Mistral Magistral Small response: '{}'", response.content);
    assert!(response.content.contains("Au"), "Should answer Au");
}

/// Test with Mistral Devstral 2 123B (code-focused, available in eu-west-1).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_mistral_devstral() {
    let provider = create_provider_with_model("mistral.devstral-2-123b").await;

    let options = CompletionOptions {
        max_tokens: Some(200),
        ..Default::default()
    };

    let response = provider
        .complete_with_options(
            "Write a Python function that returns the factorial of n. Reply with only the code.",
            &options,
        )
        .await
        .unwrap();

    println!("Mistral Devstral response: '{}'", response.content);
    assert!(
        response.content.contains("factorial") || response.content.contains("def "),
        "Should contain Python code"
    );
}

/// Test with Mistral Ministral 3 8B (small, fast, available in eu-west-1).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_mistral_ministral_8b() {
    let provider = create_provider_with_model("mistral.ministral-3-8b-instruct").await;

    let options = CompletionOptions {
        max_tokens: Some(50),
        ..Default::default()
    };

    let response = provider
        .complete_with_options("What is 4 + 6? Reply with just the number.", &options)
        .await
        .unwrap();

    println!("Ministral 8B response: '{}'", response.content);
    assert!(response.content.contains("10"), "Should answer 10");
}

/// Test with Google Gemma 3 4B IT (smallest Gemma, available in eu-west-1).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_google_gemma_3_4b() {
    let provider = create_provider_with_model("google.gemma-3-4b-it").await;

    let options = CompletionOptions {
        max_tokens: Some(50),
        ..Default::default()
    };

    let response = provider
        .complete_with_options("What is 9 + 1? Reply with just the number.", &options)
        .await
        .unwrap();

    println!("Gemma 3 4B response: '{}'", response.content);
    assert!(response.content.contains("10"), "Should answer 10");
}

/// Test with NVIDIA Nemotron Nano 3 30B (larger Nemotron, available in eu-west-1).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_nvidia_nemotron_30b() {
    let provider = create_provider_with_model("nvidia.nemotron-nano-3-30b").await;

    let options = CompletionOptions {
        max_tokens: Some(500),
        ..Default::default()
    };

    let response = provider
        .complete_with_options("What is 7 + 8? Reply with just the number.", &options)
        .await
        .unwrap();

    println!("Nemotron Nano 30B response: '{}'", response.content);
    let trimmed = response.content.trim();
    assert!(trimmed.contains("15"), "Should answer 15, got: '{trimmed}'");
}

/// Test with Qwen3 Coder 30B (code-focused Qwen, available in eu-west-1).
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_qwen3_coder() {
    let provider = create_provider_with_model("qwen.qwen3-coder-30b-a3b-v1:0").await;

    let options = CompletionOptions {
        max_tokens: Some(500),
        ..Default::default()
    };

    let response = provider
        .complete_with_options(
            "What is the capital of France? Reply in one word.",
            &options,
        )
        .await
        .unwrap();

    println!("Qwen3 Coder response: '{}'", response.content);
    let trimmed = response.content.trim();
    assert!(
        trimmed.to_lowercase().contains("paris"),
        "Should mention Paris, got: '{trimmed}'"
    );
}

/// Test with OpenAI GPT OSS 120B (available in eu-west-1).
/// NOTE: OpenAI open-source model hosted on Bedrock.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_openai_gpt_oss() {
    let provider = create_provider_with_model("openai.gpt-oss-120b-1:0").await;

    let options = CompletionOptions {
        max_tokens: Some(500),
        ..Default::default()
    };

    let response = provider
        .complete_with_options("What is 6 * 3? Reply with just the number.", &options)
        .await
        .unwrap();

    println!("OpenAI GPT OSS response: '{}'", response.content);
    let trimmed = response.content.trim();
    assert!(trimmed.contains("18"), "Should answer 18, got: '{trimmed}'");
}

/// Test with DeepSeek V3.2 (latest DeepSeek, available in us-east-1 only).
#[tokio::test]
#[ignore = "Requires AWS credentials + us-east-1/us-west-2 region"]
async fn test_bedrock_deepseek_v3_2() {
    let provider = create_provider_with_model("deepseek.v3.2").await;

    let options = CompletionOptions {
        max_tokens: Some(200),
        ..Default::default()
    };

    let messages = vec![ChatMessage::user(
        "What is 12 * 11? Reply with just the number.",
    )];

    let response = provider.chat(&messages, Some(&options)).await.unwrap();

    println!("DeepSeek V3.2 response: '{}'", response.content);
    assert!(response.content.contains("132"), "Should answer 132");
}

/// Test with Mistral Magistral Small — tool calling support.
/// NOTE: Magistral Small may not fully support tool calling via Converse API.
/// This test validates the request doesn't error even if tool_choice is set.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_mistral_magistral_tool_calling() {
    let provider = create_provider_with_model("mistral.magistral-small-2509").await;

    let tools = vec![ToolDefinition::function(
        "get_time",
        "Get the current time in a timezone",
        serde_json::json!({
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone, e.g., 'Europe/Paris'"
                }
            },
            "required": ["timezone"]
        }),
    )];

    let messages = vec![ChatMessage::user("What time is it in Tokyo?")];

    let response = provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), None)
        .await
        .unwrap();

    println!("Magistral tool response: {:?}", response.tool_calls);
    println!("Magistral tool content: '{}'", response.content);
    // Magistral may or may not call the tool — either way the request should succeed
    assert!(
        response.has_tool_calls() || !response.content.is_empty(),
        "Should either call a tool or provide a text response"
    );
}
