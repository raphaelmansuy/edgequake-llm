//! Comprehensive End-to-End AWS Bedrock Provider Tests
//!
//! Tests for AWS Bedrock provider via the Converse API.
//!
//! # Environment Variables Required
//!
//! - AWS credentials via standard credential chain (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY,
//!   AWS_PROFILE, IAM roles, etc.)
//! - `AWS_REGION` or `AWS_DEFAULT_REGION`: AWS region (default: us-east-1)
//! - `AWS_BEDROCK_MODEL`: Model ID (default: anthropic.claude-3-5-sonnet-20241022-v2:0)
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
    traits::{ChatMessage, CompletionOptions, LLMProvider, ToolChoice, ToolDefinition},
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

    assert_eq!(provider.name(), "bedrock");
    assert!(!provider.model().is_empty(), "Model should not be empty");
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
    let provider = create_provider_with_model("anthropic.claude-3-haiku-20240307-v1:0").await;

    assert_eq!(provider.model(), "anthropic.claude-3-haiku-20240307-v1:0");
    assert_eq!(provider.max_context_length(), 200_000);

    // Quick smoke test with Haiku (cheaper)
    let response = provider
        .complete("Say 'hello' and nothing else.")
        .await
        .unwrap();

    println!("Haiku response: {}", response.content);
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
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_factory_create() {
    use edgequake_llm::factory::{ProviderFactory, ProviderType};

    let (llm, embedding) = ProviderFactory::create(ProviderType::Bedrock).unwrap();
    assert_eq!(llm.name(), "bedrock");
    // Bedrock doesn't support embeddings natively, falls back to mock or OpenAI
    assert!(!embedding.name().is_empty());
}

/// Test factory creation with specific model.
#[tokio::test]
#[ignore = "Requires AWS credentials with Bedrock access"]
async fn test_bedrock_factory_create_with_model() {
    use edgequake_llm::factory::{ProviderFactory, ProviderType};

    let (llm, _) = ProviderFactory::create_with_model(
        ProviderType::Bedrock,
        Some("anthropic.claude-3-haiku-20240307-v1:0"),
    )
    .unwrap();
    assert_eq!(llm.name(), "bedrock");
    assert_eq!(llm.model(), "anthropic.claude-3-haiku-20240307-v1:0");
}
