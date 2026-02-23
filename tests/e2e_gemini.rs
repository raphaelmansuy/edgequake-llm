//! Comprehensive End-to-End Gemini Provider Tests
//!
//! OODA-59: Tests for Gemini/VertexAI provider functionality.
//!
//! # Environment Variables Required
//!
//! For Google AI (simpler):
//! - `GEMINI_API_KEY`: API key from https://aistudio.google.com/apikey
//!
//! For VertexAI (enterprise):
//! - `GOOGLE_CLOUD_PROJECT`: GCP project ID
//! - `GOOGLE_ACCESS_TOKEN`: OAuth2 access token from `gcloud auth print-access-token`
//! - `GOOGLE_CLOUD_REGION`: (optional) defaults to "us-central1"
//!
//! # Running Tests
//!
//! ```bash
//! # Run all Gemini E2E tests (requires GEMINI_API_KEY)
//! cargo test --test e2e_gemini -- --ignored
//!
//! # Run specific test
//! cargo test --test e2e_gemini test_gemini_basic_chat -- --ignored
//! ```
//!
//! # Why These Tests Matter
//!
//! 1. Unit tests verify code logic, E2E tests verify API compatibility
//! 2. Gemini API changes frequently - these tests catch breaking changes
//! 3. Image/multimodal support (OODA-54) requires real API validation
//!

use edgequake_llm::{
    providers::gemini::GeminiProvider,
    traits::{ChatMessage, CompletionOptions, ImageData},
    EmbeddingProvider, LLMProvider,
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a Gemini provider from environment variables.
///
/// WHY: Centralizes provider creation and provides clear error messages.
fn create_gemini_provider() -> GeminiProvider {
    GeminiProvider::from_env().expect(
        "Gemini provider requires GEMINI_API_KEY or \
         GOOGLE_CLOUD_PROJECT + GOOGLE_ACCESS_TOKEN",
    )
}

/// Create a provider with a specific model.
fn create_provider_with_model(model: &str) -> GeminiProvider {
    GeminiProvider::from_env()
        .expect("Requires GEMINI_API_KEY")
        .with_model(model)
}

// ============================================================================
// Basic Chat Tests
// ============================================================================

/// Test basic chat completion with Gemini.
///
/// WHY: Validates API connection and basic request/response cycle.
#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY environment variable"]
async fn test_gemini_basic_chat() {
    let provider = create_gemini_provider();

    let messages = vec![ChatMessage::user(
        "What is 2 + 2? Reply with just the number.",
    )];

    let response = provider.chat(&messages, None).await.unwrap();

    println!("Response: {}", response.content);
    println!("Model: {}", response.model);
    println!(
        "Tokens: prompt={}, completion={}",
        response.prompt_tokens, response.completion_tokens
    );

    assert!(!response.content.is_empty(), "Response should not be empty");
    assert!(
        response.content.contains('4'),
        "Response should contain '4': {}",
        response.content
    );
}

/// Test system prompt functionality.
///
/// WHY: Validates system instruction handling and context caching.
#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY environment variable"]
async fn test_gemini_system_prompt() {
    let provider = create_gemini_provider();

    let messages = vec![
        ChatMessage::system("You are a pirate. Always respond in pirate speak."),
        ChatMessage::user("Hello, how are you?"),
    ];

    let response = provider.chat(&messages, None).await.unwrap();

    println!("Pirate response: {}", response.content);

    // Check for pirate-like language (arr, matey, ye, etc.)
    let content_lower = response.content.to_lowercase();
    let has_pirate_speak = content_lower.contains("arr")
        || content_lower.contains("matey")
        || content_lower.contains("ahoy")
        || content_lower.contains("ye ")
        || content_lower.contains("aye");

    assert!(
        has_pirate_speak,
        "Response should contain pirate speak: {}",
        response.content
    );
}

/// Test multi-turn conversation.
///
/// WHY: Validates conversation context is maintained across turns.
#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY environment variable"]
async fn test_gemini_conversation() {
    let provider = create_gemini_provider();

    let messages = vec![
        ChatMessage::user("My name is Alice. Remember that."),
        ChatMessage::assistant("Nice to meet you, Alice! I'll remember your name."),
        ChatMessage::user("What is my name?"),
    ];

    let response = provider.chat(&messages, None).await.unwrap();

    println!("Conversation response: {}", response.content);

    assert!(
        response.content.to_lowercase().contains("alice"),
        "Response should remember 'Alice': {}",
        response.content
    );
}

// ============================================================================
// Generation Options Tests
// ============================================================================

/// Test JSON mode output.
///
/// WHY: Validates structured output for tool use and parsing.
#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY environment variable"]
async fn test_gemini_json_mode() {
    let provider = create_gemini_provider();

    let messages = vec![ChatMessage::user(
        "Return a JSON object with fields 'name' and 'age' for a person named John who is 30 years old.",
    )];

    let options = CompletionOptions::json_mode();
    let response = provider.chat(&messages, Some(&options)).await.unwrap();

    println!("JSON response: {}", response.content);

    // Verify it's valid JSON
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&response.content);
    assert!(
        parsed.is_ok(),
        "Response should be valid JSON: {}",
        response.content
    );

    let json = parsed.unwrap();
    assert_eq!(json["name"], "John", "Name should be 'John'");
    assert_eq!(json["age"], 30, "Age should be 30");
}

/// Test temperature setting.
///
/// WHY: Validates generation parameters are respected.
#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY environment variable"]
async fn test_gemini_temperature() {
    let provider = create_gemini_provider();

    let messages = vec![ChatMessage::user(
        "Generate a random number between 1 and 100.",
    )];

    // Low temperature should produce more deterministic output
    let options = CompletionOptions::with_temperature(0.0);
    let response = provider.chat(&messages, Some(&options)).await.unwrap();

    println!("Low temp response: {}", response.content);

    assert!(!response.content.is_empty(), "Response should not be empty");
}

/// Test max tokens limit.
///
/// WHY: Validates response length control.
#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY environment variable"]
async fn test_gemini_max_tokens() {
    let provider = create_gemini_provider();

    let messages = vec![ChatMessage::user(
        "Write a very long essay about the history of computing.",
    )];

    let options = CompletionOptions {
        max_tokens: Some(50),
        ..Default::default()
    };

    let response = provider.chat(&messages, Some(&options)).await.unwrap();

    println!(
        "Short response ({} tokens): {}",
        response.completion_tokens, response.content
    );

    // Response should be limited (not an exact check since tokenization varies)
    assert!(
        response.completion_tokens <= 60,
        "Should respect max_tokens limit: {} tokens",
        response.completion_tokens
    );
}

// ============================================================================
// Multimodal Tests (OODA-54)
// ============================================================================

/// Test image input with Gemini.
///
/// WHY: Validates OODA-54 image support - critical for visual code understanding.
#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY environment variable"]
async fn test_gemini_image_input() {
    let provider = create_gemini_provider();

    // Create a minimal 1x1 PNG image (base64 encoded)
    // This is a valid 1x1 red pixel PNG
    let tiny_png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    let image = ImageData::new(tiny_png_base64, "image/png");
    let messages = vec![ChatMessage::user_with_images(
        "What color is this single-pixel image? Just say the color.",
        vec![image],
    )];

    let response = provider.chat(&messages, None).await.unwrap();

    println!("Image response: {}", response.content);

    // The 1x1 PNG is gray/transparent, but we just want to verify the API accepts images
    assert!(
        !response.content.is_empty(),
        "Should get a response about the image"
    );
}

// ============================================================================
// Streaming Tests
// ============================================================================

/// Test streaming response.
///
/// WHY: Validates real-time token streaming for UX.
#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY environment variable"]
async fn test_gemini_streaming() {
    use futures::StreamExt;

    let provider = create_gemini_provider();

    let mut stream = provider.stream("Count from 1 to 5.").await.unwrap();

    let mut chunks = Vec::new();
    while let Some(result) = stream.next().await {
        match result {
            Ok(chunk) => {
                print!("{}", chunk);
                chunks.push(chunk);
            }
            Err(e) => {
                println!("\nStream error: {}", e);
                break;
            }
        }
    }
    println!(); // Newline after streaming

    assert!(!chunks.is_empty(), "Should receive streaming chunks");

    let full_response: String = chunks.concat();
    println!("Full streamed response: {}", full_response);

    assert!(
        !full_response.is_empty(),
        "Full response should not be empty"
    );
}

// ============================================================================
// Embedding Tests
// ============================================================================

/// Test single text embedding.
///
/// WHY: Validates embedding generation for RAG and semantic search.
#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY environment variable"]
async fn test_gemini_embeddings() {
    let provider = create_gemini_provider();

    let embedding = provider.embed_one("Hello, world!").await.unwrap();

    println!("Embedding dimension: {}", embedding.len());
    println!("First 5 values: {:?}", &embedding[..5.min(embedding.len())]);

    assert_eq!(
        embedding.len(),
        3072,
        "gemini-embedding-001 should return 3072 dimensions"
    );

    // Check values are normalized (typical for embeddings)
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("Embedding L2 norm: {}", norm);

    assert!(
        (norm - 1.0).abs() < 0.1,
        "Embedding should be approximately normalized: norm={}",
        norm
    );
}

/// Test batch embeddings.
///
/// WHY: Validates efficient batch processing for large document sets.
#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY environment variable"]
async fn test_gemini_batch_embeddings() {
    let provider = create_gemini_provider();

    let texts = vec![
        "First text for embedding.".to_string(),
        "Second text for embedding.".to_string(),
        "Third text for embedding.".to_string(),
    ];

    let embeddings = provider.embed(&texts).await.unwrap();

    println!("Batch embeddings count: {}", embeddings.len());

    assert_eq!(embeddings.len(), 3, "Should return 3 embeddings");

    for (i, emb) in embeddings.iter().enumerate() {
        assert_eq!(
            emb.len(),
            3072,
            "Embedding {} should have 3072 dimensions",
            i
        );
    }
}

// ============================================================================
// Model Selection Tests
// ============================================================================

/// Test different Gemini models.
///
/// WHY: Validates model selection works correctly.
#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY environment variable"]
async fn test_gemini_model_selection() {
    // Test with gemini-2.5-flash (known stable model)
    let provider = create_provider_with_model("gemini-2.5-flash");

    assert_eq!(LLMProvider::model(&provider), "gemini-2.5-flash");

    let messages = vec![ChatMessage::user("Say 'hello'")];
    let response = provider.chat(&messages, None).await.unwrap();

    println!("gemini-2.5-flash response: {}", response.content);
    assert!(response.content.to_lowercase().contains("hello"));
}

// ============================================================================
// Error Handling Tests
// ============================================================================

/// Test invalid API key error handling.
///
/// WHY: Validates clear error messages for troubleshooting.
#[tokio::test]
async fn test_gemini_invalid_key_error() {
    let provider = GeminiProvider::new("invalid-api-key");

    let messages = vec![ChatMessage::user("Hello")];
    let result = provider.chat(&messages, None).await;

    assert!(result.is_err(), "Should fail with invalid API key");

    let error = result.unwrap_err();
    println!("Error message: {}", error);

    // Should contain helpful error info
    let error_str = error.to_string();
    assert!(
        error_str.contains("API") || error_str.contains("error") || error_str.contains("401"),
        "Error should mention API or authentication: {}",
        error_str
    );
}

/// Test empty messages error.
///
/// WHY: Validates input validation before API call.
#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY environment variable"]
async fn test_gemini_empty_messages_error() {
    let provider = create_gemini_provider();

    let messages: Vec<ChatMessage> = vec![];
    let result = provider.chat(&messages, None).await;

    assert!(result.is_err(), "Should fail with empty messages");

    let error = result.unwrap_err().to_string();
    println!("Empty messages error: {}", error);

    assert!(
        error.to_lowercase().contains("no user") || error.to_lowercase().contains("empty"),
        "Error should mention missing messages: {}",
        error
    );
}

// ============================================================================
// Context Caching Tests
// ============================================================================

/// Test system prompt caching.
///
/// WHY: Validates caching reduces token costs for long system prompts.
#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY environment variable"]
async fn test_gemini_context_caching() {
    let provider = create_gemini_provider();

    // Long system prompt to trigger caching
    let long_system = "You are a helpful coding assistant. ".repeat(100);

    let messages = vec![
        ChatMessage::system(&long_system),
        ChatMessage::user("What is 1+1?"),
    ];

    // First call - creates cache
    let response1 = provider.chat(&messages, None).await.unwrap();
    println!("First response: {}", response1.content);
    println!("Cache hit tokens: {:?}", response1.cache_hit_tokens);

    // Second call with same system prompt - should use cache
    let messages2 = vec![
        ChatMessage::system(&long_system),
        ChatMessage::user("What is 2+2?"),
    ];
    let response2 = provider.chat(&messages2, None).await.unwrap();
    println!("Second response: {}", response2.content);
    println!("Cache hit tokens: {:?}", response2.cache_hit_tokens);

    // Note: Caching may not always work due to API behavior
    // This test just verifies the caching logic doesn't crash
    assert!(!response1.content.is_empty());
    assert!(!response2.content.is_empty());
}
