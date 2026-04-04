//! E2E tests for `OpenAICompatibleProvider` pointed at the real OpenAI API.
//!
//! These tests exercise the same code paths used for Ollama / LM Studio but
//! against `api.openai.com`, ensuring the provider works with the canonical
//! reference implementation.
//!
//! # Prerequisites
//!
//! - `OPENAI_API_KEY` environment variable set
//!
//! # Running
//!
//! ```shell
//! OPENAI_API_KEY=sk-... cargo test --test e2e_openai_api -- --ignored --nocapture
//! ```

use edgequake_llm::model_config::{
    ModelCapabilities, ModelCard, ModelType, ProviderConfig, ProviderType,
};
use edgequake_llm::providers::openai_compatible::OpenAICompatibleProvider;
use edgequake_llm::traits::{ChatMessage, CompletionOptions, LLMProvider, ToolDefinition};
use futures::StreamExt;

// ── Provider factory ──────────────────────────────────────────────────────────

fn create_openai_config(model: &str) -> ProviderConfig {
    ProviderConfig {
        name: "openai".to_string(),
        display_name: "OpenAI".to_string(),
        provider_type: ProviderType::OpenAICompatible,
        api_key_env: Some("OPENAI_API_KEY".to_string()),
        base_url: Some("https://api.openai.com/v1".to_string()),
        default_llm_model: Some(model.to_string()),
        default_embedding_model: Some("text-embedding-3-small".to_string()),
        timeout_seconds: 60,
        models: vec![
            ModelCard {
                name: model.to_string(),
                display_name: model.to_string(),
                model_type: ModelType::Llm,
                capabilities: ModelCapabilities {
                    context_length: 128_000,
                    max_output_tokens: 4_096,
                    supports_function_calling: true,
                    supports_streaming: true,
                    supports_json_mode: true,
                    ..Default::default()
                },
                ..Default::default()
            },
            ModelCard {
                name: "text-embedding-3-small".to_string(),
                display_name: "text-embedding-3-small".to_string(),
                model_type: ModelType::Embedding,
                capabilities: ModelCapabilities {
                    context_length: 8_191,
                    embedding_dimension: 1_536,
                    max_embedding_tokens: 8_191,
                    ..Default::default()
                },
                ..Default::default()
            },
        ],
        ..Default::default()
    }
}

fn make_provider(model: &str) -> OpenAICompatibleProvider {
    OpenAICompatibleProvider::from_config(create_openai_config(model))
        .expect("Failed to create OpenAI provider — is OPENAI_API_KEY set?")
}

// ── Basic chat ────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY environment variable"]
async fn test_openai_simple_chat() {
    let provider = make_provider("gpt-4o-mini");

    assert_eq!(LLMProvider::name(&provider), "openai");
    assert_eq!(LLMProvider::model(&provider), "gpt-4o-mini");

    let messages = vec![ChatMessage::user(
        "What is 2 + 2? Reply with just the number.",
    )];
    let response = provider.chat(&messages, None).await.expect("chat failed");

    eprintln!("Response: {}", response.content);
    assert!(
        response.content.contains('4'),
        "Expected '4' in response, got: {}",
        response.content
    );
    assert_eq!(response.model, "gpt-4o-mini");
    assert!(response.prompt_tokens > 0, "prompt tokens must be > 0");
    assert!(
        response.completion_tokens > 0,
        "completion tokens must be > 0"
    );
}

// ── System prompt ─────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY environment variable"]
async fn test_openai_chat_with_system_prompt() {
    let provider = make_provider("gpt-4o-mini");

    let messages = vec![
        ChatMessage::system("You are a helpful assistant that replies in exactly one word."),
        ChatMessage::user("What color is the sky?"),
    ];
    let response = provider
        .chat(&messages, None)
        .await
        .expect("chat with system prompt failed");

    eprintln!("Response: {}", response.content);
    assert!(!response.content.is_empty());
}

// ── Streaming ─────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY environment variable"]
async fn test_openai_streaming() {
    let provider = make_provider("gpt-4o-mini");

    let mut stream = provider
        .stream("Count from 1 to 5. Reply with just the numbers, one per line.")
        .await
        .expect("stream failed");

    let mut full_response = String::new();
    let mut chunk_count = 0usize;

    while let Some(chunk) = stream.next().await {
        let text = chunk.expect("stream chunk error");
        full_response.push_str(&text);
        chunk_count += 1;
    }

    eprintln!("Full response ({} chunks): {}", chunk_count, full_response);
    assert!(chunk_count > 0, "must receive at least one chunk");
    assert!(
        !full_response.is_empty(),
        "streamed response must not be empty"
    );
    // The response should contain the digits 1-5.
    for digit in ['1', '2', '3', '4', '5'] {
        assert!(
            full_response.contains(digit),
            "expected digit {} in streamed response: {}",
            digit,
            full_response
        );
    }
}

// ── CompletionOptions ─────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY environment variable"]
async fn test_openai_chat_with_options() {
    let provider = make_provider("gpt-4o-mini");

    let options = CompletionOptions {
        temperature: Some(0.0),
        max_tokens: Some(20),
        ..Default::default()
    };

    let messages = vec![ChatMessage::user("What is 1 + 1? One word only.")];
    let response = provider
        .chat(&messages, Some(&options))
        .await
        .expect("chat with options failed");

    eprintln!("Response: {}", response.content);
    assert!(!response.content.is_empty());
    // With temperature=0 and a simple arithmetic question, expect "2" or "Two".
    let lower = response.content.to_lowercase();
    assert!(
        lower.contains('2') || lower.contains("two"),
        "expected '2' or 'two', got: {}",
        response.content
    );
}

// ── Function / tool calling ───────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY environment variable"]
async fn test_openai_tool_calling() {
    let provider = make_provider("gpt-4o-mini");

    let tools = vec![ToolDefinition::function(
        "get_weather",
        "Get the current weather for a location.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country, e.g. 'London, UK'"
                }
            },
            "required": ["location"]
        }),
    )];

    let messages = vec![ChatMessage::user(
        "What is the weather like in Paris, France today?",
    )];

    let response = provider
        .chat_with_tools(&messages, &tools, None, None)
        .await
        .expect("chat_with_tools failed");

    eprintln!("Response: {:?}", response);

    // The model should call get_weather or provide a text response.
    // Either is acceptable — the point is the request is well-formed.
    assert!(!response.content.is_empty() || !response.tool_calls.is_empty());

    if !response.tool_calls.is_empty() {
        let call = &response.tool_calls[0];
        assert_eq!(call.name(), "get_weather");
        eprintln!("Tool call args: {}", call.arguments());
        // Arguments should mention Paris.
        assert!(
            call.arguments().to_lowercase().contains("paris"),
            "Expected 'paris' in tool call arguments: {}",
            call.arguments()
        );
    }
}

// ── Tool choice: none (no tools sent) ────────────────────────────────────────

/// Fix #3 regression: When tools vec is empty, `tool_choice` must NOT be sent.
/// Previously sending `tool_choice` without `tools` caused 400 errors from OpenAI.
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY environment variable"]
async fn test_openai_empty_tools_no_tool_choice_400_regression() {
    let provider = make_provider("gpt-4o-mini");

    // Pass empty tools slice — must not elicit a 400 from OpenAI.
    let messages = vec![ChatMessage::user("Say hello.")];
    let response = provider
        .chat_with_tools(&messages, &[], None, None)
        .await
        .expect("chat_with_tools with empty tools must not return 400");

    assert!(!response.content.is_empty());
}

// ── Embeddings ────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY environment variable"]
async fn test_openai_embeddings() {
    use edgequake_llm::model_config::{
        ModelCapabilities, ModelCard, ModelType, ProviderConfig, ProviderType,
    };
    use edgequake_llm::providers::openai_compatible::OpenAICompatibleProvider;
    use edgequake_llm::traits::EmbeddingProvider;

    let config = ProviderConfig {
        name: "openai".to_string(),
        display_name: "OpenAI".to_string(),
        provider_type: ProviderType::OpenAICompatible,
        api_key_env: Some("OPENAI_API_KEY".to_string()),
        base_url: Some("https://api.openai.com/v1".to_string()),
        default_embedding_model: Some("text-embedding-3-small".to_string()),
        timeout_seconds: 60,
        models: vec![ModelCard {
            name: "text-embedding-3-small".to_string(),
            display_name: "text-embedding-3-small".to_string(),
            model_type: ModelType::Embedding,
            capabilities: ModelCapabilities {
                context_length: 8_191,
                embedding_dimension: 1_536,
                max_embedding_tokens: 8_191,
                ..Default::default()
            },
            ..Default::default()
        }],
        ..Default::default()
    };

    let provider =
        OpenAICompatibleProvider::from_config(config).expect("Failed to create embedding provider");

    assert_eq!(EmbeddingProvider::name(&provider), "openai");
    assert_eq!(
        EmbeddingProvider::model(&provider),
        "text-embedding-3-small"
    );
    assert_eq!(provider.dimension(), 1_536);

    let texts = vec![
        "Hello, world!".to_string(),
        "Rust is a systems programming language.".to_string(),
    ];

    let embeddings = provider.embed(&texts).await.expect("embed failed");

    assert_eq!(embeddings.len(), 2, "must return one embedding per input");

    for (i, emb) in embeddings.iter().enumerate() {
        assert_eq!(
            emb.len(),
            1_536,
            "embedding {} must have dimension 1536, got {}",
            i,
            emb.len()
        );
        // The embedding must not be all zeros.
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.0, "embedding {} must not be zero vector", i);
    }

    // Cosine similarity: identical input must have similarity ≈ 1.0.
    let same_texts = vec!["test".to_string(), "test".to_string()];
    let same_embeddings = provider
        .embed(&same_texts)
        .await
        .expect("embed same failed");
    let cos_sim = cosine_similarity(&same_embeddings[0], &same_embeddings[1]);
    assert!(
        cos_sim > 0.99,
        "identical inputs should have cosine similarity > 0.99, got {}",
        cos_sim
    );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}
