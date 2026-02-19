//! End-to-end tests for the Mistral AI provider.
//!
//! These tests require a valid MISTRAL_API_KEY environment variable.
//!
//! # Running the tests
//!
//! ```bash
//! export MISTRAL_API_KEY=your-api-key
//! cargo test -p edgequake-llm --test e2e_mistral
//! cargo test -p edgequake-llm --test e2e_mistral test_mistral_basic_chat
//! ```
//!
//! # Test coverage
//!
//! - Basic chat completion
//! - Simple `complete()` helper
//! - JSON mode
//! - Streaming
//! - Tool / function calling
//! - Embeddings (native mistral-embed)
//! - Model listing (GET /v1/models)
//! - Provider factory auto-detection

use edgequake_llm::traits::{ChatMessage, EmbeddingProvider, LLMProvider};
use edgequake_llm::MistralProvider;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn has_mistral_key() -> bool {
    std::env::var("MISTRAL_API_KEY")
        .map(|k| !k.is_empty())
        .unwrap_or(false)
}

fn create_provider() -> MistralProvider {
    MistralProvider::from_env().expect("MISTRAL_API_KEY must be set")
}

// ---------------------------------------------------------------------------
// Basic Chat
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_mistral_basic_chat() {
    if !has_mistral_key() {
        eprintln!("Skipping test_mistral_basic_chat: MISTRAL_API_KEY not set");
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
            println!("Response: {}", resp.content);
            println!("Model: {}", resp.model);
            println!(
                "Tokens: {} in, {} out",
                resp.prompt_tokens, resp.completion_tokens
            );
            assert!(
                resp.content.contains("4"),
                "Expected '4' in response: {}",
                resp.content
            );
        }
        Err(e) => panic!("Chat failed: {:?}", e),
    }
}

#[tokio::test]
async fn test_mistral_simple_complete() {
    if !has_mistral_key() {
        eprintln!("Skipping test_mistral_simple_complete: MISTRAL_API_KEY not set");
        return;
    }

    let provider = create_provider();
    let response = provider
        .complete("Say 'hello world' and nothing else.")
        .await;

    match response {
        Ok(resp) => {
            println!("Response: {}", resp.content);
            assert!(
                resp.content.to_lowercase().contains("hello"),
                "Expected 'hello' in response: {}",
                resp.content
            );
        }
        Err(e) => panic!("Complete failed: {:?}", e),
    }
}

// ---------------------------------------------------------------------------
// JSON Mode
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_mistral_json_mode() {
    if !has_mistral_key() {
        eprintln!("Skipping test_mistral_json_mode: MISTRAL_API_KEY not set");
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
            let json_result: Result<serde_json::Value, _> =
                serde_json::from_str(resp.content.trim());
            match json_result {
                Ok(json) => {
                    assert!(json.get("name").is_some(), "Missing 'name' field");
                    assert!(json.get("age").is_some(), "Missing 'age' field");
                    assert!(json.get("active").is_some(), "Missing 'active' field");
                }
                Err(e) => panic!("Invalid JSON response: {} — Error: {}", resp.content, e),
            }
        }
        Err(e) => panic!("JSON mode failed: {:?}", e),
    }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_mistral_streaming() {
    if !has_mistral_key() {
        eprintln!("Skipping test_mistral_streaming: MISTRAL_API_KEY not set");
        return;
    }

    use futures::StreamExt;

    let provider = create_provider();
    let result = provider
        .stream("Count from 1 to 5, separated by commas.")
        .await;

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
                    Err(e) => panic!("Stream chunk error: {:?}", e),
                }
            }

            println!(
                "Streamed response ({} chunks): {}",
                chunk_count, full_response
            );
            assert!(
                chunk_count > 1,
                "Expected multiple chunks, got {}",
                chunk_count
            );
            assert!(
                !full_response.is_empty(),
                "Expected non-empty streamed response"
            );
        }
        Err(e) => panic!("Stream failed: {:?}", e),
    }
}

// ---------------------------------------------------------------------------
// Tool / Function Calling
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_mistral_tool_calling() {
    if !has_mistral_key() {
        eprintln!("Skipping test_mistral_tool_calling: MISTRAL_API_KEY not set");
        return;
    }

    use edgequake_llm::traits::{ToolChoice, ToolDefinition};

    let provider = create_provider();

    let tools = vec![ToolDefinition::function(
        "get_weather",
        "Get the current weather for a city",
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

    let messages = vec![
        ChatMessage::system("You are a helpful assistant. Use tools when appropriate."),
        ChatMessage::user("What is the weather in Paris?"),
    ];

    let response = provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), None)
        .await;

    match response {
        Ok(resp) => {
            println!("Response content: {}", resp.content);
            println!("Tool calls: {:?}", resp.tool_calls);
            // Model should call the tool or respond directly
            assert!(
                !resp.content.is_empty() || !resp.tool_calls.is_empty(),
                "Expected either content or tool calls"
            );
        }
        Err(e) => panic!("Tool calling failed: {:?}", e),
    }
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_mistral_embeddings_single() {
    if !has_mistral_key() {
        eprintln!("Skipping test_mistral_embeddings_single: MISTRAL_API_KEY not set");
        return;
    }

    let provider = create_provider();
    let texts = vec!["The quick brown fox jumps over the lazy dog.".to_string()];
    let result = provider.embed(&texts).await;

    match result {
        Ok(embeddings) => {
            println!(
                "Got {} embedding(s), dimension: {}",
                embeddings.len(),
                embeddings[0].len()
            );
            assert_eq!(embeddings.len(), 1);
            // mistral-embed produces 1024-dimensional embeddings
            assert_eq!(
                embeddings[0].len(),
                1024,
                "Expected 1024-dim, got {}",
                embeddings[0].len()
            );
        }
        Err(e) => panic!("Embedding failed: {:?}", e),
    }
}

#[tokio::test]
async fn test_mistral_embeddings_batch() {
    if !has_mistral_key() {
        eprintln!("Skipping test_mistral_embeddings_batch: MISTRAL_API_KEY not set");
        return;
    }

    let provider = create_provider();
    let texts = vec![
        "Rust is a systems programming language.".to_string(),
        "Python is great for data science.".to_string(),
        "Go is simple and efficient.".to_string(),
    ];
    let result = provider.embed(&texts).await;

    match result {
        Ok(embeddings) => {
            println!("Got {} embeddings", embeddings.len());
            assert_eq!(embeddings.len(), 3);
            for emb in &embeddings {
                assert_eq!(emb.len(), 1024, "Expected 1024-dim, got {}", emb.len());
            }
        }
        Err(e) => panic!("Batch embedding failed: {:?}", e),
    }
}

#[tokio::test]
async fn test_mistral_embedding_dimension() {
    if !has_mistral_key() {
        eprintln!("Skipping test_mistral_embedding_dimension: MISTRAL_API_KEY not set");
        return;
    }

    let provider = create_provider();
    assert_eq!(provider.dimension(), 1024, "Expected dimension 1024");
    assert_eq!(EmbeddingProvider::model(&provider), "mistral-embed");
}

// ---------------------------------------------------------------------------
// Model Listing
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_mistral_list_models() {
    if !has_mistral_key() {
        eprintln!("Skipping test_mistral_list_models: MISTRAL_API_KEY not set");
        return;
    }

    let provider = create_provider();
    let result = provider.list_models().await;

    match result {
        Ok(response) => {
            println!("Available Mistral models ({}):", response.data.len());
            for model in &response.data {
                println!("  - {}", model.id);
            }
            assert!(!response.data.is_empty(), "Expected at least one model");
            // mistral-small should always be available
            let has_small = response.data.iter().any(|m| m.id.contains("mistral-small"));
            assert!(has_small, "Expected mistral-small in model list");
        }
        Err(e) => panic!("List models failed: {:?}", e),
    }
}

// ---------------------------------------------------------------------------
// Provider metadata
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_mistral_provider_name() {
    let provider = MistralProvider::from_env().unwrap_or_else(|_| {
        // Use a placeholder key for metadata-only tests
        MistralProvider::new(
            "placeholder".to_string(),
            "mistral-small-latest".to_string(),
            "mistral-embed".to_string(),
            None,
        )
        .expect("Failed to create placeholder provider")
    });

    assert_eq!(LLMProvider::name(&provider), "mistral");
    assert_eq!(LLMProvider::model(&provider), "mistral-small-latest");
    assert_eq!(provider.dimension(), 1024);
}

#[tokio::test]
async fn test_mistral_with_model_builder() {
    let provider = MistralProvider::new(
        "placeholder".to_string(),
        "mistral-small-latest".to_string(),
        "mistral-embed".to_string(),
        None,
    )
    .expect("Failed to create provider")
    .with_model("mistral-large-latest");

    assert_eq!(LLMProvider::model(&provider), "mistral-large-latest");
}

// ---------------------------------------------------------------------------
// Factory integration
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_mistral_factory_from_env() {
    if !has_mistral_key() {
        eprintln!("Skipping test_mistral_factory_from_env: MISTRAL_API_KEY not set");
        return;
    }

    use edgequake_llm::ProviderFactory;

    // When MISTRAL_API_KEY is set, factory should auto-detect Mistral
    // (unless a higher-priority provider key is also set)
    // We test explicit selection here to be deterministic.
    let result = ProviderFactory::create(edgequake_llm::ProviderType::Mistral);
    match result {
        Ok((llm, _)) => {
            assert_eq!(llm.name(), "mistral");
        }
        Err(e) => panic!("Factory create Mistral failed: {:?}", e),
    }
}
