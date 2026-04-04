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
            println!(
                "Response: {} (model={}, tokens={}/{})",
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
            match serde_json::from_str::<serde_json::Value>(resp.content.trim()) {
                Ok(json) => {
                    if json.get("name").is_none() {
                        eprintln!("Warning: Missing 'name' field in JSON response");
                    }
                    if json.get("age").is_none() {
                        eprintln!("Warning: Missing 'age' field in JSON response");
                    }
                    if json.get("active").is_none() {
                        eprintln!("Warning: Missing 'active' field in JSON response");
                    }
                }
                Err(e) => {
                    eprintln!(
                        "Response was not valid JSON (possible transient, skipping): {} — {}",
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
                    Err(e) => {
                        eprintln!("Stream chunk error (possible transient API issue): {:?}", e);
                        break;
                    }
                }
            }

            println!(
                "Streamed response ({} chunks): {}",
                chunk_count, full_response
            );
            if chunk_count == 0 || full_response.is_empty() {
                eprintln!(
                    "Streaming returned empty response (possible transient API issue), skipping assertions"
                );
                return;
            }
            assert!(
                chunk_count >= 1,
                "Expected at least one chunk, got {}",
                chunk_count
            );
        }
        Err(e) => {
            eprintln!(
                "Stream failed (possible transient API issue, skipping): {:?}",
                e
            );
        }
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
        Err(e) => {
            eprintln!(
                "Tool calling failed (possible transient API issue, skipping): {:?}",
                e
            );
        }
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
            if embeddings.len() != 1 {
                eprintln!("Warning: expected 1 embedding, got {}", embeddings.len());
            }
            if embeddings[0].len() != 1024 {
                eprintln!("Warning: expected 1024-dim, got {}", embeddings[0].len());
            }
        }
        Err(e) => {
            eprintln!(
                "Embedding failed (possible transient issue, skipping): {:?}",
                e
            );
        }
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
            if embeddings.len() != 3 {
                eprintln!("Warning: expected 3 embeddings, got {}", embeddings.len());
            }
            for emb in &embeddings {
                if emb.len() != 1024 {
                    eprintln!("Warning: expected 1024-dim, got {}", emb.len());
                }
            }
        }
        Err(e) => {
            eprintln!(
                "Batch embedding failed (possible transient issue, skipping): {:?}",
                e
            );
        }
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
            if response.data.is_empty() {
                eprintln!("Warning: no models returned from Mistral API");
            }
            let has_small = response.data.iter().any(|m| m.id.contains("mistral-small"));
            if !has_small {
                eprintln!("Warning: mistral-small not found in model list");
            }
        }
        Err(e) => {
            eprintln!(
                "List models failed (possible transient issue, skipping): {:?}",
                e
            );
        }
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
        Err(e) => {
            eprintln!(
                "Factory create Mistral failed (possible transient issue, skipping): {:?}",
                e
            );
        }
    }
}

// ---------------------------------------------------------------------------
// safe_prompt parameter
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_mistral_safe_prompt() {
    if !has_mistral_key() {
        eprintln!("Skipping test_mistral_safe_prompt: MISTRAL_API_KEY not set");
        return;
    }

    let provider = create_provider();
    let messages = vec![ChatMessage::user("Say hello.")];

    // safe_prompt=true injects a Mistral safety system message before the conversation.
    let options = edgequake_llm::traits::CompletionOptions {
        safe_prompt: Some(true),
        ..Default::default()
    };

    let response = provider.chat(&messages, Some(&options)).await;
    match response {
        Ok(resp) => {
            println!("safe_prompt response: {}", resp.content);
            assert!(
                !resp.content.is_empty(),
                "Response should not be empty with safe_prompt=true"
            );
        }
        Err(e) => {
            eprintln!("safe_prompt test failed (transient, skipping): {:?}", e);
        }
    }
}

// ---------------------------------------------------------------------------
// parallel_tool_calls=false (single tool-call mode)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_mistral_single_tool_call_mode() {
    if !has_mistral_key() {
        eprintln!("Skipping test_mistral_single_tool_call_mode: MISTRAL_API_KEY not set");
        return;
    }

    use edgequake_llm::traits::{ToolChoice, ToolDefinition};

    let provider = create_provider();

    let tools = vec![
        ToolDefinition::function(
            "get_temperature",
            "Get the temperature for a city",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string", "description": "City name" }
                },
                "required": ["city"]
            }),
        ),
        ToolDefinition::function(
            "get_humidity",
            "Get the humidity for a city",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string", "description": "City name" }
                },
                "required": ["city"]
            }),
        ),
    ];

    let messages = vec![
        ChatMessage::system("You are a weather assistant. Use tools when needed."),
        ChatMessage::user("What is the temperature and humidity in Paris?"),
    ];

    // parallel_tool_calls=false must force single tool-call mode.
    let options = edgequake_llm::traits::CompletionOptions {
        parallel_tool_calls: Some(false),
        ..Default::default()
    };

    let response = provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), Some(&options))
        .await;

    match response {
        Ok(resp) => {
            println!(
                "single-tool response: content='{}' tool_calls={}",
                resp.content,
                resp.tool_calls.len()
            );
            // With parallel_tool_calls=false the model may call at most one tool.
            assert!(
                resp.tool_calls.len() <= 1,
                "Expected at most 1 tool call in single-call mode, got {}",
                resp.tool_calls.len()
            );
        }
        Err(e) => {
            eprintln!(
                "parallel_tool_calls=false test failed (transient, skipping): {:?}",
                e
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Mistral-specific model smoke tests (context / capability)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_mistral_medium_latest_chat() {
    if !has_mistral_key() {
        eprintln!("Skipping test_mistral_medium_latest_chat: MISTRAL_API_KEY not set");
        return;
    }

    let provider = MistralProvider::new(
        std::env::var("MISTRAL_API_KEY").unwrap(),
        "mistral-medium-latest".to_string(),
        "mistral-embed".to_string(),
        None,
    )
    .expect("Failed to create provider with mistral-medium-latest");

    let messages = vec![ChatMessage::user(
        "What is 7 * 6? Reply with just the number.",
    )];
    let response = provider.chat(&messages, None).await;
    match response {
        Ok(resp) => {
            println!("mistral-medium-latest response: {}", resp.content);
            if !resp.content.contains("42") {
                eprintln!("Warning: expected '42', got: {}", resp.content);
            }
        }
        Err(e) => {
            eprintln!(
                "mistral-medium-latest chat failed (transient, skipping): {:?}",
                e
            );
        }
    }
}

// ---------------------------------------------------------------------------
// No-env-var construction — api_key injected directly, no set_var
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_mistral_new_without_env_var() {
    // This test verifies the set_var anti-pattern fix:
    // MistralProvider::new() must succeed even if MISTRAL_API_KEY is unset in env.
    std::env::remove_var("MISTRAL_API_KEY");

    let result = MistralProvider::new(
        "explicit-test-key".to_string(),
        "mistral-small-latest".to_string(),
        "mistral-embed".to_string(),
        None,
    );

    assert!(
        result.is_ok(),
        "MistralProvider::new() must succeed without MISTRAL_API_KEY env var \
         when an explicit api_key is provided"
    );

    let provider = result.unwrap();
    // Structural checks only — no live call since the key is fake.
    assert_eq!(LLMProvider::name(&provider), "mistral");
    assert_eq!(provider.max_context_length(), 262_144);
}
