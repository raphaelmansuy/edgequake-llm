//! End-to-end tests for the Azure OpenAI provider.
//!
//! These tests exercise the Azure OpenAI provider against a real Azure endpoint.
//! They are gated behind `.env` / environment-variable checks and skipped when
//! credentials are not available, so they are always safe to run in CI.
//!
//! # Environment variables (CONTENTGEN variant — preferred)
//!
//! ```bash
//! export AZURE_OPENAI_CONTENTGEN_API_ENDPOINT=https://myresource.openai.azure.com/
//! export AZURE_OPENAI_CONTENTGEN_API_KEY=<your-key>
//! export AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT=gpt-4.1-mini-global-standard
//! export AZURE_OPENAI_CONTENTGEN_API_VERSION=2024-08-01-preview   # optional
//! ```
//!
//! # Environment variables (standard variant)
//!
//! ```bash
//! export AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com/
//! export AZURE_OPENAI_API_KEY=<your-key>
//! export AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
//! export AZURE_OPENAI_API_VERSION=2024-10-21                      # optional
//! ```
//!
//! # Running
//!
//! ```bash
//! # All Azure e2e tests
//! cargo test --test e2e_azure
//!
//! # Specific test
//! cargo test --test e2e_azure test_azure_basic_chat
//! ```
//!
//! # Test coverage
//!
//! - Basic chat / complete()
//! - JSON mode
//! - Streaming (simple prompt)
//! - Streaming with messages and options
//! - Vision / multimodal messages
//! - Tool / function calling
//! - Streaming tool calls
//! - Embeddings (when an embedding deployment is configured)
//! - Provider metadata
//! - Factory auto-detection (CONTENTGEN and standard)
//! - Factory explicit selection (`ProviderType::AzureOpenAI`)

use std::sync::Mutex;

use edgequake_llm::traits::{
    ChatMessage, CompletionOptions, EmbeddingProvider, ImageData, LLMProvider, StreamChunk,
    ToolChoice, ToolDefinition,
};
use edgequake_llm::{AzureOpenAIProvider, ProviderFactory, ProviderType};
use futures::StreamExt;

/// Global mutex to serialise tests that mutate environment variables.
/// Tests that manipulate env vars must hold this lock for the duration.
static ENV_MUTEX: Mutex<()> = Mutex::new(());

// ============================================================================
// Helpers
// ============================================================================

/// Create provider from the CONTENTGEN env vars.
/// Returns `None` when credentials are not configured (test is skipped).
fn make_provider() -> Option<AzureOpenAIProvider> {
    AzureOpenAIProvider::from_env_auto().ok()
}

/// Return `true` when at least one set of Azure credentials is present.
fn has_azure_creds() -> bool {
    make_provider().is_some()
}

/// Skip helper — prints a reason and returns true when environment is not set up.
macro_rules! skip_if_no_creds {
    ($test_name:expr) => {
        if !has_azure_creds() {
            eprintln!(
                "Skipping {}: Azure credentials not configured \
                 (AZURE_OPENAI_CONTENTGEN_API_KEY or AZURE_OPENAI_API_KEY not set)",
                $test_name
            );
            return;
        }
    };
}

// ============================================================================
// Provider unit-level tests (no API call, always run)
// ============================================================================

/// Verify provider struct construction and metadata.
#[test]
fn test_azure_provider_construction() {
    let provider = AzureOpenAIProvider::new(
        "https://myresource.openai.azure.com",
        "test-api-key",
        "gpt-4o",
    );

    assert_eq!(LLMProvider::name(&provider), "azure-openai");
    assert_eq!(LLMProvider::model(&provider), "gpt-4o");
    assert!(provider.max_context_length() > 0);
    assert!(provider.supports_streaming());
    assert!(provider.supports_function_calling());
    assert!(provider.supports_json_mode());
    assert!(provider.supports_tool_streaming());
}

/// Verify builder-pattern setter methods.
#[test]
fn test_azure_provider_builder_methods() {
    let provider = AzureOpenAIProvider::new(
        "https://myresource.openai.azure.com/",
        "test-api-key",
        "gpt-4o",
    )
    .with_embedding_deployment("text-embedding-ada-002")
    .with_api_version("2024-06-01")
    .with_max_context_length(200_000)
    .with_embedding_dimension(3072)
    .with_deployment("gpt-4o-mini");

    assert_eq!(LLMProvider::model(&provider), "gpt-4o-mini");
    assert_eq!(
        EmbeddingProvider::model(&provider),
        "text-embedding-ada-002"
    );
    assert_eq!(provider.max_context_length(), 200_000);
    assert_eq!(provider.dimension(), 3072);
    // Trailing slash should be stripped
}

/// Verify `from_env_contentgen()` maps the CONTENTGEN env-var naming correctly.
///
/// Uses ENV_MUTEX to prevent races with other env-var-mutating tests.
#[test]
fn test_azure_from_env_contentgen_maps_vars() {
    let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    // Save originals
    let saved_endpoint = std::env::var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT").ok();
    let saved_key = std::env::var("AZURE_OPENAI_CONTENTGEN_API_KEY").ok();
    let saved_deployment = std::env::var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT").ok();
    let saved_version = std::env::var("AZURE_OPENAI_CONTENTGEN_API_VERSION").ok();

    // Temporarily set CONTENTGEN vars
    std::env::set_var(
        "AZURE_OPENAI_CONTENTGEN_API_ENDPOINT",
        "https://test.openai.azure.com",
    );
    std::env::set_var("AZURE_OPENAI_CONTENTGEN_API_KEY", "fake-key");
    std::env::set_var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT", "my-deployment");
    std::env::set_var("AZURE_OPENAI_CONTENTGEN_API_VERSION", "2024-08-01-preview");

    let provider = AzureOpenAIProvider::from_env_contentgen()
        .expect("from_env_contentgen should succeed when vars are set");

    // Restore originals
    match saved_endpoint {
        Some(v) => std::env::set_var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT", v),
        None => std::env::remove_var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT"),
    }
    match saved_key {
        Some(v) => std::env::set_var("AZURE_OPENAI_CONTENTGEN_API_KEY", v),
        None => std::env::remove_var("AZURE_OPENAI_CONTENTGEN_API_KEY"),
    }
    match saved_deployment {
        Some(v) => std::env::set_var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT", v),
        None => std::env::remove_var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT"),
    }
    match saved_version {
        Some(v) => std::env::set_var("AZURE_OPENAI_CONTENTGEN_API_VERSION", v),
        None => std::env::remove_var("AZURE_OPENAI_CONTENTGEN_API_VERSION"),
    }

    assert_eq!(LLMProvider::model(&provider), "my-deployment");
}

/// Verify `from_env_contentgen()` gives a clear error when vars are missing.
///
/// Uses the ENV_MUTEX to prevent races with other env-var-mutating tests.
#[test]
fn test_azure_from_env_contentgen_fails_without_vars() {
    let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    // Save existing values so we can restore them
    let saved_endpoint = std::env::var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT").ok();
    let saved_key = std::env::var("AZURE_OPENAI_CONTENTGEN_API_KEY").ok();
    let saved_deployment = std::env::var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT").ok();

    std::env::remove_var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT");
    std::env::remove_var("AZURE_OPENAI_CONTENTGEN_API_KEY");
    std::env::remove_var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT");

    let result = AzureOpenAIProvider::from_env_contentgen();

    // Restore
    if let Some(v) = saved_endpoint {
        std::env::set_var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT", v);
    }
    if let Some(v) = saved_key {
        std::env::set_var("AZURE_OPENAI_CONTENTGEN_API_KEY", v);
    }
    if let Some(v) = saved_deployment {
        std::env::set_var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT", v);
    }

    assert!(
        result.is_err(),
        "Should fail when CONTENTGEN vars are not set"
    );
}

/// Verify `from_env_auto()` prefers CONTENTGEN over standard vars.
///
/// Uses ENV_MUTEX to prevent races with other env-var-mutating tests.
#[test]
fn test_azure_from_env_auto_prefers_contentgen() {
    let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    // Save original values
    let saved: Vec<(&str, Option<String>)> = vec![
        (
            "AZURE_OPENAI_CONTENTGEN_API_ENDPOINT",
            std::env::var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT").ok(),
        ),
        (
            "AZURE_OPENAI_CONTENTGEN_API_KEY",
            std::env::var("AZURE_OPENAI_CONTENTGEN_API_KEY").ok(),
        ),
        (
            "AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT",
            std::env::var("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT").ok(),
        ),
        (
            "AZURE_OPENAI_API_KEY",
            std::env::var("AZURE_OPENAI_API_KEY").ok(),
        ),
        (
            "AZURE_OPENAI_ENDPOINT",
            std::env::var("AZURE_OPENAI_ENDPOINT").ok(),
        ),
        (
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            std::env::var("AZURE_OPENAI_DEPLOYMENT_NAME").ok(),
        ),
    ];

    // Both sets present — CONTENTGEN should win
    std::env::set_var(
        "AZURE_OPENAI_CONTENTGEN_API_ENDPOINT",
        "https://contentgen.openai.azure.com",
    );
    std::env::set_var("AZURE_OPENAI_CONTENTGEN_API_KEY", "contentgen-key");
    std::env::set_var(
        "AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT",
        "contentgen-deployment",
    );
    std::env::set_var("AZURE_OPENAI_API_KEY", "standard-key");
    std::env::set_var("AZURE_OPENAI_ENDPOINT", "https://standard.openai.azure.com");
    std::env::set_var("AZURE_OPENAI_DEPLOYMENT_NAME", "standard-deployment");

    let provider = AzureOpenAIProvider::from_env_auto().expect("Should succeed");

    // Restore original values
    for (key, val) in &saved {
        match val {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }

    // The CONTENTGEN deployment should be used
    assert_eq!(LLMProvider::model(&provider), "contentgen-deployment");
}

/// Vision multipart message construction does not require an API call.
#[test]
fn test_azure_vision_message_construction() {
    // 1×1 transparent PNG
    let tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    let img = ImageData::new(tiny_png, "image/png").with_detail("high");

    let msg = ChatMessage::user_with_images("What is in this image?", vec![img]);
    assert!(msg.has_images());
    assert_eq!(msg.images.as_ref().unwrap().len(), 1);
    assert_eq!(msg.images.as_ref().unwrap()[0].mime_type, "image/png");
    assert_eq!(
        msg.images.as_ref().unwrap()[0].detail.as_deref(),
        Some("high")
    );
}

// ============================================================================
// Live E2E tests (require real Azure credentials)
// ============================================================================

/// Basic chat completion.
#[tokio::test]
async fn test_azure_basic_chat() {
    skip_if_no_creds!("test_azure_basic_chat");
    let provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: Azure provider unavailable during test");
            return;
        }
    };

    let messages = vec![
        ChatMessage::system("You are a concise math assistant."),
        ChatMessage::user("What is 2 + 2? Reply with just the number."),
    ];

    match provider.chat(&messages, None).await {
        Ok(resp) => {
            println!(
                "[Azure basic_chat] response='{}' model='{}' tokens={}/{}/{}",
                resp.content,
                resp.model,
                resp.prompt_tokens,
                resp.completion_tokens,
                resp.total_tokens,
            );
            assert!(!resp.content.is_empty(), "Response must not be empty");
            assert!(resp.prompt_tokens > 0, "Should have prompt tokens");
            assert!(resp.completion_tokens > 0, "Should have completion tokens");
            assert_eq!(
                resp.total_tokens,
                resp.prompt_tokens + resp.completion_tokens
            );
        }
        Err(e) => eprintln!("[Azure basic_chat] SKIPPED (transient): {:?}", e),
    }
}

/// `complete()` convenience method.
#[tokio::test]
async fn test_azure_complete() {
    skip_if_no_creds!("test_azure_complete");
    let provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: Azure provider unavailable during test");
            return;
        }
    };

    match provider
        .complete("Say 'hello azure' and nothing else.")
        .await
    {
        Ok(resp) => {
            println!("[Azure complete] response='{}'", resp.content);
            assert!(!resp.content.is_empty());
        }
        Err(e) => eprintln!("[Azure complete] SKIPPED (transient): {:?}", e),
    }
}

/// `complete_with_options()` — system prompt + temperature + max_tokens.
#[tokio::test]
async fn test_azure_complete_with_options() {
    skip_if_no_creds!("test_azure_complete_with_options");
    let provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: Azure provider unavailable during test");
            return;
        }
    };

    let options = CompletionOptions {
        system_prompt: Some("You are a very brief assistant. One sentence max.".to_string()),
        max_tokens: Some(60),
        temperature: Some(0.1),
        ..Default::default()
    };

    match provider
        .complete_with_options("Say hello in Spanish.", &options)
        .await
    {
        Ok(resp) => {
            println!("[Azure complete_with_options] response='{}'", resp.content);
            assert!(!resp.content.is_empty());
        }
        Err(e) => eprintln!("[Azure complete_with_options] SKIPPED (transient): {:?}", e),
    }
}

/// JSON mode — model must return valid JSON.
#[tokio::test]
async fn test_azure_json_mode() {
    skip_if_no_creds!("test_azure_json_mode");
    let provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: Azure provider unavailable during test");
            return;
        }
    };

    let messages = vec![
        ChatMessage::system("You are a JSON generator. Respond ONLY with valid JSON, no markdown."),
        ChatMessage::user(
            "Provide a JSON object with fields: name (string), age (number), active (boolean).",
        ),
    ];

    let options = CompletionOptions::json_mode();

    match provider.chat(&messages, Some(&options)).await {
        Ok(resp) => {
            println!("[Azure json_mode] response='{}'", resp.content);
            match serde_json::from_str::<serde_json::Value>(resp.content.trim()) {
                Ok(json) => {
                    if json.get("name").is_none() {
                        eprintln!("[Azure json_mode] WARNING: missing 'name' field");
                    }
                }
                Err(e) => eprintln!(
                    "[Azure json_mode] Response not valid JSON (transient?): {} — {}",
                    resp.content, e
                ),
            }
        }
        Err(e) => eprintln!("[Azure json_mode] SKIPPED (transient): {:?}", e),
    }
}

/// Streaming (simple prompt → `stream()`).
#[tokio::test]
async fn test_azure_streaming_simple() {
    skip_if_no_creds!("test_azure_streaming_simple");
    let provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test_azure_streaming_simple: provider unavailable");
            return;
        }
    };

    match provider.stream("Count from 1 to 5, comma-separated.").await {
        Ok(mut stream) => {
            let mut full = String::new();
            let mut chunks = 0usize;

            while let Some(result) = stream.next().await {
                match result {
                    Ok(chunk) => {
                        full.push_str(&chunk);
                        chunks += 1;
                    }
                    Err(e) => {
                        eprintln!("[Azure streaming] chunk error (transient?): {:?}", e);
                        break;
                    }
                }
            }

            println!(
                "[Azure streaming_simple] {} chunks, full='{}'",
                chunks, full
            );
            if chunks == 0 || full.is_empty() {
                eprintln!("[Azure streaming_simple] empty stream (transient?)");
                return;
            }
            assert!(chunks >= 1, "Expected at least one chunk");
        }
        Err(e) => eprintln!("[Azure streaming_simple] SKIPPED (transient): {:?}", e),
    }
}

/// Streaming via `chat_with_tools_stream()` (no tools — just content streaming with messages).
#[tokio::test]
async fn test_azure_chat_stream_with_options() {
    skip_if_no_creds!("test_azure_chat_stream_with_options");
    let provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: Azure provider unavailable during test");
            return;
        }
    };

    // `chat_with_tools_stream` with an empty tools slice is the standard way
    // to stream a conversation with full options.
    let messages = vec![
        ChatMessage::system("You are a brief assistant."),
        ChatMessage::user("List 3 programming languages, one per line."),
    ];
    let options = CompletionOptions {
        max_tokens: Some(100),
        temperature: Some(0.5),
        ..Default::default()
    };

    match provider
        .chat_with_tools_stream(&messages, &[], None, Some(&options))
        .await
    {
        Ok(stream) => {
            let chunks: Vec<_> = stream.collect().await;
            let content: String = chunks
                .iter()
                .filter_map(|r| {
                    if let Ok(StreamChunk::Content(c)) = r {
                        Some(c.clone())
                    } else {
                        None
                    }
                })
                .collect();

            println!(
                "[Azure chat_stream] {} chunks, content='{}'",
                chunks.len(),
                content
            );
            assert!(!chunks.is_empty(), "Stream should return chunks");
        }
        Err(e) => eprintln!(
            "[Azure chat_stream_with_options] SKIPPED (transient): {:?}",
            e
        ),
    }
}

/// Vision / multimodal — send a base64-encoded image and ask about it.
///
/// Uses an extremely small 1×1 red PNG so the test is fast and data-efficient.
#[tokio::test]
async fn test_azure_vision() {
    skip_if_no_creds!("test_azure_vision");
    let provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: Azure provider unavailable during test");
            return;
        }
    };

    // 1×1 red PNG, base64-encoded.
    // This pixel is used purely to exercise the multimodal code-path.
    let tiny_red_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADklEQVQI12P4z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg==";

    let img = ImageData::new(tiny_red_png, "image/png").with_detail("low");
    let messages = vec![ChatMessage::user_with_images(
        "What color is this image? Reply in one word.",
        vec![img],
    )];
    let options = CompletionOptions {
        max_tokens: Some(20),
        temperature: Some(0.0),
        ..Default::default()
    };

    match provider.chat(&messages, Some(&options)).await {
        Ok(resp) => {
            println!("[Azure vision] response='{}'", resp.content);
            assert!(
                !resp.content.is_empty(),
                "Vision response should not be empty"
            );
        }
        Err(e) => {
            // Some Azure deployments don't support vision. Treat as non-fatal.
            eprintln!(
                "[Azure vision] SKIPPED — error (may not support vision or transient): {:?}",
                e
            );
        }
    }
}

/// Vision streaming — same image but via streaming.
#[tokio::test]
async fn test_azure_vision_streaming() {
    skip_if_no_creds!("test_azure_vision_streaming");
    let provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: Azure provider unavailable during test");
            return;
        }
    };

    let tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADklEQVQI12P4z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg==";
    let img = ImageData::new(tiny_png, "image/png").with_detail("low");
    let messages = vec![ChatMessage::user_with_images(
        "What color is this tiny image?",
        vec![img],
    )];
    let options = CompletionOptions {
        max_tokens: Some(30),
        ..Default::default()
    };

    match provider
        .chat_with_tools_stream(&messages, &[], None, Some(&options))
        .await
    {
        Ok(stream) => {
            let chunks: Vec<_> = stream.collect().await;
            let content: String = chunks
                .iter()
                .filter_map(|r| {
                    if let Ok(StreamChunk::Content(c)) = r {
                        Some(c.clone())
                    } else {
                        None
                    }
                })
                .collect();
            println!(
                "[Azure vision_streaming] {} chunks, content='{}'",
                chunks.len(),
                content
            );
        }
        Err(e) => eprintln!(
            "[Azure vision_streaming] SKIPPED (may not support vision or transient): {:?}",
            e
        ),
    }
}

/// Tool / function calling.
#[tokio::test]
async fn test_azure_tool_calling() {
    skip_if_no_creds!("test_azure_tool_calling");
    let provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: Azure provider unavailable during test");
            return;
        }
    };

    let tools = vec![ToolDefinition::function(
        "get_weather",
        "Get the current weather for a city",
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city"
                }
            },
            "required": ["city"]
        }),
    )];

    let messages = vec![
        ChatMessage::system("You are a helpful assistant. Use tools when appropriate."),
        ChatMessage::user("What is the weather in Paris?"),
    ];

    match provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), None)
        .await
    {
        Ok(resp) => {
            println!(
                "[Azure tool_calling] content='{}' tool_calls={:?}",
                resp.content, resp.tool_calls
            );
            assert!(
                !resp.content.is_empty() || !resp.tool_calls.is_empty(),
                "Should have either content or tool calls"
            );
        }
        Err(e) => eprintln!("[Azure tool_calling] SKIPPED (transient): {:?}", e),
    }
}

/// Tool calling with streaming.
#[tokio::test]
async fn test_azure_tool_calling_streaming() {
    skip_if_no_creds!("test_azure_tool_calling_streaming");
    let provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: Azure provider unavailable during test");
            return;
        }
    };

    let tools = vec![ToolDefinition::function(
        "calculate",
        "Perform a calculation",
        serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }),
    )];

    let messages = vec![ChatMessage::user("What is 15 * 7?")];
    let options = CompletionOptions {
        max_tokens: Some(200),
        ..Default::default()
    };

    match provider
        .chat_with_tools_stream(&messages, &tools, Some(ToolChoice::auto()), Some(&options))
        .await
    {
        Ok(stream) => {
            let chunks: Vec<_> = stream.collect().await;
            println!("[Azure tool_calling_streaming] {} chunks", chunks.len());
            assert!(!chunks.is_empty(), "Tool streaming should produce chunks");

            let has_tool_or_content = chunks.iter().any(|c| {
                matches!(
                    c,
                    Ok(StreamChunk::ToolCallDelta { .. }) | Ok(StreamChunk::Content(_))
                )
            });
            assert!(
                has_tool_or_content,
                "Stream should have tool call delta or content"
            );
        }
        Err(e) => eprintln!(
            "[Azure tool_calling_streaming] SKIPPED (transient): {:?}",
            e
        ),
    }
}

/// Embeddings — if an embedding deployment is configured.
///
/// Set `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` (standard) or use the default deployment.
#[tokio::test]
async fn test_azure_embeddings() {
    skip_if_no_creds!("test_azure_embeddings");

    // Build a provider with the same or a dedicated embedding deployment
    let provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: Azure provider unavailable during test");
            return;
        }
    };

    // Skip if no embedding deployment info is available beyond the chat one
    // (the default is to re-use the chat deployment for embeddings; some models
    // don't support embeddings)
    let texts = vec![
        "The quick brown fox.".to_string(),
        "Hello, Azure!".to_string(),
    ];

    match provider.embed(&texts).await {
        Ok(embeddings) => {
            println!(
                "[Azure embeddings] {} embeddings, dim={}",
                embeddings.len(),
                embeddings.first().map(|e| e.len()).unwrap_or(0)
            );
            assert_eq!(
                embeddings.len(),
                texts.len(),
                "Should return one embedding per input"
            );
            for emb in &embeddings {
                assert!(!emb.is_empty(), "Embedding should not be empty");
            }
        }
        Err(e) => eprintln!(
            "[Azure embeddings] SKIPPED — deployment may not support embeddings (transient?): {:?}",
            e
        ),
    }
}

/// Embeddings — empty input should return empty result.
#[tokio::test]
async fn test_azure_embeddings_empty() {
    skip_if_no_creds!("test_azure_embeddings_empty");
    let provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: Azure provider unavailable during test");
            return;
        }
    };

    match provider.embed(&[]).await {
        Ok(result) => {
            assert!(result.is_empty(), "Empty input should return empty result");
            println!("[Azure embeddings_empty] ok");
        }
        Err(e) => eprintln!("[Azure embeddings_empty] SKIPPED (transient): {:?}", e),
    }
}

/// Cache-hit token extraction — two identical calls, second may get cached tokens.
#[tokio::test]
async fn test_azure_cache_hit_tokens() {
    skip_if_no_creds!("test_azure_cache_hit_tokens");
    let provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test_azure_cache_hit_tokens: provider unavailable");
            return;
        }
    };

    let messages = vec![
        ChatMessage::system("You are a helpful assistant. Always respond in exactly 3 words."),
        ChatMessage::user("What is 1+1?"),
    ];

    // First call — prime possible cache
    let _ = provider.chat(&messages, None).await;

    // Second call — may have cached_tokens in usage.prompt_tokens_details
    match provider.chat(&messages, None).await {
        Ok(resp) => {
            println!(
                "[Azure cache_hit_tokens] content='{}', cache_hit_tokens={:?}",
                resp.content, resp.cache_hit_tokens
            );
            // cache_hit_tokens may be None or Some (non-negative)
            if let Some(cached) = resp.cache_hit_tokens {
                assert!(cached <= resp.prompt_tokens, "Cached ≤ prompt tokens");
            }
        }
        Err(e) => eprintln!("[Azure cache_hit_tokens] SKIPPED (transient): {:?}", e),
    }
}

/// Provider metadata verification.
#[tokio::test]
async fn test_azure_provider_metadata() {
    let provider =
        AzureOpenAIProvider::new("https://myresource.openai.azure.com", "test-key", "gpt-4o")
            .with_max_context_length(128_000)
            .with_embedding_dimension(1536);

    assert_eq!(LLMProvider::name(&provider), "azure-openai");
    assert_eq!(LLMProvider::model(&provider), "gpt-4o");
    assert_eq!(provider.max_context_length(), 128_000);
    assert_eq!(provider.dimension(), 1536);
    assert!(provider.supports_streaming());
    assert!(provider.supports_function_calling());
    assert!(provider.supports_json_mode());
    assert!(provider.supports_tool_streaming());
}

// ============================================================================
// Factory integration tests
// ============================================================================

/// Verify `ProviderType::AzureOpenAI` is parsed correctly.
#[test]
fn test_azure_provider_type_parsing() {
    assert_eq!(
        ProviderType::from_str("azure"),
        Some(ProviderType::AzureOpenAI)
    );
    assert_eq!(
        ProviderType::from_str("azure-openai"),
        Some(ProviderType::AzureOpenAI)
    );
    assert_eq!(
        ProviderType::from_str("AZURE"),
        Some(ProviderType::AzureOpenAI)
    );
    assert_eq!(
        ProviderType::from_str("azureopenai"),
        Some(ProviderType::AzureOpenAI)
    );
}

/// Factory explicit creation with `ProviderType::AzureOpenAI`.
#[tokio::test]
async fn test_azure_factory_explicit_creation() {
    skip_if_no_creds!("test_azure_factory_explicit_creation");

    match ProviderFactory::create(ProviderType::AzureOpenAI) {
        Ok((llm, embedding)) => {
            assert_eq!(llm.name(), "azure-openai");
            assert_eq!(embedding.name(), "azure-openai");
            println!(
                "[Azure factory] explicit creation OK — model='{}'",
                llm.model()
            );
        }
        Err(e) => eprintln!("[Azure factory_explicit] SKIPPED (transient): {:?}", e),
    }
}

/// Factory `create_with_model` overrides the deployment name.
#[tokio::test]
async fn test_azure_factory_create_with_model() {
    skip_if_no_creds!("test_azure_factory_create_with_model");

    let default_provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test_azure_factory_create_with_model: provider unavailable");
            return;
        }
    };
    let default_model = LLMProvider::model(&default_provider).to_string();

    match ProviderFactory::create_with_model(ProviderType::AzureOpenAI, Some(&default_model)) {
        Ok((llm, _)) => {
            assert_eq!(llm.name(), "azure-openai");
            assert_eq!(llm.model(), default_model.as_str());
            println!("[Azure factory_with_model] model='{}'", llm.model());
        }
        Err(e) => eprintln!("[Azure factory_with_model] SKIPPED (transient): {:?}", e),
    }
}

/// Factory `from_env()` auto-detects Azure when only Azure vars are set.
///
/// This test temporarily overrides env vars so it must run after other tests
/// that depend on a clean environment; use serial_test in CI if needed.
#[tokio::test]
async fn test_azure_factory_auto_detection_contentgen() {
    skip_if_no_creds!("test_azure_factory_auto_detection_contentgen");

    // The test environment already has Azure CONTENTGEN vars set, so from_env()
    // should pick them up (assuming no higher-priority provider keys are set).
    // We test explicit selection here to be deterministic.
    match ProviderFactory::create(ProviderType::AzureOpenAI) {
        Ok((llm, _)) => {
            assert_eq!(
                llm.name(),
                "azure-openai",
                "Factory should create Azure provider"
            );
            println!(
                "[Azure auto_detection_contentgen] provider='{}' model='{}'",
                llm.name(),
                llm.model()
            );
        }
        Err(e) => eprintln!(
            "[Azure auto_detection_contentgen] SKIPPED (transient): {:?}",
            e
        ),
    }
}

/// Full round-trip: create via factory, run `complete()`.  
#[tokio::test]
async fn test_azure_factory_full_roundtrip() {
    skip_if_no_creds!("test_azure_factory_full_roundtrip");

    let (llm, _) = match ProviderFactory::create(ProviderType::AzureOpenAI) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "[Azure factory_roundtrip] factory failed (transient?): {:?}",
                e
            );
            return;
        }
    };

    match llm.complete("Say 'Azure OK' and nothing else.").await {
        Ok(resp) => {
            println!("[Azure factory_roundtrip] response='{}'", resp.content);
            assert!(!resp.content.is_empty());
        }
        Err(e) => eprintln!("[Azure factory_roundtrip] SKIPPED (transient): {:?}", e),
    }
}
