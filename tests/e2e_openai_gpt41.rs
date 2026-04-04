//! End-to-end tests for the OpenAI provider focused on **gpt-4.1**.
//!
//! These tests verify correct error classification (no string heuristics),
//! streaming, tool calling, and retry-strategy behaviour for all edge cases
//! that surfaced during the rate-limit bug investigation.
//!
//! # What is covered
//!
//! | Category            | Test                                         |
//! |---------------------|----------------------------------------------|
//! | Basic chat          | `test_gpt41_basic_chat`                      |
//! | Streaming           | `test_gpt41_streaming`                       |
//! | Tool calling        | `test_gpt41_tool_calling`                    |
//! | Tool calling stream | `test_gpt41_tool_calling_stream`             |
//! | Max tokens          | `test_gpt41_max_tokens_respected`            |
//! | Error: auth         | `test_gpt41_invalid_auth_is_auth_error`      |
//! | Error: bad model    | `test_gpt41_bad_model_is_model_not_found`    |
//! | Error: bad request  | `test_gpt41_bad_request_is_invalid_request`  |
//! | Unit: error codes   | `test_error_classification_unit_*`            |
//! | Unit: retry-after   | `test_retry_after_parsing_*`                 |
//! | Unit: retry strat   | `test_retry_strategy_*`                      |
//!
//! # Environment variables
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! ```
//!
//! # Running
//!
//! ```bash
//! # All tests (including integration)
//! cargo test --test e2e_openai_gpt41
//!
//! # Only the unit tests (no API key needed)
//! cargo test --test e2e_openai_gpt41 unit
//!
//! # Only live API tests
//! cargo test --test e2e_openai_gpt41 -- --ignored
//! ```

use edgequake_llm::providers::openai::OpenAIProvider;
use edgequake_llm::traits::{
    ChatMessage, CompletionOptions, LLMProvider, StreamChunk, ToolChoice, ToolDefinition,
};
use edgequake_llm::{LlmError, RetryStrategy};
use futures::StreamExt;
use std::time::Duration;

// ============================================================================
// Test helpers
// ============================================================================

const MODEL: &str = "gpt-4.1";

/// Create a provider pointing at gpt-4.1.
/// Returns `None` (and the caller skips the test) when `OPENAI_API_KEY` is absent.
fn make_provider() -> Option<OpenAIProvider> {
    let api_key = std::env::var("OPENAI_API_KEY").ok()?;
    Some(OpenAIProvider::new(api_key).with_model(MODEL))
}

// ============================================================================
// Live API tests  (require OPENAI_API_KEY)
// ============================================================================

/// Basic chat completion with gpt-4.1.
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_gpt41_basic_chat() {
    let provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("OPENAI_API_KEY not set, skipping");
            return;
        }
    };

    let messages = vec![ChatMessage::user(
        "What is 3 + 4? Reply with just the number.",
    )];
    let response = provider.chat(&messages, None).await.expect("Chat failed");

    assert!(!response.content.is_empty(), "Response must not be empty");
    assert!(response.content.contains('7'), "3+4 must equal 7");
    assert!(response.prompt_tokens > 0, "prompt_tokens must be > 0");
    assert!(
        response.completion_tokens > 0,
        "completion_tokens must be > 0"
    );
    assert_eq!(
        response.total_tokens,
        response.prompt_tokens + response.completion_tokens,
        "total_tokens must equal prompt + completion"
    );

    println!(
        "gpt-4.1 basic chat: '{}' | tokens {}/{}/{}",
        response.content, response.prompt_tokens, response.completion_tokens, response.total_tokens
    );
}

/// Streaming chat with gpt-4.1.
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_gpt41_streaming() {
    let provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("OPENAI_API_KEY not set, skipping");
            return;
        }
    };

    let stream = provider
        .stream("Count from 1 to 3, one number per line.")
        .await
        .expect("Stream failed to start");

    let chunks: Vec<_> = stream.collect().await;
    assert!(!chunks.is_empty(), "Stream must produce at least one chunk");

    let full: String = chunks.into_iter().filter_map(|r| r.ok()).collect();

    assert!(!full.is_empty(), "Stream content must not be empty");
    assert!(
        full.contains('1') && full.contains('3'),
        "Stream should contain numbers 1 and 3, got: {:?}",
        full
    );

    println!("gpt-4.1 stream: {:?}", full);
}

/// Chat with tools (non-streaming).
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_gpt41_tool_calling() {
    let provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("OPENAI_API_KEY not set, skipping");
            return;
        }
    };

    let tools = vec![ToolDefinition::function(
        "get_capital",
        "Return the capital city of a country",
        serde_json::json!({
            "type": "object",
            "properties": {
                "country": {
                    "type": "string",
                    "description": "The country name"
                }
            },
            "required": ["country"]
        }),
    )];

    let messages = vec![ChatMessage::user("What is the capital of Japan?")];
    let options = CompletionOptions {
        max_tokens: Some(200),
        ..Default::default()
    };

    let response = provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), Some(&options))
        .await
        .expect("Tool calling failed");

    // Either the model returns a tool call or a direct answer
    let has_output = !response.content.is_empty() || !response.tool_calls.is_empty();
    assert!(has_output, "Response must contain content or a tool call");

    println!(
        "gpt-4.1 tools: content={:?}, tool_calls={}",
        response.content,
        response.tool_calls.len()
    );
}

/// Streaming tool calling with gpt-4.1.
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_gpt41_tool_calling_stream() {
    let provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("OPENAI_API_KEY not set, skipping");
            return;
        }
    };

    let tools = vec![ToolDefinition::function(
        "get_weather",
        "Get current weather for a city",
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }),
    )];

    let messages = vec![ChatMessage::user("What is the weather in Tokyo?")];
    let options = CompletionOptions {
        max_tokens: Some(200),
        ..Default::default()
    };

    let stream = provider
        .chat_with_tools_stream(&messages, &tools, Some(ToolChoice::auto()), Some(&options))
        .await
        .expect("Tool stream failed to start");

    let chunks: Vec<_> = stream.collect().await;
    assert!(!chunks.is_empty(), "Tool stream must produce chunks");

    // All chunks must be Ok (no errors in the stream)
    for chunk in &chunks {
        if let Err(e) = chunk {
            panic!("Unexpected stream error: {}", e);
        }
    }

    let has_tool_or_content = chunks.iter().any(|c| {
        matches!(
            c,
            Ok(StreamChunk::ToolCallDelta { .. }) | Ok(StreamChunk::Content(_))
        )
    });
    assert!(
        has_tool_or_content,
        "Stream must contain ToolCallDelta or Content chunks"
    );

    println!("gpt-4.1 tool stream: {} chunks", chunks.len());
}

/// Verify that max_tokens is respected: the response must be very short.
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_gpt41_max_tokens_respected() {
    let provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("OPENAI_API_KEY not set, skipping");
            return;
        }
    };

    let options = CompletionOptions {
        max_tokens: Some(5),
        ..Default::default()
    };

    let messages = vec![ChatMessage::user(
        "Describe the entire history of the universe.",
    )];
    let response = provider
        .chat(&messages, Some(&options))
        .await
        .expect("Chat with max_tokens failed");

    assert!(
        response.completion_tokens <= 10,
        "completion_tokens ({}) should be near the 5-token limit",
        response.completion_tokens
    );

    println!(
        "gpt-4.1 max_tokens: {:?} | completion_tokens={}",
        response.content, response.completion_tokens
    );
}

/// An invalid API key must produce `LlmError::AuthError`, not `ApiError`.
/// Verifies that `From<OpenAIError>` uses the `code` field, not message heuristics.
#[tokio::test]
#[ignore = "Requires network access"]
async fn test_gpt41_invalid_auth_is_auth_error() {
    let provider =
        OpenAIProvider::new("sk-invalid_key_000000000000000000000000000000000000".to_string())
            .with_model(MODEL);

    let messages = vec![ChatMessage::user("Hello")];
    let err = provider
        .chat(&messages, None)
        .await
        .expect_err("Expected auth error, got success");

    assert!(
        matches!(err, LlmError::AuthError(_)),
        "Invalid API key must produce AuthError, got: {}",
        err
    );

    println!("Auth error correctly classified: {}", err);
}

/// A non-existent model name must produce `LlmError::ModelNotFound`.
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_gpt41_bad_model_is_model_not_found() {
    let provider = match make_provider() {
        Some(p) => p.with_model("gpt-4.1-this-model-does-not-exist"),
        None => {
            eprintln!("OPENAI_API_KEY not set, skipping");
            return;
        }
    };

    let messages = vec![ChatMessage::user("Hello")];
    let err = provider
        .chat(&messages, None)
        .await
        .expect_err("Expected model-not-found error, got success");

    assert!(
        matches!(err, LlmError::ModelNotFound(_)),
        "Non-existent model must produce ModelNotFound, got: {}",
        err
    );

    println!("Model-not-found correctly classified: {}", err);
}

/// An empty messages list is an invalid request and must not panic.
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_gpt41_bad_request_is_invalid_request() {
    let provider = match make_provider() {
        Some(p) => p,
        None => {
            eprintln!("OPENAI_API_KEY not set, skipping");
            return;
        }
    };

    // Sending an empty message with no body - OpenAI should reject it.
    let messages = vec![ChatMessage::user("")];
    let options = CompletionOptions {
        max_tokens: Some(1),
        ..Default::default()
    };

    // We just check that we don't panic. The error type depends on what the
    // server actually returns; accept either ApiError or InvalidRequest.
    let result = provider.chat(&messages, Some(&options)).await;
    match result {
        Ok(_) => println!("Empty message accepted (server returned a response)"),
        Err(e) => {
            println!("Empty message rejected as expected: {}", e);
            // Not ModelNotFound, not AuthError
            assert!(
                !matches!(e, LlmError::AuthError(_)),
                "Should not be AuthError: {}",
                e
            );
            assert!(
                !matches!(e, LlmError::ModelNotFound(_)),
                "Should not be ModelNotFound: {}",
                e
            );
        }
    }
}

// ============================================================================
// Unit tests — no network required
// ============================================================================

/// Verify that `LlmError::from(OpenAIError::ApiError)` uses the `code` field.
///
/// This directly tests the `map_openai_api_error` logic end-to-end through
/// the public `From` impl.
mod unit {
    use super::*;
    use async_openai::error::{ApiError, OpenAIError};

    fn api_error(message: &str, r#type: Option<&str>, code: Option<&str>) -> OpenAIError {
        OpenAIError::ApiError(ApiError {
            message: message.to_string(),
            r#type: r#type.map(str::to_string),
            param: None,
            code: code.map(str::to_string),
        })
    }

    // ── code-field driven classification ─────────────────────────────────

    #[test]
    fn test_error_classification_unit_rate_limit_exceeded_code() {
        // code="rate_limit_exceeded" must produce RateLimited regardless of message
        let err = api_error(
            "Rate limit reached for gpt-4.1 on tokens per min",
            Some("tokens"),
            Some("rate_limit_exceeded"),
        );
        let llm_err = LlmError::from(err);
        assert!(
            matches!(llm_err, LlmError::RateLimited(_)),
            "rate_limit_exceeded code must produce RateLimited, got: {}",
            llm_err
        );
    }

    #[test]
    fn test_error_classification_unit_insufficient_quota_code() {
        // code="insufficient_quota" must produce ApiError (quota exhausted — not a transient rate limit)
        let err = api_error(
            "You exceeded your current quota",
            Some("insufficient_quota"),
            Some("insufficient_quota"),
        );
        let llm_err = LlmError::from(err);
        assert!(
            matches!(llm_err, LlmError::ApiError(_)),
            "insufficient_quota code must produce ApiError, got: {}",
            llm_err
        );
    }

    #[test]
    fn test_error_classification_unit_invalid_api_key_code() {
        let err = api_error(
            "Incorrect API key provided",
            Some("authentication_error"),
            Some("invalid_api_key"),
        );
        let llm_err = LlmError::from(err);
        assert!(
            matches!(llm_err, LlmError::AuthError(_)),
            "invalid_api_key code must produce AuthError, got: {}",
            llm_err
        );
    }

    #[test]
    fn test_error_classification_unit_context_length_exceeded_code() {
        let err = api_error(
            "This model's maximum context length is 128000 tokens",
            Some("invalid_request_error"),
            Some("context_length_exceeded"),
        );
        let llm_err = LlmError::from(err);
        assert!(
            matches!(llm_err, LlmError::TokenLimitExceeded { .. }),
            "context_length_exceeded code must produce TokenLimitExceeded, got: {}",
            llm_err
        );
    }

    #[test]
    fn test_error_classification_unit_model_not_found_code() {
        let err = api_error(
            "The model 'gpt-4.1-nonexistent' does not exist",
            Some("invalid_request_error"),
            Some("model_not_found"),
        );
        let llm_err = LlmError::from(err);
        assert!(
            matches!(llm_err, LlmError::ModelNotFound(_)),
            "model_not_found code must produce ModelNotFound, got: {}",
            llm_err
        );
    }

    #[test]
    fn test_error_classification_unit_content_filter_code() {
        let err = api_error(
            "Your request was rejected by the content management policy",
            Some("invalid_request_error"),
            Some("content_filter"),
        );
        let llm_err = LlmError::from(err);
        assert!(
            matches!(llm_err, LlmError::ApiError(_)),
            "content_filter code must produce ApiError, got: {}",
            llm_err
        );
    }

    // ── type-field fallback (code is absent) ──────────────────────────────

    #[test]
    fn test_error_classification_unit_type_tokens_no_code() {
        // type="tokens" with no code field must still produce RateLimited
        let err = api_error("Rate limit reached on tokens per min", Some("tokens"), None);
        let llm_err = LlmError::from(err);
        assert!(
            matches!(llm_err, LlmError::RateLimited(_)),
            "type=tokens without code must produce RateLimited, got: {}",
            llm_err
        );
    }

    #[test]
    fn test_error_classification_unit_type_requests_no_code() {
        let err = api_error(
            "Rate limit reached on requests per min",
            Some("requests"),
            None,
        );
        let llm_err = LlmError::from(err);
        assert!(
            matches!(llm_err, LlmError::RateLimited(_)),
            "type=requests without code must produce RateLimited, got: {}",
            llm_err
        );
    }

    #[test]
    fn test_error_classification_unit_type_server_error_no_code() {
        // type="server_error" must produce ProviderError (retryable via server_backoff)
        let err = api_error(
            "The server had an error processing your request",
            Some("server_error"),
            None,
        );
        let llm_err = LlmError::from(err);
        assert!(
            matches!(llm_err, LlmError::ProviderError(_)),
            "type=server_error must produce ProviderError, got: {}",
            llm_err
        );
    }

    #[test]
    fn test_error_classification_unit_type_auth_error_no_code() {
        let err = api_error("Invalid authentication", Some("authentication_error"), None);
        let llm_err = LlmError::from(err);
        assert!(
            matches!(llm_err, LlmError::AuthError(_)),
            "type=authentication_error must produce AuthError, got: {}",
            llm_err
        );
    }

    #[test]
    fn test_error_classification_unit_type_invalid_request_no_code() {
        let err = api_error(
            "Invalid value for 'messages'",
            Some("invalid_request_error"),
            None,
        );
        let llm_err = LlmError::from(err);
        assert!(
            matches!(llm_err, LlmError::InvalidRequest(_)),
            "type=invalid_request_error must produce InvalidRequest, got: {}",
            llm_err
        );
    }

    // ── StreamError variant ───────────────────────────────────────────────

    #[test]
    fn test_error_classification_unit_stream_error_is_provider_error() {
        // A StreamError (e.g. SSE connection lost) must become ProviderError, not panic.
        let err = OpenAIError::StreamError(Box::new(
            async_openai::error::StreamError::EventStream("connection reset".to_string()),
        ));
        let llm_err = LlmError::from(err);
        assert!(
            matches!(llm_err, LlmError::ProviderError(_)),
            "StreamError must produce ProviderError, got: {}",
            llm_err
        );
    }

    // ── InvalidArgument variant ───────────────────────────────────────────

    #[test]
    fn test_error_classification_unit_invalid_argument_is_invalid_request() {
        let err = OpenAIError::InvalidArgument("temperature must be in [0, 2]".to_string());
        let llm_err = LlmError::from(err);
        assert!(
            matches!(llm_err, LlmError::InvalidRequest(_)),
            "InvalidArgument must produce InvalidRequest, got: {}",
            llm_err
        );
    }

    // ── Retry-after parsing ───────────────────────────────────────────────

    fn retry_strategy_for_rate_limited(msg: &str) -> RetryStrategy {
        LlmError::RateLimited(msg.to_string()).retry_strategy()
    }

    #[test]
    fn test_retry_after_parsing_seconds_only() {
        // "Please try again in 29.662s." → ~32 666 ms (29662 + 10% buffer)
        let strategy =
            retry_strategy_for_rate_limited("Rate limit reached. Please try again in 29.662s.");
        match strategy {
            RetryStrategy::WaitAndRetry { wait } => {
                // 29.662s * 1.1 = 32.628s; with ms rounding it's in [32s, 34s]
                assert!(
                    wait >= Duration::from_secs(29),
                    "Wait must be at least the suggested delay"
                );
                assert!(
                    wait <= Duration::from_secs(120),
                    "Wait must be capped at 120s"
                );
                println!("Parsed retry-after (29.662s): {:?}", wait);
            }
            other => panic!("Expected WaitAndRetry, got {:?}", other),
        }
    }

    #[test]
    fn test_retry_after_parsing_minutes_and_seconds() {
        // "try again in 1m23.5s" → 83.5s * 1.1 = 91.85s → 91 850 ms
        let strategy = retry_strategy_for_rate_limited(
            "You are being rate limited. Please try again in 1m23.5s.",
        );
        match strategy {
            RetryStrategy::WaitAndRetry { wait } => {
                assert!(
                    wait >= Duration::from_secs(83),
                    "1m23.5s wait must be at least 83s"
                );
                assert!(
                    wait <= Duration::from_secs(120),
                    "Wait must be capped at 120s"
                );
                println!("Parsed retry-after (1m23.5s): {:?}", wait);
            }
            other => panic!("Expected WaitAndRetry, got {:?}", other),
        }
    }

    #[test]
    fn test_retry_after_parsing_minutes_only() {
        // "try again in 2m" → 120s * 1.1 = 132s → capped to 120s
        let strategy = retry_strategy_for_rate_limited("Rate limited. Please try again in 2m.");
        match strategy {
            RetryStrategy::WaitAndRetry { wait } => {
                assert_eq!(
                    wait,
                    Duration::from_secs(120),
                    "2m should be capped to 120s"
                );
                println!("Parsed retry-after capped (2m): {:?}", wait);
            }
            other => panic!("Expected WaitAndRetry, got {:?}", other),
        }
    }

    #[test]
    fn test_retry_after_parsing_no_hint_uses_default() {
        // Message without a time hint → 60s default
        let strategy = retry_strategy_for_rate_limited("Rate limit reached for gpt-4.1.");
        match strategy {
            RetryStrategy::WaitAndRetry { wait } => {
                assert_eq!(
                    wait,
                    Duration::from_secs(60),
                    "No hint must fall back to 60s default"
                );
                println!("Default retry-after: {:?}", wait);
            }
            other => panic!("Expected WaitAndRetry, got {:?}", other),
        }
    }

    // ── Retry strategy correctness ────────────────────────────────────────

    #[test]
    fn test_retry_strategy_rate_limited_should_retry() {
        let e = LlmError::RateLimited("rate limit".to_string());
        assert!(e.retry_strategy().should_retry(), "RateLimited must retry");
    }

    #[test]
    fn test_retry_strategy_auth_error_no_retry() {
        let e = LlmError::AuthError("invalid key".to_string());
        assert!(
            !e.retry_strategy().should_retry(),
            "AuthError must not retry"
        );
    }

    #[test]
    fn test_retry_strategy_model_not_found_no_retry() {
        let e = LlmError::ModelNotFound("gpt-x".to_string());
        assert!(
            !e.retry_strategy().should_retry(),
            "ModelNotFound must not retry"
        );
    }

    #[test]
    fn test_retry_strategy_invalid_request_no_retry() {
        let e = LlmError::InvalidRequest("bad param".to_string());
        assert!(
            !e.retry_strategy().should_retry(),
            "InvalidRequest must not retry"
        );
    }

    #[test]
    fn test_retry_strategy_network_error_should_retry() {
        let e = LlmError::NetworkError("connection reset".to_string());
        assert!(e.retry_strategy().should_retry(), "NetworkError must retry");
    }

    #[test]
    fn test_retry_strategy_timeout_should_retry() {
        let e = LlmError::Timeout;
        assert!(e.retry_strategy().should_retry(), "Timeout must retry");
    }

    #[test]
    fn test_retry_strategy_provider_error_should_retry() {
        let e = LlmError::ProviderError("server error".to_string());
        assert!(
            e.retry_strategy().should_retry(),
            "ProviderError must retry (server_backoff)"
        );
    }

    #[test]
    fn test_retry_strategy_token_limit_exceeded_reduce_context() {
        let e = LlmError::TokenLimitExceeded {
            max: 4096,
            got: 5000,
        };
        assert!(
            matches!(e.retry_strategy(), RetryStrategy::ReduceContext),
            "TokenLimitExceeded must use ReduceContext strategy"
        );
    }
}
