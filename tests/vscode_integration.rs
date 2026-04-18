//! Integration tests for the VSCode Copilot provider.
//!
//! Direct mode uses the local VS Code Copilot authentication cache. Proxy mode
//! remains available for legacy copilot-api setups.

use edgequake_llm::{LLMProvider, VsCodeCopilotProvider};

fn is_verified_global_rate_limit(err: &impl std::fmt::Display) -> bool {
    let msg = err.to_string().to_ascii_lowercase();
    msg.contains("user_weekly_rate_limited")
        || msg.contains("user_global_rate_limited")
        || msg.contains("global-chat:global-cogs-7-day-key")
}

#[tokio::test]
#[ignore] // Run with: cargo test --ignored test_vscode_health_check
async fn test_vscode_health_check() {
    let provider = VsCodeCopilotProvider::new().build().unwrap();

    // Test that we can create the provider
    assert_eq!(provider.name(), "vscode-copilot");
    assert_eq!(provider.model(), "gpt-5-mini");
    assert!(provider.supports_streaming());
    assert!(provider.supports_json_mode());
}

#[tokio::test]
#[ignore] // Run with: cargo test --ignored test_vscode_simple_completion
async fn test_vscode_simple_completion() {
    let provider = VsCodeCopilotProvider::new().build().unwrap();

    let response = provider.complete("What is 2 + 2?").await;

    match response {
        Ok(res) => {
            println!("Response: {}", res.content);
            println!("Model: {}", res.model);
            println!(
                "Tokens: prompt={}, completion={}",
                res.prompt_tokens, res.completion_tokens
            );
            assert!(!res.content.is_empty());
            assert!(res.content.contains("4") || res.content.contains("four"));
        }
        Err(e) => {
            if is_verified_global_rate_limit(&e) {
                eprintln!(
                    "Skipping live completion due to upstream Copilot rate limit: {}",
                    e
                );
                return;
            }
            panic!("Request failed: {}", e);
        }
    }
}

#[tokio::test]
#[ignore] // Run with: cargo test --ignored test_vscode_chat
async fn test_vscode_chat() {
    use edgequake_llm::traits::ChatMessage;

    let provider = VsCodeCopilotProvider::new()
        .model("copilot/gpt-4.1")
        .build()
        .unwrap();

    let messages = vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("Hello!"),
    ];

    let response = provider.chat(&messages, None).await;

    match response {
        Ok(res) => {
            println!("Chat response: {}", res.content);
            assert!(!res.content.is_empty());
            assert!(res.total_tokens > 0);
        }
        Err(e) => {
            if is_verified_global_rate_limit(&e) {
                eprintln!(
                    "Skipping live gpt-4.1 chat due to upstream Copilot rate limit: {}",
                    e
                );
                return;
            }
            panic!("Chat request failed: {}", e);
        }
    }
}

#[tokio::test]
#[ignore] // Run with: cargo test --ignored test_vscode_streaming
async fn test_vscode_streaming() {
    use futures::stream::StreamExt;

    let provider = VsCodeCopilotProvider::new().build().unwrap();

    let mut stream = match provider.stream("Count to 5").await {
        Ok(stream) => stream,
        Err(e) => {
            if is_verified_global_rate_limit(&e) {
                eprintln!(
                    "Skipping live streaming due to upstream Copilot rate limit: {}",
                    e
                );
                return;
            }
            panic!("Failed to start stream: {}", e);
        }
    };
    let mut chunks = Vec::new();

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(text) => {
                print!("{}", text);
                chunks.push(text);
            }
            Err(e) => {
                panic!("Stream error: {}", e);
            }
        }
    }

    println!(); // New line after streaming
    let full_text = chunks.join("");
    assert!(!full_text.is_empty());
}

#[tokio::test]
#[ignore] // Run with: cargo test --ignored test_vscode_with_options
async fn test_vscode_with_options() {
    use edgequake_llm::traits::CompletionOptions;

    let provider = VsCodeCopilotProvider::new().build().unwrap();

    let options = CompletionOptions {
        temperature: Some(0.7),
        max_tokens: Some(100),
        ..Default::default()
    };

    let response = provider
        .complete_with_options("Explain Rust ownership briefly", &options)
        .await;

    match response {
        Ok(res) => {
            println!("Response: {}", res.content);
            assert!(!res.content.is_empty());
        }
        Err(e) => {
            if is_verified_global_rate_limit(&e) {
                eprintln!(
                    "Skipping live completion due to upstream Copilot rate limit: {}",
                    e
                );
                return;
            }
            panic!("Request failed: {}", e);
        }
    }
}

#[tokio::test]
#[ignore] // Run with: cargo test --ignored test_vscode_error_handling
async fn test_vscode_error_handling() {
    // Test with invalid proxy URL
    let provider = VsCodeCopilotProvider::new()
        .proxy_url("http://localhost:9999")
        .build()
        .unwrap();

    let response = provider.complete("Test").await;

    assert!(response.is_err());
    let err = response.unwrap_err();
    println!("Expected error: {}", err);
}
