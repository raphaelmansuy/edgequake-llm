//! Integration test for VSCode Copilot provider.
//!
//! This test requires copilot-api proxy to be running on localhost:4141

use edgequake_llm::{LLMProvider, VsCodeCopilotProvider};

#[tokio::test]
#[ignore] // Run with: cargo test --ignored test_vscode_health_check
async fn test_vscode_health_check() {
    let provider = VsCodeCopilotProvider::new().build().unwrap();

    // Test that we can create the provider
    assert_eq!(provider.name(), "vscode-copilot");
    assert_eq!(provider.model(), "gpt-4o-mini");
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
            panic!("Request failed: {}", e);
        }
    }
}

#[tokio::test]
#[ignore] // Run with: cargo test --ignored test_vscode_chat
async fn test_vscode_chat() {
    use edgequake_llm::traits::ChatMessage;

    let provider = VsCodeCopilotProvider::new()
        .model("gpt-4o-mini")
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
            panic!("Chat request failed: {}", e);
        }
    }
}

#[tokio::test]
#[ignore] // Run with: cargo test --ignored test_vscode_streaming
async fn test_vscode_streaming() {
    use futures::stream::StreamExt;

    let provider = VsCodeCopilotProvider::new().build().unwrap();

    let mut stream = provider.stream("Count to 5").await.unwrap();
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
