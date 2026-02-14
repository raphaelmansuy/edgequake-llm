//! E2E tests for OpenAI-compatible provider (Z.ai)
//!
//! These tests require:
//! - ZAI_API_KEY environment variable set
//!
//! Run with: cargo test -p edgequake-llm --test e2e_openai_compatible -- --ignored

use edgequake_llm::model_config::{
    ModelCapabilities, ModelCard, ModelType, ProviderConfig, ProviderType,
};
use edgequake_llm::providers::openai_compatible::OpenAICompatibleProvider;
use edgequake_llm::traits::{ChatMessage, LLMProvider};

fn create_zai_config() -> ProviderConfig {
    let mut headers = std::collections::HashMap::new();
    headers.insert("Accept-Language".to_string(), "en-US,en".to_string());

    ProviderConfig {
        name: "zai".to_string(),
        display_name: "Z.AI Platform".to_string(),
        provider_type: ProviderType::OpenAICompatible,
        api_key_env: Some("ZAI_API_KEY".to_string()),
        base_url: Some("https://api.z.ai/api/paas/v4".to_string()),
        default_llm_model: Some("glm-4.7-flash".to_string()),
        headers,
        supports_thinking: false,
        models: vec![
            ModelCard {
                name: "glm-4.7".to_string(),
                display_name: "GLM-4.7 (Premium)".to_string(),
                model_type: ModelType::Llm,
                capabilities: ModelCapabilities {
                    context_length: 128000,
                    max_output_tokens: 16384,
                    supports_function_calling: true,
                    supports_streaming: true,
                    supports_json_mode: true,
                    ..Default::default()
                },
                ..Default::default()
            },
            ModelCard {
                name: "glm-4.7-flash".to_string(),
                display_name: "GLM-4.7 Flash".to_string(),
                model_type: ModelType::Llm,
                capabilities: ModelCapabilities {
                    context_length: 128000,
                    max_output_tokens: 8192,
                    supports_function_calling: true,
                    supports_streaming: true,
                    supports_json_mode: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        ],
        ..Default::default()
    }
}

#[tokio::test]
#[ignore = "Requires ZAI_API_KEY environment variable"]
async fn test_zai_simple_chat() {
    let config = create_zai_config();
    let provider =
        OpenAICompatibleProvider::from_config(config).expect("Failed to create Z.ai provider");

    assert_eq!(LLMProvider::name(&provider), "zai");
    assert_eq!(LLMProvider::model(&provider), "glm-4.7-flash");
    assert_eq!(provider.max_context_length(), 128000);

    // Simple chat test
    let messages = vec![ChatMessage::user(
        "What is 2 + 2? Reply with just the number.",
    )];

    let response = provider.chat(&messages, None).await;

    match response {
        Ok(resp) => {
            println!("Response: {}", resp.content);
            println!(
                "Tokens: {} prompt, {} completion",
                resp.prompt_tokens, resp.completion_tokens
            );
            assert!(!resp.content.is_empty());
            assert!(resp.content.contains("4") || resp.content.to_lowercase().contains("four"));
        }
        Err(e) => {
            panic!("Chat request failed: {}", e);
        }
    }
}

#[tokio::test]
#[ignore = "Requires ZAI_API_KEY environment variable"]
async fn test_zai_programming_riddle() {
    let config = create_zai_config();
    let provider =
        OpenAICompatibleProvider::from_config(config).expect("Failed to create Z.ai provider");

    // Programming riddle
    let messages = vec![
        ChatMessage::system("You are a helpful programming assistant. Answer concisely."),
        ChatMessage::user(
            "Write a Rust function that reverses a string. Just the function, no explanation.",
        ),
    ];

    let response = provider.chat(&messages, None).await;

    match response {
        Ok(resp) => {
            println!("Response: {}", resp.content);
            assert!(!resp.content.is_empty());
            // Should contain Rust code
            assert!(
                resp.content.contains("fn ") || resp.content.contains("pub fn"),
                "Response should contain a Rust function"
            );
            assert!(
                resp.content.contains("reverse") || resp.content.contains("chars"),
                "Response should contain string reversal logic"
            );
        }
        Err(e) => {
            panic!("Programming riddle failed: {}", e);
        }
    }
}

#[tokio::test]
#[ignore = "Requires ZAI_API_KEY environment variable"]
async fn test_zai_with_premium_model() {
    let config = create_zai_config();
    let provider = OpenAICompatibleProvider::from_config(config)
        .expect("Failed to create Z.ai provider")
        .with_model("glm-4.7");

    assert_eq!(LLMProvider::model(&provider), "glm-4.7");
    assert_eq!(provider.max_context_length(), 128000);

    let messages = vec![ChatMessage::user("Hello! What's your name?")];

    let response = provider.chat(&messages, None).await;

    match response {
        Ok(resp) => {
            println!("GLM-4.7 Response: {}", resp.content);
            assert!(!resp.content.is_empty());
        }
        Err(e) => {
            panic!("GLM-4.7 request failed: {}", e);
        }
    }
}

#[tokio::test]
#[ignore = "Requires ZAI_API_KEY environment variable"]
async fn test_zai_function_calling() {
    use edgequake_llm::traits::{ToolChoice, ToolDefinition};

    let config = create_zai_config();
    let provider =
        OpenAICompatibleProvider::from_config(config).expect("Failed to create Z.ai provider");

    // Define a simple tool
    let tools = vec![ToolDefinition::function(
        "get_weather",
        "Get the current weather in a given location",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }),
    )];

    let messages = vec![ChatMessage::user(
        "What's the weather like in Paris, France?",
    )];

    let response = provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), None)
        .await;

    match response {
        Ok(resp) => {
            println!("Function calling response: {}", resp.content);
            println!("Tool calls: {:?}", resp.tool_calls);
            // Either we get a response or a tool call
            assert!(!resp.content.is_empty() || !resp.tool_calls.is_empty());
        }
        Err(e) => {
            panic!("Function calling failed: {}", e);
        }
    }
}
