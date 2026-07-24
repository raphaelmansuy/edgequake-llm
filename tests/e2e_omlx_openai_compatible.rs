//! Live e2e against a running oMLX server.
//!
//! ```bash
//! OMLX_E2E=1 cargo test --test e2e_omlx_openai_compatible -- --ignored --nocapture
//! ```

use edgequake_llm::{ChatMessage, CompletionOptions, LLMProvider, OmlxProvider, ToolDefinition};
use serde_json::json;

fn e2e_enabled() -> bool {
    matches!(
        std::env::var("OMLX_E2E").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

fn skip_if_disabled() -> bool {
    if e2e_enabled() {
        return false;
    }
    eprintln!("skip: set OMLX_E2E=1 with oMLX serving on OMLX_HOST (default :8000)");
    true
}

#[tokio::test]
#[ignore = "requires OMLX_E2E=1 and running oMLX"]
async fn e2e_omlx_list_models() {
    if skip_if_disabled() {
        return;
    }
    let provider = OmlxProvider::from_env().expect("build omlx");
    let models = provider.list_models().await.expect("list models");
    eprintln!("omlx models: {models:?}");
    // Empty is allowed if no models loaded; health still reached the API.
}

#[tokio::test]
#[ignore = "requires OMLX_E2E=1 and running oMLX"]
async fn e2e_omlx_chat() {
    if skip_if_disabled() {
        return;
    }
    let model = std::env::var("OMLX_MODEL").unwrap_or_else(|_| "default".into());
    let provider = OmlxProvider::from_env_with_model(&model).expect("build omlx");
    let models = provider.list_models().await.unwrap_or_default();
    let model = if models.iter().any(|m| m == &model) {
        model
    } else {
        models
            .first()
            .cloned()
            .expect("need at least one loaded model on oMLX")
    };
    let provider = OmlxProvider::from_env_with_model(&model).expect("rebuild");
    let resp = provider
        .chat(
            &[ChatMessage::user(
                "Reply with exactly the word pong and nothing else.",
            )],
            Some(&CompletionOptions {
                max_tokens: Some(32),
                temperature: Some(0.0),
                ..Default::default()
            }),
        )
        .await
        .expect("chat");
    eprintln!("omlx chat: {}", resp.content);
    assert!(!resp.content.trim().is_empty());
}

#[tokio::test]
#[ignore = "requires OMLX_E2E=1, tool-capable model, running oMLX"]
async fn e2e_omlx_tools() {
    if skip_if_disabled() {
        return;
    }
    let model = std::env::var("OMLX_MODEL").unwrap_or_else(|_| "default".into());
    let provider = OmlxProvider::from_env_with_model(&model).expect("build");
    let models = provider.list_models().await.unwrap_or_default();
    let model = models.first().cloned().unwrap_or(model);
    let provider = OmlxProvider::from_env_with_model(&model).expect("rebuild");

    let tools = vec![ToolDefinition::function(
        "get_time",
        "Get the current time in a city.",
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" }
            },
            "required": ["city"]
        }),
    )];

    let resp = provider
        .chat_with_tools(
            &[ChatMessage::user(
                "What time is it in Paris? Use the get_time tool.",
            )],
            &tools,
            None,
            Some(&CompletionOptions {
                max_tokens: Some(256),
                temperature: Some(0.0),
                ..Default::default()
            }),
        )
        .await;

    match resp {
        Ok(r) => {
            eprintln!(
                "omlx tools: content={} tool_calls={:?}",
                r.content, r.tool_calls
            );
            // Soft assert: some models may still prose; presence of either is ok for e2e smoke.
            assert!(!r.content.is_empty() || !r.tool_calls.is_empty());
        }
        Err(e) => {
            eprintln!("omlx tools skipped/failed (model may lack tools): {e}");
        }
    }
}
