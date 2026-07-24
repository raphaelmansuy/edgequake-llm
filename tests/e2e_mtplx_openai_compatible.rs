//! Live e2e against a running MTPLX server.
//!
//! ```bash
//! mtplx quickstart --host 127.0.0.1 --port 8000
//! MTPLX_E2E=1 cargo test --test e2e_mtplx_openai_compatible -- --ignored --nocapture
//! ```

use edgequake_llm::{
    list_cached_model_ids, resolve_mtplx_runtime_config, ChatMessage, CompletionOptions,
    LLMProvider, MtplxProvider, ToolDefinition,
};
use serde_json::json;

fn e2e_enabled() -> bool {
    matches!(
        std::env::var("MTPLX_E2E").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

fn skip_if_disabled() -> bool {
    if e2e_enabled() {
        return false;
    }
    eprintln!("skip: set MTPLX_E2E=1 with MTPLX serving (mtplx quickstart)");
    true
}

#[tokio::test]
#[ignore = "requires MTPLX_E2E=1 and running MTPLX"]
async fn e2e_mtplx_list_models() {
    if skip_if_disabled() {
        return;
    }
    let cfg = resolve_mtplx_runtime_config();
    eprintln!("mtplx host={}", cfg.host);
    let provider = MtplxProvider::from_env().expect("build mtplx");
    let models = provider.list_models().await.expect("list models");
    eprintln!("mtplx models ({}): {:?}", models.len(), models);
    let cached = list_cached_model_ids();
    eprintln!("cached models: {cached:?}");
}

#[tokio::test]
#[ignore = "requires MTPLX_E2E=1 and running MTPLX"]
async fn e2e_mtplx_chat() {
    if skip_if_disabled() {
        return;
    }
    let provider = MtplxProvider::from_env().expect("build");
    let models = provider.list_models().await.unwrap_or_default();
    let model = std::env::var("MTPLX_MODEL")
        .ok()
        .filter(|m| !m.is_empty())
        .or_else(|| models.first().cloned())
        .unwrap_or_else(|| "default".into());
    let provider = MtplxProvider::from_env_with_model(&model).expect("rebuild");
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
    eprintln!("mtplx chat: {}", resp.content);
    assert!(!resp.content.trim().is_empty());
}

#[tokio::test]
#[ignore = "requires MTPLX_E2E=1, tool-capable model, running MTPLX"]
async fn e2e_mtplx_tools() {
    if skip_if_disabled() {
        return;
    }
    let provider = MtplxProvider::from_env().expect("build");
    let models = provider.list_models().await.unwrap_or_default();
    let model = models.first().cloned().unwrap_or_else(|| "default".into());
    let provider = MtplxProvider::from_env_with_model(&model).expect("rebuild");
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
    match provider
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
        .await
    {
        Ok(r) => {
            eprintln!(
                "mtplx tools: content={} tool_calls={:?}",
                r.content, r.tool_calls
            );
            assert!(!r.content.is_empty() || !r.tool_calls.is_empty());
        }
        Err(e) => eprintln!("mtplx tools soft-fail: {e}"),
    }
}
