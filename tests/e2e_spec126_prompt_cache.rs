//! SPEC-126 e2e: provider KV / prompt-cache wire format (August 2026).
//!
//! HTTP mocks only — no vendor keys. A live OpenAI hit test is `#[ignore]`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use edgequake_llm::{
    AnthropicProvider, ChatMessage, CompletionOptions, LLMProvider, OpenAIProvider, OpenAiChatApi,
    OpenRouterProvider,
};
use serde_json::Value;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn chat_ok(cached: u64, write: u64) -> Value {
    serde_json::json!({
        "id": "chatcmpl-spec126",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 1,
            "total_tokens": 1201,
            "prompt_tokens_details": {
                "cached_tokens": cached,
                "cache_write_tokens": write
            }
        }
    })
}

fn anthropic_ok() -> Value {
    serde_json::json!({
        "id": "msg_spec126",
        "type": "message",
        "role": "assistant",
        "content": [{ "type": "text", "text": "ok" }],
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 20,
            "output_tokens": 1,
            "cache_creation_input_tokens": 18,
            "cache_read_input_tokens": 0
        }
    })
}

type JsonResponder = Box<dyn Fn(&Value, usize) -> ResponseTemplate + Send + Sync>;

struct CapturingJson {
    bodies: Arc<std::sync::Mutex<Vec<Value>>>,
    hits: AtomicUsize,
    responder: JsonResponder,
}

impl wiremock::Respond for CapturingJson {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let n = self.hits.fetch_add(1, Ordering::SeqCst);
        let body = serde_json::from_slice::<Value>(&request.body).unwrap_or(Value::Null);
        if let Ok(mut g) = self.bodies.lock() {
            g.push(body.clone());
        }
        (self.responder)(&body, n)
    }
}

fn extract_messages() -> Vec<ChatMessage> {
    vec![
        ChatMessage::system("Stable extract instructions for knowledge graph JSON."),
        ChatMessage::user("chunk-17 unique payload"),
    ]
}

#[tokio::test]
async fn e2e_openai_native_sends_explicit_breakpoint_and_key() {
    let server = MockServer::start().await;
    let bodies = Arc::new(std::sync::Mutex::new(Vec::new()));
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(CapturingJson {
            bodies: Arc::clone(&bodies),
            hits: AtomicUsize::new(0),
            responder: Box::new(|_, _| ResponseTemplate::new(200).set_body_json(chat_ok(0, 1100))),
        })
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new("sk-test")
        .with_base_url(format!("{}/v1", server.uri()))
        .with_model("eq-spec126-native");
    assert!(OpenAiChatApi::Native.allows_explicit_breakpoints());

    let opts = CompletionOptions::default().with_role_cache("extract", &provider);
    let resp = provider
        .chat(&extract_messages(), Some(&opts))
        .await
        .expect("native chat");
    assert_eq!(resp.content, "ok");

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 1);
    let body = &bodies[0];
    assert_eq!(body["prompt_cache_options"]["mode"], "explicit");
    assert_eq!(body["prompt_cache_options"]["ttl"], "30m");
    assert!(
        body["prompt_cache_key"]
            .as_str()
            .unwrap()
            .starts_with("eq:extract:"),
        "key={}",
        body["prompt_cache_key"]
    );
    assert_eq!(
        body["messages"][0]["content"][0]["prompt_cache_breakpoint"]["mode"],
        "explicit"
    );
    assert!(body["messages"][1]["content"].is_string());
}

#[tokio::test]
async fn e2e_openai_compatible_omits_gpt56_fields_keeps_key() {
    let server = MockServer::start().await;
    let bodies = Arc::new(std::sync::Mutex::new(Vec::new()));
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(CapturingJson {
            bodies: Arc::clone(&bodies),
            hits: AtomicUsize::new(0),
            responder: Box::new(|_, _| ResponseTemplate::new(200).set_body_json(chat_ok(0, 0))),
        })
        .mount(&server)
        .await;

    let provider = OpenAIProvider::compatible("sk-test", format!("{}/v1", server.uri()))
        .with_model("mistral-small-latest");
    let opts = CompletionOptions::default().with_role_cache("extract", &provider);
    provider
        .chat(&extract_messages(), Some(&opts))
        .await
        .expect("compat chat");

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 1);
    assert!(
        bodies[0].get("prompt_cache_options").is_none(),
        "compatible hosts must not receive GPT-5.6 fields: {}",
        bodies[0]
    );
    assert!(bodies[0].get("prompt_cache_key").is_some());
}

#[tokio::test]
async fn e2e_openai_native_retries_without_explicit_fields_on_param_400() {
    let server = MockServer::start().await;
    let bodies = Arc::new(std::sync::Mutex::new(Vec::new()));
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(CapturingJson {
            bodies: Arc::clone(&bodies),
            hits: AtomicUsize::new(0),
            responder: Box::new(|body, _| {
                if body.get("prompt_cache_options").is_some() {
                    ResponseTemplate::new(400).set_body_json(serde_json::json!({
                        "error": {
                            "message": "Unknown parameter",
                            "type": "invalid_request_error",
                            "param": "prompt_cache_options",
                            "code": "unknown_parameter"
                        }
                    }))
                } else {
                    ResponseTemplate::new(200).set_body_json(chat_ok(0, 0))
                }
            }),
        })
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new("sk-test")
        .with_base_url(format!("{}/v1", server.uri()))
        .with_model("eq-spec126-unaware");
    let opts = CompletionOptions::default().with_role_cache("extract", &provider);
    let resp = provider
        .chat(&extract_messages(), Some(&opts))
        .await
        .expect("retry after 400");
    assert_eq!(resp.content, "ok");

    let bodies = bodies.lock().unwrap();
    assert!(
        bodies.len() >= 2,
        "expected explicit attempt then unmarked retry, got {}",
        bodies.len()
    );
    assert!(bodies[0].get("prompt_cache_options").is_some());
    assert!(bodies[1].get("prompt_cache_options").is_none());
}

#[tokio::test]
async fn e2e_anthropic_marks_system_cache_control() {
    let server = MockServer::start().await;
    let bodies = Arc::new(std::sync::Mutex::new(Vec::new()));
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(CapturingJson {
            bodies: Arc::clone(&bodies),
            hits: AtomicUsize::new(0),
            responder: Box::new(|_, _| ResponseTemplate::new(200).set_body_json(anthropic_ok())),
        })
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new("sk-ant-test")
        .with_base_url(server.uri())
        .with_model("claude-sonnet-4-6");
    let opts = CompletionOptions::default().with_role_cache("extract", &provider);
    provider
        .chat(&extract_messages(), Some(&opts))
        .await
        .expect("anthropic chat");

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 1);
    let _sys = bodies[0]["system"].as_array();
    let raw = bodies[0].to_string();
    assert!(
        raw.contains("cache_control") && raw.contains("ephemeral"),
        "Anthropic request must mark cache_control: {raw}"
    );
}

#[tokio::test]
async fn e2e_openrouter_sends_session_id_key_and_cache_control() {
    let server = MockServer::start().await;
    let bodies = Arc::new(std::sync::Mutex::new(Vec::new()));
    Mock::given(method("POST"))
        .and(path("/api/v1/chat/completions"))
        .respond_with(CapturingJson {
            bodies: Arc::clone(&bodies),
            hits: AtomicUsize::new(0),
            responder: Box::new(|_, _| ResponseTemplate::new(200).set_body_json(chat_ok(0, 0))),
        })
        .mount(&server)
        .await;

    let provider = OpenRouterProvider::new("sk-or-test")
        .with_base_url(format!("{}/api/v1", server.uri()))
        .with_model("anthropic/claude-sonnet-4");
    let opts = CompletionOptions::default().with_role_cache("extract", &provider);
    provider
        .chat(&extract_messages(), Some(&opts))
        .await
        .expect("openrouter chat");

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 1);
    let body = &bodies[0];
    let key = body["prompt_cache_key"].as_str().expect("prompt_cache_key");
    assert_eq!(body["session_id"].as_str(), Some(key));
    let raw = body.to_string();
    assert!(
        raw.contains("cache_control"),
        "OpenRouter must forward cache_control for upstream translation: {raw}"
    );
}

/// Live proof: second extract-style call reports `cached_tokens` > 0.
///
/// Run:
/// `EDGEQUAKE_LIVE_PROMPT_CACHE=1 OPENAI_API_KEY=... cargo test -p edgequake-llm --test e2e_spec126_prompt_cache live_openai -- --ignored --nocapture`
#[tokio::test]
#[ignore = "live OpenAI; set EDGEQUAKE_LIVE_PROMPT_CACHE=1"]
async fn live_openai_second_extract_reads_cached_tokens() {
    let enabled = std::env::var("EDGEQUAKE_LIVE_PROMPT_CACHE").unwrap_or_default();
    assert!(
        matches!(enabled.to_ascii_lowercase().as_str(), "1" | "true" | "yes"),
        "refusing to spend tokens without EDGEQUAKE_LIVE_PROMPT_CACHE=1"
    );
    let provider = OpenAIProvider::from_env().expect("OPENAI_API_KEY");
    let system = format!(
        "You extract entities as JSON. Follow the schema strictly.\n{}",
        "Instruction padding for the 1024-token GPT-5.6 cache floor. ".repeat(80)
    );
    let mut opts = CompletionOptions::default().with_role_cache("extract", &provider);
    opts.temperature = Some(0.0);
    let first = provider
        .chat(
            &[
                ChatMessage::system(&system),
                ChatMessage::user("Chunk A: Alice works at Acme."),
            ],
            Some(&opts),
        )
        .await
        .expect("first live chat");
    let second = provider
        .chat(
            &[
                ChatMessage::system(&system),
                ChatMessage::user("Chunk B: Bob works at Globex."),
            ],
            Some(&opts),
        )
        .await
        .expect("second live chat");
    eprintln!(
        "live cache: first hit={:?} write={:?} second hit={:?} write={:?}",
        first.cache_hit_tokens,
        first.cache_write_tokens,
        second.cache_hit_tokens,
        second.cache_write_tokens
    );
    assert!(
        second.cache_hit_tokens.unwrap_or(0) > 0,
        "second extract-style call must read cached prefix tokens (got hit={:?} write={:?})",
        second.cache_hit_tokens,
        second.cache_write_tokens
    );
}
