//! SPEC-131 e2e: omit-temperature / omit-effort / Responses API (wiremock).
//!
//! HTTP mocks only — no vendor keys.
//!
//! `ENV_LOCK` serializes process-global env mutation across async tests; holding
//! the std mutex across `.await` is intentional and cannot deadlock these mocks.

#![allow(clippy::await_holding_lock)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

use edgequake_llm::{
    ApiFormat, ChatMessage, CompletionOptions, LLMProvider, OpenAIProvider, ENV_API_FORMAT,
    ENV_OMIT_REASONING_EFFORT, ENV_OMIT_TEMPERATURE,
};
use futures::StreamExt;
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

/// Process-wide lock so SPEC-131 env knobs do not race across parallel tests.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn chat_ok() -> Value {
    json!({
        "id": "chatcmpl-spec131",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    })
}

fn responses_ok() -> Value {
    json!({
        "id": "resp_spec131",
        "object": "response",
        "model": "test",
        "status": "completed",
        "output": [
            { "type": "reasoning", "summary": [] },
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{ "type": "output_text", "text": "resp-ok" }]
            }
        ],
        "usage": { "input_tokens": 2, "output_tokens": 1 }
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

fn clear_spec131_env() {
    std::env::remove_var(ENV_OMIT_TEMPERATURE);
    std::env::remove_var(ENV_OMIT_REASONING_EFFORT);
    std::env::remove_var(ENV_API_FORMAT);
}

/// E2E-131-01: omit-temp → Chat Completions body lacks temperature.
#[tokio::test]
async fn e2e_131_01_omit_temperature_chat_completions() {
    let _guard = ENV_LOCK.lock().unwrap();
    clear_spec131_env();
    std::env::set_var(ENV_OMIT_TEMPERATURE, "true");

    let server = MockServer::start().await;
    let bodies = Arc::new(std::sync::Mutex::new(Vec::new()));
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(CapturingJson {
            bodies: Arc::clone(&bodies),
            hits: AtomicUsize::new(0),
            responder: Box::new(|_, _| ResponseTemplate::new(200).set_body_json(chat_ok())),
        })
        .mount(&server)
        .await;

    let provider = OpenAIProvider::compatible("sk-test", format!("{}/v1", server.uri()))
        .with_model("google.gemma-4-31b");
    let opts = CompletionOptions {
        temperature: Some(0.0),
        max_tokens: Some(16),
        ..Default::default()
    };
    provider
        .chat(&[ChatMessage::user("hi")], Some(&opts))
        .await
        .expect("chat");

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 1);
    assert!(
        bodies[0].get("temperature").is_none(),
        "temperature must be omitted: {}",
        bodies[0]
    );
    clear_spec131_env();
}

/// E2E-131-02: gpt-5-nano with None temperature lacks field without omit env.
#[tokio::test]
async fn e2e_131_02_gpt5_none_temperature_without_omit_env() {
    let _guard = ENV_LOCK.lock().unwrap();
    clear_spec131_env();
    let server = MockServer::start().await;
    let bodies = Arc::new(std::sync::Mutex::new(Vec::new()));
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(CapturingJson {
            bodies: Arc::clone(&bodies),
            hits: AtomicUsize::new(0),
            responder: Box::new(|_, _| ResponseTemplate::new(200).set_body_json(chat_ok())),
        })
        .mount(&server)
        .await;

    let provider = OpenAIProvider::compatible("sk-test", format!("{}/v1", server.uri()))
        .with_model("gpt-5-nano");
    let opts = CompletionOptions {
        temperature: None,
        ..Default::default()
    };
    provider
        .chat(&[ChatMessage::user("hi")], Some(&opts))
        .await
        .expect("chat");
    let bodies = bodies.lock().unwrap();
    assert!(bodies[0].get("temperature").is_none());
}

/// E2E-131-03: omit-effort → no reasoning_effort on wire.
#[tokio::test]
async fn e2e_131_03_omit_reasoning_effort() {
    let _guard = ENV_LOCK.lock().unwrap();
    clear_spec131_env();
    std::env::set_var(ENV_OMIT_REASONING_EFFORT, "1");

    let server = MockServer::start().await;
    let bodies = Arc::new(std::sync::Mutex::new(Vec::new()));
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(CapturingJson {
            bodies: Arc::clone(&bodies),
            hits: AtomicUsize::new(0),
            responder: Box::new(|_, _| ResponseTemplate::new(200).set_body_json(chat_ok())),
        })
        .mount(&server)
        .await;

    let provider = OpenAIProvider::compatible("sk-test", format!("{}/v1", server.uri()))
        .with_model("gpt-5-mini");
    let opts = CompletionOptions {
        reasoning_effort: Some("high".into()),
        ..Default::default()
    };
    provider
        .chat(&[ChatMessage::user("hi")], Some(&opts))
        .await
        .expect("chat");
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].get("reasoning_effort").is_none(),
        "reasoning_effort must be omitted: {}",
        bodies[0]
    );
    clear_spec131_env();
}

/// E2E-131-04 / E2E-131-08: Responses path + store:false + skip reasoning items.
#[tokio::test]
async fn e2e_131_04_responses_store_false_and_text() {
    let _guard = ENV_LOCK.lock().unwrap();
    clear_spec131_env();
    std::env::set_var(ENV_API_FORMAT, "responses");
    assert!(ApiFormat::from_env().unwrap().is_responses());

    let server = MockServer::start().await;
    let bodies = Arc::new(std::sync::Mutex::new(Vec::new()));
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(CapturingJson {
            bodies: Arc::clone(&bodies),
            hits: AtomicUsize::new(0),
            responder: Box::new(|_, _| ResponseTemplate::new(200).set_body_json(responses_ok())),
        })
        .mount(&server)
        .await;

    let provider = OpenAIProvider::compatible("sk-test", format!("{}/v1", server.uri()))
        .with_model("openai.gpt-5.6-luna");
    let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
    let opts = CompletionOptions {
        response_format: Some("json_schema".into()),
        response_schema: Some(schema),
        temperature: Some(0.0),
        ..Default::default()
    };
    std::env::set_var(ENV_OMIT_TEMPERATURE, "true");
    let resp = provider
        .chat(&[ChatMessage::user("extract")], Some(&opts))
        .await
        .expect("responses chat");
    assert_eq!(resp.content, "resp-ok");

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 1);
    assert_eq!(bodies[0]["store"], false);
    assert_eq!(bodies[0]["text"]["format"]["type"], "json_schema");
    assert!(bodies[0].get("temperature").is_none());
    clear_spec131_env();
}

/// E2E-131-05: Responses SSE deltas → content stream.
#[tokio::test]
async fn e2e_131_05_responses_stream_deltas() {
    let _guard = ENV_LOCK.lock().unwrap();
    clear_spec131_env();
    std::env::set_var(ENV_API_FORMAT, "responses");

    let server = MockServer::start().await;
    let sse = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hel\"}\n\n\
               data: {\"type\":\"response.output_text.delta\",\"delta\":\"lo\"}\n\n\
               data: {\"type\":\"response.completed\"}\n\n";
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;

    let provider = OpenAIProvider::compatible("sk-test", format!("{}/v1", server.uri()))
        .with_model("test-model");
    let mut stream = provider.stream("hi").await.expect("stream");
    let mut acc = String::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(s) => acc.push_str(&s),
            Err(e) => {
                // Some EventSource impls error after completed; accept partial text.
                if acc.is_empty() {
                    panic!("chunk: {e}");
                }
                break;
            }
        }
    }
    assert_eq!(acc, "Hello");
    clear_spec131_env();
}
