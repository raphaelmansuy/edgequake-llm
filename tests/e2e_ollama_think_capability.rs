//! SPEC-113 — Ollama thinking capability gate (wiremock).
//!
//! Covers T-113-10..14, 22: outbound `/api/chat` must honor live capabilities.

use edgequake_llm::providers::ollama_capabilities::{
    capabilities_include_thinking, thinking_support_from_json_capabilities, ThinkingSupport,
};
use edgequake_llm::{ChatMessage, CompletionOptions, LLMProvider, OllamaProvider};
use serde_json::Value;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn chat_ok_body() -> Value {
    serde_json::json!({
        "model": "qwen3-vl:8b",
        "message": { "role": "assistant", "content": "pong" },
        "done": true,
        "prompt_eval_count": 1,
        "eval_count": 1
    })
}

struct CapturingChat {
    bodies: Arc<std::sync::Mutex<Vec<Value>>>,
    hits: AtomicUsize,
}

impl wiremock::Respond for CapturingChat {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        self.hits.fetch_add(1, Ordering::SeqCst);
        if let Ok(v) = serde_json::from_slice::<Value>(&request.body) {
            if let Ok(mut g) = self.bodies.lock() {
                g.push(v);
            }
        }
        ResponseTemplate::new(200).set_body_json(chat_ok_body())
    }
}

struct CountingShow {
    hits: Arc<AtomicUsize>,
    body: Value,
}

impl wiremock::Respond for CountingShow {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        self.hits.fetch_add(1, Ordering::SeqCst);
        ResponseTemplate::new(200).set_body_json(self.body.clone())
    }
}

#[tokio::test]
async fn t113_10_vl_caps_omit_think_on_chat() {
    let server = MockServer::start().await;
    let bodies = Arc::new(std::sync::Mutex::new(Vec::new()));

    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "capabilities": ["completion", "vision"],
            "details": { "family": "qwen3" }
        })))
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(CapturingChat {
            bodies: Arc::clone(&bodies),
            hits: AtomicUsize::new(0),
        })
        .mount(&server)
        .await;

    let provider = OllamaProvider::builder()
        .host(server.uri())
        .model("qwen3-vl:8b")
        .build()
        .unwrap();

    let resp = provider
        .chat(&[ChatMessage::user("ping")], None)
        .await
        .expect("chat should succeed without think");

    assert_eq!(resp.content, "pong");
    let captured = bodies.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert!(
        captured[0].get("think").is_none(),
        "VL fixture must omit think: {}",
        captured[0]
    );
}

#[tokio::test]
async fn t113_11_thinking_caps_auto_sends_think_true() {
    let server = MockServer::start().await;
    let bodies = Arc::new(std::sync::Mutex::new(Vec::new()));

    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "capabilities": ["completion", "tools", "thinking"],
            "details": { "family": "qwen3" }
        })))
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(CapturingChat {
            bodies: Arc::clone(&bodies),
            hits: AtomicUsize::new(0),
        })
        .mount(&server)
        .await;

    let provider = OllamaProvider::builder()
        .host(server.uri())
        .model("qwen3:8b")
        .build()
        .unwrap();

    provider
        .chat(&[ChatMessage::user("ping")], None)
        .await
        .unwrap();

    let captured = bodies.lock().unwrap();
    assert_eq!(captured[0]["think"], true);
}

#[tokio::test]
async fn t113_12_show_500_still_chats_without_think() {
    let server = MockServer::start().await;
    let bodies = Arc::new(std::sync::Mutex::new(Vec::new()));

    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(CapturingChat {
            bodies: Arc::clone(&bodies),
            hits: AtomicUsize::new(0),
        })
        .mount(&server)
        .await;

    let provider = OllamaProvider::builder()
        .host(server.uri())
        .model("qwen3-vl:8b")
        .build()
        .unwrap();

    provider
        .chat(&[ChatMessage::user("ping")], None)
        .await
        .expect("Unknown → omit think → chat ok");

    assert!(bodies.lock().unwrap()[0].get("think").is_none());
}

#[tokio::test]
async fn t113_13_cache_second_lookup_skips_show() {
    let server = MockServer::start().await;
    let show_hits = Arc::new(AtomicUsize::new(0));
    let bodies = Arc::new(std::sync::Mutex::new(Vec::new()));

    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(CountingShow {
            hits: Arc::clone(&show_hits),
            body: serde_json::json!({
                "capabilities": ["completion", "vision"]
            }),
        })
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(CapturingChat {
            bodies: Arc::clone(&bodies),
            hits: AtomicUsize::new(0),
        })
        .mount(&server)
        .await;

    let provider = OllamaProvider::builder()
        .host(server.uri())
        .model("qwen3-vl:8b")
        .build()
        .unwrap();

    for _ in 0..2 {
        provider
            .chat(&[ChatMessage::user("ping")], None)
            .await
            .unwrap();
    }

    assert_eq!(
        show_hits.load(Ordering::SeqCst),
        1,
        "TTL cache must reuse show"
    );
    assert_eq!(provider.capability_cache().show_request_count(), 1);
}

#[tokio::test]
async fn t113_14_different_hosts_do_not_share_cache() {
    let server_a = MockServer::start().await;
    let server_b = MockServer::start().await;

    for (server, caps) in [
        (&server_a, serde_json::json!(["completion", "thinking"])),
        (&server_b, serde_json::json!(["completion", "vision"])),
    ] {
        Mock::given(method("POST"))
            .and(path("/api/show"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "capabilities": caps
            })))
            .mount(server)
            .await;

        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_ok_body()))
            .mount(server)
            .await;
    }

    // Shared cache instance across two providers with different hosts.
    let cache =
        Arc::new(edgequake_llm::providers::ollama_capabilities::OllamaCapabilityCache::new());
    // Build two providers then swap caches via cloning pattern — use resolver directly.
    let client = reqwest::Client::new();
    let resolver =
        edgequake_llm::providers::ollama_capabilities::OllamaCapabilityResolver::with_ttl_timeout(
            Arc::clone(&cache),
            Duration::from_secs(300),
            Duration::from_millis(2000),
        );

    let a = resolver
        .thinking_support(&client, &server_a.uri(), "m")
        .await;
    let b = resolver
        .thinking_support(&client, &server_b.uri(), "m")
        .await;
    assert_eq!(a, ThinkingSupport::Yes);
    assert_eq!(b, ThinkingSupport::No);
    assert_eq!(cache.show_request_count(), 2);
}

#[tokio::test]
async fn t113_22_stream_and_nonstream_honor_gate() {
    let server = MockServer::start().await;
    let bodies = Arc::new(std::sync::Mutex::new(Vec::new()));

    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "capabilities": ["completion", "vision"]
        })))
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(CapturingChat {
            bodies: Arc::clone(&bodies),
            hits: AtomicUsize::new(0),
        })
        .mount(&server)
        .await;

    let provider = OllamaProvider::builder()
        .host(server.uri())
        .model("qwen3-vl:8b")
        .build()
        .unwrap();

    provider
        .chat(
            &[ChatMessage::user("ping")],
            Some(&CompletionOptions::default()),
        )
        .await
        .unwrap();

    // Stream path uses same gate; mock returns a single JSON object (non-NDJSON) which
    // may fail parse — we only need the outbound request captured.
    let _ = provider.stream("ping").await;

    let captured = bodies.lock().unwrap();
    assert!(!captured.is_empty());
    for body in captured.iter() {
        assert!(
            body.get("think").is_none(),
            "stream/non-stream must omit think: {body}"
        );
    }
}

/// Env mode is read per request — changing after build must take effect (no rebuild).
#[tokio::test]
async fn t113_env_mode_live_after_build_force_off() {
    let server = MockServer::start().await;
    let bodies = Arc::new(std::sync::Mutex::new(Vec::new()));

    // Thinking caps — Auto would send think:true unless force_off wins.
    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "capabilities": ["completion", "thinking"]
        })))
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(CapturingChat {
            bodies: Arc::clone(&bodies),
            hits: AtomicUsize::new(0),
        })
        .mount(&server)
        .await;

    let provider = OllamaProvider::builder()
        .host(server.uri())
        .model("qwen3:8b")
        .build()
        .unwrap();

    let prev = std::env::var("EDGEQUAKE_OLLAMA_THINK_CAPABILITY").ok();
    std::env::set_var("EDGEQUAKE_OLLAMA_THINK_CAPABILITY", "force_off");
    let result = provider.chat(&[ChatMessage::user("ping")], None).await;
    match prev {
        Some(v) => std::env::set_var("EDGEQUAKE_OLLAMA_THINK_CAPABILITY", v),
        None => std::env::remove_var("EDGEQUAKE_OLLAMA_THINK_CAPABILITY"),
    }
    result.expect("chat");

    assert!(
        bodies.lock().unwrap()[0].get("think").is_none(),
        "force_off after build must omit think"
    );
}

/// Unknown show failure: second chat after short TTL must re-hit /api/show.
#[tokio::test]
async fn t113_unknown_expires_and_reprobes() {
    let server = MockServer::start().await;
    let show_hits = Arc::new(AtomicUsize::new(0));
    let bodies = Arc::new(std::sync::Mutex::new(Vec::new()));

    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(CountingShow {
            hits: Arc::clone(&show_hits),
            body: serde_json::json!({}), // missing capabilities → Unknown
        })
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(CapturingChat {
            bodies: Arc::clone(&bodies),
            hits: AtomicUsize::new(0),
        })
        .mount(&server)
        .await;

    let provider = OllamaProvider::builder()
        .host(server.uri())
        .model("m")
        .build()
        .unwrap();

    provider
        .chat(&[ChatMessage::user("ping")], None)
        .await
        .unwrap();
    assert_eq!(show_hits.load(Ordering::SeqCst), 1);

    // Within Unknown TTL — no re-probe.
    provider
        .chat(&[ChatMessage::user("ping")], None)
        .await
        .unwrap();
    assert_eq!(show_hits.load(Ordering::SeqCst), 1);

    tokio::time::sleep(Duration::from_millis(5100)).await;

    provider
        .chat(&[ChatMessage::user("ping")], None)
        .await
        .unwrap();
    assert!(
        show_hits.load(Ordering::SeqCst) >= 2,
        "Unknown must expire (~5s) and re-probe"
    );
}

#[test]
fn t113_17_discovery_parse_matches_resolver() {
    let thinking = ["completion", "thinking"];
    let vision = ["completion", "vision"];
    assert_eq!(
        capabilities_include_thinking(&thinking),
        thinking_support_from_json_capabilities(&serde_json::json!(thinking))
            == ThinkingSupport::Yes
    );
    assert_eq!(
        !capabilities_include_thinking(&vision),
        thinking_support_from_json_capabilities(&serde_json::json!(vision)) == ThinkingSupport::No
    );
}

// =============================================================================
// T-113-23 — Live Ollama (requires local daemon + models)
//
// Setup for #369 false-positive class:
//   ollama cp granite4:latest qwen3-fake-vl:test   # caps lack thinking; name has qwen3
// Thinking-capable smoke model:
//   ollama pull deepseek-r1:1.5b
//
//   cargo test --test e2e_ollama_think_capability -- --ignored --nocapture
// =============================================================================

fn live_ollama_host() -> String {
    std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".into())
}

async fn live_ollama_up(host: &str) -> bool {
    let Ok(resp) = reqwest::Client::new()
        .get(format!("{}/api/tags", host.trim_end_matches('/')))
        .timeout(Duration::from_secs(2))
        .send()
        .await
    else {
        return false;
    };
    resp.status().is_success()
}

async fn live_show_caps(host: &str, model: &str) -> Vec<String> {
    let resp = reqwest::Client::new()
        .post(format!("{}/api/show", host.trim_end_matches('/')))
        .json(&serde_json::json!({ "model": model }))
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .expect("show");
    assert!(resp.status().is_success(), "show {model}");
    let body: Value = resp.json().await.expect("json");
    body.get("capabilities")
        .and_then(|c| c.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// #369 class: name contains `qwen3`, caps lack thinking → Auto chat must succeed (omit think).
#[tokio::test]
#[ignore = "live Ollama: ollama cp granite4:latest qwen3-fake-vl:test"]
async fn t113_23_live_qwen3_name_non_thinking_auto_chat() {
    let host = live_ollama_host();
    assert!(
        live_ollama_up(&host).await,
        "Ollama not reachable at {host}"
    );

    let model =
        std::env::var("EDGEQUAKE_T113_VL_MODEL").unwrap_or_else(|_| "qwen3-fake-vl:test".into());
    let caps = live_show_caps(&host, &model).await;
    assert!(
        !capabilities_include_thinking(&caps),
        "{model} must lack thinking for this fixture; caps={caps:?}"
    );

    let provider = OllamaProvider::builder()
        .host(&host)
        .model(&model)
        .build()
        .unwrap();

    let resp = provider
        .chat(
            &[ChatMessage::user("Reply with exactly: OK")],
            None, // Auto
        )
        .await
        .unwrap_or_else(|e| {
            panic!("Auto chat against {model} must succeed (SPEC-113 omit think); err={e}")
        });
    assert!(
        !resp.content.trim().is_empty(),
        "expected non-empty reply from {model}"
    );
    eprintln!(
        "T-113-23 VL-class PASS model={model} caps={caps:?} reply={}",
        resp.content.chars().take(80).collect::<String>()
    );
}

/// Thinking-capable model: Auto may enable think; chat must still succeed.
#[tokio::test]
#[ignore = "live Ollama: ollama pull deepseek-r1:1.5b"]
async fn t113_23_live_thinking_model_auto_chat() {
    let host = live_ollama_host();
    assert!(
        live_ollama_up(&host).await,
        "Ollama not reachable at {host}"
    );

    let model =
        std::env::var("EDGEQUAKE_T113_THINK_MODEL").unwrap_or_else(|_| "deepseek-r1:1.5b".into());
    let caps = live_show_caps(&host, &model).await;
    assert!(
        capabilities_include_thinking(&caps),
        "{model} must include thinking; caps={caps:?}"
    );

    let provider = OllamaProvider::builder()
        .host(&host)
        .model(&model)
        .build()
        .unwrap();

    let resp = provider
        .chat(
            &[ChatMessage::user("Reply with exactly: OK")],
            None, // Auto → think:true when capable
        )
        .await
        .unwrap_or_else(|e| panic!("Auto chat against thinking model {model} failed: {e}"));
    assert!(!resp.content.trim().is_empty());
    eprintln!(
        "T-113-23 thinking PASS model={model} caps={caps:?} thinking_tokens={:?} reply={}",
        resp.thinking_tokens,
        resp.content.chars().take(80).collect::<String>()
    );
}

/// granite4 (no qwen3 in name, no thinking): Auto must omit think and succeed.
#[tokio::test]
#[ignore = "live Ollama: granite4:latest"]
async fn t113_23_live_granite_non_thinking_auto_chat() {
    let host = live_ollama_host();
    assert!(
        live_ollama_up(&host).await,
        "Ollama not reachable at {host}"
    );
    let model = "granite4:latest";
    let caps = live_show_caps(&host, model).await;
    assert!(!capabilities_include_thinking(&caps), "caps={caps:?}");

    let provider = OllamaProvider::builder()
        .host(&host)
        .model(model)
        .build()
        .unwrap();
    let resp = provider
        .chat(&[ChatMessage::user("Reply with exactly: OK")], None)
        .await
        .expect("granite Auto chat");
    assert!(!resp.content.trim().is_empty());
    eprintln!("T-113-23 granite PASS caps={caps:?} reply={}", resp.content);
}
