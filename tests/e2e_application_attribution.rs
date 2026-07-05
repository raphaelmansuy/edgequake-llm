//! E2E tests for application attribution header propagation.
//!
//! Uses local TCP mock servers — no external API keys required.

use edgequake_llm::application_context::{
    ApplicationContext, ApplicationContextBuilder, AttributionPolicy,
};
use edgequake_llm::factory::ProviderFactory;
use edgequake_llm::http::attribution::append_goog_api_client;
use edgequake_llm::model_config::{
    ModelCapabilities, ModelCard, ModelType, ProviderConfig, ProviderType as ConfigProviderType,
};
use edgequake_llm::providers::openai_compatible::OpenAICompatibleProvider;
use edgequake_llm::providers::openrouter::OpenRouterProvider;
use edgequake_llm::traits::{ChatMessage, LLMProvider};
use edgequake_llm::LlmError;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const MOCK_CHAT_RESPONSE: &str = r#"{"id":"chatcmpl-test","object":"chat.completion","created":0,"model":"test","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;

async fn read_http_request(socket: &mut tokio::net::TcpStream) -> String {
    let mut buf = vec![0u8; 8192];
    let n = socket.read(&mut buf).await.expect("read request");
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

async fn write_json_response(socket: &mut tokio::net::TcpStream, body: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    socket
        .write_all(response.as_bytes())
        .await
        .expect("write response");
}

#[tokio::test]
async fn openai_compatible_forwards_client_request_id() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_http_request(&mut socket).await;
        assert!(
            request
                .to_ascii_lowercase()
                .contains("x-client-request-id: req-e2e-1"),
            "expected X-Client-Request-Id header, got:\n{request}"
        );
        write_json_response(&mut socket, MOCK_CHAT_RESPONSE).await;
    });

    let config = ProviderConfig {
        name: "mock-compat".into(),
        display_name: "Mock Compatible".into(),
        provider_type: ConfigProviderType::OpenAICompatible,
        base_url: Some(format!("http://127.0.0.1:{}", addr.port())),
        default_llm_model: Some("test-model".into()),
        api_key: Some("test-key".into()),
        models: vec![ModelCard {
            name: "test-model".into(),
            display_name: "Test".into(),
            model_type: ModelType::Llm,
            capabilities: ModelCapabilities {
                context_length: 8192,
                supports_streaming: true,
                ..Default::default()
            },
            ..Default::default()
        }],
        ..Default::default()
    };

    let ctx = ApplicationContextBuilder::new()
        .app_id("testapp")
        .request_id("req-e2e-1")
        .build()
        .unwrap();

    let provider = OpenAICompatibleProvider::from_config(config)
        .expect("provider")
        .with_application_context(ctx);

    let response = provider
        .chat(&[ChatMessage::user("hi")], None)
        .await
        .expect("chat");
    assert_eq!(response.content, "ok");

    server.await.unwrap();
}

#[tokio::test]
async fn openrouter_forwards_referer_and_title() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_http_request(&mut socket).await;
        let lower = request.to_ascii_lowercase();
        assert!(
            lower.contains("http-referer: https://app.example.com"),
            "expected HTTP-Referer, got:\n{request}"
        );
        assert!(
            lower.contains("x-openrouter-title: my app"),
            "expected X-OpenRouter-Title, got:\n{request}"
        );
        assert!(
            lower.contains("x-title: my app"),
            "expected legacy X-Title, got:\n{request}"
        );
        write_json_response(&mut socket, MOCK_CHAT_RESPONSE).await;
    });

    let ctx = ApplicationContextBuilder::new()
        .app_url("https://app.example.com")
        .app_name("My App")
        .build()
        .unwrap();

    let provider = OpenRouterProvider::new("test-key")
        .with_base_url(format!("http://127.0.0.1:{}", addr.port()))
        .with_model("test-model")
        .with_application_context(ctx);

    let response = provider
        .chat(&[ChatMessage::user("hi")], None)
        .await
        .expect("chat");
    assert_eq!(response.content, "ok");

    server.await.unwrap();
}

#[test]
fn factory_require_app_id_errors_on_vscode_copilot() {
    let ctx = ApplicationContextBuilder::new()
        .app_id("my-backend")
        .build()
        .unwrap();
    let err = ProviderFactory::create_llm_provider_with_context_policy(
        "vscode-copilot",
        "default",
        ctx,
        AttributionPolicy::RequireAppId,
    );
    assert!(matches!(err, Err(LlmError::AttributionError(_))));
}

#[test]
fn gemini_appends_goog_api_client_without_clobbering() {
    let merged = append_goog_api_client(Some("existing/1.0"), "edgequake-app/backend");
    assert_eq!(merged, "existing/1.0 edgequake-app/backend");

    let ctx = ApplicationContext {
        app_id: Some("backend".into()),
        ..Default::default()
    };
    let resolved = edgequake_llm::http::attribution::resolve_attribution(
        edgequake_llm::application_context::AttributionProviderKind::Gemini,
        &ctx,
    );
    let token = resolved.headers.get("x-goog-api-client").expect("token");
    let merged = append_goog_api_client(Some("existing/1.0"), token);
    assert!(merged.contains("existing/1.0"));
    assert!(merged.contains("edgequake-app/backend"));
}
