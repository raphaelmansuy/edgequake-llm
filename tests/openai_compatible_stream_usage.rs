//! Streaming `Finished` chunk of the OpenAI-compatible provider (wiremock).
//!
//! HTTP mocks only — no vendor keys.

use edgequake_llm::model_config::{ProviderConfig, ProviderType};
use edgequake_llm::providers::openai_compatible::OpenAICompatibleProvider;
use edgequake_llm::traits::{ChatMessage, LLMProvider, StreamChunk};
use futures::StreamExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn stream_chunks(sse_body: &str) -> Vec<StreamChunk> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = OpenAICompatibleProvider::from_config(ProviderConfig {
        name: "mock".to_string(),
        display_name: "Mock".to_string(),
        provider_type: ProviderType::OpenAICompatible,
        api_key: Some("sk-test".to_string()),
        base_url: Some(format!("{}/v1", server.uri())),
        default_llm_model: Some("test-model".to_string()),
        ..Default::default()
    })
    .expect("provider");

    let stream = provider
        .chat_with_tools_stream(&[ChatMessage::user("hi")], &[], None, None)
        .await
        .expect("stream");
    stream
        .map(|chunk| chunk.expect("chunk"))
        .collect::<Vec<_>>()
        .await
}

/// Finish reason and (prompt, completion, cache hit) tokens of each `Finished` chunk.
type FinishedSummary = (String, Option<(usize, usize, Option<usize>)>);

fn finished(chunks: &[StreamChunk]) -> Vec<FinishedSummary> {
    chunks
        .iter()
        .filter_map(|chunk| match chunk {
            StreamChunk::Finished { reason, usage, .. } => Some((
                reason.clone(),
                usage
                    .as_ref()
                    .map(|u| (u.prompt_tokens, u.completion_tokens, u.cache_hit_tokens)),
            )),
            _ => None,
        })
        .collect()
}

/// With `include_usage` the server sends `finish_reason` on the last content chunk and the
/// usage on one more chunk with empty `choices`.
#[tokio::test]
async fn stream_ends_with_finished_carrying_reason_and_usage() {
    let body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7,",
        "\"total_tokens\":18,\"prompt_tokens_details\":{\"cached_tokens\":5}}}\n\n",
        "data: [DONE]\n\n",
    );
    let chunks = stream_chunks(body).await;

    assert!(chunks
        .iter()
        .any(|chunk| matches!(chunk, StreamChunk::Content(text) if text == "Hi")));
    assert_eq!(
        finished(&chunks),
        vec![("length".to_string(), Some((11, 7, Some(5))))]
    );
}

/// A server that sends no usage still reports why the stream ended.
#[tokio::test]
async fn stream_without_usage_still_reports_finish_reason() {
    let body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    let chunks = stream_chunks(body).await;

    assert_eq!(finished(&chunks), vec![("tool_calls".to_string(), None)]);
}
