//! Streamed tool calls of the OpenAI and Azure OpenAI providers (wiremock).
//!
//! HTTP mocks only — no vendor keys.

use edgequake_llm::providers::azure_openai::AzureOpenAIProvider;
use edgequake_llm::providers::openai::OpenAIProvider;
use edgequake_llm::traits::{ChatMessage, LLMProvider, StreamChunk};
use futures::StreamExt;
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Index, id, name and argument fragment of a `ToolCallDelta`.
type ToolDelta = (usize, Option<String>, Option<String>, Option<String>);

fn event(delta: &str) -> String {
    format!(
        "data: {{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{{\"index\":0,\"delta\":{delta},\"finish_reason\":null}}]}}\n\n"
    )
}

fn sse(events: &[String]) -> String {
    let mut body = events.concat();
    body.push_str("data: [DONE]\n\n");
    body
}

async fn mock_server(sse_body: String) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex(r"chat/completions$"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;
    server
}

async fn collect(
    stream: futures::stream::BoxStream<'static, edgequake_llm::error::Result<StreamChunk>>,
) -> Vec<StreamChunk> {
    stream
        .map(|chunk| chunk.expect("chunk"))
        .collect::<Vec<_>>()
        .await
}

async fn openai_chunks(sse_body: String) -> Vec<StreamChunk> {
    let server = mock_server(sse_body).await;
    let provider = OpenAIProvider::new("sk-test").with_base_url(format!("{}/v1", server.uri()));
    let stream = provider
        .chat_with_tools_stream(&[ChatMessage::user("hi")], &[], None, None)
        .await
        .expect("stream");
    collect(stream).await
}

async fn azure_chunks(sse_body: String) -> Vec<StreamChunk> {
    let server = mock_server(sse_body).await;
    let provider = AzureOpenAIProvider::new(server.uri(), "key", "deployment");
    let stream = provider
        .chat_with_tools_stream(&[ChatMessage::user("hi")], &[], None, None)
        .await
        .expect("stream");
    collect(stream).await
}

fn tool_deltas(chunks: &[StreamChunk]) -> Vec<ToolDelta> {
    chunks
        .iter()
        .filter_map(|chunk| match chunk {
            StreamChunk::ToolCallDelta {
                index,
                id,
                function_name,
                function_arguments,
                ..
            } => Some((
                *index,
                id.clone(),
                function_name.clone(),
                function_arguments.clone(),
            )),
            _ => None,
        })
        .collect()
}

fn parallel_calls_in_one_event() -> String {
    sse(&[event(
        r#"{"tool_calls":[{"index":0,"id":"call_paris","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}},{"index":1,"id":"call_london","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"London\"}"}}]}"#,
    )])
}

fn text_next_to_a_call() -> String {
    sse(&[event(
        r#"{"content":"Let me check.","tool_calls":[{"index":0,"id":"call_paris","type":"function","function":{"name":"get_weather","arguments":"{}"}}]}"#,
    )])
}

fn expected_parallel() -> Vec<ToolDelta> {
    vec![
        (
            0,
            Some("call_paris".to_string()),
            Some("get_weather".to_string()),
            Some("{\"city\":\"Paris\"}".to_string()),
        ),
        (
            1,
            Some("call_london".to_string()),
            Some("get_weather".to_string()),
            Some("{\"city\":\"London\"}".to_string()),
        ),
    ]
}

#[tokio::test]
async fn openai_keeps_every_tool_call_of_one_event() {
    let chunks = openai_chunks(parallel_calls_in_one_event()).await;
    assert_eq!(tool_deltas(&chunks), expected_parallel());
}

#[tokio::test]
async fn openai_keeps_tool_call_sent_next_to_text() {
    let chunks = openai_chunks(text_next_to_a_call()).await;
    assert!(chunks
        .iter()
        .any(|c| matches!(c, StreamChunk::Content(t) if t == "Let me check.")));
    assert_eq!(tool_deltas(&chunks).len(), 1);
}

#[tokio::test]
async fn azure_keeps_every_tool_call_of_one_event() {
    let chunks = azure_chunks(parallel_calls_in_one_event()).await;
    assert_eq!(tool_deltas(&chunks), expected_parallel());
}

#[tokio::test]
async fn azure_keeps_tool_call_sent_next_to_text() {
    let chunks = azure_chunks(text_next_to_a_call()).await;
    assert!(chunks
        .iter()
        .any(|c| matches!(c, StreamChunk::Content(t) if t == "Let me check.")));
    assert_eq!(tool_deltas(&chunks).len(), 1);
}
