//! SPEC-131 — shared HTTP client for `POST …/responses`.

use futures::stream::{self, BoxStream};
use serde_json::Value;

use crate::error::{LlmError, Result};
use crate::responses_map::{
    build_responses_request, parse_responses_body, parse_sse_data_line, responses_url,
};
use crate::traits::{ChatMessage, CompletionOptions, LLMResponse, StreamChunk};

/// Non-streaming Responses call.
pub async fn responses_chat(
    base_url: &str,
    api_key: &str,
    model: &str,
    messages: &[ChatMessage],
    options: &CompletionOptions,
) -> Result<LLMResponse> {
    let url = responses_url(base_url);
    let body = build_responses_request(model, messages, options, false);
    let client = reqwest::Client::new();
    let mut req = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&body);
    if !api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {api_key}"));
    }
    let response = req
        .send()
        .await
        .map_err(|e| LlmError::NetworkError(format!("Responses request failed: {e}")))?;
    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|e| LlmError::NetworkError(format!("Responses read failed: {e}")))?;
    if !status.is_success() {
        return Err(LlmError::ApiError(format!(
            "Responses API {status}: {text}"
        )));
    }
    let json: Value = serde_json::from_str(&text).map_err(LlmError::SerializationError)?;
    Ok(parse_responses_body(&json, model))
}

/// Streaming Responses call → string content deltas (matches `LLMProvider::stream`).
pub async fn responses_stream(
    base_url: &str,
    api_key: &str,
    model: &str,
    messages: &[ChatMessage],
    options: &CompletionOptions,
) -> Result<BoxStream<'static, Result<String>>> {
    let url = responses_url(base_url);
    let body = build_responses_request(model, messages, options, true);
    let client = reqwest::Client::new();
    let mut req = client
        .post(&url)
        .header("Content-Type", "application/json")
        .header("Accept", "text/event-stream")
        .json(&body);
    if !api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {api_key}"));
    }
    let response = req
        .send()
        .await
        .map_err(|e| LlmError::NetworkError(format!("Responses stream failed: {e}")))?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(LlmError::ApiError(format!(
            "Responses stream {status}: {text}"
        )));
    }

    // Buffer whole body then emit deltas — sufficient for EdgeQuake query streams
    // and reliable under wiremock (avoids EventSource Content-Type quirks).
    let text = response
        .text()
        .await
        .map_err(|e| LlmError::NetworkError(format!("Responses stream read: {e}")))?;
    let mut chunks: Vec<Result<String>> = Vec::new();
    for line in text.lines() {
        if let Some(data) = line.strip_prefix("data:") {
            match parse_sse_data_line(data) {
                Some(StreamChunk::Content(s)) => chunks.push(Ok(s)),
                Some(StreamChunk::Finished { .. }) => break,
                _ => {}
            }
        }
    }
    Ok(Box::pin(stream::iter(chunks)))
}
