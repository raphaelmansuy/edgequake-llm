//! Server-Sent Events (SSE) streaming support for VSCode Copilot.
//!
//! # Architecture
//!
//! This module handles parsing of Server-Sent Events (SSE) from the Copilot API.
//! SSE is a streaming protocol where the server pushes data to the client in
//! a line-oriented format.
//!
//! ## SSE Parsing Flow
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     SSE Stream Parsing                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                   │
//! │  ┌─────────────────┐                                             │
//! │  │ HTTP Response   │  Chunked bytes from reqwest                 │
//! │  │ bytes_stream()  │                                             │
//! │  └────────┬────────┘                                             │
//! │           │                                                       │
//! │           ▼                                                       │
//! │  ┌─────────────────┐                                             │
//! │  │ String Buffer   │  WHY: HTTP chunks may split lines           │
//! │  │ Accumulate      │  Buffer until newline found                 │
//! │  └────────┬────────┘                                             │
//! │           │                                                       │
//! │           ▼                                                       │
//! │  ┌─────────────────┐      Recognized Prefixes:                   │
//! │  │ Parse SSE Line  │      - data: → JSON content or [DONE]       │
//! │  │ strip_prefix()  │      - event: → ignored                     │
//! │  └────────┬────────┘      - id: → ignored                        │
//! │           │               - : → comment, ignored                  │
//! │           │                                                       │
//! │           ├───────────────────┬───────────────────┐              │
//! │           │                   │                   │               │
//! │           ▼                   ▼                   ▼               │
//! │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐         │
//! │  │ data: JSON  │     │ data: [DONE]│     │ Other Lines │         │
//! │  │ Deserialize │     │ End stream  │     │ Ignore/warn │         │
//! │  └──────┬──────┘     └─────────────┘     └─────────────┘         │
//! │         │                                                         │
//! │         ▼                                                         │
//! │  ┌─────────────────┐                                             │
//! │  │ Extract Content │  choices[0].delta.content                   │
//! │  │ Yield String    │  Empty content → skip                       │
//! │  └─────────────────┘                                             │
//! │                                                                   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Buffer Strategy
//!
//! WHY: HTTP chunked transfer can split SSE lines at arbitrary byte boundaries.
//! For example, a chunk might end in the middle of a JSON object:
//!
//! ```text
//! Chunk 1: "data: {\"id\":\"abc\",\"content\":"
//! Chunk 2: "\"Hello\"}\n"
//! ```
//!
//! The buffer accumulates bytes until a complete line (ending in `\n`) is found,
//! then processes complete lines while retaining partial data for the next chunk.
//!
//! ## Error Handling
//!
//! - Network errors → `VsCodeError::Stream`
//! - JSON parse errors → `VsCodeError::Stream` with context
//! - Unknown line formats → warning logged, line ignored
//!
//! ## References
//!
//! - [SSE Specification](https://html.spec.whatwg.org/multipage/server-sent-events.html)
//! - [OpenAI Streaming](https://platform.openai.com/docs/api-reference/chat/create#stream)

use futures::stream::{BoxStream, TryStreamExt};
use reqwest::Response;
use tracing::{debug, warn};

use super::error::{Result, VsCodeError};
use super::types::ChatCompletionChunk;

/// Parse SSE stream from HTTP response.
///
/// The Copilot API returns Server-Sent Events in the format:
/// ```text
/// data: {"id":"...","object":"chat.completion.chunk",...}
///
/// data: {"id":"...","object":"chat.completion.chunk",...}
///
/// data: [DONE]
/// ```
///
/// This function parses the stream and extracts content deltas.
pub(super) fn parse_sse_stream(response: Response) -> BoxStream<'static, Result<String>> {
    // Convert bytes stream to lines, buffering partial lines
    let mut buffer = String::new();

    let stream = response
        .bytes_stream()
        .map_err(|e| VsCodeError::Stream(e.to_string()))
        .try_filter_map(move |chunk| {
            // Add new bytes to buffer
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Extract complete lines
            let mut lines = Vec::new();
            while let Some(idx) = buffer.find('\n') {
                let line = buffer[..idx].trim().to_string();
                buffer.drain(..=idx);

                if !line.is_empty() {
                    lines.push(line);
                }
            }

            futures::future::ready(Ok(if lines.is_empty() {
                None
            } else {
                Some(futures::stream::iter(lines.into_iter().map(Ok)))
            }))
        })
        .try_flatten()
        .try_filter_map(|line| async move {
            // Parse SSE data lines
            if let Some(data) = line.strip_prefix("data: ") {
                // Check for [DONE] signal
                if data.trim() == "[DONE]" {
                    debug!("Received [DONE] signal, ending stream");
                    return Ok(None);
                }

                // Parse JSON chunk
                match serde_json::from_str::<ChatCompletionChunk>(data) {
                    Ok(chunk) => {
                        // Extract content from first choice
                        if let Some(choice) = chunk.choices.first() {
                            if let Some(content) = &choice.delta.content {
                                if !content.is_empty() {
                                    debug!(content_len = content.len(), "Received content delta");
                                    return Ok(Some(content.clone()));
                                }
                            }
                        }
                        // Empty delta, skip
                        Ok(None)
                    }
                    Err(e) => {
                        warn!(error = %e, data = %data, "Failed to parse SSE chunk");
                        Err(VsCodeError::Stream(format!("Failed to parse chunk: {}", e)))
                    }
                }
            } else if line.starts_with("event: ") || line.starts_with("id: ") {
                // Ignore event type and id lines
                Ok(None)
            } else if line.starts_with(':') {
                // Ignore comments
                Ok(None)
            } else {
                // Unknown line format
                warn!(line = %line, "Unexpected SSE line format");
                Ok(None)
            }
        });

    Box::pin(stream)
}

/// Parse SSE stream with tool call support (OODA-05).
///
/// Unlike `parse_sse_stream` which returns String content, this function
/// returns `StreamChunk` to support the full range of streaming events:
/// - Content chunks (`StreamChunk::Content`)
/// - Tool call deltas (`StreamChunk::ToolCallDelta`)
/// - Finish reason (`StreamChunk::Finished`)
///
/// This enables the React agent to use the streaming path with real-time
/// token counting and progress display.
///
/// # Flow
///
/// ```text
/// SSE bytes → buffer → parse line → match delta type → StreamChunk
///
/// delta.content → StreamChunk::Content(text)
/// delta.tool_calls → StreamChunk::ToolCallDelta {index, id, name, args}
/// finish_reason → StreamChunk::Finished {reason}
/// ```
pub(super) fn parse_sse_stream_with_tools(
    response: Response,
) -> BoxStream<'static, Result<crate::traits::StreamChunk>> {
    use crate::traits::StreamChunk;

    let mut buffer = String::new();

    let stream = response
        .bytes_stream()
        .map_err(|e| VsCodeError::Stream(e.to_string()))
        .try_filter_map(move |chunk| {
            // Add new bytes to buffer
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Extract complete lines
            let mut lines = Vec::new();
            while let Some(idx) = buffer.find('\n') {
                let line = buffer[..idx].trim().to_string();
                buffer.drain(..=idx);

                if !line.is_empty() {
                    lines.push(line);
                }
            }

            futures::future::ready(Ok(if lines.is_empty() {
                None
            } else {
                Some(futures::stream::iter(lines.into_iter().map(Ok)))
            }))
        })
        .try_flatten()
        .try_filter_map(|line| async move {
            // Parse SSE data lines
            if let Some(data) = line.strip_prefix("data: ") {
                // Check for [DONE] signal
                if data.trim() == "[DONE]" {
                    debug!("Received [DONE] signal, ending stream");
                    return Ok(Some(StreamChunk::Finished {
                        reason: "stop".to_string(),
                        ttft_ms: None,
                    }));
                }

                // Parse JSON chunk
                match serde_json::from_str::<ChatCompletionChunk>(data) {
                    Ok(chunk) => {
                        if let Some(choice) = chunk.choices.first() {
                            // Check for finish reason first
                            if let Some(ref finish_reason) = choice.finish_reason {
                                debug!(reason = %finish_reason, "Stream finished");
                                return Ok(Some(StreamChunk::Finished {
                                    reason: finish_reason.clone(),
                                    ttft_ms: None,
                                }));
                            }

                            // Check for tool calls (OODA-05)
                            if let Some(ref tool_calls) = choice.delta.tool_calls {
                                if let Some(tc) = tool_calls.first() {
                                    let function_name =
                                        tc.function.as_ref().and_then(|f| f.name.clone());
                                    let function_arguments =
                                        tc.function.as_ref().and_then(|f| f.arguments.clone());

                                    debug!(
                                        index = tc.index,
                                        id = ?tc.id,
                                        name = ?function_name,
                                        "Received tool call delta"
                                    );

                                    return Ok(Some(StreamChunk::ToolCallDelta {
                                        index: tc.index,
                                        id: tc.id.clone(),
                                        function_name,
                                        function_arguments,
                                    }));
                                }
                            }

                            // Check for content
                            if let Some(ref content) = choice.delta.content {
                                if !content.is_empty() {
                                    debug!(content_len = content.len(), "Received content delta");
                                    return Ok(Some(StreamChunk::Content(content.clone())));
                                }
                            }
                        }
                        // Empty delta, skip
                        Ok(None)
                    }
                    Err(e) => {
                        warn!(error = %e, data = %data, "Failed to parse SSE chunk");
                        Err(VsCodeError::Stream(format!("Failed to parse chunk: {}", e)))
                    }
                }
            } else if line.starts_with("event: ") || line.starts_with("id: ") {
                // Ignore event type and id lines
                Ok(None)
            } else if line.starts_with(':') {
                // Ignore comments
                Ok(None)
            } else {
                // Unknown line format
                warn!(line = %line, "Unexpected SSE line format");
                Ok(None)
            }
        });

    Box::pin(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // SSE Line Recognition Tests
    // WHY: Verify that all SSE line types are correctly identified and handled
    // =========================================================================

    #[test]
    fn test_parse_done_signal() {
        // Test that [DONE] signal is recognized
        let data = "data: [DONE]";
        assert!(data.starts_with("data: "));
        let content = &data[6..];
        assert_eq!(content.trim(), "[DONE]");
    }

    #[test]
    fn test_done_signal_with_whitespace() {
        // WHY: The API may include trailing whitespace
        let variations = [
            "data: [DONE]",
            "data: [DONE] ",
            "data: [DONE]\r",
            "data:  [DONE]",
        ];

        for data in variations {
            assert!(data.starts_with("data:"), "Should start with data:");
            let content = data.strip_prefix("data:").unwrap().trim();
            assert_eq!(content, "[DONE]", "Failed for: {:?}", data);
        }
    }

    #[test]
    fn test_done_signal_is_case_sensitive() {
        // WHY: [DONE] must be uppercase per OpenAI spec
        let invalid = ["data: [done]", "data: [Done]", "data: done"];

        for data in invalid {
            let content = data.strip_prefix("data: ").unwrap_or("").trim();
            assert_ne!(content, "[DONE]", "[DONE] check should be case-sensitive");
        }
    }

    #[test]
    fn test_sse_event_line_prefix() {
        // WHY: SSE can include event type lines which we ignore
        let line = "event: message";
        assert!(line.starts_with("event: "), "Should recognize event prefix");
    }

    #[test]
    fn test_sse_id_line_prefix() {
        // WHY: SSE can include message ID lines which we ignore
        let line = "id: 12345";
        assert!(line.starts_with("id: "), "Should recognize id prefix");
    }

    #[test]
    fn test_sse_comment_line_prefix() {
        // WHY: SSE comments start with colon - used for keep-alive
        let comment = ": this is a comment";
        assert!(comment.starts_with(':'), "Should recognize comment prefix");
    }

    #[test]
    fn test_sse_data_line_prefix() {
        // WHY: Content lines start with "data: "
        let data_line = "data: {\"content\":\"hello\"}";
        assert!(data_line.starts_with("data: "));

        let json = data_line.strip_prefix("data: ").unwrap();
        assert!(json.starts_with('{'));
    }

    // =========================================================================
    // JSON Chunk Parsing Tests
    // WHY: Verify deserialization of various chunk formats
    // =========================================================================

    #[test]
    fn test_parse_chunk_format() {
        let json = r#"{"id":"test","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;

        let chunk: std::result::Result<ChatCompletionChunk, _> = serde_json::from_str(json);
        assert!(chunk.is_ok());

        let chunk = chunk.unwrap();
        assert_eq!(chunk.id, "test");
        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
    }

    #[test]
    fn test_chunk_with_empty_content() {
        // WHY: First chunk often has empty content (role-only delta)
        let json = r#"{"id":"test","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();

        // Content exists but is empty
        assert_eq!(chunk.choices[0].delta.content, Some("".to_string()));
    }

    #[test]
    fn test_chunk_with_no_content() {
        // WHY: Final chunk may have no content, just finish_reason
        let json = r#"{"id":"test","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();

        assert!(chunk.choices[0].delta.content.is_none());
        assert_eq!(chunk.choices[0].finish_reason, Some("stop".to_string()));
    }

    #[test]
    fn test_chunk_with_role_only() {
        // WHY: First assistant chunk typically has role but no content
        let json = r#"{"id":"test","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();

        assert_eq!(chunk.choices[0].delta.role, Some("assistant".to_string()));
        assert!(chunk.choices[0].delta.content.is_none());
    }

    #[test]
    fn test_chunk_malformed_json() {
        // WHY: Malformed JSON should produce a parse error
        let bad_json = r#"{"id":"test", broken json"#;

        let result: std::result::Result<ChatCompletionChunk, _> = serde_json::from_str(bad_json);
        assert!(result.is_err(), "Malformed JSON should fail to parse");
    }

    #[test]
    fn test_chunk_multiple_choices() {
        // WHY: API can return multiple choices (n > 1), we use first
        let json = r#"{
            "id":"test",
            "object":"chat.completion.chunk",
            "created":123,
            "model":"gpt-4o",
            "choices":[
                {"index":0,"delta":{"content":"First"},"finish_reason":null},
                {"index":1,"delta":{"content":"Second"},"finish_reason":null}
            ]
        }"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();

        assert_eq!(chunk.choices.len(), 2);
        assert_eq!(chunk.choices[0].delta.content, Some("First".to_string()));
        assert_eq!(chunk.choices[1].delta.content, Some("Second".to_string()));
    }

    #[test]
    fn test_chunk_empty_choices() {
        // WHY: Edge case - choices array is empty
        let json = r#"{"id":"test","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[]}"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();

        assert!(chunk.choices.is_empty());
        // In parse_sse_stream, this would result in None (skipped)
    }

    // =========================================================================
    // Content Extraction Logic Tests
    // WHY: Verify the content extraction logic matches expected behavior
    // =========================================================================

    #[test]
    fn test_extract_content_from_choice() {
        let json = r#"{"id":"abc","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"World"},"finish_reason":null}]}"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();

        // Simulate the extraction logic from parse_sse_stream
        let content = chunk
            .choices
            .first()
            .and_then(|c| c.delta.content.as_ref())
            .filter(|s| !s.is_empty());

        assert_eq!(content, Some(&"World".to_string()));
    }

    #[test]
    fn test_extract_content_filters_empty() {
        let json = r#"{"id":"abc","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":""},"finish_reason":null}]}"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();

        let content = chunk
            .choices
            .first()
            .and_then(|c| c.delta.content.as_ref())
            .filter(|s| !s.is_empty());

        assert!(content.is_none(), "Empty content should be filtered out");
    }

    #[test]
    fn test_extract_content_with_unicode() {
        // WHY: Content may contain unicode, emojis, etc.
        let json = r#"{"id":"abc","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello 世界 🌍"},"finish_reason":null}]}"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();

        assert_eq!(
            chunk.choices[0].delta.content,
            Some("Hello 世界 🌍".to_string())
        );
    }

    #[test]
    fn test_extract_content_with_newlines() {
        // WHY: Content may contain newlines (code, multi-line text)
        let json = r#"{"id":"abc","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"line1\nline2\nline3"},"finish_reason":null}]}"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();

        let content = chunk.choices[0].delta.content.as_ref().unwrap();
        assert!(content.contains('\n'));
        assert_eq!(content.lines().count(), 3);
    }

    // =========================================================================
    // Finish Reason Tests
    // WHY: Understand how finish_reason affects streaming behavior
    // =========================================================================

    #[test]
    fn test_finish_reason_values() {
        // WHY: Various finish reasons indicate different end conditions
        let reasons = ["stop", "length", "content_filter", "tool_calls"];

        for reason in reasons {
            let json = format!(
                r#"{{"id":"test","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{{"index":0,"delta":{{}},"finish_reason":"{}"}}]}}"#,
                reason
            );

            let chunk: ChatCompletionChunk = serde_json::from_str(&json).unwrap();
            assert_eq!(chunk.choices[0].finish_reason, Some(reason.to_string()));
        }
    }

    #[test]
    fn test_finish_reason_null_during_streaming() {
        // WHY: During streaming, finish_reason is null until the final chunk
        let json = r#"{"id":"test","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"streaming..."},"finish_reason":null}]}"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();

        assert!(chunk.choices[0].finish_reason.is_none());
        assert!(chunk.choices[0].delta.content.is_some());
    }
}
