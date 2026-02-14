//! GenAI Event Emission following OpenTelemetry Semantic Conventions
//!
//! This module implements event emission according to:
//! https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/
//!
//! Events are recorded as OpenTelemetry span events with the event name:
//! "gen_ai.client.inference.operation.details"
//!
//! The events contain structured JSON data with the following attributes:
//! - gen_ai.input.messages: Array of input messages
//! - gen_ai.output.messages: Array of output messages

use crate::traits::{ChatMessage, ChatRole};
use serde::{Deserialize, Serialize};
use std::env;

/// Check if content capture is enabled via environment variable
pub fn should_capture_content() -> bool {
    env::var("EDGECODE_CAPTURE_CONTENT")
        .map(|v| v.to_lowercase() == "true" || v == "1")
        .unwrap_or(false)
}

// No need for a separate logger getter function

/// GenAI message part (text or tool_call or tool_result)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GenAIMessagePart {
    Text { text: String },
    ToolCall { tool_call: GenAIToolCall },
    ToolResult { tool_result: GenAIToolResult },
}

/// GenAI tool call structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenAIToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// GenAI tool result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenAIToolResult {
    pub tool_call_id: String,
    pub content: String,
}

/// GenAI message following OpenTelemetry schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenAIMessage {
    pub role: String,
    pub content: Vec<GenAIMessagePart>,
}

/// Convert ChatMessage to GenAI message format
pub fn convert_to_genai_messages(messages: &[ChatMessage]) -> Vec<GenAIMessage> {
    messages
        .iter()
        .map(|msg| {
            let role = match msg.role {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
                ChatRole::Tool => "tool",
                ChatRole::Function => "function",
            };

            let mut all_parts = vec![GenAIMessagePart::Text {
                text: msg.content.clone(),
            }];

            // Add tool calls if present
            if let Some(tool_calls) = &msg.tool_calls {
                for tc in tool_calls {
                    all_parts.push(GenAIMessagePart::ToolCall {
                        tool_call: GenAIToolCall {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        },
                    });
                }
            }

            GenAIMessage {
                role: role.to_string(),
                content: all_parts,
            }
        })
        .collect()
}

/// Emit a gen_ai.client.inference.operation.details event as a span event
///
/// # Arguments
/// * `input_messages` - Input messages sent to the LLM
/// * `output_messages` - Output messages received from the LLM
/// * `response` - The LLM response containing metadata (response_id, finish_reason, etc.)
/// * `options` - Optional request options (temperature, max_tokens, etc.)
///
/// This function adds an event to the current active span using tracing macros.
/// The event will be exported as part of the span to Jaeger.
///
/// # OODA-13: Extended Metadata Capture
/// This function now captures comprehensive metadata following OpenTelemetry GenAI semantic conventions:
/// - Response ID (gen_ai.response.id) - Unique identifier from LLM provider
/// - Finish reason (gen_ai.response.finish_reasons) - Why generation stopped
/// - Cache hits (gen_ai.usage.cache_hit_tokens) - Tokens served from cache
/// - Request options (temperature, max_tokens, top_p, penalties)
pub fn emit_inference_event(
    input_messages: &[ChatMessage],
    output_messages: &[ChatMessage],
    response: &crate::traits::LLMResponse,
    options: Option<&crate::traits::CompletionOptions>,
) {
    // Check if content capture is enabled
    if !should_capture_content() {
        tracing::debug!("Content capture disabled (EDGECODE_CAPTURE_CONTENT not set to true)");
        return;
    }

    // Convert messages to GenAI format
    let input = convert_to_genai_messages(input_messages);
    let output = convert_to_genai_messages(output_messages);

    // Serialize to JSON
    let input_json = match serde_json::to_string(&input) {
        Ok(json) => json,
        Err(e) => {
            tracing::warn!("Failed to serialize input messages: {}", e);
            return;
        }
    };

    let output_json = match serde_json::to_string(&output) {
        Ok(json) => json,
        Err(e) => {
            tracing::warn!("Failed to serialize output messages: {}", e);
            return;
        }
    };

    // OODA-13: Extract metadata for event emission
    // Extract response_id from metadata HashMap
    let response_id = response
        .metadata
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let finish_reason = response.finish_reason.as_deref().unwrap_or("");

    // OODA-13: Extract optional parameters with defaults for Jaeger compatibility
    // tracing-opentelemetry may not properly export Option<T> fields, so we use f64 with sentinel
    let temperature_val = options.and_then(|o| o.temperature).unwrap_or(-1.0) as f64;
    let max_tokens_val = options.and_then(|o| o.max_tokens).unwrap_or(0) as i64;
    let top_p_val = options.and_then(|o| o.top_p).unwrap_or(-1.0) as f64;
    let frequency_penalty_val = options.and_then(|o| o.frequency_penalty).unwrap_or(-999.0) as f64;
    let presence_penalty_val = options.and_then(|o| o.presence_penalty).unwrap_or(-999.0) as f64;
    let cache_hit_tokens_val = response.cache_hit_tokens.unwrap_or(0) as i64;

    // Emit the event using tracing::event! macro which adds it to the current span
    // The event will appear in Jaeger as a span event (log entry within the span timeline)
    // OODA-13: Now includes comprehensive metadata per OpenTelemetry GenAI conventions
    tracing::event!(
        target: "gen_ai.events",
        tracing::Level::INFO,
        event.name = "gen_ai.client.inference.operation.details",
        gen_ai.input.messages = %input_json,
        gen_ai.output.messages = %output_json,
        gen_ai.response.id = %response_id,
        gen_ai.response.finish_reasons = %finish_reason,
        gen_ai.usage.input_tokens = response.prompt_tokens as i64,
        gen_ai.usage.output_tokens = response.completion_tokens as i64,
        gen_ai.usage.cache_hit_tokens = cache_hit_tokens_val,
        gen_ai.request.temperature = temperature_val,
        gen_ai.request.max_tokens = max_tokens_val,
        gen_ai.request.top_p = top_p_val,
        gen_ai.request.frequency_penalty = frequency_penalty_val,
        gen_ai.request.presence_penalty = presence_penalty_val,
        "GenAI inference completed"
    );

    tracing::debug!(
        "Emitted gen_ai.client.inference.operation.details event with response_id={} finish_reason={}",
        response_id,
        finish_reason
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolCall;

    #[test]
    fn test_convert_simple_text_message() {
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello, world!".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            cache_control: None,
            images: None,
        }];

        let genai = convert_to_genai_messages(&messages);
        assert_eq!(genai.len(), 1);
        assert_eq!(genai[0].role, "user");
        assert_eq!(genai[0].content.len(), 1);

        match &genai[0].content[0] {
            GenAIMessagePart::Text { text } => {
                assert_eq!(text, "Hello, world!");
            }
            _ => panic!("Expected text part"),
        }
    }

    #[test]
    fn test_convert_with_tool_calls() {
        let messages = vec![ChatMessage {
            role: ChatRole::Assistant,
            content: "Let me search for that.".to_string(),
            name: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_123".to_string(),
                call_type: "function".to_string(),
                function: crate::traits::FunctionCall {
                    name: "web_search".to_string(),
                    arguments: r#"{"query":"test"}"#.to_string(),
                },
            }]),
            tool_call_id: None,
            cache_control: None,
            images: None,
        }];

        let genai = convert_to_genai_messages(&messages);
        assert_eq!(genai.len(), 1);
        assert_eq!(genai[0].role, "assistant");
        assert_eq!(genai[0].content.len(), 2); // text + tool_call
    }

    #[test]
    fn test_should_capture_content_enabled() {
        env::set_var("EDGECODE_CAPTURE_CONTENT", "true");
        assert!(should_capture_content());
        env::remove_var("EDGECODE_CAPTURE_CONTENT");
    }

    #[test]
    fn test_should_capture_content_disabled() {
        env::remove_var("EDGECODE_CAPTURE_CONTENT");
        assert!(!should_capture_content());
    }

    #[test]
    fn test_json_serialization() {
        let genai = GenAIMessage {
            role: "user".to_string(),
            content: vec![GenAIMessagePart::Text {
                text: "Test message".to_string(),
            }],
        };

        let json = serde_json::to_string(&genai).unwrap();
        assert!(json.contains("user"));
        assert!(json.contains("Test message"));
        assert!(json.contains("\"type\":\"text\""));
    }
}
