//! Streaming tool-call assembly helpers used by EdgeCrab's provider call path.

use std::collections::BTreeMap;

use crate::error::{LlmError, Result};
use crate::traits::{FunctionCall, ToolCall};

/// Partial tool call accumulated from [`crate::traits::StreamChunk::ToolCallDelta`] events.
#[derive(Debug, Clone, Default)]
pub struct PartialStreamToolCall {
    pub id: Option<String>,
    pub function_name: Option<String>,
    pub arguments: String,
    pub thought_signature: Option<String>,
}

/// Options for [`finalize_streamed_tool_calls_with_repair`].
#[derive(Debug, Clone, Copy, Default)]
pub struct FinalizeStreamToolCallsOptions {
    /// When true, empty / null-like argument payloads become `{}`.
    pub empty_args_fallback: bool,
}

/// Finalize indexed partial tool calls into wire-ready [`ToolCall`] values.
///
/// `repair` is an optional hook (EdgeCrab argument pipeline) applied to each
/// argument string before the call is emitted.
pub fn finalize_streamed_tool_calls_with_repair(
    partials: BTreeMap<usize, PartialStreamToolCall>,
    options: FinalizeStreamToolCallsOptions,
    repair: Option<fn(&str) -> String>,
) -> Result<Vec<ToolCall>> {
    let mut out = Vec::with_capacity(partials.len());
    for (index, partial) in partials {
        let name = partial.function_name.unwrap_or_default();
        if name.trim().is_empty() {
            return Err(LlmError::InvalidRequest(format!(
                "streamed tool call at index {index} missing function name"
            )));
        }
        let id = partial
            .id
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| format!("call_{index}"));

        let mut args = partial.arguments;
        if let Some(repair_fn) = repair {
            args = repair_fn(&args);
        } else if options.empty_args_fallback {
            let trimmed = args.trim();
            if trimmed.is_empty() || trimmed == "null" || trimmed == "None" {
                args = "{}".to_string();
            }
        }

        let trimmed = args.trim();
        if trimmed.is_empty() || trimmed == "null" || trimmed == "None" {
            if options.empty_args_fallback {
                args = "{}".to_string();
            } else {
                return Err(LlmError::InvalidRequest(format!(
                    "streamed tool call at index {index} finished without arguments"
                )));
            }
        } else if !options.empty_args_fallback
            && serde_json::from_str::<serde_json::Value>(trimmed).is_err()
        {
            return Err(LlmError::InvalidRequest(format!(
                "streamed tool call at index {index} has invalid JSON arguments"
            )));
        }

        out.push(ToolCall {
            id,
            call_type: "function".into(),
            function: FunctionCall {
                name,
                arguments: args,
            },
            thought_signature: partial.thought_signature,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finalize_applies_repair_and_default_id() {
        let mut map = BTreeMap::new();
        map.insert(
            0,
            PartialStreamToolCall {
                id: None,
                function_name: Some("read_file".into()),
                arguments: "null".into(),
                thought_signature: None,
            },
        );
        let calls = finalize_streamed_tool_calls_with_repair(
            map,
            FinalizeStreamToolCallsOptions {
                empty_args_fallback: true,
            },
            Some(|s| {
                if s.trim() == "null" {
                    "{}".into()
                } else {
                    s.to_string()
                }
            }),
        )
        .expect("ok");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[0].function.arguments, "{}");
    }
}
