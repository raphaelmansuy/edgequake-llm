//! Incremental streamed tool-call accumulator (EdgeCrab proxy / clients).

use std::collections::BTreeMap;

use crate::error::Result;
use crate::stream_tools::finalize_streamed_tool_calls_with_repair;
use crate::traits::ToolCall;

pub use crate::stream_tools::{FinalizeStreamToolCallsOptions, PartialStreamToolCall};

/// Accumulates [`crate::traits::StreamChunk::ToolCallDelta`] fragments by index.
#[derive(Debug, Default)]
pub struct StreamToolCallAccumulator {
    partials: BTreeMap<usize, PartialStreamToolCall>,
}

impl StreamToolCallAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply_delta(
        &mut self,
        index: usize,
        id: Option<String>,
        function_name: Option<String>,
        function_arguments: Option<String>,
        thought_signature: Option<String>,
    ) {
        let entry = self.partials.entry(index).or_default();
        if let Some(id) = id.filter(|s| !s.is_empty()) {
            entry.id = Some(id);
        }
        if let Some(name) = function_name.filter(|s| !s.is_empty()) {
            entry.function_name = Some(name);
        }
        if let Some(args) = function_arguments {
            entry.arguments.push_str(&args);
        }
        if let Some(sig) = thought_signature.filter(|s| !s.is_empty()) {
            entry.thought_signature = Some(sig);
        }
    }

    pub fn finalize(self, options: FinalizeStreamToolCallsOptions) -> Result<Vec<ToolCall>> {
        finalize_streamed_tool_calls_with_repair(self.partials, options, None)
    }
}
