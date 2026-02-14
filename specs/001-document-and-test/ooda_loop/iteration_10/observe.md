# Iteration 10 - Observe
## Observation Target
Version history, public API surface, breaking changes between internal EdgeCode module and standalone crate.

## Findings
- Current version: 0.2.0 (first standalone release)
- MSRV: 1.75.0
- 15 public modules re-exported from lib.rs
- Feature flag `otel` gates OpenTelemetry dependencies
- CHANGELOG shows single release (0.2.0) with all features listed as "Added"
- No prior public releases to migrate from, but library was extracted from EdgeCode project
- Key new types: ToolCall, StreamChunk enum, thinking_content/thinking_tokens on LLMResponse
