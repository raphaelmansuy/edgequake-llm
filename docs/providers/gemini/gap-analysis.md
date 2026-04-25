# Gemini Gap Analysis (Code vs Current API Guidance)

Date: 2026-04-23

## Sources used

- https://ai.google.dev/gemini-api/docs/function-calling
- https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
- https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings

## Summary

Status: Most major Gemini and Vertex flows are implemented. Key reliability gaps were closed in this update (function-call ID roundtrip + SSE chunk-boundary parsing). Remaining gaps are mostly optional enhancements.

## Detailed Gap Table

| Area | Expected by docs | Previous status | Current status | Evidence |
|---|---|---|---|---|
| Function call id returned by model | Preserve and reuse id in functionResponse | Potentially remapped/generated | Closed | src/providers/gemini.rs uses functionCall.id -> ToolCall.id and tool_result -> functionResponse.id |
| Tool response mapping | Match functionResponse.id to originating call | Name/response only | Closed | convert_messages now writes functionResponse.id from tool_call_id |
| Streaming parse safety | Handle fragmented SSE lines across byte chunks | Line-by-line on each chunk only | Closed | extract_sse_payloads helper with carry buffer |
| Streamed tool-call identity | Preserve model function call id in deltas | Random UUID used for streamed deltas | Closed | StreamChunk::ToolCallDelta now uses functionCall.id when present |
| Thought signature replay | Preserve signature on original parts | Implemented but sensitive to part/chunk placement | Closed/validated | pending_sig strategy + signature forwarding in tool calls |
| Vertex chat path | Covered in provider | Covered | Covered + tested | new e2e test test_vertex_ai_basic_chat |
| Vertex embedding :predict path | Covered in provider | Covered | Covered + tested | new e2e test test_vertex_ai_embeddings |
| Parallel tool call result ordering | API maps by id, ordering can vary | Grouping logic present | Covered by design | tool results grouped as one user content with multiple functionResponse parts |

## Remaining Optional Improvements

- Add explicit e2e for multiple parallel tool calls in one model turn with asynchronous result order.
- Add e2e for thought-signature validation on Gemini 3 with thinking enabled + tool call in same run.
- Add live test artifact generation that snapshots model/version list and endpoint response metadata.

## Proof Checklist

- Provider code patched for id continuity and chunk-safe SSE parsing.
- Gemini E2E expanded with tool-call id roundtrip test.
- Vertex AI E2E coverage added for both chat and embeddings paths.
- Validation run commands documented in docs/providers/gemini/README.md.

## Risk Notes

- Vertex tests are environment-gated and skipped if Vertex credentials are absent.
- Google AI model behavior can still choose direct answers instead of tool calls; tests allow this branch but still validate no-regression behavior.
