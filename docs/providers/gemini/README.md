# Gemini Provider (Google AI + Vertex AI)

This document is the implementation-level reference for the Gemini provider in this repository.

## Scope

The provider in src/providers/gemini.rs supports:

- Google AI Gemini endpoint via GEMINI_API_KEY
- Vertex AI Gemini endpoint via GOOGLE_CLOUD_PROJECT + GOOGLE_ACCESS_TOKEN (+ optional GOOGLE_CLOUD_REGION)
- Chat completion (single + multi-turn)
- Streaming (SSE)
- Tool/function calling
- Thinking configuration and thought signature replay
- Embeddings (Google AI + Vertex AI :predict path)
- Context caching for system instructions (Google AI cachedContents)

## Configuration

### Google AI

Required:

- GEMINI_API_KEY

Optional:

- model override via provider.with_model(...)

### Vertex AI

Required:

- GOOGLE_CLOUD_PROJECT
- GOOGLE_ACCESS_TOKEN

Optional:

- GOOGLE_CLOUD_REGION (default: us-central1)

## Capability Matrix

| Capability | Google AI | Vertex AI | Notes |
|---|---|---|---|
| Chat | Yes | Yes | generateContent |
| Stream | Yes | Yes | streamGenerateContent + alt=sse |
| Tool calling | Yes | Yes | tools/functionDeclarations + toolConfig |
| Tool streaming | Yes | Yes | StreamChunk::ToolCallDelta |
| Thinking config | Yes | Yes | generationConfig.thinkingConfig |
| Thought signatures | Yes | Yes | captured + replayed on parts |
| Embeddings (single) | Yes | Yes | Vertex uses :predict |
| Embeddings (batch) | Yes | Yes | provider batches client-side for Vertex |
| Model listing | Yes | Yes | provider maps to endpoint-specific list path |
| Context cache | Yes | Partial | cache create/reuse is Google AI native |

## Edge-Case Guarantees Covered

The current implementation explicitly handles:

- Function call ID continuity:
  - functionCall.id from Gemini responses is preserved into ToolCall.id.
  - functionResponse.id is populated from tool_call_id when sending tool results.
- Thinking signature continuity:
  - thought_signature is preserved on tool-call parts and replayed as required.
  - fallback behavior carries pending signatures between parts/chunks when needed.
- SSE chunk-boundary safety:
  - stream parsers buffer partial lines and only parse complete data: payloads.
- Parallel tool call stability in streaming:
  - each streamed tool call gets a stable monotonic index.
- Empty/no-candidate and prompt-blocked responses:
  - promptFeedback.blockReason surfaces as actionable API errors.
- Rate-limit normalization:
  - 429 / RESOURCE_EXHAUSTED mapped to RateLimited errors.

## Tests

Primary file:

- tests/e2e_gemini.rs

Important coverage buckets:

- Basic chat/system/conversation
- JSON mode
- Streaming
- Multimodal image input
- Embeddings + batch embeddings
- Tool-call ID roundtrip regression (new)
- Vertex AI chat regression (new)
- Vertex AI embedding path regression (new)

### Run commands

Google AI (ignored E2E):

```bash
cargo test --test e2e_gemini -- --ignored
```

Focused tool-call roundtrip:

```bash
cargo test --test e2e_gemini test_gemini_tool_call_id_roundtrip -- --ignored
```

Vertex-only focused checks:

```bash
cargo test --test e2e_gemini test_vertex_ai_basic_chat -- --ignored
cargo test --test e2e_gemini test_vertex_ai_embeddings -- --ignored
```

## References

- Gemini function calling docs (id + functionResponse.id mapping)
- Gemini thought signatures guidance
- Vertex inference request/response reference
- Vertex text embeddings endpoint

See gap-analysis.md for detailed compliance status.
