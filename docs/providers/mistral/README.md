# Mistral Provider Coverage (edgequake-llm)

This folder documents Mistral capability coverage, implementation details, and remaining work.

## Scope

Primary implementation:
- `src/providers/mistral.rs`
- `src/providers/openai_compatible.rs`
- `tests/e2e_mistral.rs`

Primary external references:
- https://docs.mistral.ai/api/endpoint/chat
- https://docs.mistral.ai/api/endpoint/embeddings
- https://docs.mistral.ai/api/endpoint/models
- https://docs.mistral.ai/api/endpoint/ocr
- https://docs.mistral.ai/api/endpoint/audio/speech
- https://docs.mistral.ai/api/endpoint/audio/transcriptions
- https://docs.mistral.ai/api/endpoint/audio/voices
- https://docs.mistral.ai/models/overview

## Capability Matrix

| Capability | Mistral API | edgequake status | Notes |
|---|---|---|---|
| Chat completions | `POST /v1/chat/completions` | Supported | Uses OpenAI-compatible path, includes tools + streaming + JSON mode |
| Vision in chat | multipart `messages[].content` | Supported | Uses image parts through OpenAI-compatible message conversion |
| Thinking/reasoning | `reasoning_effort` + reasoning content | Supported | Propagated via `CompletionOptions.reasoning_effort` |
| Embeddings | `POST /v1/embeddings` | Supported | Native Mistral implementation in provider |
| List models | `GET /v1/models` | Supported | Native Mistral implementation in provider |
| OCR | `POST /v1/ocr` | Supported | Added native Mistral helper method |
| Text to speech | `POST /v1/audio/speech` | Supported | Added native Mistral helper method |
| Audio transcription | `POST /v1/audio/transcriptions` | Supported | Added native Mistral helper method |
| Voices API | `/v1/audio/voices*` | Supported | List/create/get/update/delete + sample retrieval wrappers |
| Transcription streaming | `POST /v1/audio/transcriptions#stream` | Supported | Added raw SSE helper method |
| Transcription file upload | multipart `file` | Supported | Added dedicated multipart upload helper |

## Architecture

```ascii
+------------------------------------------------------------------------+
|                          edgequake mistral flow                        |
+------------------------------------------------------------------------+
|                                                                        |
|  +--------------------+            +--------------------------------+  |
|  |  MistralProvider   |            | OpenAICompatibleProvider       |  |
|  |  (src/providers/   |--chat----->| /v1/chat/completions           |  |
|  |   mistral.rs)      |--stream--->| tools/json/reasoning/image     |  |
|  +--------------------+            +--------------------------------+  |
|           |                                                            |
|           +--embeddings----> POST /v1/embeddings                       |
|           +--models--------> GET  /v1/models                           |
|           +--ocr-----------> POST /v1/ocr                              |
|           +--speech------->  POST /v1/audio/speech                     |
|           +--transcribe--->  POST /v1/audio/transcriptions             |
|           +--voices------->  /v1/audio/voices* (list/create/get/...)   |
|           +--speech(sse)->   POST /v1/audio/speech   (stream=true)     |
|           +--asr(sse)---->   POST /v1/audio/transcriptions#stream       |
|                                                                        |
+------------------------------------------------------------------------+
```

## Cross-reference Map

- Generic provider abstractions: `src/traits.rs`
- Provider registration/config defaults: `src/model_config.rs`
- Mistral implementation: `src/providers/mistral.rs`
- OpenAI-compatible transport logic: `src/providers/openai_compatible.rs`
- Rust E2E coverage: `tests/e2e_mistral.rs`
- Python LiteLLM parity tests: `edgequake-litellm/tests/test_e2e_mistral.py`

## Live Model Inventory

A live model inventory snapshot is stored in `docs/providers/mistral/live-models-2026-04-23.md`.

## Gap Report

Detailed implemented gaps and remaining limitations are in `docs/providers/mistral/gap-analysis.md`.
