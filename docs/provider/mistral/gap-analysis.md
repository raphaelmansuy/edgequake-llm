# Mistral Gap Analysis and Implementation Notes

## Baseline (before this work)

Implemented in edgequake:
- Chat completion + tools + streaming via OpenAI-compatible adapter.
- Vision-in-chat multipart message support.
- JSON mode and reasoning effort passthrough.
- Embeddings (`/v1/embeddings`).
- Model listing (`/v1/models`).

Not implemented in Mistral provider:
- No major endpoint family gap remains for public Mistral chat/embedding/ocr/audio paths.

## What was implemented now

Code changes:
- Added native Mistral methods in `src/providers/mistral.rs`:
  - `speech(...)` -> `POST /v1/audio/speech`
  - `speech_stream_raw(...)` -> `POST /v1/audio/speech` with `stream=true`
  - `transcribe(...)` -> `POST /v1/audio/transcriptions`
  - `transcribe_stream_raw(...)` -> `POST /v1/audio/transcriptions#stream`
  - `transcribe_file_upload(...)` -> multipart file upload mode
  - `ocr(...)` -> `POST /v1/ocr`
  - `list_audio_voices()` -> `GET /v1/audio/voices`
  - `create_audio_voice(...)` -> `POST /v1/audio/voices`
  - `get_audio_voice(...)` -> `GET /v1/audio/voices/{voice_id}`
  - `update_audio_voice(...)` -> `PATCH /v1/audio/voices/{voice_id}`
  - `delete_audio_voice(...)` -> `DELETE /v1/audio/voices/{voice_id}`
  - `get_audio_voice_sample(...)` -> `GET /v1/audio/voices/{voice_id}/sample`
- Added request/response structs for these APIs in `src/providers/mistral.rs`.
- Added e2e smoke tests in `tests/e2e_mistral.rs`:
  - list voices
  - text-to-speech
  - transcription from URL
  - voice details and sample retrieval
  - OCR from document URL
- Added live model snapshot:
  - `docs/provider/mistral/live-models-2026-04-23.md`

## Capability coverage by endpoint family

```ascii
+--------------------------------------------------------------------------------+
|                         Mistral API coverage in edgequake                      |
+-------------------------------+----------------------+-------------------------+
| Endpoint family               | Status               | Location                |
+-------------------------------+----------------------+-------------------------+
| /v1/chat/completions          | implemented          | openai_compatible.rs    |
| /v1/embeddings                | implemented          | mistral.rs              |
| /v1/models                    | implemented          | mistral.rs              |
| /v1/ocr                       | implemented          | mistral.rs              |
| /v1/audio/speech              | implemented          | mistral.rs              |
| /v1/audio/speech (stream)     | implemented          | mistral.rs              |
| /v1/audio/transcriptions      | implemented          | mistral.rs              |
| /v1/audio/transcriptions SSE  | implemented          | mistral.rs              |
| multipart file upload ASR     | implemented          | mistral.rs              |
| /v1/audio/voices CRUD/sample  | implemented          | mistral.rs              |
+-------------------------------+----------------------+-------------------------+
```

## Model support interpretation

Published model inventory is dynamic and authoritative in:
- `GET /v1/models` (snapshot in `docs/provider/mistral/live-models-2026-04-23.md`)

Current practical support in edgequake:
- Any chat model ID can be passed to `MistralProvider::new(..., model, ...)`.
- Embedding model is configurable with `MISTRAL_EMBEDDING_MODEL` or `with_embedding_model(...)`.
- OCR/audio model IDs are caller-provided in the new native request structs.

This means edgequake does not hard-block newer model IDs when Mistral publishes updates.

## Cross-references

Core files:
- `src/providers/mistral.rs`
- `src/providers/openai_compatible.rs`
- `src/model_config.rs`
- `src/traits.rs`

Tests:
- `tests/e2e_mistral.rs`
- `edgequake-litellm/tests/test_e2e_mistral.py`

Primary docs used:
- https://docs.mistral.ai/api/endpoint/chat
- https://docs.mistral.ai/api/endpoint/embeddings
- https://docs.mistral.ai/api/endpoint/models
- https://docs.mistral.ai/api/endpoint/ocr
- https://docs.mistral.ai/api/endpoint/audio/speech
- https://docs.mistral.ai/api/endpoint/audio/transcriptions
- https://docs.mistral.ai/api/endpoint/audio/voices
- https://docs.mistral.ai/models/overview

## Remaining prioritized gaps

1. Add typed OCR options (annotation format, page selection, table format) in a dedicated typed request wrapper.
2. Expand e2e suite with deterministic quality assertions for selected tasks (beyond non-empty checks) while keeping transient-failure resilience.
