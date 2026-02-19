---
title: "Add Mistral provider (chat, embeddings, list-models) — RFC & implementation spec"
labels: [enhancement, provider, needs-spec]
---

# Add Mistral provider (chat, embeddings, list-models)

Summary
- Goal: Add a first-class Mistral provider implementing chat completions, SSE streaming, function/tool calling, JSON output, embeddings, and model listing. Provider must implement `LLMProvider` and `EmbeddingProvider` and follow the project's factory/registration pattern.

Motivation
- Mistral offers OpenAI-like chat + SSE streaming semantics and a separate embeddings endpoint. Adding a provider lets edgequake users call Mistral models using existing abstractions and tooling.

Scope
- LLM Chat: synchronous chat, streaming (SSE), tool/function calling, JSON mode. Implement `chat`, `chat_with_tools`, `chat_with_tools_stream`, `stream`, and capability flags.
- Embeddings: batch + single embedding methods implementing `EmbeddingProvider::embed` and `embed_one`.
- List models: model discovery endpoint mapping into `ModelCard` or a helper that can be used to populate `models.toml` entries.

Mistral API endpoints (authoritative: https://docs.mistral.ai/api)
- Chat completions: `POST {base_url}/v1/chat/completions`
  - Supports: `messages`, `model`, `temperature`, `top_p`, `max_tokens`, `stream` (SSE), `response_format`/JSON mode, `tools`, `tool_choice`, penalties, `stop`, `presence/frequency`.
  - SSE stream terminates with a data-only `[DONE]` event.
- Embeddings: `POST {base_url}/v1/embeddings` (body: `{ model, input: ["...", ...] }`)
- Models list: `GET {base_url}/v1/models` (returns model IDs and metadata/capabilities)

Authentication & configuration
- Auth: `Authorization: Bearer <MISTRAL_API_KEY>`.
- Provider must support `api_key_env` and `base_url` overrides and timeouts; configuration follows the `ProviderConfig` pattern used by `OpenAICompatibleProvider`.

Design (mapping Mistral ↔ edgequake traits)
- `LLMProvider`
  - `name()` -> "mistral"; `model()` -> configured model id; `max_context_length()` -> model capability (from listing or config).
  - `chat(messages, options)`
    - Convert `ChatMessage` → Mistral `messages` (preserve roles, include images via `ImageData` to data URIs when present).
    - Map `CompletionOptions` → Mistral fields: `temperature`, `top_p`, `max_tokens`, `stop`, `response_format`.
    - POST to `{base_url}/v1/chat/completions`, parse `choices[0].message` into `LLMResponse.content` and usage into tokens.
    - Extract `tool_calls` if present and map to `ToolCall` entries in `LLMResponse`.
  - `chat_with_tools(...)`
    - Serialize `ToolDefinition` → Mistral `tools` JSON schema and set `tool_choice`.
  - Streaming (`stream` / `chat_with_tools_stream`)
    - Use SSE (`reqwest_eventsource::EventSource` per existing providers) and map SSE chunks to `StreamChunk` variants:
      - token deltas → `StreamChunk::Content`
      - tool/function deltas → `StreamChunk::ToolCallDelta` (index/id/name/args)
      - reasoning/thinking content (if exposed) → `StreamChunk::ThinkingContent`
      - finalization → `StreamChunk::Finished`

- `EmbeddingProvider`
  - `embed(texts)` → POST `{base_url}/v1/embeddings` with `{model, input}` and parse `data[].embedding` to Vec<Vec<f32>>.
  - `dimension()` sourced from model config or validated against returned vector length.

- List models
  - Helper `list_models()` calling `GET {base_url}/v1/models` and returning a simple list or converted `ModelCard` entries for `model_config` population.

Error mapping & retries
- Map HTTP & JSON errors to `crate::error::LlmError` (ApiError, NetworkError, RateLimit/429 metadata). Surface `Retry-After` header to retry logic. Follow `retry.rs` patterns used by `openai_compatible`.

Security & telemetry
- Respect API key in env; avoid logging full key (log prefix only). Integrate tracing/genai_events like other providers for telemetry.

Implementation notes (Rust)
- Follow `OpenAICompatibleProvider` and `OpenAIProvider` as canonical references for structure, SSE handling, message conversion, and error mapping.
- Add file `src/providers/mistral.rs` implementing:
  - `struct MistralProvider { client: Client, api_key: String, base_url: String, model: String, embedding_model: Option<String>, model_card: Option<ModelCard>, ... }`
  - `pub fn from_config(config: ProviderConfig) -> Result<Self>` similar to `OpenAICompatibleProvider::from_config`.
  - `impl LLMProvider for MistralProvider` and `impl EmbeddingProvider for MistralProvider`.
- Export provider in `src/providers/mod.rs`.

Examples
- Chat request (JSON):
```json
{
  "model": "mistral-small-latest",
  "messages": [{"role":"user","content":"Hello"}],
  "max_tokens": 256,
  "temperature": 0.2
}
```

- Embedding request (JSON):
```json
{ "model": "mistral-emb-small", "input": ["hello world"] }
```

Tests
- Unit tests: message conversion (text vs multimodal), SSE parsing to `StreamChunk`, embedding parsing and dimension validation, error mapping.
- E2E tests: `tests/e2e_mistral.rs` guarded by `MISTRAL_API_KEY` env var: chat simple completion, streaming, embeddings.
- Mock tests: reuse `mock` provider patterns to simulate SSE and tool-calling flows.

Files to add / modify
- Add: `src/providers/mistral.rs`
- Edit: `src/providers/mod.rs` — export `mistral::MistralProvider`
- Add tests: `tests/e2e_mistral.rs` and unit tests inside `src/providers/mistral.rs`
- Add example: `examples/mistral_integration.rs`

Acceptance criteria
- Provider compiles and integrates with factory/registration.
- `LLMProvider::chat` returns correct `LLMResponse` for non-streaming completions.
- Streaming (`chat_with_tools_stream`) returns `StreamChunk` deltas for content and tool calls.
- `EmbeddingProvider::embed` returns embeddings with correct dimensions.

Implementation checklist
- [ ] Create `src/providers/mistral.rs` with provider struct and `from_config()`.
- [ ] Implement `LLMProvider` methods with chat/tool streaming mapping.
- [ ] Implement `EmbeddingProvider` methods.
- [ ] Add model listing helper and optional sync to `model_config`.
- [ ] Add provider export in `src/providers/mod.rs`.
- [ ] Add unit tests for conversion/parsing and SSE mapping.
- [ ] Add e2e test `tests/e2e_mistral.rs` (guarded by `MISTRAL_API_KEY`).
- [ ] Document provider in `docs/providers.md` and add `examples/mistral_integration.rs`.

If you'd like, I can scaffold `src/providers/mistral.rs` and a minimal e2e test and open a PR.

---

Created-by: edgequake-llm automated RFC generator
