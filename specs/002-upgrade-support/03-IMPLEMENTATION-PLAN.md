# Implementation Plan — Provider Upgrade & Gap Closure

> **Version**: 2.0 — 2026-07-04 (updated with implementation status)
>
> **Target**: edgequake-llm v0.8.0
>
> **Principles**: First Principles · DRY · SOLID · Ascending Compatibility · Battle-Tested
>
> ## Implementation Status (2026-07-04)
>
> | Feature | Status | Files Modified/Created |
> |---------|--------|----------------------|
> | FEAT-040: Cohere Provider | ✅ COMPLETE | `src/providers/cohere.rs` (new), `src/providers/mod.rs`, `src/lib.rs`, `src/factory.rs`, `src/discovery/registry.rs` |
> | FEAT-041: Bedrock Embeddings | ✅ ALREADY DONE | No changes needed |
> | FEAT-042: NVIDIA Embeddings | ✅ COMPLETE | `src/providers/nvidia.rs` |
> | FEAT-043: OpenAI Image Gen | ✅ COMPLETE | `src/imagegen/providers/openai_imagegen.rs` (new), `src/imagegen/providers/mod.rs`, `src/imagegen/mod.rs`, `src/imagegen/factory.rs` |
> | FEAT-044: Model Catalog | ✅ COMPLETE | `src/providers/openai.rs`, `src/discovery/registry.rs` |
> | FEAT-051: Mistral reasoning_effort | ✅ COMPLETE | `src/providers/mistral.rs` |
> | FEAT-052: Anthropic effort param | ✅ COMPLETE | `src/providers/anthropic.rs` |
> | FEAT-053: NVIDIA Tool Calling | ✅ COMPLETE | `src/providers/nvidia.rs` |
> | FEAT-054: xAI Image Gen | ✅ COMPLETE | `src/imagegen/providers/xai_imagegen.rs` (new), `src/imagegen/providers/mod.rs` |
> | FEAT-055: Gemini Embedding-2 | ✅ COMPLETE | `src/providers/gemini.rs` |
> | FEAT-056: Gemini 3.x Image Models | ✅ COMPLETE | `src/imagegen/providers/gemini.rs` |
> | E2E Tests | ✅ COMPLETE | `tests/e2e_provider_upgrade.rs` (new), `tests/e2e_discovery.rs`, `tests/e2e_anthropic.rs`, `tests/e2e_mistral.rs`, `tests/e2e_xai.rs` |
>
> **Test Results**: 1,416 tests passing (1,344 lib + 72 integration). 0 failures.

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Architecture Decisions](#2-architecture-decisions)
3. [Phase 1 — Foundation (P0 Gaps)](#3-phase-1--foundation-p0-gaps)
4. [Phase 2 — Feature Parity (P1 Gaps)](#4-phase-2--feature-parity-p1-gaps)
5. [Phase 3 — Polish (P2/P3 Gaps)](#5-phase-3--polish-p2p3-gaps)
6. [Ascending Compatibility Strategy](#6-ascending-compatibility-strategy)
7. [Testing Strategy](#7-testing-strategy)
8. [Risk Register](#8-risk-register)

---

## 1. Design Principles

### First Principles

Every change must answer: **"Does this reduce the distance between the user's intent and working code?"**

1. **Trait boundary is the API contract** — `LLMProvider`, `EmbeddingProvider`, `ImageGenProvider` are stable interfaces. New capabilities extend via new methods with default implementations (Open/Closed Principle).

2. **Provider-specific options belong in `CompletionOptions`** — Provider-unique fields (e.g., `safe_prompt`, `gemini_thinking_level`) use `Option<T>` with `skip_serializing_if = "Option::is_none"`. Non-applicable providers silently ignore them (Liskov Substitution).

3. **Model catalogs are runtime data, not compile-time constants** — Model names, context windows, and capabilities should be discoverable at runtime via the existing `ModelDiscoveryService`, not hardcoded.

4. **Auth patterns are provider-local** — Each provider owns its authentication. No shared auth abstraction that forces inappropriate coupling (Interface Segregation).

### DRY Enforcement

| Pattern | How |
|---------|-----|
| OpenAI-compatible providers | Reuse `openai_compatible.rs` base for NVIDIA, LM Studio, Ollama (OpenAI compat path), Cohere (if Chat Completions compat), future providers |
| Embed request/response | Shared `EmbedRequest` / `EmbedResponse` types with `input_type` field |
| Image gen request/response | Shared `ImageGenRequest` / `ImageGenResponse` with provider-specific options in `ImageGenOptions` |
| HTTP client creation | Shared `build_provider_client(base_url, api_key, timeout)` utility |
| Error mapping | Shared `map_http_error(status, body) -> LlmError` |
| Model catalog updates | Single `models.toml` or similar, loaded at startup |

### SOLID Application

| Principle | Application |
|-----------|-------------|
| **S**ingle Responsibility | Each provider file handles one API. Cross-cutting concerns (retry, tracing, caching) are middleware. |
| **O**pen/Closed | New capabilities added via new default trait methods. Existing providers don't break. |
| **L**iskov Substitution | Any `Box<dyn LLMProvider>` works in any slot. Provider-specific options are optional. |
| **I**nterface Segregation | `LLMProvider`, `EmbeddingProvider`, `ImageGenProvider` are separate traits. Providers implement only what they support. |
| **D**ependency Inversion | Business logic depends on traits, not concrete providers. Factory pattern creates providers from config. |

---

## 2. Architecture Decisions

### ADR-001: Cohere Provider Architecture

**Context**: Cohere has a unique API that differs significantly from OpenAI-compatible endpoints.

**Decision**: Implement `CohereProvider` as a standalone provider (like Anthropic) that implements `LLMProvider`, `EmbeddingProvider`, and optionally `Reranker`.

**Rationale**: Cohere's Chat API uses different field names (`message` vs `messages`, `preamble` vs `system`, `connectors` for RAG). Forcing it through `openai_compatible.rs` would create a leaky abstraction.

**Ref**: https://docs.cohere.com/docs/chat-api

### ADR-002: Image Generation via Unified Trait

**Context**: Image generation APIs differ wildly (OpenAI token-based, xAI REST, Gemini native, Vertex Imagen).

**Decision**: Extend existing `ImageGenProvider` trait. Each provider gets its own implementation file under `src/imagegen/providers/`. The `ImageGenFactory` dispatches by provider name.

**Rationale**: The existing pattern (`GeminiImageGenProvider`, `VertexAIImageGen`, `FalImageGen`) is clean and extensible. No reason to break it.

### ADR-003: Embeddings for Bedrock and NVIDIA

**Context**: Both Bedrock and NVIDIA offer embeddings but via different APIs (Bedrock InvokeModel, NVIDIA OpenAI-compat).

**Decision**:
- **NVIDIA**: Implement as `NvidiaEmbeddingProvider` using OpenAI-compatible `/v1/embeddings` (DRY: reuse HTTP client pattern from `openai_compatible.rs`).
- **Bedrock**: Implement as `BedrockEmbeddingProvider` using AWS SDK `InvokeModel` for Titan Embed V2.

**Rationale**: NVIDIA's API is identical to OpenAI's embeddings endpoint. Bedrock requires AWS SDK for SigV4 auth and model-specific request/response formats.

### ADR-004: Model Catalog Updates Are Non-Breaking

**Context**: Model lists (defaults, context windows) are outdated across all providers.

**Decision**: Model catalog updates are pure data changes. Update default model names, context window sizes, and pricing in configuration. No trait changes. Use feature flags or `models.toml` where possible.

**Ascending compatibility**: Old model names continue to work (providers accept them). New defaults point to current flagships. Discovery API returns live model lists.

### ADR-005: Reasoning/Effort Unification

**Context**: Multiple providers now support configurable reasoning depth, but with different field names:
- OpenAI: `reasoning_effort` (high/medium/low)
- Anthropic: `effort` parameter (high/medium/low) — new in Opus 4.8 / Sonnet 5
- Mistral: `reasoning_effort` (high/medium/low/none)
- Gemini: `thinkingLevel` (minimal/low/medium/high)
- xAI: Reasoning model variants (implicit)

**Decision**: Use `CompletionOptions::reasoning_effort` as the unified field. Each provider maps it to its native API:

```
"high"   → OpenAI reasoning_effort:"high", Anthropic effort:"high", Mistral reasoning_effort:"high", Gemini thinkingLevel:"high"
"medium" → ...
"low"    → ...
"none"   → Mistral reasoning_effort:"none", Gemini thinkingLevel:"minimal"
```

**Rationale**: Single field, provider-local mapping. No breaking changes. Silently ignored by providers that don't support it.

---

## 3. Phase 1 — Foundation (P0 Gaps)

> **Goal**: Close critical gaps. ~2 weeks of focused work.

### 3.1 FEAT-040: Cohere Provider (LLM + Embeddings + Rerank)

**Scope**: New provider implementing `LLMProvider` + `EmbeddingProvider`.

**Files**:
- `src/providers/cohere.rs` — Main provider
- `src/providers/mod.rs` — Module registration
- `src/lib.rs` — Public exports

**Implementation**:

```
CohereProvider {
    api_key: SecretString,
    base_url: String,          // default: https://api.cohere.com/v2
    model: String,             // default: command-a-plus-05-2026
    embed_model: String,       // default: embed-v4.0
    http_client: reqwest::Client,
}
```

**LLM endpoints**:
- `POST /v2/chat` → `LLMProvider::chat()`
- Streaming via SSE → `LLMProvider::chat_with_tools_stream()`
- Tool use with multi-step → `LLMProvider::chat_with_tools()`

**Embedding endpoint**:
- `POST /v2/embed` → `EmbeddingProvider::embed()`
- Support `input_type` (search_document / search_query / classification / clustering)
- Support Matryoshka dimensions via new `EmbedOptions` struct
- Support multimodal input (images, PDFs) via content blocks

**Rerank** (stretch):
- `POST /v2/rerank` → `Reranker` trait implementation
- Integrate with existing `HttpReranker` pattern

**Models to support**:
| Model ID | Type | Context |
|----------|------|---------|
| `command-a-plus-05-2026` | Chat (flagship) | 128k |
| `command-a-04-2025` | Chat | 128k |
| `command-r-plus-08-2024` | Chat (RAG) | 128k |
| `embed-v4.0` | Embed (multimodal) | 128k |
| `embed-english-v3.0` | Embed (text) | 512 |
| `embed-multilingual-v3.0` | Embed (text) | 512 |
| `rerank-v4.0-pro` | Rerank | 32k |
| `rerank-v4.0-fast` | Rerank | 32k |

**Edge cases**:
- Cohere uses `preamble` instead of `system` for system messages
- Citations in RAG responses need to be captured in `metadata`
- `input_type` is required for embeddings (not optional)
- Matryoshka dimensions: validate against `[256, 512, 1024, 1536]`

**Ref**: https://docs.cohere.com/docs/models
**Ref**: https://docs.cohere.com/changelog/embed-multimodal-v4

### 3.2 FEAT-041: Bedrock Embeddings — ALREADY IMPLEMENTED ✅

> **Status**: Bedrock embeddings already exist via `InvokeModel` supporting Titan v1/v2,
> Cohere v3/v4, and Nova embedding models. No work needed here.
>
> **Remaining gap**: Add Nova 2 Multimodal Embeddings support (P2, deferred to Phase 3).

### 3.3 FEAT-042: NVIDIA NIM Embeddings

**Scope**: Add `EmbeddingProvider` to existing `NvidiaProvider`.

**Implementation**:
- Use OpenAI-compatible `POST /v1/embeddings`
- Reuse HTTP client and auth from existing `NvidiaProvider`
- Default model: `nvidia/llama-nemotron-embed-1b-v2`
- Support `input_type` (query/passage) via model name suffix or custom field
- Support Matryoshka dimensions

**Edge cases**:
- `input_type` is non-standard OpenAI field — NVIDIA adds it as extra body field
- Alternatively, suffix model name: `nvidia/llama-nemotron-embed-1b-v2-query`
- Dimension support: 384, 2048 (default)
- `encoding_format`: "float" (default), "int8", "binary"

**Ref**: https://docs.nvidia.com/nim/nemo-retriever/text-embedding/1.13.0/use-the-api-openai.html

### 3.4 FEAT-043: OpenAI Image Generation

**Scope**: New `ImageGenProvider` for OpenAI `gpt-image-2`.

**File**: `src/imagegen/providers/openai.rs`

**Implementation**:
- `POST /v1/images/generations` with token-based billing
- Support `gpt-image-2`, `gpt-image-1`, `gpt-image-1-mini`
- Map `ImageGenRequest` fields to OpenAI API:
  - `prompt` → `prompt`
  - `aspect_ratio` → `size` (e.g., "1024x1024", "1792x1024")
  - `count` → `n`
  - `quality` → `quality` ("standard", "hd")
  - `response_format` → `response_format` ("url", "b64_json")

**Edge cases**:
- Token-based pricing (different from per-image pricing)
- Image editing endpoint (`POST /v1/images/edits`) — stretch goal
- Content policy rejections (return structured error, not panic)
- Large base64 responses may exceed memory — stream where possible

**Ref**: https://developers.openai.com/api/docs/images

### 3.5 FEAT-044: Model Catalog Update (All Providers)

**Scope**: Update all model defaults and context window constants.

**Changes per provider**:

| Provider | Current Default | New Default | Context Window |
|----------|----------------|-------------|----------------|
| OpenAI | `gpt-4o` | `gpt-5.5` | 1,048,576 |
| Azure | `gpt-4` | `gpt-5.5` | 1,048,576 |
| Anthropic | `claude-3-5-sonnet-20241022` | `claude-sonnet-5` | 1,000,000 |
| Gemini | `gemini-2.5-flash` | `gemini-3.5-flash` | 1,048,576 |
| Mistral | `mistral-large-latest` | `mistral-medium-3-5` | 256,000 |
| xAI | `grok-3-latest` | `grok-4.3` | 256,000 |
| NVIDIA | Generic NIM | `meta/llama-3.3-70b-instruct` | 128,000 |
| Bedrock | `anthropic.claude-3-sonnet` | `anthropic.claude-sonnet-5` | 1,000,000 |

**Ascending compatibility**: Old model names still work (providers accept them). Update only the defaults used when no model is specified.

**Discovery integration**: Update `src/discovery/providers/*.rs` with current model families for each provider.

---

## 4. Phase 2 — Feature Parity (P1 Gaps)

> **Goal**: Close feature gaps with existing providers. ~2 weeks.

### 4.1 FEAT-050: Mistral Vision Support

**Scope**: Add image input support to `MistralProvider`.

**Implementation**:
- Mistral Medium 3.5 accepts image content blocks in messages
- Format: OpenAI-compatible `image_url` in `content` array
- Update `build_request_body()` to serialize `ChatMessage::images`

**Edge cases**:
- Only Medium 3.5 and Pixtral models support vision
- Variable aspect ratio support (custom vision encoder)
- Error handling for non-vision models receiving images

**Ref**: https://docs.mistral.ai/capabilities/vision/

### 4.2 FEAT-051: Mistral Reasoning Effort

**Scope**: Map `CompletionOptions::reasoning_effort` to Mistral's `reasoning_effort` field.

**Implementation**:
- Add `reasoning_effort` to Mistral request body when present
- Values: "high", "medium", "low", "none"
- Direct 1:1 mapping from `CompletionOptions`

**Ref**: https://docs.mistral.ai/studio-api/conversations/reasoning

### 4.3 FEAT-052: Anthropic Effort Parameter

**Scope**: Map `CompletionOptions::reasoning_effort` to Anthropic's `effort` parameter.

**Implementation**:
- Anthropic Opus 4.8 / Sonnet 5 support `effort` (high/medium/low)
- Map from `reasoning_effort` field
- Adaptive thinking (Fable 5) is always-on — `effort` controls depth

**Edge cases**:
- Fable 5: Adaptive thinking cannot be disabled
- Opus 4.8: Default effort is "high"
- Sonnet 5: Default effort is "high" on API
- Haiku 4.5: Does NOT support effort (uses extended thinking instead)

**Ref**: https://platform.claude.com/docs/en/about-claude/models/overview

### 4.4 FEAT-053: NVIDIA Tool Calling

**Scope**: Enable function/tool calling in `NvidiaProvider`.

**Implementation**:
- NIM's `/v1/chat/completions` supports OpenAI-compatible tools
- Override `supports_function_calling() → true`
- Override `supports_tool_streaming() → true`
- Implement `chat_with_tools()` and `chat_with_tools_stream()`

**Edge cases**:
- Not all NIM models support tool calling — depends on deployed model
- `supports_function_calling()` should ideally check model capabilities

### 4.5 FEAT-054: xAI Image Generation

**Scope**: New `ImageGenProvider` for xAI Grok Imagine.

**File**: `src/imagegen/providers/xai.rs`

**Implementation**:
- `POST /v1/images/generations` (OpenAI-compatible)
- Models: `grok-imagine-image-quality`, `grok-imagine-image`
- Support `aspect_ratio` parameter (1:1, 16:9, 9:16, etc.)
- Support batch generation via `n` parameter
- Support `response_format`: "url" (default), "b64_json"

**Ref**: https://docs.x.ai/developers/model-capabilities/images/generation

### 4.6 FEAT-055: Gemini Embedding-2 (Multimodal)

**Scope**: Upgrade Gemini embeddings to support `gemini-embedding-2`.

**Implementation**:
- New `embedContent` request format supporting multimodal input
- Support `output_dimensionality` parameter (128–3072)
- For text-only: same API, new model name
- For multimodal: `content.parts[]` with `inlineData` (images, audio, etc.)

**Edge cases**:
- Max 6 images per request
- Max 180s audio, 120s video
- Max 6 PDF pages
- Different from text embedding dimension defaults

**Ref**: https://ai.google.dev/gemini-api/docs/embeddings
**Ref**: https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2

### 4.7 FEAT-056: Gemini 3.x Image Generation Models

**Scope**: Update `GeminiImageGenProvider` for Gemini 3.x image models.

**Implementation**:
- Add `gemini-3.1-flash-image`, `gemini-3-pro-image` to available models
- Update default model from `gemini-2.0-flash-exp`
- Support new response format for Nano Banana 2 models

**Ref**: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions

### 4.8 FEAT-057: Bedrock Image Generation (Stability AI)

**Scope**: New `ImageGenProvider` for Stability AI on Bedrock.

**File**: `src/imagegen/providers/bedrock_stability.rs`

**Implementation**:
- Use Bedrock `InvokeModel` with Stability AI request format
- Models: `stability.stable-image-core-v1:1`, `stability.stable-image-ultra-v1:0`
- Request format: `{"text_prompts": [{"text": "..."}], "cfg_scale": 7, ...}`
- Response: base64 image data

**Feature gate**: Behind `bedrock` feature flag.

---

## 5. Phase 3 — Polish (P2/P3 Gaps)

> **Goal**: Incremental improvements. Ongoing.

### 5.1 FEAT-060: Azure Managed Identity Auth

**Scope**: Support Azure Entra ID / Managed Identity for Azure OpenAI.

**Implementation**:
- Add `AzureCredential` enum (ApiKey | ManagedIdentity | EntraId)
- For ManagedIdentity: use IMDS endpoint to get token
- For EntraId: use client certificate / client secret flow
- Token refresh with automatic expiry handling

### 5.2 FEAT-061: Vertex AI ADC Authentication

**Scope**: Support Application Default Credentials for Google Cloud.

**Implementation**:
- Detect ADC via `GOOGLE_APPLICATION_CREDENTIALS` or metadata server
- Support Workload Identity Federation for GKE/Cloud Run
- Auto-refresh tokens using service account key or metadata

### 5.3 FEAT-062: Structured Outputs (JSON Schema)

**Scope**: Support OpenAI-style `response_format: { type: "json_schema", json_schema: {...} }`.

**Implementation**:
- Add `response_schema: Option<serde_json::Value>` to `CompletionOptions`
- OpenAI: `response_format.json_schema`
- Anthropic: Not natively supported (use tool_use pattern)
- Gemini: `responseMimeType + responseSchema`
- Mistral: `response_format.json_schema`

### 5.4 FEAT-063: Ollama Embed Dimensions

**Scope**: Expose `dimensions` parameter in Ollama embed requests.

**Implementation**:
- Add `dimensions: Option<usize>` to embed request
- Pass through to `/api/embed` body

### 5.5 FEAT-064: OpenRouter Fallback Arrays

**Scope**: Support OpenRouter's `models` array for fallback routing.

**Implementation**:
- Add `fallback_models: Option<Vec<String>>` to `CompletionOptions` or provider config
- When set, use `models` field instead of `model` in request body

### 5.6 FEAT-065: Bedrock Cross-Region Inference

**Scope**: Support geo/global inference IDs for Bedrock.

**Implementation**:
- Allow model IDs with `us.`, `eu.`, `global.` prefix
- Auto-detect from region configuration
- Support `bedrock-mantle` endpoint as alternative

---

## 6. Ascending Compatibility Strategy

### Guarantees

1. **No breaking changes to public traits** — `LLMProvider`, `EmbeddingProvider`, `ImageGenProvider` signatures are frozen. New methods have default implementations.

2. **Old model names continue to work** — Providers accept deprecated model IDs (the API handles redirection). edgequake-llm should log a deprecation warning but not reject.

3. **New fields in `CompletionOptions` are always `Option<T>`** — Existing code that doesn't set them gets default behavior.

4. **Feature gates protect optional dependencies** — `bedrock` feature flag continues to gate AWS SDK dependencies.

5. **Semantic versioning** — v0.8.0 for new features (minor bump). Breaking changes only in v1.0.0.

### Migration Path for Users

```rust
// Before (v0.7.0) — still works in v0.8.0
let provider = OpenAIProvider::new("sk-...");
let response = provider.chat(&messages, None).await?;

// After (v0.8.0) — new capabilities available
let provider = OpenAIProvider::new("sk-...");
let mut opts = CompletionOptions::default();
opts.reasoning_effort = Some("high".to_string());
let response = provider.chat(&messages, Some(&opts)).await?;
```

### Deprecation Warnings

Add tracing warnings for deprecated models:
```rust
if model.starts_with("gpt-4") || model.starts_with("gpt-3.5") {
    tracing::warn!(
        model = %model,
        "Model is deprecated. Consider upgrading to gpt-5.5. See: https://developers.openai.com/api/docs/deprecations"
    );
}
```

---

## 7. Testing Strategy

### Unit Tests

Each new provider/capability gets:
1. **Request serialization test** — Verify the JSON body matches the official API spec
2. **Response deserialization test** — Parse sample API responses correctly
3. **Error handling test** — HTTP 4xx/5xx → appropriate `LlmError` variant
4. **Edge case tests** — Empty input, max batch size, unsupported model

### Integration Tests (E2E)

Using existing `tests/` pattern:
1. **Auth validation** — Provider accepts configured credentials
2. **Basic completion** — Simple prompt → valid response
3. **Streaming** — Token-by-token delivery works
4. **Embeddings** — Vector dimensions match expected
5. **Image generation** — Returns valid image data

### Mock Tests

Extend `MockProvider` for new capabilities:
1. `MockCohereProvider` — Chat + embeddings + rerank
2. `MockImageGenProvider` — Already exists, extend for OpenAI/xAI

### CI Pipeline

- Unit tests: `cargo test` (no API keys needed)
- Integration tests: `cargo test --features bedrock` with secrets
- All providers: `cargo test -- --ignored` for live API tests

---

## 8. Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Cohere API v2 is unstable | Medium | Pin to specific API version, use v1 fallback |
| AWS SDK version drift | Medium | Pin `aws-sdk-bedrockruntime` to exact minor |
| OpenAI deprecates Chat Completions API | Low | Responses API is additive; Chat Completions still supported |
| Gemini 2.5 models retire (Oct 2026) | Medium | Update defaults to 3.x; 2.5 still works until retirement |
| NVIDIA NIM model availability varies | Low | Discovery-based model selection handles this |
| Rate limiting during E2E tests | Medium | Use mock providers for CI; live tests are `--ignored` |
| Anthropic changes thinking API | Low | `effort` parameter is stable; `thinking` budget still works |
| xAI image API changes | Medium | Use versioned endpoint; monitor docs |

---

## Appendix A: File Change Summary

### New Files

| File | Purpose |
|------|---------|
| `src/providers/cohere.rs` | Cohere LLM + Embed provider |
| `src/imagegen/providers/openai.rs` | OpenAI image generation |
| `src/imagegen/providers/xai.rs` | xAI Grok Imagine |
| `src/imagegen/providers/bedrock_stability.rs` | Stability AI on Bedrock |

### Modified Files

| File | Changes |
|------|---------|
| `src/providers/mod.rs` | Register Cohere module |
| `src/providers/mistral.rs` | Vision + reasoning_effort |
| `src/providers/anthropic.rs` | `effort` parameter + model catalog |
| `src/providers/nvidia.rs` | Embeddings + tool calling |
| `src/providers/bedrock.rs` | Embeddings + model catalog |
| `src/providers/openai.rs` | Model catalog update |
| `src/providers/azure_openai.rs` | Model catalog update |
| `src/providers/gemini.rs` | Model catalog + embedding-2 |
| `src/providers/xai.rs` | Model catalog |
| `src/providers/ollama.rs` | Embed dimensions |
| `src/imagegen/mod.rs` | Export new providers |
| `src/imagegen/factory.rs` | Register new providers |
| `src/imagegen/providers/mod.rs` | Register new providers |
| `src/imagegen/providers/gemini.rs` | Gemini 3.x image models |
| `src/lib.rs` | Export Cohere, new image providers |
| `src/traits.rs` | Potentially add `embed_options` parameter |
| `Cargo.toml` | Version bump to 0.8.0 |
| `src/discovery/providers/*.rs` | Update model families |

### Test Files

| File | Purpose |
|------|---------|
| `tests/e2e_cohere.rs` | Cohere integration tests |
| `tests/e2e_openai_imagegen.rs` | OpenAI image gen tests |
| `tests/e2e_xai_imagegen.rs` | xAI image gen tests |
| `tests/e2e_nvidia_embed.rs` | NVIDIA embedding tests |
| `tests/e2e_bedrock_embed.rs` | Bedrock embedding tests |

---

## Appendix B: Dependency Changes

### No New Dependencies Required

| Feature | How |
|---------|-----|
| Cohere | Uses `reqwest` (already in deps) |
| OpenAI Image Gen | Uses `reqwest` + `async-openai` (already in deps) |
| xAI Image Gen | Uses `reqwest` (already in deps) |
| NVIDIA Embeddings | Uses `reqwest` (already in deps) |
| Bedrock Embeddings | Uses `aws-sdk-bedrockruntime` (already in deps, feature-gated) |

Zero new crate dependencies. All implementations use existing HTTP client infrastructure.

---

## Appendix C: Official Documentation Links

| Provider | API Reference | Models | Embeddings | Image Gen |
|----------|--------------|--------|------------|-----------|
| OpenAI | [API Docs](https://developers.openai.com/api/docs) | [Models](https://developers.openai.com/api/docs/models/all) | [Embeddings](https://developers.openai.com/api/docs/embeddings) | [Images](https://developers.openai.com/api/docs/images) |
| Azure | [Foundry Models](https://learn.microsoft.com/en-us/azure/foundry/) | [Models List](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure) | [Embeddings](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/embeddings) | [Image Gen](https://learn.microsoft.com/en-us/azure/ai-services/openai/dall-e-quickstart) |
| Anthropic | [Platform Docs](https://platform.claude.com/docs) | [Models](https://platform.claude.com/docs/en/about-claude/models/overview) | [Embeddings (Voyage)](https://platform.claude.com/docs/en/build-with-claude/embeddings) | N/A |
| Gemini | [API Docs](https://ai.google.dev/gemini-api/docs) | [Model Versions](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions) | [Embeddings](https://ai.google.dev/gemini-api/docs/embeddings) | [Image Gen](https://ai.google.dev/gemini-api/docs/image-generation) |
| Vertex AI | [Cloud Docs](https://docs.cloud.google.com/vertex-ai/generative-ai/docs) | [Migration Guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/migrate) | Via Gemini | [Imagen](https://cloud.google.com/vertex-ai/generative-ai/docs/image/overview) |
| Bedrock | [AWS Docs](https://docs.aws.amazon.com/bedrock/latest/userguide/) | [Endpoint Availability](https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html) | [Titan Embed](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html) | [Stability AI](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-stability-diffusion.html) |
| Mistral | [API Docs](https://docs.mistral.ai/) | [Models](https://docs.mistral.ai/getting-started/models/) | [Embeddings](https://docs.mistral.ai/api/endpoint/embeddings) | N/A (agent tool) |
| xAI | [Developer Docs](https://docs.x.ai/) | [Grok 4.3](https://docs.x.ai/developers/models/) | N/A | [Image Gen](https://docs.x.ai/developers/model-capabilities/images/generation) |
| Cohere | [API Docs](https://docs.cohere.com/) | [Models](https://docs.cohere.com/docs/models) | [Embed](https://docs.cohere.com/docs/cohere-embed) | N/A |
| OpenRouter | [API Docs](https://openrouter.ai/docs) | [Models](https://openrouter.ai/docs/guides/overview/models) | N/A | N/A |
| NVIDIA | [NIM Docs](https://docs.nvidia.com/nim/) | [build.nvidia.com](https://build.nvidia.com) | [NeMo Embeddings](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/) | [Visual GenAI](https://docs.nvidia.com/nim/visual-genai/) |
| Ollama | [API Docs](https://docs.ollama.com/api) | Local models | [Embed](https://docs.ollama.com/api/embed) | Experimental |
| LM Studio | [Developer Docs](https://lmstudio.ai/docs/developer) | Local models | [Embeddings](https://lmstudio.ai/docs/developer/openai-compat/embeddings) | N/A |
