# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.6] - 2026-02-21

### edgequake-llm (Rust crate)

#### Fixed
- **Vision images silently dropped in Ollama provider (Issue #15):** `OllamaMessage` struct
  was missing the `images` field. Added `images: Option<Vec<String>>` with
  `skip_serializing_if = "Option::is_none"` and updated `convert_messages` to extract
  raw base64 strings (without data-URI prefix) from `ChatMessage.images`, matching the
  Ollama chat API spec.
- **OpenAI `temperature` rejected by gpt-4.1-nano / o-series (Issue #15):** Provider
  unconditionally forwarded `temperature` even for models that only accept `1.0`. Added
  an `f32::EPSILON` guard at both `chat()` and `chat_with_tools_stream()` sites — the
  field is now skipped when value equals 1.0.
- **Vision images not forwarded in LM Studio provider:** `ChatMessageRequest.content`
  was typed as `String`, making multimodal content-parts impossible. Refactored to
  `serde_json::Value` with `build_content` / `build_image_part` / `map_role` helpers.
- **Vision images not forwarded in OpenRouter provider:** Same `content: String`
  anti-pattern. Refactored to `serde_json::Value` with `openrouter_build_content` /
  `openrouter_build_image_part` helpers.

#### Added
- **`tests/e2e_ollama_vision.rs`** — 3 `#[ignore]` live e2e tests (glm-ocr:latest
  verified returning "red background") + 2 inline unit tests.
- **`tests/e2e_lmstudio_vision.rs`** — 2 `#[ignore]` live e2e tests (env:
  `LMSTUDIO_VISION_MODEL`) + 2 inline unit tests.
- 8 new unit tests across Ollama, LM Studio, and OpenRouter provider modules.

## [0.2.5] - 2026-02-21

### edgequake-llm (Rust crate)

#### Changed
- **`async-openai` upgraded from 0.24 → 0.33.0** — fully adapts to the new namespaced
  type layout (`async_openai::types::chat::*`, `async_openai::types::embeddings::*`)
  and explicit feature flags (`["chat-completion", "embedding"]`).

#### Fixed
- **Issue #13 — `max_completion_tokens` for o-series and gpt-4.1 model families.**
  `CreateChatCompletionRequest` now exposes `max_completion_tokens` natively in 0.33,
  eliminating the raw-HTTP JSON injection workaround (~150 lines removed).
  Models like `o1`, `o3-mini`, `o4-mini`, `gpt-4.1`, and `gpt-4.1-nano` that only
  accept `max_completion_tokens` (not the deprecated `max_tokens`) now work correctly
  out of the box.
- `ChatCompletionTools` is now an enum (`Function(ChatCompletionTool) | Custom(...)`)
  — updated provider code accordingly, removing the obsolete `ChatCompletionToolType`.
- `ChatCompletionToolChoiceOption` variants updated:
  `Auto/Required` → `Mode(ToolChoiceOptions::Auto/Required)`.
- `OpenAIError::JSONDeserialize` now carries two arguments `(serde_json::Error, String)`;
  all match arms updated in `error.rs`.

#### Added
- **Cache-hit token extraction** via `usage.prompt_tokens_details.cached_tokens`
  populated into `LLMResponse::cache_hit_tokens`.
- **Reasoning token extraction** via `usage.completion_tokens_details.reasoning_tokens`
  populated into `LLMResponse::thinking_tokens`.
- **12 new unit tests** covering the 0.33 API changes: tool wrapping, tool choice
  serialisation, `max_completion_tokens` builder, cache/reasoning token extraction,
  `FinishReason` variants, and `JSONDeserialize` error conversion.
- **11 new e2e regression tests** (`tests/e2e_openai_033.rs`) covering:
  basic chat, `gpt-4.1-nano` with `max_completion_tokens`, system prompt, streaming,
  embeddings, tool-calling via streaming, vision/multimodal, cache-hit token extraction,
  and `complete_with_options`. All e2e tests are `#[ignore]`-gated on `OPENAI_API_KEY`.

#### Removed
- `OpenAIProvider::requires_completion_tokens_param()` — no longer needed.
- `OpenAIProvider::chat_with_completion_tokens()` — superseded by native 0.33 field.
- `OpenAIProvider::stream_with_completion_tokens()` — superseded by native 0.33 field.
- `http_client: reqwest::Client` field from `OpenAIProvider` — no longer needed.

## [0.2.4] - 2026-02-20

### edgequake-llm (Rust crate)

#### Documentation
- README: added Python package section, PyPI badge, drop-in migration guide, compatibility table, and streaming/embedding examples
- README: added `edgequake-litellm` to features list and quick-nav callout at top

### edgequake-litellm (Python package — `py-v0.1.0`, PyPI)

#### Added
- **`edgequake-litellm` 0.1.0** — drop-in LiteLLM replacement backed by Rust, published to [PyPI](https://pypi.org/project/edgequake-litellm/)
  - `pip install edgequake-litellm` — pre-built wheels for 7 platform/arch combos
  - Import as `import edgequake_litellm as litellm` — identical API to LiteLLM
  - `completion()` / `acompletion()` — sync and async chat completions
  - `embedding()` — text embeddings
  - Streaming via `acompletion(stream=True)` → `AsyncGenerator[StreamChunkCompat, None]`
  - `stream_chunk_builder(chunks)` — reconstruct full response from stream
  - `ModelResponseCompat`: `resp.choices[0].message.content`, `.id`, `.created`, `.object`, `.response_ms`, `.to_dict()`, `.finish_reason`, dict-style access
  - `StreamChunkCompat`: `.choices[0].delta.content`
  - `EmbeddingResponseCompat`: `.data[0].embedding`, list iteration/indexing
  - Extra params: `max_completion_tokens`, `seed`, `user`, `timeout`, `api_base`, `base_url`, `api_key`
  - `response_format` accepts `dict` (`{"type": "json_object"}`) in addition to `str`
  - Module globals: `set_verbose`, `drop_params`, `NotFoundError` (alias of `ModelNotFoundError`)
  - Wheels: Linux manylinux/musllinux × x86\_64/aarch64, macOS x86\_64/arm64, Windows x86\_64, sdist
  - CI: preflight → 7 platform builds → smoke tests (ubuntu/macos/windows) → PyPI publish via API token (`PYPI_API_TOKEN` secret)
  - Publish trigger: `py-v*` git tag on `python-publish.yml`
- **`litellm_study/` research documents** — LiteLLM compatibility audit and DX improvement roadmap

## [0.2.3] - 2026-07-10

### Added
- **Mistral AI provider** (`MistralProvider`): full support for Mistral La Plateforme API
  - Chat completions via OpenAI-compatible `/v1/chat/completions` endpoint
  - Text embeddings using `mistral-embed` (1024-dimensional) via native `/v1/embeddings`
  - Model listing via `GET /v1/models`
  - Streaming (SSE), tool/function calling, JSON mode
  - Auto-detected by `ProviderFactory::from_env()` when `MISTRAL_API_KEY` is set
  - Registered in `ProviderType` enum (factory and model_config)
  - `builtin_defaults()` entry with 5 models (small, medium, large, codestral, embed)
  - Closes [#7](https://github.com/raphaelmansuy/edgequake-llm/issues/7)
- **Example**: `examples/mistral_chat.rs` — chat, streaming, embeddings, model listing, tool calling
- **E2E tests**: `tests/e2e_mistral.rs` — 13 tests covering all provider capabilities

### Environment Variables (Mistral)
| Variable | Required | Default |
|----------|----------|---------|
| `MISTRAL_API_KEY` | ✅ | — |
| `MISTRAL_MODEL` | ❌ | `mistral-small-latest` |
| `MISTRAL_EMBEDDING_MODEL` | ❌ | `mistral-embed` |
| `MISTRAL_BASE_URL` | ❌ | `https://api.mistral.ai/v1` |

## [0.2.2] - 2026-02-19

### Fixed
- **OpenAI vision**: `OpenAIProvider::convert_messages()` now correctly handles the
  `ChatMessage.images` field. User messages with images are serialized as multipart
  `content` arrays (`[{type: "text", ...}, {type: "image_url", ...}]`) instead of
  silently dropping image data. Fixes [#3](https://github.com/raphaelmansuy/edgequake-llm/issues/3).
- **Azure OpenAI vision**: `AzureOpenAIProvider::convert_messages()` similarly updated
  to produce multipart `content` arrays for messages with images, enabling vision
  requests against Azure-hosted models.

### Added
- Unit tests covering multimodal message conversion in both `OpenAIProvider` and
  `AzureOpenAIProvider` (detail levels, data URI encoding, array/string content selection).

### Documentation
- Add provider-families.md: Deep comparison of OpenAI vs Anthropic vs Gemini API patterns
- Add performance-tuning.md: Latency, throughput, cost optimization strategies
- Add security.md: API key management, input validation, privacy best practices
- Add examples/README.md: Prerequisites, running instructions, and planned examples
- Expand FAQ troubleshooting section from 4 to 19 entries covering auth, rate limits, network
- Expand README documentation section with all 15 guide links organized by category
- Add See Also sections to architecture.md and providers.md
- Document image formats, tool calling, and best use cases for each provider family
- Include roadmap for provider-specific interface extensions

### Examples (12 total, expanded)
- streaming_chat.rs: Async streaming responses with real-time output
- embeddings.rs: Text embeddings and semantic similarity search
- reranking.rs: BM25 and RRF document reranking (no API key needed)
- local_llm.rs: Ollama and LM Studio local providers
- tool_calling.rs: Function calling with tool definitions
- chatbot.rs: Interactive multi-turn conversation with history
- vision.rs: Multimodal image analysis with GPT-4V/GPT-4o
- cost_tracking.rs: Session-level cost tracking and budget management
- retry_handling.rs: Error handling with retry strategies
- middleware.rs: Custom middleware for logging, metrics, and validation

### Dependencies
- Added base64 v0.22 for image encoding in vision example

### Tests
- Add factory.rs unit tests (13 new tests for embedding provider fallbacks)
- Add 41 new provider unit tests (917 → 958 total, now 971+)
- lmstudio.rs: 11 new tests (constants, builder, supports_*, EmbeddingProvider)
- gemini.rs: 14 new tests (constants, supports_*, cache_ttl, API types)
- anthropic.rs: 9 new tests (constants, streaming, cache tokens, error parsing)
- openai_compatible.rs: 7 new tests (streaming, ThinkingConfig, ResponseFormat)

## [0.2.0] - 2026-02-14

### Added
- Initial standalone release extracted from EdgeCode project
- Multi-provider LLM support (9 providers)
- Response caching with configurable TTL
- Rate limiting with exponential backoff
- Session-level cost tracking and metrics
- Reranking capabilities (BM25, RRF, hybrid strategies)
- OpenTelemetry integration for observability
- Comprehensive integration test suite

### Providers
- OpenAI (GPT-4, GPT-5 series)
- Anthropic (Claude 3+, Claude 4)
- Google Gemini (2.0, 2.5, 3.0 Flash/Pro)
- xAI (Grok 2, 3, 4 series)
- OpenRouter (616+ models from all providers)
- Ollama (local model support)
- LMStudio (local OpenAI-compatible API)
- HuggingFace Hub (open-source models)
- VSCode Copilot (GitHub Copilot API)

### Features
- ⚡ Async/await based API
- 🎯 Trait-based provider abstraction
- 💰 Token usage and cost tracking
- 📦 Response caching (memory and persistent)
- 🔄 Automatic retry with exponential backoff
- 🚦 Rate limiting per provider
- 📊 OpenTelemetry metrics and tracing
- 🔍 Advanced reranking algorithms
- 🧪 Mock provider for testing

[0.2.0]: https://github.com/raphaelmansuy/edgequake-llm/releases/tag/v0.2.0
