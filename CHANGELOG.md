# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
