# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.6] - 2026-04-18

### Changed

- Clarified the GitHub Copilot documentation around direct auth, Auto-first routing, and the WHY behind session-based model selection so downstream consumers do not cargo-cult a brittle fixed-model setup.
- Synced the Rust and Python package metadata for `edgequake-litellm` so release automation and published artifacts now agree on the same version line.

### Fixed

- Updated the remaining VS Code Copilot endpoint assertions to match the current verified individual-account API hostname, keeping the release test suite green.

## [0.6.5] - 2026-04-18

### Fixed

- Prefer the live VS Code Copilot auth cache over stale local token copies when refreshing Copilot credentials, which improves parity with the real VS Code authentication flow.
- Default Copilot chat selection now follows official Auto mode semantics, because GitHub's own router is the source of truth for which chat-capable model and session token should be used.
- Auto-session resolution now skips responses-only models and forwards the Copilot session token when GitHub requires it, which is why EdgeCrab can now behave like the real VS Code client instead of guessing.
- Long-lived weekly/global Copilot throttles now fail fast with the upstream scope preserved, instead of being retried like short burst limits.
- EdgeCrab setup, doctor, and Copilot E2E detection now recognize VS Code’s real auth cache locations on macOS and Linux.

## [0.6.4] - 2026-04-18

### Fixed

- **Copilot model alias + Auto resolution parity.** The VS Code provider now normalizes vendor-prefixed model IDs like `copilot/gpt-4.1`, resolves `copilot/auto` from the live Copilot model catalog using server-advertised metadata only, removes heuristic fallbacks, and adds verified live E2E coverage for `gpt-5-mini` and `copilot/gpt-4.1` while preserving genuine upstream weekly-rate-limit responses verbatim.

## [0.6.3] - 2026-04-17

### Changed

- **VS Code Copilot parity refresh.** The direct Copilot provider now mirrors the modern VS Code client more closely with current request headers, interaction IDs, richer rate-limit classification, better token import discovery, and a default chat model of GPT-5 mini.

### Fixed

- **Actionable Copilot throttling diagnostics.** Weekly and long-window rate limits now preserve the upstream error code, scope, and retry window instead of collapsing into a generic retry loop.
- **Cross-platform token discovery for Copilot auth.** The token manager now checks the real GitHub Copilot auth cache locations used on macOS and Linux, improving out-of-the-box direct-mode authentication.

## [0.6.1] - 2026-06-15

### Changed

- **Default Ollama LLM model changed to `gemma4:latest`** (First Principle).
  `DEFAULT_OLLAMA_MODEL` is now `"gemma4:latest"` (was `"gemma3:12b"`). This
  aligns the Ollama provider default with the project-wide `DEFAULT_LLM_MODEL`
  constant. The default embedding model remains `embeddinggemma:latest`.

## [0.6.0] - 2026-06-15

### Added

- **Ollama Cloud API key support (Issue #162).** New `OLLAMA_API_KEY` environment
  variable enables Bearer token authentication for Ollama Cloud and any Ollama
  endpoint requiring auth. Builder method: `.api_key("key")`. Proxy bypass
  (`no_proxy`) now only applies to localhost connections.
- **Embedding batch size control (Issue #165).** New `embed_batched()` method on
  `EmbeddingProvider` trait automatically splits large embedding requests into
  chunks. Configurable via `EDGEQUAKE_EMBEDDING_BATCH_SIZE` environment variable
  (default: 2048). Prevents 422 errors from servers with batch limits (e.g.,
  HuggingFace TEI default of 32).
- `max_batch_size()` method on `EmbeddingProvider` trait for provider-specific
  batch size configuration.

### Fixed

- **Lenient embedding deserialization (Issue #164).** OpenAI provider now uses a
  manual HTTP client for embeddings instead of async-openai's strict types. This
  supports HuggingFace TEI, Infinity, FastEmbed, and other OpenAI-compatible
  embedding servers that omit the cosmetic `object` field in responses.

## [0.5.1] - 2026-04-05

### Fixed

- **Streaming usage propagation now survives terminal chunks across providers.**
  When a provider supplies authoritative usage metadata only at the end of a
  stream, the Rust core now carries that usage on the terminal stream chunk
  instead of dropping it before downstream consumers can persist or display it.
- **LiteLLM Python fallback streaming now preserves final usage accounting.**
  The synthetic `StreamChunk::Finished` produced from the completed fallback
  response now includes prompt, completion, cache-hit, and thinking token
  counts so Python streaming consumers receive the same authoritative usage as
  non-streaming calls.

## [0.5.0] - 2026-04-04

### Added

- **Image generation API in the Rust crate** via `ImageGenProvider`,
  `ImageGenFactory`, `ImageGenRequest`, and `ImageGenResponse`.
- **Image generation provider support** for Gemini image generation, Vertex
  Imagen, FAL, and a mock image backend for local tests.
- **xAI Grok 4.20 model coverage** for the latest Grok release line.

### Fixed

- **Bedrock model resolution and request parsing** were hardened after the
  `0.4.0` release, including stricter Converse request handling and clearer
  failures for invalid model IDs.
- **OpenAI / Azure compatibility** now preserves tool call formatting,
  `ToolChoice::Function`, temperature guards, and rate-limit classification
  based on provider error codes instead of brittle message matching.
- **OpenAI-compatible gateways** now handle embeddings, reasoning-effort
  settings, and LM Studio / Ollama compatibility more consistently.
- **OpenRouter, Anthropic, Ollama, and Mistral provider audits** closed API
  compliance gaps around tool call history, JSON mode, duplicate streaming
  terminals, thread-safe API key handling, and safe prompt behavior.

### Changed

- **AWS Bedrock internals** moved further toward a typed Converse-based path,
  with multimodal handling, thinking support, retry behavior, and AWS SDK
  updates aligned with the post-`0.4.0` history.
- **Provider and release documentation** now matches the actual published
  history and the full current provider surface, including Rust-only image
  generation providers and the tag-driven crates.io / PyPI flow.

## [0.4.0] - 2026-04-04

### Added

- **`ProviderType::VertexAI`** — new distinct enum variant separate from `Gemini`; strings
  `"vertex"` / `"vertexai"` now parse to `VertexAI` instead of `Gemini` (fixes #31).
- **`vertex_host()` helper** — returns `aiplatform.googleapis.com` (global endpoint) so
  Gemini-3.x models no longer fail with regional endpoint errors (fixes #26, #37).
- **`default_thinking_config()`** — Gemini-3 uses `thinking_level`, Gemini-2.5 uses
  `thinking_budget`; applied automatically in chat / stream / tool-stream paths (fixes #28).
- **`thought_signature: Option<String>`** added to `ToolCall` struct — propagated across
  all providers and serialised only when `Some` (fixes #41, #42).
- **`thought_signature: Option<String>`** added to `StreamChunk::ToolCallDelta` — enables
  cross-chunk thought-signature propagation during streaming (fixes #42).
- **xAI timeout raised to 600 s** for Grok 4 / Grok 4-latest deep-reasoning models that
  may stream for several minutes before the first token arrives (fixes xai long-reasoning).
- **Local CI script** `scripts/ci-local.sh` and guide `docs/local-ci.md` — run every CI
  check locally before pushing; supports per-job (`fmt`, `test`, `python`, …) execution.
- **`audit.toml`** — acknowledges known unmaintained transitive crates (`backoff`,
  `instant`) with no upstream fix; keeps `cargo audit` clean.
- **Release artifacts on GitHub Releases** — the crates.io publish workflow now
  attaches the packaged `.crate` file and release notes extracted from `CHANGELOG.md`.

### Fixed

- **Multi-tool streaming** — replaced `map` with `flat_map` so all `Part` values in a
  streamed response are emitted; added a monotonic per-call index counter to keep tool
  call ordering stable (fixes #33).
- **Tool result grouping** — consecutive tool results are now grouped into a single user
  `Content` with multiple `functionResponse` Parts, matching the Gemini API requirement
  (fixes #33).
- **HTTP 429 / `RESOURCE_EXHAUSTED`** — both the blocking and streaming Gemini paths now
  map this status to `LlmError::RateLimited` instead of a generic error (fixes #35).
- **VertexAI URL construction** — `build_base_url()` and `build_cache_url()` now use
  `vertex_host()` so requests always target the correct endpoint (fixes #26).
- **`thought_signature` echo in assistant turns** — `functionCall` Parts are sent back
  with the original `thought_signature` echoed from the assistant turn (fixes #41).
- **Security dependency bumps** — `aws-lc-rs` 1.16.0 → 1.16.2 (RUSTSEC-2026-0044–0048),
  `aws-lc-sys` 0.37.1 → 0.39.1, `quinn-proto` 0.11.13 → 0.11.14 (RUSTSEC-2026-0037),
  `rustls-webpki` 0.103.9 → 0.103.10 (RUSTSEC-2026-0049).
- **aarch64 wheel build CI** — replaced slow/fragile QEMU cross-compilation with native
  `ubuntu-24.04-arm` GitHub-hosted runner; `continue-on-error` safety blanket removed.

### Changed

- `VertexAI` factory arm now routes through a dedicated `VertexAI` provider variant;
  the `vertexai:` model-string prefix in the `Gemini` arm is retained for backward
  compatibility.
- Documentation was rewritten to reflect the current provider surface, environment
  variables, release tags, and dual-package CI/CD flow.

---

## [0.3.0] - 2026-03-01

### Added

**Bedrock provider — native embedding support**

- **`EmbeddingProvider` implementation**: Native embedding support via the
  `invoke_model` API (not Converse). Supports Amazon Titan Embed Text v2/v1
  and Cohere Embed English v3 / Multilingual v3 / v4.
- **`with_embedding_model()`**: Builder method to set a custom embedding model.
- **`with_embedding_dimension()`**: Builder to override auto-detected dimension.
- **Default embedding model**: `amazon.titan-embed-text-v2:0` (1024 dimensions).
- **`AWS_BEDROCK_EMBEDDING_MODEL`**: New environment variable to configure the
  embedding model at startup.
- **Batch embedding**: Cohere models use native batch embedding; Titan processes
  one text per API call.
- **Factory native embedding**: `ProviderFactory::create(Bedrock)` now returns
  native `BedrockProvider` for both LLM and embedding (no OpenAI fallback).

**Bedrock provider — comprehensive model coverage (12 providers, 30+ models)**

- **New inference profile families**: Added `deepseek.*`, `mistral.pixtral*`,
  and `writer.*` to cross-region inference profile auto-resolution.
- **Context length estimates**: Added accurate context lengths for all model
  families: Google Gemma (128K), NVIDIA Nemotron (128K), MiniMax (1M), Z.AI
  GLM (128K), OpenAI OSS (128K), Mistral Pixtral (128K), Devstral (256K).
- **54 E2E tests** across all supported model providers:
  - Amazon Nova: Lite, Micro, Pro, Nova 2 Lite/Pro
  - Anthropic Claude: 3, 3.5, 3.7, Sonnet 4, Haiku 4.5, Sonnet 4.5
  - Meta Llama: 3.2 (1B, 3B), Llama 4 Scout/Maverick
  - Mistral: Large, Pixtral Large, Magistral Small, Devstral 2, Ministral 8B
  - Google Gemma: 3 27B, 3 4B
  - NVIDIA Nemotron: Nano 12B, Nano 30B
  - Qwen: Qwen3 32B, Qwen3 Coder 30B
  - MiniMax: M2, M2.1 (reasoning models with chain-of-thought)
  - DeepSeek: R1, V3.2
  - Z.AI: GLM 4.7 Flash
  - OpenAI OSS: GPT OSS 120B
  - Cohere: Command R+
  - Writer: Palmyra X4
  - Embedding tests: Titan v2, v1, Cohere v3, v4, batch, factory
  - Tool calling: auto, required, multi-turn (with tool results)

### Fixed

**Bedrock provider — inference profile auto-resolution & error reporting**

- **Default model changed** from `anthropic.claude-3-5-sonnet-20241022-v2:0`
  (requires inference profile) to `amazon.nova-lite-v1:0` (works across all
  regions without geo-restrictions).
- **Inference profile auto-resolution**: bare model IDs (e.g.,
  `amazon.nova-lite-v1:0`) are now automatically resolved to cross-region
  inference profile IDs based on the configured AWS region (e.g.,
  `eu.amazon.nova-lite-v1:0` in `eu-west-1`, `us.amazon.nova-lite-v1:0` in
  `us-east-1`). Fully-qualified IDs, ARNs, and `global.` prefixes are passed
  through unchanged.
- **Detailed AWS error messages**: `SdkError<E>::Display` only prints generic
  "service error". Added `format_sdk_error()` helper using
  `ProvideErrorMetadata` to extract the actual error code and message from the
  Bedrock API (e.g., `"ValidationException: The provided model identifier is
  invalid."`).
- **Factory `block_on` panic**: `Handle::block_on()` panics when called from
  within a `#[tokio::test]` async context. Changed to
  `tokio::task::block_in_place(|| handle.block_on(...))` in all factory methods.
- **Blank text ContentBlock**: assistant messages with tool calls but no text
  content no longer include an empty text block (Bedrock rejects blank text
  ContentBlocks in multi-turn tool calling flows).
- **Mistral Pixtral inference profile**: `mistral.pixtral-large-2502-v1:0` now
  correctly resolves to inference profile ID (e.g., `eu.mistral.pixtral-*`).
- **MiniMax M2 empty response**: reasoning models (MiniMax M2/M2.1) use
  `reasoningContent` blocks for chain-of-thought. Tests now use sufficient
  `max_tokens` to allow the model to complete reasoning + text output.

## [0.2.9] - 2026-03-01

### edgequake-llm (Rust crate)

#### Added

**AWS Bedrock provider — Converse API (Issue #20)**

*Provider implementation* (`src/providers/bedrock.rs`)
- `BedrockProvider` — new LLM provider backed by the AWS Bedrock Runtime
  Converse API, gated behind the `bedrock` Cargo feature flag.
- `BedrockProvider::from_env()` — reads AWS credentials from the standard
  credential chain (env vars, `~/.aws/credentials`, IAM roles, SSO).
- `BedrockProvider::with_model(model_id)` — sets the Bedrock model ID
  (e.g. `anthropic.claude-3-5-sonnet-20241022-v2:0`, `amazon.titan-text-express-v1`).
- `BedrockProvider::with_max_context_length(len)` — overrides the default
  context window (200 000 tokens).
- Full `LLMProvider` trait implementation: `complete()`, `complete_with_options()`,
  `chat()`, `chat_with_tools()`, `stream()`.
- Streaming via `converse_stream()` with `futures::stream::unfold` — matches
  the pattern used by other providers.
- Tool calling support with `ToolConfiguration` and `ToolChoice` mapping
  (`Auto`, `Required`, `Function`).
- Manual `json_to_document()` / `document_to_json()` converters for
  `aws_smithy_types::Document` ↔ `serde_json::Value` (Document does not
  implement serde traits).
- 34 unit tests covering message conversion, inference config, tool config,
  JSON ↔ Document roundtrip, content extraction, stop-reason mapping.

*Feature flag*
- `bedrock = ["dep:aws-sdk-bedrockruntime", "dep:aws-config", "dep:aws-smithy-types"]`
  — opt-in with zero cost when disabled.
- Dependencies: `aws-sdk-bedrockruntime` v1, `aws-config` v1
  (`behavior-version-latest`), `aws-smithy-types` v1.

*ProviderFactory integration*
- `ProviderType::Bedrock` — new variant parsed from `"bedrock"`, `"aws-bedrock"`,
  `"aws_bedrock"` (case-insensitive).
- `ProviderFactory::create(ProviderType::Bedrock)` and
  `ProviderFactory::create_with_model(ProviderType::Bedrock, Some("model-id"))`.
- Embedding provider returns mock fallback with warning (Bedrock embedding
  support is a future enhancement).

*E2E tests* (`tests/e2e_bedrock.rs`)
- 18 integration tests: basic chat, complete, complete with options, multi-turn,
  system prompt, streaming, tool calling (auto/required/multi-turn), provider
  metadata, `with_model()`, edge cases (empty messages, long prompt, max tokens
  1, temperature 0, stop sequences), factory create.
- All tests `#[ignore]`-gated on AWS credentials with Bedrock access.

#### Fixed
- **Reasoning model false positives for Qwen3-VL and Qwen3-VL-Embedding
  (Issue #19):** `is_reasoning_model()` used overly broad `contains("qwen3")`
  which matched all Qwen3 variants, including non-reasoning models like
  `qwen3-vl-embedding-2b` and `qwen3-vl-4b`. LM Studio rejected requests with
  "Model does not support reasoning configuration." Replaced with explicit
  size-specific allowlist (`qwen3-30b`, `qwen3-14b`, etc.) that excludes VL,
  embedding, and coder variants. Added regression tests for false-positive cases.

#### Documentation
- Provider count updated from 12 to 13 in README.md.
- Bedrock added to Supported Providers table in README.md.
- Provider Setup section includes AWS Bedrock configuration guide.
- Feature flag usage documented: `edgequake-llm = { version = "0.2", features = ["bedrock"] }`.

## [0.2.8] - 2026-02-23

### edgequake-llm (Rust crate)

#### Added

**Gemini & Vertex AI — full feature parity**

*Provider enhancements*
- `GeminiProvider::with_embedding_dimension(dim)` — builder method to request
  a specific output dimension (128–3072) when using `gemini-embedding-001`.
- `output_dimensionality` is now forwarded to the API in every embed call
  (single, batch, VertexAI `:predict`) when a custom dimension is configured.
- `GeminiProvider::get_access_token_from_gcloud()` now falls back to
  `gcloud auth application-default print-access-token` (ADC) before failing,
  enabling service-account and CI/CD authentication without interactive login.
- Updated model table covers all GA and preview releases:
  `gemini-1.0-pro`, `gemini-1.5-flash/pro`, `gemini-2.0-flash/flash-lite`,
  `gemini-2.5-flash/pro/flash-lite`, `gemini-3.0-flash/pro`,
  `gemini-3.1-pro`, `gemini-embedding-001`, `text-embedding-004/005`.

*Examples — Gemini (Google AI endpoint)*
- `examples/gemini/demo.rs` (`gemini_demo`) — full API walkthrough: completion,
  chat, streaming, tool calling, JSON mode, vision, embeddings, model listing.
- `examples/gemini/chat.rs` (`gemini_chat`) — Q&A, system prompts, multi-turn,
  temperature sweep, JSON mode.
- `examples/gemini/streaming.rs` (`gemini_streaming`) — text stream, thinking
  content display, tool-call deltas, TTFT measurement.
- `examples/gemini/vision.rs` (`gemini_vision`) — base64 PNG/JPEG, multi-image
  comparison, vision+JSON classification.
- `examples/gemini/embeddings.rs` (`gemini_embeddings`) — single / batch,
  semantic search, similarity matrix, custom dimensions, clustering.
- `examples/gemini/tool_calling.rs` (`gemini_tool_calling`) — single tool,
  multi-tool selection, forced tool choice, multi-step tool use.

*Examples — Vertex AI endpoint*
- `examples/vertexai/demo.rs` (`vertexai_demo`) — full API walkthrough
  (completion, chat, streaming, tools, JSON, vision, embeddings, thinking).
- `examples/vertexai/chat.rs` (`vertexai_chat`) — conversations, personas,
  temperature sweep, JSON mode.
- `examples/vertexai/streaming.rs` (`vertexai_streaming`) — text stream,
  thinking content, tool-call deltas, TTFT measurement.
- `examples/vertexai/vision.rs` (`vertexai_vision`) — base64 PNG/JPEG,
  multi-image, vision+JSON, explicit model selection.
- `examples/vertexai/embeddings.rs` (`vertexai_embeddings`) — single/batch via
  `:predict` endpoint, semantic search, similarity matrix, custom dims.
- `examples/vertexai/tool_calling.rs` (`vertexai_tool_calling`) — single/multi
  tools, forced choice, multi-step with result feeding, complex schemas.

#### Fixed
- `ProviderFactory::create_embedding_provider("gemini", …)` test updated to
  accept both `"gemini"` (when `GEMINI_API_KEY` is present) and `"mock"
  (fallback when no credentials are available).

#### Documentation
- `examples/README.md` — added Gemini and Vertex AI sections with env-var setup
  and per-example descriptions.
- `docs/providers.md` — expanded Gemini section: full model table, embedding
  models, `GOOGLE_ACCESS_TOKEN` env var, ADC authentication note.


### edgequake-llm (Rust crate)

#### Added

**Azure OpenAI provider — complete official-crate rewrite**
- Provider now uses the `async-openai` `AzureConfig` type (via the `Config` trait)
  instead of hand-rolled HTTP, eliminating ~400 lines of bespoke request building.
- **`AzureOpenAIProvider::from_env_contentgen()`** — reads the
  `AZURE_OPENAI_CONTENTGEN_*` naming scheme used by enterprise deployments
  (`AZURE_OPENAI_CONTENTGEN_API_ENDPOINT`, `AZURE_OPENAI_CONTENTGEN_API_KEY`,
  `AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT`, `AZURE_OPENAI_CONTENTGEN_API_VERSION`).
- **`AzureOpenAIProvider::from_env_auto()`** — tries `from_env_contentgen()` first
  and falls back to the standard `from_env()` (`AZURE_OPENAI_*`) transparently.
- **`AzureOpenAIProvider::with_deployment(name)`** — builder method to override the
  active chat deployment at runtime (mirrors `with_embedding_deployment()`).

**Azure full E2E test suite** (`tests/e2e_azure.rs`) — 25 tests covering:
  - Basic chat (`chat()`, `complete()`, `complete_with_options()`)
  - JSON mode, streaming, streaming tool calls
  - Vision / multimodal (base64 JPEG, URL images, detail levels)
  - Tool / function calling
  - Embeddings (single and batch; graceful skip when deployment lacks embed support)
  - Cache-hit token extraction, provider metadata assertions
  - Factory auto-detection and `ProviderType::AzureOpenAI` explicit creation
  - Env-var isolation via `ENV_MUTEX` for safe parallel test execution

**`ProviderFactory` Azure support**
- `ProviderType::AzureOpenAI` — new variant parsed from `"azure"`, `"azure-openai"`,
  `"azure_openai"`, `"azureopenai"` (case-insensitive).
- `ProviderFactory::create(ProviderType::AzureOpenAI)`,
  `create_with_model(AzureOpenAI, Some("deployment"))`,
  `create_azure_openai()`, `create_azure_openai_with_deployment()`
- `ProviderFactory::from_env()` now auto-detects Azure when
  `AZURE_OPENAI_CONTENTGEN_API_KEY` (or `AZURE_OPENAI_API_KEY`) plus endpoint are set.
- `ConfigProviderType::Azure` fully wired in `ProviderFactory::from_config()`.
- 7 new Azure unit tests inside `src/factory.rs`.

**`ImageData` URL-native helpers** (`src/traits.rs`)
- `ImageData::from_url(url)` — stores a URL directly (sets `mime_type = "url"`).
- `ImageData::is_url() -> bool` — detects whether the image is a URL vs base64.
- `ImageData::to_api_url() -> String` — returns the raw URL directly when
  `is_url()`, otherwise wraps base64 data in a `data:<mime>;base64,…` URI.
- Both `openai.rs` and `azure_openai.rs` `build_user_content` now use
  `img.to_api_url()` — URL images are passed directly to the API without
  unnecessary base64 wrapping.

**OpenAI provider improvements** (`src/providers/openai.rs`)
- `OpenAIProvider::from_env()` — new constructor that reads `OPENAI_API_KEY`
  (and optionally `OPENAI_BASE_URL`) with dotenvy support.
- `OpenAIProvider::supports_json_mode()` now returns `true` for all models with
  prefixes `gpt-4`, `gpt-3.5-turbo`, `gpt-5`, `o1`, `o3`, `o4`.
- Content-filter guardrail: responses with `finish_reason = "content_filter"` are
  surfaced as an `LlmError::ApiError` with a descriptive message.

**Examples reorganised into provider subfolders**
- `examples/openai/` — `basic_completion.rs`, `chatbot.rs`, `demo.rs`,
  `embeddings.rs`, `streaming.rs`, `tool_calling.rs`, `vision.rs`
- `examples/azure/` — `env_check.rs`, `full_demo.rs`
- `examples/mistral/` — `chat.rs`
- `examples/local/` — `local_llm.rs`
- `examples/advanced/` — `cost_tracking.rs`, `middleware.rs`, `multi_provider.rs`,
  `reranking.rs`, `retry_handling.rs`
- `examples/README.md` — clean 130-line provider-organised reference

**OpenAI example improvements** (now part of `openai/` subfolder)
- `openai/demo.rs` — 8 sections: provider from env, simple completion, multi-turn
  chat, streaming, tool calling, JSON mode, vision (URL image via `gpt-4o-mini`),
  model families reference. `temperature` omitted for gpt-5-mini compatibility.
- `openai/tool_calling.rs` — removed `temperature: Some(0.0)` incompatible with
  gpt-5-mini; added explanatory comment.
- `openai/vision.rs` — 5 sections: URL image, base64 image, multiple images, detail
  level comparison (low vs high), vision + JSON classification. Uses `gpt-4o-mini`.
  All images sourced from Azure's own sample dataset (no rate limits, no content filter).

**Azure full demo** (`examples/azure/full_demo.rs`)
- Section 6a: URL image via `landmark.jpg` (Azure sample, no content filter).
- Section 6b: base64 image downloaded from same URL.
- Section 7: content-filter demo — 7a intentionally triggers Azure content filter
  with `faces.jpg`; 7b uses `AZURE_OPENAI_NO_FILTER_DEPLOYMENT_NAME` to bypass.

**Documentation**
- `docs/providers.md` Azure section rewritten: 3-constructor table, CONTENTGEN env
  vars, programmatic builder pattern, `with_deployment()` runtime switching,
  content-filter notes, reliable Azure sample image URLs.
- Azure Vision updated to `Y` in the feature comparison table.

#### Fixed
- **`test_from_env_fallback_to_mock` flaky in full suite** (`src/factory.rs`) — test
  now clears all Azure env var variants (`AZURE_OPENAI_CONTENTGEN_*`, `AZURE_OPENAI_*`)
  before asserting mock fallback. Root cause: `from_env()` checks CONTENTGEN vars
  first; a prior serial test left them populated.

### edgequake-litellm (Python package)

#### Added
- `"azure"` provider added to `list_providers()` — callers can now pass
  `model="azure/<deployment-name>"` to route to Azure OpenAI.
- Error message for unknown providers updated to include `azure`.
- New example `examples/09_azure_openai.py` — 5 sections: provider auto-detection,
  `azure/<deployment>` chat, JSON mode, streaming, `list_providers` assertion.
- `examples/README.md` updated with example 09 in table and quick-start section.

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
