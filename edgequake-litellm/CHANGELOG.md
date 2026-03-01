# Changelog — edgequake-litellm

All notable changes to this package are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-01

### Added

- **AWS Bedrock provider support** — pass `model="bedrock/<model-id>"` to route
  completions and embeddings to AWS Bedrock Runtime (Converse API).
  - Supports 12 model providers: Amazon Nova, Anthropic Claude, Meta Llama,
    Mistral, Google Gemma, NVIDIA Nemotron, Qwen, MiniMax, DeepSeek, Z.AI,
    OpenAI OSS, Cohere, Writer.
  - Inference profile auto-resolution: bare model IDs automatically resolve to
    cross-region inference profile IDs based on the configured AWS region.
  - Native Bedrock embedding: supports Amazon Titan Embed Text v2/v1 and
    Cohere Embed English v3 / Multilingual v3 / v4.
  - Credentials read from standard AWS chain (env vars, `~/.aws/credentials`,
    IAM roles, SSO).
- `"bedrock"` added to `list_providers()` return value.
- Error message for unknown provider names now includes `bedrock` in the list.

### Changed

- **Rust backend upgraded to `edgequake-llm` v0.3.0**, which includes:
  - Full AWS Bedrock provider with native embedding support.
  - Default Bedrock model changed to `amazon.nova-lite-v1:0` (works in all regions).
  - 54 E2E tests across all supported Bedrock model providers.
  - Inference profile auto-resolution for cross-region model access.
  - Detailed AWS error reporting via `format_sdk_error()`.

## [0.1.4] - 2026-02-23

### Changed

- Tracks **edgequake-llm v0.2.8** — picks up all Gemini / Vertex AI improvements:
  - `gemini/<model>` and `vertexai/<model>` routing now benefits from full model
    coverage (Gemini 3.0 Flash/Pro, Gemini 3.1 Pro) and `gemini-embedding-001`
    custom-dimension support.
  - `gcloud auth application-default` ADC fallback for Vertex AI credentials in
    CI/CD environments.

## [0.1.3] - 2026-02-22

### Added

- **Azure OpenAI provider support** — pass `model="azure/<deployment-name>"` to route
  completions to Azure OpenAI Service.  Credentials are read from the standard
  `AZURE_OPENAI_*` or enterprise `AZURE_OPENAI_CONTENTGEN_*` environment variables.
- `"azure"` added to `list_providers()` return value.
- Error message for unknown provider names now includes `azure` in the list of valid options.
- **New example `examples/09_azure_openai.py`** — demonstrates:
  - Provider auto-detection (`detect_provider()`)
  - `azure/<deployment>` simple chat
  - JSON mode (`response_format={"type": "json_object"}`)
  - Streaming
  - `list_providers()` assertion
- `examples/README.md` updated to include example 09 in the table and quick-start section.

### Changed

- **Rust backend upgraded to `edgequake-llm` 0.2.7**, which includes the full Azure
  OpenAI rewrite on top of `async-openai AzureConfig`, `ImageData::from_url()` /
  `to_api_url()`, `OpenAIProvider::from_env()`, and all example reorganisation.



### Fixed

- **Vision images now forwarded correctly for Ollama, LM Studio, and OpenRouter providers.**
  Backed by `edgequake-llm 0.2.6` which resolves Issue #15:
  - Ollama: base64 image strings now sent in the `images` array field.
  - LM Studio / OpenRouter: content changed from plain `String` to multimodal
    content-parts (`serde_json::Value`), enabling image URL parts.
  - OpenAI: `temperature=1.0` no longer forwarded to gpt-4.1-nano / o-series
    that reject non-default temperature values.

### Changed

- **Rust backend upgraded to `edgequake-llm` 0.2.6.**

## [0.1.1] - 2026-02-21

### Changed

- **Rust backend upgraded to `edgequake-llm` 0.2.5**, which upgrades `async-openai`
  from 0.24 to 0.33.0 internally.

### Fixed

- **`max_completion_tokens` now works for all OpenAI model families.**
  Models that only accept `max_completion_tokens` (not the deprecated `max_tokens`) —
  including `o1`, `o3-mini`, `o4-mini`, `gpt-4.1`, `gpt-4.1-nano` — now receive
  the parameter correctly.  Previously these models would return a 400 Bad Request error.

### Added

- `resp.cache_hit_tokens` convenience property on `ModelResponseCompat`.
  Returns the number of tokens served from the provider cache
  (`None` if the provider does not support caching or no cache hit occurred).
  Equivalent to `resp.usage.cache_read_input_tokens`.

- `resp.thinking_tokens` convenience property on `ModelResponseCompat`.
  Returns the number of reasoning/thinking tokens used by the model
  (OpenAI o-series `reasoning_tokens`, Anthropic thinking blocks), or `None`.
  Equivalent to `resp.usage.reasoning_tokens`.

- Both new properties are included in `resp.to_dict()` output.

## [0.1.0] - 2026-02-20

### Added

- Initial release — drop-in LiteLLM replacement backed by Rust.
- `completion()` / `acompletion()` — sync and async chat completions.
- `embedding()` / `aembedding()` — text embeddings.
- `stream()` — async generator streaming.
- `acompletion(stream=True)` — litellm-style streaming.
- Multi-provider routing via `provider/model` strings:
  OpenAI, Anthropic, Gemini, Mistral, OpenRouter, xAI, Ollama, LM Studio, HuggingFace, Mock.
- `ModelResponseCompat` with litellm-compatible `resp.choices[0].message.content` access.
- `EmbeddingResponseCompat` with `resp.data[0].embedding` and list access.
- `StreamChunkCompat` with `chunk.choices[0].delta.content`.
- `stream_chunk_builder()` for reconstructing full response from stream.
- Full type stubs (`py.typed`, `_elc_core.pyi`).
- ABI3 wheels for Python 3.9-3.13+ (one wheel per platform).
- Zero runtime Python dependencies.
