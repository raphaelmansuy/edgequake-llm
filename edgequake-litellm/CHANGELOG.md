# Changelog — edgequake-litellm

All notable changes to this package are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
