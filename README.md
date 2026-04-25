# EdgeQuake LLM

[![Crates.io](https://img.shields.io/crates/v/edgequake-llm.svg)](https://crates.io/crates/edgequake-llm)
[![Docs.rs](https://docs.rs/edgequake-llm/badge.svg)](https://docs.rs/edgequake-llm)
[![PyPI](https://img.shields.io/pypi/v/edgequake-litellm.svg)](https://pypi.org/project/edgequake-litellm/)
[![Rust CI](https://github.com/raphaelmansuy/edgequake-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/raphaelmansuy/edgequake-llm/actions/workflows/ci.yml)
[![Python CI](https://github.com/raphaelmansuy/edgequake-llm/actions/workflows/python-ci.yml/badge.svg)](https://github.com/raphaelmansuy/edgequake-llm/actions/workflows/python-ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE-APACHE)

`edgequake-llm` is a Rust AI runtime with a single abstraction over cloud APIs,
local gateways, enterprise deployments, and testing backends. It ships
first-class support for chat, streaming, tool calling, embeddings, image
generation, caching, retries, rate limiting, cost tracking, and release-grade
CI/CD.

Python users should use [`edgequake-litellm`](edgequake-litellm/README.md), the LiteLLM-compatible package backed by this crate.

## What It Covers

- One trait-based surface for LLMs, embeddings, and Rust image generation.
- Production backends: OpenAI, Azure OpenAI, Anthropic, Gemini, Vertex AI, xAI, OpenRouter, NVIDIA NIM, Mistral, AWS Bedrock.
- Local and gateway backends: Ollama, LM Studio, GitHub Copilot direct mode (proxy optional), generic OpenAI-compatible APIs.
- Additional embedding backend: Jina.
- Image generation backends in the Rust crate: Gemini image generation, Vertex Imagen, FAL, mock image generation.
- Operational layers: caching, retry, rate limiting, cost tracking, tracing, reranking, mock providers.

## Install

```toml
[dependencies]
edgequake-llm = "0.6.14"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

`bedrock` is feature-gated:

```toml
[dependencies]
edgequake-llm = { version = "0.6.14", features = ["bedrock"] }
```

Note: the repository is now pinned to Rust 1.95.0, and the Bedrock integration is verified against the latest published AWS SDK crate set, including the current Bedrock runtime release.

Provider compatibility highlights in this release:

- Anthropic-compatible adapters now preserve final streamed tool-call deltas even when the upstream SSE stream ends without a trailing newline.
- Provider-side schema normalization now aligns Anthropic, Bedrock, and Gemini tool declarations with the stricter subsets those APIs actually accept.
- Normalized usage reporting distinguishes cache writes from cache hits where the upstream provider reports both.
- Gemini provider now preserves function-call IDs across assistant tool calls, streamed deltas, and tool-result follow-ups.
- Mistral provider now includes native audio (`speech`, `transcriptions`, `voices`) and OCR endpoint wrappers.

Latest model IDs validated in docs on 2026-04-23:

- Gemini API: `gemini-2.5-flash` (default), `gemini-2.5-pro`, `gemini-3-flash-preview`, `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite-preview`
- Vertex AI Gemini: `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3-flash-preview`, `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite-preview`
- Mistral: `mistral-small-latest` (default), `mistral-medium-latest`, `mistral-large-latest`, `magistral-small-latest`, `magistral-medium-latest`, `codestral-latest`, `devstral-latest`

## Quick Start

```rust
use edgequake_llm::{ChatMessage, LLMProvider, OpenAIProvider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = OpenAIProvider::from_env()?;
    let messages = vec![ChatMessage::user("Explain Rust ownership in one sentence.")];
    let response = provider.chat(&messages, None).await?;

    println!("{}", response.content);
    Ok(())
}
```

Environment:

```bash
export OPENAI_API_KEY=sk-...
```

## Provider Matrix

| Provider | Prefix / Type | Chat | Stream | Tools | Embeddings | Notes |
|----------|----------------|------|--------|-------|------------|-------|
| OpenAI | `openai` | Yes | Yes | Yes | Yes | GPT, o-series, vision |
| Azure OpenAI | `azure` | Yes | Yes | Yes | Yes | Deployment-based |
| Anthropic | `anthropic` | Yes | Yes | Yes | No | Claude thinking + caching |
| Gemini | `gemini` | Yes | Yes | Yes | Yes | Google AI Studio |
| Vertex AI | `vertexai` | Yes | Yes | Yes | Yes | Gemini on GCP auth |
| xAI | `xai` | Yes | Yes | Yes | No | Grok models |
| OpenRouter | `openrouter` | Yes | Yes | Yes | No | Multi-provider gateway |
| Mistral | `mistral` | Yes | Yes | Yes | Yes | La Plateforme |
| NVIDIA NIM | `nvidia` | Yes | Yes | Yes | No | OpenAI-compatible + dynamic model listing + 202 polling |
| AWS Bedrock | `bedrock` | Yes | Yes | Yes | Yes | Feature-gated |
| HuggingFace | `huggingface` | Yes | Yes | Limited | No | Inference API |
| OpenAI Compatible | `openai-compatible` | Yes | Yes | Yes | Yes | Groq, Together, DeepSeek, custom |
| Ollama | `ollama` | Yes | Yes | Yes | Yes | Local runtime |
| LM Studio | `lmstudio` | Yes | Yes | Yes | Yes | Local OpenAI-compatible |
| VSCode Copilot | `vscode-copilot` | Yes | Yes | Yes | Yes | Direct auth by default, proxy optional |
| Jina | embedding only | No | No | No | Yes | Dedicated embeddings |
| Mock | `mock` | Yes | No | Yes | Yes | Tests and offline dev |

## Image Generation Providers

Rust-only image generation support is exposed through `ImageGenProvider` and
`ImageGenFactory`:

| Provider | Type | Auth / Environment | Notes |
|----------|------|--------------------|-------|
| Gemini image generation | `GeminiImageGenProvider` | `GEMINI_API_KEY` or Vertex AI auth | Default model: `gemini-2.5-flash-image` |
| Vertex Imagen | `VertexAIImageGen` | `GOOGLE_CLOUD_PROJECT` and ADC / `GOOGLE_ACCESS_TOKEN` | Default model: `imagen-4.0-generate-001` |
| FAL | `FalImageGen` | `FAL_KEY` | Default model: `fal-ai/flux/dev` |
| Mock | `MockImageGenProvider` | none | Tests and offline development |

## Common Setup

| Provider | Required environment |
|----------|----------------------|
| OpenAI | `OPENAI_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT_NAME` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| Vertex AI | `GOOGLE_CLOUD_PROJECT` and ADC / `GOOGLE_ACCESS_TOKEN` |
| xAI | `XAI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| NVIDIA NIM | `NVIDIA_API_KEY` |
| AWS Bedrock | standard AWS credential chain plus `AWS_REGION` |
| HuggingFace | `HF_TOKEN` or `HUGGINGFACE_TOKEN` |
| OpenAI Compatible | `OPENAI_COMPATIBLE_BASE_URL`, optional `OPENAI_COMPATIBLE_API_KEY` |
| Ollama | optional `OLLAMA_HOST` |
| LM Studio | optional `LMSTUDIO_HOST` |
| VSCode Copilot | optional `VSCODE_COPILOT_PROXY_URL`; otherwise reuses the official VS Code Copilot auth cache or a fresh device login |
| Jina | `JINA_API_KEY` |

### GitHub Copilot direct mode

Use `vscode-copilot/auto` unless you have a strong reason to pin a specific model.

Why this is now the default:

- GitHub's live Auto routing knows which chat-capable model family is actually available for the current account and session.
- Some Copilot catalog entries are responses-only or temporarily throttled; Auto avoids hard-coding a brittle premium path.
- Reusing the real VS Code auth cache keeps parity with the official extension instead of depending on stale local token copies.

Legacy proxy setups still work through `VSCODE_COPILOT_PROXY_URL`, but no proxy is required for the normal path anymore.

Image generation environment:

| Provider | Required environment |
|----------|----------------------|
| Gemini image generation | `GEMINI_API_KEY` or Vertex AI auth |
| Vertex Imagen | `GOOGLE_CLOUD_PROJECT` and ADC / `GOOGLE_ACCESS_TOKEN` |
| FAL | `FAL_KEY` |

## Factory Usage

`ProviderFactory` is the fastest way to wire environments or provider/model routing:

```rust
use edgequake_llm::{ProviderFactory, ProviderType};

let (llm, embedding) = ProviderFactory::from_env()?;
println!("llm={} embedding={}", llm.name(), embedding.name());

let (vertex_llm, _) = ProviderFactory::create_with_model(
    ProviderType::VertexAI,
    Some("gemini-2.5-flash"),
)?;

let custom = ProviderFactory::create_llm_provider(
    "openai-compatible",
    "deepseek-chat",
)?;
```

For generic OpenAI-compatible routing, set:

```bash
export OPENAI_COMPATIBLE_BASE_URL=https://api.groq.com/openai/v1
export OPENAI_COMPATIBLE_API_KEY=...
```

For Rust image generation, use:

```rust,ignore
use edgequake_llm::{ImageGenFactory, ImageGenRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ImageGenFactory::from_env()?;
    let response = provider
        .generate(&ImageGenRequest::new("Editorial product photo on a concrete desk"))
        .await?;

    println!("generated {} image(s)", response.images.len());
    Ok(())
}
```

## Python Package

`edgequake-litellm` is the Python package in this repo. It is a drop-in LiteLLM replacement backed by the Rust runtime:

```python
import edgequake_litellm as litellm

resp = litellm.completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.choices[0].message.content)
```

Install:

```bash
pip install edgequake-litellm
```

See [`edgequake-litellm/README.md`](edgequake-litellm/README.md) for provider routing, migration notes, wheel coverage, and release instructions.
The Python package does not expose the Rust image-generation APIs yet.

## Development

Local validation:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --locked
cargo doc --workspace --no-deps --all-features
```

Python package validation:

```bash
cd edgequake-litellm
pip install . -v
pytest -q -k "not e2e"
```

## Release

Release guides:

- [`docs/providers.md`](docs/providers.md): provider-by-provider setup
- [`docs/releasing.md`](docs/releasing.md): release checklist, tags, registry setup
- [`docs/release-cycle.md`](docs/release-cycle.md): end-to-end CI/CD flow
- [`CHANGELOG.md`](CHANGELOG.md): release notes for the Rust crate
- [`edgequake-litellm/CHANGELOG.md`](edgequake-litellm/CHANGELOG.md): release notes for the Python package

Tag conventions:

- Rust crate: `vX.Y.Z`
- Python package: `py-vX.Y.Z`

Both publish workflows validate versions before publishing and can attach release artifacts to GitHub Releases.

## License

Apache-2.0. See [`LICENSE-APACHE`](LICENSE-APACHE).
