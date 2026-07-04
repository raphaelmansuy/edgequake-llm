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
- Production backends: [OpenAI](https://platform.openai.com/docs/models), [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/), [Anthropic](https://docs.anthropic.com/en/docs/about-claude/models), [Gemini](https://ai.google.dev/gemini-api/docs/models), [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models), [xAI](https://docs.x.ai/docs/models), [OpenRouter](https://openrouter.ai/docs/models), [NVIDIA NIM](https://docs.api.nvidia.com), [Mistral](https://docs.mistral.ai/getting-started/models/), [AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html).
- Local and gateway backends: [Ollama](https://github.com/ollama/ollama), [LM Studio](https://lmstudio.ai/docs/api), GitHub Copilot direct mode (proxy optional), generic OpenAI-compatible APIs.
- Additional embedding backend: [Jina](https://jina.ai/embeddings/).
- Image generation backends in the Rust crate: Gemini image generation, Vertex Imagen, [FAL](https://fal.ai), mock image generation.
- **Model Discovery** — programmatic capability discovery across 11 providers with zero heuristics.
- Operational layers: caching, retry, rate limiting, cost tracking, tracing, reranking, mock providers.

## Install

```toml
[dependencies]
edgequake-llm = "0.7.0"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

`bedrock` is feature-gated:

```toml
[dependencies]
edgequake-llm = { version = "0.7.0", features = ["bedrock"] }
```

Note: the repository is now pinned to Rust 1.95.0, and the Bedrock integration is verified against the latest published AWS SDK crate set, including the current Bedrock runtime release.

Provider compatibility highlights in this release:

- Anthropic-compatible adapters now preserve final streamed tool-call deltas even when the upstream SSE stream ends without a trailing newline.
- Provider-side schema normalization now aligns Anthropic, Bedrock, and Gemini tool declarations with the stricter subsets those APIs actually accept.
- Normalized usage reporting distinguishes cache writes from cache hits where the upstream provider reports both.
- Gemini provider now preserves function-call IDs across assistant tool calls, streamed deltas, and tool-result follow-ups.
- Mistral provider now includes native audio (`speech`, `transcriptions`, `voices`) and OCR endpoint wrappers.

Latest model IDs validated on 2026-07-04 from official provider documentation:

- **OpenAI** ([docs](https://platform.openai.com/docs/models)): `gpt-5.5`, `gpt-5.4`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o`, `o3`, `o4-mini`
- **Anthropic** ([docs](https://docs.anthropic.com/en/docs/about-claude/models)): `claude-fable-5`, `claude-opus-4-8`, `claude-sonnet-5`, `claude-opus-4-7`, `claude-sonnet-4-6`
- **Gemini** ([docs](https://ai.google.dev/gemini-api/docs/models)): `gemini-3.5-flash`, `gemini-2.5-flash` (default), `gemini-2.5-pro`, `gemini-3.1-pro-preview`
- **Mistral** ([docs](https://docs.mistral.ai/getting-started/models/)): `mistral-small-latest` (default), `mistral-large-latest`, `codestral-latest`, `magistral-medium-latest`
- **xAI** ([docs](https://docs.x.ai/docs/models)): `grok-4`, `grok-3`, `grok-3-mini`

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

## Header Propagation

All five major providers support caller-supplied HTTP headers for B2B/multi-tenant deployments
(closes [edgequake#132](https://github.com/raphaelmansuy/edgequake/issues/132)):

| Provider | Since |
|----------|-------|
| `OpenAICompatibleProvider` | 0.6.16 |
| `MistralProvider` | 0.6.16 |
| `AnthropicProvider` | 0.6.17 |
| `GeminiProvider` | 0.6.17 |
| `NvidiaProvider` | 0.6.17 |

Use `with_extra_headers()` to inject custom headers — trace IDs, tenant identifiers, HMAC
tokens, or `traceparent` — into every outgoing LLM API call:

```rust
use edgequake_llm::{AnthropicProvider, GeminiProvider, MistralProvider};

// Anthropic
let provider = AnthropicProvider::from_env()?
    .with_extra_headers([
        ("x-request-id".to_string(), "req-abc-123".to_string()),
        ("x-tenant-id".to_string(), "tenant-42".to_string()),
    ]);

// Gemini (Google AI or VertexAI)
let provider = GeminiProvider::from_env()?
    .with_extra_headers([
        ("x-correlation-id".to_string(), "corr-xyz".to_string()),
    ]);

// Mistral
let provider = MistralProvider::from_env()?
    .with_extra_headers([
        ("traceparent".to_string(), "00-abc123-def456-01".to_string()),
    ]);
```

**Reserved headers** (`authorization`, `x-api-key`, `anthropic-version`, `content-type`,
`content-length`, `host`, `user-agent`) are silently dropped to prevent accidental credential
overrides. All other headers pass through to every request made by that provider instance.

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

## Model Discovery

Programmatic model discovery across all providers — no hardcoded model IDs, no name-pattern heuristics:

```rust
use edgequake_llm::{ModelDiscoveryService, CapabilityFilter};

let service = ModelDiscoveryService::new();

// Find all reasoning models with 100K+ context
let filter = CapabilityFilter {
    requires_thinking: Some(true),
    min_context_length: Some(100_000),
    ..Default::default()
};
let models = service.find_models(&filter).await?;
```

Python:

```python
import edgequake_litellm as litellm

# Find vision-capable models
models = litellm.discovery.find_models(requires_vision=True, requires_tools=True)
for m in models:
    print(f"  {m.provider}/{m.id}: {m.context_length} tokens")
```

Features:
- **11 providers** with dynamic, hybrid, or static discovery strategies
- **Per-provider caching** with configurable TTL and stale-on-error fallback
- **Capability filtering** — vision, tools, thinking, context length, cost, deprecation
- **Zero heuristics** — capabilities from API responses or cited documentation only
- **Static registry** — 30+ models with verified data for offline/fallback use

See [`docs/discovery.md`](docs/discovery.md) for full documentation.

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

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/discovery.md`](docs/discovery.md) | Model discovery system architecture and usage |
| [`docs/providers.md`](docs/providers.md) | Provider-by-provider setup and model tables |
| [`docs/architecture.md`](docs/architecture.md) | System design and trait architecture |
| [`docs/caching.md`](docs/caching.md) | Response caching (prompt + completion) |
| [`docs/cost-tracking.md`](docs/cost-tracking.md) | Session-level cost tracking and budgets |
| [`docs/rate-limiting.md`](docs/rate-limiting.md) | Per-provider rate limiting |
| [`docs/observability.md`](docs/observability.md) | OpenTelemetry integration |
| [`docs/reranking.md`](docs/reranking.md) | BM25, RRF, and cross-encoder reranking |
| [`docs/testing.md`](docs/testing.md) | Testing guide (mock providers, e2e) |
| [`docs/security.md`](docs/security.md) | API key management and security |
| [`docs/migration-guide.md`](docs/migration-guide.md) | Version migration guide |
| [`docs/faq.md`](docs/faq.md) | Troubleshooting and FAQ |

## Specifications

Design specifications are tracked in [`specs/001-edgequake-llm/`](specs/001-edgequake-llm/00-INDEX.md):

| # | Document | Purpose |
|---|----------|---------|
| 01 | [5-WHY Analysis](specs/001-edgequake-llm/01-FIVE-WHY-ANALYSIS.md) | Root-cause analysis |
| 02 | [Provider Conformance Audit](specs/001-edgequake-llm/02-PROVIDER-CONFORMANCE-AUDIT.md) | Gap analysis per provider |
| 03 | [Model Discovery API](specs/001-edgequake-llm/03-MODEL-DISCOVERY-API.md) | Core discovery API design |
| 04 | [Provider Discovery Approaches](specs/001-edgequake-llm/04-PROVIDER-DISCOVERY-APPROACHES.md) | Per-provider strategy |
| 05 | [Architecture & Implementation](specs/001-edgequake-llm/05-ARCHITECTURE-IMPLEMENTATION.md) | DRY/SOLID plan |
| 06 | [Model Capability Registry](specs/001-edgequake-llm/06-MODEL-CAPABILITY-REGISTRY.md) | Type system design |
| 07 | [Edge Cases & Migration](specs/001-edgequake-llm/07-EDGE-CASES-MIGRATION.md) | Compatibility and edge cases |
| 08 | [Research Findings](specs/001-edgequake-llm/08-RESEARCH-FINDINGS-JULY-2026.md) | Ground-truth corrections |
| 09 | [Implementation Plan](specs/001-edgequake-llm/09-IMPLEMENTATION-PLAN-FINAL.md) | Phased plan with roadblocks |

## Release

Release guides:

- [`docs/releasing.md`](docs/releasing.md): release checklist, tags, registry setup
- [`docs/release-cycle.md`](docs/release-cycle.md): end-to-end CI/CD flow
- [`CHANGELOG.md`](CHANGELOG.md): release notes for the Rust crate
- [`edgequake-litellm/CHANGELOG.md`](edgequake-litellm/CHANGELOG.md): release notes for the Python package

Tag conventions:

- Rust crate: `vX.Y.Z` → [crates.io](https://crates.io/crates/edgequake-llm)
- Python package: `py-vX.Y.Z` → [PyPI](https://pypi.org/project/edgequake-litellm/)

Both publish workflows validate versions before publishing and attach release artifacts to [GitHub Releases](https://github.com/raphaelmansuy/edgequake-llm/releases).

## License

Apache-2.0. See [`LICENSE-APACHE`](LICENSE-APACHE).
