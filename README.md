# EdgeQuake LLM

[![Crates.io](https://img.shields.io/crates/v/edgequake-llm.svg)](https://crates.io/crates/edgequake-llm)
[![Documentation](https://docs.rs/edgequake-llm/badge.svg)](https://docs.rs/edgequake-llm)
[![PyPI](https://img.shields.io/pypi/v/edgequake-litellm.svg)](https://pypi.org/project/edgequake-litellm/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE-APACHE)
[![CI](https://github.com/raphaelmansuy/edgequake-llm/workflows/CI/badge.svg)](https://github.com/raphaelmansuy/edgequake-llm/actions)

A unified Rust library providing LLM and embedding provider abstraction with support for multiple backends, intelligent caching, rate limiting, and cost tracking.

> **Python users**: see [`edgequake-litellm`](#python-package-edgequake-litellm) — a drop-in LiteLLM replacement backed by this Rust library.

## Features

- 🤖 **13 LLM Providers**: OpenAI, Anthropic, Gemini, xAI, Mistral AI, OpenRouter, Ollama, LMStudio, HuggingFace, VSCode Copilot, Azure OpenAI, AWS Bedrock, OpenAI Compatible
- 📦 **Response Caching**: Reduce costs with intelligent caching (memory + persistent)
- ⚡ **Rate Limiting**: Built-in API rate limit management with exponential backoff
- 💰 **Cost Tracking**: Session-level cost monitoring and metrics
- 🔄 **Retry Logic**: Automatic retry with configurable strategies
- 🎯 **Reranking**: BM25, RRF, and hybrid reranking strategies
- 📊 **Observability**: OpenTelemetry integration for metrics and tracing
- 🧪 **Testing**: Mock provider for unit tests
- 🐍 **Python bindings**: `edgequake-litellm` — LiteLLM-compatible Python package

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
edgequake-llm = "0.2"
tokio = { version = "1.0", features = ["full"] }
```

### Basic Usage

```rust
use edgequake_llm::{OpenAIProvider, LLMProvider, ChatMessage, ChatRole};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize provider
    let provider = OpenAIProvider::new("your-api-key", "gpt-4");

    // Create message
    let messages = vec![
        ChatMessage {
            role: ChatRole::User,
            content: "What is Rust?".to_string(),
            ..Default::default()
        }
    ];

    // Get completion
    let response = provider.complete(&messages, None).await?;
    println!("{}", response.content);

    Ok(())
}
```

## Supported Providers

| Provider | Models | Streaming | Embeddings | Tool Use |
|----------|--------|-----------|------------|----------|
| OpenAI | GPT-4, GPT-5 | ✅ | ✅ | ✅ |
| Azure OpenAI | Azure GPT | ✅ | ✅ | ✅ |
| Anthropic | Claude 3+, 4 | ✅ | ❌ | ✅ |
| Gemini | Gemini 2.0+, 3.0 | ✅ | ✅ | ✅ |
| xAI | Grok 2, 3, 4 | ✅ | ❌ | ✅ |
| Mistral AI | Mistral Small/Large, Codestral | ✅ | ✅ | ✅ |
| OpenRouter | 616+ models | ✅ | ❌ | ✅ |
| Ollama | Local models | ✅ | ✅ | ✅ |
| LMStudio | Local models | ✅ | ✅ | ✅ |
| HuggingFace | Open-source | ✅ | ❌ | ⚠️ |
| VSCode Copilot | GitHub models | ✅ | ❌ | ✅ |
| AWS Bedrock | Claude, Titan, Llama, Mistral | ✅ | ❌ | ✅ |
| OpenAI Compatible | Custom | ✅ | ✅ | ✅ |

## Examples

### Multi-Provider Abstraction

```rust
use edgequake_llm::{LLMProvider, OpenAIProvider, AnthropicProvider};

async fn try_providers() -> Result<(), Box<dyn std::error::Error>> {
    let providers: Vec<Box<dyn LLMProvider>> = vec![
        Box::new(OpenAIProvider::from_env()),
        Box::new(AnthropicProvider::from_env()),
    ];

    for provider in providers {
        println!("Testing: {}", provider.name());
        // Use provider...
    }

    Ok(())
}
```

### Response Caching

```rust
use edgequake_llm::{OpenAIProvider, CachedProvider, CacheConfig};

let provider = OpenAIProvider::from_env();
let cache_config = CacheConfig {
    ttl_seconds: 3600,  // 1 hour
    max_entries: 1000,
};

let cached = CachedProvider::new(provider, cache_config);
// Subsequent identical requests served from cache
```

### Cost Tracking

```rust
use edgequake_llm::SessionCostTracker;

let tracker = SessionCostTracker::new();

// After each completion
tracker.add_completion(
    "openai",
    "gpt-4",
    prompt_tokens,
    completion_tokens,
);

// Get summary
let summary = tracker.summary();
println!("Total cost: ${:.4}", summary.total_cost);
```

### Rate Limiting

```rust
use edgequake_llm::{RateLimitedProvider, RateLimiterConfig};

let config = RateLimiterConfig {
    max_requests_per_minute: 60,
    max_tokens_per_minute: 100_000,
};

let limited = RateLimitedProvider::new(provider, config);
// Automatic rate limiting with exponential backoff
```

## Provider Setup

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
```

```rust
let provider = OpenAIProvider::new("your-key", "gpt-4");
// or
let provider = OpenAIProvider::from_env();
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

```rust
let provider = AnthropicProvider::from_env();
```

### Gemini

```bash
export GOOGLE_API_KEY=...
```

```rust
let provider = GeminiProvider::from_env();
```

### OpenRouter

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

```rust
let provider = OpenRouterProvider::new("your-key");
```

### AWS Bedrock

Enable the `bedrock` feature flag:

```toml
edgequake-llm = { version = "0.2", features = ["bedrock"] }
```

AWS credentials are resolved via the standard credential chain (env vars, `~/.aws/credentials`, IAM roles, SSO):

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1
```

```rust
use edgequake_llm::BedrockProvider;

let provider = BedrockProvider::from_env().await
    .with_model("anthropic.claude-3-5-sonnet-20241022-v2:0");
```

### Local Providers

```rust
// Ollama (assumes running on localhost:11434)
let provider = OllamaProvider::new("http://localhost:11434");

// LMStudio (assumes running on localhost:1234)
let provider = LMStudioProvider::new("http://localhost:1234");
```

## Advanced Features

### OpenTelemetry Integration

Enable with `otel` feature:

```toml
edgequake-llm = { version = "0.2", features = ["otel"] }
```

```rust
use edgequake_llm::TracingProvider;

let provider = OpenAIProvider::from_env();
let traced = TracingProvider::new(provider, "my-service");
// Automatic span creation and GenAI semantic conventions
```

### Reranking

```rust
use edgequake_llm::{BM25Reranker, Reranker};

let reranker = BM25Reranker::new();
let results = reranker.rerank(query, documents, top_k).await?;
```

## Documentation

### API Documentation
- [Rust Docs](https://docs.rs/edgequake-llm) - Auto-generated API reference

### Guides
- [Provider Families](docs/provider-families.md) - Deep comparison of OpenAI vs Anthropic vs Gemini
- [Providers Guide](docs/providers.md) - Setup and configuration for all 13 providers
- [Architecture](docs/architecture.md) - System design and patterns
- [Examples](examples/) - Runnable code examples

### Features
- [Caching](docs/caching.md) - Response caching strategies
- [Cost Tracking](docs/cost-tracking.md) - Token usage and cost monitoring
- [Rate Limiting](docs/rate-limiting.md) - API rate limit handling
- [Reranking](docs/reranking.md) - BM25, RRF, and hybrid strategies
- [Observability](docs/observability.md) - OpenTelemetry integration

### Operations
- [Performance Tuning](docs/performance-tuning.md) - Latency, throughput, cost optimization
- [Security](docs/security.md) - API keys, input validation, privacy best practices

### Reference
- [Testing](docs/testing.md) - Testing strategies and mock provider
- [Migration Guide](docs/migration-guide.md) - Upgrading between versions
- [FAQ](docs/faq.md) - Frequently asked questions and troubleshooting

---

## Python Package: edgequake-litellm

`edgequake-litellm` is a drop-in replacement for [LiteLLM](https://github.com/BerriAI/litellm) backed by this Rust library. It exposes the same Python API as LiteLLM so existing code can migrate with a one-line import change.

[![PyPI](https://img.shields.io/pypi/v/edgequake-litellm.svg)](https://pypi.org/project/edgequake-litellm/)
[![Python](https://img.shields.io/pypi/pyversions/edgequake-litellm.svg)](https://pypi.org/project/edgequake-litellm/)

### Install

```bash
pip install edgequake-litellm
```

Pre-built wheels ship for:

| Platform | Architectures |
|----------|---------------|
| Linux (glibc) | x86\_64, aarch64 |
| Linux (musl / Alpine) | x86\_64, aarch64 |
| macOS | x86\_64, arm64 (Apple Silicon) |
| Windows | x86\_64 |

### Drop-in migration

```python
# Before
import litellm

# After — one line change
import edgequake_litellm as litellm

# All calls stay identical
response = litellm.completion(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Async & streaming

```python
import asyncio
import edgequake_litellm as litellm

async def main():
    # Async completion
    response = await litellm.acompletion(
        model="anthropic/claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Explain Rust in one sentence."}],
    )
    print(response.choices[0].message.content)

    # Streaming
    stream = await litellm.acompletion(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Count to 5"}],
        stream=True,
    )
    async for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)

asyncio.run(main())
```

### Embeddings

```python
import edgequake_litellm as litellm

response = litellm.embedding(
    model="openai/text-embedding-3-small",
    input=["Hello world", "Rust is fast"],
)
vector = response.data[0].embedding
print(f"Embedding dim: {len(vector)}")
```

### Compatibility surface

| LiteLLM feature | edgequake-litellm |
|-----------------|-------------------|
| `completion()` / `acompletion()` | ✅ |
| `embedding()` | ✅ |
| Streaming (`stream=True`) | ✅ |
| `response.choices[0].message.content` | ✅ |
| `response.to_dict()` | ✅ |
| `stream_chunk_builder(chunks)` | ✅ |
| `AuthenticationError`, `RateLimitError`, `NotFoundError` | ✅ |
| `set_verbose`, `drop_params` globals | ✅ |
| `max_completion_tokens`, `seed`, `user`, `timeout` params | ✅ |

### Source

The Python package lives in [`edgequake-litellm/`](edgequake-litellm/) and is published to [PyPI](https://pypi.org/project/edgequake-litellm/) via the [python-publish workflow](.github/workflows/python-publish.yml).

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License


Licensed under the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE)).

## Credits

Extracted from the [EdgeCode](https://github.com/raphaelmansuy/edgecode) project, a Rust coding agent with OODA loop decision framework.
