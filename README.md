# EdgeQuake LLM

[![Crates.io](https://img.shields.io/crates/v/edgequake-llm.svg)](https://crates.io/crates/edgequake-llm)
[![Documentation](https://docs.rs/edgequake-llm/badge.svg)](https://docs.rs/edgequake-llm)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![CI](https://github.com/raphaelmansuy/edgequake-llm/workflows/CI/badge.svg)](https://github.com/raphaelmansuy/edgequake-llm/actions)

A unified Rust library providing LLM and embedding provider abstraction with support for multiple backends, intelligent caching, rate limiting, and cost tracking.

## Features

- 🤖 **9 LLM Providers**: OpenAI, Anthropic, Gemini, xAI, OpenRouter, Ollama, LMStudio, HuggingFace, VSCode Copilot
- 📦 **Response Caching**: Reduce costs with intelligent caching (memory + persistent)
- ⚡ **Rate Limiting**: Built-in API rate limit management with exponential backoff
- 💰 **Cost Tracking**: Session-level cost monitoring and metrics
- 🔄 **Retry Logic**: Automatic retry with configurable strategies
- 🎯 **Reranking**: BM25, RRF, and hybrid reranking strategies
- 📊 **Observability**: OpenTelemetry integration for metrics and tracing
- 🧪 **Testing**: Mock provider for unit tests

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
| Anthropic | Claude 3+, 4 | ✅ | ❌ | ✅ |
| Gemini | Gemini 2.0+, 3.0 | ✅ | ✅ | ✅ |
| xAI | Grok 2, 3, 4 | ✅ | ❌ | ✅ |
| OpenRouter | 616+ models | ✅ | ❌ | ✅ |
| Ollama | Local models | ✅ | ✅ | ✅ |
| LMStudio | Local models | ✅ | ✅ | ✅ |
| HuggingFace | Open-source | ✅ | ❌ | ⚠️ |
| Copilot | GitHub models | ✅ | ❌ | ✅ |

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
- [Providers Guide](docs/providers.md) - Setup and configuration for all 11 providers
- [Architecture](docs/architecture.md) - System design and patterns
- [Examples](examples/) - Runnable code examples

### Features
- [Caching](docs/caching.md) - Response caching strategies
- [Cost Tracking](docs/cost-tracking.md) - Token usage and cost monitoring
- [Rate Limiting](docs/rate-limiting.md) - API rate limit handling
- [Reranking](docs/reranking.md) - BM25, RRF, and hybrid strategies
- [Observability](docs/observability.md) - OpenTelemetry integration

### Reference
- [Testing](docs/testing.md) - Testing strategies and mock provider
- [Migration Guide](docs/migration-guide.md) - Upgrading between versions
- [FAQ](docs/faq.md) - Frequently asked questions

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Credits

Extracted from the [EdgeCode](https://github.com/raphaelmansuy/edgecode) project, a Rust coding agent with OODA loop decision framework.
