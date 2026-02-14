# Testing

EdgeQuake LLM ships with a layered testing strategy: unit tests inside each module, integration tests in `tests/`, and mock providers for deterministic verification without live API calls.

---

## Test Architecture

```text
+------------------+     +---------------------+     +-------------------+
| Unit Tests       |     | Integration Tests   |     | E2E Tests         |
| (src/**/tests)   |     | (tests/*.rs)        |     | (tests/e2e_*.rs)  |
+------------------+     +---------------------+     +-------------------+
| - Pure logic     |     | - Provider factory  |     | - Real API calls  |
| - No network     |     | - Env auto-detect   |     | - Requires keys   |
| - Deterministic  |     | - Middleware stack   |     | - Gated by env    |
+------------------+     +---------------------+     +-------------------+
        |                         |                         |
        v                         v                         v
    MockProvider           serial_test macro         Provider-specific
    MockAgentProvider      Env var isolation         Feature validation
```

---

## Running Tests

```bash
# All unit and integration tests (no API keys needed)
cargo test

# Specific module
cargo test --lib cache
cargo test --lib cost_tracker
cargo test --lib reranker

# Integration tests only
cargo test --test e2e_llm_providers
cargo test --test e2e_provider_factory

# E2E tests requiring API keys
OPENAI_API_KEY=sk-... cargo test --test e2e_llm_providers
GEMINI_API_KEY=... cargo test --test e2e_gemini
XAI_API_KEY=... cargo test --test e2e_xai

# Run with output visible
cargo test -- --nocapture
```

---

## MockProvider

`MockProvider` is a queue-based mock for basic LLM and embedding testing. It implements both `LLMProvider` and `EmbeddingProvider`.

### Features

- **Queue-based responses**: Add responses with `add_response()`, consumed in FIFO order
- **Default fallback**: Returns `"Mock response"` when queue is empty
- **Embedding support**: Returns 1536-dimensional vectors by default
- **No network**: Fully deterministic, no API calls

### Usage

```rust
use edgequake_llm::{MockProvider, LLMProvider, EmbeddingProvider};

let provider = MockProvider::new();

// Queue specific responses
provider.add_response("First answer").await;
provider.add_response("Second answer").await;

let r1 = provider.complete("question 1").await?;
assert_eq!(r1.content, "First answer");

let r2 = provider.complete("question 2").await?;
assert_eq!(r2.content, "Second answer");

// Falls back to default when queue is empty
let r3 = provider.complete("question 3").await?;
assert_eq!(r3.content, "Mock response");

// Embedding support
let embedding = provider.embed_one("test text").await?;
assert_eq!(embedding.len(), 1536);
```

---

## MockAgentProvider

`MockAgentProvider` extends the mock concept for tool-calling agent workflows. It supports deterministic tool call responses and streaming.

### Features

- **Tool call responses**: Queue responses with `ToolCall` vectors
- **Streaming support**: `chat_with_tools_stream()` emits `Content`, `ToolCallDelta`, and `Finished` chunks
- **Call counting**: Track how many responses were consumed
- **Exhaustion check**: `is_exhausted()` verifies all queued responses were used
- **Default task_complete**: Returns a `task_complete` tool call when queue is empty
- **Custom model names**: `with_model("claude-sonnet-4-20250514")` for model-specific tests
- **Sync setup**: `add_response_sync()` / `add_tool_response_sync()` for non-async test setup

### Usage

```rust
use edgequake_llm::providers::MockAgentProvider;
use edgequake_llm::traits::{ToolCall, FunctionCall, ChatMessage, LLMProvider};

let provider = MockAgentProvider::new();

// Queue a tool-calling response
provider.add_tool_response(
    "I'll create that file.",
    vec![ToolCall {
        id: "call_1".into(),
        call_type: "function".into(),
        function: FunctionCall {
            name: "write_file".into(),
            arguments: r#"{"path":"test.txt","content":"hello"}"#.into(),
        },
    }],
).await;

// Queue a final text response
provider.add_response("Done!").await;

// First call returns tool call
let r1 = provider.chat_with_tools(&messages, &tools, None, None).await?;
assert_eq!(r1.tool_calls[0].function.name, "write_file");

// Second call returns text
let r2 = provider.chat_with_tools(&messages, &tools, None, None).await?;
assert_eq!(r2.content, "Done!");

// Verify all responses consumed
assert!(provider.is_exhausted().await);
assert_eq!(provider.call_count(), 2);
```

---

## Provider Factory Tests

Tests in `tests/e2e_provider_factory.rs` verify environment-based provider auto-detection. They use the `serial_test` crate to isolate environment variables:

```rust
use serial_test::serial;

#[tokio::test]
#[serial]  // Prevents parallel env var conflicts
async fn test_provider_auto_detection_ollama() {
    std::env::set_var("OLLAMA_HOST", "http://localhost:11434");
    let (llm, embedding) = ProviderFactory::from_env().unwrap();
    assert_eq!(llm.name(), "ollama");
    std::env::remove_var("OLLAMA_HOST");
}
```

The auto-detection priority order:
1. `EDGEQUAKE_LLM_PROVIDER` (explicit override)
2. Ollama environment variables
3. OpenAI API key
4. Other provider keys
5. Mock fallback (no keys set)

---

## Testing Patterns

### Decorator Testing

Decorators like `TracingProvider`, `CachedProvider`, and `RateLimitedProvider` wrap `MockProvider` for isolated testing:

```rust
use edgequake_llm::providers::TracingProvider;

let mock = MockProvider::new();
let traced = TracingProvider::new(mock);

// Test that the decorator delegates correctly
assert_eq!(traced.name(), "mock");
let response = traced.complete("Hello").await?;
assert_eq!(response.content, "Mock response");
```

### Cache Testing

```rust
use edgequake_llm::{CachedProvider, CacheConfig, LLMCache};

let cache = LLMCache::new(CacheConfig::default());
let provider = MockProvider::new();
let cached = CachedProvider::new(provider, cache.clone());

// First call: cache miss
let r1 = cached.complete("test").await?;
assert_eq!(cache.stats().hits, 0);

// Second call: cache hit
let r2 = cached.complete("test").await?;
assert_eq!(cache.stats().hits, 1);
assert_eq!(r1.content, r2.content);
```

### Rate Limiter Testing

```rust
use edgequake_llm::{RateLimitedProvider, RateLimiterConfig};

let config = RateLimiterConfig {
    max_concurrent: 2,
    requests_per_minute: 60,
    tokens_per_minute: 100_000,
};
let provider = MockProvider::new();
let limited = RateLimitedProvider::new(provider, config);

let response = limited.complete("test").await?;
assert_eq!(response.content, "Mock response");
```

### Reranker Testing

```rust
use edgequake_llm::reranker::{BM25Reranker, Reranker};

let reranker = BM25Reranker::new();
let results = reranker.rerank("search query", &documents, 5).await?;
assert!(results[0].score >= results[1].score); // Sorted by relevance
```

---

## Coverage Measurement

### Using cargo-tarpaulin

```bash
# Install
cargo install cargo-tarpaulin

# Run with HTML report
cargo tarpaulin --out Html --output-dir coverage/

# Run with specific packages
cargo tarpaulin --packages edgequake-llm --out Lcov
```

### Using cargo-llvm-cov

```bash
# Install
rustup component add llvm-tools-preview
cargo install cargo-llvm-cov

# Run with HTML report
cargo llvm-cov --html --output-dir coverage/

# Show summary
cargo llvm-cov --summary-only
```

---

## Test File Index

| File | Scope | API Keys Required |
|------|-------|-------------------|
| `src/cache.rs` (mod tests) | LRU cache, TTL, stats | No |
| `src/cost_tracker.rs` (mod tests) | Pricing, budgets, sessions | No |
| `src/rate_limiter.rs` (mod tests) | Token bucket, concurrency | No |
| `src/middleware.rs` (mod tests) | Middleware stack, hooks | No |
| `src/inference_metrics.rs` (mod tests) | TTFT, rate, formatting | No |
| `src/providers/tracing.rs` (mod tests) | Span delegation | No |
| `src/providers/genai_events.rs` (mod tests) | Event conversion | No |
| `src/providers/mock.rs` (mod tests) | Mock providers | No |
| `src/reranker/tests.rs` | BM25, RRF, hybrid, overlap | No |
| `tests/e2e_llm_providers.rs` | MockProvider, cache, rate limit | No |
| `tests/e2e_provider_factory.rs` | Env auto-detection | No |
| `tests/e2e_gemini.rs` | Gemini API | `GEMINI_API_KEY` |
| `tests/e2e_xai.rs` | xAI/Grok API | `XAI_API_KEY` |
| `tests/e2e_openai_compatible.rs` | OpenAI-compatible endpoints | `OPENAI_API_KEY` |

---

## See Also

- [Providers](providers.md) - provider-specific configuration
- [Caching](caching.md) - cache configuration for testing
- [Observability](observability.md) - TracingProvider test patterns
