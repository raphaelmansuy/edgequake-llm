# FAQ

Frequently asked questions about edgequake-llm.

---

## General

### What is edgequake-llm?

A Rust library that provides a unified async API for multiple LLM providers (OpenAI, Anthropic, Gemini, xAI, Ollama, and more). It handles caching, rate limiting, cost tracking, and observability so you can swap providers without changing application code.

### Which Rust version do I need?

The minimum supported Rust version (MSRV) is **1.75.0**. The library requires async/await and the `async-trait` crate.

### Is this production-ready?

Version 0.2.0 is the first standalone release. The API may evolve before 1.0, but the core traits (`LLMProvider`, `EmbeddingProvider`) are stable. Pin your dependency version in `Cargo.toml`.

---

## Providers

### How do I add a new provider?

1. Implement `LLMProvider` for your struct (and `EmbeddingProvider` if applicable)
2. Add a variant to `ProviderType` in `src/factory.rs`
3. Register the provider in `ProviderFactory::create()`
4. Add environment variable detection in `ProviderFactory::from_env()`

See [providers.md](providers.md) for the full list and configuration.

### Which provider should I use?

| Use Case | Recommended Provider |
|----------|---------------------|
| Best quality | Anthropic (Claude), OpenAI (GPT-4/o-series) |
| Lowest cost | Ollama (local, free) or OpenRouter (compare 600+ models) |
| Fastest response | Gemini Flash, xAI Grok |
| Privacy-sensitive | Ollama or LMStudio (local, no data leaves your machine) |
| Multi-provider routing | OpenRouter |

### How does auto-detection work?

`ProviderFactory::from_env()` checks environment variables in priority order:

1. `EDGEQUAKE_LLM_PROVIDER` (explicit override)
2. `OLLAMA_HOST` (Ollama)
3. `LMSTUDIO_HOST` (LMStudio)
4. `OPENAI_API_KEY` (OpenAI)
5. `ANTHROPIC_API_KEY` (Anthropic)
6. `GEMINI_API_KEY` or `GOOGLE_API_KEY` (Gemini)
7. `XAI_API_KEY` (xAI)
8. `OPENROUTER_API_KEY` (OpenRouter)
9. `AZURE_OPENAI_API_KEY` (Azure OpenAI)
10. Mock fallback (no keys set)

### Can I use multiple providers simultaneously?

Yes. Create separate provider instances and use them independently:

```rust
let openai = ProviderFactory::create(ProviderType::OpenAI)?;
let ollama = ProviderFactory::create(ProviderType::Ollama)?;

// Use OpenAI for complex tasks
let analysis = openai.0.chat(&complex_messages, None).await?;

// Use Ollama for simple tasks (free, local)
let summary = ollama.0.chat(&simple_messages, None).await?;
```

---

## Caching

### How does response caching work?

`CachedProvider` wraps any provider with an LRU cache keyed by (provider, model, prompt/messages hash). Cache hits return instantly without making API calls.

```rust
let cache = LLMCache::new(CacheConfig { max_entries: 1000, ttl_seconds: 3600 });
let cached = CachedProvider::new(provider, cache);
```

See [caching.md](caching.md) for configuration details.

### What is prompt caching vs response caching?

- **Response caching** (`CachedProvider`): Stores complete LLM responses locally. Same input returns cached output immediately.
- **Prompt caching** (`CachePromptConfig`): Tells the provider API to cache the prompt prefix server-side for faster processing. Supported by Anthropic (cache_control breakpoints) and reduces cost by up to 90%.

---

## Cost Tracking

### How do I track costs?

Use `SessionCostTracker` to accumulate costs across multiple LLM calls:

```rust
let tracker = SessionCostTracker::new();
tracker.add_pricing("gpt-4", ModelPricing::new(0.03, 0.06)); // per 1K tokens
tracker.record(&response);
println!("Total: ${:.4}", tracker.total_cost());
```

### Can I set a budget limit?

Yes. Use `set_budget()` and check `is_over_budget()`:

```rust
tracker.set_budget(1.00); // $1.00 max
if tracker.is_over_budget() {
    eprintln!("Budget exceeded!");
}
```

See [cost-tracking.md](cost-tracking.md) for details.

---

## Rate Limiting

### How does rate limiting work?

`RateLimitedProvider` uses a three-layer token bucket algorithm:

1. **Semaphore**: Limits concurrent in-flight requests
2. **Request bucket**: Limits requests per minute
3. **Token bucket**: Limits tokens per minute

```rust
let config = RateLimiterConfig::for_provider("openai");
let limited = RateLimitedProvider::new(provider, config);
```

See [rate-limiting.md](rate-limiting.md) for provider presets.

### What happens when I hit a rate limit?

The library automatically waits and retries. Provider-returned `Retry-After` headers are respected. The `RetryExecutor` handles exponential backoff with configurable max retries.

---

## Streaming

### How do I stream responses?

Use `stream()` for simple text streaming or `chat_with_tools_stream()` for rich streaming with tool calls:

```rust
use futures::StreamExt;

let mut stream = provider.stream("Tell me a story").await?;
while let Some(chunk) = stream.next().await {
    print!("{}", chunk?);
}
```

For rich streaming:

```rust
let mut stream = provider.chat_with_tools_stream(&messages, &tools, None, None).await?;
while let Some(chunk) = stream.next().await {
    match chunk? {
        StreamChunk::Content(text) => print!("{text}"),
        StreamChunk::ThinkingContent { text, .. } => { /* reasoning */ },
        StreamChunk::ToolCallDelta { function_name, .. } => { /* tool call */ },
        StreamChunk::Finished { reason, ttft_ms } => break,
    }
}
```

### Which providers support streaming?

All providers support basic text streaming. Tool-call streaming (`supports_tool_streaming()`) is provider-dependent. Check with `provider.supports_tool_streaming()`.

---

## Observability

### How do I add tracing?

Wrap your provider with `TracingProvider`:

```rust
use edgequake_llm::providers::TracingProvider;

let traced = TracingProvider::new(provider);
// All calls emit OpenTelemetry GenAI spans
```

See [observability.md](observability.md) for Jaeger setup and attribute reference.

### Are prompts logged?

No. Content capture is opt-in for privacy. Set `EDGECODE_CAPTURE_CONTENT=true` to include prompts and responses in traces.

---

## Testing

### How do I test without API keys?

Use `MockProvider` or `MockAgentProvider`:

```rust
let provider = MockProvider::new();
provider.add_response("Expected answer").await;
let response = provider.complete("test").await?;
assert_eq!(response.content, "Expected answer");
```

See [testing.md](testing.md) for patterns.

### How do I measure test coverage?

```bash
cargo install cargo-tarpaulin
cargo tarpaulin --out Html --output-dir coverage/
```

---

## Troubleshooting

### "Authentication error" on startup

Check that the correct API key environment variable is set for your provider. Each provider expects a specific variable (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). See [providers.md](providers.md).

### "Rate limit exceeded" errors

Wrap your provider with `RateLimitedProvider` using appropriate limits for your API tier. The library will automatically queue and retry requests.

### Compilation is slow

Enable incremental compilation and use the latest Rust toolchain:

```bash
export CARGO_INCREMENTAL=1
rustup update stable
```

For development, use `cargo check` instead of full builds when only checking types.

### `cargo doc` warnings about unresolved links

Run `cargo doc --no-deps` and fix any broken intra-doc links. Common cause: referencing types without the full module path.

---

## See Also

- [Architecture](architecture.md) - system design
- [Providers](providers.md) - all 11 providers
- [Migration Guide](migration-guide.md) - upgrading between versions
