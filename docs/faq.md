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

### Authentication Errors

#### "Authentication error: invalid_api_key"

**Cause**: API key is missing, expired, or incorrect.

**Solutions**:
1. Verify the environment variable is set:
   ```bash
   echo $OPENAI_API_KEY  # Should not be empty
   ```
2. Check for leading/trailing whitespace in your key
3. Ensure the key hasn't been rotated or revoked in your provider's dashboard
4. For Azure OpenAI, verify both `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` are set

#### "AuthError: token expired"

**Cause**: Service account tokens (GCP, Azure AD) have expired.

**Solutions**:
1. For Gemini/VertexAI: Run `gcloud auth application-default login`
2. For Azure: Run `az login` to refresh credentials
3. Check token TTL and implement refresh logic in long-running applications

#### Provider detects wrong API key format

**Cause**: API keys have different formats per provider.

| Provider | Key Format |
|----------|------------|
| OpenAI | `sk-...` (51 chars) |
| Anthropic | `sk-ant-...` |
| Gemini | `AIza...` |
| xAI | `xai-...` |

**Solution**: Ensure you're using the correct key for the provider you're configuring.

### Rate Limiting Issues

#### "Rate limit exceeded" with immediate failures

**Cause**: Not using `RateLimitedProvider` wrapper.

**Solution**:
```rust
use edgequake_llm::{RateLimiterConfig, RateLimitedProvider};

let config = RateLimiterConfig::openai_gpt4(); // or custom limits
let limited = RateLimitedProvider::new(provider, config);
```

#### "429 Too Many Requests" despite rate limiting

**Cause**: Your API tier has lower limits than the default config.

**Solution**: Check your provider dashboard for actual limits and configure accordingly:
```rust
let config = RateLimiterConfig {
    requests_per_minute: 60,    // Adjust to your tier
    tokens_per_minute: 40_000,  // Adjust to your tier
    max_concurrent: 5,
    ..Default::default()
};
```

#### Requests queue indefinitely

**Cause**: Token bucket depleted faster than it refills.

**Solution**: 
1. Reduce `max_concurrent` to allow bucket refill
2. Increase `tokens_per_minute` if your tier allows
3. Use `RateLimiterConfig::for_provider("provider_name")` for presets

### Token Limit Errors

#### "Token limit exceeded: max X, got Y"

**Cause**: Input prompt + expected output exceeds model's context window.

**Solutions**:
1. Reduce input size using the tokenizer:
   ```rust
   let tokenizer = Tokenizer::for_model("gpt-4o");
   let truncated = tokenizer.truncate(&text, 8000);
   ```
2. Chunk large documents:
   ```rust
   let chunks = tokenizer.chunk(&text, 4000, 200); // with overlap
   ```
3. Use a model with larger context (e.g., `gpt-4o` has 128K tokens)

#### How to estimate token count before sending?

```rust
let tokenizer = Tokenizer::for_model("gpt-4o");
let count = tokenizer.count_tokens(&prompt);
println!("Estimated tokens: {}", count);
```

**Note**: Chat messages have additional overhead (~3 tokens per message).

### Network Errors

#### "Network error: connection refused"

**Cause**: Provider endpoint unreachable.

**Solutions**:
1. Check your internet connection
2. Verify firewall/proxy settings
3. For local providers (Ollama, LMStudio):
   ```bash
   # Verify Ollama is running
   curl http://localhost:11434/api/tags
   
   # Verify LMStudio is running
   curl http://localhost:1234/v1/models
   ```

#### "Request timed out"

**Cause**: Long-running requests or slow network.

**Solutions**:
1. Increase timeout in completion options:
   ```rust
   let options = CompletionOptions {
       timeout_ms: Some(120_000), // 2 minutes
       ..Default::default()
   };
   ```
2. Use streaming for long responses to keep connection alive
3. Retry with backoff:
   ```rust
   let executor = RetryExecutor::new();
   let result = executor.execute(
       &RetryStrategy::network_backoff(),
       || async { provider.chat(&messages, None).await }
   ).await;
   ```

#### DNS resolution failures

**Cause**: DNS server issues or network configuration.

**Solution**: Try using IP addresses directly or configure DNS:
```bash
# Test DNS resolution
nslookup api.openai.com

# Use alternative DNS
export RES_NAMESERVERS="8.8.8.8"
```

### Provider-Specific Issues

#### Ollama: "model not found"

**Cause**: Model not pulled locally.

**Solution**:
```bash
# List available models
ollama list

# Pull a model
ollama pull llama3.2:latest
```

#### LMStudio: No response / empty content

**Cause**: No model loaded in LMStudio.

**Solution**:
1. Open LMStudio app
2. Go to "Local Server" tab
3. Select and load a model
4. Verify server is running (green status)

#### Gemini: "PERMISSION_DENIED"

**Cause**: API not enabled or quota exceeded.

**Solutions**:
1. Enable the Generative Language API:
   ```bash
   gcloud services enable generativelanguage.googleapis.com
   ```
2. Check quota in Google Cloud Console
3. Verify billing is enabled for your project

#### Azure OpenAI: "DeploymentNotFound"

**Cause**: Model deployment name mismatch.

**Solution**: Use your exact deployment name, not the model name:
```rust
// Wrong: model = "gpt-4o"
// Right: model = "my-gpt4o-deployment"
let provider = AzureOpenAIProvider::new(&endpoint, &api_key, "my-gpt4o-deployment");
```

#### xAI: "model does not exist"

**Cause**: Using incorrect model identifier.

**Solutions**:
- Use `grok-2` or `grok-2-vision-1212` (not `grok-beta`)
- Check [api.x.ai](https://api.x.ai) for current model names

### Build & Development Issues

#### Compilation is slow

Enable incremental compilation and use the latest Rust toolchain:

```bash
export CARGO_INCREMENTAL=1
rustup update stable
```

For development, use `cargo check` instead of full builds when only checking types.

#### `cargo doc` warnings about unresolved links

Run `cargo doc --no-deps` and fix any broken intra-doc links. Common cause: referencing types without the full module path.

#### "cannot find type X in this scope"

Import the required types explicitly:
```rust
use edgequake_llm::{
    LLMProvider, LLMResponse, ChatMessage, ChatRole,
    CompletionOptions, ToolDefinition, ToolCall,
};
```

---

## See Also

- [Architecture](architecture.md) - system design
- [Providers](providers.md) - all 11 providers
- [Migration Guide](migration-guide.md) - upgrading between versions
