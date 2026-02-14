# Performance Tuning

Best practices for optimizing LLM performance, latency, and throughput with edgequake-llm.

---

## Latency Optimization

### Time to First Token (TTFT)

TTFT measures how quickly the LLM starts generating output. To minimize TTFT:

1. **Use streaming responses**
   ```rust
   let mut stream = provider.stream("Your prompt").await?;
   while let Some(chunk) = stream.next().await {
       print!("{}", chunk?); // First chunk arrives quickly
   }
   ```

2. **Keep prompts concise** - Longer prompts take longer to process
3. **Use faster models** - Flash/mini variants prioritize speed over quality
   - Gemini: `gemini-2.0-flash` (faster) vs `gemini-1.5-pro`
   - OpenAI: `gpt-4o-mini` (faster) vs `gpt-4o`
   - Anthropic: `claude-3-haiku` (faster) vs `claude-3-opus`

4. **Monitor TTFT with metrics**
   ```rust
   let mut metrics = InferenceMetrics::new();
   // ... after first token
   metrics.record_first_token();
   println!("TTFT: {:?}ms", metrics.ttft_ms());
   ```

### Request Latency

To reduce total request time:

| Strategy | Impact | Trade-off |
|----------|--------|-----------|
| Shorter max_tokens | Lower latency | May truncate output |
| Streaming | Perceived faster | Same total time |
| Caching | Near-instant hits | Memory usage |
| Local models | No network | Lower quality |

---

## Throughput Optimization

### Concurrent Requests

Maximize throughput with parallel requests:

```rust
use futures::future::join_all;

let prompts = vec!["Prompt 1", "Prompt 2", "Prompt 3"];
let futures: Vec<_> = prompts
    .iter()
    .map(|p| provider.complete(p, None))
    .collect();

let results = join_all(futures).await;
```

**Note**: Respect rate limits. Use `RateLimitedProvider` to manage concurrency:

```rust
let config = RateLimiterConfig {
    max_concurrent: 10,  // Max parallel requests
    requests_per_minute: 500,
    tokens_per_minute: 100_000,
    ..Default::default()
};
let limited = RateLimitedProvider::new(provider, config);
```

### Batch Processing

For embedding workloads, batch texts to reduce API overhead:

```rust
// ❌ Slow: One API call per text
for text in texts {
    let embedding = provider.embed(&[text]).await?;
}

// ✅ Fast: One API call for all texts
let embeddings = provider.embed(&texts).await?;
```

Most providers support up to 96-100 texts per batch.

---

## Memory Optimization

### Cache Configuration

Balance hit rate vs memory usage:

```rust
let cache = LLMCache::new(CacheConfig {
    max_entries: 1000,  // Adjust based on available memory
    ttl: Duration::from_secs(3600), // 1 hour
    cache_completions: true,
    cache_embeddings: true, // Embeddings use more memory
});
```

**Memory estimation**: ~2KB per cached completion, ~6KB per cached embedding (1536 dims).

### Token Counting

Use lazy tokenizer initialization to avoid memory overhead:

```rust
// Create tokenizer only when needed
fn count_tokens_if_needed(text: &str, model: &str) -> Option<usize> {
    if text.len() > 10_000 {
        let tokenizer = Tokenizer::for_model(model);
        Some(tokenizer.count_tokens(text))
    } else {
        None // Estimate: chars / 4
    }
}
```

---

## Cost Optimization

### Model Selection by Task

| Task | Recommended Model | Why |
|------|-------------------|-----|
| Simple extraction | `gpt-4o-mini` | 10x cheaper than full GPT-4o |
| Complex reasoning | `claude-3-opus` | Best quality |
| High volume | Ollama (local) | Free after setup |
| Classification | `gpt-3.5-turbo` | Fast and cheap |

### Caching Strategy

Response caching provides the biggest cost savings:

```rust
// Repeated identical queries are free
let cached = CachedProvider::new(provider, cache);

// First call: API request (~$0.01)
cached.complete("Summarize this document...").await?;

// Second identical call: Cache hit (~$0.00)
cached.complete("Summarize this document...").await?;
```

### Prompt Caching (Server-Side)

For Anthropic providers, use prompt caching to reduce input token costs by ~90%:

```rust
let config = CachePromptConfig::default();
let messages = apply_cache_control(&config, &messages);
// Anthropic caches the system prompt and reuses it
```

---

## Connection Pooling

### HTTP Client Reuse

The library reuses HTTP clients internally. For optimal performance:

```rust
// ✅ Good: Create provider once, reuse for all requests
let provider = OpenAIProvider::new(&api_key);
for prompt in prompts {
    provider.complete(prompt, None).await?;
}

// ❌ Bad: New provider per request (creates new HTTP client)
for prompt in prompts {
    let provider = OpenAIProvider::new(&api_key);
    provider.complete(prompt, None).await?;
}
```

### Keep-Alive

Connections are kept alive by default (HTTP/2 for most providers). For long-running applications, this eliminates connection setup overhead.

---

## Streaming Best Practices

### Display Responsiveness

For chat applications, flush output frequently:

```rust
use std::io::{stdout, Write};

let mut stream = provider.stream(prompt).await?;
while let Some(chunk) = stream.next().await {
    print!("{}", chunk?);
    stdout().flush()?; // Immediate display
}
```

### Token Rate Tracking

Monitor generation speed:

```rust
let mut metrics = InferenceMetrics::new();
while let Some(chunk) = stream.next().await {
    let text = chunk?;
    metrics.add_output_tokens(text.len() / 4); // Rough estimate
    println!("\r{:.1} t/s", metrics.tokens_per_second());
}
```

---

## Provider-Specific Tips

### OpenAI

- Use `gpt-4o-mini` for most tasks (best price/performance)
- Enable `stream: true` for faster perceived latency
- Batch embeddings up to 96 texts

### Ollama

- Pull models before first use: `ollama pull llama3.2`
- Use GPU acceleration: `OLLAMA_GPU_LAYERS=99`
- Keep Ollama running as a service for instant responses

### Gemini

- Use context caching for long documents (>32K tokens)
- Flash models are optimized for speed
- Batch up to 100 texts for embeddings

### Anthropic

- Use prompt caching (`cache_control`) for system prompts
- Claude Haiku: Fastest response time
- Claude Opus: Best for complex reasoning

---

## Benchmarking

### Measure Actual Performance

```rust
use std::time::Instant;

let start = Instant::now();
let response = provider.complete(prompt, None).await?;
let duration = start.elapsed();

println!("Total time: {:?}", duration);
println!("Tokens: {:?}", response.usage);
println!("Tokens/sec: {:.1}", 
    response.usage.completion_tokens as f64 / duration.as_secs_f64());
```

### Compare Providers

Run identical prompts across providers to find the best fit:

```rust
let providers = vec![
    ("openai", ProviderFactory::create(ProviderType::OpenAI)?),
    ("anthropic", ProviderFactory::create(ProviderType::Anthropic)?),
    ("ollama", ProviderFactory::create(ProviderType::Ollama)?),
];

for (name, (llm, _)) in providers {
    let start = Instant::now();
    let response = llm.complete(prompt, None).await?;
    println!("{}: {:?}", name, start.elapsed());
}
```

---

## See Also

- [Caching](caching.md) - Response and prompt caching details
- [Rate Limiting](rate-limiting.md) - Controlling request throughput
- [Cost Tracking](cost-tracking.md) - Monitoring API costs
- [Observability](observability.md) - Tracing and metrics
