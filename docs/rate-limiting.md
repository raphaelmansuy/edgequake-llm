# Rate Limiting

EdgeQuake LLM provides async-aware rate limiting using a token bucket algorithm
to prevent API overload and respect provider limits.

## How It Works

```text
  Incoming Request
       |
       v
  +----+----+--------------------+
  |         Semaphore            |  Layer 1: Max concurrent requests
  |  (max_concurrent permits)   |  Blocks if all slots taken
  +---------+--------------------+
            |
            v
  +---------+--------------------+
  |    Request Token Bucket      |  Layer 2: Requests per minute
  |  refills at RPM/60 per sec  |  Waits if bucket empty
  +---------+--------------------+
            |
            v
  +---------+--------------------+
  |     Token Token Bucket       |  Layer 3: Tokens per minute
  |  refills at TPM/60 per sec  |  Waits if bucket empty
  +---------+--------------------+
            |
            v
  Provider API Call
```

**Why token bucket?** LLM APIs enforce rate limits as requests/min and
tokens/min. Token buckets allow short bursts (up to bucket capacity)
while enforcing sustained rates, matching real provider behavior.

## Configuration

```rust,ignore
use edgequake_llm::rate_limiter::RateLimiterConfig;

// Custom configuration
let config = RateLimiterConfig::new(60, 90_000)  // 60 RPM, 90K TPM
    .with_max_concurrent(10)
    .with_retry_delay(Duration::from_secs(1));
```

### Provider Presets

| Preset | RPM | TPM | Concurrent |
|--------|-----|-----|------------|
| `default()` | 60 | 90,000 | 10 |
| `openai_gpt4()` | 500 | 30,000 | 50 |
| `openai_gpt4o_mini()` | 5,000 | 200,000 | 100 |
| `openai_gpt35()` | 3,500 | 90,000 | 100 |
| `anthropic_claude()` | 60 | 100,000 | 10 |

```rust,ignore
// Use a preset
let config = RateLimiterConfig::openai_gpt4();
let limiter = RateLimiter::new(config);
```

## Using the Rate Limiter

### Direct Usage

```rust,ignore
use edgequake_llm::rate_limiter::RateLimiter;

let limiter = RateLimiter::new(RateLimiterConfig::openai_gpt4());

// Acquire rate limit permission (waits if necessary)
let estimated_tokens = 1000;
let guard = limiter.acquire(estimated_tokens).await;

// Make API call...
let response = provider.chat(&messages, None).await?;

// Record actual usage for accuracy
limiter.record_usage(response.total_tokens, estimated_tokens).await;

// Guard is dropped automatically, releasing concurrent slot
drop(guard);
```

### Try Without Waiting

```rust,ignore
// Non-blocking check
if let Some(guard) = limiter.try_acquire(1000).await {
    // Make request
} else {
    // Rate limited - handle gracefully
    println!("Rate limited, try again later");
}
```

### Wrapping a Provider

```rust,ignore
use edgequake_llm::rate_limiter::{RateLimitedProvider, RateLimiter, RateLimiterConfig};
use std::sync::Arc;

let provider = OpenAIProvider::new("sk-...");
let limiter = Arc::new(RateLimiter::new(RateLimiterConfig::openai_gpt4()));

// RateLimitedProvider implements LLMProvider
let limited = RateLimitedProvider::new(Arc::new(provider), limiter);

// All calls are now rate-limited automatically
let response = limited.chat(&messages, None).await?;
```

## Monitoring

```rust,ignore
// Check available capacity
let avail_requests = limiter.available_requests().await;
let avail_tokens = limiter.available_tokens().await;

println!("Available requests: {:.0}", avail_requests);
println!("Available tokens: {:.0}", avail_tokens);
```

## Token Bucket Algorithm

```text
  Bucket capacity = RPM (or TPM)
  Refill rate = capacity / 60 tokens per second

  Time 0:    [====================]  bucket full (60 tokens)
  Request 1: [=================== ]  59 tokens remain
  Request 2: [==================  ]  58 tokens remain
  ...
  Wait 1s:   [==================..]  59 tokens (1 refilled)
  ...
  Empty:     [                    ]  0 tokens - must wait
  Wait 1s:   [.                   ]  1 token refilled
```

The algorithm refills tokens continuously based on elapsed time,
enabling smooth rate limiting without sharp per-minute boundaries.

## Source Files

| File | Purpose |
|------|---------|
| `src/rate_limiter.rs` | Token bucket rate limiter implementation |
