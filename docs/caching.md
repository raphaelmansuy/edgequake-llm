# Caching

EdgeQuake LLM provides two complementary caching layers to reduce API costs
and improve response latency.

## Caching Layers

```text
  Request
    |
    v
  +---------------------------+
  |   Response Cache (LRU)    |   Layer 1: Eliminates API calls
  |   Same prompt? Return     |   for repeated queries
  |   cached response.        |
  +---------------------------+
    |
    | Cache miss
    v
  +---------------------------+
  |   Prompt Cache (KV)       |   Layer 2: Reduces cost per call
  |   Mark stable prefixes    |   via provider-side token caching
  |   for provider caching.   |   (90% discount on cached tokens)
  +---------------------------+
    |
    v
  Provider API Call
```

**Why two layers?** They serve different optimization goals:

- **Response cache** avoids the network round-trip entirely when the exact
  same prompt has been seen before. Best for deterministic queries
  (temperature = 0) with repeated inputs.

- **Prompt cache** reduces the per-token cost when the prompt shares a
  prefix with previous calls. Best for conversational contexts where the
  system prompt and early messages are stable across turns.

---

## Response Cache (`src/cache.rs`)

In-memory LRU cache that stores complete `LLMResponse` objects.

### Configuration

```rust,ignore
use edgequake_llm::cache::{CacheConfig, CachedProvider, LLMCache};
use std::time::Duration;

// Basic configuration
let config = CacheConfig::new(1000)           // Max 1000 entries
    .with_ttl(Duration::from_secs(3600))      // 1 hour TTL
    .with_completion_caching(true)             // Cache chat completions
    .with_embedding_caching(true);             // Cache embeddings

let cache = LLMCache::new(config);
```

### Wrapping a Provider

```rust,ignore
use edgequake_llm::cache::CachedProvider;
use std::sync::Arc;

let provider = OpenAIProvider::new("sk-...");
let cache = Arc::new(LLMCache::new(CacheConfig::default()));

// CachedProvider implements LLMProvider + EmbeddingProvider
let cached = CachedProvider::new(Arc::new(provider), cache.clone());

// First call: cache miss, calls API
let response1 = cached.complete("Hello").await?;

// Second call: cache hit, returns instantly
let response2 = cached.complete("Hello").await?;
```

### Cache Statistics

```rust,ignore
let stats = cache.stats().await;
println!("Entries: {}", stats.entries);
println!("Hits: {}", stats.hits);
println!("Misses: {}", stats.misses);
println!("Hit rate: {:.1}%", stats.hit_rate() * 100.0);
println!("Evictions: {}", stats.evictions);
```

### How Keys Are Generated

Cache keys are computed by hashing the prompt text:

```text
  "What is Rust?"  -->  hash()  -->  CacheKey { hash: 0x7a3f... }
```

For chat messages, the full message array is serialized and hashed.

### LRU Eviction

When the cache reaches `max_entries`, the least recently used entry
is evicted. The algorithm:

1. Find the entry with the lowest `access_count`
2. Among ties, prefer the oldest entry (`created_at`)
3. Remove that entry, increment `evictions` counter

### TTL Expiration

Entries older than `ttl` are evicted on the next access attempt.
A `get_completion` for an expired entry returns `None` and removes
the stale entry.

### Configuration Guide

| Scenario | max_entries | ttl | completions | embeddings |
|----------|-------------|-----|-------------|------------|
| Development | 100 | 5 min | Yes | Yes |
| Batch processing | 10000 | 24 hours | Yes | Yes |
| Real-time chat | 500 | 30 min | Yes | No |
| Embeddings only | 5000 | 1 hour | No | Yes |

---

## Prompt Cache (`src/cache_prompt.rs`)

Provider-side KV-cache optimization for Anthropic Claude models. Marks
message sections as cacheable so the provider can skip recomputing
attention for stable prefixes.

### How It Works

```text
  Turn 1:
  +------------------+---------------------------+--------+
  | System Prompt    | User: "Read this file..." | New    |
  | (cacheable)      | (cacheable, large)        | query  |
  +------------------+---------------------------+--------+
       ^                    ^
       |                    |
    cache_control:       cache_control:
    ephemeral            ephemeral

  Turn 2:
  +------------------+---------------------------+--------+
  | System Prompt    | User: "Read this file..." | New    |
  | (CACHE HIT)     | (CACHE HIT)               | query  |
  +------------------+---------------------------+--------+
       90% discount       90% discount
```

### Configuration

```rust,ignore
use edgequake_llm::cache_prompt::CachePromptConfig;

// Default: cache system + large messages + last 3 user messages
let config = CachePromptConfig::default();

// Only cache system prompts
let config = CachePromptConfig::system_only();

// Maximum caching (cache more content)
let config = CachePromptConfig::aggressive();

// Disable caching
let config = CachePromptConfig::disabled();
```

### Configuration Fields

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Enable/disable cache marking |
| `min_content_length` | `1000` | Min chars to auto-cache user messages |
| `cache_system_prompt` | `true` | Always cache system prompt |
| `cache_last_n_messages` | `3` | Number of recent messages to cache |

### Applying Cache Control

```rust,ignore
use edgequake_llm::cache_prompt::{apply_cache_control, CachePromptConfig};
use edgequake_llm::traits::ChatMessage;

let config = CachePromptConfig::default();
let mut messages = vec![
    ChatMessage::system("You are a code reviewer. Be thorough."),
    ChatMessage::user("Review this 500-line file:\n..."),
    ChatMessage::user("Now check the tests"),
];

apply_cache_control(&mut messages, &config);
// Messages now have cache_control set where appropriate
```

### Cache Statistics

```rust,ignore
use edgequake_llm::cache_prompt::CacheStats;

let stats = CacheStats::new(10000, 1000, 8000, 0);

println!("Cache hit rate: {:.0}%", stats.cache_hit_rate() * 100.0);
println!("Savings: ${:.4}", stats.savings());
println!("Cost per call: ${:.4}", stats.cost_per_call());
println!("Effective: {}", stats.is_effective());
```

### Presets

| Preset | System | Min Length | Last N | Use Case |
|--------|--------|-----------|--------|----------|
| `default()` | Yes | 1000 | 3 | General purpose |
| `system_only()` | Yes | MAX | 0 | Minimal caching |
| `aggressive()` | Yes | 100 | 10 | Maximum savings |
| `disabled()` | No | - | - | No caching |

---

## Composing Both Layers

For maximum cost reduction, use both layers together:

```rust,ignore
use edgequake_llm::cache::{CacheConfig, CachedProvider, LLMCache};
use edgequake_llm::cache_prompt::{apply_cache_control, CachePromptConfig};
use std::sync::Arc;

// Layer 1: Response cache
let cache = Arc::new(LLMCache::new(CacheConfig::new(1000)));
let cached_provider = CachedProvider::new(Arc::new(provider), cache);

// Layer 2: Prompt cache (applied to messages before sending)
let prompt_config = CachePromptConfig::default();
apply_cache_control(&mut messages, &prompt_config);

// Now the request benefits from both layers:
// - If exact match exists -> response cache hit (free)
// - If no match but shared prefix -> prompt cache discount (90% off)
let response = cached_provider.chat(&messages, None).await?;
```

## Performance Impact

| Metric | Without Cache | With Response Cache | With Both |
|--------|--------------|--------------------|-----------| 
| Latency (repeated) | 500-2000ms | <1ms | <1ms |
| Cost (repeated) | Full price | Free | Free |
| Cost (new, shared prefix) | Full price | Full price | ~10% |
| Memory | None | ~1MB per 1000 entries | Same |

## Source Files

| File | Purpose |
|------|---------|
| `src/cache.rs` | LRU response cache implementation |
| `src/cache_prompt.rs` | Prompt-level KV-cache optimization |
