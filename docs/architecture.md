# Architecture

EdgeQuake LLM is a unified Rust library that abstracts multiple LLM and embedding
providers behind a common trait interface. This document explains the system design,
key patterns, and the reasoning behind architectural decisions.

## System Overview

```text
+-----------------------------------------------------------------------+
|                         User Application                              |
|                                                                       |
|   let (llm, embed) = ProviderFactory::from_env()?;                   |
|   let response = llm.chat(&messages, options).await?;                 |
+-----------------------------------------------------------------------+
         |                           |
         v                           v
+------------------+     +----------------------+
| ProviderFactory  |     |  ProviderRegistry    |
| (env auto-detect)|     |  (dynamic lookup)    |
+------------------+     +----------------------+
         |                           |
         +----------+----------------+
                    |
                    v
+-----------------------------------------------------------------------+
|                      Middleware Pipeline                               |
|                                                                       |
|   Request --> [Logging] --> [Metrics] --> [Cost] --> Provider.chat()   |
|   Response <-- [Logging] <-- [Metrics] <-- [Cost] <-- Provider        |
+-----------------------------------------------------------------------+
                    |
                    v
+-----------------------------------------------------------------------+
|                    Provider Trait Layer                                |
|                                                                       |
|   trait LLMProvider          trait EmbeddingProvider                   |
|     fn complete()              fn embed()                             |
|     fn chat()                  fn embed_batch()                       |
|     fn chat_with_tools()                                              |
|     fn stream()                                                       |
+-----------------------------------------------------------------------+
         |              |              |              |
         v              v              v              v
+----------+   +-----------+   +-----------+   +----------+
|  OpenAI  |   | Anthropic |   |  Gemini   |   |  Ollama  |
|  xAI     |   | OpenRouter|   | HuggingFace|  | LMStudio |
|  Azure   |   |  VSCode   |   |   Mock    |   | OAI-Compat|
+----------+   +-----------+   +-----------+   +----------+
         |              |              |              |
         v              v              v              v
+-----------------------------------------------------------------------+
|                    Infrastructure Layer                                |
|                                                                       |
|   +--------+  +-----------+  +--------+  +-------+  +----------+     |
|   | Cache  |  |Rate Limiter| | Retry  |  | Cost  |  | Tokenizer|     |
|   | (LRU)  |  |(Tok Bucket)| |(Backoff)|  |Tracker|  | (tiktoken)|    |
|   +--------+  +-----------+  +--------+  +-------+  +----------+     |
|                                                                       |
|   +---------------+  +------------------+                             |
|   | Cache Prompt  |  | Inference Metrics|                             |
|   | (KV-cache)    |  | (Streaming Stats)|                             |
|   +---------------+  +------------------+                             |
+-----------------------------------------------------------------------+
```

## Core Traits

### LLMProvider (`src/traits.rs`)

The central abstraction. Every provider implements this trait to expose
a uniform API regardless of the underlying LLM service.

```rust
#[async_trait]
pub trait LLMProvider: Send + Sync {
    fn name(&self) -> &str;
    fn model(&self) -> &str;
    fn max_context_length(&self) -> usize;

    async fn complete(&self, prompt: &str) -> Result<LLMResponse>;
    async fn chat(&self, messages: &[ChatMessage], options: Option<&CompletionOptions>) -> Result<LLMResponse>;
    async fn chat_with_tools(&self, messages: &[ChatMessage], tools: &[ToolDefinition], ...) -> Result<LLMResponse>;
    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>>;

    fn supports_streaming(&self) -> bool;
    fn supports_function_calling(&self) -> bool;
    fn supports_json_mode(&self) -> bool;
}
```

**Why traits over enums?** Traits enable:
- External code to implement custom providers without forking
- Compile-time verification of provider contracts
- Dynamic dispatch via `Arc<dyn LLMProvider>` for runtime flexibility
- Clean testing with `MockProvider`

### EmbeddingProvider (`src/traits.rs`)

Separate trait for vector embedding generation. Not all LLM providers
support embeddings, so this is decoupled.

```rust
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    fn name(&self) -> &str;
    fn embedding_dimensions(&self) -> usize;
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}
```

## Provider Creation

### ProviderFactory (`src/factory.rs`)

Creates providers from environment variables. Auto-detection priority:

```text
  EDGEQUAKE_LLM_PROVIDER env var set?
  |
  +-- Yes --> Use specified provider
  |
  +-- No  --> Check for provider-specific env vars:
              1. OLLAMA_HOST or OLLAMA_MODEL --> Ollama
              2. OPENAI_API_KEY             --> OpenAI
              3. (none)                      --> Mock
```

**Why factory + auto-detection?** Reduces boilerplate for common setups.
A developer sets their API key and gets a working provider without
explicit configuration code.

### ProviderRegistry (`src/registry.rs`)

For applications needing multiple providers simultaneously:

```rust
let mut registry = ProviderRegistry::new();
registry.register_llm("openai", Arc::new(openai_provider));
registry.register_llm("ollama", Arc::new(ollama_provider));

// Runtime provider selection
let provider = registry.get_llm("openai").unwrap();
```

**Why registry pattern?** Supports dynamic provider selection at runtime
(e.g., route expensive requests to cheaper providers, failover scenarios).

## Middleware Pipeline (`src/middleware.rs`)

Request/response middleware for cross-cutting concerns:

```text
    Incoming Request
         |
         v
  +------+------+
  | before()    |  LoggingMiddleware: log request details
  +------+------+
         |
         v
  +------+------+
  | before()    |  MetricsMiddleware: start timer
  +------+------+
         |
         v
  +------+------+
  | provider    |  LLMProvider::chat()
  | .chat()     |
  +------+------+
         |
         v
  +------+------+
  | after()     |  MetricsMiddleware: record latency, tokens
  +------+------+
         |
         v
  +------+------+
  | after()     |  LoggingMiddleware: log response summary
  +------+------+
         |
         v
    Response
```

**Why middleware?** Separates observability, cost tracking, and auditing
from provider logic. Adding new cross-cutting concerns requires no
changes to provider implementations.

## Infrastructure Components

### Response Cache (`src/cache.rs`)

LRU cache with TTL expiration for LLM responses:

- **Key**: Hash of (messages + options + model)
- **Eviction**: LRU when capacity exceeded, TTL for staleness
- **Skip**: Non-deterministic requests (temperature > 0) can be excluded

**Why LRU?** LLM prompts exhibit temporal locality - recent queries are
most likely to repeat. LRU provides bounded memory with O(1) operations.

### Prompt Cache (`src/cache_prompt.rs`)

Provider-side KV-cache optimization (Anthropic-style). Inserts cache
breakpoints into message sequences so providers can skip re-computing
attention for stable prefixes.

**Why separate from response cache?** Response cache stores complete
answers; prompt cache optimizes the provider's internal computation.
They compose: prompt cache reduces cost per request, response cache
eliminates the request entirely.

### Rate Limiter (`src/rate_limiter.rs`)

Token bucket algorithm with:
- Requests-per-minute limiting
- Tokens-per-minute limiting
- Max concurrent request limiting (semaphore)

**Why token bucket?** Matches how LLM APIs enforce limits. Allows bursts
within the bucket capacity while enforcing sustained rates.

### Retry Executor (`src/retry.rs`)

Automatic retry with strategies mapped to error types:

| Error Type | Strategy | Behavior |
|------------|----------|----------|
| Network    | ExponentialBackoff | 125ms, 250ms, 500ms, ... up to 30s |
| RateLimit  | WaitAndRetry | Wait for `retry_after` header duration |
| TokenLimit | ReduceContext | Caller must reduce input size |
| Auth       | NoRetry | Permanent failure |

### Cost Tracker (`src/cost_tracker.rs`)

Session-level aggregation of API costs by model and provider.
Uses per-million-token pricing and supports cached token discounts.

### Inference Metrics (`src/inference_metrics.rs`)

Real-time streaming metrics: time-to-first-token, tokens-per-second,
total latency, token counts.

## Reranker Module (`src/reranker/`)

Standalone reranking pipeline for search result quality improvement:

```text
  Query + Candidate Documents
         |
         v
  +------+------+
  | BM25 Score  |  Term frequency / inverse document frequency
  +------+------+
         |
         v
  +------+------+
  | Term Overlap|  Simple keyword matching score
  +------+------+
         |
         v
  +------+------+
  | HTTP Rerank |  External reranker API (Cohere, Jina)
  +------+------+
         |
         v
  +------+------+
  | RRF Fusion  |  Reciprocal Rank Fusion combines scores
  +------+------+
         |
         v
  Reranked Results
```

## Error Handling (`src/error.rs`)

Errors carry retry strategies so callers know how to recover:

```rust
match error {
    LlmError::RateLimited { retry_after } => {
        // Wait and retry
        tokio::time::sleep(retry_after).await;
    }
    LlmError::TokenLimitExceeded { .. } => {
        // Reduce context and retry
    }
    LlmError::AuthError(_) => {
        // Permanent - check credentials
    }
}
```

## Data Flow: Complete Request

```text
  1. User calls provider.chat(messages, options)
  2. CachedProvider checks LRU cache
     |- Cache HIT  --> return cached response
     |- Cache MISS --> continue
  3. RateLimitedProvider acquires rate limit token
  4. RetryExecutor wraps the call with retry strategy
  5. Provider sends HTTP request to LLM API
  6. Provider parses response into LLMResponse
  7. CachedProvider stores response in cache
  8. Middleware after() hooks run (logging, metrics, cost)
  9. LLMResponse returned to caller
```

## File Reference

| File | Purpose |
|------|---------|
| `src/traits.rs` | LLMProvider, EmbeddingProvider traits |
| `src/error.rs` | Error types with retry strategies |
| `src/factory.rs` | Environment-based provider creation |
| `src/registry.rs` | Dynamic provider registry |
| `src/middleware.rs` | Request/response middleware pipeline |
| `src/cache.rs` | LRU response cache |
| `src/cache_prompt.rs` | Provider-side KV-cache optimization |
| `src/cost_tracker.rs` | Session cost tracking |
| `src/rate_limiter.rs` | Token bucket rate limiter |
| `src/retry.rs` | Retry executor with backoff |
| `src/tokenizer.rs` | Token counting (tiktoken) |
| `src/model_config.rs` | Model configuration and pricing |
| `src/inference_metrics.rs` | Streaming inference metrics |
| `src/providers/` | Provider implementations |
| `src/reranker/` | Search result reranking |

---

## See Also

- [Providers](providers.md) - All 11 providers configuration
- [Provider Families](provider-families.md) - Deep comparison of OpenAI vs Anthropic vs Gemini
- [Performance Tuning](performance-tuning.md) - Optimization strategies
- [Security](security.md) - API keys and privacy best practices
