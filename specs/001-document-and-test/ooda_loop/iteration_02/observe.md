# OODA Iteration 02 - Observe

## Mission Re-Read
Re-read `specs/001-document-and-test.md`. Focus: Create `docs/architecture.md`.

## Observations

### Architecture Summary (verified from code)

1. **Trait-Based Abstraction**: `LLMProvider` and `EmbeddingProvider` traits in `src/traits.rs`
   - `LLMProvider`: complete(), chat(), chat_with_tools(), stream(), chat_with_tools_stream()
   - `EmbeddingProvider`: embed(), embed_batch()
   - Both: Send + Sync for async safety

2. **Provider Implementations** (11 total in `src/providers/`):
   - Cloud: OpenAI, Anthropic, Gemini, xAI, OpenRouter, HuggingFace
   - Local: Ollama, LMStudio
   - IDE: VSCode Copilot
   - Generic: OpenAI Compatible
   - Testing: Mock

3. **Factory Pattern** (`src/factory.rs`):
   - `ProviderFactory::from_env()` auto-detects provider from env vars
   - `ProviderFactory::create()` creates from explicit `ProviderType`
   - Priority: EDGEQUAKE_LLM_PROVIDER > Ollama env > OpenAI env > Mock

4. **Registry Pattern** (`src/registry.rs`):
   - Dynamic registration of providers
   - Runtime lookup by name
   - Extensible without factory modification

5. **Middleware Pipeline** (`src/middleware.rs`):
   - `LLMMiddleware` trait: before() + after() hooks
   - `LLMMiddlewareStack`: ordered middleware chain
   - Built-in: LoggingLLMMiddleware, MetricsLLMMiddleware

6. **Cross-Cutting Concerns**:
   - Cache: LRU response caching (`src/cache.rs`)
   - Cache Prompt: Anthropic-style prompt caching (`src/cache_prompt.rs`)
   - Cost Tracker: Session-level cost tracking (`src/cost_tracker.rs`)
   - Rate Limiter: Token bucket algorithm (`src/rate_limiter.rs`)
   - Retry: Exponential backoff (`src/retry.rs`)
   - Metrics: Streaming inference metrics (`src/inference_metrics.rs`)

7. **Reranker Module** (`src/reranker/`):
   - BM25 scoring
   - Reciprocal Rank Fusion (RRF)
   - Hybrid reranking
   - HTTP-based reranking
   - Term overlap scoring

8. **Error Handling** (`src/error.rs`):
   - Typed errors with retry strategy suggestions
   - Categories: Auth, RateLimit, TokenLimit, Network, Model, Timeout, etc.
