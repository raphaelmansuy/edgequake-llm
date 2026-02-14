# OODA Iteration 04 - Observe

## Mission Re-Read
Re-read `specs/001-document-and-test.md`. Focus: Create `docs/caching.md`.

## Caching Architecture (verified from source)

### Two Caching Layers

1. **Response Cache** (`src/cache.rs`, 571 lines)
   - In-memory LRU cache for LLM responses
   - Hash-based key from prompt/messages
   - TTL expiration + LRU eviction
   - CachedProvider wraps any LLMProvider/EmbeddingProvider
   - CacheConfig: max_entries, ttl, cache_completions, cache_embeddings
   - CacheStats: hits, misses, entries, evictions, hit_rate()

2. **Prompt Cache** (`src/cache_prompt.rs`, 648 lines)
   - Anthropic-style KV-cache optimization
   - Inserts cache_control breakpoints into messages
   - CachePromptConfig: enabled, min_content_length, cache_system_prompt, cache_last_n_messages
   - CacheStats: input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens
   - 90% cost reduction on cached tokens
   - Three presets: default, system_only, aggressive

### Key Insight

Response cache eliminates API calls entirely (same prompt = same response).
Prompt cache reduces cost per API call (shared prefix = discounted tokens).
They are complementary and compose well.
