# Iteration 17 - Observe
## cache.rs (571 lines, 11 existing tests)
Existing tests cover: CacheKey creation, CacheConfig builder, basic stats/miss/hit/clear, hit_rate, embedding cache, disabled caching.
Missing: TTL expiration (completion + embedding), LRU eviction at capacity, CachedProvider delegation (name/model/complete/embed/clear), access_count tracking, edge cases (empty prompts, overwrite, clear eviction count).
