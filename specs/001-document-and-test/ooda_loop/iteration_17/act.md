# Iteration 17 - Act
Added 16 new tests to cache.rs: TTL expiration (completion + embedding), LRU eviction at capacity (completion + embedding), CachedProvider delegation (name/model/complete/embed/clear/with_default_cache), access_count increments, empty prompt/texts keys, overwrite same key, clear eviction count, config with_embedding_caching.

## Evidence
27 tests passed, 0 failed
