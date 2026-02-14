# Iteration 17 - Orient
TTL expiration and LRU eviction are the core cache invariants - untested = high risk. CachedProvider is the user-facing API that delegates to inner provider + cache. Adding 16 tests covers all untested paths.
