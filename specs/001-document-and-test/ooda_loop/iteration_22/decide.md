# OODA Iteration 22 — Decide

## Plan: Add 16 tests to rate_limiter.rs

1. Config Debug/Clone traits
2. Config builder chaining
3. TokenBucket refill capped at max
4. TokenBucket exact boundary acquire
5. try_acquire fails on request rate exhaustion
6. try_acquire fails on token rate exhaustion
7. record_usage exact match (actual == estimated)
8. RateLimitedProvider::new() and accessors
9. RateLimitedProvider::with_limiter() shared limiter
10. RateLimitedProvider LLMProvider delegation
11. RateLimitedProvider EmbeddingProvider delegation
12. RateLimitedProvider complete()
13. RateLimitedProvider embed()
14. RateLimitedProvider chat()
15. Concurrent acquire and release
