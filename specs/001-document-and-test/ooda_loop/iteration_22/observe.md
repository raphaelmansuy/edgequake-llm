# OODA Iteration 22 — Observe

## Focus: rate_limiter.rs test coverage gaps

### Current State
- `src/rate_limiter.rs`: 694 lines, 19 existing tests
- Tests cover: TokenBucket basic ops, RateLimiter creation/acquire/try_acquire, config presets, available capacity, record_usage

### Gaps Identified
1. **Config Debug/Clone traits** — never tested
2. **Builder chaining** — with_max_concurrent + with_retry_delay together
3. **TokenBucket refill cap** — tokens should never exceed max_tokens
4. **TokenBucket exact boundary** — acquiring exactly all remaining tokens
5. **try_acquire failure paths** — request rate exhaustion and token rate exhaustion (separate from concurrency)
6. **record_usage exact match** — actual == estimated
7. **RateLimitedProvider** — never tested: new(), with_limiter(), inner(), limiter()
8. **RateLimitedProvider LLMProvider delegation** — name, model, max_context_length, supports_streaming, supports_json_mode
9. **RateLimitedProvider EmbeddingProvider delegation** — name, model, dimension, max_tokens
10. **RateLimitedProvider complete/chat/embed** — end-to-end through rate limiter
11. **Concurrent acquire/release** — multiple slots, release, re-acquire
