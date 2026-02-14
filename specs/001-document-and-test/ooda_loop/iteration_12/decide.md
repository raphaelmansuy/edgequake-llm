# Iteration 12 - Decide

## Decision
Add comprehensive unit tests to `src/error.rs` to cover all untested branches:

1. `user_description()` - test all 13 error variants
2. `retry_strategy()` - test ApiError with 500/502/503, ProviderError, Unknown, SerializationError
3. `RetryStrategy::server_backoff()` - verify values
4. `is_recoverable()` - cover all variants
5. `retry_strategy()` ReduceContext should_retry check

## Priority: HIGH
- error.rs is the foundation module; comprehensive testing prevents regression
- Pure unit tests, fast to run, high signal value
