# OODA-53 Orient

## Analysis
Retry handling is critical for production LLM applications due to:
- Transient network errors
- Rate limiting from providers
- Server-side errors (503, 500)
- Token limit issues requiring context reduction

## API Design
- `RetryStrategy` - enum defining how to handle failures
- `RetryExecutor` - wraps async operations with retry logic
- `LlmError::retry_strategy()` - maps errors to appropriate strategies
- `LlmError::is_recoverable()` - quick check for transient errors

## Pattern
Error → Strategy → Executor → Result
