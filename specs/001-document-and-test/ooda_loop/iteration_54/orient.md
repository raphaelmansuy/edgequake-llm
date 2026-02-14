# OODA-54 Orient

## Analysis
Middleware system enables:
- Logging all LLM requests/responses
- Metrics collection (tokens, latency)
- Custom validation/audit logic
- Clean separation of cross-cutting concerns

## API Learnings
- `LLMMiddlewareStack::new()` - create empty stack
- `stack.add(Arc<dyn LLMMiddleware>)` - add middleware
- `stack.before(&request)` - run before hooks
- `stack.after(&request, &response, duration_ms)` - run after hooks
- Custom middleware implements `LLMMiddleware` trait

## Pattern
Request → [before hooks] → LLM call → [after hooks (reverse)] → Response
