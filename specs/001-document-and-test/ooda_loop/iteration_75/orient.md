## Orient: Test Coverage Strategy

### Key Insight
Mock provider tests are 100% covered. The coverage gap is in HTTP-dependent provider code.

### Testable Without HTTP
1. **tracing.rs (12%)**: TracingProvider wraps any provider - can test with MockProvider
2. **cache.rs edge cases**: Cache miss/hit paths
3. **error.rs paths**: Error construction and retry strategy selection

### Not Testable Without HTTP Mock Infrastructure
- Provider HTTP request/response handling
- Stream parsing
- Auth token refresh flows

### Recommendation
Focus on tracing.rs - can achieve significant coverage gains by testing TracingProvider with MockProvider.
