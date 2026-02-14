# Iteration 09 - Orient

## Analysis
The testing guide needs to document the two-layer mock system, common testing patterns, and how to run different test suites. Users need to know:
1. Which tests require API keys vs which are fully offline
2. How to use MockProvider and MockAgentProvider for their own tests
3. Coverage measurement tools
4. The testing patterns used throughout the codebase (decorator wrapping, cache testing, rate limiter testing)

## Key Pattern
All decorator types (TracingProvider, CachedProvider, RateLimitedProvider) are tested by wrapping MockProvider, which isolates the decorator logic from network concerns.
