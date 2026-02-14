# OODA Loop Iteration 34 - Decide

**Date:** 2025-01-14
**Focus:** Gemini Provider Unit Tests

## Decision
Add 12 new tests to gemini.rs covering gaps identified.

## Tests to Implement
1. `test_constants` - Verify GEMINI_API_BASE, DEFAULT_GEMINI_MODEL, DEFAULT_EMBEDDING_MODEL
2. `test_google_ai_provider_name` - Google AI endpoint returns "gemini"
3. `test_supports_streaming` - Returns true
4. `test_supports_json_mode` - Check actual return value
5. `test_with_cache_ttl` - Builder method sets cache TTL
6. `test_embedding_provider_name` - EmbeddingProvider::name()
7. `test_embedding_provider_model` - EmbeddingProvider::model()
8. `test_embedding_provider_max_tokens` - EmbeddingProvider::max_tokens()
9. `test_embed_empty_input` - Error on empty vec
10. `test_generation_config_serialization` - GenerationConfig to JSON
11. `test_gemini_models_response_deserialization` - List models response parsing
12. `test_function_call_deserialization` - FunctionCall struct parsing

## Implementation Order
Simple tests first, then struct tests.

## Expected Outcome
- 29 total tests (17 existing + 12 new)
- All tests pass
