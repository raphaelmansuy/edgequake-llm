# OODA Loop Iteration 27 - Decide

**Date:** 2025-01-14
**Focus:** OpenAI Provider Unit Tests

## Decision
Add 17 comprehensive unit tests to `src/providers/openai.rs`:

### Context Length Tests (6 tests)
1. `test_context_length_gpt5_series` - GPT-5/5.1/5.2/5-nano/5-mini → 200k/128k
2. `test_context_length_o_series` - o1/o3/o4 reasoning models → 200k
3. `test_context_length_gpt4_variants` - gpt-4o/turbo/32k/base → various
4. `test_context_length_gpt35_variants` - gpt-3.5-turbo-16k/base → 16k/4k
5. `test_context_length_detection` - via with_model() updates max_context_length
6. `test_context_length_unknown_defaults_high` - unknown defaults to 128k

### Embedding Dimension Tests (3 tests)
7. `test_embedding_dimension_detection` - text-embedding-3-large → 3072
8. `test_dimension_ada_model` - text-embedding-ada → 1536
9. `test_dimension_unknown_defaults` - unknown → 1536

### Provider Trait Tests (5 tests)
10. `test_provider_name` - LLMProvider::name() returns "openai"
11. `test_provider_embedding_model` - EmbeddingProvider::model()
12. `test_provider_dimension` - EmbeddingProvider::dimension()
13. `test_provider_max_context_length` - with_model sets context length
14. `test_provider_builder` - builder pattern chaining

### Feature Support Tests (3 tests)
15. `test_supports_streaming` - returns true
16. `test_supports_json_mode_gpt4` - gpt-4o supports JSON mode
17. `test_supports_json_mode_gpt35` - gpt-3.5-turbo supports JSON mode
18. `test_supports_json_mode_default_is_false` - default model doesn't

### Message Conversion Tests (2 tests)
19. `test_message_conversion` - user/assistant/system messages
20. `test_message_conversion_tool_role` - tool role fallback to user

## Rationale
- Covers all branches in context_length_for_model() match statement
- Tests 2025 models (GPT-5, O-series) which are critical for current users
- Proper trait disambiguation prevents future confusion
- supports_json_mode behavior tested for different model families
