# OODA Loop Iteration 32 - Decide

**Date:** 2025-01-14
**Focus:** Ollama Provider Unit Tests

## Decision
Add 14 comprehensive unit tests to `src/providers/ollama.rs`:

### Builder Tests (4 tests)
1. `test_constants` - DEFAULT_OLLAMA_HOST, MODEL, EMBEDDING_MODEL
2. `test_builder_default_values` - Default trait implementation
3. `test_builder_custom_context_length` - max_context_length override
4. `test_builder_custom_embedding_dimension` - embedding_dimension override

### Provider Creation Tests (2 tests)
5. `test_default_local_creation` - OllamaProvider::default_local()
6. `test_from_env_uses_defaults` - from_env falls back to defaults

### Trait Implementation Tests (3 tests)
7. `test_supports_streaming` - Returns true
8. `test_supports_json_mode` - Returns false (not overridden)
9. `test_message_conversion_tool_role` - Tool → "user" conversion

### EmbeddingProvider Tests (3 tests)
10. `test_embedding_provider_name` - Returns "ollama"
11. `test_embedding_provider_dimension` - Custom dimension
12. `test_embedding_provider_max_tokens` - Returns 8192

### Serialization Tests (2 tests)
13. `test_ollama_message_serialization` - OllamaMessage JSON
14. `test_chat_options_temperature_serialization` - Full ChatOptions

## Rationale
- Local development depends on Ollama - tests must be comprehensive
- Tool role behavior is Ollama-specific and critical to understand
- JSON mode returns false - users need to know this limitation
