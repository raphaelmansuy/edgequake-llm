# OODA Loop Iteration 28 - Decide

**Date:** 2025-01-14
**Focus:** Azure OpenAI Provider Unit Tests

## Decision
Add 12 comprehensive unit tests to `src/providers/azure_openai.rs`:

### Message Conversion (1 test)
1. `test_message_conversion_tool_role` - Tool role message conversion

### Provider Defaults & Configuration (4 tests)
2. `test_provider_defaults` - Verify default API version, context length, dimension
3. `test_supports_streaming` - Returns true
4. `test_supports_json_mode` - Returns true
5. `test_max_context_length` - with_max_context_length works correctly

### EmbeddingProvider Trait (2 tests)
6. `test_embedding_provider_name` - Name returns "azure-openai"
7. `test_embedding_provider_dimension` - Dimension accessor works

### URL Building (3 tests)
8. `test_endpoint_trailing_slash_handling` - Strips trailing slashes
9. `test_build_url_embeddings` - URL for embedding deployments
10. `test_build_url_custom_api_version` - Custom API version in URL

### Error Handling (1 test)
11. `test_from_env_missing_endpoint` - Returns error on missing env var

### Serialization (1 test)
12. `test_azure_message_serialization` - AzureMessage struct serializes correctly

## Rationale
- Enterprise deployments need confidence in configuration handling
- Azure-specific URL patterns must be tested
- from_env error handling is critical for deployment scripts
- Serialization tests catch breaking changes in API wire format
