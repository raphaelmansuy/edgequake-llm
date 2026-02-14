# OODA Loop Iteration 35 - Decide

**Date:** 2025-01-14
**Focus:** Anthropic Provider Unit Tests

## Decision
Add 10 new tests to anthropic.rs covering identified gaps.

## Tests to Implement
1. `test_constants` - Verify ANTHROPIC_API_BASE, ANTHROPIC_API_VERSION, DEFAULT_MODEL
2. `test_supports_streaming` - Returns true
3. `test_supports_json_mode` - Returns false
4. `test_supports_function_calling` - Returns true for Claude 3+
5. `test_anthropic_usage_with_cache_tokens` - Cache fields deserialize
6. `test_anthropic_error_response_deserialization` - Error response parsing
7. `test_stream_event_message_start` - StreamEvent enum deserialization
8. `test_stream_event_ping` - Ping event deserialization
9. `test_image_source_serialization` - ImageSource JSON format
10. `test_content_block_tool_use_serialization` - Tool use content block

## Implementation Order
Simple tests first, API type tests last.

## Expected Outcome
- 34 total tests (24 existing + 10 new)
- All tests pass
