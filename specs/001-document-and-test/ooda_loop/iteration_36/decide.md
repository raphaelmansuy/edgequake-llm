# OODA Loop Iteration 36 - Decide

**Date:** 2025-01-14
**Focus:** OpenAI Compatible Provider Unit Tests

## Decision
Add 8 new tests to openai_compatible.rs covering identified gaps.

## Tests to Implement
1. `test_supports_streaming` - From model capabilities
2. `test_supports_function_calling` - Already working, verify
3. `test_thinking_config_serialization` - Z.ai thinking mode
4. `test_response_format_serialization` - JSON mode format
5. `test_embedding_provider_name` - Returns config name
6. `test_embedding_provider_dimension` - From model card
7. `test_tool_call_request_serialization` - Tool call format
8. `test_function_call_request_serialization` - Function call format

## Implementation Order
Simple tests first, serialization tests after.

## Expected Outcome
- 21 total tests (13 existing + 8 new)
- All tests pass
