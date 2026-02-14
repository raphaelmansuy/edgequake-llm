# OODA Loop Iteration 27 - Orient

**Date:** 2025-01-14
**Focus:** OpenAI Provider Unit Tests

## Analysis
OpenAI provider is one of the most important providers in the crate, being the industry standard API that many other providers emulate. Testing various model-specific logic paths is critical.

## Technical Considerations
1. **Trait disambiguation**: OpenAIProvider implements both LLMProvider and EmbeddingProvider with overlapping method names. Must use `LLMProvider::name(&provider)` syntax.
2. **Model variants**: GPT-5 series, O-series (o1/o3/o4), GPT-4 variants, GPT-3.5 variants all have different context lengths
3. **JSON mode support**: Only gpt-4 and gpt-3.5-turbo models support JSON mode with current implementation
4. **Default model**: gpt-5-mini is the default (200k context)

## Priority
- HIGH: Test context_length_for_model with new 2025 models (GPT-5, O-series)
- HIGH: Test provider trait methods with proper disambiguation
- MEDIUM: Test embedding dimension detection
- MEDIUM: Test message conversion including tool role
- MEDIUM: Test supports_json_mode with different models

## Lessons from Previous Iterations
- Always use `LLMProvider::name(&provider)` not `provider.name()` when both traits are implemented
- Test actual behavior not assumptions (JSON mode is NOT supported on default model)
