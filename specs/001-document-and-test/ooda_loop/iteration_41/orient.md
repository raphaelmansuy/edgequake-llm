# OODA Iteration 41 - Orient

## First Principles Analysis

### Coverage Gap Root Cause

The factory module has 77% uncovered code (283/372 lines) because:

1. **API Key Dependencies**: Many `create_llm_provider()` branches require real API keys
2. **Network I/O**: Provider creation often attempts HTTP connections
3. **Fallback Paths**: Mock fallback paths for providers without embeddings are untested

### Risk Assessment

| Approach | Benefit | Risk |
|----------|---------|------|
| Test mock fallbacks | +5-10% coverage | Low - no API keys needed |
| Test config parsing | +3-5% coverage | Low - pure logic |
| Test auto-detection | +5% coverage | Medium - env var conflicts |
| Test with real APIs | +20% coverage | High - flaky, expensive |

### Strategic Insight

The `create_embedding_provider()` function has 10 match arms, but only 2 are tested (mock, unknown). The other 8 can be tested:
- 5 fall back to mock (Anthropic, OpenRouter, xAI, HuggingFace, Gemini)
- 3 create providers (Ollama, LMStudio, VsCodeCopilot)

This is **low-hanging fruit** that adds significant coverage without HTTP mocking.

### Recommended Actions

1. **High Signal / Low Risk**: Test all `create_embedding_provider` mock fallback paths
2. **Medium Signal**: Test `from_config` with OpenAI/Ollama/LMStudio config types
3. **Low Signal**: Add edge case tests for empty API keys

### Expected Outcome

Adding ~12 new tests targeting factory.rs should increase coverage by 5-8%, moving from 51.44% toward 55%+.
