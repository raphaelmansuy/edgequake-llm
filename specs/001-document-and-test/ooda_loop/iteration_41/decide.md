# OODA Iteration 41 - Decide

## Decision: Add Factory Module Tests

### Priority Actions

1. **Test create_embedding_provider mock fallbacks** (5 tests)
   - Anthropic -> mock
   - OpenRouter -> mock
   - xAI -> mock
   - HuggingFace -> mock
   - Gemini -> mock

2. **Test create_embedding_provider provider creation** (3 tests)
   - Ollama -> OllamaProvider
   - LMStudio -> LMStudioProvider
   - VsCodeCopilot -> VsCodeCopilotProvider

3. **Test from_config branches** (3 tests)
   - OpenAI config type
   - Ollama config type
   - LMStudio config type

4. **Test create_with_model provider paths** (2 tests)
   - Ollama with model
   - LMStudio with model

### Implementation Plan

Add 13 new tests to `src/factory.rs` in the `#[cfg(test)] mod tests` section:

```rust
#[test]
fn test_create_embedding_provider_anthropic_fallback()
#[test]
fn test_create_embedding_provider_openrouter_fallback()
#[test]
fn test_create_embedding_provider_xai_fallback()
#[test]
fn test_create_embedding_provider_huggingface_fallback()
#[test]
fn test_create_embedding_provider_gemini_fallback()
#[test]
fn test_create_embedding_provider_ollama()
#[test]
fn test_create_embedding_provider_lmstudio()
#[test]
fn test_create_embedding_provider_vscode_copilot()
#[test]
fn test_from_config_openai()
#[test]
fn test_from_config_ollama()
#[test]
fn test_from_config_lmstudio()
#[test]
fn test_create_with_model_ollama()
#[test]
fn test_create_with_model_lmstudio()
```

### Success Criteria

- All 13 new tests pass
- factory.rs coverage increases from 23% to 35%+
- Overall coverage increases toward 53%+
