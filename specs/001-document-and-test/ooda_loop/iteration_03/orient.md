# OODA Iteration 03 - Orient

## Analysis

The providers doc must cover all 11 providers (9 production + Mock + OpenAICompatible).
Each provider section needs: setup, env vars, features, models, code example.

### Organization Strategy

Group by deployment model:
1. Cloud APIs (OpenAI, Anthropic, Gemini, xAI, OpenRouter, HuggingFace, Azure)
2. Local (Ollama, LMStudio)
3. IDE Integration (VSCode Copilot)
4. Generic (OpenAI Compatible)
5. Testing (Mock)

### Key Points to Document

- Each provider's unique capabilities (vision, thinking, model discovery)
- Wrapper providers (xAI, HuggingFace) that use OpenAICompatible internally
- Environment-based configuration pattern
- Builder pattern for local providers
