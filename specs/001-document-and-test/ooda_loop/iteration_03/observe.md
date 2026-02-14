# OODA Iteration 03 - Observe

## Mission Re-Read
Re-read `specs/001-document-and-test.md`. Focus: Create `docs/providers.md`.

## Provider Inventory (verified from source)

| Provider | Source File | Lines | Auth | Chat | Embed | Stream | Tools |
|----------|------------|-------|------|------|-------|--------|-------|
| OpenAI | `openai.rs` | 539 | API Key | Yes | Yes | Yes | Yes |
| Anthropic | `anthropic.rs` | 1424 | API Key | Yes | No | Yes | Yes |
| Gemini | `gemini.rs` | 2040 | API Key/OAuth | Yes | Yes | Yes | Yes |
| xAI | `xai.rs` | 433 | API Key | Yes | No | Yes | Yes |
| OpenRouter | `openrouter.rs` | 1760 | API Key | Yes | Yes | Yes | Yes |
| Ollama | `ollama.rs` | 1153 | None | Yes | Yes | Yes | Yes |
| LMStudio | `lmstudio.rs` | 1939 | None | Yes | Yes | Yes | Yes |
| HuggingFace | `huggingface.rs` | 546 | Token | Yes | No | Yes | No |
| Azure OpenAI | `azure_openai.rs` | 932 | API Key | Yes | Yes | Yes | Yes |
| VSCode Copilot | `vscode/` | ~4000 | OAuth | Yes | Yes | Yes | Yes |
| OpenAI Compatible | `openai_compatible.rs` | 1499 | API Key | Yes | Yes | Yes | Yes |
| Mock | `mock.rs` | 564 | None | Yes | Yes | No | Yes |

## Environment Variables (verified)

- OpenAI: OPENAI_API_KEY, OPENAI_BASE_URL
- Anthropic: ANTHROPIC_API_KEY
- Gemini: GEMINI_API_KEY, GOOGLE_APPLICATION_CREDENTIALS
- xAI: XAI_API_KEY, XAI_MODEL, XAI_BASE_URL
- OpenRouter: OPENROUTER_API_KEY
- Ollama: OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL
- LMStudio: LMSTUDIO_HOST, LMSTUDIO_MODEL, LMSTUDIO_EMBEDDING_MODEL
- HuggingFace: HF_TOKEN, HF_MODEL
- Azure: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME
