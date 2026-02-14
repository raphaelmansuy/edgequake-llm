# OODA Iteration 42 - Observe

## Deep Coverage Analysis

Current coverage: **~51.44%** (2950/5735 lines)

### Modules with Extensive Tests Already

| Module | Coverage | Test Count | Notes |
|--------|----------|------------|-------|
| factory.rs | 23% → 30%+ | 36 | OODA-41 added 13 tests |
| model_config.rs | 75% | 25+ | Full config testing |
| error.rs | 65% | 20+ | Retry strategies covered |
| retry.rs | 72% | 15+ | Async retry logic |
| traits.rs | 67% | 50+ | Chat message helpers |
| ollama.rs | 46% | 25+ | Builder, message conversion |
| xai.rs | 51% | 20+ | Model catalog |
| huggingface.rs | 51% | 25+ | Model catalog |
| openrouter.rs | 33% | 20+ | SSE buffering tests |
| gemini.rs | 47% | 31 | OODA-34 improvements |
| anthropic.rs | 53% | 33 | OODA-35 improvements |
| openai_compatible.rs | 28% | 20 | OODA-36 improvements |
| lmstudio.rs | 15% | 30 | OODA-33 improvements |

### Modules Requiring HTTP Mocking for >97% Coverage

| Module | Coverage | Lines Uncovered | Blocker |
|--------|----------|-----------------|---------|
| tracing.rs | 12% | 240 | OpenTelemetry span emission |
| vscode/stream.rs | 0% | 94 | SSE streaming |
| vscode/client.rs | 33% | 173 | HTTP requests |
| vscode/auth.rs | 7% | 48 | OAuth flow |
| vscode/token.rs | 19% | 90 | Token management |
| azure_openai.rs | 28% | 122 | Azure API calls |
| reranker/http.rs | 40% | 73 | HTTP reranking |

### Key Insight

The remaining ~45% coverage gap is dominated by:
1. **HTTP I/O paths** (chat, complete, stream methods)
2. **OpenTelemetry instrumentation** (span creation, event emission)
3. **OAuth/Token flows** (VSCode authentication)

These require either:
- **HTTP mocking** (mockito, wiremock)
- **Test doubles** for reqwest::Client
- **Tracing subscriber** setup for span testing

### Low-Hanging Fruit Exhausted

Most pure logic (builders, conversions, configs, catalogs) is now tested.
The remaining uncovered code is primarily:
- Async HTTP request handlers
- Response parsing from live API calls
- Error branches that only trigger on network failures
