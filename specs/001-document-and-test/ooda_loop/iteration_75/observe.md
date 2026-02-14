## Observe: Coverage Analysis for OODA-75

### Current Coverage: 49.96% (2865/5735 lines)

**Fully Covered Modules (100%):**
- mock.rs: 77/77 lines
- tokenizer.rs: 40/40 lines
- vscode/error.rs: 15/15 lines
- vscode/types.rs: 24/24 lines

**High Coverage (>70%):**
- traits.rs: 109/124 (88%)
- registry.rs: 46/50 (92%)
- bm25.rs: 123/134 (92%)
- reranker/hybrid.rs: 18/22 (82%)
- middleware.rs: 89/106 (84%)
- rate_limiter.rs: 104/152 (68%)

**Low Coverage (HTTP-dependent):**
- tracing.rs: 33/273 (12%) - decorates providers, needs mock-based tests
- lmstudio.rs: 89/569 (16%) - HTTP client code
- vscode/stream.rs: 0/94 (0%) - streaming code
- vscode/auth.rs: 4/52 (8%) - auth flow code
- azure_openai.rs: 39/200 (20%)
- huggingface.rs: 34/187 (18%)

### gap: HTTP-dependent code cannot be unit tested without mocking HTTP layer
