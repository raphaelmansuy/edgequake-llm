# OODA Iteration 02 - Orient

## Analysis

### What `docs/architecture.md` Should Contain

Based on code analysis, the architecture has 4 layers:

1. **Client Layer**: User code calls `ProviderFactory` or `ProviderRegistry`
2. **Middleware Layer**: Request/response pipeline (logging, metrics, cost)
3. **Provider Layer**: Trait implementations for each LLM backend
4. **Infrastructure Layer**: Cache, rate limiter, retry, tokenizer

### Key Design Decisions to Document (WHY)

1. **Why traits over enums?** Extensibility - external code can add providers
2. **Why factory + registry?** Factory for simple env-based startup, registry for dynamic multi-provider
3. **Why middleware pattern?** Cross-cutting concerns (logging, metrics, cost) without provider modification
4. **Why LRU cache?** Bounded memory with good temporal locality for repeated prompts
5. **Why token bucket rate limiting?** Matches API provider paradigm of requests/min + tokens/min

### Documentation Style

- Use ASCII diagrams (no emojis)
- Focus on WHY explanations
- Include code examples
- Cross-reference source files
