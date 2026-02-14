# OODA Iteration 23 — Observe
## Focus: registry.rs test coverage gaps
### Current State
- `src/registry.rs`: 451 lines, 10 existing tests
- Tests cover: new(), get_mock, get_nonexistent, register_custom LLM/embedding, list, remove, count, overwrite
### Gaps Identified
1. Default impl (empty registry)
2. has_llm/has_embedding on empty registry
3. get on empty registry
4. remove_nonexistent (LLM and embedding)
5. remove_embedding (only remove_llm was tested)
6. clear_llm / clear_embedding
7. Multiple custom provider registration
8. clear_llm does not affect embedding (and vice versa)
9. get_returns_cloned_arc (Arc::ptr_eq)
10. llm_count after removal
