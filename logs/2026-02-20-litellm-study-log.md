# Task Log — 2026-02-20 — LiteLLM Study

## Actions
- Fetched LiteLLM GitHub repo and `__init__.py` to extract full API surface (100+ providers, ModelResponse, streaming protocol, exception hierarchy)
- Read edgequake-llm `src/lib.rs`, `src/traits.rs`, `Cargo.toml` to map existing providers and data types
- Fetched maturin docs to establish PyO3/maturin build strategy
- Created `litellm_study/` directory with 8 comprehensive study documents

## Decisions
- Use PyO3 `abi3-py39` stable ABI to produce one wheel per platform (not per Python minor version)
- JSON string bridge (not PyO3 object traversal) for message/opts passing — keeps GIL release clean
- `Arc<Mutex<BoxStream>>` pattern for streaming iterator to satisfy `Send + Sync` requirements
- Phase 1 `acompletion` uses `run_in_executor` as safe fallback; Phase 2 adopts `pyo3-asyncio`
- Python package name: `edgequake-python` (distinct from Rust crate `edgequake-llm`)
- Provider fallback: unknown providers silently delegate to `litellm` if installed

## Next Steps
- Start Phase 1 implementation: modify `Cargo.toml`, create `src/python/` module, write `pyproject.toml`
- Implement `complete()` PyO3 binding for OpenAI provider as proof of concept
- Run `maturin develop --features python` to verify build pipeline end-to-end
- Write smoke test against OpenAI gpt-4o-mini to validate response shape

## Lessons / Insights
- LiteLLM's core surface is actually narrow (completion + embedding + streaming) — the 100+ providers are the real scope; edgequake-llm already covers the 8 most important ones
- The GIL/tokio boundary is the single biggest implementation risk; `Arc<Mutex<BoxStream>>` + `py.allow_threads()` is the established pattern used by polars, pydantic-core, tantivy-py
- maturin + abi3-py39 is the current industry standard (ruff, polars, orjson all use it)
- Streaming in PyO3 requires careful ownership: `BoxStream<'static, ...>` avoids lifetime issues across FFI boundary
