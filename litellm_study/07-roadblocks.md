# Roadblocks, Risks & Mitigations

## Overview

```
Risk Matrix
───────────────────────────────────────────────────────────────────────
ID   Category           Severity  Likelihood  Mitigation
───────────────────────────────────────────────────────────────────────
R01  GIL & threading      HIGH      HIGH       block_on + allow_threads
R02  Async compatibility  HIGH      MEDIUM     pyo3-asyncio + executor
R03  ABI stability        MEDIUM    LOW        abi3-py39 stable ABI
R04  Streaming ownership  HIGH      HIGH       Arc<Mutex<BoxStream>>
R05  Type mismatch        MEDIUM    MEDIUM     JSON bridge + Pydantic
R06  Provider coverage    HIGH      HIGH       Phased plan, fallback
R07  Test surface         HIGH      MEDIUM     LiteLLM diff-test harness
R08  Windows builds       MEDIUM    MEDIUM     rustls (no OpenSSL dep)
R09  Licensing            LOW       LOW        Apache-2.0 compatible
R10  PyO3 version sync    MEDIUM    MEDIUM     lock PyO3 + maturin matrix
───────────────────────────────────────────────────────────────────────
```

---

## R01 — GIL and Threading

### Problem

PyO3 requires that all Python object access happens while the GIL is held.
Rust async code (tokio) runs on its own thread pool.
Misuse causes segfaults or deadlocks.

### Solution

```
Python call enters _eq_core.complete()
         │
         ▼ py.allow_threads(|| {  ← GIL RELEASED
             tokio::block_on(async { ... })   ← Rust I/O here
         })                       ← GIL RE-ACQUIRED
         │
         ▼ Python: build ModelResponse
```

Rules enforced in code:
1. Never hold a Python object (`Py<T>`) without the GIL
2. Use `Arc<Mutex<>>` for anything shared between tokio tasks and Python
3. `PyStreamIter.__next__()` calls `py.allow_threads()` for each chunk

### Detection

Compile-time: PyO3's `Send` bound enforcement catches most cases.
Runtime test: Run with `PYTHONFAULTHANDLER=1` to get crash traces.

---

## R02 — Async Compatibility (asyncio vs tokio)

### Problem

Python has its own event loop (asyncio). Rust uses tokio.
`pyo3-asyncio` bridges them, but:
- The asyncio loop must be running
- `anyio` / `trio` users need different bridges
- `nest_asyncio` scenarios (Jupyter) need care

### Solutions by scenario

```
Scenario                      Solution
────────────────────────────────────────────────────────────────────
Standard asyncio script       pyo3_asyncio_0_21::tokio::future_into_py()
IPython / Jupyter             loop.run_in_executor(None, sync_fn)
anyio/trio                    anyio.to_thread.run_sync(sync_fn)
Sync-only usage               bridge::block_on() — no asyncio needed
FastAPI (asyncio)             pyo3-asyncio with get_event_loop()
```

### Short-term workaround (Phase 1)

```python
async def acompletion(model, messages, **kwargs):
    # Phase 1: run sync version in thread pool (correct, slightly slower)
    import asyncio
    return await asyncio.get_event_loop().run_in_executor(
        None, lambda: completion(model, messages, **kwargs)
    )
```

### Phase 2: native async with pyo3-asyncio

```rust
#[pyfunction]
fn acomplete<'py>(py: Python<'py>, ...) -> PyResult<Bound<'py, PyAny>> {
    pyo3_asyncio_0_21::tokio::future_into_py(py, async move {
        // runs on tokio, resolves to Python awaitable
        let resp = provider.chat(&messages, &opts).await?;
        Python::with_gil(|py| Ok(PyLLMResponse::from(resp).into_py(py)))
    })
}
```

---

## R03 — ABI Stability

### Problem

By default, Python extension modules are compiled for a specific
Python version (e.g. `cp312`). This means separate wheels for each
Python 3.x minor version.

### Solution

```toml
# Cargo.toml
pyo3 = { version = "0.23", features = ["abi3-py39"] }
```

This uses CPython's Stable ABI (PEP 384), producing a single wheel tag
`cp39-abi3` that works on Python 3.9, 3.10, 3.11, 3.12, 3.13.

**Trade-off**: Can't use some bleeding-edge PyO3 APIs that rely on
version-specific internals. In practice this is rarely a limitation.

---

## R04 — Streaming Ownership Across Thread Boundaries

### Problem

`BoxStream<'static, Result<StreamChunk>>` is not `Send` by default
unless the inner stream is. PyO3 `#[pyclass]` requires `Send + Sync`.

### Solution

```rust
#[pyclass]
pub struct PyStreamIter {
    // Arc<Mutex<...>> makes the stream Send + Sync
    inner: Arc<Mutex<BoxStream<'static, Result<StreamChunk>>>>,
    rt:    &'static Runtime,   // 'static lifetime, safe
}
```

The `Mutex` ensures only one thread polls the stream at a time
(correct for Python's sequential iteration model).

### Lifetime management

```
Python GC calls __del__ on PyStreamIter
    → Rust Drop runs
    → Arc ref count decrements
    → if last ref: BoxStream dropped, connection closed
```

This is correct because the stream is `'static` — it owns all its data.

---

## R05 — JSON Bridge Type Mismatch

### Problem

Passing structured data as JSON strings (messages, opts) is safe but
introduces serialisation overhead and potential mismatch between what
Python passes and what Rust expects.

### Quantification

```
messages = [{"role": "user", "content": "Hello"}]
JSON serialise:  ~2 µs   (Python json.dumps)
Rust deserialise: ~1 µs  (serde_json)
Total overhead:   ~3 µs  → negligible vs ~100ms network latency
```

### Alternative considered: PyO3 direct object access

```rust
// Could use PyDict directly:
fn complete(py: Python, messages: &PyList) -> PyResult<...> {
    for item in messages.iter() {
        let role = item.downcast::<PyDict>()?.get_item("role")?;
    }
}
```

But this holds the GIL during parsing, blocking other Python threads.
→ **JSON bridge is the right choice** for I/O-bound LLM calls.

---

## R06 — Provider Coverage Gap

### Problem

edgequake-llm supports ~10 providers; LiteLLM supports 100+.
Missing major providers: HuggingFace, Cohere, Bedrock (AWS), Vertex AI.

### Mitigation Strategy

```
Tier    Providers                     Strategy
────────────────────────────────────────────────────────────────────────
Core    OpenAI, Azure, Anthropic,     Implemented in Rust (Phase 1-2)
        Gemini, Mistral, Ollama,
        OpenRouter, xAI

Bridge  HuggingFace, Cohere,          Rust → OpenAI-compatible endpoint
        many OpenAI-compatible APIs   via OpenAICompatibleProvider

Fallback Bedrock, VertexAI, etc.      Python-level fallback: if provider
                                      not in Rust, delegate to litellm
                                      (if installed)
```

### Python-level fallback

```python
def completion(model, messages, **kwargs):
    provider, _ = _parse_model(model)
    if provider in RUST_SUPPORTED_PROVIDERS:
        return _rust_complete(...)
    else:
        # Graceful degradation: try litellm if available
        try:
            import litellm as _ll
            return _ll.completion(model, messages, **kwargs)
        except ImportError:
            raise UnsupportedProviderError(
                f"Provider '{provider}' not yet supported. "
                f"Install litellm as fallback: pip install litellm"
            )
```

---

## R07 — Test Surface

### Problem

LiteLLM has ~1000 test files covering subtle edge cases.
We must not accidentally break compatibility for common patterns.

### Test Pyramid

```
Unit tests (no network):
  - _parse_model() edge cases
  - _to_model_response() field mapping
  - Exception mapping from Rust errors
  - Streaming chunk assembly
  - Config/env-var resolution

Integration tests (real API, use API keys):
  - completion() for each provider
  - Streaming iteration
  - Tool calling round-trip
  - Vision (base64 image)
  - Embedding shapes

LiteLLM diff tests:
  - Structural comparison of ModelResponse
  - Streaming delta identity
  - Exception type equality
```

### CI strategy

```yaml
- name: Run unit tests (no network)
  run: pytest tests/python/unit/ -v

- name: Run integration tests
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: pytest tests/python/integration/ -v -m "not slow"

- name: Run LiteLLM compat tests
  env:
    LITELLM_COMPAT_TESTS: "1"
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: pip install litellm && pytest tests/python/compat/ -v
```

---

## R08 — Windows Build Complexity

### Problem

- Rust on Windows links MSVCRT; Python may use a different CRT
- OpenSSL dependency is painful on Windows
- Cross-compilation is not supported (need Windows runner)

### Solution

```toml
# Cargo.toml — use rustls (pure Rust TLS, no OpenSSL)
reqwest = { version = "0.12", default-features = false, features = [
    "json",
    "rustls-tls",    ← no OpenSSL needed
    "stream",
] }
```

edgequake-llm already uses `rustls-tls`. Windows builds will work
out of the box in GitHub Actions `windows-latest`.

---

## R09 — Licensing

### Analysis

```
edgequake-llm Rust crate:  Apache-2.0
LiteLLM:                   MIT
PyO3:                       MIT or Apache-2.0
maturin:                    MIT or Apache-2.0
pydantic:                   MIT
```

All dependencies are permissive. **No GPL/LGPL contamination.**

edgequake-python published under **Apache-2.0** (same as Rust crate).

Note: We are NOT copying LiteLLM code — only reimplementing the public API
shape (function signatures + response types). This is API compatibility,
not code copying.

---

## R10 — PyO3 / maturin Version Synchronisation

### Problem

PyO3 and maturin must be version-compatible:
- pyo3 0.23.x requires maturin 1.6+
- pyo3-asyncio 0.21.x requires pyo3 0.21+

### Pin strategy

```toml
# Cargo.toml
pyo3          = "0.23"    # minor updates OK, not major
pyo3-asyncio  = "0.21"    # keep in sync with pyo3

# pyproject.toml
[build-system]
requires = ["maturin>=1.7,<2.0"]   # allow maturin patch updates
```

Review the [PyO3 migration guide](https://pyo3.rs/latest/migration.html)
before any major version bump.

---

## Summary: Risk-Ranked Work Items

```
Priority  Risk  Action
──────────────────────────────────────────────────────────────────────
1   R01/R04  Implement GIL-safe streaming iterator (Arc<Mutex<BoxStream>>)
2   R02      Phase 1: use run_in_executor; Phase 2: pyo3-asyncio native
3   R06      Document provider fallback; implement Python-level delegator
4   R07      Build diff-test harness against litellm before any release
5   R05      Benchmark JSON bridge overhead; switch to PyO3 object access only if >1ms
6   R08      Verify Windows CI build with rustls on day 1
7   R03      abi3-py39 in Cargo.toml from day 1 - avoids wheel matrix explosion
```
