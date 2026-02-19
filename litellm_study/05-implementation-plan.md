# Implementation Plan: Phased Roadmap

## Phase Overview

```
Phase 1  [2 weeks]  Foundation
         ─────────────────────
         Cargo.toml changes, maturin scaffolding, PyO3 module init,
         synchronous completion for OpenAI, basic test suite.

Phase 2  [1 week]   Streaming + More Providers
         ─────────────────────────────────────
         Streaming iterators (sync + async), add Anthropic/Gemini/Mistral,
         exception mapping, litellm diff-test harness.

Phase 3  [1 week]   Embeddings + Config + Polish
         ────────────────────────────────────────
         Embedding endpoints, global config vars, env-var resolution,
         type stubs (.pyi), py.typed marker.

Phase 4  [1 week]   CI + PyPI + Docs
         ────────────────────────────
         GitHub Actions matrix (Linux/macOS/Windows × py3.9–3.13),
         manylinux wheels, SDist, publish to PyPI, README, migration guide.

Phase 5  [ongoing]  Parity
         ─────────────────
         More providers, Router class, caching, budget tracking,
         callback system.
```

---

## Phase 1 — Foundation

### 1.1 Cargo.toml modifications

```toml
[lib]
crate-type = ["rlib", "cdylib"]   # add "cdylib"

[features]
python = ["pyo3", "pyo3-asyncio", "once_cell"]

[dependencies]
# ... existing ...
pyo3          = { version = "0.23", features = ["extension-module","abi3-py39"], optional = true }
pyo3-asyncio  = { version = "0.21", features = ["tokio-runtime"], optional = true }
once_cell     = "1"
```

### 1.2 File creation order

```
1. src/python/mod.rs        — pymodule entry point
2. src/python/bridge.rs     — tokio runtime, block_on()
3. src/python/types.rs      — PyLLMResponse, PyStreamChunk
4. src/python/completion.rs — complete(), stream()
5. src/python/embedding.rs  — embed()
6. pyproject.toml           — maturin + PEP-621 config
7. python/edgequake_python/__init__.py
8. python/edgequake_python/_types.py
9. python/edgequake_python/completion.py
10. python/edgequake_python/exceptions.py
```

### 1.3 Add to `src/lib.rs`

```rust
// At the very end of src/lib.rs:
#[cfg(feature = "python")]
pub mod python;
```

### 1.4 Smoke test

```python
# tests/python/test_smoke.py
import os
import edgequake_python as eq

def test_completion_openai():
    response = eq.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Reply with just: OK"}],
    )
    assert response.choices[0].message.content is not None
    assert response.usage.total_tokens > 0

def test_model_response_fields():
    response = eq.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hi"}],
    )
    assert response.id.startswith("chatcmpl")
    assert response.object == "chat.completion"
    assert response.model != ""
```

---

## Phase 2 — Streaming + More Providers

### 2.1 Streaming iterator design (Rust side)

```rust
// PyStreamIter must be:
// - Send + Sync (required for PyO3 classes in GIL-released contexts)
// - backed by Arc<Mutex<BoxStream>> so it can cross thread boundaries
// - __next__ releases GIL via py.allow_threads()

#[pyclass]
pub struct PyStreamIter {
    inner: Arc<Mutex<BoxStream<'static, Result<StreamChunk>>>>,
    rt: &'static Runtime,
}
```

### 2.2 Streaming test

```python
def test_streaming():
    collected = []
    for chunk in eq.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Count 1 to 5"}],
        stream=True,
    ):
        delta = chunk.choices[0].delta.content or ""
        collected.append(delta)
    full = "".join(collected)
    assert "1" in full and "5" in full
```

### 2.3 LiteLLM diff-test harness

```python
# tests/python/test_litellm_compat.py
"""
Tests that edgequake_python responses are structurally identical to litellm.
Run only when LITELLM_COMPAT_TESTS=1 to avoid requiring both packages.
"""
import pytest, os
ENABLED = os.getenv("LITELLM_COMPAT_TESTS") == "1"

@pytest.mark.skipif(not ENABLED, reason="set LITELLM_COMPAT_TESTS=1")
def test_response_shape_compat():
    import litellm
    import edgequake_python as eq

    msg = [{"role": "user", "content": "Hi, say hello"}]
    a = eq.completion("openai/gpt-4o-mini", msg)
    b = litellm.completion("openai/gpt-4o-mini", msg)

    # Same top-level fields
    assert set(a.model_fields.keys()) >= {"id","object","created","model","choices","usage"}
    assert a.choices[0].message.role == b.choices[0].message.role
    assert type(a.usage.prompt_tokens) == type(b.usage.prompt_tokens)
```

---

## Phase 3 — Embeddings, Config, Polish

### 3.1 Embedding Rust binding

```rust
#[pyfunction]
pub fn embed(
    py: Python<'_>,
    provider: &str,
    model: &str,
    inputs: Vec<String>,
    api_key: &str,
    api_base: &str,
) -> PyResult<PyEmbeddingResult> {
    let config = build_config(provider, model, api_key, api_base);
    block_on(py, async move {
        let prov = ProviderFactory::create_embed_from_config(&config)?;
        prov.embed(&inputs).await
    }).map(PyEmbeddingResult::from)
}
```

### 3.2 Global config (Python module-level `__setattr__` intercept)

```python
# edgequake_python/__init__.py addition:
import sys

class _Module(sys.modules[__name__].__class__):
    def __setattr__(self, name, value):
        # Propagate key changes to the Rust layer
        if name.endswith("_key") or name in ("api_base", "num_retries"):
            from . import config as _cfg
            setattr(_cfg, name, value)
            # Also push into Rust global config if needed:
            # _eq_core.set_config(name, str(value))
        super().__setattr__(name, value)

sys.modules[__name__].__class__ = _Module
```

---

## Phase 4 — CI, Wheels, PyPI

### 4.1 GitHub Actions matrix

```yaml
# .github/workflows/python-publish.yml
name: Build & Publish Python Wheels

on:
  push:
    tags: ["py-v*"]

jobs:
  build:
    strategy:
      matrix:
        include:
          # Linux (manylinux)
          - os: ubuntu-latest
            target: x86_64
            manylinux: "2014"
          - os: ubuntu-latest
            target: aarch64
            manylinux: "2014"
          # macOS
          - os: macos-13        # Intel
            target: x86_64
          - os: macos-14        # Apple Silicon
            target: aarch64
          # Windows
          - os: windows-latest
            target: x86_64

    steps:
      - uses: actions/checkout@v4
      - uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.target }}
          manylinux: ${{ matrix.manylinux || 'off' }}
          args: --release --features python --out dist
          sccache: true

      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}-${{ matrix.target }}
          path: dist

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          pattern: wheels-*
          path: dist
          merge-multiple: true
      - uses: pypa/gh-action-pypi-publish@release/v1
```

### 4.2 Local test before publish

```bash
# 1. Build all wheels locally
maturin build --release --features python

# 2. Install and smoke test
pip install target/wheels/edgequake_python-*.whl
python -c "import edgequake_python as eq; print(eq.__version__)"

# 3. Upload to TestPyPI first
maturin upload --repository testpypi target/wheels/*.whl
pip install --index-url https://test.pypi.org/simple/ edgequake-python

# 4. Upload to real PyPI
maturin upload target/wheels/*.whl
```

---

## Phase 5 — Parity Roadmap

```
Feature                        Priority  Effort  Notes
─────────────────────────────────────────────────────────
Tool calling (streaming)        P0        M       StreamChunk::ToolCallDelta
Anthropic vision                P0        S       ImageData in traits.rs
Gemini vision                   P0        S
Ollama local models             P0        XS      already in edgequake-llm
Router class                    P1        L       Python-side round-robin/fallback
Budget tracking                 P1        M       wrap SessionCostTracker
Caching (Redis/in-memory)      P1        M       CachedProvider already exists
Callback logging (Langfuse)    P2        L       new Python integration layer
Bedrock (AWS)                  P2        XL      new Rust provider needed
Vertex AI                      P2        XL      new Rust provider needed
Async streaming (pyo3-asyncio) P0        M       PyAsyncStreamIter
```

---

## Folder Structure After Phase 1 Complete

```
edgequake-llm/
├── Cargo.toml                ← modified: cdylib + python feature
├── pyproject.toml            ← NEW
├── src/
│   ├── lib.rs                ← +mod python at bottom
│   └── python/               ← NEW
│       ├── mod.rs
│       ├── bridge.rs
│       ├── types.rs
│       ├── completion.rs
│       └── embedding.rs
├── python/                   ← NEW
│   └── edgequake_python/
│       ├── __init__.py
│       ├── _types.py
│       ├── completion.py
│       ├── embedding.py
│       ├── streaming.py
│       ├── exceptions.py
│       ├── config.py
│       └── py.typed
├── tests/
│   └── python/
│       ├── test_smoke.py
│       ├── test_streaming.py
│       ├── test_embedding.py
│       └── test_litellm_compat.py
├── .github/
│   └── workflows/
│       └── python-publish.yml  ← NEW
└── litellm_study/             ← this directory
```
