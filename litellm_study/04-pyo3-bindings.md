# PyO3 Bindings & Maturin Integration

## 1. Technology Stack

```
┌─────────────────────────────────────────────────────────────────┐
│  Python 3.9+                                                     │
├─────────────────────────────────────────────────────────────────┤
│  pyo3  v0.23+          High-level Rust↔Python bindings          │
│  pyo3-asyncio v0.21+   Bridges tokio futures ↔ asyncio          │
│  maturin v1.7+         Build system: cargo → wheel              │
│  tokio v1.x            Async runtime inside Rust extension      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Cargo.toml Changes

Add a new section to the **existing** `Cargo.toml`:

```toml
[lib]
# Keep the existing rlib for Rust consumers AND add cdylib for Python
crate-type = ["rlib", "cdylib"]

[features]
default = []
python = ["pyo3", "pyo3-asyncio"]
otel   = ["opentelemetry", "tracing-opentelemetry"]

[dependencies]
# ... existing deps ...

# Python bindings (optional, activated by `python` feature)
pyo3            = { version = "0.23", features = ["extension-module", "abi3-py39"], optional = true }
pyo3-asyncio    = { version = "0.21", features = ["tokio-runtime"], optional = true }
once_cell       = "1"     # for global RUNTIME lazy init
```

> **Note**: `abi3-py39` builds a stable-ABI wheel that works on Python 3.9–3.x
> without recompiling per minor version. One wheel covers Python 3.9, 3.10, 3.11, 3.12, 3.13.

---

## 3. pyproject.toml

```toml
[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[project]
name = "edgequake-python"
version = "0.1.0"
description = "High-performance LiteLLM-compatible LLM client backed by Rust"
readme = "python/README_PYPI.md"
license = { text = "Apache-2.0" }
requires-python = ">=3.9"
keywords = ["llm", "openai", "anthropic", "litellm", "ai", "rust"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Rust",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "pydantic>=2.0",
]

[project.optional-dependencies]
async = ["anyio>=4.0"]
dev   = ["pytest", "pytest-asyncio", "maturin[patchelf]"]

[project.urls]
Homepage   = "https://github.com/raphaelmansuy/edgequake-llm"
Repository = "https://github.com/raphaelmansuy/edgequake-llm"

[tool.maturin]
# Python package lives in python/edgequake_python/
python-source  = "python"
module-name    = "edgequake_python._eq_core"
features       = ["python"]
# Build universal2 on macOS for Apple Silicon + Intel
# universal2 = true  # uncomment for macOS release builds
```

---

## 4. Rust Module Entry Point (`src/python/mod.rs`)

```rust
// src/python/mod.rs
#![cfg(feature = "python")]

use pyo3::prelude::*;

mod bridge;
mod completion;
mod embedding;
mod types;

/// Python extension module `_eq_core`.
///
/// Registered as `edgequake_python._eq_core` via pyproject.toml.
#[pymodule]
#[pyo3(name = "_eq_core")]
pub fn init_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialise global tokio runtime early
    bridge::init_runtime();

    // Register classes
    m.add_class::<types::PyLLMResponse>()?;
    m.add_class::<types::PyEmbeddingResult>()?;
    m.add_class::<types::PyStreamIter>()?;
    m.add_class::<types::PyAsyncStreamIter>()?;

    // Register free functions
    m.add_function(wrap_pyfunction!(completion::complete, m)?)?;
    m.add_function(wrap_pyfunction!(completion::stream, m)?)?;
    m.add_function(wrap_pyfunction!(embedding::embed, m)?)?;

    // Async variants  (pyo3-asyncio)
    m.add_function(wrap_pyfunction!(completion::acomplete, m)?)?;
    m.add_function(wrap_pyfunction!(completion::astream, m)?)?;
    m.add_function(wrap_pyfunction!(embedding::aembed, m)?)?;

    Ok(())
}
```

---

## 5. Tokio Runtime Bridge (`src/python/bridge.rs`)

```rust
// src/python/bridge.rs
#![cfg(feature = "python")]

use once_cell::sync::OnceCell;
use pyo3::prelude::*;
use tokio::runtime::{Builder, Runtime};

/// Global shared Tokio runtime.
/// Created once on first call to `init_runtime()`.
static RUNTIME: OnceCell<Runtime> = OnceCell::new();

pub fn init_runtime() {
    RUNTIME.get_or_init(|| {
        Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .thread_name("eq-tokio")
            .build()
            .expect("Failed to create tokio runtime")
    });
}

pub fn runtime() -> &'static Runtime {
    RUNTIME.get().expect("runtime not initialised; call init_runtime() first")
}

/// Run an async Rust future from a synchronous PyO3 call.
/// Releases the GIL while the future is running.
pub fn block_on<F, T>(py: Python<'_>, fut: F) -> PyResult<T>
where
    F: std::future::Future<Output = crate::error::Result<T>> + Send + 'static,
    T: Send + 'static,
{
    py.allow_threads(|| {
        runtime().block_on(fut).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e))
        })
    })
}
```

---

## 6. Completion Bindings (`src/python/completion.rs`)

```rust
// src/python/completion.rs
#![cfg(feature = "python")]

use pyo3::prelude::*;
use serde_json::Value;
use std::collections::HashMap;

use crate::factory::{ProviderFactory, ProviderType};
use crate::traits::{ChatMessage, ChatRole, CompletionOptions};
use super::bridge::block_on;
use super::types::{PyLLMResponse, PyStreamIter};

/// Synchronous completion — called from Python's `completion()`.
#[pyfunction]
#[pyo3(signature = (provider, model, messages, opts, api_key="", api_base=""))]
pub fn complete(
    py: Python<'_>,
    provider: &str,
    model: &str,
    messages: &str,    // JSON string
    opts: &str,        // JSON string
    api_key: &str,
    api_base: &str,
) -> PyResult<PyLLMResponse> {
    let messages = parse_messages(messages)?;
    let options  = parse_options(opts)?;
    let config   = build_config(provider, model, api_key, api_base);

    block_on(py, async move {
        let provider_impl = ProviderFactory::create_from_config(&config)
            .map_err(|e| crate::error::LlmError::Other(e.to_string()))?;
        provider_impl.chat(&messages, &options).await
    })
    .map(PyLLMResponse::from)
}

/// Streaming completion — returns a PyStreamIter.
#[pyfunction]
#[pyo3(signature = (provider, model, messages, opts, api_key="", api_base=""))]
pub fn stream(
    py: Python<'_>,
    provider: &str,
    model: &str,
    messages: &str,
    opts: &str,
    api_key: &str,
    api_base: &str,
) -> PyResult<PyStreamIter> {
    // Build provider and return a blocking stream iterator
    let messages = parse_messages(messages)?;
    let options  = parse_options(opts)?;
    let config   = build_config(provider, model, api_key, api_base);

    let boxed_stream = block_on(py, async move {
        let prov = ProviderFactory::create_from_config(&config)?;
        // Get the stream handle (a BoxStream<StreamChunk>)
        Ok(prov.chat_stream(&messages, &options).await?)
    })?;

    Ok(PyStreamIter::new(boxed_stream))
}

// --- helpers omitted for brevity (parse_messages, parse_options, build_config) ---
```

---

## 7. PyO3 Type Wrappers (`src/python/types.rs`)

```rust
// src/python/types.rs
#![cfg(feature = "python")]

use pyo3::prelude::*;
use pyo3::types::PyList;
use crate::traits::{LLMResponse, StreamChunk, ToolCall};
use crate::error::Result as EqResult;
use tokio::sync::Mutex;
use futures::stream::BoxStream;
use std::sync::Arc;

/// LLM response exposed to Python.
#[pyclass(name = "PyLLMResponse")]
pub struct PyLLMResponse {
    #[pyo3(get)] pub content:           String,
    #[pyo3(get)] pub prompt_tokens:     usize,
    #[pyo3(get)] pub completion_tokens: usize,
    #[pyo3(get)] pub total_tokens:      usize,
    #[pyo3(get)] pub model:             String,
    #[pyo3(get)] pub finish_reason:     Option<String>,
    #[pyo3(get)] pub tool_calls_json:   String,    // JSON-encoded Vec<ToolCall>
    #[pyo3(get)] pub thinking_tokens:   Option<usize>,
}

impl From<LLMResponse> for PyLLMResponse {
    fn from(r: LLMResponse) -> Self {
        Self {
            content: r.content,
            prompt_tokens: r.prompt_tokens,
            completion_tokens: r.completion_tokens,
            total_tokens: r.total_tokens,
            model: r.model,
            finish_reason: r.finish_reason,
            tool_calls_json: serde_json::to_string(&r.tool_calls).unwrap_or_default(),
            thinking_tokens: r.thinking_tokens,
        }
    }
}

/// Synchronous stream iterator for Python.
#[pyclass(name = "PyStreamIter")]
pub struct PyStreamIter {
    // Arc<Mutex<...>> allows Send across the PyO3 boundary
    inner: Arc<Mutex<BoxStream<'static, EqResult<StreamChunk>>>>,
    rt:    &'static tokio::runtime::Runtime,
}

#[pymethods]
impl PyStreamIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<PyStreamChunk>> {
        use futures::StreamExt;
        let inner = Arc::clone(&self.inner);
        let rt = self.rt;
        py.allow_threads(|| {
            rt.block_on(async move {
                let mut guard = inner.lock().await;
                guard.next().await
                    .transpose()
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
                    .map(|opt| opt.map(PyStreamChunk::from))
            })
        })
    }
}

/// One streaming chunk.
#[pyclass(name = "PyStreamChunk")]
pub struct PyStreamChunk {
    #[pyo3(get)] pub content:       Option<String>,
    #[pyo3(get)] pub finish_reason: Option<String>,
    #[pyo3(get)] pub prompt_tokens:     usize,
    #[pyo3(get)] pub completion_tokens: usize,
    #[pyo3(get)] pub total_tokens:      usize,
}

impl From<StreamChunk> for PyStreamChunk {
    fn from(c: StreamChunk) -> Self {
        match c {
            StreamChunk::Content(s) => PyStreamChunk {
                content: Some(s),
                finish_reason: None,
                prompt_tokens: 0, completion_tokens: 0, total_tokens: 0,
            },
            StreamChunk::Finished { reason, .. } => PyStreamChunk {
                content: None,
                finish_reason: Some(reason),
                prompt_tokens: 0, completion_tokens: 0, total_tokens: 0,
            },
            _ => PyStreamChunk {
                content: None, finish_reason: None,
                prompt_tokens: 0, completion_tokens: 0, total_tokens: 0,
            },
        }
    }
}
```

---

## 8. Build Commands

```bash
# Development install (fast, no optimisations)
cd edgequake-llm
pip install maturin
maturin develop --features python

# Release build (optimised)
maturin build --release --features python

# Build universal macOS wheel (Intel + Apple Silicon)
maturin build --release --features python --universal2

# Build manylinux wheel for PyPI
docker run --rm -v $(pwd):/io \
  ghcr.io/pyo3/maturin \
  build --release --features python --manylinux 2014

# Install locally
pip install target/wheels/edgequake_python-0.1.0-*.whl
```

---

## 9. Async Python Support with pyo3-asyncio

```rust
// acompletion in Rust — returns a Python coroutine
#[pyfunction]
pub fn acomplete<'py>(
    py: Python<'py>,
    provider: String,
    model: String,
    messages: String,
    opts: String,
    api_key: String,
    api_base: String,
) -> PyResult<Bound<'py, PyAny>> {
    // pyo3_asyncio::tokio::future_into_py returns a Python coroutine
    // that the Python asyncio event loop can await.
    pyo3_asyncio_0_21::tokio::future_into_py(py, async move {
        let messages = parse_messages(&messages)?;
        let options  = parse_options(&opts)?;
        let config   = build_config(&provider, &model, &api_key, &api_base);
        let prov = ProviderFactory::create_from_config(&config)?;
        let resp = prov.chat(&messages, &options).await?;
        Python::with_gil(|py| Ok(PyLLMResponse::from(resp).into_py(py)))
    })
}
```

On the Python side:
```python
# This works with asyncio, trio, anyio:
response = await _eq_core.acomplete(
    provider, model, messages_json, opts_json, api_key, api_base
)
```
