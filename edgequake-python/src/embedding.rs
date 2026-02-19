//! Embedding functions exposed to Python.
//!
//! # Functions
//!
//! * `embed(provider, model, texts)` → `List[List[float]]`
//! * `aembed(provider, model, texts)` → `Awaitable[List[List[float]]]`

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyList;

use edgequake_llm::factory::{ProviderFactory, ProviderType};

use crate::bridge::runtime;
use crate::types::to_py_err;

fn parse_provider(s: &str) -> PyResult<ProviderType> {
    ProviderType::from_str(s)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown provider '{}'", s)))
}

// ---------------------------------------------------------------------------
// Synchronous embed — releases GIL
// ---------------------------------------------------------------------------

/// Generate embeddings synchronously.
///
/// Args:
///     provider: Provider name (e.g. "openai", "ollama", "huggingface").
///     model:    Embedding model name (e.g. "text-embedding-3-small").
///     texts:    List of strings to embed.
///
/// Returns:
///     List[List[float]] where each inner list is one embedding vector.
#[pyfunction]
pub fn embed(
    py: Python<'_>,
    provider: &str,
    model: &str,
    texts: Vec<String>,
) -> PyResult<Vec<Vec<f32>>> {
    let provider_type = parse_provider(provider)?;
    let model_owned = model.to_string();

    let _ = py; // GIL held during blocking I/O
    runtime().block_on(async move {
        let (_, embedding) =
            ProviderFactory::create_with_model(provider_type, Some(&model_owned))
                .map_err(to_py_err)?;

        embedding.embed(&texts).await.map_err(to_py_err)
    })
}

// ---------------------------------------------------------------------------
// Asynchronous embed — returns Python coroutine
// ---------------------------------------------------------------------------

/// Generate embeddings asynchronously.
///
/// Returns an awaitable coroutine.
///
/// Args: same as `embed`.
///
/// Returns:
///     Awaitable[List[List[float]]]
#[pyfunction]
pub fn aembed<'py>(
    py: Python<'py>,
    provider: &str,
    model: &str,
    texts: Vec<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let provider_type = parse_provider(provider)?;
    let model_owned = model.to_string();

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let (_, embedding) =
            ProviderFactory::create_with_model(provider_type, Some(&model_owned))
                .map_err(to_py_err)?;

        let vecs = embedding.embed(&texts).await.map_err(to_py_err)?;

        // Convert to Python list of lists inside a brief GIL re-acquisition.
        Python::try_attach(|py| -> PyResult<Py<PyAny>> {
            let outer: Vec<Py<PyAny>> = vecs
                .into_iter()
                .map(|v| {
                    let inner: Vec<f32> = v;
                    Ok(PyList::new(py, inner)?.into_any().unbind())
                })
                .collect::<PyResult<_>>()?;
            let result = PyList::new(py, outer)?;
            Ok(result.into_any().unbind())
        })
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Python interpreter not available"))?
    })
}
