//! edgequake-litellm: Python bindings for edgequake-llm.
//!
//! This crate exposes a native `_elc_core` Python extension module built with PyO3.
//! The Python package `edgequake_litellm` wraps this core module with a
//! LiteLLM-compatible high-level API — enabling `import edgequake_litellm as litellm`.
//!
//! # Architecture
//!
//! ```
//! Python callers
//!      │
//!      ▼
//! edgequake_litellm/  (pure Python, LiteLLM-compatible API)
//!      │ imports
//!      ▼
//! _elc_core  (this crate — PyO3 cdylib)
//!      │ uses
//!      ▼
//! edgequake-llm  (Rust LLM core, multi-provider, tokio async)
//! ```

use pyo3::prelude::*;

mod bridge;
mod completion;
mod embedding;
mod types;

/// The `_elc_core` native extension module.
///
/// Do not use this module directly — import `edgequake_litellm` instead.
#[pymodule]
fn _elc_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // -----------------------------------------------------------------------
    // Exported types
    // -----------------------------------------------------------------------
    m.add_class::<types::PyModelResponse>()?;
    m.add_class::<types::PyUsage>()?;
    m.add_class::<types::PyToolCall>()?;
    m.add_class::<types::PyToolCallDelta>()?;
    m.add_class::<types::PyStreamChunk>()?;

    // -----------------------------------------------------------------------
    // Completion functions
    // -----------------------------------------------------------------------
    m.add_function(wrap_pyfunction!(completion::completion, m)?)?;
    m.add_function(wrap_pyfunction!(completion::acompletion, m)?)?;
    m.add_function(wrap_pyfunction!(completion::stream_completion, m)?)?;

    // -----------------------------------------------------------------------
    // Provider info
    // -----------------------------------------------------------------------
    m.add_function(wrap_pyfunction!(completion::list_providers, m)?)?;
    m.add_function(wrap_pyfunction!(completion::detect_provider, m)?)?;

    // -----------------------------------------------------------------------
    // Embedding functions
    // -----------------------------------------------------------------------
    m.add_function(wrap_pyfunction!(embedding::embed, m)?)?;
    m.add_function(wrap_pyfunction!(embedding::aembed, m)?)?;

    // -----------------------------------------------------------------------
    // Version
    // -----------------------------------------------------------------------
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
