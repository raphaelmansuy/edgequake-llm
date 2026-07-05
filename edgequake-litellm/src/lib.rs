//! edgequake-litellm: Python bindings for edgequake-llm.
//!
//! This crate exposes a native `_elc_core` Python extension module built with PyO3.
//! The Python package `edgequake_litellm` wraps this core module with a
//! LiteLLM-compatible high-level API — enabling `import edgequake_litellm as litellm`.
//!
//! # Architecture
//!
//! ```text
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

mod attribution;
mod bridge;
mod completion;
mod discovery;
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
    m.add_class::<attribution::PyApplicationContext>()?;

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
    m.add_function(wrap_pyfunction!(attribution::get_provider_attribution, m)?)?;
    m.add_function(wrap_pyfunction!(attribution::list_provider_attribution, m)?)?;

    // -----------------------------------------------------------------------
    // Embedding functions
    // -----------------------------------------------------------------------
    m.add_function(wrap_pyfunction!(embedding::embed, m)?)?;
    m.add_function(wrap_pyfunction!(embedding::aembed, m)?)?;

    // -----------------------------------------------------------------------
    // Discovery functions
    // -----------------------------------------------------------------------
    m.add_class::<discovery::PyDiscoveredModel>()?;
    m.add_class::<discovery::PyModelSearchMatch>()?;
    m.add_function(wrap_pyfunction!(discovery::discover_all, m)?)?;
    m.add_function(wrap_pyfunction!(discovery::adiscover_all, m)?)?;
    m.add_function(wrap_pyfunction!(discovery::find_models, m)?)?;
    m.add_function(wrap_pyfunction!(discovery::find_static_models, m)?)?;
    m.add_function(wrap_pyfunction!(discovery::search_models, m)?)?;
    m.add_function(wrap_pyfunction!(
        discovery::search_static_models_by_name,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(discovery::lookup_model_by_name, m)?)?;
    m.add_function(wrap_pyfunction!(discovery::asearch_models, m)?)?;
    m.add_function(wrap_pyfunction!(discovery::afind_models, m)?)?;
    m.add_function(wrap_pyfunction!(discovery::get_model, m)?)?;
    m.add_function(wrap_pyfunction!(discovery::aget_model, m)?)?;
    m.add_function(wrap_pyfunction!(discovery::discover_provider, m)?)?;
    m.add_function(wrap_pyfunction!(discovery::list_discovery_providers, m)?)?;
    m.add_function(wrap_pyfunction!(discovery::discovery_info, m)?)?;

    // -----------------------------------------------------------------------
    // Version
    // -----------------------------------------------------------------------
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
