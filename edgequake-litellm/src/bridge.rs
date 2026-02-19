//! Tokio runtime bridge for synchronous PyO3 wrappers.
//!
//! Provides a global multi-threaded tokio runtime used by blocking
//! Python-facing functions. Async functions use pyo3-async-runtimes
//! directly instead of this bridge.

use std::sync::OnceLock;
use tokio::runtime::Runtime;

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Get (or lazily create) the global tokio runtime for blocking calls.
pub fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .thread_name("edgequake-worker")
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime")
    })
}
