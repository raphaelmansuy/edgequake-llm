//! Model Discovery System
//!
//! Unified model discovery across all LLM providers. Agent harnesses use this
//! module to programmatically discover available models, query capabilities,
//! and select the best model for a given task.
//!
//! # Design Principles
//!
//! 1. **No Heuristics** — capabilities come from API responses or verified
//!    documentation, never from model name patterns
//! 2. **DRY** — per-provider adapters normalize to `DiscoveredModel`
//! 3. **SOLID** — `ModelDiscoveryProvider` trait (ISP), `ModelDiscoveryService`
//!    depends on trait abstraction (DIP)
//! 4. **Zero Breaking Changes** — additive only, existing APIs untouched
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use edgequake_llm::discovery::{ModelDiscoveryService, CapabilityFilter};
//!
//! let service = ModelDiscoveryService::new();
//!
//! // Find models with vision and tool support
//! let filter = CapabilityFilter {
//!     requires_vision: Some(true),
//!     requires_tools: Some(true),
//!     min_context_length: Some(100_000),
//!     ..Default::default()
//! };
//! let candidates = service.find_models(&filter).await?;
//! ```

pub mod cache;
pub mod providers;
pub mod registry;
pub mod service;
pub mod traits;
pub mod types;

pub use registry::{all_static_models, static_lookup, unknown_model};
pub use service::{ModelDiscoveryService, ModelDiscoveryServiceBuilder};
pub use traits::ModelDiscoveryProvider;
pub use types::{
    CapabilityFilter, DiscoveredModel, DiscoveryError, DiscoverySource, DiscoveryStrategy,
};
