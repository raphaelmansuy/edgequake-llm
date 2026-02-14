//! Reranking functionality for improved retrieval quality.
//!
//! This module provides reranking capabilities to improve search result relevance
//! by scoring documents against a query using specialized reranking models.
//!
//! # Architecture
//!
//! ```ascii
//!                    ┌─────────────────────────────┐
//!                    │      Query + Documents      │
//!                    └──────────────┬──────────────┘
//!                                   │
//!                                   ▼
//!     ┌─────────────────────────────────────────────────────┐
//!     │                  Reranker Trait                      │
//!     │   rerank(query, docs, top_n) → Vec<RerankResult>    │
//!     └──────────────────────────┬──────────────────────────┘
//!                                │
//!        ┌───────────────────────┼───────────────────────┐
//!        ▼                       ▼                       ▼
//! ┌──────────────┐      ┌──────────────┐        ┌──────────────┐
//! │ HttpReranker │      │ BM25Reranker │        │HybridReranker│
//! │ (Jina,Cohere)│      │  (Local)     │        │  (Combined)  │
//! └──────────────┘      └──────────────┘        └──────────────┘
//! ```
//!
//! # Module Structure (OODA-02)
//!
//! ```ascii
//! reranker/
//! ├── mod.rs         ─► This file (re-exports)
//! ├── config.rs      ─► RerankConfig, ScoreAggregation
//! ├── result.rs      ─► RerankResult
//! ├── traits.rs      ─► Reranker trait
//! ├── http.rs        ─► HttpReranker (Jina, Cohere, Aliyun)
//! ├── term_overlap.rs─► TermOverlapReranker, MockReranker
//! ├── bm25.rs        ─► BM25Reranker, TokenizerConfig
//! ├── rrf.rs         ─► RRFReranker
//! └── hybrid.rs      ─► HybridReranker
//! ```
//!
//! # Implements
//!
//! - **FEAT0774**: Reranking for improved retrieval
//! - **FEAT0775**: Multi-provider reranker support
//! - **FEAT0776**: BM25 keyword fallback scoring
//!
//! # Enforces
//!
//! - **BR0774**: Top-k results after reranking
//! - **BR0775**: Fallback to BM25 if reranker unavailable
//!
//! # Providers
//!
//! | Provider | Type | Notes |
//! |----------|------|-------|
//! | Jina AI | HTTP | Production multilingual |
//! | Cohere | HTTP | Production rerank-v3.5 |
//! | Aliyun | HTTP | DashScope gte-rerank-v2 |
//! | BM25 | Local | No API key needed |
//! | TermOverlap | Local | Fast testing fallback |
//! | Hybrid | Combined | BM25 + Neural fusion |
//!
//! # Example
//!
//! ```ignore
//! use edgequake_llm::reranker::{BM25Reranker, Reranker};
//!
//! let reranker = BM25Reranker::new();
//! let results = reranker.rerank("rust async", &docs, Some(10)).await?;
//! ```

// Sub-modules (SRP: each module has a single responsibility)
mod bm25;
mod config;
mod http;
mod hybrid;
mod result;
mod rrf;
mod term_overlap;
mod traits;

// Re-export public types - maintains backward compatibility with existing API
pub use bm25::{BM25Reranker, TokenizerConfig};
pub use config::{RerankConfig, ScoreAggregation};
pub use http::HttpReranker;
pub use hybrid::HybridReranker;
pub use result::RerankResult;
pub use rrf::RRFReranker;
pub use term_overlap::{MockReranker, TermOverlapReranker};
pub use traits::Reranker;

#[cfg(test)]
mod tests;
