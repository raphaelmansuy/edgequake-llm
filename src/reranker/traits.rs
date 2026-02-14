//! Reranker trait definition.
//!
//! This module defines the core `Reranker` trait that all reranking implementations must satisfy.
//!
//! # Architecture
//!
//! ```ascii
//!                      ┌─────────────────┐
//!                      │  Reranker Trait │
//!                      └────────┬────────┘
//!                               │
//!        ┌──────────────────────┼──────────────────────┐
//!        │                      │                      │
//!        ▼                      ▼                      ▼
//! ┌──────────────┐     ┌──────────────┐      ┌──────────────┐
//! │ HttpReranker │     │ BM25Reranker │      │ HybridRerank │
//! │ (Jina,Cohere)│     │ (Local BM25) │      │ (Combined)   │
//! └──────────────┘     └──────────────┘      └──────────────┘
//! ```
//!
//! # Implementations
//!
//! - [`super::HttpReranker`] - Remote API-based reranking
//! - [`super::TermOverlapReranker`] - Simple term matching
//! - [`super::BM25Reranker`] - BM25 keyword scoring
//! - [`super::RRFReranker`] - Reciprocal Rank Fusion
//! - [`super::HybridReranker`] - Combines multiple strategies

use async_trait::async_trait;

use super::result::RerankResult;
use crate::error::Result;

/// Trait for reranking providers.
///
/// All rerankers implement this trait to provide a consistent interface
/// for document relevance scoring.
///
/// # Required Methods
///
/// - [`name`](Reranker::name) - Identifier for the reranker
/// - [`model`](Reranker::model) - Model/algorithm being used
/// - [`rerank`](Reranker::rerank) - Main reranking operation
///
/// # Provided Methods
///
/// - [`rerank_str`](Reranker::rerank_str) - Convenience for `&str` slices
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Get the name of this reranker.
    fn name(&self) -> &str;

    /// Get the model being used.
    fn model(&self) -> &str;

    /// Rerank documents based on relevance to a query.
    ///
    /// # Arguments
    ///
    /// - `query`: The search query to rank against
    /// - `documents`: Documents to rerank
    /// - `top_n`: Maximum number of results to return (None = all)
    ///
    /// # Returns
    ///
    /// Vector of [`RerankResult`] sorted by relevance (highest first).
    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_n: Option<usize>,
    ) -> Result<Vec<RerankResult>>;

    /// Rerank with documents as string slices.
    ///
    /// Convenience method that converts `&str` to `String`.
    async fn rerank_str(
        &self,
        query: &str,
        documents: &[&str],
        top_n: Option<usize>,
    ) -> Result<Vec<RerankResult>> {
        let docs: Vec<String> = documents.iter().map(|s| s.to_string()).collect();
        self.rerank(query, &docs, top_n).await
    }
}
