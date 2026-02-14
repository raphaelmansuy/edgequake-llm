//! Reranking result types.
//!
//! This module contains the result types returned by reranking operations.

use serde::{Deserialize, Serialize};

/// Result from reranking a document.
///
/// # Fields
///
/// - `index`: Position of the document in the original input list
/// - `relevance_score`: Computed relevance score (higher = more relevant)
///
/// # Example
///
/// ```ignore
/// let results = reranker.rerank(query, documents, Some(10)).await?;
/// for result in results {
///     println!("Doc {} has score {:.3}", result.index, result.relevance_score);
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    /// Index of the document in the original list.
    pub index: usize,
    /// Relevance score (higher is more relevant).
    pub relevance_score: f64,
}
