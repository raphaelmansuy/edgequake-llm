//! Term overlap reranker implementation.
//!
//! A simple, fast reranker using Jaccard-like term overlap scoring.
//!
//! # When to Use
//!
//! - Testing and development (no API keys required)
//! - Fallback when BM25 is overkill
//! - Simple use cases with short documents
//!
//! # Limitations
//!
//! - No IDF weighting (rare terms not prioritized)
//! - No term frequency consideration
//! - No length normalization
//!
//! For production, prefer [`super::BM25Reranker`].

use async_trait::async_trait;
use std::collections::HashSet;

use super::result::RerankResult;
use super::traits::Reranker;
use crate::error::Result;

/// Term overlap reranker using simple Jaccard-like scoring.
///
/// # Algorithm
///
/// ```ascii
/// Query Terms:  {capital, of, France}
///                      │
///                      ▼
/// Document:     "The capital of France is Paris"
///                      │
///                      ▼
/// Doc Terms:    {the, capital, of, France, is, Paris}
///                      │
///                      ▼
/// Score = |query ∩ doc| / |query| = 3/3 = 1.0
/// ```
///
/// # Example
///
/// ```ignore
/// use edgequake_llm::reranker::{TermOverlapReranker, Reranker};
///
/// let reranker = TermOverlapReranker::new();
/// let results = reranker.rerank("capital France", &docs, Some(10)).await?;
/// ```
pub struct TermOverlapReranker {
    model: String,
}

impl TermOverlapReranker {
    /// Create a new term overlap reranker.
    pub fn new() -> Self {
        Self {
            model: "term-overlap-reranker".to_string(),
        }
    }
}

impl Default for TermOverlapReranker {
    fn default() -> Self {
        Self::new()
    }
}

/// Backward compatibility alias for `TermOverlapReranker`.
///
/// **Deprecated**: Use `TermOverlapReranker` for new code.
pub type MockReranker = TermOverlapReranker;

#[async_trait]
impl Reranker for TermOverlapReranker {
    fn name(&self) -> &str {
        "term-overlap"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_n: Option<usize>,
    ) -> Result<Vec<RerankResult>> {
        // Score based on query term overlap (Jaccard-like metric)
        let query_lower = query.to_lowercase();
        let query_terms: HashSet<String> = query_lower
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let mut results: Vec<RerankResult> = documents
            .iter()
            .enumerate()
            .map(|(idx, doc)| {
                let doc_lower = doc.to_lowercase();
                let doc_terms: HashSet<String> = doc_lower
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect();

                let overlap = query_terms.intersection(&doc_terms).count();
                let max_terms = query_terms.len().max(1);
                let score = overlap as f64 / max_terms as f64;

                RerankResult {
                    index: idx,
                    relevance_score: score,
                }
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply top_n
        if let Some(n) = top_n {
            results.truncate(n);
        }

        Ok(results)
    }
}
