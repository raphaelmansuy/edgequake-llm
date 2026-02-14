//! Reciprocal Rank Fusion (RRF) reranker.
//!
//! Combines multiple ranking signals without needing score normalization.
//!
//! # Algorithm
//!
//! ```ascii
//! RRF Score = Σ 1/(k + rank) for each ranking list
//!
//! Where:
//! - k = smoothing constant (default 60)
//! - rank = 1-indexed position in each list
//! ```
//!
//! # Use Cases
//!
//! - Combining vector similarity + BM25 rankings
//! - Combining results from multiple queries
//! - Hybrid search scenarios

use async_trait::async_trait;

use super::bm25::BM25Reranker;
use super::result::RerankResult;
use super::traits::Reranker;
use crate::error::Result;

/// Reciprocal Rank Fusion reranker.
///
/// # Example
///
/// ```ignore
/// use edgequake_llm::reranker::RRFReranker;
///
/// let rrf = RRFReranker::new();
///
/// // Combine BM25 and vector rankings
/// let bm25_ranking = vec![0, 2, 1]; // doc 0 best, then 2, then 1
/// let vector_ranking = vec![1, 0, 2]; // doc 1 best, then 0, then 2
///
/// let fused = rrf.fuse(&[bm25_ranking, vector_ranking], 3);
/// // Result balances both signals
/// ```
pub struct RRFReranker {
    /// Ranking constant (higher = lower-ranked docs have more influence).
    k: u32,
    /// Model name for trait compliance.
    model: String,
}

impl RRFReranker {
    /// Create a new RRF reranker with default k=60.
    pub fn new() -> Self {
        Self {
            k: 60,
            model: "rrf-reranker".to_string(),
        }
    }

    /// Create with custom k value.
    pub fn with_k(k: u32) -> Self {
        Self {
            k: k.max(1),
            model: "rrf-reranker".to_string(),
        }
    }

    /// Fuse multiple ranked lists using RRF.
    ///
    /// Each inner Vec contains document indices in ranked order (best first).
    ///
    /// # Algorithm
    ///
    /// ```ascii
    /// For each ranking list:
    ///   For each (rank, doc_idx) in list:
    ///     scores[doc_idx] += 1 / (k + rank + 1)
    ///
    /// Return docs sorted by total score
    /// ```
    pub fn fuse(&self, ranked_lists: &[Vec<usize>], num_docs: usize) -> Vec<RerankResult> {
        let mut scores = vec![0.0f64; num_docs];

        for ranked_list in ranked_lists {
            for (rank, &doc_idx) in ranked_list.iter().enumerate() {
                if doc_idx < num_docs {
                    scores[doc_idx] += 1.0 / (self.k as f64 + rank as f64 + 1.0);
                }
            }
        }

        let mut results: Vec<RerankResult> = scores
            .into_iter()
            .enumerate()
            .filter(|(_, score)| *score > 0.0)
            .map(|(idx, score)| RerankResult {
                index: idx,
                relevance_score: score,
            })
            .collect();

        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }
}

impl Default for RRFReranker {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Reranker for RRFReranker {
    fn name(&self) -> &str {
        "rrf"
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
        // RRF alone uses BM25 as the single ranking signal.
        // For true RRF, use the fuse() method with multiple sources.
        let bm25 = BM25Reranker::new();
        let mut results = bm25.rerank(query, documents, None).await?;

        if let Some(n) = top_n {
            results.truncate(n);
        }

        Ok(results)
    }
}
