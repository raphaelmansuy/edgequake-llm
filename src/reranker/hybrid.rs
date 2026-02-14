//! Hybrid reranker combining BM25 with vector similarity.
//!
//! Uses RRF to fuse text-based and vector-based rankings.
//!
//! # Architecture
//!
//! ```ascii
//! ┌─────────────┐     ┌─────────────┐
//! │    Query    │     │   Vector    │
//! └──────┬──────┘     │   Search    │
//!        │            └──────┬──────┘
//!        ▼                   │
//! ┌──────────────┐           │
//! │ BM25Reranker │           │ (optional pre-computed)
//! └──────┬───────┘           │
//!        │                   │
//!        ▼                   ▼
//! ┌─────────────────────────────────┐
//! │         RRF Fusion              │
//! │  score = 1/(k+rank_bm25) +      │
//! │          1/(k+rank_vector)      │
//! └─────────────┬───────────────────┘
//!               │
//!               ▼
//!        Fused Results
//! ```

use async_trait::async_trait;

use super::bm25::BM25Reranker;
use super::result::RerankResult;
use super::rrf::RRFReranker;
use super::traits::Reranker;
use crate::error::Result;

/// Hybrid reranker combining BM25 with vector similarity boosting.
///
/// # Example
///
/// ```ignore
/// use edgequake_llm::reranker::HybridReranker;
///
/// let reranker = HybridReranker::new();
///
/// // With both text and vector rankings
/// let vector_rankings = vec![2, 0, 1]; // From vector search
/// let results = reranker.rerank_hybrid(
///     "machine learning",
///     &documents,
///     Some(vector_rankings),
///     Some(10)
/// ).await?;
/// ```
pub struct HybridReranker {
    bm25: BM25Reranker,
    rrf: RRFReranker,
    model: String,
}

impl HybridReranker {
    /// Create a new hybrid reranker.
    pub fn new() -> Self {
        Self {
            bm25: BM25Reranker::new(),
            rrf: RRFReranker::new(),
            model: "hybrid-reranker".to_string(),
        }
    }

    /// Rerank with both text and vector signals.
    ///
    /// # Arguments
    ///
    /// - `query`: Search query for BM25
    /// - `documents`: Document texts
    /// - `vector_rankings`: Pre-sorted indices from vector search (best first)
    /// - `top_n`: Maximum results to return
    pub async fn rerank_hybrid(
        &self,
        query: &str,
        documents: &[String],
        vector_rankings: Option<Vec<usize>>,
        top_n: Option<usize>,
    ) -> Result<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        // Get BM25 ranking
        let bm25_results = self.bm25.rerank(query, documents, None).await?;
        let bm25_ranking: Vec<usize> = bm25_results.iter().map(|r| r.index).collect();

        // Combine with vector ranking if provided
        let mut ranked_lists = vec![bm25_ranking];
        if let Some(vec_ranking) = vector_rankings {
            ranked_lists.push(vec_ranking);
        }

        // Use RRF to fuse rankings
        let mut results = self.rrf.fuse(&ranked_lists, documents.len());

        if let Some(n) = top_n {
            results.truncate(n);
        }

        Ok(results)
    }
}

impl Default for HybridReranker {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Reranker for HybridReranker {
    fn name(&self) -> &str {
        "hybrid"
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
        // Without vector rankings, just use BM25
        self.bm25.rerank(query, documents, top_n).await
    }
}
