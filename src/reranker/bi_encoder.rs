//! Bi-encoder reranker using embedding cosine similarity.
//!
//! Fallback when HTTP cross-encoders (Jina/Cohere/Aliyun) are unavailable but an
//! [`EmbeddingProvider`] is configured.

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::{LlmError, Result};
use crate::traits::EmbeddingProvider;

use super::result::RerankResult;
use super::traits::Reranker;

/// Rerank documents by cosine similarity between query and document embeddings.
pub struct BiEncoderReranker {
    embedding: Arc<dyn EmbeddingProvider>,
}

impl BiEncoderReranker {
    pub fn new(embedding: Arc<dyn EmbeddingProvider>) -> Self {
        Self { embedding }
    }
}

#[async_trait]
impl Reranker for BiEncoderReranker {
    fn name(&self) -> &str {
        "bi-encoder"
    }

    fn model(&self) -> &str {
        "embedding-cosine"
    }

    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_n: Option<usize>,
    ) -> Result<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let mut texts = Vec::with_capacity(documents.len() + 1);
        texts.push(query.to_string());
        texts.extend(documents.iter().cloned());

        let embeddings = self.embedding.embed(&texts).await?;
        let query_emb = embeddings.first().ok_or_else(|| {
            LlmError::ProviderError("empty embedding response from bi-encoder reranker".into())
        })?;

        let mut scored: Vec<(usize, f64)> = documents
            .iter()
            .enumerate()
            .filter_map(|(idx, _)| {
                embeddings.get(idx + 1).map(|doc_emb| {
                    let score = cosine_similarity(query_emb, doc_emb) as f64;
                    (idx, score)
                })
            })
            .collect();

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let limit = top_n.unwrap_or(documents.len());
        Ok(scored
            .into_iter()
            .take(limit)
            .map(|(index, relevance_score)| RerankResult {
                index,
                relevance_score,
            })
            .collect())
    }
}

pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a <= 0.0 || norm_b <= 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors_score_one() {
        let v = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors_score_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }
}
