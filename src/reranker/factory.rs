//! Production reranker factory (SPEC-024 / EdgeQuake query bootstrap SSOT).
//!
//! Resolves rerankers from environment variables so API and SDK share identical
//! reranking behavior.

use std::sync::Arc;

use tracing::{info, warn};

use crate::traits::EmbeddingProvider;

use super::bi_encoder::BiEncoderReranker;
use super::bm25::BM25Reranker;
use super::http::HttpReranker;
use super::traits::Reranker;

/// Create BM25 reranker (enhanced by default; set `BM25_ENHANCED=false` for minimal).
pub fn create_bm25_reranker() -> Arc<dyn Reranker> {
    if std::env::var("BM25_ENHANCED").unwrap_or_default() == "false" {
        info!("Using minimal BM25 reranker (BM25_ENHANCED=false)");
        Arc::new(BM25Reranker::new())
    } else {
        info!("Using enhanced BM25 reranker with stemming and Unicode normalization");
        Arc::new(BM25Reranker::new_enhanced())
    }
}

/// Try to build an HTTP cross-encoder reranker from env API keys.
///
/// Honors `EDGEQUAKE_RERANKER_PROVIDER` (`jina`, `cohere`, `aliyun`) or auto-detects
/// from `JINA_API_KEY`, `COHERE_API_KEY`, `DASHSCOPE_API_KEY`, `ALIYUN_API_KEY`.
pub fn try_http_cross_encoder_reranker() -> Option<Arc<dyn Reranker>> {
    let provider = std::env::var("EDGEQUAKE_RERANKER_PROVIDER")
        .unwrap_or_default()
        .to_ascii_lowercase();

    match provider.as_str() {
        "jina" => env_api_key("JINA_API_KEY").map(|k| {
            info!("Using Jina cross-encoder reranker");
            Arc::new(HttpReranker::jina(k)) as Arc<dyn Reranker>
        }),
        "cohere" => env_api_key("COHERE_API_KEY").map(|k| {
            info!("Using Cohere cross-encoder reranker");
            Arc::new(HttpReranker::cohere(k)) as Arc<dyn Reranker>
        }),
        "aliyun" => env_api_key("DASHSCOPE_API_KEY")
            .or_else(|| env_api_key("ALIYUN_API_KEY"))
            .map(|k| {
                info!("Using Aliyun cross-encoder reranker");
                Arc::new(HttpReranker::aliyun(k)) as Arc<dyn Reranker>
            }),
        _ => {
            if let Some(key) = env_api_key("JINA_API_KEY") {
                info!("Using Jina cross-encoder reranker (auto-detected JINA_API_KEY)");
                return Some(Arc::new(HttpReranker::jina(key)));
            }
            if let Some(key) = env_api_key("COHERE_API_KEY") {
                info!("Using Cohere cross-encoder reranker (auto-detected COHERE_API_KEY)");
                return Some(Arc::new(HttpReranker::cohere(key)));
            }
            if let Some(key) =
                env_api_key("DASHSCOPE_API_KEY").or_else(|| env_api_key("ALIYUN_API_KEY"))
            {
                info!("Using Aliyun cross-encoder reranker (auto-detected API key)");
                return Some(Arc::new(HttpReranker::aliyun(key)));
            }
            None
        }
    }
}

/// Cross-encoder chain: HTTP API → bi-encoder embedding fallback → BM25.
pub fn create_cross_encoder_reranker(
    embedding: Option<Arc<dyn EmbeddingProvider>>,
) -> Arc<dyn Reranker> {
    if let Some(reranker) = try_http_cross_encoder_reranker() {
        return reranker;
    }

    if let Some(provider) = embedding {
        info!("Using bi-encoder reranker (embedding cosine similarity)");
        return Arc::new(BiEncoderReranker::new(provider));
    }

    warn!("EDGEQUAKE_RERANKER=cross_encoder but no HTTP API key or embedding provider; using BM25");
    create_bm25_reranker()
}

/// Production reranker from `EDGEQUAKE_RERANKER` (default: BM25).
///
/// Set `EDGEQUAKE_RERANKER=cross_encoder` for neural reranking. Pass `embedding` to
/// enable bi-encoder fallback when no reranker API key is configured.
pub fn create_production_reranker(
    embedding: Option<Arc<dyn EmbeddingProvider>>,
) -> Arc<dyn Reranker> {
    match std::env::var("EDGEQUAKE_RERANKER")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "cross_encoder" => create_cross_encoder_reranker(embedding),
        _ => create_bm25_reranker(),
    }
}

fn env_api_key(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}
