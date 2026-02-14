//! Reranker configuration types.
//!
//! This module contains configuration structures for reranking providers.
//!
//! # Architecture
//!
//! ```ascii
//! ┌─────────────────────────────────────────────────────────┐
//! │                    RerankConfig                          │
//! ├─────────────────────────────────────────────────────────┤
//! │ model: String         ─────► Which model to use         │
//! │ base_url: String      ─────► API endpoint               │
//! │ api_key: Option       ─────► Authentication             │
//! │ top_n: Option<usize>  ─────► Max results to return      │
//! │ timeout: Duration     ─────► Request timeout            │
//! │ enable_chunking: bool ─────► Split long docs?           │
//! │ max_tokens_per_doc    ─────► Chunk size limit           │
//! └─────────────────────────────────────────────────────────┘
//! ```

use std::time::Duration;

/// Configuration for a reranker.
///
/// # Provider-Specific Configurations
///
/// Use the factory methods for common providers:
/// - [`RerankConfig::jina`] - Jina AI Reranker
/// - [`RerankConfig::cohere`] - Cohere Rerank
/// - [`RerankConfig::aliyun`] - Aliyun DashScope
///
/// # Example
///
/// ```ignore
/// // Jina reranker with top 10 results
/// let config = RerankConfig::jina("your-api-key")
///     .with_top_n(10)
///     .with_chunking(true);
/// ```
#[derive(Debug, Clone)]
pub struct RerankConfig {
    /// Model name to use.
    pub model: String,
    /// Base URL for the reranker API.
    pub base_url: String,
    /// API key for authentication.
    pub api_key: Option<String>,
    /// Maximum number of results to return.
    pub top_n: Option<usize>,
    /// Request timeout.
    pub timeout: Duration,
    /// Enable document chunking for long documents.
    pub enable_chunking: bool,
    /// Maximum tokens per document for chunking.
    pub max_tokens_per_doc: usize,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            model: "jina-reranker-v2-base-multilingual".to_string(),
            base_url: "https://api.jina.ai/v1/rerank".to_string(),
            api_key: None,
            top_n: None,
            timeout: Duration::from_secs(30),
            enable_chunking: false,
            max_tokens_per_doc: 480,
        }
    }
}

impl RerankConfig {
    /// Create a new Jina reranker config.
    ///
    /// Uses the `jina-reranker-v2-base-multilingual` model.
    pub fn jina(api_key: impl Into<String>) -> Self {
        Self {
            model: "jina-reranker-v2-base-multilingual".to_string(),
            base_url: "https://api.jina.ai/v1/rerank".to_string(),
            api_key: Some(api_key.into()),
            ..Default::default()
        }
    }

    /// Create a new Cohere reranker config.
    ///
    /// Uses the `rerank-v3.5` model with 4096 max tokens.
    pub fn cohere(api_key: impl Into<String>) -> Self {
        Self {
            model: "rerank-v3.5".to_string(),
            base_url: "https://api.cohere.com/v2/rerank".to_string(),
            api_key: Some(api_key.into()),
            max_tokens_per_doc: 4096,
            ..Default::default()
        }
    }

    /// Create a new Aliyun DashScope reranker config.
    ///
    /// Uses the `gte-rerank-v2` model.
    pub fn aliyun(api_key: impl Into<String>) -> Self {
        Self {
            model: "gte-rerank-v2".to_string(),
            base_url:
                "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
                    .to_string(),
            api_key: Some(api_key.into()),
            ..Default::default()
        }
    }

    /// Set the model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the top N results to return.
    pub fn with_top_n(mut self, top_n: usize) -> Self {
        self.top_n = Some(top_n);
        self
    }

    /// Enable document chunking.
    pub fn with_chunking(mut self, enable: bool) -> Self {
        self.enable_chunking = enable;
        self
    }

    /// Set max tokens per document for chunking.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens_per_doc = max_tokens;
        self
    }
}

/// Strategy for aggregating chunk scores.
///
/// When documents are split into chunks for processing, this determines
/// how the individual chunk scores are combined into a final document score.
///
/// # Variants
///
/// - `Max`: Use the highest score from any chunk (default, most common)
/// - `Mean`: Average all chunk scores
/// - `First`: Use only the first chunk's score
#[derive(Debug, Clone, Copy, Default)]
pub enum ScoreAggregation {
    /// Use the maximum score from all chunks.
    #[default]
    Max,
    /// Use the mean of all chunk scores.
    Mean,
    /// Use the score from the first chunk.
    First,
}
