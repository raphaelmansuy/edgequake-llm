//! HTTP-based reranker implementation.
//!
//! This module provides reranking via HTTP APIs (Jina, Cohere, Aliyun).
//!
//! # Supported Providers
//!
//! | Provider | API | Features |
//! |----------|-----|----------|
//! | Jina AI | REST | Multilingual, fast |
//! | Cohere | REST | rerank-v3.5, 4K context |
//! | Aliyun | REST | DashScope gte-rerank-v2 |

use async_trait::async_trait;
use reqwest::Client;
use std::collections::HashMap;
use tracing::{debug, warn};

use super::config::{RerankConfig, ScoreAggregation};
use super::result::RerankResult;
use super::traits::Reranker;
use crate::error::{LlmError, Result};

/// HTTP-based reranker that supports multiple providers.
///
/// # Architecture
///
/// ```ascii
/// ┌────────────────┐    HTTP     ┌─────────────────┐
/// │  HttpReranker  │ ─────────►  │  Provider API   │
/// │                │             │  (Jina/Cohere)  │
/// └────────┬───────┘             └────────┬────────┘
///          │                              │
///          │   RerankConfig               │  JSON Response
///          │   - model                    │  - results[]
///          │   - base_url                 │    - index
///          │   - api_key                  │    - relevance_score
///          └──────────────────────────────┘
/// ```
pub struct HttpReranker {
    client: Client,
    config: RerankConfig,
    /// Response format for parsing results.
    response_format: ResponseFormat,
    /// Request format for building payloads.
    request_format: RequestFormat,
}

#[derive(Debug, Clone, Copy)]
enum ResponseFormat {
    /// Standard format: {"results": [{"index": 0, "relevance_score": 0.9}]}
    Standard,
    /// Aliyun format: {"output": {"results": [...]}}
    Aliyun,
}

#[derive(Debug, Clone, Copy)]
enum RequestFormat {
    /// Standard format: {"query": "...", "documents": [...]}
    Standard,
    /// Aliyun format: {"input": {"query": "...", "documents": [...]}}
    Aliyun,
}

impl HttpReranker {
    /// Create a new HTTP reranker with the given config.
    pub fn new(config: RerankConfig) -> Self {
        let (response_format, request_format) = Self::detect_format(&config.base_url);

        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            config,
            response_format,
            request_format,
        }
    }

    /// Create a Jina reranker.
    pub fn jina(api_key: impl Into<String>) -> Self {
        Self::new(RerankConfig::jina(api_key))
    }

    /// Create a Cohere reranker.
    pub fn cohere(api_key: impl Into<String>) -> Self {
        Self::new(RerankConfig::cohere(api_key))
    }

    /// Create an Aliyun reranker.
    pub fn aliyun(api_key: impl Into<String>) -> Self {
        let config = RerankConfig::aliyun(api_key);
        Self {
            client: Client::builder()
                .timeout(config.timeout)
                .build()
                .expect("Failed to build HTTP client"),
            config,
            response_format: ResponseFormat::Aliyun,
            request_format: RequestFormat::Aliyun,
        }
    }

    fn detect_format(base_url: &str) -> (ResponseFormat, RequestFormat) {
        if base_url.contains("dashscope.aliyuncs.com") {
            (ResponseFormat::Aliyun, RequestFormat::Aliyun)
        } else {
            (ResponseFormat::Standard, RequestFormat::Standard)
        }
    }

    fn build_request(
        &self,
        query: &str,
        documents: &[String],
        top_n: Option<usize>,
    ) -> serde_json::Value {
        match self.request_format {
            RequestFormat::Standard => {
                let mut payload = serde_json::json!({
                    "model": self.config.model,
                    "query": query,
                    "documents": documents,
                });
                if let Some(n) = top_n {
                    payload["top_n"] = serde_json::json!(n);
                }
                payload
            }
            RequestFormat::Aliyun => {
                let mut params = serde_json::Map::new();
                if let Some(n) = top_n {
                    params.insert("top_n".to_string(), serde_json::json!(n));
                }
                serde_json::json!({
                    "model": self.config.model,
                    "input": {
                        "query": query,
                        "documents": documents,
                    },
                    "parameters": params,
                })
            }
        }
    }

    fn parse_response(&self, response: serde_json::Value) -> Result<Vec<RerankResult>> {
        let results = match self.response_format {
            ResponseFormat::Standard => response
                .get("results")
                .and_then(|r| r.as_array())
                .cloned()
                .unwrap_or_default(),
            ResponseFormat::Aliyun => response
                .get("output")
                .and_then(|o| o.get("results"))
                .and_then(|r| r.as_array())
                .cloned()
                .unwrap_or_default(),
        };

        if results.is_empty() {
            warn!("Rerank API returned empty results");
            return Ok(vec![]);
        }

        let mut rerank_results = Vec::with_capacity(results.len());
        for result in results {
            let index = result
                .get("index")
                .and_then(|i| i.as_u64())
                .ok_or_else(|| LlmError::Unknown("Missing index in rerank result".to_string()))?
                as usize;
            let score = result
                .get("relevance_score")
                .and_then(|s| s.as_f64())
                .ok_or_else(|| {
                    LlmError::Unknown("Missing relevance_score in rerank result".to_string())
                })?;

            rerank_results.push(RerankResult {
                index,
                relevance_score: score,
            });
        }

        // Sort by relevance score descending
        rerank_results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(rerank_results)
    }

    /// Chunk documents that exceed the token limit.
    pub fn chunk_documents(&self, documents: &[String]) -> (Vec<String>, Vec<usize>) {
        if !self.config.enable_chunking {
            let indices: Vec<usize> = (0..documents.len()).collect();
            return (documents.to_vec(), indices);
        }

        let max_chars = self.config.max_tokens_per_doc * 4; // Approximate 1 token ≈ 4 chars
        let overlap_chars = 32 * 4; // 32 tokens overlap

        let mut chunked = Vec::new();
        let mut indices = Vec::new();

        for (idx, doc) in documents.iter().enumerate() {
            if doc.len() <= max_chars {
                chunked.push(doc.clone());
                indices.push(idx);
            } else {
                // Split into overlapping chunks
                let mut start = 0;
                while start < doc.len() {
                    let end = (start + max_chars).min(doc.len());
                    let chunk = doc[start..end].to_string();
                    chunked.push(chunk);
                    indices.push(idx);

                    if end >= doc.len() {
                        break;
                    }
                    start = end.saturating_sub(overlap_chars);
                }
            }
        }

        debug!(
            "Chunked {} documents into {} chunks",
            documents.len(),
            chunked.len()
        );
        (chunked, indices)
    }

    /// Aggregate chunk scores back to original documents.
    pub fn aggregate_scores(
        &self,
        chunk_results: Vec<RerankResult>,
        doc_indices: &[usize],
        num_docs: usize,
        aggregation: ScoreAggregation,
    ) -> Vec<RerankResult> {
        let mut doc_scores: HashMap<usize, Vec<f64>> = HashMap::new();
        for i in 0..num_docs {
            doc_scores.insert(i, Vec::new());
        }

        for result in chunk_results {
            if result.index < doc_indices.len() {
                let original_idx = doc_indices[result.index];
                if let Some(scores) = doc_scores.get_mut(&original_idx) {
                    scores.push(result.relevance_score);
                }
            }
        }

        let mut aggregated: Vec<RerankResult> = doc_scores
            .into_iter()
            .filter(|(_, scores)| !scores.is_empty())
            .map(|(idx, scores)| {
                let final_score = match aggregation {
                    ScoreAggregation::Max => {
                        scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                    }
                    ScoreAggregation::Mean => scores.iter().sum::<f64>() / scores.len() as f64,
                    ScoreAggregation::First => scores[0],
                };
                RerankResult {
                    index: idx,
                    relevance_score: final_score,
                }
            })
            .collect();

        aggregated.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        aggregated
    }
}

#[async_trait]
impl Reranker for HttpReranker {
    fn name(&self) -> &str {
        if self.config.base_url.contains("jina.ai") {
            "jina"
        } else if self.config.base_url.contains("cohere.com") {
            "cohere"
        } else if self.config.base_url.contains("aliyuncs.com") {
            "aliyun"
        } else {
            "http"
        }
    }

    fn model(&self) -> &str {
        &self.config.model
    }

    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_n: Option<usize>,
    ) -> Result<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        // Handle chunking
        let (chunked_docs, doc_indices) = self.chunk_documents(documents);
        let original_top_n = top_n;

        // When chunking, disable top_n at API level to get all chunk scores
        let api_top_n = if self.config.enable_chunking {
            None
        } else {
            top_n
        };

        let payload = self.build_request(query, &chunked_docs, api_top_n);

        debug!(
            "Rerank request: {} documents, model: {}",
            chunked_docs.len(),
            self.config.model
        );

        let mut request = self
            .client
            .post(&self.config.base_url)
            .header("Content-Type", "application/json");

        if let Some(ref api_key) = self.config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = request
            .json(&payload)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "Rerank API error ({}): {}",
                status.as_u16(),
                error_text
            )));
        }

        let response_json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::Unknown(format!("Failed to parse rerank response: {}", e)))?;

        let mut results = self.parse_response(response_json)?;

        // Aggregate chunk scores if chunking was enabled
        if self.config.enable_chunking && chunked_docs.len() != documents.len() {
            results = self.aggregate_scores(
                results,
                &doc_indices,
                documents.len(),
                ScoreAggregation::Max,
            );
        }

        // Apply top_n limit at document level
        if let Some(n) = original_top_n {
            results.truncate(n);
        }

        Ok(results)
    }
}
