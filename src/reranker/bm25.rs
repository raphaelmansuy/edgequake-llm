//! BM25 reranker implementation.
//!
//! Industry-standard BM25 ranking algorithm with BM25+ extension.
//!
//! # Why BM25?
//!
//! BM25 is the ranking algorithm used by Elasticsearch, Lucene, and most
//! production search systems. It provides:
//!
//! - **IDF weighting**: Rare terms score higher (e.g., "ENVY" vs "the")
//! - **Term frequency saturation**: Diminishing returns for repeated terms
//! - **Length normalization**: Long docs don't dominate short focused ones
//!
//! # Algorithm
//!
//! ```ascii
//! Standard BM25:
//! score = Σ IDF(q) × f(q,D)×(k1+1) / (f(q,D) + k1×(1-b+b×|D|/avgdl))
//!
//! BM25+ (when delta > 0):
//! score = Σ IDF(q) × (f(q,D)×(k1+1) / (f(q,D) + k1×(1-b+b×|D|/avgdl)) + delta)
//!
//! Where:
//! - f(q,D) = term frequency of q in document D
//! - |D| = document length
//! - avgdl = average document length
//! - k1 ∈ [1.2, 2.0] = term frequency saturation
//! - b ∈ [0, 1] = length normalization
//! - delta ≥ 0 = BM25+ extension for long docs
//! ```
//!
//! # IDF Formula (SOTA)
//!
//! `IDF(q) = ln((N - n(q) + 0.5) / (n(q) + 0.5) + 1)`
//!
//! The `+1` inside ln() ensures IDF is always non-negative.
//!
//! # References
//!
//! - Robertson, S., Zaragoza, H. (2009). The Probabilistic Relevance Framework
//! - Lv, Y., Zhai, C. (2011). Lower-Bounding Term Frequency Normalization (BM25+)

use async_trait::async_trait;
use rust_stemmers::{Algorithm, Stemmer};
use std::collections::{HashMap, HashSet};
use unicode_normalization::UnicodeNormalization;

use super::result::RerankResult;
use super::traits::Reranker;
use crate::error::Result;

/// BM25 reranker for high-quality text-based reranking.
///
/// # Presets
///
/// | Preset | k1 | b | delta | Use Case |
/// |--------|----|----|-------|----------|
/// | `new()` | 1.5 | 0.75 | 0 | General purpose |
/// | `for_short_docs()` | 1.2 | 0.3 | 0 | Tweets, titles |
/// | `for_long_docs()` | 1.5 | 0.75 | 1.0 | Papers, articles |
/// | `for_technical()` | 2.0 | 0.5 | 0 | Code, APIs |
/// | `for_rag()` | 1.5 | 0.75 | 0.5 | Knowledge graphs |
///
/// # Example
///
/// ```
/// use edgequake_llm::reranker::BM25Reranker;
///
/// // General purpose
/// let reranker = BM25Reranker::new();
///
/// // Optimized for RAG queries
/// let rag_reranker = BM25Reranker::for_rag();
/// ```
pub struct BM25Reranker {
    /// Term frequency saturation parameter (k1).
    /// WHY: Controls how quickly TF saturates. Higher = more TF weight.
    pub k1: f64,
    /// Length normalization parameter (b).
    /// WHY: Controls doc length penalty. 0 = none, 1 = full.
    pub b: f64,
    /// BM25+ delta for long document handling.
    /// WHY: Standard BM25 can over-penalize long docs.
    pub delta: f64,
    /// Phrase boost for adjacent term matching.
    /// WHY: "knowledge graph" should score higher than "graph knowledge".
    pub phrase_boost: f64,
    /// Model name for trait compliance.
    pub model: String,
    /// Tokenizer configuration.
    pub tokenizer_config: TokenizerConfig,
}

/// Configuration for the BM25 tokenizer.
///
/// # Options
///
/// ```ascii
/// ┌────────────────────────────────────────────────────┐
/// │              TokenizerConfig                        │
/// ├────────────────────────────────────────────────────┤
/// │ enable_stemming: bool   ──► Porter2 stemming       │
/// │ stemmer_algorithm       ──► English, French, etc.  │
/// │ enable_stop_words: bool ──► Filter common words    │
/// │ min_token_length: usize ──► Skip short tokens      │
/// └────────────────────────────────────────────────────┘
/// ```
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Enable Porter2 stemming for English text.
    pub enable_stemming: bool,
    /// Stemmer algorithm to use.
    pub stemmer_algorithm: Algorithm,
    /// Enable stop word filtering.
    pub enable_stop_words: bool,
    /// Minimum token length.
    pub min_token_length: usize,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            enable_stemming: true,
            stemmer_algorithm: Algorithm::English,
            enable_stop_words: true,
            min_token_length: 2,
        }
    }
}

impl TokenizerConfig {
    /// Create minimal tokenizer (backward compatible, no stemming).
    pub fn minimal() -> Self {
        Self {
            enable_stemming: false,
            stemmer_algorithm: Algorithm::English,
            enable_stop_words: false,
            min_token_length: 2,
        }
    }

    /// Create enhanced tokenizer with stemming and stop words.
    pub fn enhanced() -> Self {
        Self::default()
    }

    /// Create French tokenizer.
    pub fn french() -> Self {
        Self {
            enable_stemming: true,
            stemmer_algorithm: Algorithm::French,
            enable_stop_words: true,
            min_token_length: 2,
        }
    }
}

/// Common English stop words.
const ENGLISH_STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it",
    "its", "of", "on", "or", "that", "the", "to", "was", "were", "will", "with", "this", "but",
    "they", "have", "had", "what", "when", "where", "who", "which", "you", "your", "we", "our",
    "can", "all", "there", "their", "been", "would", "could", "should", "may", "might", "must",
    "do", "does", "did", "if", "not", "no", "so", "up", "out", "just", "than", "then", "too",
    "very", "also",
];

impl BM25Reranker {
    /// Create with default parameters (k1=1.5, b=0.75, standard BM25).
    pub fn new() -> Self {
        Self {
            k1: 1.5,
            b: 0.75,
            delta: 0.0,
            phrase_boost: 0.0,
            model: "bm25-reranker".to_string(),
            tokenizer_config: TokenizerConfig::minimal(),
        }
    }

    /// Create with enhanced tokenization (stemming + stop words).
    pub fn new_enhanced() -> Self {
        Self {
            k1: 1.5,
            b: 0.75,
            delta: 0.0,
            phrase_boost: 0.0,
            model: "bm25-enhanced-reranker".to_string(),
            tokenizer_config: TokenizerConfig::enhanced(),
        }
    }

    /// Create BM25+ reranker (delta=1.0) for better long doc handling.
    pub fn bm25_plus() -> Self {
        Self {
            k1: 1.5,
            b: 0.75,
            delta: 1.0,
            phrase_boost: 0.0,
            model: "bm25-plus-reranker".to_string(),
            tokenizer_config: TokenizerConfig::minimal(),
        }
    }

    /// Preset for short documents (tweets, titles).
    pub fn for_short_docs() -> Self {
        Self {
            k1: 1.2,
            b: 0.3,
            delta: 0.0,
            phrase_boost: 0.0,
            model: "bm25-short-docs".to_string(),
            tokenizer_config: TokenizerConfig::enhanced(),
        }
    }

    /// Preset for long documents (papers, articles).
    pub fn for_long_docs() -> Self {
        Self {
            k1: 1.5,
            b: 0.75,
            delta: 1.0,
            phrase_boost: 0.0,
            model: "bm25-long-docs".to_string(),
            tokenizer_config: TokenizerConfig::enhanced(),
        }
    }

    /// Preset for technical content (code, APIs).
    pub fn for_technical() -> Self {
        Self {
            k1: 2.0,
            b: 0.5,
            delta: 0.0,
            phrase_boost: 0.0,
            model: "bm25-technical".to_string(),
            tokenizer_config: TokenizerConfig::minimal(),
        }
    }

    /// Preset for RAG/knowledge graph queries.
    pub fn for_rag() -> Self {
        Self {
            k1: 1.5,
            b: 0.75,
            delta: 0.5,
            phrase_boost: 0.3,
            model: "bm25-rag".to_string(),
            tokenizer_config: TokenizerConfig::enhanced(),
        }
    }

    /// Preset for semantic queries with phrase boosting.
    pub fn for_semantic() -> Self {
        Self {
            k1: 1.5,
            b: 0.75,
            delta: 0.5,
            phrase_boost: 0.5,
            model: "bm25-semantic".to_string(),
            tokenizer_config: TokenizerConfig::enhanced(),
        }
    }

    /// Create with custom k1 and b parameters.
    pub fn with_params(k1: f64, b: f64) -> Self {
        Self {
            k1: k1.clamp(0.0, 3.0),
            b: b.clamp(0.0, 1.0),
            delta: 0.0,
            phrase_boost: 0.0,
            model: "bm25-reranker".to_string(),
            tokenizer_config: TokenizerConfig::minimal(),
        }
    }

    /// Create with full custom parameters including delta.
    pub fn with_full_params(k1: f64, b: f64, delta: f64) -> Self {
        Self {
            k1: k1.clamp(0.0, 3.0),
            b: b.clamp(0.0, 1.0),
            delta: delta.max(0.0),
            phrase_boost: 0.0,
            model: if delta > 0.0 {
                "bm25-plus-reranker".to_string()
            } else {
                "bm25-reranker".to_string()
            },
            tokenizer_config: TokenizerConfig::minimal(),
        }
    }

    /// Set custom tokenizer configuration.
    pub fn with_tokenizer_config(mut self, config: TokenizerConfig) -> Self {
        self.tokenizer_config = config;
        self
    }

    /// Set phrase boost factor (0.0-2.0).
    pub fn with_phrase_boost(mut self, boost: f64) -> Self {
        self.phrase_boost = boost.clamp(0.0, 2.0);
        self
    }

    /// Check if a word is a stop word.
    fn is_stop_word(word: &str) -> bool {
        ENGLISH_STOP_WORDS.binary_search(&word).is_ok()
    }

    /// Tokenize with configured settings (stemming, stop words).
    pub(crate) fn tokenize_with_config(&self, text: &str) -> Vec<String> {
        let normalized: String = text
            .to_lowercase()
            .nfkd()
            .filter(|c| !unicode_normalization::char::is_combining_mark(*c))
            .collect();

        let tokens: Vec<String> = normalized
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() >= self.tokenizer_config.min_token_length)
            .map(|s| s.to_string())
            .collect();

        let filtered: Vec<String> = if self.tokenizer_config.enable_stop_words {
            tokens
                .into_iter()
                .filter(|t| !Self::is_stop_word(t))
                .collect()
        } else {
            tokens
        };

        if self.tokenizer_config.enable_stemming {
            let stemmer = Stemmer::create(self.tokenizer_config.stemmer_algorithm);
            filtered
                .into_iter()
                .map(|t| stemmer.stem(&t).to_string())
                .collect()
        } else {
            filtered
        }
    }

    /// Basic tokenization (backward compatible).
    pub(crate) fn tokenize(text: &str) -> Vec<String> {
        let normalized: String = text
            .to_lowercase()
            .nfkd()
            .filter(|c| !unicode_normalization::char::is_combining_mark(*c))
            .collect();

        normalized
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() > 1)
            .map(|s| s.to_string())
            .collect()
    }

    /// Compute IDF for a term across documents.
    #[allow(dead_code)]
    pub(crate) fn compute_idf(term: &str, doc_terms_list: &[Vec<String>]) -> f64 {
        let n = doc_terms_list.len() as f64;
        let containing_docs = doc_terms_list
            .iter()
            .filter(|terms| terms.contains(&term.to_string()))
            .count() as f64;

        Self::compute_idf_from_df(n, containing_docs)
    }

    /// Compute IDF from pre-computed document frequency.
    #[inline]
    pub(crate) fn compute_idf_from_df(n: f64, df: f64) -> f64 {
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Build document frequency map for all terms.
    pub(crate) fn compute_document_frequencies(
        doc_terms_list: &[Vec<String>],
    ) -> HashMap<String, usize> {
        let mut df_map: HashMap<String, usize> = HashMap::new();

        for doc_terms in doc_terms_list {
            let unique_terms: HashSet<&String> = doc_terms.iter().collect();
            for term in unique_terms {
                *df_map.entry(term.clone()).or_insert(0) += 1;
            }
        }

        df_map
    }

    /// Compute phrase match bonus for adjacent query terms.
    pub(crate) fn compute_phrase_bonus(&self, query_terms: &[String], doc_terms: &[String]) -> f64 {
        if query_terms.len() < 2 || doc_terms.len() < 2 {
            return 0.0;
        }

        let mut phrase_matches = 0;
        let total_pairs = query_terms.len().saturating_sub(1);

        for window in query_terms.windows(2) {
            let (term_a, term_b) = (&window[0], &window[1]);
            for doc_window in doc_terms.windows(2) {
                if &doc_window[0] == term_a && &doc_window[1] == term_b {
                    phrase_matches += 1;
                    break;
                }
            }
        }

        phrase_matches as f64 / total_pairs.max(1) as f64
    }

    /// Compute BM25/BM25+ score for a single document.
    fn compute_bm25_score(
        &self,
        query_terms: &[String],
        doc_terms: &[String],
        avgdl: f64,
        idf_cache: &HashMap<String, f64>,
    ) -> f64 {
        let doc_len = doc_terms.len() as f64;
        let length_norm = 1.0 - self.b + self.b * (doc_len / avgdl);

        let mut score = 0.0;
        for term in query_terms {
            let tf = doc_terms.iter().filter(|t| t == &term).count() as f64;
            if tf > 0.0 {
                let idf = idf_cache.get(term).copied().unwrap_or(0.0);
                let tf_component = (tf * (self.k1 + 1.0)) / (tf + self.k1 * length_norm);
                score += idf * (tf_component + self.delta);
            }
        }
        score
    }
}

impl Default for BM25Reranker {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Reranker for BM25Reranker {
    fn name(&self) -> &str {
        "bm25"
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
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let query_terms =
            if self.tokenizer_config.enable_stemming || self.tokenizer_config.enable_stop_words {
                self.tokenize_with_config(query)
            } else {
                Self::tokenize(query)
            };

        if query_terms.is_empty() {
            let results: Vec<RerankResult> = documents
                .iter()
                .enumerate()
                .map(|(idx, _)| RerankResult {
                    index: idx,
                    relevance_score: 0.0,
                })
                .collect();
            return Ok(results);
        }

        let doc_terms_list: Vec<Vec<String>> =
            if self.tokenizer_config.enable_stemming || self.tokenizer_config.enable_stop_words {
                documents
                    .iter()
                    .map(|d| self.tokenize_with_config(d))
                    .collect()
            } else {
                documents.iter().map(|d| Self::tokenize(d)).collect()
            };

        let avgdl = doc_terms_list.iter().map(|d| d.len()).sum::<usize>() as f64
            / doc_terms_list.len().max(1) as f64;
        let avgdl = avgdl.max(1.0);

        let df_map = Self::compute_document_frequencies(&doc_terms_list);
        let n = doc_terms_list.len() as f64;

        let mut idf_cache = HashMap::new();
        for term in &query_terms {
            let df = df_map.get(term).copied().unwrap_or(0) as f64;
            let idf = Self::compute_idf_from_df(n, df);
            idf_cache.insert(term.clone(), idf);
        }

        let mut results: Vec<RerankResult> = doc_terms_list
            .iter()
            .enumerate()
            .map(|(idx, doc_terms)| {
                let bm25_score =
                    self.compute_bm25_score(&query_terms, doc_terms, avgdl, &idf_cache);

                let phrase_bonus = if self.phrase_boost > 0.0 {
                    self.compute_phrase_bonus(&query_terms, doc_terms)
                } else {
                    0.0
                };

                let final_score = bm25_score + (self.phrase_boost * phrase_bonus);
                RerankResult {
                    index: idx,
                    relevance_score: final_score,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(n) = top_n {
            results.truncate(n);
        }

        Ok(results)
    }
}
