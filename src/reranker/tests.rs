//! Reranker tests.
//!
//! Comprehensive test suite for all reranker implementations.

use super::*;

#[test]
fn test_rerank_config_defaults() {
    let config = RerankConfig::default();
    assert_eq!(config.model, "jina-reranker-v2-base-multilingual");
    assert!(config.api_key.is_none());
}

#[test]
fn test_jina_config() {
    let config = RerankConfig::jina("test-key");
    assert_eq!(config.api_key, Some("test-key".to_string()));
    assert!(config.base_url.contains("jina.ai"));
}

#[test]
fn test_cohere_config() {
    let config = RerankConfig::cohere("test-key");
    assert!(config.base_url.contains("cohere.com"));
    assert_eq!(config.max_tokens_per_doc, 4096);
}

#[test]
fn test_aliyun_config() {
    let config = RerankConfig::aliyun("test-key");
    assert!(config.base_url.contains("aliyuncs.com"));
}

#[tokio::test]
async fn test_mock_reranker() {
    let reranker = MockReranker::new();
    let query = "capital of France";
    let documents = vec![
        "The capital of France is Paris.".to_string(),
        "Tokyo is the capital of Japan.".to_string(),
        "London is the capital of England.".to_string(),
    ];

    let results = reranker.rerank(query, &documents, Some(2)).await.unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].index, 0);
    assert!(results[0].relevance_score > 0.0);
}

#[tokio::test]
async fn test_mock_reranker_empty_docs() {
    let reranker = MockReranker::new();
    let results = reranker.rerank("test", &[], None).await.unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_score_aggregation() {
    let reranker = HttpReranker::new(RerankConfig::default());

    let chunk_results = vec![
        RerankResult {
            index: 0,
            relevance_score: 0.9,
        },
        RerankResult {
            index: 1,
            relevance_score: 0.8,
        },
        RerankResult {
            index: 2,
            relevance_score: 0.7,
        },
    ];
    let doc_indices = vec![0, 0, 1];

    let aggregated =
        reranker.aggregate_scores(chunk_results, &doc_indices, 2, ScoreAggregation::Max);

    assert_eq!(aggregated.len(), 2);
    assert_eq!(aggregated[0].index, 0);
    assert!((aggregated[0].relevance_score - 0.9).abs() < 0.001);
}

#[test]
fn test_chunking() {
    let config = RerankConfig::default()
        .with_chunking(true)
        .with_max_tokens(50);
    let reranker = HttpReranker::new(config);

    let documents = vec!["Short document.".to_string(), "A".repeat(1000)];

    let (chunked, indices) = reranker.chunk_documents(&documents);

    assert!(chunked.len() > 2);
    assert!(indices.iter().all(|&i| i < 2));
}

// =========== BM25 Reranker Tests ===========

#[tokio::test]
async fn test_bm25_reranker_basic() {
    let reranker = BM25Reranker::new();
    let query = "capital of France";
    let documents = vec![
        "The capital of France is Paris.".to_string(),
        "Tokyo is the capital of Japan.".to_string(),
        "London is the capital of England.".to_string(),
    ];

    let results = reranker.rerank(query, &documents, None).await.unwrap();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].index, 0);
    assert!(results[0].relevance_score > results[1].relevance_score);
}

#[tokio::test]
async fn test_bm25_idf_weighting() {
    let reranker = BM25Reranker::new();
    let query = "Peugeot ENVY";
    let documents = vec![
        "The Peugeot 2008 ENVY is a great car.".to_string(),
        "Peugeot makes many cars.".to_string(),
        "Peugeot 208 is also available.".to_string(),
    ];

    let results = reranker.rerank(query, &documents, None).await.unwrap();

    assert_eq!(results[0].index, 0);
    assert!(results[0].relevance_score > results[1].relevance_score * 1.5);
}

#[tokio::test]
async fn test_bm25_2008_vs_208_precision() {
    let reranker = BM25Reranker::new();
    let query = "2008";
    let documents = vec![
        "The Peugeot 208 is a compact car.".to_string(),
        "The Peugeot 2008 is an SUV.".to_string(),
        "The Peugeot 3008 is a larger SUV.".to_string(),
    ];

    let results = reranker.rerank(query, &documents, None).await.unwrap();

    assert_eq!(results[0].index, 1, "2008 document should be first");
    assert_ne!(results[0].index, 0, "208 document should NOT be first");
}

#[tokio::test]
async fn test_bm25_french_accents() {
    let reranker = BM25Reranker::new();
    let query = "vehicule electrique";
    let documents = vec![
        "Le véhicule électrique est l'avenir.".to_string(),
        "Une voiture classique fonctionne à essence.".to_string(),
    ];

    let results = reranker.rerank(query, &documents, None).await.unwrap();

    assert_eq!(results[0].index, 0);
    assert!(results[0].relevance_score > 0.0);
}

#[tokio::test]
async fn test_bm25_empty_documents() {
    let reranker = BM25Reranker::new();
    let results = reranker.rerank("test", &[], None).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_bm25_empty_query() {
    let reranker = BM25Reranker::new();
    let documents = vec!["Some document.".to_string()];
    let results = reranker.rerank("", &documents, None).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].relevance_score, 0.0);
}

#[tokio::test]
async fn test_bm25_top_n() {
    let reranker = BM25Reranker::new();
    let documents = vec![
        "Alpha document.".to_string(),
        "Beta document.".to_string(),
        "Gamma document.".to_string(),
    ];

    let results = reranker
        .rerank("document", &documents, Some(2))
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_bm25_tokenization() {
    let tokens = BM25Reranker::tokenize("Hello, World! Test-123 véhicule");
    assert!(tokens.contains(&"hello".to_string()));
    assert!(tokens.contains(&"world".to_string()));
    assert!(tokens.contains(&"test".to_string()));
    assert!(tokens.contains(&"123".to_string()));
    assert!(tokens.contains(&"vehicule".to_string()));
}

#[test]
fn test_bm25_custom_params() {
    let reranker = BM25Reranker::with_params(2.0, 0.5);
    assert_eq!(reranker.k1, 2.0);
    assert_eq!(reranker.b, 0.5);
}

// =========== BM25+ Extension Tests ===========

#[test]
fn test_bm25_plus_constructor() {
    let reranker = BM25Reranker::bm25_plus();
    assert_eq!(reranker.k1, 1.5);
    assert_eq!(reranker.b, 0.75);
    assert_eq!(reranker.delta, 1.0);
    assert_eq!(reranker.model(), "bm25-plus-reranker");
}

#[test]
fn test_bm25_with_full_params() {
    let reranker = BM25Reranker::with_full_params(1.2, 0.8, 0.5);
    assert_eq!(reranker.k1, 1.2);
    assert_eq!(reranker.b, 0.8);
    assert_eq!(reranker.delta, 0.5);
}

// =========== Domain-Specific Preset Tests ===========

#[test]
fn test_for_short_docs_preset() {
    let reranker = BM25Reranker::for_short_docs();
    assert_eq!(reranker.k1, 1.2);
    assert_eq!(reranker.b, 0.3);
    assert_eq!(reranker.delta, 0.0);
    assert_eq!(reranker.model(), "bm25-short-docs");
    assert!(reranker.tokenizer_config.enable_stemming);
}

#[test]
fn test_for_long_docs_preset() {
    let reranker = BM25Reranker::for_long_docs();
    assert_eq!(reranker.k1, 1.5);
    assert_eq!(reranker.b, 0.75);
    assert_eq!(reranker.delta, 1.0);
    assert_eq!(reranker.model(), "bm25-long-docs");
    assert!(reranker.tokenizer_config.enable_stemming);
}

#[test]
fn test_for_technical_preset() {
    let reranker = BM25Reranker::for_technical();
    assert_eq!(reranker.k1, 2.0);
    assert_eq!(reranker.b, 0.5);
    assert_eq!(reranker.delta, 0.0);
    assert_eq!(reranker.model(), "bm25-technical");
    assert!(!reranker.tokenizer_config.enable_stemming);
}

#[test]
fn test_for_rag_preset() {
    let reranker = BM25Reranker::for_rag();
    assert_eq!(reranker.k1, 1.5);
    assert_eq!(reranker.b, 0.75);
    assert_eq!(reranker.delta, 0.5);
    assert_eq!(reranker.model(), "bm25-rag");
    assert!(reranker.tokenizer_config.enable_stemming);
    assert_eq!(reranker.phrase_boost, 0.3);
}

// =========== Phrase Boosting Tests ===========

#[test]
fn test_for_semantic_preset() {
    let reranker = BM25Reranker::for_semantic();
    assert_eq!(reranker.k1, 1.5);
    assert_eq!(reranker.b, 0.75);
    assert_eq!(reranker.phrase_boost, 0.5);
    assert_eq!(reranker.model(), "bm25-semantic");
}

#[test]
fn test_with_phrase_boost_builder() {
    let reranker = BM25Reranker::new().with_phrase_boost(0.7);
    assert_eq!(reranker.phrase_boost, 0.7);

    let clamped = BM25Reranker::new().with_phrase_boost(5.0);
    assert_eq!(clamped.phrase_boost, 2.0);
}

#[test]
fn test_phrase_bonus_calculation() {
    let reranker = BM25Reranker::new();

    let query = vec!["knowledge".to_string(), "graph".to_string()];
    let doc_with_phrase = vec![
        "some".to_string(),
        "knowledge".to_string(),
        "graph".to_string(),
        "extraction".to_string(),
    ];
    let doc_without_phrase = vec![
        "graph".to_string(),
        "of".to_string(),
        "knowledge".to_string(),
    ];

    let bonus_with = reranker.compute_phrase_bonus(&query, &doc_with_phrase);
    let bonus_without = reranker.compute_phrase_bonus(&query, &doc_without_phrase);

    assert!(
        bonus_with > 0.0,
        "Should have phrase bonus for adjacent terms"
    );
    assert_eq!(
        bonus_without, 0.0,
        "Should have no bonus for non-adjacent terms"
    );
}

#[tokio::test]
async fn test_phrase_boost_ranking_effect() {
    let no_boost = BM25Reranker::new();
    let with_boost = BM25Reranker::for_semantic();

    let query = "knowledge graph";
    let documents = vec![
        "This document discusses knowledge graph extraction.".to_string(),
        "The graph of knowledge is complex.".to_string(),
        "Something about graphs and some knowledge.".to_string(),
    ];

    let results_no_boost = no_boost.rerank(query, &documents, None).await.unwrap();
    let results_with_boost = with_boost.rerank(query, &documents, None).await.unwrap();

    let phrase_doc_score_boosted = results_with_boost
        .iter()
        .find(|r| r.index == 0)
        .unwrap()
        .relevance_score;
    let non_phrase_doc_score_boosted = results_with_boost
        .iter()
        .find(|r| r.index == 1)
        .unwrap()
        .relevance_score;

    assert!(
        phrase_doc_score_boosted > non_phrase_doc_score_boosted,
        "Phrase match should score higher with boost"
    );

    let phrase_score_no_boost = results_no_boost
        .iter()
        .find(|r| r.index == 0)
        .unwrap()
        .relevance_score;
    assert!(
        phrase_doc_score_boosted > phrase_score_no_boost,
        "Boosted score should be higher"
    );
}

// =========== TermOverlapReranker Tests ===========

#[tokio::test]
async fn test_term_overlap_reranker() {
    let reranker = TermOverlapReranker::new();
    let query = "capital of France";
    let documents = vec![
        "The capital of France is Paris.".to_string(),
        "Tokyo is the capital of Japan.".to_string(),
    ];

    let results = reranker.rerank(query, &documents, None).await.unwrap();

    assert_eq!(results[0].index, 0);
    assert_eq!(reranker.name(), "term-overlap");
}

#[test]
fn test_mock_reranker_alias() {
    let mock: MockReranker = MockReranker::new();
    let term_overlap: TermOverlapReranker = TermOverlapReranker::new();

    assert_eq!(mock.name(), term_overlap.name());
}

// =========== RRF Reranker Tests ===========

#[test]
fn test_rrf_fusion_basic() {
    let rrf = RRFReranker::new();

    let list1 = vec![0, 1, 2];
    let list2 = vec![2, 1, 0];

    let results = rrf.fuse(&[list1, list2], 3);

    assert!(!results.is_empty());
}

#[test]
fn test_rrf_fusion_clear_winner() {
    let rrf = RRFReranker::with_k(1);

    let list1 = vec![0, 1, 2];
    let list2 = vec![0, 2, 1];

    let results = rrf.fuse(&[list1, list2], 3);

    assert_eq!(results[0].index, 0);
    assert!((results[0].relevance_score - 1.0).abs() < 0.01);
}

#[tokio::test]
async fn test_rrf_reranker_trait() {
    let reranker = RRFReranker::new();
    let query = "test query";
    let documents = vec![
        "First document about test.".to_string(),
        "Second document.".to_string(),
    ];

    let results = reranker.rerank(query, &documents, None).await.unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].index, 0);
}

// =========== Hybrid Reranker Tests ===========

#[tokio::test]
async fn test_hybrid_reranker_without_vector() {
    let reranker = HybridReranker::new();
    let query = "test query";
    let documents = vec![
        "This is a test document.".to_string(),
        "Another document here.".to_string(),
    ];

    let results = reranker.rerank(query, &documents, None).await.unwrap();

    assert_eq!(results[0].index, 0);
}

#[tokio::test]
async fn test_hybrid_reranker_with_vector() {
    let reranker = HybridReranker::new();
    let query = "test query";
    let documents = vec![
        "This is a test document.".to_string(),
        "Another document here.".to_string(),
        "Third one with test.".to_string(),
    ];

    let vector_rankings = vec![1, 2, 0];

    let results = reranker
        .rerank_hybrid(query, &documents, Some(vector_rankings), None)
        .await
        .unwrap();

    assert_eq!(results.len(), 3);
}

#[test]
fn test_hybrid_reranker_defaults() {
    let reranker = HybridReranker::new();
    assert_eq!(reranker.name(), "hybrid");
    assert_eq!(reranker.model(), "hybrid-reranker");
}

// =========== Tokenizer Config Tests ===========

#[test]
fn test_tokenizer_config_default() {
    let config = TokenizerConfig::default();
    assert!(config.enable_stemming);
    assert!(config.enable_stop_words);
    assert_eq!(config.min_token_length, 2);
}

#[test]
fn test_tokenizer_config_minimal() {
    let config = TokenizerConfig::minimal();
    assert!(!config.enable_stemming);
    assert!(!config.enable_stop_words);
}

#[test]
fn test_tokenizer_config_enhanced() {
    let config = TokenizerConfig::enhanced();
    assert!(config.enable_stemming);
    assert!(config.enable_stop_words);
}

#[test]
fn test_enhanced_tokenizer_stemming() {
    let reranker = BM25Reranker::new_enhanced();
    let tokens = reranker.tokenize_with_config("running jumps played");
    assert!(tokens.contains(&"run".to_string()));
    assert!(tokens.contains(&"jump".to_string()));
    assert!(tokens.contains(&"play".to_string()));
}

#[test]
fn test_enhanced_tokenizer_stop_words() {
    let reranker = BM25Reranker::new_enhanced();
    let tokens = reranker.tokenize_with_config("the quick brown fox");
    assert!(!tokens.iter().any(|t| t == "the"));
    assert!(tokens.len() >= 2);
}

// =========== IDF Optimization Tests ===========

#[test]
fn test_document_frequency_computation() {
    let docs = vec![
        vec!["the".to_string(), "quick".to_string(), "brown".to_string()],
        vec!["the".to_string(), "lazy".to_string(), "dog".to_string()],
        vec!["quick".to_string(), "fox".to_string()],
    ];

    let df_map = BM25Reranker::compute_document_frequencies(&docs);

    assert_eq!(df_map.get("the"), Some(&2));
    assert_eq!(df_map.get("quick"), Some(&2));
    assert_eq!(df_map.get("brown"), Some(&1));
    assert_eq!(df_map.get("fox"), Some(&1));
    assert_eq!(df_map.get("missing"), None);
}

#[test]
fn test_idf_from_df_equivalence() {
    let docs = vec![
        vec!["apple".to_string(), "banana".to_string()],
        vec!["apple".to_string(), "cherry".to_string()],
        vec!["banana".to_string(), "date".to_string()],
    ];

    let n = docs.len() as f64;
    let df_map = BM25Reranker::compute_document_frequencies(&docs);

    let idf_old = BM25Reranker::compute_idf("apple", &docs);
    let idf_new = BM25Reranker::compute_idf_from_df(n, *df_map.get("apple").unwrap() as f64);
    assert!((idf_old - idf_new).abs() < 1e-10);
}

#[test]
fn test_bm25_params_clamping() {
    let reranker = BM25Reranker::with_full_params(10.0, 2.0, -1.0);
    assert_eq!(reranker.k1, 3.0);
    assert_eq!(reranker.b, 1.0);
    assert_eq!(reranker.delta, 0.0);
}

// =========== Edge Case Tests ===========

#[tokio::test]
async fn test_bm25_single_document() {
    let reranker = BM25Reranker::new();
    let query = "test query";
    let documents = vec!["Single document with test.".to_string()];

    let results = reranker.rerank(query, &documents, None).await.unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].index, 0);
    assert!(results[0].relevance_score > 0.0);
}

#[tokio::test]
async fn test_bm25_boundary_top_n_zero() {
    let reranker = BM25Reranker::new();
    let query = "test";
    let documents = vec!["test document".to_string()];

    let results = reranker.rerank(query, &documents, Some(0)).await.unwrap();

    assert!(results.is_empty());
}

#[tokio::test]
async fn test_bm25_boundary_top_n_larger_than_docs() {
    let reranker = BM25Reranker::new();
    let query = "test";
    let documents = vec!["test one".to_string(), "test two".to_string()];

    let results = reranker.rerank(query, &documents, Some(100)).await.unwrap();

    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn test_rrf_empty_rankings() {
    let reranker = RRFReranker::new();
    let results = reranker.fuse(&[], 5);
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_rrf_single_ranking() {
    let reranker = RRFReranker::new();
    let single_list = vec![2, 0, 1];
    let results = reranker.fuse(&[single_list], 3);

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].index, 2);
}
