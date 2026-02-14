//! Reranking example
//!
//! Demonstrates document reranking to improve search result relevance.
//!
//! Run with: cargo run --example reranking
//! No API key required - uses local BM25 algorithm
//!
//! This example shows:
//! - Creating a BM25 reranker (local, no API needed)
//! - Reranking search results for improved relevance
//! - Different presets for different use cases
//! - Reciprocal Rank Fusion (RRF) for combining scores

use edgequake_llm::reranker::{BM25Reranker, RRFReranker, RerankResult, Reranker};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔀 EdgeQuake LLM - Reranking Example\n");

    // Sample documents (simulating search results)
    let documents = vec![
        "Rust is a systems programming language focused on safety and performance.",
        "The Python programming language is popular for machine learning.",
        "Memory safety in Rust is achieved without a garbage collector.",
        "JavaScript is the language of the web browser.",
        "Rust's ownership system prevents data races at compile time.",
        "Go is a statically typed language designed at Google.",
        "Rust async/await provides efficient concurrency without threads.",
        "C++ is a general-purpose programming language.",
        "Rust compiles to native code with zero-cost abstractions.",
        "Java runs on the JVM with automatic garbage collection.",
    ];

    let doc_strings: Vec<String> = documents.iter().map(|s| s.to_string()).collect();

    // Query to rerank documents for
    let query = "Rust memory safety without garbage collection";

    println!("📝 Query: \"{}\"", query);
    println!("{}", "─".repeat(60));

    // === BM25 Reranking ===
    println!("\n📊 BM25 Reranking (General Purpose)\n");

    let bm25 = BM25Reranker::new();
    println!("Reranker: {} ({})", bm25.name(), bm25.model());

    let results = bm25.rerank(query, &doc_strings, Some(5)).await?;
    print_results(&results, &documents);

    // === BM25 for RAG ===
    println!("\n📊 BM25 for RAG (Optimized for Knowledge Retrieval)\n");

    let bm25_rag = BM25Reranker::for_rag();
    println!("Reranker: {} ({})", bm25_rag.name(), bm25_rag.model());
    println!(
        "Settings: k1={}, b={}, delta={}",
        bm25_rag.k1, bm25_rag.b, bm25_rag.delta
    );

    let rag_results = bm25_rag.rerank(query, &doc_strings, Some(5)).await?;
    print_results(&rag_results, &documents);

    // === Reciprocal Rank Fusion (RRF) ===
    println!("\n📊 Reciprocal Rank Fusion (Combining Multiple Rankings)\n");

    // RRF combines multiple ranked lists into one
    // Useful when you have multiple retrieval methods
    let rrf = RRFReranker::new();
    println!("Reranker: {} (k={})", rrf.name(), 60); // Default RRF k=60

    let rrf_results = rrf.rerank(query, &doc_strings, Some(5)).await?;
    print_results(&rrf_results, &documents);

    // === Different Presets ===
    println!("\n📋 Available BM25 Presets:\n");
    println!("  new()           - General purpose (k1=1.5, b=0.75)");
    println!("  for_short_docs() - Tweets, titles (k1=1.2, b=0.3)");
    println!("  for_long_docs()  - Papers, articles (delta=1.0)");
    println!("  for_technical()  - Code, APIs (k1=2.0, b=0.5)");
    println!("  for_rag()        - Knowledge retrieval (delta=0.5)");

    println!("\n{}", "─".repeat(60));
    println!("✅ Reranking complete!");

    Ok(())
}

/// Print rerank results with scores.
fn print_results(results: &[RerankResult], documents: &[&str]) {
    for (rank, result) in results.iter().enumerate() {
        let doc_preview: String = documents[result.index].chars().take(50).collect();
        let preview = if doc_preview.len() < documents[result.index].len() {
            format!("{}...", doc_preview)
        } else {
            doc_preview
        };
        println!(
            "  {}. [score={:.4}] {}",
            rank + 1,
            result.relevance_score,
            preview
        );
    }
}
