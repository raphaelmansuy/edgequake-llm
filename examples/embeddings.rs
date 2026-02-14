//! Embeddings example
//!
//! Demonstrates generating text embeddings and calculating similarity.
//!
//! Run with: cargo run --example embeddings
//! Requires: OPENAI_API_KEY environment variable
//!
//! This example shows:
//! - Creating an embedding provider
//! - Generating embeddings for text
//! - Calculating cosine similarity between vectors
//! - Basic semantic search concepts

use edgequake_llm::{EmbeddingProvider, OpenAIProvider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    // Initialize provider (OpenAI supports both LLM and embeddings)
    let provider = OpenAIProvider::new(&api_key);

    println!("🔮 EdgeQuake LLM - Embeddings Example\n");
    println!("Provider: {}", EmbeddingProvider::name(&provider));
    println!("Model: {}", EmbeddingProvider::model(&provider));
    println!("Dimension: {}", provider.dimension());
    println!("{}", "─".repeat(50));

    // Sample documents for semantic search
    let documents = vec![
        "Rust is a systems programming language focused on safety.",
        "Python is great for data science and machine learning.",
        "JavaScript powers most web applications.",
        "Memory safety without garbage collection is Rust's key feature.",
        "Machine learning models can be trained on large datasets.",
    ];

    // Query to search for
    let query = "Which language is best for safe systems programming?";

    println!("\n📝 Query: {}\n", query);

    // Generate embeddings for documents
    println!(
        "📊 Generating embeddings for {} documents...",
        documents.len()
    );

    let doc_texts: Vec<String> = documents.iter().map(|s| s.to_string()).collect();
    let doc_embeddings = provider.embed(&doc_texts).await?;

    // Generate embedding for query
    let query_embedding = provider.embed_one(query).await?;

    println!(
        "✅ Generated {} embeddings (dim={})\n",
        doc_embeddings.len(),
        query_embedding.len()
    );

    // Calculate cosine similarity and rank documents
    println!("🔍 Similarity Rankings:\n");

    let mut similarities: Vec<(usize, f32)> = doc_embeddings
        .iter()
        .enumerate()
        .map(|(i, doc_emb)| (i, cosine_similarity(&query_embedding, doc_emb)))
        .collect();

    // Sort by similarity (descending)
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (rank, (idx, sim)) in similarities.iter().enumerate() {
        println!("{}. [sim={:.4}] {}", rank + 1, sim, documents[*idx]);
    }

    println!("\n{}", "─".repeat(50));
    println!("📈 Most relevant: \"{}\"", documents[similarities[0].0]);

    Ok(())
}

/// Calculate cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}
