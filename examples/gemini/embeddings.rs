//! Gemini embeddings example — semantic search and similarity.
//!
//! Run: cargo run --example gemini_embeddings
//! Requires: GEMINI_API_KEY
//!
//! Shows:
//!   1. Single text embedding (gemini-embedding-001, 3072 dims)
//!   2. Batch embeddings
//!   3. Semantic search with cosine similarity
//!   4. Custom embedding dimensions (768, 1536, 3072)
//!   5. Different task types (RETRIEVAL_DOCUMENT, SEMANTIC_SIMILARITY)

use edgequake_llm::providers::gemini::GeminiProvider;
use edgequake_llm::traits::EmbeddingProvider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Default: gemini-embedding-001, 3072 dimensions
    let provider = GeminiProvider::from_env()?;
    println!("Embedding model: {}", EmbeddingProvider::model(&provider));
    println!("Dimension: {}", provider.dimension());
    println!("Max tokens: {}", provider.max_tokens());
    println!("{}", "-".repeat(60));

    // ── 1. Single embedding ──────────────────────────────────────────────
    println!("\n=== 1. Single embedding ===");
    let embedding = provider.embed_one("Hello, world!").await?;
    println!("Dimension: {}", embedding.len());
    println!("First 5 values: {:?}", &embedding[..5.min(embedding.len())]);
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("L2 norm: {:.6}", norm);

    // ── 2. Batch embeddings ─────────────────────────────────────────────
    println!("\n=== 2. Batch embeddings ===");
    let documents = vec![
        "Rust is a systems programming language focused on safety and performance.".to_string(),
        "Python is excellent for data science and machine learning.".to_string(),
        "JavaScript is the language of the web browser.".to_string(),
        "Memory safety without garbage collection is Rust's key innovation.".to_string(),
        "Machine learning models can be trained on large datasets using Python.".to_string(),
        "TypeScript adds static types to JavaScript.".to_string(),
    ];

    let doc_embeddings = provider.embed(&documents).await?;
    println!(
        "Generated {} embeddings (dim={})",
        doc_embeddings.len(),
        doc_embeddings[0].len()
    );

    // ── 3. Semantic search ──────────────────────────────────────────────
    println!("\n=== 3. Semantic search ===");
    let query = "Which language is best for safe systems programming?";
    println!("Query: \"{}\"\n", query);

    let query_embedding = provider.embed_one(query).await?;

    let mut results: Vec<(usize, f32)> = doc_embeddings
        .iter()
        .enumerate()
        .map(|(i, doc_emb)| (i, cosine_similarity(&query_embedding, doc_emb)))
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Ranked results:");
    for (rank, (idx, sim)) in results.iter().enumerate() {
        let marker = if rank == 0 { " <-- best match" } else { "" };
        println!("  {}. [{:.4}] {}{}", rank + 1, sim, documents[*idx], marker);
    }

    // ── 4. Custom dimensions ────────────────────────────────────────────
    println!("\n=== 4. Custom dimensions ===");

    // Create a provider with 768-dim output (recommended for storage efficiency)
    let provider_768 = GeminiProvider::from_env()?.with_embedding_dimension(768);
    println!(
        "Custom provider: {} (dim={})",
        EmbeddingProvider::model(&provider_768),
        provider_768.dimension()
    );

    let emb_768 = provider_768.embed_one("Hello, custom dimensions!").await?;
    println!(
        "768-dim embedding: {} values, first 5: {:?}",
        emb_768.len(),
        &emb_768[..5.min(emb_768.len())]
    );

    // ── 5. Similarity matrix ────────────────────────────────────────────
    println!("\n=== 5. Similarity matrix ===");
    let concepts = vec![
        "artificial intelligence".to_string(),
        "machine learning".to_string(),
        "quantum computing".to_string(),
        "web development".to_string(),
    ];

    let concept_embeddings = provider.embed(&concepts).await?;

    println!(
        "              {:>12} {:>12} {:>12} {:>12}",
        "AI", "ML", "Quantum", "Web"
    );
    for (i, label) in ["AI", "ML", "Quantum", "Web"].iter().enumerate() {
        print!("{:>12}", label);
        for j in 0..concepts.len() {
            let sim = cosine_similarity(&concept_embeddings[i], &concept_embeddings[j]);
            print!(" {:>11.4}", sim);
        }
        println!();
    }

    println!("\n=== Done! Embedding features demonstrated. ===");
    Ok(())
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}
