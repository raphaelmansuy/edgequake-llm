//! VertexAI embeddings examples — semantic search, similarity, batch, custom dimensions.
//!
//! VertexAI embeddings use the :predict endpoint instead of the GoogleAI embedContent API.
//!
//! Run: cargo run --example vertexai_embeddings
//! Requires:
//!   - GOOGLE_CLOUD_PROJECT
//!   - Authenticated via `gcloud auth login` (or GOOGLE_ACCESS_TOKEN)

use edgequake_llm::providers::gemini::GeminiProvider;
use edgequake_llm::traits::{EmbeddingProvider, LLMProvider};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = GeminiProvider::from_env_vertex_ai()?;
    println!(
        "VertexAI Embeddings Examples — model: {}\n",
        LLMProvider::model(&provider)
    );
    println!("Embedding model : {}", EmbeddingProvider::model(&provider));
    println!("Default dimension: {}\n", provider.dimension());

    // ────────────────────────────────────────────────────────────── 1 ──
    // Single embedding
    // ────────────────────────────────────────────────────────────── 1 ──
    println!("=== 1. Single embedding ===");
    let embedding = provider.embed_one("Hello, Vertex AI!").await?;
    println!("Dimensions: {}", embedding.len());
    println!("First 8: {:?}", &embedding[..8.min(embedding.len())]);
    println!(
        "L2 norm: {:.6}",
        embedding.iter().map(|x| x * x).sum::<f32>().sqrt()
    );
    println!();

    // ────────────────────────────────────────────────────────────── 2 ──
    // Batch embeddings
    // ────────────────────────────────────────────────────────────── 2 ──
    println!("=== 2. Batch embeddings ===");
    let texts = vec![
        "Machine learning is a subset of artificial intelligence.".to_string(),
        "Deep learning uses neural networks with many layers.".to_string(),
        "Natural language processing handles text and speech.".to_string(),
        "Computer vision enables machines to interpret images.".to_string(),
        "Reinforcement learning trains agents via rewards.".to_string(),
    ];
    let embeddings = provider.embed(&texts).await?;
    println!("Embedded {} texts", embeddings.len());
    for (i, emb) in embeddings.iter().enumerate() {
        println!("  [{}] {} dims — first 3: {:?}", i, emb.len(), &emb[..3]);
    }
    println!();

    // ────────────────────────────────────────────────────────────── 3 ──
    // Semantic search — find most relevant document
    // ────────────────────────────────────────────────────────────── 3 ──
    println!("=== 3. Semantic search ===");
    let corpus = [
        "Rust is a systems programming language focused on safety and performance.",
        "Python is popular for data science and machine learning.",
        "JavaScript runs in web browsers and on servers with Node.js.",
        "Go was designed at Google for scalable cloud services.",
        "TypeScript adds static types to JavaScript.",
    ];
    let corpus_strings: Vec<String> = corpus.iter().map(|s| s.to_string()).collect();
    let corpus_embs = provider.embed(&corpus_strings).await?;

    let query = "Which language is best for building high-performance systems?";
    let query_emb = provider.embed_one(query).await?;

    println!("Query: \"{}\"\n", query);
    let mut scored: Vec<(usize, f32)> = corpus_embs
        .iter()
        .enumerate()
        .map(|(i, emb)| (i, cosine_similarity(&query_emb, emb)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Results (ranked by similarity):");
    for (i, (idx, sim)) in scored.iter().enumerate() {
        let marker = if i == 0 { " <-- best match" } else { "" };
        println!("  {:.4} — \"{}\"{}", sim, corpus[*idx], marker);
    }
    println!();

    // ────────────────────────────────────────────────────────────── 4 ──
    // Similarity matrix
    // ────────────────────────────────────────────────────────────── 4 ──
    println!("=== 4. Similarity matrix ===");
    let items = vec![
        "The cat sat on the mat.".to_string(),
        "A kitten rested on the rug.".to_string(),
        "Stock prices rose sharply today.".to_string(),
        "The financial markets surged.".to_string(),
    ];
    let item_embs = provider.embed(&items).await?;

    println!("{:>40} | {:>6} {:>6} {:>6} {:>6}", "", "0", "1", "2", "3");
    println!("{}", "-".repeat(75));
    for (i, emb_i) in item_embs.iter().enumerate() {
        let label = if items[i].len() > 35 {
            format!("{}...", &items[i][..32])
        } else {
            items[i].clone()
        };
        print!("{:>40} |", label);
        for emb_j in &item_embs {
            print!(" {:.4}", cosine_similarity(emb_i, emb_j));
        }
        println!();
    }
    println!();

    // ────────────────────────────────────────────────────────────── 5 ──
    // Custom embedding dimensions
    // ────────────────────────────────────────────────────────────── 5 ──
    println!("=== 5. Custom embedding dimensions ===");
    for dim in [256, 512, 1024, 3072] {
        let custom_provider = GeminiProvider::from_env_vertex_ai()?.with_embedding_dimension(dim);

        let emb = custom_provider.embed_one("Test dimensionality").await?;
        println!("  Requested: {} → Got: {} dims", dim, emb.len());
    }
    println!();

    // ────────────────────────────────────────────────────────────── 6 ──
    // Clustering example
    // ────────────────────────────────────────────────────────────── 6 ──
    println!("=== 6. Simple clustering (by average similarity) ===");
    let group_a = vec![
        "I love hiking in the mountains.".to_string(),
        "Trail running is my favorite exercise.".to_string(),
        "Camping under the stars is peaceful.".to_string(),
    ];
    let group_b = vec![
        "Quantum computing uses qubits.".to_string(),
        "Superposition is a key quantum concept.".to_string(),
        "Quantum entanglement defies intuition.".to_string(),
    ];

    let embs_a = provider.embed(&group_a).await?;
    let embs_b = provider.embed(&group_b).await?;

    let avg_within_a = avg_pairwise_similarity(&embs_a);
    let avg_within_b = avg_pairwise_similarity(&embs_b);
    let avg_between = avg_cross_similarity(&embs_a, &embs_b);

    println!("  Within 'Outdoor' group  : {:.4}", avg_within_a);
    println!("  Within 'Quantum' group  : {:.4}", avg_within_b);
    println!("  Between groups          : {:.4}", avg_between);
    println!("  (Within-group similarity should be higher than between-group)");
    println!();

    println!("=== Done! ===");
    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn avg_pairwise_similarity(embs: &[Vec<f32>]) -> f32 {
    let n = embs.len();
    if n < 2 {
        return 1.0;
    }
    let mut sum = 0.0;
    let mut count = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            sum += cosine_similarity(&embs[i], &embs[j]);
            count += 1;
        }
    }
    sum / count as f32
}

fn avg_cross_similarity(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    let mut sum = 0.0;
    let mut count = 0;
    for ea in a {
        for eb in b {
            sum += cosine_similarity(ea, eb);
            count += 1;
        }
    }
    sum / count as f32
}
