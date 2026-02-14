# Reranking

EdgeQuake LLM provides a modular reranking pipeline for improving search
result relevance using multiple scoring strategies.

## Architecture

```text
  Query + Candidate Documents
       |
       v
  +----+----+
  | Reranker |  trait Reranker: rerank(query, docs, top_n) -> Vec<RerankResult>
  +----+----+
       |
       +----------+-----------+-----------+------------+
       |          |           |           |            |
       v          v           v           v            v
  +--------+ +--------+ +---------+ +--------+ +----------+
  |  BM25  | |  RRF   | | Hybrid  | |  HTTP  | |TermOverlap|
  | (local)| |(fusion)| |(combined| |(remote)| |  (test)   |
  +--------+ +--------+ +---------+ +--------+ +----------+
```

## Reranker Trait

All rerankers implement a common interface:

```rust,ignore
#[async_trait]
pub trait Reranker: Send + Sync {
    fn name(&self) -> &str;
    fn model(&self) -> &str;
    async fn rerank(&self, query: &str, documents: &[String], top_n: Option<usize>) -> Result<Vec<RerankResult>>;
}
```

## BM25 Reranker

Industry-standard BM25 ranking (used by Elasticsearch, Lucene).

```text
  BM25 Score = SUM IDF(q) x f(q,D) x (k1+1)
                          / (f(q,D) + k1 x (1 - b + b x |D|/avgdl))

  Where:
    f(q,D)  = term frequency of query term q in document D
    |D|     = document length
    avgdl   = average document length across corpus
    k1      = term frequency saturation (default: 1.5)
    b       = length normalization (default: 0.75)
```

**Why BM25?** Best-in-class keyword matching that handles term frequency
saturation and document length normalization. No external API needed.

### Presets

| Preset | k1 | b | delta | Use Case |
|--------|----|----|-------|----------|
| `new()` | 1.5 | 0.75 | 0 | General purpose |
| `for_short_docs()` | 1.2 | 0.3 | 0 | Tweets, titles |
| `for_long_docs()` | 1.5 | 0.75 | 1.0 | Papers, articles |
| `for_technical()` | 2.0 | 0.5 | 0 | Code, API docs |
| `for_rag()` | 1.5 | 0.75 | 0.5 | RAG retrieval |

### Example

```rust,ignore
use edgequake_llm::reranker::{BM25Reranker, Reranker};

let reranker = BM25Reranker::for_rag();

let docs = vec![
    "Rust is a systems programming language".to_string(),
    "Python is great for data science".to_string(),
    "Rust async runtime uses tokio".to_string(),
];

let results = reranker.rerank("rust async", &docs, Some(2)).await?;
for result in &results {
    println!("Doc {}: score {:.4}", result.index, result.relevance_score);
}
```

### Tokenizer Configuration

BM25 includes configurable tokenization:

- **Stemming**: Porter stemmer for English (reduces "running" to "run")
- **Stop words**: Common words filtered (configurable)
- **Unicode normalization**: NFC normalization for consistent matching
- **Case folding**: Case-insensitive matching

## Reciprocal Rank Fusion (RRF)

Combines multiple ranking signals without score normalization.

```text
  RRF Score = SUM 1/(k + rank) for each ranking list

  Example with k=60:
    Doc A: rank 1 in BM25, rank 3 in vector  =>  1/61 + 1/63 = 0.0322
    Doc B: rank 2 in BM25, rank 1 in vector  =>  1/62 + 1/61 = 0.0325
    Doc B wins (balanced across both signals)
```

**Why RRF?** Rank-based fusion avoids the need to normalize incompatible
score distributions (BM25 scores vs cosine similarity values).

### Example

```rust,ignore
use edgequake_llm::reranker::RRFReranker;

let rrf = RRFReranker::new();           // k=60 (default)
let rrf = RRFReranker::with_k(10);      // Lower k = more weight to top results
```

## Hybrid Reranker

Combines BM25 keyword scoring with another reranker (typically neural) using RRF.

```text
  Query + Documents
       |
       +--------+--------+
       |                 |
       v                 v
  +---------+     +----------+
  |  BM25   |     |  Neural  |
  | Scoring |     | Reranker |
  +---------+     +----------+
       |                 |
       v                 v
  +---------+     +----------+
  | Ranking |     | Ranking  |
  +---------+     +----------+
       |                 |
       +--------+--------+
                |
                v
          +-----+-----+
          | RRF Fusion |
          +-----+-----+
                |
                v
          Final Ranking
```

### Example

```rust,ignore
use edgequake_llm::reranker::{HybridReranker, BM25Reranker, HttpReranker};

// Combine BM25 with a neural reranker
let bm25 = BM25Reranker::new();
let neural = HttpReranker::jina("your-api-key");
let hybrid = HybridReranker::new(bm25, neural);

let results = hybrid.rerank("async rust tokio", &docs, Some(10)).await?;
```

## HTTP Reranker

Remote API-based reranking via Jina AI, Cohere, or Aliyun DashScope.

### Providers

| Provider | Model | Endpoint |
|----------|-------|----------|
| Jina AI | `jina-reranker-v2-base-multilingual` | `api.jina.ai` |
| Cohere | `rerank-english-v3.0` | `api.cohere.com` |
| Aliyun | `gte-rerank-v2` | `dashscope.aliyuncs.com` |

### Example

```rust,ignore
use edgequake_llm::reranker::HttpReranker;

// Jina AI reranker
let reranker = HttpReranker::jina("your-jina-key");

// Cohere reranker
let reranker = HttpReranker::cohere("your-cohere-key");

let results = reranker.rerank("search query", &docs, Some(5)).await?;
```

## Term Overlap Reranker

Simple keyword matching for testing and fast fallback.

```rust,ignore
use edgequake_llm::reranker::TermOverlapReranker;

let reranker = TermOverlapReranker::new();
// Scores based on proportion of query terms found in each document
```

## Mock Reranker

For testing without any scoring logic.

```rust,ignore
use edgequake_llm::reranker::MockReranker;

let reranker = MockReranker::new();
// Returns documents in original order with fixed scores
```

## Source Files

| File | Purpose |
|------|---------|
| `src/reranker/mod.rs` | Module exports |
| `src/reranker/traits.rs` | Reranker trait |
| `src/reranker/bm25.rs` | BM25 reranker |
| `src/reranker/rrf.rs` | Reciprocal Rank Fusion |
| `src/reranker/hybrid.rs` | Hybrid BM25 + Neural |
| `src/reranker/http.rs` | Remote API rerankers |
| `src/reranker/term_overlap.rs` | Term overlap + Mock |
| `src/reranker/config.rs` | Configuration |
| `src/reranker/result.rs` | Result types |
