# OODA Iteration 42 - Decide

## Decision: Add Jina Provider Tests

### Rationale

Looking at remaining modules, **jina.rs** (62/78 = 79%) has:
- Relatively high coverage already
- Simple API (embeddings only)
- Potential for a few more unit tests without HTTP mocking

### Tests to Add

1. Test `from_env` with missing API key
2. Test constants (BASE_URL, PROVIDER_NAME)
3. Test model dimension lookup
4. Test EmbeddingProvider trait implementation

### Implementation Plan

Add 5-8 tests to `src/providers/jina.rs`:

```rust
// Test environment error handling
#[test]
fn test_from_env_missing_api_key()

// Test constants
#[test]
fn test_constants()

// Test trait implementation
#[test]
fn test_embedding_provider_name()
#[test]
fn test_embedding_provider_dimension()
#[test]
fn test_embedding_provider_max_tokens()
```

### Expected Outcome

- jina.rs coverage: 79% → 90%+
- Overall coverage: ~51.5% → ~52%
- Demonstrates continued testing effort

### Alternative Decision

If Jina tests prove insufficient, pivot to:
- Add more config validation tests
- Add serde roundtrip tests for request/response types
