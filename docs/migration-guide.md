# Migration Guide

This guide covers upgrading between edgequake-llm versions.

---

## Upgrading to 0.2.0 (from EdgeCode internal)

Version 0.2.0 is the first standalone release of edgequake-llm, extracted from the EdgeCode project. If you were using the library as an internal module, follow these steps.

### 1. Update Cargo.toml

```toml
[dependencies]
edgequake-llm = "0.2.0"
```

### 2. Update Import Paths

All types are re-exported from the crate root and from their respective modules:

```rust
// Before (internal module)
use edgecode::llm::{LLMProvider, ChatMessage, CompletionOptions};

// After (standalone crate)
use edgequake_llm::{LLMProvider, ChatMessage, CompletionOptions};
use edgequake_llm::traits::{LLMProvider, EmbeddingProvider};
use edgequake_llm::providers::OpenAIProvider;
```

### 3. Provider Construction

Providers are created through `ProviderFactory` or directly:

```rust
// Auto-detect from environment variables
let (llm, embedding) = ProviderFactory::from_env()?;

// Explicit provider type
let (llm, embedding) = ProviderFactory::create(ProviderType::OpenAI)?;

// Direct construction
use edgequake_llm::providers::OpenAIProvider;
let provider = OpenAIProvider::new("sk-...");
```

### 4. Error Type Changes

All errors now use `edgequake_llm::LlmError`:

```rust
use edgequake_llm::{LlmError, Result};

match provider.chat(&messages, None).await {
    Ok(response) => println!("{}", response.content),
    Err(LlmError::RateLimit { retry_after, .. }) => {
        tokio::time::sleep(retry_after).await;
    }
    Err(LlmError::AuthenticationError(msg)) => {
        eprintln!("Check your API key: {msg}");
    }
    Err(e) => eprintln!("Error: {e}"),
}
```

### 5. Feature Flags

OpenTelemetry support is now behind a feature flag:

```toml
[dependencies]
edgequake-llm = { version = "0.2.0", features = ["otel"] }
```

Without the `otel` feature, `TracingProvider` still works with the `tracing` crate but does not depend on `opentelemetry` or `tracing-opentelemetry`.

### 6. New Modules in 0.2.0

| Module | Purpose |
|--------|---------|
| `cache_prompt` | Anthropic-style prompt caching with `CachePromptConfig` |
| `cost_tracker` | `SessionCostTracker` with budget management |
| `inference_metrics` | `InferenceMetrics` for streaming display (TTFT, t/s) |
| `model_config` | Model configuration and presets |
| `registry` | Dynamic provider registry for runtime lookup |
| `reranker` | BM25, RRF, hybrid, HTTP, term overlap rerankers |
| `middleware` | `LLMMiddleware` trait with logging and metrics |

### 7. Trait Changes

The `LLMProvider` trait now includes:

```rust
#[async_trait]
pub trait LLMProvider: Send + Sync {
    fn name(&self) -> &str;
    fn model(&self) -> &str;
    fn max_context_length(&self) -> usize;

    async fn complete(&self, prompt: &str) -> Result<LLMResponse>;
    async fn complete_with_options(&self, prompt: &str, options: &CompletionOptions) -> Result<LLMResponse>;
    async fn chat(&self, messages: &[ChatMessage], options: Option<&CompletionOptions>) -> Result<LLMResponse>;
    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>>;

    // New in 0.2.0
    async fn chat_with_tools(
        &self, messages: &[ChatMessage], tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>, options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> { /* default: unsupported error */ }

    async fn chat_with_tools_stream(
        &self, messages: &[ChatMessage], tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>, options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>> { /* default: unsupported error */ }

    fn supports_streaming(&self) -> bool { true }
    fn supports_tool_streaming(&self) -> bool { false }
    fn supports_json_mode(&self) -> bool { false }
    fn supports_function_calling(&self) -> bool { false }
}
```

Tool-related methods have default implementations that return an error, so existing provider implementations compile without changes.

### 8. LLMResponse New Fields

```rust
pub struct LLMResponse {
    pub content: String,
    pub model: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub finish_reason: Option<String>,
    pub tool_calls: Vec<ToolCall>,           // New
    pub thinking_content: Option<String>,    // New (Claude, o-series)
    pub thinking_tokens: Option<usize>,      // New
    pub cache_hit_tokens: Option<usize>,     // New
    pub metadata: HashMap<String, Value>,    // New
}
```

### 9. StreamChunk Enum

Streaming now uses `StreamChunk` instead of plain strings for rich streaming:

```rust
pub enum StreamChunk {
    Content(String),
    ThinkingContent { text: String, token_count: Option<usize> },
    ToolCallDelta { index: usize, id: Option<String>, function_name: Option<String>, function_arguments: Option<String> },
    Finished { reason: String, ttft_ms: Option<f64> },
}
```

The old `stream()` method still returns `BoxStream<'static, Result<String>>` for backward compatibility.

---

## Version Compatibility

| edgequake-llm | Rust | tokio | reqwest |
|---------------|------|-------|---------|
| 0.2.x | >= 1.75 | 1.x | 0.12 |

---

## See Also

- [Architecture](architecture.md) - system design overview
- [Providers](providers.md) - provider-specific setup
