# Migration Guide

This guide covers upgrading between edgequake-llm versions.

---

## Upgrading to 0.10.0 (from 0.9.0)

Version 0.10.0 adds **application attribution** — propagate caller identity to upstream providers and OTEL spans.

### 1. Update dependencies

```toml
[dependencies]
edgequake-llm = "0.10.0"
```

```bash
pip install --upgrade edgequake-litellm
```

### 2. Application context (Rust)

Prefer the typed API over raw header vectors:

```rust
use edgequake_llm::{ApplicationContext, ApplicationContextBuilder, ProviderFactory};

let ctx = ApplicationContextBuilder::new()
    .app_id("my-backend")
    .app_name("My Service")
    .app_url("https://app.example.com")
    .request_id("req-123")
    .build()?;

let provider = ProviderFactory::create_llm_provider_with_context("openrouter", "anthropic/claude-3.5-sonnet", ctx)?;
```

Ingress headers (`X-EdgeQuake-App-Id`, etc.) can be parsed with `ApplicationContext::from_ingress_headers()`.

`create_llm_provider_with_headers()` still works but delegates to the context API.

Use `AttributionPolicy::RequireAppId` to fail when a provider cannot propagate attribution (e.g. VS Code Copilot).

### 3. Application context (Python)

Optional keyword arguments on `completion`, `acompletion`, and `stream_completion`:

```python
import edgequake_litellm as eq

eq.completion(
    "openrouter/anthropic/claude-3.5-sonnet",
    [{"role": "user", "content": "hi"}],
    application_id="my-backend",
    application_name="My Service",
    application_url="https://app.example.com",
    request_id="req-123",
)
```

Or build a reusable context object:

```python
from edgequake_litellm import ApplicationContext, completion

ctx = ApplicationContext(
    application_id="my-backend",
    application_name="My Service",
    request_id="req-123",
)
completion("mock/test-model", [{"role": "user", "content": "hi"}], application_context=ctx)
```

Ingress headers from a web framework:

```python
ctx = ApplicationContext.from_headers(request.headers)
```

Query per-provider support before routing:

```python
from edgequake_litellm import get_provider_attribution, list_provider_attribution

assert get_provider_attribution("openai") == "full"
assert get_provider_attribution("ollama") == "passthrough"
levels = list_provider_attribution()  # dict[str, str]
```

### 4. Observability

When wrapping providers with `TracingProvider::with_application_context()`, spans include `gen_ai.application.id`, `gen_ai.application.name`, and `gen_ai.application.url`.

Each LLM span may also emit attribution audit events: `edgequake.attribution.resolved`, `edgequake.attribution.warning`, and `edgequake.attribution.unsupported` (see [observability.md](observability.md)).

W3C trace context injection defaults to **on** when the `otel` feature is enabled (`EDGEQUAKE_OTEL_INJECT_TRACE_CONTEXT=true`). Set to `false`/`0`/`no`/`off` to disable. Optional baggage promotion: `EDGEQUAKE_OTEL_PROMOTE_APP_TO_BAGGAGE`, `EDGEQUAKE_PROPAGATE_BAGGAGE_TO_PROVIDERS`.

Check `ProviderCatalog::get(id).attribution_support()` for per-provider capability metadata.

**Ollama / LM Studio**: attribution is passthrough-only (e.g. `X-Client-Request-Id` from `request_id`). No provider billing fields — use OTEL spans for app identity.

```rust
use edgequake_llm::{ApplicationContextBuilder, OllamaProvider};

let provider = OllamaProvider::from_env()?
    .with_application_context(
        ApplicationContextBuilder::new()
            .app_id("my-backend")
            .request_id("req-456")
            .build()?,
    );
```

### 5. Ollama Cloud (optional)

When `OLLAMA_API_KEY` is set, `OllamaProvider::from_env()` defaults to `https://ollama.com`. Use explicit cloud init:

```rust
use edgequake_llm::OllamaProvider;

let provider = OllamaProvider::from_env_cloud()?;
assert!(provider.is_cloud());
```

See `cargo run --example ollama_cloud` and [Ollama Cloud docs](https://docs.ollama.com/cloud).

---

## Upgrading to 0.9.0 (from 0.8.0)

Version 0.9.0 is **additive only** — no breaking changes to existing completion, embedding, or discovery APIs.

### 1. Update dependencies

```toml
[dependencies]
edgequake-llm = "0.9.0"
```

```bash
pip install --upgrade edgequake-litellm
```

### 2. New APIs (optional)

**List providers** (replaces hardcoded lists):

```rust
use edgequake_llm::ProviderFactory;
let providers = ProviderFactory::list_providers();
```

**Search models by capability (offline)**:

```rust
use edgequake_llm::{find_static_models, CapabilityFilter, ModelCapability};

let models = find_static_models(
    &CapabilityFilter::default()
        .requiring(ModelCapability::Vision)
        .with_min_context_length(100_000)
        .with_max_output_tokens(32_768),
);
```

**Search models by name / fuzzy**:

```rust
use edgequake_llm::{search_static_models, ModelSearchQuery};

let hits = search_static_models(
    &ModelSearchQuery::new("claude sonnet")
        .fuzzy(true)
        .with_min_context_length(200_000),
);
```

Python equivalents: `litellm.discovery.find_static_models()`, `search_static_models_by_name()`, `lookup_model_by_name()`.

### 3. Python `list_providers()` change

`litellm.list_providers()` now returns the live catalog from Rust (includes `cohere`, `nvidia`, and current aliases). Update tests that assert a fixed hardcoded list.

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
