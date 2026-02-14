# Providers Guide

EdgeQuake LLM supports 11 provider implementations across cloud APIs, local
inference engines, IDE integrations, and testing.

## Feature Comparison

```text
+------------------+------+-------+--------+-------+---------+--------+
| Provider         | Chat | Embed | Stream | Tools | Vision  | Think  |
+------------------+------+-------+--------+-------+---------+--------+
| OpenAI           |  Y   |   Y   |   Y    |   Y   |   Y     |   -    |
| Anthropic        |  Y   |   -   |   Y    |   Y   |   Y     |   Y    |
| Gemini           |  Y   |   Y   |   Y    |   Y   |   Y     |   Y    |
| xAI (Grok)       |  Y   |   -   |   Y    |   Y   |   Y     |   -    |
| OpenRouter       |  Y   |   Y   |   Y    |   Y   |   -     |   -    |
| HuggingFace      |  Y   |   -   |   Y    |   -   |   -     |   -    |
| Azure OpenAI     |  Y   |   Y   |   Y    |   Y   |   -     |   -    |
| Ollama           |  Y   |   Y   |   Y    |   Y   |   -     |   -    |
| LM Studio        |  Y   |   Y   |   Y    |   Y   |   -     |   -    |
| VSCode Copilot   |  Y   |   Y   |   Y    |   Y   |   -     |   -    |
| OpenAI Compatible|  Y   |   Y   |   Y    |   Y   |   Y     |   Y    |
| Mock             |  Y   |   Y   |   -    |   Y   |   -     |   -    |
+------------------+------+-------+--------+-------+---------+--------+
```

---

## Cloud Providers

### OpenAI

Direct integration with OpenAI's API using the `async-openai` crate.

**Environment Variables**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | API key from platform.openai.com |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | Custom endpoint |

**Models**

| Model | Context | Notes |
|-------|---------|-------|
| `gpt-5-mini` | 200K | Default. Cost-effective reasoning |
| `gpt-4o` | 128K | Multimodal flagship |
| `gpt-4o-mini` | 128K | Smaller, faster |
| `gpt-3.5-turbo` | 16K | Legacy, low cost |

**Example**

```rust,ignore
use edgequake_llm::{OpenAIProvider, LLMProvider, ChatMessage};

let provider = OpenAIProvider::new("sk-...")
    .with_model("gpt-4o");

let response = provider.chat(
    &[ChatMessage::user("Explain trait objects in Rust")],
    None,
).await?;

println!("{}", response.content);
```

### Anthropic (Claude)

Direct integration with Anthropic's Messages API. Supports extended
thinking, vision, and prompt caching.

**Environment Variables**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | - | API key from console.anthropic.com |

**Models**

| Model | Context | Notes |
|-------|---------|-------|
| `claude-sonnet-4-5-20250929` | 200K | Default. Best balance |
| `claude-opus-4-5-20250929` | 200K | Most capable |
| `claude-3-5-sonnet-20241022` | 200K | Previous generation |
| `claude-3-5-haiku-20241022` | 200K | Fast, affordable |

**Unique Features**
- Extended thinking (reasoning traces visible in responses)
- Prompt caching with cache breakpoints (~90% cost reduction)
- Vision via base64 image source format

**Example**

```rust,ignore
use edgequake_llm::{AnthropicProvider, LLMProvider, ChatMessage};

let provider = AnthropicProvider::new("sk-ant-...");

let response = provider.chat(
    &[ChatMessage::user("What is the meaning of life?")],
    None,
).await?;

// Check for thinking content (extended thinking)
if let Some(thinking) = &response.thinking_content {
    println!("Reasoning: {}", thinking);
}
println!("Answer: {}", response.content);
```

### Gemini

Google AI Gemini API with support for both Google AI and VertexAI endpoints.

**Environment Variables**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes (Google AI) | - | API key from ai.google.dev |
| `GOOGLE_APPLICATION_CREDENTIALS` | Yes (VertexAI) | - | Service account JSON path |
| `GOOGLE_CLOUD_PROJECT` | Yes (VertexAI) | - | GCP project ID |
| `GOOGLE_CLOUD_REGION` | No | `us-central1` | GCP region |

**Models**

| Model | Context | Notes |
|-------|---------|-------|
| `gemini-2.5-flash` | 1M | Default. Fast, large context |
| `gemini-2.5-pro` | 1M | Most capable |
| `gemini-2.0-flash` | 1M | Previous generation |

**Unique Features**
- Context caching (KV-cache with TTL)
- 1M token context window
- Dual endpoint support (Google AI + VertexAI)
- Embeddings via `text-embedding-004`

**Example**

```rust,ignore
use edgequake_llm::GeminiProvider;

// Google AI endpoint
let provider = GeminiProvider::from_env()?;

// VertexAI endpoint
let provider = GeminiProvider::vertex_ai(
    "my-project",
    "us-central1",
    "access-token",
);
```

### xAI (Grok)

Direct access to xAI's Grok models via api.x.ai. Uses OpenAI-compatible
API format internally.

**Environment Variables**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `XAI_API_KEY` | Yes | - | API key from console.x.ai |
| `XAI_MODEL` | No | `grok-4` | Default model |
| `XAI_BASE_URL` | No | `https://api.x.ai` | API endpoint |

**Models**

| Model | Context | Notes |
|-------|---------|-------|
| `grok-4` | 128K | Flagship reasoning model |
| `grok-4-0709` | 128K | July 2025 release |
| `grok-4-1-fast` | 2M | Fast agentic, tool calling |
| `grok-3` | 128K | Previous generation |
| `grok-3-mini` | 128K | Smaller, faster |
| `grok-2-vision-1212` | 32K | Image understanding |

**Example**

```rust,ignore
use edgequake_llm::XAIProvider;

let provider = XAIProvider::from_env()?;
let response = provider.complete("Write a haiku about Rust").await?;
```

### OpenRouter

Unified gateway to 200+ models from multiple providers. Supports dynamic
model discovery with caching.

**Environment Variables**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | - | API key from openrouter.ai |

**Default Model**: `anthropic/claude-3.5-sonnet`

**Unique Features**
- Access to 200+ models via single API key
- Dynamic model discovery: `list_models_cached()`
- Automatic routing and fallbacks
- Pay-per-token pricing across providers

**Example**

```rust,ignore
use edgequake_llm::{OpenRouterProvider, LLMProvider};
use std::time::Duration;

let provider = OpenRouterProvider::from_env()?
    .with_model("anthropic/claude-3.5-sonnet");

// List available models
let models = provider.list_models_cached(Duration::from_secs(3600)).await?;
for model in models.iter().take(5) {
    println!("{}: {}K context", model.id, model.context_length / 1000);
}
```

### HuggingFace

Access to open-source models via HuggingFace's Inference API.

**Environment Variables**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | - | Token from huggingface.co/settings/tokens |
| `HUGGINGFACE_TOKEN` | Alt | - | Alternative token variable |
| `HF_MODEL` | No | `meta-llama/Meta-Llama-3.1-70B-Instruct` | Default model |

**Models**

| Model | Context | Notes |
|-------|---------|-------|
| `meta-llama/Meta-Llama-3.1-70B-Instruct` | 128K | Default |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 128K | Smaller |
| `mistralai/Mistral-7B-Instruct-v0.3` | 32K | Mistral |
| `Qwen/Qwen2.5-72B-Instruct` | 128K | Qwen |

**Example**

```rust,ignore
use edgequake_llm::providers::huggingface::HuggingFaceProvider;

let provider = HuggingFaceProvider::from_env()?;
let response = provider.complete("Explain transformers").await?;
```

### Azure OpenAI

Enterprise Azure OpenAI Service with deployment-based model selection.

**Environment Variables**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Yes | - | e.g., `https://myresource.openai.azure.com` |
| `AZURE_OPENAI_API_KEY` | Yes | - | API key |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Yes | - | Chat model deployment name |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` | No | - | Embedding deployment name |
| `AZURE_OPENAI_API_VERSION` | No | `2024-10-21` | API version |

**Example**

```rust,ignore
use edgequake_llm::AzureOpenAIProvider;

let provider = AzureOpenAIProvider::from_env()?;
let response = provider.complete("Hello from Azure").await?;
```

---

## Local Providers

### Ollama

Local LLM inference via Ollama. No API key required.

**Environment Variables**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OLLAMA_HOST` | No | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | No | `gemma3:12b` | Default chat model |
| `OLLAMA_EMBEDDING_MODEL` | No | `embeddinggemma:latest` | Embedding model |

**Setup**

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull gemma3:12b

# Verify it's running
curl http://localhost:11434/api/tags
```

**Example**

```rust,ignore
use edgequake_llm::OllamaProvider;

// Auto-detect from environment
let provider = OllamaProvider::from_env()?;

// Or use builder
let provider = OllamaProvider::builder()
    .host("http://localhost:11434")
    .model("mistral")
    .embedding_model("nomic-embed-text")
    .build()?;
```

### LM Studio

Local OpenAI-compatible API. Supports model loading and management.

**Environment Variables**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LMSTUDIO_HOST` | No | `http://localhost:1234` | Server URL |
| `LMSTUDIO_MODEL` | No | `gemma2-9b-it` | Chat model |
| `LMSTUDIO_EMBEDDING_MODEL` | No | `nomic-embed-text-v1.5` | Embedding model |
| `LMSTUDIO_EMBEDDING_DIM` | No | `768` | Embedding dimension |

**Example**

```rust,ignore
use edgequake_llm::LMStudioProvider;

let provider = LMStudioProvider::builder()
    .host("http://localhost:1234")
    .model("mistral-7b-instruct")
    .build()?;
```

---

## IDE Integration

### VSCode Copilot

Integration with GitHub Copilot via the copilot-api proxy server.

**Setup**

```bash
# Clone and set up copilot-api proxy
cd copilot-api
bun install
bun run auth   # Authenticate with GitHub
bun run start  # Start proxy on localhost:4141
```

**Example**

```rust,ignore
use edgequake_llm::VsCodeCopilotProvider;

let provider = VsCodeCopilotProvider::new()
    .model("gpt-4o-mini")
    .build()?;

let response = provider.complete("Hello from Copilot").await?;
```

---

## Generic Provider

### OpenAI Compatible

Connects to any API following the OpenAI chat completions format.
Used internally by xAI and HuggingFace providers.

**Configuration** (via `models.yaml`)

```yaml
providers:
  - name: deepseek
    type: openai_compatible
    api_key_env: DEEPSEEK_API_KEY
    base_url: https://api.deepseek.com
    default_llm_model: deepseek-chat
    models:
      - name: deepseek-chat
        context_length: 128000
```

**Example**

```rust,ignore
use edgequake_llm::OpenAICompatibleProvider;

let provider = OpenAICompatibleProvider::new(
    "https://api.deepseek.com",
    "sk-...",
    "deepseek-chat",
);
```

---

## Testing

### Mock Provider

Returns configurable responses without making API calls. Essential for
unit and integration testing.

**Example**

```rust,ignore
use edgequake_llm::{MockProvider, LLMProvider, ChatMessage};
use edgequake_llm::providers::mock::MockResponse;

// Simple mock
let provider = MockProvider::new();
let response = provider.complete("test").await?;
assert_eq!(response.content, "Mock response");

// Custom responses
let provider = MockProvider::with_responses(vec![
    MockResponse::new("First response"),
    MockResponse::new("Second response"),
]);
```

---

## Provider Selection

### Auto-Detection (ProviderFactory)

```text
  EDGEQUAKE_LLM_PROVIDER set?
    |
    +-- Yes --> Create specified provider
    |
    +-- No  --> Check environment variables:
                1. ANTHROPIC_API_KEY  --> Anthropic
                2. GEMINI_API_KEY     --> Gemini
                3. XAI_API_KEY        --> xAI
                4. OLLAMA_HOST/MODEL  --> Ollama
                5. OPENAI_API_KEY     --> OpenAI
                6. (none)             --> Mock
```

### Explicit Selection

```rust,ignore
use edgequake_llm::{ProviderFactory, ProviderType};

// Auto-detect
let (llm, embed) = ProviderFactory::from_env()?;

// Explicit
let (llm, embed) = ProviderFactory::create(ProviderType::Anthropic)?;
```

### Registry (Multi-Provider)

```rust,ignore
use edgequake_llm::ProviderRegistry;

let mut registry = ProviderRegistry::new();
registry.register_llm("fast", Arc::new(openai_provider));
registry.register_llm("smart", Arc::new(anthropic_provider));
registry.register_llm("local", Arc::new(ollama_provider));

// Dynamic selection
let provider = registry.get_llm("fast").unwrap();
```

## Adding Custom Providers

Implement `LLMProvider` and optionally `EmbeddingProvider`:

```rust,ignore
use edgequake_llm::traits::{LLMProvider, LLMResponse, ChatMessage, CompletionOptions};
use async_trait::async_trait;

struct MyProvider { /* ... */ }

#[async_trait]
impl LLMProvider for MyProvider {
    fn name(&self) -> &str { "my-provider" }
    fn model(&self) -> &str { "my-model" }
    fn max_context_length(&self) -> usize { 128_000 }

    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        // Your implementation
    }

    async fn chat(&self, messages: &[ChatMessage], options: Option<&CompletionOptions>) -> Result<LLMResponse> {
        // Your implementation
    }

    // Override capability flags
    fn supports_streaming(&self) -> bool { true }
    fn supports_function_calling(&self) -> bool { true }
}
```
