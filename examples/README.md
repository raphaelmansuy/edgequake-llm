# EdgeQuake LLM — Examples

Examples are organized by provider in subdirectories. Each is a self-contained
Rust binary: `cargo run --example <name>`.

```
examples/
├── openai/          # OpenAI API
├── azure/           # Azure OpenAI Service
├── gemini/          # Google Gemini (Google AI endpoint)
├── vertexai/        # Google Gemini (Vertex AI endpoint)
├── mistral/         # Mistral AI
├── local/           # Ollama / LM Studio (local inference)
└── advanced/        # Cross-provider: cost tracking, middleware, etc.
```

---

## Prerequisites

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Azure OpenAI
export AZURE_OPENAI_ENDPOINT="https://myresource.openai.azure.com"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o"
# optional:
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME="text-embedding-3-large"
export AZURE_OPENAI_API_VERSION="2024-10-21"

# Google AI (Gemini API)
export GEMINI_API_KEY="AIza..."

# Vertex AI
export GOOGLE_CLOUD_PROJECT="my-project-id"
export GOOGLE_CLOUD_REGION="us-central1"    # optional, default: us-central1
# authenticate once:
# gcloud auth login
# gcloud auth application-default login

# Mistral AI
export MISTRAL_API_KEY="..."

# Local providers — no key needed; start Ollama or LM Studio first
```

---

## OpenAI (`examples/openai/`)

| File | Binary name | Description |
|------|-------------|-------------|
| [demo.rs](openai/demo.rs) | `openai_demo` | Full walkthrough: completion, chat, streaming, tools |
| [basic_completion.rs](openai/basic_completion.rs) | `openai_basic_completion` | Minimal `complete()` call |
| [chatbot.rs](openai/chatbot.rs) | `openai_chatbot` | Multi-turn conversation with history |
| [embeddings.rs](openai/embeddings.rs) | `openai_embeddings` | Embeddings + cosine similarity |
| [streaming.rs](openai/streaming.rs) | `openai_streaming` | Real-time streaming token output |
| [tool_calling.rs](openai/tool_calling.rs) | `openai_tool_calling` | Function / tool calling |
| [vision.rs](openai/vision.rs) | `openai_vision` | Multimodal image analysis |

```bash
cargo run --example openai_demo
cargo run --example openai_basic_completion
cargo run --example openai_chatbot
cargo run --example openai_embeddings
cargo run --example openai_streaming
cargo run --example openai_tool_calling
cargo run --example openai_vision
```

---

## Azure OpenAI (`examples/azure/`)

| File | Binary name | Description |
|------|-------------|-------------|
| [env_check.rs](azure/env_check.rs) | `azure_env_check` | Verify `.env` / env-var loading |
| [full_demo.rs](azure/full_demo.rs) | `azure_full_demo` | Chat, streaming, embeddings, tool calling |

```bash
cargo run --example azure_env_check
cargo run --example azure_full_demo
```

---

## Google Gemini (`examples/gemini/`)

Uses the Google AI (`ai.google.dev`) endpoint. Requires `GEMINI_API_KEY`.

| File | Binary name | Description |
|------|-------------|-------------|
| [demo.rs](gemini/demo.rs) | `gemini_demo` | Full walkthrough: completion, chat, streaming, tools, vision, embeddings |
| [chat.rs](gemini/chat.rs) | `gemini_chat` | Q&A, system prompts, multi-turn, temperature sweep, JSON mode |
| [streaming.rs](gemini/streaming.rs) | `gemini_streaming` | Text stream, thinking content, tool-call deltas, TTFT |
| [vision.rs](gemini/vision.rs) | `gemini_vision` | Base64 PNG/JPEG, multi-image comparison, vision+JSON |
| [embeddings.rs](gemini/embeddings.rs) | `gemini_embeddings` | Single/batch, semantic search, similarity matrix, custom dims |
| [tool_calling.rs](gemini/tool_calling.rs) | `gemini_tool_calling` | Single tool, multi-tool, forced choice, multi-step |

```bash
export GEMINI_API_KEY="AIza..."
cargo run --example gemini_demo
cargo run --example gemini_chat
cargo run --example gemini_streaming
cargo run --example gemini_vision
cargo run --example gemini_embeddings
cargo run --example gemini_tool_calling
```

---

## Vertex AI (`examples/vertexai/`)

Uses the Google Cloud Vertex AI endpoint. Requires `GOOGLE_CLOUD_PROJECT` and
a valid gcloud session (`gcloud auth login` or `gcloud auth application-default login`).

| File | Binary name | Description |
|------|-------------|-------------|
| [demo.rs](vertexai/demo.rs) | `vertexai_demo` | Full walkthrough: completion, chat, streaming, tools, vision, embeddings, thinking |
| [chat.rs](vertexai/chat.rs) | `vertexai_chat` | Conversations, personas, temperature sweep, JSON mode |
| [streaming.rs](vertexai/streaming.rs) | `vertexai_streaming` | Text stream, thinking content, tool-call deltas, TTFT |
| [vision.rs](vertexai/vision.rs) | `vertexai_vision` | Base64 PNG/JPEG, multi-image, vision+JSON, model selection |
| [embeddings.rs](vertexai/embeddings.rs) | `vertexai_embeddings` | Single/batch via `:predict`, semantic search, custom dims |
| [tool_calling.rs](vertexai/tool_calling.rs) | `vertexai_tool_calling` | Single/multi tools, forced choice, multi-step with result feeding |

```bash
export GOOGLE_CLOUD_PROJECT="my-project-id"
gcloud auth login                      # one-time interactive auth
cargo run --example vertexai_demo
cargo run --example vertexai_chat
cargo run --example vertexai_streaming
cargo run --example vertexai_vision
cargo run --example vertexai_embeddings
cargo run --example vertexai_tool_calling
```

---

## Mistral (`examples/mistral/`)

| File | Binary name | Description |
|------|-------------|-------------|
| [chat.rs](mistral/chat.rs) | `mistral_chat` | Chat, streaming, embeddings, model listing |

```bash
cargo run --example mistral_chat
```

---

## Local Inference (`examples/local/`)

Requires Ollama (`ollama serve`) or LM Studio running locally. No API key needed.

| File | Binary name | Description |
|------|-------------|-------------|
| [local_llm.rs](local/local_llm.rs) | `local_llm` | Ollama + LM Studio usage |

```bash
cargo run --example local_llm
```

---

## Advanced (`examples/advanced/`)

Cross-provider patterns and infrastructure.

| File | Binary name | Description |
|------|-------------|-------------|
| [cost_tracking.rs](advanced/cost_tracking.rs) | `cost_tracking` | Session-level cost budgets |
| [middleware.rs](advanced/middleware.rs) | `middleware` | Logging, metrics, custom middleware |
| [multi_provider.rs](advanced/multi_provider.rs) | `multi_provider` | Provider-agnostic abstraction |
| [reranking.rs](advanced/reranking.rs) | `reranking` | BM25 document reranking (no API needed) |
| [retry_handling.rs](advanced/retry_handling.rs) | `retry_handling` | Retry strategies and error handling |

```bash
cargo run --example cost_tracking
cargo run --example middleware
cargo run --example multi_provider
cargo run --example reranking
cargo run --example retry_handling
```

---

## Related Documentation

- [Providers Guide](../docs/providers.md)
- [Provider Families](../docs/provider-families.md)
- [Architecture](../docs/architecture.md)
- [Reranking](../docs/reranking.md)


---

## Prerequisites

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Azure OpenAI
export AZURE_OPENAI_ENDPOINT="https://myresource.openai.azure.com"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o"
# optional:
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME="text-embedding-3-large"
export AZURE_OPENAI_API_VERSION="2024-10-21"

# Mistral AI
export MISTRAL_API_KEY="..."

# Local providers — no key needed; start Ollama or LM Studio first
```

---

## OpenAI (`examples/openai/`)

| File | Binary name | Description |
|------|-------------|-------------|
| [demo.rs](openai/demo.rs) | `openai_demo` | Full walkthrough: completion, chat, streaming, tools |
| [basic_completion.rs](openai/basic_completion.rs) | `openai_basic_completion` | Minimal `complete()` call |
| [chatbot.rs](openai/chatbot.rs) | `openai_chatbot` | Multi-turn conversation with history |
| [embeddings.rs](openai/embeddings.rs) | `openai_embeddings` | Embeddings + cosine similarity |
| [streaming.rs](openai/streaming.rs) | `openai_streaming` | Real-time streaming token output |
| [tool_calling.rs](openai/tool_calling.rs) | `openai_tool_calling` | Function / tool calling |
| [vision.rs](openai/vision.rs) | `openai_vision` | Multimodal image analysis |

```bash
cargo run --example openai_demo
cargo run --example openai_basic_completion
cargo run --example openai_chatbot
cargo run --example openai_embeddings
cargo run --example openai_streaming
cargo run --example openai_tool_calling
cargo run --example openai_vision
```

---

## Azure OpenAI (`examples/azure/`)

| File | Binary name | Description |
|------|-------------|-------------|
| [env_check.rs](azure/env_check.rs) | `azure_env_check` | Verify `.env` / env-var loading |
| [full_demo.rs](azure/full_demo.rs) | `azure_full_demo` | Chat, streaming, embeddings, tool calling |

```bash
cargo run --example azure_env_check
cargo run --example azure_full_demo
```

---

## Mistral (`examples/mistral/`)

| File | Binary name | Description |
|------|-------------|-------------|
| [chat.rs](mistral/chat.rs) | `mistral_chat` | Chat, streaming, embeddings, model listing |

```bash
cargo run --example mistral_chat
```

---

## Local Inference (`examples/local/`)

Requires Ollama (`ollama serve`) or LM Studio running locally. No API key needed.

| File | Binary name | Description |
|------|-------------|-------------|
| [local_llm.rs](local/local_llm.rs) | `local_llm` | Ollama + LM Studio usage |

```bash
cargo run --example local_llm
```

---

## Advanced (`examples/advanced/`)

Cross-provider patterns and infrastructure.

| File | Binary name | Description |
|------|-------------|-------------|
| [cost_tracking.rs](advanced/cost_tracking.rs) | `cost_tracking` | Session-level cost budgets |
| [middleware.rs](advanced/middleware.rs) | `middleware` | Logging, metrics, custom middleware |
| [multi_provider.rs](advanced/multi_provider.rs) | `multi_provider` | Provider-agnostic abstraction |
| [reranking.rs](advanced/reranking.rs) | `reranking` | BM25 document reranking (no API needed) |
| [retry_handling.rs](advanced/retry_handling.rs) | `retry_handling` | Retry strategies and error handling |

```bash
cargo run --example cost_tracking
cargo run --example middleware
cargo run --example multi_provider
cargo run --example reranking
cargo run --example retry_handling
```

---

## Related Documentation

- [Providers Guide](../docs/providers.md)
- [Provider Families](../docs/provider-families.md)
- [Architecture](../docs/architecture.md)
- [Reranking](../docs/reranking.md)
