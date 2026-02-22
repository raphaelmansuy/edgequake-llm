# EdgeQuake LLM — Examples

Examples are organized by provider in subdirectories. Each is a self-contained
Rust binary: `cargo run --example <name>`.

```
examples/
├── openai/          # OpenAI API
├── azure/           # Azure OpenAI Service
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
