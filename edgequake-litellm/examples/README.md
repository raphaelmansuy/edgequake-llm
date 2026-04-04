# edgequake-litellm Examples

A collection of runnable examples demonstrating `edgequake_litellm` capabilities.

## Setup

```bash
cd edgequake-litellm
pip install -e ".[dev]"
# or after publishing:
pip install edgequake-litellm
```

## Examples

| File | Description | Requires |
|------|-------------|---------|
| [01_basic_completion.py](01_basic_completion.py) | Sync completion across all providers | Any API key |
| [02_async_completion.py](02_async_completion.py) | Async completion + concurrent requests | `OPENAI_API_KEY` |
| [03_streaming.py](03_streaming.py) | Real-time streaming with any provider | Any API key |
| [04_tool_calling.py](04_tool_calling.py) | Function / tool calling with multi-turn | `OPENAI_API_KEY` or `MISTRAL_API_KEY` |
| [05_embeddings.py](05_embeddings.py) | Text embeddings + cosine similarity | `OPENAI_API_KEY` or Ollama |
| [06_drop_in_litellm_replacement.py](06_drop_in_litellm_replacement.py) | Migration from `litellm` | `OPENAI_API_KEY` |
| [07_ollama_local.py](07_ollama_local.py) | Local models via Ollama (no API key) | Ollama running |
| [08_multimodal_gemini.py](08_multimodal_gemini.py) | Vision / multimodal with Gemini | `GEMINI_API_KEY` |
| [09_azure_openai.py](09_azure_openai.py) | Azure OpenAI Service — chat, JSON mode, streaming | `AZURE_OPENAI_*` or `AZURE_OPENAI_CONTENTGEN_*` |

For providers without a dedicated example file yet, reuse `01_basic_completion.py`
with a different `provider/model` string:

| Provider | Example model string | Required environment |
|----------|----------------------|----------------------|
| Vertex AI | `vertexai/gemini-2.5-flash` | `GOOGLE_CLOUD_PROJECT` + ADC |
| OpenAI Compatible | `openai-compatible/deepseek-chat` | `OPENAI_COMPATIBLE_BASE_URL`, optional `OPENAI_COMPATIBLE_API_KEY` |
| VSCode Copilot | `vscode-copilot/gpt-4o-mini` | optional `VSCODE_COPILOT_PROXY_URL` |
| Bedrock | `bedrock/amazon.nova-lite-v1:0` | AWS credential chain |
| HuggingFace | `huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct` | `HF_TOKEN` |
| xAI | `xai/grok-4` | `XAI_API_KEY` |
| OpenRouter | `openrouter/meta-llama/llama-3.1-70b-instruct` | `OPENROUTER_API_KEY` |
| Mistral | `mistral/mistral-large-latest` | `MISTRAL_API_KEY` |
| Jina embeddings | `jina/jina-embeddings-v3` via `embedding()` | `JINA_API_KEY` |

## Quick Start

```bash
# Sync completion
OPENAI_API_KEY=sk-... python examples/01_basic_completion.py

# Async + concurrent
OPENAI_API_KEY=sk-... python examples/02_async_completion.py

# Streaming
OPENAI_API_KEY=sk-... python examples/03_streaming.py

# Tool calling
OPENAI_API_KEY=sk-... python examples/04_tool_calling.py

# Embeddings
OPENAI_API_KEY=sk-... python examples/05_embeddings.py

# Drop-in litellm replacement
OPENAI_API_KEY=sk-... python examples/06_drop_in_litellm_replacement.py

# Local Ollama (no API key)
ollama pull llama3.2 nomic-embed-text
python examples/07_ollama_local.py

# Gemini multimodal
GEMINI_API_KEY=... python examples/08_multimodal_gemini.py

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com \
AZURE_OPENAI_API_KEY=... \
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o \
    python examples/09_azure_openai.py
```

## Drop-in Migration from litellm

Change **one line** in your existing code:

```python
# Before
import litellm

# After
import edgequake_litellm as litellm
```

All `litellm.completion()`, `litellm.acompletion()`, `litellm.stream()`,
`litellm.embedding()`, and exception types remain compatible.

## Supported Providers

| Provider | Chat | Stream | Embed | Tool Calls |
|----------|------|--------|-------|------------|
| OpenAI | ✅ | ✅ | ✅ | ✅ |
| Anthropic | ✅ | ✅ | — | ✅ |
| Google Gemini | ✅ | ✅ | — | ✅ |
| Mistral | ✅ | ✅ | ✅ | ✅ |
| xAI (Grok) | ✅ | ✅ | — | — |
| OpenRouter | ✅ | ✅ | — | depends |
| Ollama (local) | ✅ | ✅ | ✅ | depends |
| Azure OpenAI | ✅ | ✅ | ✅ | ✅ |
| HuggingFace | ✅ | — | — | — |
| Vertex AI | ✅ | ✅ | ✅ | ✅ |
| OpenAI Compatible | ✅ | ✅ | ✅ | ✅ |
| VSCode Copilot | ✅ | ✅ | ✅ | ✅ |
| AWS Bedrock | ✅ | ✅ | ✅ | ✅ |
| Jina | — | — | ✅ | — |
