# edgequake-litellm

[![PyPI](https://img.shields.io/pypi/v/edgequake-litellm)](https://pypi.org/project/edgequake-litellm/)
[![Python](https://img.shields.io/pypi/pyversions/edgequake-litellm)](https://pypi.org/project/edgequake-litellm/)
[![Python CI](https://github.com/raphaelmansuy/edgequake-llm/actions/workflows/python-ci.yml/badge.svg)](https://github.com/raphaelmansuy/edgequake-llm/actions/workflows/python-ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](../LICENSE-APACHE)

`edgequake-litellm` is a LiteLLM-compatible Python package backed by the Rust `edgequake-llm` core. The intent is simple: keep the LiteLLM call shape, replace the Python network path with a native implementation, and preserve operational features such as streaming, tool calling, embeddings, and provider routing.

```python
# Before
import litellm

# After
import edgequake_litellm as litellm
```

## Install

```bash
pip install edgequake-litellm
```

Supported wheel targets:

| Platform | Architectures |
|----------|---------------|
| Linux (glibc) | `x86_64`, `aarch64` |
| Linux (musl) | `x86_64`, `aarch64` |
| macOS | `x86_64`, `arm64` |
| Windows | `x86_64` |

The package uses `abi3-py39`, so one wheel per platform covers Python 3.9+.

Scope note: this package covers the LiteLLM-compatible chat and embedding API
surface. The Rust crate also ships image-generation providers, but those APIs
are not exposed through `edgequake-litellm` yet.

## Quick Start

```python
import asyncio
import edgequake_litellm as litellm

messages = [{"role": "user", "content": "Explain Rust ownership in one sentence."}]

# Sync
resp = litellm.completion("openai/gpt-4o-mini", messages, max_tokens=128)
print(resp.choices[0].message.content)

# Async
async def main() -> None:
    resp = await litellm.acompletion("anthropic/claude-3-5-haiku-20241022", messages)
    print(resp.content)

    stream = await litellm.acompletion("openai/gpt-4o-mini", messages, stream=True)
    async for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)

asyncio.run(main())
```

Embeddings:

```python
import edgequake_litellm as litellm

result = litellm.embedding(
    "openai/text-embedding-3-small",
    ["hello world", "rust is fast"],
)

print(result.data[0].embedding[:3])
print(len(result[0]))
```

## Provider Routing

Pass `provider/model` as the `model` argument:

| Provider | Example |
|----------|---------|
| OpenAI | `openai/gpt-4o-mini` |
| Azure OpenAI | `azure/my-gpt4o-deployment` |
| Anthropic | `anthropic/claude-3-5-sonnet-20241022` |
| Gemini | `gemini/gemini-2.5-flash` |
| Vertex AI | `vertexai/gemini-2.5-flash` |
| xAI | `xai/grok-4` |
| OpenRouter | `openrouter/meta-llama/llama-3.1-70b-instruct` |
| Mistral | `mistral/mistral-large-latest` |
| AWS Bedrock | `bedrock/amazon.nova-lite-v1:0` |
| HuggingFace | `huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct` |
| OpenAI Compatible | `openai-compatible/deepseek-chat` |
| Ollama | `ollama/llama3.2` |
| LM Studio | `lmstudio/local-model` |
| VSCode Copilot | `vscode-copilot/auto` |
| Mock | `mock/test-model` |

Embedding-only backend:

| Provider | Example |
|----------|---------|
| Jina | `jina/jina-embeddings-v3` |

## Supported Features

| Provider | Chat | Stream | Tools | Embeddings | Notes |
|----------|------|--------|-------|------------|-------|
| OpenAI | Yes | Yes | Yes | Yes | includes `max_completion_tokens` handling |
| Azure OpenAI | Yes | Yes | Yes | Yes | deployment-based routing |
| Anthropic | Yes | Yes | Yes | No | Claude extended thinking surfaced in response metadata |
| Gemini | Yes | Yes | Yes | Yes | Google AI Studio |
| Vertex AI | Yes | Yes | Yes | Yes | GCP auth / ADC |
| xAI | Yes | Yes | Yes | No | Grok |
| OpenRouter | Yes | Yes | Yes | No | gateway models |
| Mistral | Yes | Yes | Yes | Yes | native embeddings |
| AWS Bedrock | Yes | Yes | Yes | Yes | backed by the Rust Bedrock feature |
| HuggingFace | Yes | Yes | Limited | No | Inference API |
| OpenAI Compatible | Yes | Yes | Yes | Yes | Groq, Together, DeepSeek, custom gateways |
| Ollama | Yes | Yes | Yes | Yes | local runtime |
| LM Studio | Yes | Yes | Yes | Yes | local OpenAI-compatible server |
| VSCode Copilot | Yes | Yes | Yes | Yes | direct auth by default, proxy optional |
| Jina | No | No | No | Yes | embeddings only |
| Mock | Yes | No | Yes | Yes | unit tests / local development |

## Environment Setup

| Provider | Required environment |
|----------|----------------------|
| OpenAI | `OPENAI_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT_NAME` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| Vertex AI | `GOOGLE_CLOUD_PROJECT` and ADC or `GOOGLE_ACCESS_TOKEN` |
| xAI | `XAI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| AWS Bedrock | standard AWS credential chain plus `AWS_REGION` |
| HuggingFace | `HF_TOKEN` or `HUGGINGFACE_TOKEN` |
| OpenAI Compatible | `OPENAI_COMPATIBLE_BASE_URL`, optional `OPENAI_COMPATIBLE_API_KEY` |
| Ollama | optional `OLLAMA_HOST` |
| LM Studio | optional `LMSTUDIO_HOST` |
| VSCode Copilot | optional `VSCODE_COPILOT_PROXY_URL`; otherwise reuse the official VS Code Copilot auth cache |
| Jina | `JINA_API_KEY` |

Module defaults:

```python
import edgequake_litellm as litellm

litellm.set_default_provider("anthropic")
litellm.set_default_model("claude-3-5-haiku-20241022")
```

Environment defaults:

- `LITELLM_EDGE_PROVIDER`
- `LITELLM_EDGE_MODEL`
- `LITELLM_EDGE_TIMEOUT`
- `LITELLM_EDGE_MAX_RETRIES`
- `LITELLM_EDGE_VERBOSE`

## LiteLLM Compatibility

Implemented:

- `completion()`
- `acompletion()`
- `embedding()`
- `aembedding()`
- `stream=True` on `acompletion()`
- `stream()` async generator
- `response.choices[0].message.content`
- `response.to_dict()`
- `AuthenticationError`, `RateLimitError`, `NotFoundError`, `Timeout`
- module globals `set_verbose` and `drop_params`

Behavior notes:

- synchronous streaming is intentionally not supported; use `acompletion(..., stream=True)` or `stream()`
- unsupported or extra keyword arguments are dropped for LiteLLM parity
- per-call `api_key`, `api_base`, and `timeout` parameters are accepted at the Python layer but not yet wired into the Rust core for every provider

## Provider Examples

OpenAI-compatible custom gateway:

```bash
export OPENAI_COMPATIBLE_BASE_URL=https://api.groq.com/openai/v1
export OPENAI_COMPATIBLE_API_KEY=...
```

```python
import edgequake_litellm as litellm

resp = litellm.completion(
    "openai-compatible/llama-3.3-70b-versatile",
    [{"role": "user", "content": "Write a one-line changelog summary."}],
)
print(resp.content)
```

Vertex AI:

```bash
export GOOGLE_CLOUD_PROJECT=my-project
gcloud auth application-default login
```

```python
resp = litellm.completion(
    "vertexai/gemini-2.5-flash",
    [{"role": "user", "content": "Summarise this design review."}],
)
```

Jina embeddings:

```python
import edgequake_litellm as litellm

vectors = litellm.embedding(
    "jina/jina-embeddings-v3",
    ["retrieval query", "retrieval document"],
)
print(len(vectors[0]))
```

## Development

```bash
git clone https://github.com/raphaelmansuy/edgequake-llm.git
cd edgequake-llm/edgequake-litellm

python -m venv .venv
source .venv/bin/activate

pip install "maturin>=1.7" "pytest>=8" "pytest-asyncio>=0.24" "ruff>=0.3" "mypy>=1.8"
pip install . -v

pytest -q -k "not e2e"
ruff check python/
mypy python/edgequake_litellm --ignore-missing-imports
```

## Release

Release tags are separate from the Rust crate:

- Rust crate: `vX.Y.Z`
- Python package: `py-vX.Y.Z`

Publish flow for `edgequake-litellm`:

1. bump `edgequake-litellm/Cargo.toml`
2. bump `edgequake-litellm/pyproject.toml`
3. update [`CHANGELOG.md`](CHANGELOG.md)
4. push the release-prep commit
5. wait for `python-ci.yml` to go green
6. push `py-vX.Y.Z`

`python-publish.yml` builds the sdist and wheels, smoke-tests the native wheels, publishes to PyPI, and can attach built artifacts to the GitHub Release.

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md) for the current release line and published history.

## License

Apache-2.0. See [`../LICENSE-APACHE`](../LICENSE-APACHE).
