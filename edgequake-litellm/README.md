# edgequake-litellm

**Drop-in LiteLLM replacement backed by Rust — same API, lower overhead.**

[![PyPI](https://img.shields.io/pypi/v/edgequake-litellm)](https://pypi.org/project/edgequake-litellm/)
[![Python](https://img.shields.io/pypi/pyversions/edgequake-litellm)](https://pypi.org/project/edgequake-litellm/)
[![CI](https://github.com/raphaelmansuy/edgequake-llm/actions/workflows/python-ci.yml/badge.svg)](https://github.com/raphaelmansuy/edgequake-llm/actions/workflows/python-ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](../LICENSE-APACHE)

`edgequake-litellm` wraps the [`edgequake-llm`](https://crates.io/crates/edgequake-llm) Rust core via [PyO3](https://pyo3.rs/), providing a high-performance drop-in for [LiteLLM](https://github.com/BerriAI/litellm). Swap the import — the rest of your code stays unchanged.

```python
# Before
import litellm

# After — same API, Rust-backed
import edgequake_litellm as litellm
```

## Features

- **LiteLLM-compatible API** — `completion()`, `acompletion()`, `stream()`, `embedding()`, same call signatures, same response shape (`resp.choices[0].message.content`).
- **Multi-provider routing** — OpenAI, Anthropic, Gemini, Mistral, OpenRouter, xAI, Ollama, LM Studio, HuggingFace, and more, via `provider/model` strings.
- **Async-native** — built on Tokio; sync and async Python both supported.
- **Single wheel per platform** — uses PyO3's `abi3-py39` stable ABI, one `.whl` covers Python 3.9–3.13+.
- **Zero Python runtime dependencies** — the Rust extension is self-contained.
- **Full type annotations** — ships with `py.typed` and `.pyi` stubs.

## Installation

```bash
pip install edgequake-litellm
```

## Quick Start

```python
import edgequake_litellm as litellm   # drop-in import alias

# ── Synchronous chat ────────────────────────────────────────────────────────
resp = litellm.completion(
    "openai/gpt-4o-mini",
    [{"role": "user", "content": "Hello, world!"}],
)
# litellm-compatible access
print(resp.choices[0].message.content)
# convenience shortcut
print(resp.content)

# ── Asynchronous chat ───────────────────────────────────────────────────────
import asyncio

async def main():
    resp = await litellm.acompletion(
        "anthropic/claude-3-5-haiku-20241022",
        [{"role": "user", "content": "Tell me a joke."}],
        max_tokens=128,
        temperature=0.8,
    )
    print(resp.choices[0].message.content)

asyncio.run(main())

# ── Streaming (async generator) ─────────────────────────────────────────────
async def stream_example():
    messages = [{"role": "user", "content": "Count to five."}]
    async for chunk in litellm.acompletion("openai/gpt-4o", messages, stream=True):
        print(chunk.choices[0].delta.content or "", end="", flush=True)

# ── Embeddings ──────────────────────────────────────────────────────────────
result = litellm.embedding(
    "openai/text-embedding-3-small",
    ["Hello world", "Rust is fast"],
)
# litellm-compatible access
print(result.data[0].embedding[:3])
# legacy list access still works
print(len(result), len(result[0]))  # 2 1536
```

## Provider Routing

Pass `provider/model` as the first argument — the prefix selects the provider:

| Provider     | Example model string                                |
|--------------|-----------------------------------------------------|
| OpenAI       | `openai/gpt-4o`                                    |
| Anthropic    | `anthropic/claude-3-5-sonnet-20241022`              |
| Google Gemini| `gemini/gemini-2.0-flash`                          |
| Mistral      | `mistral/mistral-large-latest`                      |
| OpenRouter   | `openrouter/meta-llama/llama-3.1-70b-instruct`      |
| xAI          | `xai/grok-3-beta`                                  |
| Ollama       | `ollama/llama3.2`                                  |
| LM Studio    | `lmstudio/local-model`                             |
| HuggingFace  | `huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1` |
| Mock (tests) | `mock/any-name`                                    |

## API Reference

### `completion(model, messages, **kwargs) → ModelResponseCompat`

Synchronous chat completion. Blocks but releases the GIL during Rust I/O so other Python threads keep running.

```python
resp = litellm.completion(
    "openai/gpt-4o",
    messages,
    max_tokens=256,
    temperature=0.7,
    system="You are a helpful assistant.",
    max_completion_tokens=256,  # alias for max_tokens
    seed=42,
    response_format={"type": "json_object"},  # or "text" / "json_object"
)

# All of these access the same content:
resp.choices[0].message.content   # litellm path
resp.content                       # shortcut
resp["choices"][0]["message"]["content"]  # dict-style

resp.usage.total_tokens
resp.model
resp.response_ms   # latency in milliseconds
resp.to_dict()     # plain dict
```

### `acompletion(model, messages, stream=False, **kwargs)`

Async chat completion. Returns `ModelResponseCompat` or (if `stream=True`) `AsyncGenerator[StreamChunkCompat, None]`.

```python
# Non-streaming
resp = await litellm.acompletion("openai/gpt-4o", messages)

# Streaming
async for chunk in await litellm.acompletion("openai/gpt-4o", messages, stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```

### `stream(model, messages, **kwargs) → AsyncGenerator[StreamChunk, None]`

Low-level streaming. Raw `StreamChunk` objects:

```python
async for chunk in litellm.stream("openai/gpt-4o", messages):
    if chunk.content:
        print(chunk.content, end="")
    elif chunk.is_finished:
        print(f"\n[stop: {chunk.finish_reason}]")
```

### `embedding(model, input, **kwargs) → EmbeddingResponseCompat`

Synchronous embeddings. Returns an `EmbeddingResponseCompat` that supports both litellm-style and legacy list-style access:

```python
result = litellm.embedding("openai/text-embedding-3-small", ["foo", "bar"])

# litellm path
result.data[0].embedding

# backwards-compatible list access
for vec in result:          # iterates List[float]
    print(len(vec))
result[0]                   # List[float]
len(result)                 # number of vectors
```

### `aembedding(model, input, **kwargs) → EmbeddingResponseCompat`

Async embeddings — same return type as `embedding()`.

### `stream_chunk_builder(chunks, messages=None) → ModelResponseCompat`

Reconstruct a full `ModelResponseCompat` from a collected list of streaming chunks:

```python
from edgequake_litellm import stream_chunk_builder

chunks = []
async for chunk in litellm.stream("openai/gpt-4o", messages):
    chunks.append(chunk)

full = stream_chunk_builder(chunks, messages=messages)
print(full.content)
```

## Configuration

Module-level globals mirror `litellm`:

```python
import edgequake_litellm as litellm

litellm.set_verbose = True      # enable debug logging
litellm.drop_params = True      # drop unknown params (always True)

# Set default provider / model
litellm.set_default_provider("anthropic")
litellm.set_default_model("claude-3-5-haiku-20241022")

# Now the provider prefix can be omitted:
resp = litellm.completion("claude-3-5-haiku-20241022", messages)
```

## Exception Hierarchy

Exceptions mirror LiteLLM for painless migration:

```python
import edgequake_litellm as litellm

try:
    resp = litellm.completion("openai/gpt-4o", messages)
except litellm.AuthenticationError as e:
    print(f"Check your API key: {e}")
except litellm.RateLimitError:
    time.sleep(5)
except litellm.ContextWindowExceededError:
    # trim messages and retry
    pass
except litellm.NotFoundError:      # alias for ModelNotFoundError
    pass
except litellm.APIConnectionError:
    pass
```

All exceptions (`AuthenticationError`, `RateLimitError`, `ContextWindowExceededError`, `ModelNotFoundError`, `Timeout`, `APIConnectionError`, `APIError`) are also available from `edgequake_litellm.exceptions`.

## Environment Variables

Provider credentials follow the standard naming convention:

| Provider     | Environment variable                                      |
|--------------|-----------------------------------------------------------|
| OpenAI       | `OPENAI_API_KEY`                                         |
| Anthropic    | `ANTHROPIC_API_KEY`                                      |
| Gemini       | `GEMINI_API_KEY`                                         |
| Mistral      | `MISTRAL_API_KEY`                                        |
| OpenRouter   | `OPENROUTER_API_KEY`                                     |
| xAI          | `XAI_API_KEY`                                            |
| HuggingFace  | `HF_TOKEN`                                               |
| Ollama       | `OLLAMA_HOST` (default: `http://localhost:11434`)         |
| LM Studio    | `LMSTUDIO_HOST` (default: `http://localhost:1234`)        |

Defaults can also be set via `LITELLM_EDGE_PROVIDER` / `LITELLM_EDGE_MODEL`.

## Development

### Prerequisites

- Rust ≥ 1.83 (`rustup toolchain install stable`)
- Python ≥ 3.9
- `pip install maturin`

### Build from source

```bash
git clone https://github.com/raphaelmansuy/edgequake-llm.git
cd edgequake-llm/edgequake-litellm

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install maturin pytest pytest-asyncio ruff mypy

# Build & install in dev mode (incremental Rust + Python)
maturin develop --release

# Run unit tests (mock provider — no API keys needed)
pytest tests/ -k "not e2e" -v
```

### Running E2E tests

```bash
export OPENAI_API_KEY=sk-...
pytest tests/test_e2e_openai.py -v
```

### Publishing

```bash
# Bump version in pyproject.toml AND Cargo.toml (must match), then:
git tag py-v0.2.0
git push --tags
# GitHub Actions builds and publishes to PyPI automatically.
```

## License

Apache-2.0 — see [LICENSE-APACHE](../LICENSE-APACHE).

