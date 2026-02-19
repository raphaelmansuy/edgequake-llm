# edgequake-python

**High-performance, LiteLLM-compatible Python LLM library backed by Rust.**

`edgequake-python` wraps the [`edgequake-llm`](https://crates.io/crates/edgequake-llm) Rust crate via [PyO3](https://pyo3.rs/), providing a drop-in replacement for [LiteLLM](https://github.com/BerriAI/litellm) with dramatically lower overhead for latency-sensitive applications.

## Features

- **LiteLLM-compatible API** — `completion()`, `acompletion()`, `stream()`, `embedding()`, etc.
- **Multi-provider routing** — OpenAI, Anthropic, Gemini, Mistral, OpenRouter, xAI, Ollama, LM Studio, HuggingFace, and more, via `provider/model` string.
- **Async-native** — built on Tokio; both sync and async Python are supported.
- **Single wheel per platform** — uses PyO3's `abi3-py39` stable ABI, one `.whl` covers Python 3.9–3.13+.
- **Zero Python dependencies** — the Rust extension is self-contained.
- **Full type annotations** — ships with `py.typed` and `.pyi` stubs.

## Installation

```bash
pip install edgequake-python
```

## Quick Start

```python
from edgequake_python import completion, acompletion, embedding, stream

# ── Synchronous chat ────────────────────────────────────────────────────────
resp = completion(
    "openai/gpt-4o-mini",
    [{"role": "user", "content": "Hello, world!"}],
)
print(resp.content)
print(resp.usage.total_tokens)

# ── Asynchronous chat ───────────────────────────────────────────────────────
import asyncio

async def main():
    resp = await acompletion(
        "anthropic/claude-3-5-haiku-20241022",
        [{"role": "user", "content": "Tell me a joke."}],
        max_tokens=128,
        temperature=0.8,
    )
    print(resp.content)

asyncio.run(main())

# ── Streaming ───────────────────────────────────────────────────────────────
async def stream_example():
    async for chunk in stream("gemini/gemini-2.0-flash", messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)

# ── Embeddings ──────────────────────────────────────────────────────────────
vectors = embedding(
    "openai/text-embedding-3-small",
    ["Hello world", "Rust is fast"],
)
print(len(vectors), len(vectors[0]))  # 2 1536
```

## Provider Routing

Pass `provider/model` as the first argument — the prefix is the provider name:

| Provider     | Example model string                           |
|--------------|------------------------------------------------|
| OpenAI       | `openai/gpt-4o`                               |
| Anthropic    | `anthropic/claude-3-5-sonnet-20241022`         |
| Google Gemini| `gemini/gemini-2.0-flash`                     |
| Mistral      | `mistral/mistral-large-latest`                 |
| OpenRouter   | `openrouter/meta-llama/llama-3.1-70b-instruct` |
| xAI          | `xai/grok-beta`                               |
| Ollama       | `ollama/llama3.2`                             |
| LM Studio    | `lmstudio/local-model`                        |
| HuggingFace  | `huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1` |
| Mock (tests) | `mock/any-name`                               |

## API Reference

### `completion(model, messages, **kwargs) → ModelResponse`

Synchronous completion. Blocks the calling thread but releases the Python GIL
during I/O (other Python threads keep running).

```python
resp = completion(
    "openai/gpt-4o",
    messages,
    max_tokens=256,
    temperature=0.7,
    system="You are a helpful assistant.",
    tools=[...],          # optional tool definitions
    tool_choice="auto",   # optional tool choice
)
```

### `acompletion(model, messages, **kwargs) → Awaitable[ModelResponse]`

Async completion. Use inside `async def` functions:

```python
resp = await acompletion("anthropic/claude-3-5-haiku-20241022", messages)
```

### `stream(model, messages, **kwargs) → AsyncGenerator[StreamChunk, None]`

Streaming completion. Yields `StreamChunk` objects:

```python
async for chunk in stream("openai/gpt-4o", messages):
    if chunk.content:
        print(chunk.content, end="")
    elif chunk.is_finished:
        print(f"\n[stop: {chunk.finish_reason}]")
```

### `embedding(model, input) → List[List[float]]`

Synchronous embedding:

```python
vecs = embedding("openai/text-embedding-3-small", ["foo", "bar"])
```

### `aembedding(model, input) → Awaitable[List[List[float]]]`

Async embedding:

```python
vecs = await aembedding("openai/text-embedding-3-small", ["foo", "bar"])
```

## Configuration

```python
from edgequake_python import set_default_provider, set_default_model

set_default_provider("anthropic")
set_default_model("claude-3-5-haiku-20241022")

# Now you can omit the provider prefix:
resp = completion("claude-3-5-haiku-20241022", messages)
```

## Exception Hierarchy

Exceptions mirror LiteLLM for painless migration:

```python
from edgequake_python import (
    EdgeQuakeError,           # base
    AuthenticationError,      # HTTP 401/403
    RateLimitError,           # HTTP 429
    ContextWindowExceededError,
    ModelNotFoundError,
    Timeout,
    APIConnectionError,
    APIError,
)

try:
    resp = completion("openai/gpt-4o", messages)
except AuthenticationError as e:
    print(f"Check your API key: {e}")
except RateLimitError:
    time.sleep(10)
    retry()
```

## Environment Variables

Provider credentials are read from the standard environment variables:

| Provider     | Environment variable     |
|--------------|--------------------------|
| OpenAI       | `OPENAI_API_KEY`         |
| Anthropic    | `ANTHROPIC_API_KEY`      |
| Gemini       | `GEMINI_API_KEY`         |
| Mistral      | `MISTRAL_API_KEY`        |
| OpenRouter   | `OPENROUTER_API_KEY`     |
| xAI          | `XAI_API_KEY`            |
| HuggingFace  | `HF_TOKEN`               |
| Ollama       | `OLLAMA_HOST` (default: `http://localhost:11434`) |
| LM Studio    | `LMSTUDIO_HOST` (default: `http://localhost:1234`) |

## Development

### Build from source

You need Rust (≥ 1.83) and Python (≥ 3.9):

```bash
# Clone the repo
git clone https://github.com/raphaelmansuy/edgequake-llm.git
cd edgequake-llm/edgequake-python

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install maturin
pip install maturin pytest pytest-asyncio

# Build and install in dev mode (incremental)
maturin develop --release

# Run tests (mock provider — no API keys needed)
pytest tests/ -k "not e2e" -v
```

### Running e2e tests

```bash
export OPENAI_API_KEY=sk-...
pytest tests/test_e2e_openai.py -v
```

## License

Apache-2.0 — see [LICENSE-APACHE](../LICENSE-APACHE).
