# Architecture: edgequake-python

## 1. Bird's Eye View

```
╔══════════════════════════════════════════════════════════════════════════╗
║  USER CODE                                                               ║
║  import edgequake_python as litellm                                      ║
║  response = litellm.completion("openai/gpt-4o", messages=[...])         ║
╚════════════════════════════╤═════════════════════════════════════════════╝
                             │ Python function call
                             ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  PYTHON LAYER  edgequake_python/                                         ║
║  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐ ║
║  │completion│ │embedding │ │ router   │ │exceptions│ │  _types.py   │ ║
║  │ .py      │ │ .py      │ │ .py      │ │ .py      │ │(ModelResponse│ ║
║  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────────┘ │ Embedding    │ ║
║       │            │            │                      │ StreamChunk) │ ║
║       └────────────┴────────────┘                      └──────────────┘ ║
║                    │ calls _eq_core (Rust extension module)              ║
╚════════════════════╪═════════════════════════════════════════════════════╝
                     │ PyO3 FFI  (GIL released for Rust work)
                     ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  RUST EXTENSION  _eq_core.so  (compiled by maturin)                     ║
║                                                                          ║
║  python/mod.rs                                                           ║
║  ┌───────────────────────────────────────────────────────────────────┐  ║
║  │  PyCompletion        PyEmbedding       PyStreamIter               │  ║
║  │  complete()          embed()           __next__() / __anext__()   │  ║
║  └──────────────────────────┬────────────────────────────────────────┘  ║
║                             │                                            ║
║  bridge.rs — tokio runtime bridge                                        ║
║  ┌───────────────────────────────────────────────────────────────────┐  ║
║  │  block_on()  /  spawn() — maps async Rust → sync Python call     │  ║
║  │  AsyncPyIter — wraps BoxStream<StreamChunk> for Python iteration  │  ║
║  └──────────────────────────┬────────────────────────────────────────┘  ║
║                             │                                            ║
║  edgequake-llm  (existing crate)                                         ║
║  ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ ║
║  │ProviderFac │ │LLMProvid │ │EmbedProv │ │RateLimiter│ │CostTracker│ ║
║  │ tory       │ │er trait  │ │er trait  │ │           │ │           │ ║
║  └────────────┘ └──────────┘ └──────────┘ └──────────┘ └────────────┘ ║
║  ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ ║
║  │ OpenAI     │ │Anthropic │ │ Gemini   │ │ Mistral  │ │  Ollama   │ ║
║  │ Provider   │ │ Provider │ │ Provider │ │ Provider │ │  Provider │ ║
║  └────────────┘ └──────────┘ └──────────┘ └──────────┘ └────────────┘ ║
╚══════════════════════════════════════════════════════════════════════════╝
                             │  reqwest (Rust async HTTP, rustls)
                             ▼
              ┌──────────────────────────────┐
              │  LLM Provider Endpoints      │
              │  api.openai.com              │
              │  api.anthropic.com           │
              │  generativelanguage.google.* │
              │  api.mistral.ai              │
              │  localhost:11434 (Ollama)    │
              └──────────────────────────────┘
```

---

## 2. Module Decomposition

### 2.1 Python Layer (`python/edgequake_python/`)

```
edgequake_python/
│
├── __init__.py            [re-exports litellm-compatible API]
│   └── completion, acompletion, embedding, aembedding
│       AuthenticationError, RateLimitError, ...
│
├── _types.py              [Pydantic v2 models matching litellm types]
│   ├── Message
│   ├── ModelResponse
│   ├── Choices, Delta
│   ├── Usage
│   ├── EmbeddingResponse
│   └── StreamingModelResponse
│
├── completion.py          [completion() / acompletion()]
│   ├── _parse_model()     → (provider, model_name)
│   ├── _build_messages()  → List[dict] → Rust-serializable form
│   ├── _build_opts()      → CompletionOptions struct
│   └── _to_response()     → Rust LLMResponse → ModelResponse
│
├── embedding.py           [embedding() / aembedding()]
│   ├── _parse_model()
│   └── _to_embed_response()
│
├── streaming.py           [SyncStreamWrapper / AsyncStreamWrapper]
│   ├── SyncStreamWrapper  __iter__ → yields ModelResponse chunks
│   └── AsyncStreamWrapper __aiter__ → async yields
│
├── router.py              [Router class — v2 feature]
│   └── Router(model_list=[...])
│
├── exceptions.py          [maps Rust errors → litellm exceptions]
│   ├── AuthenticationError(OpenAIError)
│   ├── RateLimitError(OpenAIError)
│   ├── ContextWindowExceededError(OpenAIError)
│   └── _rust_error_to_python()
│
└── py.typed               [PEP 561 marker]
```

### 2.2 Rust Extension (`src/python/`)

```
src/python/
│
├── mod.rs                 [#[pymodule] _eq_core, registers all classes]
│
├── types.rs               [PyO3 struct mirrors for Rust types]
│   ├── PyLLMResponse      (#[pyclass] wraps LLMResponse)
│   ├── PyEmbeddingResult  (#[pyclass] wraps Vec<Vec<f32>>)
│   └── PyStreamChunk      (#[pyclass] wraps StreamChunk)
│
├── completion.rs          [PyO3 functions for completion]
│   ├── complete()         → Py<PyLLMResponse>
│   ├── astream()          → Py<PyAsyncStreamIter>
│   └── _into_opts()       → Python dict → CompletionOptions
│
├── embedding.rs           [PyO3 functions for embedding]
│   └── embed()            → Py<PyEmbeddingResult>
│
└── bridge.rs              [tokio runtime bridge]
    ├── RUNTIME: Lazy<Runtime>     global tokio runtime
    ├── block_on(fut)              run async Rust from sync PyO3 call
    └── PyAsyncStreamIter          wraps BoxStream, impl __anext__
```

---

## 3. Data Flow: Synchronous Completion

```
Python call:
  completion("openai/gpt-4o", messages=[...], temperature=0.7)

Step 1  [Python]  parse_model("openai/gpt-4o")
           → provider = "openai", model = "gpt-4o"

Step 2  [Python]  build_opts(temperature=0.7, max_tokens=None, ...)
           → dict {"temperature": 0.7, ...}

Step 3  [PyO3]    _eq_core.complete(provider, model, messages_json, opts_json)
           → GIL released
           → Rust: bridge::block_on(async { provider.chat(messages, opts) })
           → reqwest HTTP POST to api.openai.com
           → serde JSON decode → LLMResponse struct
           → GIL re-acquired
           → returns PyLLMResponse

Step 4  [Python]  _to_model_response(PyLLMResponse)
           → ModelResponse(
               id="chatcmpl-xxx",
               choices=[Choices(message=Message(role="assistant", content="...")),
               usage=Usage(prompt_tokens=10, ...))

Step 5  Return ModelResponse to caller
```

---

## 4. Data Flow: Streaming Completion

```
Python call:
  for chunk in completion("openai/gpt-4o", messages=[...], stream=True):

Step 1–2  Same as above

Step 3  [PyO3]   _eq_core.astream(provider, model, messages_json, opts_json)
            → returns PyAsyncStreamIter (wraps BoxStream<StreamChunk>)

Step 4  [Python] SyncStreamWrapper wraps PyAsyncStreamIter
            iter.__next__() calls bridge::next_chunk() which does:
            → GIL released
            → tokio runtime polls the stream for next chunk
            → GIL re-acquired
            → returns PyStreamChunk

Step 5  [Python] _chunk_to_model_response(PyStreamChunk)
            → ModelResponse with delta field populated

Step 6  Yield to caller
        (caller sees litellm-identical streaming chunks)
```

---

## 5. Threading Model

```
Python Thread (caller)
       │
       │  GIL held during Python setup code
       │
       ├─── PyO3 call ──────────────────────────────────────────────────┐
       │    (GIL released via allow_threads)                            │
       │                                                                │
       │    Rust side:                                                  │
       │    ┌─────────────────────────────────────────────────────┐    │
       │    │  TOKIO RUNTIME  (single global, multi-threaded)     │    │
       │    │                                                      │    │
       │    │  Task: provider.chat(messages, opts)                │    │
       │    │  ├── reqwest: connect (I/O)                        │    │
       │    │  ├── reqwest: write request                        │    │
       │    │  ├── reqwest: read response / stream               │    │
       │    │  └── serde: decode JSON                            │    │
       │    └─────────────────────────────────────────────────────┘    │
       │                                                                │
       └─── return LLMResponse ─────────────────────────────────────────┘
       │
       │  GIL re-acquired
       │  Python: convert to ModelResponse
       │  Return to caller
```

Key: **no Python threads are blocked during I/O** — tokio handles all async I/O
internally. The Python thread blocks on `block_on()` but does so with GIL released,
so other Python threads / asyncio tasks remain unblocked.

---

## 6. Async Python Integration

For `acompletion` / `aembedding`, we use PyO3's `pyo3_asyncio` bridge:

```
async def acompletion(...):

  Python asyncio event loop running in caller's thread
       │
       │  asyncio.get_event_loop().run_in_executor(None, _sync_call)
       │  OR:
       │  pyo3_asyncio_0_21::tokio::future_into_py(py, async_rust_fut)
       │
       ▼
  Rust future polled by tokio
  Result sent back via oneshot channel
  asyncio.Future resolved in Python event loop
```

---

## 7. Error Propagation

```
Rust LlmError                   Python Exception
─────────────────────────────────────────────────
LlmError::Authentication        AuthenticationError
LlmError::RateLimit             RateLimitError
LlmError::ContextWindow         ContextWindowExceededError
LlmError::InvalidRequest        BadRequestError
LlmError::Timeout               Timeout
LlmError::Network               APIConnectionError
LlmError::ServiceUnavailable    ServiceUnavailableError
LlmError::Other(msg)            APIError(msg)
```

Each Rust error is converted in `bridge.rs` via `impl From<LlmError> for PyErr`.

---

## 8. Provider Registry at Runtime

```
"openai/gpt-4o"
       │
       ▼ parse_model_string()
   provider = "openai"
   model    = "gpt-4o"
       │
       ▼ ProviderFactory::create(ProviderType::OpenAI, config)
   OpenAIProvider { api_key, base_url }
       │
       ▼ trait LLMProvider::chat(messages, opts)
   HTTP POST https://api.openai.com/v1/chat/completions
```

Provider config sourced (in order):

```
1. Explicit kwarg: completion(..., api_key="sk-...")
2. edgequake_python.openai_key = "sk-..."  (global Python config)
3. Environment variable: OPENAI_API_KEY
4. ~/.config/edgequake/config.toml  (future)
```
