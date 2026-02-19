# LiteLLM Deep-Dive: API Surface & Compatibility Targets

## 1. What LiteLLM Is

LiteLLM (v1.81+, ~36k GitHub stars) is the de-facto Python "universal LLM router":
- Calls 100+ providers through a single `completion()` function
- Normalises all responses to an OpenAI-compatible shape
- Provides a proxy server (FastAPI-based) with virtual keys, budgets, routing
- 8 ms P95 latency at 1k RPS on the proxy (pure Python, aiohttp)

Our goal: **replace the inner HTTP layer with edgequake-llm** to cut that 8 ms to
~1–2 ms for the SDK path, while keeping the Python API identical.

---

## 2. Core Public API Surface

### 2.1 Completion

```python
# Synchronous
response = litellm.completion(
    model="openai/gpt-4o",          # "provider/model" routing
    messages=[{"role": "user", "content": "hi"}],
    # --- optional params ---
    temperature=0.7,
    max_tokens=1024,
    stream=False,
    tools=[...],
    tool_choice="auto",
    response_format={"type": "json_object"},
    api_key="...",                   # override global key
    api_base="...",                  # override endpoint
    timeout=30,
    num_retries=3,
    metadata={},                     # pass-through
    # many more...
)

# Async
response = await litellm.acompletion(...)

# Streaming (sync)
for chunk in litellm.completion(..., stream=True):
    print(chunk.choices[0].delta.content)

# Streaming (async)
async for chunk in await litellm.acompletion(..., stream=True):
    ...
```

### 2.2 Embedding

```python
response = litellm.embedding(
    model="openai/text-embedding-3-small",
    input=["text1", "text2"],
    dimensions=1536,
    api_key="...",
)

response = await litellm.aembedding(...)
```

### 2.3 Other Endpoints (lower priority)

```python
litellm.image_generation(model=..., prompt=...)
litellm.transcription(model="openai/whisper-1", file=...)
litellm.rerank(model="cohere/rerank-english-v3.0", query=..., documents=[...])
litellm.text_completion(model=..., prompt=...)
```

---

## 3. Response Objects

### ModelResponse (completion)

```
ModelResponse
├── id: str                         # "chatcmpl-xxxx"
├── object: str                     # "chat.completion"
├── created: int                    # unix timestamp
├── model: str                      # model actually used
├── choices: List[Choices]
│   └── Choices
│       ├── finish_reason: str      # "stop"|"length"|"tool_calls"
│       ├── index: int
│       └── message: Message
│           ├── role: str           # "assistant"
│           ├── content: str|None
│           └── tool_calls: List[ChatCompletionMessageToolCall]|None
└── usage: Usage
    ├── prompt_tokens: int
    ├── completion_tokens: int
    └── total_tokens: int
```

### EmbeddingResponse

```
EmbeddingResponse
├── object: str                     # "list"
├── data: List[Embedding]
│   └── Embedding
│       ├── object: str             # "embedding"
│       ├── embedding: List[float]
│       └── index: int
├── model: str
└── usage: Usage
```

### StreamingChunk (delta)

```
ModelResponse (streaming)
└── choices[0]
    └── delta
        ├── content: str|None
        ├── role: str|None
        └── tool_calls: List[ChoiceDeltaToolCall]|None
```

---

## 4. Model Routing Scheme

LiteLLM uses `provider/model` strings:

```
"openai/gpt-4o"
"anthropic/claude-opus-4-5"
"gemini/gemini-2.0-flash"
"mistral/mistral-large-latest"
"ollama/llama3.3"
"xai/grok-3"
"openrouter/meta-llama/llama-3.3-70b-instruct"
"azure/gpt-4o"                      # Azure OpenAI deployment
"lm_studio/llama-3.1-8b"
```

The router uses `get_llm_provider(model)` to split on `/` and map to
the correct provider class.

---

## 5. Exception Hierarchy

```
litellm.exceptions.AuthenticationError
litellm.exceptions.BadRequestError
litellm.exceptions.RateLimitError
litellm.exceptions.ServiceUnavailableError
litellm.exceptions.ContextWindowExceededError
litellm.exceptions.ContentPolicyViolationError
litellm.exceptions.Timeout
litellm.exceptions.APIConnectionError
litellm.exceptions.InternalServerError
```

All exceptions inherit from `openai.OpenAIError` for drop-in compatibility.

---

## 6. Configuration Model

```python
import litellm

# Global keys
litellm.openai_key = "sk-..."
litellm.anthropic_key = "sk-ant-..."

# Retries / fallbacks
litellm.num_retries = 3
litellm.fallbacks = [
    {"openai/gpt-4o": ["anthropic/claude-opus-4-5"]},
]

# Callbacks (observability)
litellm.success_callback = ["langfuse", "prometheus"]
litellm.failure_callback = ["slack"]

# Caching
litellm.cache = Cache(type="redis", host="...", port=6379)

# Budget
litellm.max_budget = 10.0   # USD
```

---

## 7. Providers Currently in edgequake-llm

Cross-reference with LiteLLM coverage:

```
Provider          edgequake-llm   LiteLLM   Priority
─────────────────────────────────────────────────────
OpenAI            ✓               ✓         P0
Azure OpenAI      ✓               ✓         P0
Anthropic         ✓               ✓         P0
Gemini            ✓               ✓         P0
Mistral           ✓               ✓         P0
Ollama            ✓               ✓         P0
LM Studio         ✓               ✓         P1
OpenRouter        ✓               ✓         P1
xAI (Grok)        ✓               ✓         P1
Jina (embed)      ✓               ✓         P2
Hugging Face      ✗               ✓         P2
Cohere            ✗               ✓         P2
Bedrock           ✗               ✓         P3
Vertex AI         ✗               ✓         P3
```

---

## 8. Key LiteLLM Internals We Replace

```
LiteLLM Call Path (current)
────────────────────────────
completion()
  → get_llm_provider()
  → _generate_config()
  → ProviderClass.completion()    ← pure Python HTTP (aiohttp/httpx)
  → transform_response()          ← Python dict manipulation
  → ModelResponse(...)            ← Pydantic validation

edgequake-python Call Path
───────────────────────────
completion()
  → parse_model_string()          ← Python (fast)
  → _eq_core.complete()          ← Rust (PyO3 call, GIL released)
      → edgequake provider        ← reqwest HTTP
      → LLMResponse struct        ← Rust serde
  → _to_model_response()          ← Python (thin dict wrap)
  → ModelResponse(...)
```

---

## 9. Compatibility Guarantees We Target

- **Must match**: `ModelResponse` fields, exception types, streaming iteration protocol
- **Must match**: `completion(model=..., messages=..., **kwargs)` signature
- **Best effort**: All optional params (`temperature`, `tools`, `response_format`, ...)
- **Out of scope** (v1): `Router`, `BudgetManager`, proxy server, callbacks/logging
- **Out of scope** (v1): providers not in edgequake-llm (Bedrock, HuggingFace, etc.)
