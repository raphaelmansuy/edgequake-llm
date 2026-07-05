# Cross-Reference Capability Matrix — July 2026

> **Scope**: Feature-level cross-reference of edgequake-llm v0.8.0 against provider APIs.
>
> **Legend**: ✅ Implemented | ⚠️ Partial | ❌ Gap | ➖ N/A (provider doesn't offer)
>
> **Updated**: 2026-07-04 — All P0, P1, P2, and P3 gaps from audit have been closed.

---

## 1. LLM Chat/Completion

| Provider | Chat | Streaming | Tool Calling | Tool Streaming | Vision | JSON Mode | Thinking/Reasoning | Prompt Caching |
|----------|------|-----------|-------------|----------------|--------|-----------|-------------------|----------------|
| **OpenAI** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `reasoning_effort` | ✅ `cached_tokens` |
| **Azure OpenAI** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (via OpenAI) | ✅ |
| **Anthropic** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `thinking` blocks | ✅ `cache_control` |
| **Gemini** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `thinkingConfig` | ⚠️ (cachedContent API) |
| **Vertex AI** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (via Gemini) | ⚠️ |
| **Bedrock** | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ `thinking_budget` | ❌ |
| **Mistral** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `reasoning_effort` | ❌ |
| **xAI** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `reasoning.effort` | ❌ |
| **Cohere** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| **OpenRouter** | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ⚠️ | ❌ |
| **NVIDIA** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Ollama** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **LM Studio** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ |

### Gaps vs. Provider API Reality (post-implementation)

| Provider | API Has | edgequake-llm | Status |
|----------|---------|---------------|--------|
| Mistral | Vision (Medium 3.5) | ✅ Via inner OpenAICompatibleProvider | **CLOSED** |
| Mistral | `reasoning_effort` | ✅ | **CLOSED** |
| xAI | Reasoning (model variants) | ✅ (`reasoning.effort`) | **CLOSED** |
| Cohere | Full Chat API | ✅ | **CLOSED** |
| NVIDIA | Tool calling in `/v1/chat/completions` | ✅ | **CLOSED** |
| NVIDIA | Vision (VLM models) | ✅ Via inner OpenAICompatibleProvider | **CLOSED** |

---

## 2. Embeddings

| Provider | Supported | Model(s) | Dimensions | Max Tokens | Batch | Multimodal |
|----------|-----------|----------|------------|------------|-------|------------|
| **OpenAI** | ✅ | `text-embedding-3-small/large` | 1536/3072 | 8191 | ✅ | ❌ text only |
| **Azure OpenAI** | ✅ | `text-embedding-3-small/large` | 1536/3072 | 8192 | ✅ | ❌ |
| **Anthropic** | ➖ | N/A (use Voyage AI) | — | — | — | — |
| **Gemini** | ✅ | `text-embedding-004` | 768 | 2048 | ✅ | ❌ (text only) |
| **Vertex AI** | ✅ | Via Gemini | 768 | 2048 | ✅ | ❌ |
| **Bedrock** | ✅ | Titan v1/v2, Cohere v3/v4, Nova | 1024 (Titan V2) | 8192 | ✅ | ⚠️ (Nova multimodal) |
| **Mistral** | ✅ | `mistral-embed` | 1024 | 8191 | ✅ | ❌ |
| **xAI** | ➖ | N/A (no public API) | — | — | — | — |
| **Cohere** | ✅ | `embed-v4.0` | 1536 | 128k | ✅ | ❌ |
| **OpenRouter** | ➖ | N/A (no embed endpoint) | — | — | — | — |
| **NVIDIA** | ✅ | `llama-nemotron-embed-1b-v2` | 2048 | 8192 | ✅ | ❌ |
| **Ollama** | ✅ | Local models | Variable | Variable | ✅ | ❌ |
| **LM Studio** | ✅ | Local models | Variable | Variable | ✅ | ❌ |

### Gaps vs. Provider API Reality

| Provider | API Offers | edgequake-llm | Priority |
|----------|-----------|---------------|----------|
| **Gemini** | `gemini-embedding-2` (multimodal, 128–3072 dim) | ✅ Updated to gemini-embedding-2 | **CLOSED** |
| **Bedrock** | Nova 2 Multimodal Embeddings (newest) | ✅ Nova 2 multimodal model rule added | **CLOSED** |
| **Cohere** | Embed v4 (multimodal, Matryoshka, 128k context) | ✅ Implemented | **CLOSED** |
| **NVIDIA** | `llama-nemotron-embed-1b-v2` (OpenAI-compat) | ✅ Implemented | **CLOSED** |
| **Mistral** | `mistral-embed-dim256-2510` (variable dimensions) | ✅ `with_embed_dimensions()` | **CLOSED** |

---

## 3. Image Generation

| Provider | Supported | Model(s) | API Style | Aspect Ratio | Editing |
|----------|-----------|----------|-----------|-------------|---------|
| **OpenAI** | ✅ | `gpt-image-2`, `gpt-image-1` | REST | ✅ | ❌ |
| **Azure OpenAI** | ✅ | `gpt-image-2` | REST | ✅ | ❌ |
| **Anthropic** | ➖ | N/A | — | — | — |
| **Gemini** | ✅ | `gemini-3.1-flash-image`, `gemini-3-pro-image` | Native | ✅ | ❌ |
| **Vertex AI** | ✅ | Imagen 3 | REST | ✅ | ✅ |
| **Bedrock** | ✅ | Stability AI (SD3.5, Core, Ultra) | InvokeModel | ✅ | ❌ |
| **Mistral** | ➖ | Agent tool only | — | — | — |
| **xAI** | ✅ | `grok-imagine-image-quality`, `grok-imagine-image` | REST | ✅ | ❌ |
| **Cohere** | ➖ | N/A | — | — | — |
| **OpenRouter** | ➖ | N/A | — | — | — |
| **NVIDIA** | ✅ | SD3.5, FLUX.1 (via NIM) | REST | ❌ | ❌ |
| **Ollama** | ❌ | Experimental | — | — | — |
| **LM Studio** | ➖ | N/A | — | — | — |
| **Fal.ai** | ✅ | FLUX, SD3 | REST | ✅ | ❌ |

### Gaps vs. Provider API Reality

| Provider | API Offers | edgequake-llm | Priority |
|----------|-----------|---------------|----------|
| **OpenAI** | `gpt-image-2`, `gpt-image-1`, `gpt-image-1-mini` | ✅ Implemented | **CLOSED** |
| **Azure OpenAI** | `gpt-image-2` via Azure endpoints | ✅ `AzureImageGen` provider | **CLOSED** |
| **Gemini** | `gemini-3.1-flash-image`, `gemini-3-pro-image` | ✅ Updated | **CLOSED** |
| **xAI** | `grok-imagine-image-quality`, `grok-imagine-image` | ✅ Implemented | **CLOSED** |
| **Bedrock** | Stability AI (SD3.5, Stable Image Core/Ultra) | ✅ `BedrockStabilityImageGen` (feature-gated) | **CLOSED** |
| **NVIDIA** | Visual GenAI NIM (SD3.5, FLUX.1) | ✅ `NvidiaImageGen` provider | **CLOSED** |

---

## 4. Model Catalog Currency

### Frontier Models — What's Current vs. What edgequake-llm References

| Provider | Current Flagship (July 2026) | edgequake-llm Default/Referenced |
|----------|------------------------------|----------------------------------|
| OpenAI | GPT-5.5 | GPT-4o / GPT-4 Turbo |
| Azure | GPT-5.5 | GPT-4 |
| Anthropic | Claude Fable 5 / Opus 4.8 / Sonnet 5 | Claude 3.5 Sonnet / Claude 3 |
| Gemini | Gemini 3.5 Flash / 3.1 Pro | Gemini 2.5 Pro / Flash |
| Bedrock | Claude Fable 5 + Nova 2 | Claude 3.x |
| Mistral | Mistral Medium 3.5 | Mistral Large 2 |
| xAI | Grok 4.3 / 4.20 | Grok 3 / Grok Beta |
| NVIDIA | Llama Nemotron + third-party | Generic NIM models |

---

## 5. Authentication Methods

| Provider | Supported Auth | Missing Auth Methods |
|----------|---------------|---------------------|
| OpenAI | ✅ API key | — |
| Azure | ✅ API key, ✅ Entra ID / Bearer Token | — |
| Anthropic | ✅ API key | — |
| Gemini | ✅ API key | — |
| Vertex AI | ✅ API key, ✅ ADC / Metadata server | — |
| Bedrock | ✅ AWS SigV4 | — |
| Mistral | ✅ API key | — |
| xAI | ✅ API key | — |
| Cohere | ✅ API key | — |
| OpenRouter | ✅ API key | — |
| NVIDIA | ✅ API key | — |
| Ollama | ✅ None (local) | — |
| LM Studio | ✅ None (local) | ⚠️ Auth tokens |

---

## 6. Advanced Features Coverage

| Feature | Providers That Offer It | edgequake-llm Coverage |
|---------|------------------------|----------------------|
| **Structured Outputs (JSON Schema)** | OpenAI, Anthropic, Gemini, Mistral | ✅ `response_schema` in CompletionOptions |
| **Prompt Caching** | OpenAI, Anthropic, Gemini | ✅ OpenAI + Anthropic, ⚠️ Gemini |
| **Batch/Async API** | OpenAI, Anthropic, Gemini, Bedrock | ❌ Not implemented |
| **Fine-tuning API** | OpenAI, Mistral, Cohere | ❌ Not in scope |
| **Reranking** | Cohere, NVIDIA, Jina | ⚠️ Generic HTTP reranker only |
| **Code Execution** | OpenAI, Anthropic, Gemini | ❌ Not implemented |
| **Web Search (grounding)** | OpenAI, Gemini, Anthropic | ❌ Not implemented |
| **Computer Use** | OpenAI (GPT-5.4+), Anthropic | ❌ Not in scope |
| **Audio/TTS** | OpenAI, Gemini | ❌ Not in scope |
| **Video Generation** | OpenAI (Sora), Gemini (Veo), xAI | ❌ Not in scope |
