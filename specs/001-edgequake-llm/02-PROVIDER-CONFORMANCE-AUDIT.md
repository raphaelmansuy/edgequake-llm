# 02 — Provider Conformance Audit

> **Lens**: AI Engineer  
> **Date**: 2026-07-04  
> **Status**: DRAFT  
> **Method**: Code-is-Law analysis of `src/providers/*.rs` vs official API documentation

## Audit Methodology

For each provider:
1. Read the current implementation in `src/providers/`
2. Compare against the official API documentation (July 2026)
3. Identify model coverage gaps (missing models)
4. Identify capability gaps (missing features)
5. Identify discovery gaps (no `list_models` or stale metadata)

---

## Provider-by-Provider Audit

### 1. OpenAI (`src/providers/openai.rs`)

**Official API**: `https://api.openai.com/v1`  
**SDK**: Uses `async-openai` crate v0.34  
**Discovery API**: `GET /v1/models` — returns all available models

#### Current Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Chat completion | ✅ | Full support |
| Streaming | ✅ | SSE-based |
| Tool/Function calling | ✅ | Full support |
| Vision | ✅ | ImageData → content parts |
| Embeddings | ✅ | text-embedding-3 family |
| JSON mode | ✅ | `response_format` support |
| Thinking/Reasoning | ⚠️ | `reasoning_effort` for o-series only |
| Structured Outputs | ⚠️ | `strict: true` set but not all modes |
| Model Discovery | ❌ | No `list_models()` method |

#### Model Coverage Gaps (July 2026)

| Model | In Config? | Context In | Context Out | Notes |
|-------|-----------|------------|-------------|-------|
| `gpt-4o` | ✅ | 128K | 16K | Config correct |
| `gpt-4o-mini` | ✅ | 128K | 16K | Config correct |
| `gpt-4.1` | ❌ MISSING | 1M | 32K | Released April 2026 |
| `gpt-4.1-mini` | ❌ MISSING | 1M | 32K | Released April 2026 |
| `gpt-4.1-nano` | ❌ MISSING | 1M | 32K | Released April 2026 |
| `o3` | ❌ MISSING | 200K | 100K | Reasoning model |
| `o3-mini` | ❌ MISSING | 200K | 100K | Cost-effective reasoning |
| `o4-mini` | ❌ MISSING | 200K | 100K | Latest reasoning model |
| `o1` | ❌ MISSING | 200K | 100K | Original reasoning |
| `o1-mini` | ❌ MISSING | 128K | 65K | Cost-effective |
| `gpt-4-turbo` | ❌ MISSING | 128K | 4K | Still available |
| `text-embedding-3-small` | ✅ | 8K | — | Correct |
| `text-embedding-3-large` | ✅ | 8K | — | Correct |
| `text-embedding-ada-002` | ❌ MISSING | 8K | — | Legacy but active |

#### Conformance Issues

1. **`async-openai` v0.34 may be outdated** — check if latest version supports Responses API
2. **No `/v1/models` discovery** — OpenAI's endpoint returns model IDs but limited metadata (no context length, no capabilities). Static registry is needed.
3. **Missing `max_completion_tokens`** — o-series models require this instead of `max_tokens`
4. **Missing reasoning token tracking** — `completion_tokens_details.reasoning_tokens` for o-series

---

### 2. Anthropic Claude (`src/providers/anthropic.rs`)

**Official API**: `https://api.anthropic.com/v1`  
**SDK**: Direct HTTP (no official Rust SDK)  
**Discovery API**: `GET /v1/models` — available since 2025, returns model list with capabilities

#### Current Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Chat (Messages API) | ✅ | Full support |
| Streaming | ✅ | SSE with content blocks |
| Tool calling | ✅ | Full support |
| Vision | ✅ | base64 source format |
| Extended Thinking | ✅ | thinking blocks + budget |
| Prompt Caching | ✅ | cache_control with TTL |
| Embeddings | ❌ | Anthropic doesn't offer embeddings |
| Model Discovery | ❌ | No `list_models()` despite API existing |

#### Model Coverage Gaps (July 2026)

| Model | In Config? | Context In | Context Out | Thinking? |
|-------|-----------|------------|-------------|-----------|
| `claude-sonnet-4-5-20250929` | ✅ | 200K | 8K | ✅ |
| `claude-3-5-sonnet-20241022` | ✅ | 200K | 8K | ❌ |
| `claude-3-5-haiku-20241022` | ✅ | 200K | 8K | ❌ |
| `claude-opus-4-20250514` | ❌ MISSING | 200K | 32K | ✅ |
| `claude-sonnet-4-20250514` | ❌ MISSING | 200K | 64K | ✅ |
| `claude-haiku-4-20250514` | ❌ MISSING | 200K | 8K | ❌ |
| `claude-4.5-sonnet` | ⚠️ ALIAS | 200K | 16K | ✅ |
| `claude-4.5-opus` | ❌ MISSING | 200K | 32K | ✅ |

#### Conformance Issues

1. **Max output tokens are WRONG** — Config says 8K for claude-sonnet-4-5 but latest Claude 4.x models support 32K-64K output
2. **Missing Claude 4 family** — The `claude-opus-4`, `claude-sonnet-4` models are not in config
3. **Anthropic now has `/v1/models`** — should implement dynamic discovery
4. **Missing `anthropic-beta` header management** — needed for features like extended cache, computer use
5. **Missing PDF support** — Claude supports document/PDF input (base64 encoded)
6. **Missing `citations` support** — Claude 4+ supports source citations in responses

---

### 3. Google Gemini (`src/providers/gemini.rs`)

**Official API**: `https://generativelanguage.googleapis.com/v1beta`  
**SDK**: Direct HTTP (REST API)  
**Discovery API**: `GET /v1beta/models` — returns detailed model metadata including token limits

#### Current Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Chat | ✅ | Full support |
| Streaming | ✅ | SSE-based |
| Tool calling | ✅ | Function declarations |
| Vision | ✅ | Inline data parts |
| Embeddings | ✅ | embedContent endpoint |
| Thinking (2.5) | ✅ | thinkingConfig with budget |
| Thinking (3.x) | ✅ | thinkingLevel support |
| Code execution | ❌ | Not implemented |
| Model Discovery | ❌ | API exists but not used |

#### Model Coverage Gaps (July 2026)

| Model | In Config? | Context In | Context Out | Notes |
|-------|-----------|------------|-------------|-------|
| `gemini-2.5-flash` | ✅ (README) | 1M | 65K | Current default |
| `gemini-2.5-pro` | ✅ (README) | 1M | 65K | Pro tier |
| `gemini-3-flash-preview` | ✅ (README) | 1M | 65K | Preview |
| `gemini-3.1-pro-preview` | ✅ (README) | 2M | 65K | Latest preview |
| `gemini-3.1-flash-lite-preview` | ✅ (README) | 1M | 65K | Lite preview |
| `gemini-2.0-flash` | ❌ MISSING | 1M | 8K | Still available |
| `text-embedding-004` | ❌ MISSING | 2K | — | Latest embedding |
| `embedding-001` | ❌ MISSING | 2K | — | Legacy embedding |

**NOTE**: Models are referenced in README but have **ZERO ModelCard entries** in `model_config.rs::builtin_defaults()`. The Gemini provider is completely absent from the static config system.

#### Conformance Issues

1. **NO static model cards** — Gemini provider exists but has no entries in `ModelsConfig`
2. **Should use Gemini discovery API** — `GET /v1beta/models` returns `inputTokenLimit`, `outputTokenLimit`, `supportedGenerationMethods`
3. **Missing grounding/search** — Google Search grounding is a key Gemini feature
4. **Missing audio input** — Gemini 2.5+ supports audio

---

### 3b. Google Vertex AI (`src/providers/gemini.rs` — VertexAI endpoint)

> **ADDED 2026-07-04**: Vertex AI is a separate provider type (`ProviderType::VertexAI`)
> sharing the `GeminiProvider` implementation but using OAuth2/ADC instead of API key.

**Official API**: `https://{region}-aiplatform.googleapis.com` or `https://aiplatform.googleapis.com` (global)  
**Auth**: OAuth2 Bearer token (no API key). Obtained via ADC, service account, or gcloud CLI.  
**Source**: https://docs.cloud.google.com/vertex-ai/docs/authentication

#### Current Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Chat (generateContent) | ✅ | Same as Gemini, Bearer auth |
| Streaming (streamGenerateContent) | ✅ | Same as Gemini |
| Tool calling | ✅ | Same as Gemini |
| Thinking (thinkingConfig) | ✅ | Same as Gemini |
| Embeddings (predict API) | ✅ | Uses VertexAI-specific predict endpoint |
| Image generation (Imagen) | ✅ | Via `imagegen/providers/vertexai.rs` |
| Global endpoint routing | ✅ | `requires_global_vertex` in ModelProfile |
| Auth: GOOGLE_ACCESS_TOKEN | ✅ | Direct env var |
| Auth: gcloud CLI | ✅ | `get_access_token_from_gcloud()` fallback |
| Auth: ADC (auto-detect) | ❌ MISSING | No Google Auth library |
| Auth: Service account key | ❌ MISSING | GOOGLE_APPLICATION_CREDENTIALS not used |
| Auth: Workload Identity Fed. | ❌ MISSING | No external token exchange |
| Auth: Metadata server (GCE) | ❌ MISSING | No metadata.google.internal call |
| Token refresh on expiry | ❌ MISSING | Token stored as static String |
| Model Discovery | ⚠️ PARTIAL | Uses publisher models endpoint (wrong one) |
| 3rd-party models (Model Garden) | ❌ MISSING | DeepSeek, Llama on Vertex not surfaced |

#### Auth Conformance Gap Analysis

The official Vertex AI auth documentation (https://docs.cloud.google.com/vertex-ai/docs/authentication)
defines 5 auth methods. Only 2 are implemented:

```
  Auth Method                          Current     Official
  ─────────────────────────────────────────────────────────
  GOOGLE_ACCESS_TOKEN env var          ✅          ✅
  gcloud auth print-access-token       ✅          ✅ (dev only)
  ADC (Application Default Creds)      ❌          ✅ (recommended)
  Service account key file             ❌          ✅ (legacy)
  Workload Identity Federation         ❌          ✅ (production)
  Attached SA (metadata server)        ❌          ✅ (Cloud Run/GCE)
```

**Impact**: Vertex AI will not work on Cloud Run, GKE, Cloud Functions, or
any GCP compute resource where gcloud CLI is not installed. It also cannot
be used with Workload Identity Federation for cross-cloud (AWS/Azure) deployments.

#### Discovery API Gap

The current `list_models()` for Vertex AI calls the **publisher models** endpoint
(`/v1/projects/{p}/locations/{l}/publishers/google/models`) which returns
`PublisherModel{}` objects with `supportedActions` (deploy, fine-tune) — NOT
capability metadata. The correct endpoint for model capabilities is the unified
Gemini API (`/v1beta/models`) with Bearer auth.

Additionally, `GeminiModelInfo` is missing the `thinking: bool` field that the API
returns, causing Vertex AI to not know which models support thinking.

#### Conformance Issues (Priority)

1. **AUTH CRITICAL**: Missing ADC / metadata server / WIF auth means Vertex AI
   does not work in standard GCP production deployments
2. **Token expiry**: Access tokens expire after ~1 hour with no refresh mechanism
3. **Wrong discovery endpoint**: `list_models()` uses publisher endpoint without
   `inputTokenLimit` or `outputTokenLimit`
4. **Missing `thinking` field**: `GeminiModelInfo` doesn't parse `thinking: bool`
5. **Global endpoint for 3.x**: Discovery doesn't list Gemini 3.x when user's
   region is not `"global"`
6. **3rd-party models**: Vertex AI hosts DeepSeek, Llama, Mistral via Model Garden
   but these are not surfaced by the provider

---

### 4. Ollama (`src/providers/ollama.rs`)

**Official API**: `http://localhost:11434/api`  
**Discovery API**: `GET /api/tags` (list models), `POST /api/show` (model details)

#### Current Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Chat (OpenAI compat) | ✅ | `/v1/chat/completions` |
| Native chat | ✅ | `/api/chat` |
| Streaming | ✅ | Both endpoints |
| Tool calling | ✅ | Function calling support |
| Embeddings | ✅ | `/api/embed` endpoint |
| Model Discovery | ✅ | `list_models()` exists |
| Model Details | ✅ | `get_model_info()` exists |
| Reasoning effort | ✅ | `think` parameter support |
| Vision | ⚠️ | Supported but not reflected in capabilities |

#### Model Discovery Status

```rust
// Already implemented! Returns:
pub struct OllamaModelsResponse {
    pub models: Vec<OllamaModelInfo>,
}
pub struct OllamaModelInfo {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    pub details: OllamaModelDetails,
}
pub struct OllamaModelDetails {
    pub parent_model: String,
    pub format: String,
    pub family: String,
    pub families: Option<Vec<String>>,
    pub parameter_size: String,
    pub quantization_level: String,
}
```

#### Conformance Issues

1. **Discovery exists but doesn't return capabilities** — `OllamaModelInfo` has no context length, vision, or tool support flags
2. **Must call `/api/show`** per model to get `num_ctx`, template, capabilities
3. **Not normalized to `ModelCard`** — returns Ollama-specific types
4. **Missing Ollama Cloud support** — `OLLAMA_API_KEY` for cloud is implemented but discovery doesn't use it

---

### 5. LM Studio (`src/providers/lmstudio.rs`)

**Official API**: `http://localhost:1234`  
**Discovery API**: `GET /api/v1/models` (native), `GET /v1/models` (OpenAI compat)

#### Current Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Chat (OpenAI compat) | ✅ | Via OpenAICompatibleProvider |
| Native reasoning API | ✅ | `/api/v1/chat` with reasoning blocks |
| Streaming | ✅ | Both paths |
| Tool calling | ✅ | Via inner provider |
| Embeddings | ✅ | Via inner provider |
| Model metadata sync | ✅ | `refresh_model_metadata()` |
| Auto-load models | ✅ | Via `lms` CLI |

#### Model Discovery Status

```rust
// Already has live metadata fetching:
pub struct LmStudioModelMetadata {
    pub max_context_length: Option<usize>,
    pub default_max_completion_tokens: Option<usize>,
}
```

But this only fetches metadata for the **currently loaded model**, not a catalog of available models.

#### Conformance Issues

1. **No catalog listing** — can't enumerate all downloaded models
2. **Only loaded model metadata** — `refresh_model_metadata()` queries the active model only
3. **LM Studio 0.3+ has `GET /v1/models`** — should enumerate downloadable/loaded models
4. **Missing `GET /lms/models` endpoint** — LM Studio's native management API

---

### 6. OpenRouter (`src/providers/openrouter.rs`)

**Official API**: `https://openrouter.ai/api/v1`  
**Discovery API**: `GET /api/v1/models` — rich metadata for 200+ models

#### Current Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Chat | ✅ | Full support |
| Streaming | ✅ | SSE-based |
| Tool calling | ✅ | Function calling |
| Vision | ✅ | Content parts format |
| Model Discovery | ✅ | `list_models_cached()` with TTL |
| Model Metadata | ✅ | Rich `ModelInfo` struct |

#### Model Discovery Status — **BEST IN CLASS**

```rust
// Already implemented with caching!
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub context_length: usize,
    pub pricing: ModelPricing,
    pub architecture: ModelArchitecture,
    // ... rich metadata
}
```

#### Conformance Issues

1. **Not normalized to `ModelCard`** — returns `openrouter::ModelInfo` instead
2. **Missing capability flags** — OpenRouter API returns `modality` but not tool_calling support
3. **Cache TTL not configurable** — hardcoded duration
4. **Good reference implementation** — should be the model for other dynamic providers

---

### 7. Mistral (`src/providers/mistral.rs`)

**Official API**: `https://api.mistral.ai/v1`  
**Discovery API**: `GET /v1/models` — returns model list

#### Current Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Chat | ✅ | Full support |
| Streaming | ✅ | SSE-based |
| Tool calling | ✅ | With parallel_tool_calls |
| Embeddings | ✅ | mistral-embed |
| Vision | ⚠️ | Pixtral models support it |
| Audio | ✅ | Speech, transcriptions |
| OCR | ✅ | Document OCR |
| Model Discovery | ✅ | `list_models()` exists |

#### Model Coverage Gaps (July 2026)

| Model | In Config? | Context | Notes |
|-------|-----------|---------|-------|
| `mistral-small-latest` | ✅ | 32K | Correct |
| `mistral-medium-latest` | ✅ | 128K | Config says 131072 |
| `mistral-large-latest` | ✅ | 128K | Config says 131072 |
| `codestral-latest` | ✅ | 256K | Config says 32K — **WRONG** |
| `magistral-small-latest` | ❌ MISSING | 128K | Reasoning model |
| `magistral-medium-latest` | ❌ MISSING | 128K | Reasoning model |
| `devstral-latest` | ❌ MISSING | 128K | Coding model |
| `pixtral-large-latest` | ❌ MISSING | 128K | Vision model |
| `mistral-embed` | ✅ | 8K | Correct |

#### Conformance Issues

1. **Codestral context length WRONG** — config says 32K, actual is 256K
2. **Missing Magistral models** — reasoning models not in config
3. **Missing Pixtral** — vision model not in config
4. **Discovery exists but not normalized** — returns provider-specific types

---

### 8. NVIDIA NIM (`src/providers/nvidia.rs`)

**Official API**: `https://integrate.api.nvidia.com/v1`  
**Discovery API**: `GET /v1/models` — returns available models

#### Current Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Chat | ✅ | Via OpenAICompatibleProvider |
| Streaming | ✅ | SSE-based |
| Tool calling | ✅ | Via inner provider |
| Model Discovery | ✅ | `list_models()` exists |
| 202 Async Polling | ✅ | Unique to NVIDIA |
| Reasoning effort | ✅ | For DeepSeek V4 Flash |

**NVIDIA has a static model catalog** in the provider code with rich metadata.

#### Conformance Issues

1. **Good static catalog** — but needs July 2026 update
2. **Dynamic + static blend** — one of the best implementations
3. **Not normalized** — returns `NvidiaModelInfo` instead of `ModelCard`

---

### 9. xAI Grok (`src/providers/xai.rs`)

**Official API**: `https://api.x.ai/v1`  
**Discovery API**: `GET /v1/models` — available but limited

#### Conformance Issues

1. **No model cards in config** — xAI provider has zero entries in `ModelsConfig`
2. **Missing Grok-3, Grok-4** — latest models not referenced
3. **No discovery implementation** — despite API existing
4. **Vision support unclear** — Grok models have vision but not flagged

---

### 10. Azure OpenAI (`src/providers/azure_openai.rs`)

> **ADDED 2026-07-04**: Updated with July 2026 model availability from official Microsoft documentation.

**Official API**: `https://{resource}.openai.azure.com/openai/`  
**Auth**: API key (`AZURE_OPENAI_API_KEY`) or Azure AD token  
**Source**: https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure

#### Current Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Chat Completions API | ✅ | Via `async-openai` with `AzureConfig` |
| Streaming | ✅ | Same as OpenAI |
| Tool calling | ✅ | Full support |
| Vision (image input) | ✅ | Via OpenAI types |
| Embeddings | ✅ | Separate deployment support |
| JSON mode | ✅ | Structured outputs |
| Responses API | ❌ MISSING | New Azure API (v1), supports stateful multi-turn |
| Reasoning (o-series) | ⚠️ PARTIAL | No `reasoning_effort` parameter |
| GPT-5.x series | ❌ MISSING | Not in config, context hardcoded to 128K |
| API version | ⚠️ STALE | Hardcoded to `2024-10-21`, current is `2025-06-01` |
| Model discovery | ❌ MISSING | Two APIs exist (data plane + management plane) |
| Deployment listing | ❌ MISSING | Requires ARM management API |

#### Model Coverage Gaps (July 2026)

| Model | In Config? | Context In | Max Out | Vision | Tools | Thinking |
|-------|-----------|------------|---------|--------|-------|----------|
| `gpt-5.5` | ❌ | 1M | 32K | ✅ | ✅ | ✅ |
| `gpt-5.4` | ❌ | 400K | 128K | ✅ | ✅ | ✅ |
| `gpt-5.4-pro` | ❌ | 400K | 128K | ✅ | ✅ | ✅ |
| `gpt-5.4-mini` | ❌ | 400K | 16K | ✅ | ✅ | ✅ |
| `gpt-5.4-nano` | ❌ | 400K | 16K | ✅ | ✅ | ✅ |
| `gpt-5` | ❌ | 400K | 128K | ✅ | ✅ | ✅ |
| `gpt-5-mini` | ❌ | 400K | 16K | ✅ | ✅ | ✅ |
| `gpt-5-nano` | ❌ | 400K | 16K | ✅ | ✅ | ✅ |
| `gpt-4.1` | ❌ | 1M | 32K | ✅ | ✅ | ❌ |
| `gpt-4.1-mini` | ❌ | 1M | 32K | ✅ | ✅ | ❌ |
| `gpt-4o` | ❌ | 128K | 16K | ✅ | ✅ | ❌ |
| `o4-mini` | ❌ | 200K | 100K | ✅ | ✅ | ✅ |
| `o3` | ❌ | 200K | 100K | ✅ | ✅ | ✅ |
| `o3-mini` | ❌ | 200K | 100K | ❌ | ✅ | ✅ |
| `o1` | ❌ | 200K | 100K | ✅ | ❌ | ✅ |
| `text-embedding-3-large` | ❌ | 8K | — | — | — | — |
| `text-embedding-3-small` | ❌ | 8K | — | — | — | — |

**NOTE**: The provider hardcodes `max_context_length: 128_000` for ALL deployments. There is no model-specific context length lookup — the user must manually call `.with_max_context_length()`.

#### Conformance Issues (Priority)

1. **API VERSION STALE**: Hardcoded `2024-10-21` — current is `2025-06-01` with GPT-5.x support
2. **No model registry**: Context length hardcoded to 128K for all models — wrong for GPT-5.x (400K), GPT-4.1 (1M), o-series (200K)
3. **Missing Responses API**: Azure's new Responses API (v1 endpoint) with multi-turn state is not implemented
4. **Deployment-centric**: Azure uses deployments (user-chosen names) not model IDs — discovery must map deployment → model → capabilities
5. **No reasoning support**: No `reasoning_effort` or `max_completion_tokens` parameter for o-series
6. **No model listing**: Data plane `GET /openai/models` exists but returns capabilities without context lengths

---

### 11. AWS Bedrock (`src/providers/bedrock.rs` — feature-gated)

> **ADDED 2026-07-04**: Updated with July 2026 model availability from official AWS documentation.

**Official API**: AWS Bedrock Runtime Converse API  
**Auth**: AWS credential chain (IAM roles, access keys, profiles, IMDS)  
**Feature gate**: `#[cfg(feature = "bedrock")]`  
**Source**: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html

#### Current Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Converse API (chat) | ✅ | Model-agnostic unified interface |
| Streaming (ConverseStream) | ✅ | Full streaming support |
| Tool calling | ✅ | Via Converse toolConfig |
| Tool streaming | ❌ MISSING | `supports_tool_streaming()` returns `false` |
| JSON mode | ❌ MISSING | `supports_json_mode()` returns `false` |
| Vision (image input) | ✅ | Via ImageBlock in Converse |
| Thinking/reasoning | ✅ | Via `additionalModelRequestFields.thinking` |
| Embeddings | ✅ | Via InvokeModel (model-specific payloads) |
| Inference profiles | ✅ | Auto-resolves `us.`, `eu.`, `ap.` prefixes |
| Model discovery | ❌ MISSING | `ListFoundationModels` API not used |
| Claude 5.x models | ❌ MISSING | Sonnet 5, Fable 5, Opus 4.8 not in MODEL_RULES |
| Claude 1M context | ❌ WRONG | Claude 4.6+ have 1M context, code says 200K |
| Amazon Nova 2 models | ❌ MISSING | Nova 2 Lite, Nova 2 Sonic not in rules |
| OpenAI models on Bedrock | ❌ MISSING | `gpt-oss-120b`, `gpt-oss-20b` not handled |

#### Model Coverage Gaps (July 2026)

| Model (Bedrock ID) | In Rules? | Ctx In | Max Out | Vision | Tools | Think |
|---------------------|-----------|--------|---------|--------|-------|-------|
| `anthropic.claude-sonnet-5` | ❌ | 1M | 128K | ✅ | ✅ | ✅ |
| `anthropic.claude-fable-5` | ❌ | 1M | 128K | ✅ | ✅ | ✅ |
| `anthropic.claude-opus-4-8` | ❌ | 1M | 128K | ✅ | ✅ | ✅ |
| `anthropic.claude-opus-4-7` | ⚠️ | 1M* | 128K | ✅ | ✅ | ✅ |
| `anthropic.claude-sonnet-4-6` | ⚠️ | 1M* | 128K | ✅ | ✅ | ✅ |
| `anthropic.claude-opus-4-6` | ⚠️ | 1M* | 128K | ✅ | ✅ | ✅ |
| `amazon.nova-2-lite-v1:0` | ❌ | 300K | — | ✅ | ✅ | ❌ |
| `amazon.nova-2-sonic-v1:0` | ❌ | 300K | — | 🎤 | ✅ | ❌ |
| `amazon.nova-premier-v1:0` | ❌ | 1M | — | ✅ | ✅ | ✅ |
| `meta.llama-4-maverick-17b-instruct-v1:0` | ❌ | 1M | 16K | ✅ | ✅ | ❌ |
| `meta.llama-4-scout-17b-instruct-v1:0` | ❌ | 512K | 16K | ✅ | ✅ | ❌ |
| `openai.gpt-oss-120b` | ❌ | 128K | — | ❌ | ❌ | ✅ |
| `openai.gpt-oss-20b` | ❌ | 128K | — | ❌ | ❌ | ✅ |
| `deepseek.r1-v1:0` | ✅ | 128K | — | ❌ | ❌ | ✅ |
| `qwen.qwen3-235b-a22b-instruct-v1:0` | ❌ | 131K | — | ❌ | ✅ | ✅ |

*Claude 4.6+/4.7 have 1M context but `MODEL_RULES` says 200K (matches `anthropic.claude-3`/`anthropic.claude-4` prefix)

#### Conformance Issues (Priority)

1. **CONTEXT LENGTH WRONG**: `MODEL_RULES` maps `anthropic.claude-4` prefix to 200K — but Claude 4.6+/4.7 have 1M context. The prefix match is too broad.
2. **Missing new models**: Claude Sonnet 5, Fable 5, Opus 4.8, Nova Premier, Nova 2 series, Llama 4, OpenAI gpt-oss models are all absent from `MODEL_RULES`
3. **No dynamic discovery**: `ListFoundationModels` API returns `modelId`, `inputModalities`, `outputModalities`, `responseStreamingSupported`, `modelLifecycle` — this is enough for basic discovery
4. **Tool streaming missing**: `supports_tool_streaming()` returns false — Bedrock ConverseStream does support tool use events
5. **JSON mode missing**: `supports_json_mode()` returns false — some Bedrock models (Claude, Nova) support structured output
6. **Thinking uses `{type: "enabled"}` only**: Claude 4.7+ requires `{type: "adaptive"}` — the hardcoded `"enabled"` will fail on newer models
7. **Feature-gated correctly**: Bedrock discovery must also be feature-gated

---

### 12. OpenAI Compatible (`src/providers/openai_compatible.rs`)

**Purpose**: Generic provider for Groq, Together, DeepSeek, custom endpoints  
**Discovery**: Depends on the upstream service

#### Conformance Issues

1. **No model cards** — relies on upstream's `/v1/models` if available
2. **Groq** has `/v1/models` with rich metadata
3. **Together AI** has `/v1/models` with rich metadata
4. **DeepSeek** has `/v1/models` with basic metadata
5. **Should attempt dynamic discovery** when base_url is set

---

### 13. HuggingFace (`src/providers/huggingface.rs`)

**Official API**: `https://api-inference.huggingface.co`  
**Discovery**: HuggingFace API has model metadata

#### Conformance Issues

1. **Limited tool support** — marked as "Limited" in provider matrix
2. **No model cards** — no entries in `ModelsConfig`
3. **HF API has rich model metadata** — should query it

---

## Conformance Summary Matrix

```
  Provider Conformance Heatmap (July 2026)

  Provider          Chat  Stream  Tools  Vision  Think  Embed  Discover  Config
  ─────────────────────────────────────────────────────────────────────────────
  OpenAI            ✅     ✅      ✅     ✅      ⚠️     ✅     ❌        ⚠️
  Anthropic         ✅     ✅      ✅     ✅      ✅     ❌     ❌        ⚠️
  Gemini            ✅     ✅      ✅     ✅      ✅     ✅     ❌        ❌
  Ollama            ✅     ✅      ✅     ⚠️     ✅     ✅     ✅        ⚠️
  LM Studio         ✅     ✅      ✅     ⚠️     ✅     ✅     ⚠️       ⚠️
  OpenRouter        ✅     ✅      ✅     ✅      ⚠️     ❌     ✅        ❌
  Mistral           ✅     ✅      ✅     ⚠️     ❌     ✅     ✅        ⚠️
  NVIDIA            ✅     ✅      ✅     ⚠️     ✅     ❌     ✅        ⚠️
  xAI               ✅     ✅      ✅     ⚠️     ❌     ❌     ❌        ❌
  Azure OpenAI      ✅     ✅      ✅     ✅      ⚠️     ✅     ❌        ❌
  Bedrock           ✅     ✅      ✅     ✅      ✅     ✅     ❌        ❌
  HuggingFace       ✅     ✅      ⚠️    ❌      ❌     ❌     ❌        ❌
  OAI-Compatible    ✅     ✅      ✅     ⚠️     ⚠️     ⚠️    ❌        ❌
  VSCode Copilot    ✅     ✅      ✅     ⚠️     ⚠️     ✅     ⚠️       ❌

  Legend: ✅ = Full  ⚠️ = Partial  ❌ = Missing/None
```

---

## Priority Remediation List

### P0 — Critical (Blocks Agent Harness)

1. **Create `ModelDiscoveryProvider` trait** — unified discovery interface
2. **Update OpenAI model registry** — add GPT-4.1, o3, o4-mini
3. **Update Anthropic model registry** — add Claude 4.x family
4. **Add Gemini model cards** — completely missing from config

### P1 — High (Correctness)

5. **Fix Codestral context length** — 32K → 256K
6. **Fix Claude output token limits** — 8K → 32K-64K for Claude 4.x
7. **Add missing Mistral models** — Magistral, Pixtral, Devstral
8. **Add xAI model cards** — Grok-3, Grok-4
9. **Normalize all discovery responses** — to `ModelCard`

### P2 — Medium (Completeness)

10. **Add Bedrock model cards** — Claude, Llama, Mistral on Bedrock
11. **Implement Anthropic `/v1/models` discovery**
12. **Implement Gemini `/v1beta/models` discovery**
13. **Implement xAI `/v1/models` discovery**
14. **Add `list_models()` to OpenAI-compatible providers**
