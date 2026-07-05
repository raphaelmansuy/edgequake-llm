# 04 — Provider-by-Provider Discovery Approaches

> **Lens**: Full Stack Developer  
> **Date**: 2026-07-04  
> **Status**: DRAFT  
> **Method**: Provider-specific strategy document

## Decision Framework

Each provider falls into one of three discovery strategies:

```
  ┌─────────────────────────────────────────────────────────────────┐
  │              Discovery Strategy Decision Tree                    │
  │                                                                  │
  │  Does provider have a list-models API?                           │
  │       │                                                          │
  │       ├── YES → Does it return capability metadata?              │
  │       │          │                                               │
  │       │          ├── YES (rich) → DYNAMIC strategy               │
  │       │          │   (Ollama, OpenRouter, Gemini API)             │
  │       │          │                                               │
  │       │          └── NO (IDs only) → HYBRID strategy             │
  │       │              API call + static enrichment                 │
  │       │              (Mistral, NVIDIA, OpenAI /v1/models)        │
  │       │                                                          │
  │       └── NO → STATIC strategy                                   │
  │              Built-in registry with July 2026 data               │
  │              (xAI hardcoded — no public models API)              │
  │                                                                  │
  │  Special cases:                                                  │
  │  - Bedrock: AWS SDK ListFoundationModels → HYBRID                │
  │  - OAI-Compatible: try /v1/models, fallback to STATIC           │
  │  - VSCode Copilot: uses Copilot catalog API → DYNAMIC           │
  └─────────────────────────────────────────────────────────────────┘
```

---

## 1. OpenAI — HYBRID Strategy

### API Endpoint

```
GET https://api.openai.com/v1/models
Authorization: Bearer sk-...
```

**Returns**: List of model IDs with `created` timestamp and `owned_by` — but **NO capability metadata** (no context length, no vision flag, etc.).

### Strategy

1. Call `/v1/models` to get the **live list of model IDs** the API key has access to
2. Match each ID against a **static capability registry** with full metadata
3. Unknown models get conservative defaults (128K context, no vision, tool support assumed)

### Static Registry (July 2026)

```rust
fn openai_static_registry() -> Vec<DiscoveredModel> {
    vec![
        // ── GPT-4.1 Family (April 2026) ──────────────────────
        DiscoveredModel {
            id: "gpt-4.1".into(),
            name: "GPT-4.1".into(),
            context_length: 1_048_576,  // 1M tokens
            max_output_tokens: 32_768,
            capabilities: ModelCapabilities {
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_structured_output: true,
                ..Default::default()
            },
            cost_per_m_input: Some(2.00),
            cost_per_m_output: Some(8.00),
            ..
        },
        DiscoveredModel {
            id: "gpt-4.1-mini".into(),
            name: "GPT-4.1 Mini".into(),
            context_length: 1_048_576,
            max_output_tokens: 32_768,
            capabilities: ModelCapabilities {
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_structured_output: true,
                ..Default::default()
            },
            cost_per_m_input: Some(0.40),
            cost_per_m_output: Some(1.60),
            ..
        },
        DiscoveredModel {
            id: "gpt-4.1-nano".into(),
            name: "GPT-4.1 Nano".into(),
            context_length: 1_048_576,
            max_output_tokens: 32_768,
            capabilities: ModelCapabilities {
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_structured_output: true,
                ..Default::default()
            },
            cost_per_m_input: Some(0.10),
            cost_per_m_output: Some(0.40),
            ..
        },

        // ── GPT-4o Family ────────────────────────────────────
        DiscoveredModel {
            id: "gpt-4o".into(),
            context_length: 128_000,
            max_output_tokens: 16_384,
            capabilities: ModelCapabilities {
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_structured_output: true,
                ..Default::default()
            },
            cost_per_m_input: Some(2.50),
            cost_per_m_output: Some(10.00),
            ..
        },
        DiscoveredModel {
            id: "gpt-4o-mini".into(),
            context_length: 128_000,
            max_output_tokens: 16_384,
            cost_per_m_input: Some(0.15),
            cost_per_m_output: Some(0.60),
            ..
        },

        // ── Reasoning Models (o-series) ──────────────────────
        DiscoveredModel {
            id: "o4-mini".into(),
            name: "o4-mini".into(),
            context_length: 200_000,
            max_output_tokens: 100_000,
            capabilities: ModelCapabilities {
                supports_vision: true,
                supports_function_calling: true,
                supports_thinking: true,
                supports_streaming: true,
                supports_structured_output: true,
                ..Default::default()
            },
            cost_per_m_input: Some(1.10),
            cost_per_m_output: Some(4.40),
            tags: vec!["reasoning".into()],
            ..
        },
        DiscoveredModel {
            id: "o3".into(),
            context_length: 200_000,
            max_output_tokens: 100_000,
            capabilities: ModelCapabilities {
                supports_vision: true,
                supports_function_calling: true,
                supports_thinking: true,
                supports_streaming: true,
                ..Default::default()
            },
            cost_per_m_input: Some(2.00),
            cost_per_m_output: Some(8.00),
            tags: vec!["reasoning".into()],
            ..
        },
        DiscoveredModel {
            id: "o3-mini".into(),
            context_length: 200_000,
            max_output_tokens: 100_000,
            capabilities: ModelCapabilities {
                supports_thinking: true,
                supports_function_calling: true,
                supports_streaming: true,
                ..Default::default()
            },
            cost_per_m_input: Some(1.10),
            cost_per_m_output: Some(4.40),
            tags: vec!["reasoning".into()],
            ..
        },
        DiscoveredModel {
            id: "o1".into(),
            context_length: 200_000,
            max_output_tokens: 100_000,
            capabilities: ModelCapabilities {
                supports_vision: true,
                supports_thinking: true,
                supports_function_calling: true,
                supports_streaming: true,
                ..Default::default()
            },
            cost_per_m_input: Some(15.00),
            cost_per_m_output: Some(60.00),
            tags: vec!["reasoning".into()],
            ..
        },

        // ── Embeddings ───────────────────────────────────────
        DiscoveredModel {
            id: "text-embedding-3-small".into(),
            model_type: ModelType::Embedding,
            capabilities: ModelCapabilities {
                embedding_dimension: 1536,
                max_embedding_tokens: 8191,
                ..Default::default()
            },
            cost_per_m_input: Some(0.02),
            ..
        },
        DiscoveredModel {
            id: "text-embedding-3-large".into(),
            model_type: ModelType::Embedding,
            capabilities: ModelCapabilities {
                embedding_dimension: 3072,
                max_embedding_tokens: 8191,
                ..Default::default()
            },
            cost_per_m_input: Some(0.13),
            ..
        },
    ]
}
```

### Implementation Notes

- `/v1/models` requires API key (Bearer auth)
- Response includes fine-tuned models — filter by `owned_by` prefix
- Model IDs may include snapshot dates (e.g., `gpt-4o-2024-11-20`) — match base name
- o-series models use `max_completion_tokens` instead of `max_tokens`

---

## 2. Anthropic — HYBRID Strategy (Static + API)

### API Endpoint (Available since 2025)

```
GET https://api.anthropic.com/v1/models
x-api-key: sk-ant-...
anthropic-version: 2023-06-01
```

**Returns**: Model list with IDs, display names, and creation dates.

### Strategy

1. **Primary**: Use static registry (Anthropic API returns minimal metadata)
2. **Secondary**: Call `/v1/models` to confirm which models the API key has access to
3. **Merge**: Mark models as `available: true/false` based on API response

### Static Registry (July 2026)

```
Model ID                          Context  MaxOut  Vision  Tools  Thinking
──────────────────────────────────────────────────────────────────────────
claude-opus-4-20250514            200K     32K     ✅      ✅     ✅
claude-sonnet-4-20250514          200K     64K     ✅      ✅     ✅
claude-sonnet-4-5-20250929        200K     16K     ✅      ✅     ✅
claude-haiku-4-20250514           200K     8K      ✅      ✅     ❌
claude-3-5-sonnet-20241022        200K     8K      ✅      ✅     ❌
claude-3-5-haiku-20241022         200K     8K      ✅      ✅     ❌
claude-3-opus-20240229            200K     4K      ✅      ✅     ❌
```

### Implementation Notes

- Anthropic uses `anthropic-version` header (required)
- Extended thinking requires `anthropic-beta: interleaved-thinking-2025-05-14` header
- Computer use requires `anthropic-beta: computer-use-2025-01-24` header
- Prompt caching doesn't need beta header anymore (GA since late 2025)
- Max output increased to 64K for Claude Sonnet 4 with extended thinking enabled

---

## 3. Google Gemini — DYNAMIC Strategy

### API Endpoint

```
GET https://generativelanguage.googleapis.com/v1beta/models?key=AIza...
```

**Returns**: Rich metadata per model — this is the **best discovery API** of all providers.

### Response Shape

```json
{
  "models": [
    {
      "name": "models/gemini-2.5-flash",
      "displayName": "Gemini 2.5 Flash",
      "description": "...",
      "inputTokenLimit": 1048576,
      "outputTokenLimit": 65536,
      "supportedGenerationMethods": ["generateContent", "countTokens"],
      "temperature": 2.0,
      "maxTemperature": 2.0,
      "topP": 0.95,
      "topK": 64
    }
  ]
}
```

### Strategy

1. Call `/v1beta/models` — returns everything we need
2. Map API response fields **directly** (no version inference):
   - `inputTokenLimit` → `context_length` (exact value from API)
   - `outputTokenLimit` → `max_output_tokens` (exact value from API)
   - `supportedGenerationMethods` includes `"generateContent"` → model_type: Chat
   - `supportedGenerationMethods` includes `"embedContent"` → model_type: Embedding
   - `thinking: true` field in response → `supports_thinking` (present since Gemini 2.5)
3. For capabilities NOT in the API response, use the codebase `ModelProfile` struct
   which already encodes thinking budget ranges and context lengths per prefix

### Capability Mapping (Direct, No Heuristics)

```
API Response Field                         → DiscoveredModel Field
─────────────────────────────────────────────────────────────────────
inputTokenLimit                            → context_length
outputTokenLimit                           → max_output_tokens
supportedGenerationMethods has "embed..."  → ModelType::Embedding
supportedGenerationMethods has "generate." → ModelType::Llm
thinking: true                             → supports_thinking = true
(all Gemini chat models support tools)     → supports_function_calling = true
                                             (verified: Gemini API reference, July 2026)
```

> **Note**: Vision is implicit for all Gemini chat models (1.5+). This is NOT
> inferred from the model name — it is documented in the official Gemini API
> reference: "All Gemini models accept image, video, and audio input."

### Implementation Notes

- `/v1beta/models` works without API key for listing (but not for content generation)
- Response includes deprecated models — check for `deprecated` field
- This section covers Google AI Studio only — see section 3b for Vertex AI

---

## 3b. Google Vertex AI — DYNAMIC Strategy (Separate Provider)

> **ADDED 2026-07-04**: Vertex AI is a **distinct provider** from Google AI Studio.
> They share the same Gemini models but use completely different auth, endpoints,
> billing, and quota systems. The codebase correctly separates them as
> `ProviderType::Gemini` vs `ProviderType::VertexAI` (see `factory.rs:64-82`).

### Why Vertex AI Is Not Just "Gemini With Different Auth"

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                Google AI Studio vs Vertex AI                     │
  ├────────────────────┬────────────────────┬───────────────────────┤
  │  Dimension         │  Google AI Studio  │  Vertex AI            │
  ├────────────────────┼────────────────────┼───────────────────────┤
  │  Auth              │  API key           │  OAuth2 / ADC / SA    │
  │                    │  (GEMINI_API_KEY)  │  (no API key)         │
  │  Endpoint          │  generativelanguage│  {region}-aiplatform  │
  │                    │  .googleapis.com   │  .googleapis.com      │
  │  Billing           │  Free tier + PAYG  │  GCP project billing  │
  │  Quota             │  20 RPM free tier  │  Per-project quota    │
  │  Model listing API │  /v1beta/models    │  publishers/google/   │
  │                    │                    │  models               │
  │  Response format   │  Model{}           │  PublisherModel{}     │
  │  Models available  │  Google models     │  Google + 3rd-party   │
  │                    │                    │  (DeepSeek, Llama...) │
  │  Global endpoint   │  Always global     │  Regional or global   │
  │  Image generation  │  Imagen via API    │  Imagen via predict   │
  └────────────────────┴────────────────────┴───────────────────────┘
```

### Authentication Approaches (No API Key)

Vertex AI does NOT use API keys. Authentication is via OAuth2 access tokens
obtained through one of these methods (ordered by recommendation):

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  AUTH METHOD                     │  WHERE TO USE               │
  ├─────────────────────────────────┼─────────────────────────────┤
  │  1. Attached service account     │  Cloud Run, GCE, GKE, CF   │
  │     (automatic via metadata      │  → ADC auto-discovers it   │
  │     server — zero config)        │                             │
  │                                  │                             │
  │  2. Workload Identity Federation │  AWS, Azure, on-prem,      │
  │     (keyless cross-cloud auth)   │  GitHub Actions, GitLab CI │
  │     Set GOOGLE_APPLICATION_      │  → Exchange external IdP   │
  │     CREDENTIALS to cred config   │  token for GCP token       │
  │                                  │                             │
  │  3. gcloud CLI user credentials  │  Local development         │
  │     gcloud auth application-     │  → Stores ADC creds in     │
  │     default login                │  ~/.config/gcloud/          │
  │                                  │                             │
  │  4. Service account key file     │  Legacy / not recommended  │
  │     GOOGLE_APPLICATION_          │  → JSON key file           │
  │     CREDENTIALS=/path/to/sa.json │                             │
  │                                  │                             │
  │  5. Explicit access token        │  CI/CD, scripts            │
  │     GOOGLE_ACCESS_TOKEN env var  │  → Short-lived token       │
  └─────────────────────────────────┴─────────────────────────────┘
```

**Source**: https://docs.cloud.google.com/vertex-ai/docs/authentication

### Current Codebase Auth Flow (Proved by Code)

From `src/providers/gemini.rs:806-824`:

```rust
pub fn from_env_vertex_ai() -> Result<Self> {
    let project_id = std::env::var("GOOGLE_CLOUD_PROJECT")?;       // Required
    let region = std::env::var("GOOGLE_CLOUD_REGION")
        .unwrap_or_else(|_| "us-central1".to_string());            // Optional
    let access_token = match std::env::var("GOOGLE_ACCESS_TOKEN") {
        Ok(token) if !token.is_empty() => token,
        _ => Self::get_access_token_from_gcloud()?,                // gcloud CLI
    };
    Ok(Self::vertex_ai(project_id, region, access_token))
}
```

**Roadblock**: The current implementation ONLY supports methods 3 and 5 from the
table above. It shells out to `gcloud auth print-access-token` as a fallback.
Methods 1 (attached SA), 2 (WIF), and 4 (SA key file) are NOT supported because
the code does not use the Google Auth library — it only stores a raw `access_token`
string in the `GeminiEndpoint::VertexAI` struct. See RB-13 below.

### Model Listing API Endpoints

Vertex AI has **two different model listing APIs**:

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  1. Publisher Models (Model Garden)                              │
  │                                                                  │
  │  GET https://{region}-aiplatform.googleapis.com/v1/              │
  │      publishers/google/models                                    │
  │  Auth: Bearer $(gcloud auth print-access-token)                  │
  │  Returns: PublisherModel{} — Model Garden metadata               │
  │  Fields: name, supportedActions, launchStage, frameworks         │
  │  Missing: inputTokenLimit, outputTokenLimit, thinking            │
  │  Use for: Checking which models are available in your project    │
  │                                                                  │
  │  2. Gemini Unified API (same as Google AI Studio)                │
  │                                                                  │
  │  GET https://{region}-aiplatform.googleapis.com/v1beta/          │
  │      models                                                      │
  │  Auth: Bearer $(gcloud auth print-access-token)                  │
  │  Returns: Model{} — rich capability metadata                     │
  │  Fields: inputTokenLimit, outputTokenLimit, thinking,            │
  │          supportedGenerationMethods, temperature, topK, topP     │
  │  Use for: Getting capability metadata for discovery              │
  └──────────────────────────────────────────────────────────────────┘
```

**Critical distinction**: The publisher models API (endpoint 1) returns a
`PublisherModel` resource with `supportedActions` (deploy, fine-tune, etc.) — NOT
model capabilities. The unified Gemini API (endpoint 2) returns the same rich
`Model` resource as Google AI Studio with `inputTokenLimit`, `outputTokenLimit`,
`supportedGenerationMethods`, and `thinking`.

### Current Codebase list_models() (Proved by Code)

From `src/providers/gemini.rs:1109-1154`:

The existing `list_models()` method already handles both endpoints:
- Google AI: `GET {GEMINI_API_BASE}/models?key={api_key}`
- Vertex AI: `GET https://{host}/v1/projects/{project}/locations/{region}/publishers/google/models`

**Roadblock**: The Vertex path uses the **publisher models** endpoint (endpoint 1)
which returns `PublisherModel{}` objects, NOT the unified Gemini `Model{}` objects.
The response will NOT have `inputTokenLimit`, `outputTokenLimit`, or `thinking`
fields. The code tries to deserialize both into `GeminiModelInfo` which expects
`inputTokenLimit` — this will silently produce `None` for Vertex AI results.

### Vertex AI Discovery Strategy

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  STRATEGY: HYBRID                                                │
  │                                                                  │
  │  Primary:  Unified Gemini API endpoint (v1beta/models)           │
  │            → Same rich Model{} response as Google AI Studio      │
  │            → inputTokenLimit, outputTokenLimit, thinking         │
  │                                                                  │
  │  Fallback: Publisher models endpoint + static ModelProfile data  │
  │            → When v1beta/models is unavailable in region         │
  │            → Use existing ModelProfile for context_length etc.   │
  │                                                                  │
  │  Extra:    3rd-party models on Vertex (DeepSeek, Llama, etc.)   │
  │            → Only available via publisher models endpoint         │
  │            → Static registry for capability data                 │
  └──────────────────────────────────────────────────────────────────┘
```

### Capability Mapping (Direct, No Heuristics)

Same as Google AI Studio (section 3) — the unified Gemini API returns identical
`Model{}` objects regardless of whether accessed via Google AI or Vertex AI:

```
API Response Field                         → DiscoveredModel Field
─────────────────────────────────────────────────────────────────────
inputTokenLimit                            → context_length
outputTokenLimit                           → max_output_tokens
supportedGenerationMethods has "generate." → ModelType::Llm
supportedGenerationMethods has "embed..."  → ModelType::Embedding
thinking: true                             → supports_thinking = true
```

### Global vs Regional Endpoint Routing

Gemini 3.x models require the **global** endpoint:
- Regional: `us-central1-aiplatform.googleapis.com` → HTTP 404 for Gemini 3.x
- Global: `aiplatform.googleapis.com` → works for all models

The codebase handles this via `ModelProfile.requires_global_vertex`:

```rust
// From gemini.rs:1302-1314
let effective_region: &str = if profile.requires_global_vertex {
    "global"                        // → aiplatform.googleapis.com
} else {
    region.as_str()                 // → us-central1-aiplatform.googleapis.com
};
```

For model listing, the discovery implementation should:
1. Try the regional endpoint first (covers 2.5 and earlier models)
2. If the user's region is not `"global"`, also try the global endpoint
   (covers 3.x models that may not appear in regional listing)
3. Deduplicate by model ID

### Implementation Notes

- Vertex AI access tokens expire (default 1 hour). Discovery must handle 401
  responses by refreshing the token via `get_access_token_from_gcloud()`
- The `GeminiModelInfo` struct is missing the `thinking: bool` field from the API
  response — this must be added (see RB-14)
- Vertex AI supports 3rd-party models (DeepSeek, Llama, Mistral via Model Garden)
  that are NOT available on Google AI Studio. These are listed as publisher models
  with provider-specific metadata. Discovery should surface them with
  `DiscoverySource::DynamicApi` but their capabilities come from their own
  provider's static registry
- The `GOOGLE_CLOUD_PROJECT` env var is REQUIRED for Vertex AI discovery
- Image generation on Vertex AI (Imagen 4.0) uses a separate predict endpoint
  (`src/imagegen/providers/vertexai.rs`) — discovery should include image models

---

## 4. Ollama — DYNAMIC Strategy

### API Endpoints

```
GET  http://localhost:11434/api/tags          → list installed models
POST http://localhost:11434/api/show          → model details (per-model)
```

### Strategy (REVISED 2026-07-04 — NO HEURISTICS)

> **CORRECTION**: Since Ollama 0.30.0, `/api/tags` returns a `capabilities` array
> directly in each model object. This **eliminates the need** for parsing
> modelfiles, family-based inference, or template inspection.

1. Call `/api/tags` — models now include `capabilities[]` directly
2. Map `capabilities[]` values **directly** to `ModelCapabilities` fields
3. For `context_length` only: call `/api/show` (returns `model_info.{arch}.context_length`)
4. Cache `/api/show` results per model (expensive call)

### `/api/tags` Response (Ollama 0.30.0+)

```json
{
  "models": [
    {
      "name": "gemma4:27b",
      "model": "gemma4:27b",
      "size": 17241792000,
      "capabilities": ["completion", "vision", "tools", "thinking"],
      "details": {
        "family": "gemma4",
        "parameter_size": "27.4B",
        "quantization_level": "Q4_K_M"
      }
    },
    {
      "name": "nomic-embed-text:latest",
      "model": "nomic-embed-text:latest",
      "size": 274302450,
      "capabilities": ["embedding"],
      "details": {
        "family": "nomic-bert",
        "parameter_size": "137M"
      }
    }
  ]
}
```

### Capability Mapping (Direct, No Heuristics)

```
capabilities[] value   → DiscoveredModel Field
─────────────────────────────────────────────────────────
"completion"          → model_type = Llm
"embedding"           → model_type = Embedding
"vision"              → supports_vision = true
"tools"               → supports_function_calling = true
"thinking"            → supports_thinking = true
"image"               → supports_image_generation = true
"insert"              → (fill-in-the-middle, tracked in tags)
```

All capabilities NOT in the array are `false`. No guessing.

### `/api/show` — Context Length Only

```json
{
  "model_info": {
    "general.architecture": "gemma4",
    "gemma4.context_length": 131072
  }
}
```

The field key is `{architecture}.context_length`. This is the ONLY
reason to call `/api/show` — everything else comes from `/api/tags`.

### Implementation Notes

- `/api/show` is expensive — cache per `(model_name, model_digest)` pair
- Digest from `/api/tags` serves as cache-invalidation key (changes on model update)
- Default num_ctx can be overridden at runtime but discovery reports the model's max
- Ollama Cloud (`OLLAMA_API_KEY`) uses same endpoints with Bearer auth
- Cache TTL should be short (60s) — new models appear after `ollama pull`
- **No family-based inference** — the `capabilities[]` array is the source of truth

---

## 5. LM Studio — DYNAMIC Strategy

### API Endpoints

```
GET http://localhost:1234/v1/models           → OpenAI-compatible model list
GET http://localhost:1234/api/v1/models        → Native model list (richer)
```

### Strategy

1. Prefer native `/api/v1/models` endpoint (returns `max_context_length`)
2. Fall back to OpenAI-compatible `/v1/models`
3. Use `refresh_model_metadata()` pattern already in codebase

### Native API Response

```json
{
  "data": [
    {
      "id": "gemma2-9b-it",
      "object": "model",
      "type": "llm",
      "publisher": "google",
      "arch": "gemma2",
      "compatibility_type": "gguf",
      "quantization": "Q4_K_M",
      "max_context_length": 8192,
      "default_max_completion_tokens": 4096
    }
  ]
}
```

### Implementation Notes

- LM Studio 0.3+ exposes rich `/api/v1/models` endpoint
- Loaded models vs downloaded models — differentiate in `available` field
- Tool calling support depends on model, not LM Studio itself
- Vision support depends on model architecture (CLIP-based multimodal)

---

## 6. OpenRouter — DYNAMIC Strategy

### API Endpoint

```
GET https://openrouter.ai/api/v1/models
```

**Returns**: The richest model metadata of any provider (200+ models).

### Response Shape

```json
{
  "data": [
    {
      "id": "anthropic/claude-3.5-sonnet",
      "name": "Claude 3.5 Sonnet",
      "context_length": 200000,
      "pricing": {
        "prompt": "0.000003",
        "completion": "0.000015"
      },
      "architecture": {
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"]
      },
      "top_provider": {
        "max_completion_tokens": 8192,
        "is_moderated": false
      }
    }
  ]
}
```

### Strategy

1. Call `/api/v1/models` (no auth required for listing)
2. Parse `architecture.modality` for vision/audio support
3. Parse `pricing` for cost data
4. Already implemented with caching — adapt to `DiscoveredModel`

### Capability Mapping (Direct, No Heuristics)

> **REVISED 2026-07-04**: OpenRouter's `supported_parameters` field (array of 
> strings) directly declares which parameters each model accepts. This eliminates
> the need for family-based heuristic inference.

```
API Response Field                              → DiscoveredModel Field
────────────────────────────────────────────────────────────────────────
context_length                                  → context_length
top_provider.max_completion_tokens              → max_output_tokens
architecture.input_modalities contains "image"  → supports_vision = true
architecture.input_modalities contains "audio"  → supports_audio_input = true
supported_parameters contains "tools"           → supports_function_calling = true
supported_parameters contains "reasoning"       → supports_thinking = true
pricing.prompt (per-token string)               → cost_per_m_input
pricing.completion (per-token string)           → cost_per_m_output
```

### Implementation Notes

- `supported_parameters` field MUST be added to the existing `ModelInfo` struct
  (current codebase does not parse this field — see `openrouter.rs`)
- No family-based inference needed — `supported_parameters` is the API truth
- Already has excellent caching via `list_models_cached()`

---

## 7. Mistral — HYBRID Strategy

### API Endpoint

```
GET https://api.mistral.ai/v1/models
Authorization: Bearer ...
```

**Returns**: Model IDs and basic metadata.

### Strategy

1. Call `/v1/models` for live model list
2. Enrich with static capability registry (context lengths, vision, etc.)

### Static Registry (July 2026)

```
Model ID                  Context  Vision  Tools  Thinking  Embed
──────────────────────────────────────────────────────────────────
mistral-small-latest      32K      ❌      ✅     ❌        ❌
mistral-medium-latest     128K     ❌      ✅     ❌        ❌
mistral-large-latest      128K     ❌      ✅     ❌        ❌
codestral-latest          256K     ❌      ✅     ❌        ❌
magistral-small-latest    128K     ❌      ✅     ✅        ❌
magistral-medium-latest   128K     ❌      ✅     ✅        ❌
devstral-latest           128K     ❌      ✅     ❌        ❌
pixtral-large-latest      128K     ✅      ✅     ❌        ❌
mistral-embed             8K       ❌      ❌     ❌        ✅
```

---

## 8. NVIDIA NIM — HYBRID Strategy

### API Endpoint

```
GET https://integrate.api.nvidia.com/v1/models
Authorization: Bearer nvapi-...
```

### Strategy

Already well-implemented — has both dynamic `list_models()` and a rich static catalog.

### Key Models (July 2026)

```
nvidia/llama-3.3-nemotron-super-49b-v1   128K  thinking  free
nvidia/nemotron-3-nano-30b-a3b           1M    thinking  free
nvidia/nemotron-3-super-120b-a12b        1M    thinking  free
deepseek-ai/deepseek-v4-flash            64K   thinking  free
meta/llama-4-maverick-17b-128e-instruct  1M    vision    free
qwen/qwq-32b                            128K  thinking  free
moonshotai/kimi-k2-instruct              128K  —         free
```

---

## 9. xAI Grok — STATIC Strategy

### API Endpoint

```
GET https://api.x.ai/v1/models
```

The xAI API does have a models endpoint but returns minimal metadata.

### Strategy

Static registry is the most reliable approach.

### Static Registry (July 2026)

```
Model ID              Context   Vision  Tools  Thinking
───────────────────────────────────────────────────────
grok-3                131K      ✅      ✅     ✅
grok-3-mini           131K      ❌      ✅     ✅
grok-4                131K      ✅      ✅     ✅
grok-4-mini           131K      ❌      ✅     ✅
grok-2-vision-1212    32K       ✅      ❌     ❌
grok-2-1212           131K      ❌      ✅     ❌
```

---

## 10. Azure OpenAI — HYBRID Strategy (Deployment-Centric)

> **ADDED 2026-07-04**: Azure OpenAI uses a deployment model — users create named
> deployments backed by specific models. Discovery must map deployments to capabilities.

### API Endpoints

```
Data Plane (model catalog):
  GET https://{resource}.openai.azure.com/openai/models?api-version=2024-10-21
  Auth: api-key header
  Returns: Model objects with capabilities{} (chat_completion, embeddings, etc.)

Management Plane (active deployments):
  GET https://management.azure.com/subscriptions/{sub}/resourceGroups/{rg}/
      providers/Microsoft.CognitiveServices/accounts/{acct}/deployments
      ?api-version=2025-06-01
  Auth: Azure AD Bearer token (NOT api-key)
  Returns: Deployment objects with model name + version
```

### Strategy

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  STRATEGY: HYBRID (Deployment Mapping)                           │
  │                                                                  │
  │  1. Data plane /openai/models returns model catalog              │
  │     → capabilities{}, lifecycle_status, deprecation dates        │
  │     → But NO context_length or max_output_tokens                 │
  │                                                                  │
  │  2. Management plane lists actual deployments                    │
  │     → Maps deployment_name → model_name + version                │
  │     → But requires Azure AD token (not just api-key)             │
  │                                                                  │
  │  3. Static registry provides context lengths + capabilities      │
  │     → Deployment name mapped to model → lookup in registry       │
  │     → Source: Microsoft Learn docs (models sold by Azure)        │
  │                                                                  │
  │  Fallback: If management API unavailable, use data plane models  │
  │  + static registry only. User's deployment name ≈ model name.   │
  └──────────────────────────────────────────────────────────────────┘
```

### Capability Mapping (Direct, No Heuristics)

```
Data Plane Response Field                  → DiscoveredModel Field
─────────────────────────────────────────────────────────────────────
capabilities.chat_completion: true         → ModelType::Llm
capabilities.embeddings: true              → ModelType::Embedding
capabilities.inference: true               → available = true
lifecycle_status: "generally-available"    → deprecated = false
lifecycle_status: "preview"               → tags: ["preview"]
deprecation.inference (unix timestamp)     → deprecated_at
```

Context lengths and max output tokens are NOT in the API response — they come
from the static registry with citations to official Microsoft docs.

### Roadblock: Deployment Names ≠ Model IDs

Azure deployments have user-chosen names. A deployment named `my-gpt4` could be
backed by `gpt-4o` or `gpt-4.1`. The management API reveals the actual model,
but requires Azure AD auth. Without it, the discovery system can only assume
the deployment name contains the model name (which is the common convention).

### Implementation Notes

- API version `2024-10-21` is the latest stable; `2025-06-01` is for management plane
- Data plane `GET /openai/models` works with just the API key (no Azure AD needed)
- The existing `AzureOpenAIProvider` uses `async-openai` with `AzureConfig` — discovery can use the same crate's `models().list()` method
- Azure supports both Entra ID (Azure AD) and API key auth
- Vision support depends on the underlying model, not the deployment

---

## 11. AWS Bedrock — HYBRID Strategy (Feature-Gated)

> **ADDED 2026-07-04**: Bedrock provides `ListFoundationModels` API which returns
> model IDs, modalities, and lifecycle status. Capability details require static enrichment.

### API Endpoints

```
Model Discovery:
  GET /foundation-models
  Service: bedrock (NOT bedrock-runtime)
  Auth: AWS SigV4 (IAM credentials)
  Returns: FoundationModelSummary[] with modelId, modalities, lifecycle

Model Details:
  GET /foundation-model/{modelId}
  Service: bedrock
  Returns: Full model details for a specific model

OpenAI-Compatible:
  GET /models  (on bedrock-mantle endpoint)
  Returns: OpenAI-format model list
```

### Strategy

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  STRATEGY: HYBRID (API + Static Enrichment)                      │
  │  FEATURE GATE: #[cfg(feature = "bedrock")]                       │
  │                                                                  │
  │  1. ListFoundationModels API provides:                           │
  │     → modelId, modelName, providerName                           │
  │     → inputModalities: ["TEXT", "IMAGE"]                         │
  │     → outputModalities: ["TEXT"]                                 │
  │     → responseStreamingSupported: true/false                     │
  │     → modelLifecycle: { status, endOfLifeTime }                  │
  │     → inferenceTypesSupported: ["ON_DEMAND", "PROVISIONED"]      │
  │                                                                  │
  │  2. Static registry provides (API does NOT return these):        │
  │     → context_length (from AWS docs per model)                   │
  │     → max_output_tokens                                          │
  │     → supports_function_calling (Converse tool config support)   │
  │     → supports_thinking (additionalModelRequestFields support)   │
  │                                                                  │
  │  3. Inference profile resolution is REQUIRED for some models     │
  │     → bare `anthropic.claude-sonnet-5` must become               │
  │       `us.anthropic.claude-sonnet-5` in us-east-1                │
  └──────────────────────────────────────────────────────────────────┘
```

### Capability Mapping (Direct, No Heuristics)

```
API Response Field                         → DiscoveredModel Field
─────────────────────────────────────────────────────────────────────
modelId                                    → id
modelName                                  → name
providerName                               → tags: ["provider:{name}"]
inputModalities contains "IMAGE"           → supports_vision = true
outputModalities contains "IMAGE"          → supports_image_generation = true
outputModalities contains "EMBEDDING"      → ModelType::Embedding
responseStreamingSupported                 → supports_streaming
modelLifecycle.status = "ACTIVE"           → available = true
modelLifecycle.status = "LEGACY"           → deprecated = true
modelLifecycle.endOfLifeTime               → deprecated_at
```

Tool calling, thinking support, context_length, and max_output_tokens are NOT in the
`ListFoundationModels` response — they come from the static registry.

### Implementation Notes

- `ListFoundationModels` uses the `bedrock` service client (NOT `bedrock-runtime`)
- The codebase currently only imports `aws_sdk_bedrockruntime` — adding discovery requires also depending on `aws_sdk_bedrock`
- Feature gate must apply to both discovery code AND the `aws_sdk_bedrock` dependency
- Inference profile resolution (`resolve_model_id_for_region`) already exists and should be reused
- New Claude models (Fable 5, Sonnet 5, Opus 4.8) use simplified IDs without date suffixes: `anthropic.claude-sonnet-5` instead of `anthropic.claude-sonnet-5-YYYYMMDD-v1:0`
- Bedrock also now supports OpenAI's open-weight models (`openai.gpt-oss-120b`, `openai.gpt-oss-20b`) and 100+ Model Marketplace models

---

## 12. OpenAI-Compatible — TRY-DYNAMIC Strategy

### Strategy

1. **Attempt** `GET {base_url}/models` (many OAI-compat APIs support this)
2. If successful, return model IDs (minimal metadata)
3. If 404/failure, return empty list with warning
4. Agent harness can manually register known capabilities

### Known Compatible APIs

| Service | `/v1/models` | Rich Metadata | Notes |
|---------|-------------|--------------|-------|
| Groq | ✅ | ✅ (context, active) | Rich response |
| Together AI | ✅ | ✅ (context, type) | Rich response |
| DeepSeek | ✅ | ⚠️ (IDs only) | Minimal |
| Fireworks | ✅ | ✅ | Rich response |
| Perplexity | ❌ | ❌ | No models endpoint |
| Cohere | ✅ | ✅ | Different API format |

---

## 13. VSCode Copilot — DYNAMIC Strategy

### Existing Implementation

The VSCode Copilot provider already has `ModelsResponse` and `Model` types for catalog queries via the GitHub Copilot API.

### Strategy

Use existing `list_models()` implementation, normalize to `DiscoveredModel`.

---

## Summary: Strategy Assignment

```
  ┌───────────────────────────────────────────────────────┐
  │        Provider Discovery Strategy Summary             │
  ├──────────────────┬────────────┬────────────────────────┤
  │ Provider         │ Strategy   │ API Endpoint            │
  ├──────────────────┼────────────┼────────────────────────┤
  │ OpenAI           │ HYBRID     │ /v1/models + static    │
  │ Anthropic        │ HYBRID     │ /v1/models + static    │
  │ Gemini           │ DYNAMIC    │ /v1beta/models         │
  │ Vertex AI        │ HYBRID     │ /v1beta/models + publs │
  │ Ollama           │ DYNAMIC    │ /api/tags + /api/show  │
  │ LM Studio        │ DYNAMIC    │ /api/v1/models         │
  │ OpenRouter       │ DYNAMIC    │ /api/v1/models         │
  │ Mistral          │ HYBRID     │ /v1/models + static    │
  │ NVIDIA NIM       │ HYBRID     │ /v1/models + static    │
  │ xAI              │ STATIC     │ (built-in registry)    │
  │ Azure OpenAI     │ HYBRID     │ /openai/models + static│
  │ AWS Bedrock      │ HYBRID     │ AWS SDK + static       │
  │ OAI-Compatible   │ TRY-DYN    │ /v1/models (try)       │
  │ HuggingFace      │ STATIC     │ (built-in registry)    │
  │ VSCode Copilot   │ DYNAMIC    │ Copilot catalog API    │
  └──────────────────┴────────────┴────────────────────────┘
```
