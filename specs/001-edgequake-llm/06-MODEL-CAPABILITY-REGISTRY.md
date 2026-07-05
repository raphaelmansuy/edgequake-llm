# 06 — Model Capability Registry & Type System

> **Lens**: AI Engineer  
> **Date**: 2026-07-04  
> **Status**: DRAFT  
> **Purpose**: Definitive model registry for all providers as of July 2026

## Type System Design

### Capability Taxonomy

```
  Model Capability Taxonomy:

  ┌─────────────────────────────────────────────────────────┐
  │                   ModelCapabilities                       │
  ├─────────────────────────────────────────────────────────┤
  │                                                          │
  │  ┌─ Token Limits ──────────────────────────────────┐    │
  │  │  context_length: usize      (max input tokens)  │    │
  │  │  max_output_tokens: usize   (max output tokens) │    │
  │  │  embedding_dimension: usize (for embed models)  │    │
  │  │  max_embedding_tokens: usize                    │    │
  │  └─────────────────────────────────────────────────┘    │
  │                                                          │
  │  ┌─ Input Modalities ──────────────────────────────┐    │
  │  │  supports_vision: bool       (image input)       │    │
  │  │  supports_audio_input: bool  (audio input)       │    │
  │  │  supports_document_input: bool (PDF input)       │    │
  │  │  input_modalities: Vec<str>  (["text","image"])  │    │
  │  └─────────────────────────────────────────────────┘    │
  │                                                          │
  │  ┌─ Output Capabilities ───────────────────────────┐    │
  │  │  supports_streaming: bool                        │    │
  │  │  supports_function_calling: bool (tool use)      │    │
  │  │  supports_json_mode: bool   (json output)        │    │
  │  │  supports_structured_output: bool (JSON schema)  │    │
  │  │  supports_image_generation: bool                 │    │
  │  │  output_modalities: Vec<str> (["text"])          │    │
  │  └─────────────────────────────────────────────────┘    │
  │                                                          │
  │  ┌─ Reasoning/Thinking ────────────────────────────┐    │
  │  │  supports_thinking: bool                         │    │
  │  │  thinking_budget_range: Option<(min,max)>        │    │
  │  │  default_temperature: f32                        │    │
  │  └─────────────────────────────────────────────────┘    │
  │                                                          │
  │  ┌─ Advanced Features ─────────────────────────────┐    │
  │  │  supports_system_message: bool                   │    │
  │  │  supports_prompt_caching: bool                   │    │
  │  │  supports_web_search: bool                       │    │
  │  │  supports_computer_use: bool                     │    │
  │  │  max_images: Option<usize>                       │    │
  │  └─────────────────────────────────────────────────┘    │
  └─────────────────────────────────────────────────────────┘
```

### Backward Compatibility Check

All new fields use `#[serde(default)]` with `false`/`None` defaults. Existing `ModelCapabilities` struct is extended, not replaced. Existing serialized TOML configs deserialize without issues.

---

## Complete Model Registry (July 2026)

### OpenAI Models

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  OPENAI MODEL REGISTRY — July 2026                                           │
  ├──────────────────────┬────────┬────────┬───────┬───────┬────────┬───────────┤
  │ Model ID             │Ctx In  │Max Out │Vision │Tools  │Think   │$/M In Out │
  ├──────────────────────┼────────┼────────┼───────┼───────┼────────┼───────────┤
  │ gpt-4.1              │  1M    │ 32K    │  ✅   │  ✅   │  ❌    │ 2.0  8.0  │
  │ gpt-4.1-mini         │  1M    │ 32K    │  ✅   │  ✅   │  ❌    │ 0.4  1.6  │
  │ gpt-4.1-nano         │  1M    │ 32K    │  ✅   │  ✅   │  ❌    │ 0.1  0.4  │
  │ gpt-4o               │ 128K   │ 16K    │  ✅   │  ✅   │  ❌    │ 2.5 10.0  │
  │ gpt-4o-mini          │ 128K   │ 16K    │  ✅   │  ✅   │  ❌    │ 0.15 0.6  │
  │ o4-mini              │ 200K   │100K    │  ✅   │  ✅   │  ✅    │ 1.1  4.4  │
  │ o3                   │ 200K   │100K    │  ✅   │  ✅   │  ✅    │ 2.0  8.0  │
  │ o3-mini              │ 200K   │100K    │  ❌   │  ✅   │  ✅    │ 1.1  4.4  │
  │ o1                   │ 200K   │100K    │  ✅   │  ✅   │  ✅    │15.0 60.0  │
  │ o1-mini              │ 128K   │ 65K    │  ❌   │  ❌   │  ✅    │ 1.1  4.4  │
  ├──────────────────────┼────────┼────────┼───────┼───────┼────────┼───────────┤
  │ EMBEDDINGS           │Ctx     │Dim     │       │       │        │ $/M       │
  ├──────────────────────┼────────┼────────┼───────┼───────┼────────┼───────────┤
  │ text-embedding-3-sm  │ 8K     │ 1536   │       │       │        │ 0.02      │
  │ text-embedding-3-lg  │ 8K     │ 3072   │       │       │        │ 0.13      │
  │ text-embedding-ada   │ 8K     │ 1536   │       │       │        │ 0.10      │
  └──────────────────────┴────────┴────────┴───────┴───────┴────────┴───────────┘

  Notes:
  - GPT-4.1 family: released April 2026, 1M context window
  - o-series: use max_completion_tokens instead of max_tokens
  - o-series: support reasoning_effort parameter ("low","medium","high")
  - All GPT-4.1/4o models: support structured output (strict JSON schema)
```

### Anthropic Claude Models

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  ANTHROPIC MODEL REGISTRY — July 2026                                        │
  ├──────────────────────────────┬────────┬────────┬───────┬───────┬────────────┤
  │ Model ID                     │Ctx In  │Max Out │Vision │Tools  │Think │$/M  │
  ├──────────────────────────────┼────────┼────────┼───────┼───────┼──────┼─────┤
  │ claude-opus-4-20250514       │ 200K   │ 32K    │  ✅   │  ✅   │  ✅  │15/75│
  │ claude-sonnet-4-20250514     │ 200K   │ 64K*   │  ✅   │  ✅   │  ✅  │ 3/15│
  │ claude-sonnet-4-5-20250929   │ 200K   │ 16K    │  ✅   │  ✅   │  ✅  │ 3/15│
  │ claude-haiku-4-20250514      │ 200K   │  8K    │  ✅   │  ✅   │  ❌  │0.8/4│
  │ claude-3-5-sonnet-20241022   │ 200K   │  8K    │  ✅   │  ✅   │  ❌  │ 3/15│
  │ claude-3-5-haiku-20241022    │ 200K   │  8K    │  ✅   │  ✅   │  ❌  │0.8/4│
  │ claude-3-opus-20240229       │ 200K   │  4K    │  ✅   │  ✅   │  ❌  │15/75│
  └──────────────────────────────┴────────┴────────┴───────┴───────┴──────┴─────┘

  * Claude Sonnet 4: 64K output with extended thinking enabled

  Aliases (latest versions):
  - "claude-sonnet-4-5-latest" → claude-sonnet-4-5-20250929
  - "claude-opus-4-latest"     → claude-opus-4-20250514
  - "claude-sonnet-4-latest"   → claude-sonnet-4-20250514
  - "claude-haiku-4-latest"    → claude-haiku-4-20250514

  Features:
  - All models: prompt caching (cache_control), vision (base64+URL)
  - Claude 4+: extended thinking (budget_tokens), tool use
  - Claude 4+: PDF/document input (base64 encoded)
  - Claude 4+: citations in responses
  - anthropic-version: 2023-06-01 (required header)
```

### Google Gemini Models

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  GEMINI MODEL REGISTRY — July 2026                                           │
  ├────────────────────────────────┬────────┬────────┬───────┬───────┬──────────┤
  │ Model ID                       │Ctx In  │Max Out │Vision │Tools  │Think     │
  ├────────────────────────────────┼────────┼────────┼───────┼───────┼──────────┤
  │ gemini-2.5-flash               │  1M    │ 65K    │  ✅   │  ✅   │ budget   │
  │ gemini-2.5-pro                 │  1M    │ 65K    │  ✅   │  ✅   │ budget   │
  │ gemini-3-flash-preview         │  1M    │ 65K    │  ✅   │  ✅   │ level    │
  │ gemini-3.1-pro-preview         │  2M    │ 65K    │  ✅   │  ✅   │ level    │
  │ gemini-3.1-flash-lite-preview  │  1M    │ 65K    │  ✅   │  ✅   │ level    │
  │ gemini-2.0-flash               │  1M    │  8K    │  ✅   │  ✅   │ ❌       │
  ├────────────────────────────────┼────────┼────────┼───────┼───────┼──────────┤
  │ EMBEDDINGS                     │Ctx     │Dim     │       │       │          │
  ├────────────────────────────────┼────────┼────────┼───────┼───────┼──────────┤
  │ text-embedding-004             │ 2K     │ 768    │       │       │          │
  │ embedding-001                  │ 2K     │ 768    │       │       │          │
  └────────────────────────────────┴────────┴────────┴───────┴───────┴──────────┘

  Notes:
  - Gemini 2.5: thinkingConfig.thinkingBudget (token count)
  - Gemini 3.x: thinkingConfig.thinkingLevel ("minimal","low","medium","high")
  - All models: support audio input, code execution
  - Context: 1M for most, 2M for gemini-3.1-pro-preview
  - Discovery API: GET /v1beta/models returns full metadata
  - Source: https://ai.google.dev/gemini-api/docs/models (July 2026)
```

### Vertex AI Models (Google + 3rd Party)

> **ADDED 2026-07-04**: Vertex AI serves the same Google Gemini models as
> Google AI Studio (above) PLUS 3rd-party models via Model Garden. All Google
> models have identical capabilities on both endpoints. This section covers
> only the additional 3rd-party models unique to Vertex AI.

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  VERTEX AI — 3RD-PARTY MODELS (Model Garden) — July 2026                    │
  ├──────────────────────────┬────────┬────────┬───────┬───────┬───────────────┤
  │ Model ID                 │Ctx In  │Max Out │Vision │Tools  │Think          │
  ├──────────────────────────┼────────┼────────┼───────┼───────┼───────────────┤
  │ deepseek-ai/deepseek-r1  │  64K   │  8K    │  ❌   │  ❌   │ ✅ (CoT)     │
  │ meta-llama/llama-4-scout │ 512K   │ 16K    │  ✅   │  ✅   │ ❌           │
  │ meta-llama/llama-4-mav.  │ 1M     │ 16K    │  ✅   │  ✅   │ ❌           │
  │ mistralai/mistral-large  │ 128K   │ 32K    │  ❌   │  ✅   │ ❌           │
  └──────────────────────────┴────────┴────────┴───────┴───────┴───────────────┘

  VERTEX AI — AUTH REQUIREMENTS
  ┌────────────────────────────────────────────────────────────────────┐
  │ All models require: GOOGLE_CLOUD_PROJECT env var                  │
  │ All models use: OAuth2 Bearer token (NOT API key)                 │
  │                                                                    │
  │ Auth flow (current code):                                          │
  │   1. Check GOOGLE_ACCESS_TOKEN env var                             │
  │   2. Fallback: gcloud auth print-access-token                      │
  │   3. Fallback: gcloud auth application-default print-access-token  │
  │                                                                    │
  │ Missing auth flows:                                                │
  │   - ADC auto-detect (metadata server, WIF, SA key)                │
  │   - Token refresh on expiry (1-hour TTL)                          │
  │   - Service account impersonation                                  │
  │                                                                    │
  │ Discovery endpoint:                                                │
  │   Google models: GET /v1beta/models (Bearer auth)                  │
  │   3rd-party: GET /v1/.../publishers/{publisher}/models             │
  │                                                                    │
  │ Source: https://docs.cloud.google.com/vertex-ai/docs/authentication│
  └────────────────────────────────────────────────────────────────────┘

  Notes:
  - Google Gemini models on Vertex AI have IDENTICAL capabilities to Google AI
    Studio — do NOT duplicate entries; discovery reuses the same ModelProfile
  - 3rd-party model capabilities are from their respective providers' docs
  - 3rd-party models on Vertex AI use the MaaS (Model as a Service) pricing
  - Global endpoint (aiplatform.googleapis.com) required for Gemini 3.x
  - Imagen 4.0 on Vertex AI uses a separate predict endpoint, not generateContent
```

### Mistral Models

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  MISTRAL MODEL REGISTRY — July 2026                                          │
  ├─────────────────────────┬────────┬────────┬───────┬───────┬────────┬────────┤
  │ Model ID                │Ctx In  │Max Out │Vision │Tools  │Think   │$/M     │
  ├─────────────────────────┼────────┼────────┼───────┼───────┼────────┼────────┤
  │ mistral-large-latest    │ 128K   │ 32K    │  ❌   │  ✅   │  ❌    │ 2.0/6.0│
  │ mistral-medium-latest   │ 128K   │ 32K    │  ❌   │  ✅   │  ❌    │ 1.0/3.0│
  │ mistral-small-latest    │  32K   │  8K    │  ❌   │  ✅   │  ❌    │ 0.1/0.3│
  │ codestral-latest        │ 256K   │ 32K    │  ❌   │  ✅   │  ❌    │ 0.3/0.9│
  │ magistral-small-latest  │ 128K   │ 40K    │  ❌   │  ✅   │  ✅    │ 0.5/1.5│
  │ magistral-medium-latest │ 128K   │ 40K    │  ❌   │  ✅   │  ✅    │ 2.0/6.0│
  │ devstral-latest         │ 128K   │ 32K    │  ❌   │  ✅   │  ❌    │ 0.1/0.3│
  │ pixtral-large-latest    │ 128K   │ 32K    │  ✅   │  ✅   │  ❌    │ 2.0/6.0│
  ├─────────────────────────┼────────┼────────┼───────┼───────┼────────┼────────┤
  │ EMBEDDINGS              │Ctx     │Dim     │       │       │        │$/M     │
  ├─────────────────────────┼────────┼────────┼───────┼───────┼────────┼────────┤
  │ mistral-embed           │  8K    │ 1024   │       │       │        │ 0.1    │
  └─────────────────────────┴────────┴────────┴───────┴───────┴────────┴────────┘

  Notes:
  - Magistral: reasoning/thinking models (supports extended thinking)
  - Codestral: 256K context (was incorrectly 32K in current config)
  - Pixtral: vision model (image input support)
  - Devstral: coding-optimized
  - All: support safe_prompt, parallel_tool_calls options
```

### xAI Grok Models

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  XAI MODEL REGISTRY — July 2026                                              │
  ├─────────────────────────┬────────┬────────┬───────┬───────┬────────┬────────┤
  │ Model ID                │Ctx In  │Max Out │Vision │Tools  │Think   │$/M     │
  ├─────────────────────────┼────────┼────────┼───────┼───────┼────────┼────────┤
  │ grok-4                  │ 131K   │ 32K    │  ✅   │  ✅   │  ✅    │ 3/15   │
  │ grok-4-mini             │ 131K   │ 16K    │  ❌   │  ✅   │  ✅    │ 0.6/4  │
  │ grok-3                  │ 131K   │ 32K    │  ✅   │  ✅   │  ✅    │ 3/15   │
  │ grok-3-mini             │ 131K   │ 16K    │  ❌   │  ✅   │  ✅    │ 0.3/0.5│
  │ grok-2-1212             │ 131K   │  8K    │  ❌   │  ✅   │  ❌    │ 2/10   │
  │ grok-2-vision-1212      │  32K   │  8K    │  ✅   │  ❌   │  ❌    │ 2/10   │
  └─────────────────────────┴────────┴────────┴───────┴───────┴────────┴────────┘

  Notes:
  - Grok-3/4: OpenAI-compatible API at api.x.ai/v1
  - Grok-3+: extended thinking (reasoning tokens)
  - Vision: image input on Grok-4 and Grok-2-vision
```

### Azure OpenAI Models (July 2026)

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  AZURE OPENAI MODEL REGISTRY — July 2026                                    │
  │  Source: learn.microsoft.com/azure/foundry/foundry-models/concepts/         │
  │          models-sold-directly-by-azure                                       │
  ├──────────────────────────┬────────┬────────┬───────┬───────┬───────────────┤
  │ Model ID                 │Ctx In  │Max Out │Vision │Tools  │Think          │
  ├──────────────────────────┼────────┼────────┼───────┼───────┼───────────────┤
  │ gpt-5.5                  │  1M    │ 32K    │  ✅   │  ✅   │ ✅            │
  │ gpt-5.4                  │ 400K   │ 128K   │  ✅   │  ✅   │ ✅            │
  │ gpt-5.4-pro              │ 400K   │ 128K   │  ✅   │  ✅   │ ✅            │
  │ gpt-5.4-mini             │ 400K   │  16K   │  ✅   │  ✅   │ ✅            │
  │ gpt-5.4-nano             │ 400K   │  16K   │  ✅   │  ✅   │ ✅            │
  │ gpt-5                    │ 400K   │ 128K   │  ✅   │  ✅   │ ✅            │
  │ gpt-5-mini               │ 400K   │  16K   │  ✅   │  ✅   │ ✅            │
  │ gpt-5-nano               │ 400K   │  16K   │  ✅   │  ✅   │ ✅            │
  │ gpt-4.1                  │  1M    │  32K   │  ✅   │  ✅   │ ❌            │
  │ gpt-4.1-mini             │  1M    │  32K   │  ✅   │  ✅   │ ❌            │
  │ gpt-4.1-nano             │  1M    │  32K   │  ✅   │  ✅   │ ❌            │
  │ gpt-4o                   │ 128K   │  16K   │  ✅   │  ✅   │ ❌            │
  │ gpt-4o-mini              │ 128K   │  16K   │  ✅   │  ✅   │ ❌            │
  │ o4-mini                  │ 200K   │ 100K   │  ✅   │  ✅   │ ✅ (reason)   │
  │ o3                       │ 200K   │ 100K   │  ✅   │  ✅   │ ✅ (reason)   │
  │ o3-mini                  │ 200K   │ 100K   │  ❌   │  ✅   │ ✅ (reason)   │
  │ o3-pro                   │ 200K   │ 100K   │  ❌   │  ❌   │ ✅ (reason)   │
  │ o1                       │ 200K   │ 100K   │  ✅   │  ❌   │ ✅ (reason)   │
  ├──────────────────────────┼────────┼────────┼───────┼───────┼───────────────┤
  │ EMBEDDINGS               │Ctx     │Dim     │       │       │               │
  ├──────────────────────────┼────────┼────────┼───────┼───────┼───────────────┤
  │ text-embedding-3-large   │  8K    │ 3072   │       │       │               │
  │ text-embedding-3-small   │  8K    │ 1536   │       │       │               │
  │ text-embedding-ada-002   │  8K    │ 1536   │       │       │               │
  └──────────────────────────┴────────┴────────┴───────┴───────┴───────────────┘

  Notes:
  - Azure uses DEPLOYMENTS not model IDs — deployment name ≠ model ID
  - Context/capabilities depend on the underlying model, not deployment settings
  - o-series models require max_completion_tokens (not max_tokens)
  - GPT-5.x models support both Chat Completions API and Responses API
  - API version must be 2025-06-01+ for GPT-5.x access
  - Discovery: GET /openai/models returns capabilities{}, not context lengths
```

### AWS Bedrock Models (July 2026)

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  AWS BEDROCK MODEL REGISTRY — July 2026                                     │
  │  Source: docs.aws.amazon.com/bedrock/latest/userguide/                       │
  │          conversation-inference-supported-models-features.html               │
  ├──────────────────────────────────────────┬──────┬──────┬─────┬─────┬────────┤
  │ Model ID (Bedrock)                       │Ctx   │Out   │Vis  │Tool │Think   │
  ├──────────────────────────────────────────┼──────┼──────┼─────┼─────┼────────┤
  │ anthropic.claude-sonnet-5                │ 1M   │128K  │ ✅  │ ✅  │ ✅     │
  │ anthropic.claude-fable-5                 │ 1M   │128K  │ ✅  │ ✅  │ ✅     │
  │ anthropic.claude-opus-4-8                │ 1M   │128K  │ ✅  │ ✅  │ ✅     │
  │ anthropic.claude-opus-4-7                │ 1M   │128K  │ ✅  │ ✅  │ ✅     │
  │ anthropic.claude-sonnet-4-6              │ 1M   │128K  │ ✅  │ ✅  │ ✅     │
  │ anthropic.claude-opus-4-6               │ 1M   │128K  │ ✅  │ ✅  │ ✅     │
  │ anthropic.claude-haiku-4-5-*-v1:0        │200K  │ 8K   │ ✅  │ ✅  │ ❌     │
  │ anthropic.claude-sonnet-4-5-*-v1:0       │200K  │ 8K   │ ✅  │ ✅  │ ✅     │
  │ anthropic.claude-3-5-sonnet-*-v2:0       │200K  │ 8K   │ ✅  │ ✅  │ ❌     │
  ├──────────────────────────────────────────┼──────┼──────┼─────┼─────┼────────┤
  │ amazon.nova-premier-v1:0                 │ 1M   │ 32K  │ ✅  │ ✅  │ ✅     │
  │ amazon.nova-pro-v1:0                     │300K  │ 5K   │ ✅  │ ✅  │ ❌     │
  │ amazon.nova-lite-v1:0                    │300K  │ 5K   │ ✅  │ ✅  │ ❌     │
  │ amazon.nova-micro-v1:0                   │128K  │ 5K   │ ❌  │ ✅  │ ❌     │
  │ amazon.nova-2-lite-v1:0                  │300K  │ 5K   │ ✅  │ ✅  │ ❌     │
  │ amazon.nova-2-sonic-v1:0                 │300K  │ 5K   │ 🎤  │ ✅  │ ❌     │
  ├──────────────────────────────────────────┼──────┼──────┼─────┼─────┼────────┤
  │ meta.llama-4-maverick-17b-instruct-v1:0  │ 1M   │ 16K  │ ✅  │ ✅  │ ❌     │
  │ meta.llama-4-scout-17b-instruct-v1:0     │512K  │ 16K  │ ✅  │ ✅  │ ❌     │
  │ meta.llama3-1-405b-instruct-v1:0         │128K  │ 4K   │ ❌  │ ✅  │ ❌     │
  ├──────────────────────────────────────────┼──────┼──────┼─────┼─────┼────────┤
  │ openai.gpt-oss-120b                      │128K  │ —    │ ❌  │ ❌  │ ✅     │
  │ openai.gpt-oss-20b                       │128K  │ —    │ ❌  │ ❌  │ ✅     │
  │ mistral.mistral-large-2407-v1:0          │ 32K  │ —    │ ❌  │ ✅  │ ❌     │
  │ mistral.magistral-medium-2506            │128K  │ —    │ ❌  │ ✅  │ ✅     │
  │ deepseek.r1-v1:0                         │128K  │ —    │ ❌  │ ❌  │ ✅     │
  │ qwen.qwen3-235b-a22b-instruct-v1:0      │131K  │ —    │ ❌  │ ✅  │ ✅     │
  ├──────────────────────────────────────────┼──────┼──────┼─────┼─────┼────────┤
  │ EMBEDDINGS                               │Ctx   │Dim   │     │     │        │
  ├──────────────────────────────────────────┼──────┼──────┼─────┼─────┼────────┤
  │ amazon.titan-embed-text-v2:0             │ 8K   │1024  │     │     │        │
  │ amazon.nova-embed-v1:0                   │ 8K   │1024  │     │     │        │
  │ cohere.embed-english-v3                  │ 2K   │1024  │     │     │        │
  │ cohere.embed-v4:0                        │ 2K   │1536  │     │     │        │
  └──────────────────────────────────────────┴──────┴──────┴─────┴─────┴────────┘

  Notes:
  - Feature gate: #[cfg(feature = "bedrock")] applies to all Bedrock code
  - Inference profiles: bare IDs auto-resolved to us./eu./ap. based on region
  - Claude 5.x on Bedrock uses simplified IDs (no date suffix)
  - Claude 4.6+ on Bedrock have 1M context (NOT 200K as currently coded)
  - ListFoundationModels returns modalities but NOT context lengths
  - OpenAI gpt-oss models are open-weight reasoning models on Bedrock
  - 🎤 = audio modality (Nova 2 Sonic supports speech input)
```

### Common OpenAI-Compatible Models (Groq, DeepSeek, Together)

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  OPENAI-COMPATIBLE MODELS — July 2026                                        │
  ├─────────────────────────────────────┬────────┬────────┬───────┬──────┬──────┤
  │ Model ID                            │Service │Ctx In  │Max Out│Tools │Think │
  ├─────────────────────────────────────┼────────┼────────┼───────┼──────┼──────┤
  │ deepseek-chat (V3)                  │DeepSeek│  64K   │  8K   │  ✅  │  ❌  │
  │ deepseek-reasoner (R1)              │DeepSeek│  64K   │  8K   │  ❌  │  ✅  │
  │ llama-3.3-70b-versatile             │Groq    │ 128K   │ 32K   │  ✅  │  ❌  │
  │ llama-3.1-8b-instant                │Groq    │ 128K   │  8K   │  ✅  │  ❌  │
  │ gemma2-9b-it                        │Groq    │  8K    │  8K   │  ✅  │  ❌  │
  │ meta-llama/Meta-Llama-3.1-405B      │Together│ 128K   │  4K   │  ✅  │  ❌  │
  │ mistralai/Mixtral-8x22B-Instruct    │Together│  64K   │  4K   │  ✅  │  ❌  │
  │ Qwen/Qwen2.5-72B-Instruct          │Together│ 128K   │  4K   │  ✅  │  ❌  │
  └─────────────────────────────────────┴────────┴────────┴───────┴──────┴──────┘
```

---

## Capability Resolution (Anti-Heuristic Design)

> **REVISION (2026-07-04)**: The original `infer_capabilities()` function based on 
> model ID string matching has been **removed entirely**. String-pattern inference
> is fragile, produces wrong results for new models, and is unnecessary given that
> 11 of 13 providers now return structured capability metadata from their APIs.

### Design Principle: Three Sources Only

Capability data comes from exactly three sources, tried in order:

```
  ┌──────────────────────────────────────────────────────────┐
  │  Source 1: PROVIDER API RESPONSE (highest trust)         │
  │  The provider's own API declares model capabilities.     │
  │  Examples:                                               │
  │    - Anthropic /v1/models → capabilities{}               │
  │    - Ollama /api/tags → capabilities[]                   │
  │    - Mistral /v1/models → capabilities{}                 │
  │    - OpenRouter /api/v1/models → full metadata           │
  │                                                          │
  │  Source 2: STATIC REGISTRY (verified documentation)      │
  │  Hand-curated entries, each citing official doc URL.      │
  │  Used only when the API doesn't return capability data.  │
  │  Examples: OpenAI, xAI (APIs return IDs only)            │
  │                                                          │
  │  Source 3: EXPLICIT UNKNOWN                              │
  │  Model not found in any source → all capabilities are    │
  │  set to false/0 with DiscoverySource::Unknown.           │
  │  The caller sees "unknown" and must decide how to        │
  │  proceed (e.g., prompt the user, try anyway, reject).    │
  └──────────────────────────────────────────────────────────┘
```

### Why Name-Pattern Inference Is Banned

| Failure Mode | Example |
|---|---|
| **Wrong positive** | `grok-2-vision-1212` matches `id.contains("grok-4")` false but `id.contains("vision")` true — but this model does NOT support tool calling |
| **Wrong negative** | `claude-fable-5` was not matched by any pattern in the original code — returns 200K context when it should be 1M |
| **Stale data** | `o1-mini` was banned from tools, but OpenAI later enabled tool calling on it |
| **Ambiguity** | `gemini-3.5-flash-image` contains "gemini-3" but is an image generation model, not a chat model |

### Replacement: Static Registry With Citations

```rust
/// Static model registry. Every entry MUST include:
/// 1. Official doc URL in a comment
/// 2. Date when the entry was last verified
/// 3. Explicit values for ALL capability fields (no defaults)
///
/// Unknown values are represented as `0` or `false`, never guessed.
fn openai_static_registry() -> Vec<DiscoveredModel> {
    vec![
        // Source: https://platform.openai.com/docs/models/gpt-5.5
        // Verified: 2026-07-04
        DiscoveredModel {
            id: "gpt-5.5".into(),
            context_length: 1_048_576,
            max_output_tokens: 32_768,
            capabilities: ModelCapabilities {
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_thinking: false,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry {
                url: "https://platform.openai.com/docs/models/gpt-5.5",
                verified_date: "2026-07-04",
            },
            ..Default::default()
        },
        // ... more models, each with source citation ...
    ]
}
```

---

## Agent Harness Integration Patterns

### Pattern: Context Window Budgeting

```rust
let model = service.get_model("openai", "gpt-4.1").await?
    .ok_or("model not found")?;

let system_prompt_tokens = tokenizer.count(&system_prompt);
let available_for_history = model.context_length
    - system_prompt_tokens
    - model.max_output_tokens
    - 500; // safety margin

// Truncate history to fit
let trimmed = trim_messages_to_tokens(history, available_for_history);
```

### Pattern: Capability-Based Routing

```rust
fn select_model_for_task(task: &AgentTask, service: &ModelDiscoveryService)
    -> Result<DiscoveredModel>
{
    let filter = match task.task_type {
        TaskType::CodeReview => CapabilityFilter {
            requires_tools: Some(true),
            min_context_length: Some(100_000),
            requires_thinking: Some(true),
            ..Default::default()
        },
        TaskType::ImageAnalysis => CapabilityFilter {
            requires_vision: Some(true),
            requires_tools: Some(true),
            ..Default::default()
        },
        TaskType::QuickAnswer => CapabilityFilter {
            min_context_length: Some(8_000),
            max_cost_per_m_input: Some(1.0),
            ..Default::default()
        },
    };

    let candidates = service.find_models_sync(&filter)?;
    // Select cheapest that meets requirements
    candidates.into_iter()
        .min_by(|a, b| {
            a.cost_per_m_input.unwrap_or(f64::MAX)
                .partial_cmp(&b.cost_per_m_input.unwrap_or(f64::MAX))
                .unwrap()
        })
        .ok_or_else(|| "No model matches requirements".into())
}
```

### Pattern: Fallback Chain

```rust
let preferred = vec!["anthropic/claude-opus-4", "openai/gpt-4.1", "openai/o3"];

for model_id in &preferred {
    let (provider, model) = model_id.split_once('/').unwrap();
    if let Some(info) = service.get_model(provider, model).await? {
        if info.available && info.capabilities.supports_function_calling {
            return Ok(info);
        }
    }
}
// Fallback: any available model with tool support
let filter = CapabilityFilter {
    requires_tools: Some(true),
    ..Default::default()
};
service.find_models(&filter).await?.into_iter().next()
    .ok_or("No models available")
```
