# Provider Audit — July 2026

> **Scope**: Audit of edgequake-llm v0.7.0 provider implementations against the **latest official API specs** as of 2026-07-04.
>
> **Method**: Source-code-is-law analysis cross-referenced against live provider documentation.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Provider-by-Provider Audit](#2-provider-by-provider-audit)
   - [2.1 OpenAI](#21-openai)
   - [2.2 Azure OpenAI (Microsoft Foundry)](#22-azure-openai-microsoft-foundry)
   - [2.3 Anthropic (Claude)](#23-anthropic-claude)
   - [2.4 Google Gemini (AI Studio)](#24-google-gemini-ai-studio)
   - [2.5 Google Vertex AI](#25-google-vertex-ai)
   - [2.6 AWS Bedrock](#26-aws-bedrock)
   - [2.7 Mistral AI](#27-mistral-ai)
   - [2.8 xAI (Grok)](#28-xai-grok)
   - [2.9 Cohere](#29-cohere)
   - [2.10 OpenRouter](#210-openrouter)
   - [2.11 NVIDIA NIM](#211-nvidia-nim)
   - [2.12 Ollama](#212-ollama)
   - [2.13 LM Studio](#213-lm-studio)
3. [Cross-Reference Matrix](#3-cross-reference-matrix)
4. [Methodology & Sources](#4-methodology--sources)

---

## 1. Executive Summary

edgequake-llm v0.7.0 has strong chat/LLM coverage across 15 providers but significant gaps in **embeddings** and **image generation** coverage for newer providers. Key findings:

| Area | Current Coverage | Gap Severity |
|------|-----------------|--------------|
| LLM Chat/Completion | 15/13 providers (100%) | Low — model lists outdated |
| Streaming | 13/13 supported | Low — some missing SSE features |
| Tool/Function Calling | 10/13 supported | Medium — Cohere, NVIDIA need it |
| Embeddings | 8/13 providers (incl. Bedrock) | **High** — missing Cohere, NVIDIA |
| Image Generation | 3/13 providers (Gemini, Vertex, Fal) | **Critical** — missing OpenAI, xAI, Stability (NVIDIA) |
| Adaptive/Extended Thinking | 4/13 (Claude, Gemini, OpenAI, Bedrock) | Medium — missing Mistral, xAI reasoning_effort |
| Vision (image input) | 8/13 providers | Medium — missing Mistral, Cohere, NVIDIA VLM |
| Reranking | External (BM25/HTTP) | Low — Cohere Rerank v4 is market leader |

### Critical Gaps (P0)

1. **No Cohere provider** — Command A+, Embed v4, Rerank v4 are industry standards
2. **OpenAI models outdated** — still references GPT-4/GPT-3.5; GPT-5.5 is the current flagship
3. **No image generation for OpenAI** — gpt-image-2 is the successor to deprecated DALL-E
4. **No embeddings for NVIDIA** — NIM has OpenAI-compatible `/v1/embeddings`
5. **Anthropic has no native embeddings** — Voyage AI is the official recommendation (not a gap, by design)

> **Note**: Bedrock embeddings are already implemented (Titan, Cohere, Nova). Mistral has
> partial vision support. xAI defaults to `grok-4.3` (1M context). These were initially
> flagged as gaps but confirmed as implemented during deep code analysis.

---

## 2. Provider-by-Provider Audit

### 2.1 OpenAI

**File**: `src/providers/openai.rs`
**Official Docs**: https://developers.openai.com/api/docs

#### Current State in edgequake-llm

| Capability | Implemented | Details |
|-----------|-------------|---------|
| LLM Chat | ✅ | Via `async-openai` crate |
| Streaming | ✅ | SSE with tool call deltas |
| Tool Calling | ✅ | Function tools + streaming |
| Embeddings | ✅ | `text-embedding-3-small/large` |
| Vision (image input) | ✅ | Base64 + URL images |
| Image Generation | ❌ | **Not implemented** |
| Extended Thinking | ✅ | `reasoning_effort` for o-series |
| JSON Mode | ✅ | `response_format: json_object` |
| Cache Hit Tracking | ✅ | `cached_tokens` from usage |

#### Latest Official API (July 2026)

| Feature | Current Status | edgequake-llm |
|---------|---------------|---------------|
| GPT-5.5 (flagship) | GA | ❌ Not in model lists |
| GPT-5.5 Pro | GA | ❌ Not in model lists |
| GPT-5.4 / mini / nano | GA | ❌ Not in model lists |
| gpt-image-2 | GA (replaces DALL-E) | ❌ **No image gen provider** |
| gpt-image-1 / mini | GA | ❌ |
| text-embedding-3-large | GA | ✅ Supported |
| text-embedding-3-small | GA | ✅ Supported |
| Responses API | GA | ❌ Using Chat Completions only |
| o3 / o3-pro | Deprecated → GPT-5.5 | ⚠️ Still referenced |
| GPT-4o / GPT-4 Turbo | Legacy | ⚠️ Still primary models |
| Structured Outputs | GA | ⚠️ Partial (json_mode only, no JSON schema) |
| Tool Search | GA (5.4+) | ❌ Not supported |
| Prompt Caching | GA | ✅ cache_hit_tokens tracked |

**Ref**: https://developers.openai.com/api/docs/models/all
**Ref**: https://developers.openai.com/api/docs/deprecations

#### Gaps

1. **P0**: No image generation support — `gpt-image-2` is the current standard
2. **P1**: Model catalog outdated — GPT-5.5, GPT-5.4 family not listed
3. **P1**: No Responses API support (new unified API replacing Chat Completions)
4. **P2**: No structured output with JSON schema (only json_mode)
5. **P2**: o3/o3-pro deprecated, should warn users

---

### 2.2 Azure OpenAI (Microsoft Foundry)

**File**: `src/providers/azure_openai.rs`
**Official Docs**: https://learn.microsoft.com/en-us/azure/foundry/

#### Current State

| Capability | Implemented | Details |
|-----------|-------------|---------|
| LLM Chat | ✅ | Via `async-openai` with Azure config |
| Streaming | ✅ | Delegated to OpenAI provider |
| Tool Calling | ✅ | Via OpenAI compat |
| Embeddings | ✅ | Via OpenAI compat |
| Vision | ✅ | Via OpenAI compat |
| Image Generation | ❌ | **Not implemented** |

#### Latest Official API (July 2026)

| Feature | Azure Status | edgequake-llm |
|---------|-------------|---------------|
| GPT-5.5 | GA | ❌ Not in defaults |
| GPT-5.4 series | GA | ❌ |
| gpt-image-2 | GA | ❌ **No image gen** |
| text-embedding-3-large | GA | ✅ |
| Managed Identity auth | GA | ⚠️ API key only |
| Data Zones | GA | ❌ Not configurable |

**Ref**: https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure
**Ref**: https://azure.microsoft.com/en-us/pricing/details/azure-openai/

#### Gaps

1. **P1**: Model defaults outdated (still GPT-4o/GPT-4)
2. **P1**: No Azure Managed Identity / Entra ID auth (only API key)
3. **P2**: No Data Zone support for sovereignty requirements
4. **P2**: No image generation via Azure endpoints

---

### 2.3 Anthropic (Claude)

**File**: `src/providers/anthropic.rs`
**Official Docs**: https://platform.claude.com/docs

#### Current State

| Capability | Implemented | Details |
|-----------|-------------|---------|
| LLM Chat | ✅ | Messages API (`/v1/messages`) |
| Streaming | ✅ | SSE with tool call deltas |
| Tool Calling | ✅ | Full function calling |
| Extended Thinking | ✅ | `thinking` blocks + budget |
| Prompt Caching | ✅ | `cache_control: ephemeral` + 1h TTL |
| Vision | ✅ | Base64 images in content blocks |
| Embeddings | ❌ | **Not available** (Anthropic has no embeddings API) |
| Image Generation | ❌ | **Not available** |

#### Latest Official API (July 2026)

| Feature | Status | edgequake-llm |
|---------|--------|---------------|
| Claude Fable 5 | GA | ❌ Not in model lists |
| Claude Opus 4.8 | GA | ❌ Not in model lists |
| Claude Sonnet 5 | GA | ❌ Not in model lists |
| Claude Haiku 4.5 | GA | ❌ Not in model lists |
| Adaptive Thinking (always-on for Fable 5) | GA | ⚠️ `thinking` still uses budget style |
| `effort` parameter | GA | ❌ Not mapped to `reasoning_effort` |
| Code Execution tool | GA | ❌ Not supported |
| MCP Connector | GA | ❌ Not supported |
| Files API | GA | ❌ Not supported |
| Claude Opus 4 / Sonnet 4 | Retired | ⚠️ May still be in model lists |

**Ref**: https://platform.claude.com/docs/en/about-claude/models/overview
**Ref**: https://platform.claude.com/docs/en/release-notes/overview

#### Gaps

1. **P0**: Model catalog severely outdated — Fable 5, Opus 4.8, Sonnet 5 are current
2. **P1**: No `effort` parameter mapping (new API for controlling thinking depth)
3. **P1**: Adaptive thinking for Fable 5 / Opus 4.8 (different from old budget-based thinking)
4. **P2**: Embeddings are N/A — recommend Voyage AI integration as separate provider
5. **INFO**: Retired models (Opus 4, Sonnet 4) should be removed/warned

---

### 2.4 Google Gemini (AI Studio)

**File**: `src/providers/gemini.rs`
**Official Docs**: https://ai.google.dev/gemini-api/docs

#### Current State

| Capability | Implemented | Details |
|-----------|-------------|---------|
| LLM Chat | ✅ | REST API `generateContent` |
| Streaming | ✅ | `streamGenerateContent?alt=sse` |
| Tool Calling | ✅ | Function declarations |
| Embeddings | ✅ | `embedContent` API |
| Vision | ✅ | `inlineData` image parts |
| Image Generation | ✅ | `GeminiImageGenProvider` |
| Thinking (2.5/3.x) | ✅ | `thinkingConfig` with budget/level |
| `thought_signature` | ✅ | Gemini 3.x opaque signatures |
| JSON Mode | ✅ | `responseMimeType: application/json` |

#### Latest Official API (July 2026)

| Feature | Status | edgequake-llm |
|---------|--------|---------------|
| Gemini 3.5 Flash | GA | ❌ Not in model lists |
| Gemini 3.1 Flash-Lite | GA | ❌ |
| Gemini 3.1 Pro (Preview) | Preview | ❌ |
| Gemini 2.5 Pro / Flash | GA (retiring Oct 2026) | ✅ Supported |
| gemini-embedding-2 (multimodal) | GA | ❌ Only text embeddings |
| gemini-3.1-flash-image | GA | ❌ Not in image gen |
| gemini-3-pro-image | GA | ❌ Not in image gen |
| Media resolution control | GA (3.x) | ❌ Not supported |
| Combined tool use (Search + Code + Functions) | GA | ❌ Only function calling |
| `thinkingLevel` (3.x) | GA | ✅ Supported |
| `GEMINI_API_KEY` env var | Standard | ✅ |

**Ref**: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions
**Ref**: https://ai.google.dev/gemini-api/docs/embeddings
**Ref**: https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5

#### Gaps

1. **P0**: Model catalog outdated — Gemini 3.x family is current
2. **P1**: `gemini-embedding-2` multimodal embeddings not supported (images, audio, video, PDF)
3. **P1**: Gemini 3.x image generation models not in image gen provider
4. **P2**: Media resolution control for images/video/PDF input
5. **P2**: Google Search grounding tool not exposed
6. **INFO**: Gemini 1.x models fully retired

---

### 2.5 Google Vertex AI

**File**: Examples exist (`examples/vertexai/`) but no dedicated provider. Uses `GeminiProvider` with Vertex AI configuration.
**Official Docs**: https://docs.cloud.google.com/vertex-ai/generative-ai/docs

#### Current State

| Capability | Implemented | Details |
|-----------|-------------|---------|
| LLM Chat | ✅ | Via GeminiProvider with Vertex config |
| Streaming | ✅ | Via Gemini |
| Tool Calling | ✅ | Via Gemini |
| Embeddings | ✅ | Via Gemini embeddings |
| Image Generation | ✅ | `VertexAIImageGen` (Imagen) |
| Service Account Auth | ⚠️ | Requires manual token management |
| ADC (Application Default Credentials) | ❌ | Not auto-detected |

#### Latest Official API (July 2026)

| Feature | Status | edgequake-llm |
|---------|--------|---------------|
| Vertex AI SDK deprecated (June 24, 2026) | **REMOVED** | ⚠️ Must use Google GenAI SDK patterns |
| Gemini Enterprise Agent Platform rebrand | Active | ❌ No awareness |
| `aiplatform.googleapis.com` REST endpoint | Still active | ✅ Used |
| RAG Engine Serverless | Preview | ❌ |
| Gemma 4 26B | Preview (Model Garden) | ❌ |
| Third-party models (Claude, Llama) | GA | ❌ Not routed |

**Ref**: https://therouter.ai/news/vertex-ai-sdk-migration-gemini-enterprise-agent-platform/
**Ref**: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/release-notes

#### Gaps

1. **P1**: No ADC/Workload Identity Federation auth (critical for GCP deployments)
2. **P1**: Vertex AI SDK deprecation — ensure direct REST API patterns are used
3. **P2**: No Model Garden third-party model routing
4. **P2**: Imagen 3 models for image generation should be updated

---

### 2.6 AWS Bedrock

**File**: `src/providers/bedrock.rs` (feature-gated: `bedrock`)
**Official Docs**: https://docs.aws.amazon.com/bedrock/latest/userguide/

#### Current State

| Capability | Implemented | Details |
|-----------|-------------|---------|
| LLM Chat | ✅ | Converse API |
| Streaming | ✅ | `ConverseStream` |
| Tool Calling | ✅ | Converse tool use |
| Extended Thinking | ✅ | `thinking_budget_tokens` for Claude |
| Vision | ✅ | Image blocks in Converse |
| Embeddings | ✅ | Via `InvokeModel` — Titan v1/v2, Cohere v3/v4, Nova |
| Image Generation | ❌ | **Not implemented** |
| Auth | ✅ | AWS SDK SigV4 via `aws-config` |
| Inference Profiles | ✅ | Auto-resolves to `us.`/`eu.`/`ap.` prefixed IDs |

#### Latest Official API (July 2026)

| Feature | Status | edgequake-llm |
|---------|--------|---------------|
| Claude Fable 5 (`anthropic.claude-fable-5`) | GA | ❌ Not in model lists |
| Claude Opus 4.8 | GA | ❌ |
| Claude Sonnet 5 | GA | ❌ |
| Nova 2 Lite / Sonic | GA | ❌ Not supported |
| Titan Text Embeddings V2 | GA | ✅ Supported via InvokeModel |
| Nova 2 Multimodal Embeddings | GA | ❌ Not yet |
| Cohere Embed v3/v4 on Bedrock | GA | ✅ Supported via InvokeModel |
| Stability AI (image gen) | GA | ❌ **No image gen** |
| Nova Canvas | Legacy (EOL Sep 2026) | ❌ |
| `bedrock-mantle` endpoint (OpenAI-compat) | GA | ❌ Not used |
| Messages API on Bedrock | GA | ❌ Using Converse only |
| Geo / Global inference IDs | GA | ⚠️ Inference profiles auto-resolved |

**Ref**: https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
**Ref**: https://hidekazu-konishi.com/entry/amazon_bedrock_model_catalog_2026.html

#### Gaps

1. **P0**: Model catalog outdated — Claude Fable 5, Opus 4.8, Sonnet 5 are current
2. **P1**: No image generation — Stability AI is available on Bedrock
3. **P1**: No `bedrock-mantle` OpenAI-compatible endpoint support
4. **P1**: Streaming tool calls not supported (`supports_tool_streaming()` returns `false`)
5. **P2**: No Nova 2 model family support
6. **INFO**: Embeddings already supported (Titan v1/v2, Cohere v3/v4, Nova) — upgrade to Embed v4

---

### 2.7 Mistral AI

**File**: `src/providers/mistral.rs`
**Official Docs**: https://docs.mistral.ai/

#### Current State

| Capability | Implemented | Details |
|-----------|-------------|---------|
| LLM Chat | ✅ | `/v1/chat/completions` |
| Streaming | ✅ | SSE |
| Tool Calling | ✅ | Function tools |
| Embeddings | ✅ | `/v1/embeddings` with `mistral-embed` |
| Vision | ⚠️ | Partial — Large 3, Medium 3.5, Small 4, Ministral models support vision |
| Image Generation | ❌ | Via agents only (not standard API) |
| `safe_prompt` | ✅ | Mistral-specific option |
| `parallel_tool_calls` | ✅ | Mistral-specific option |
| Model Discovery | ✅ | `/v1/models` |

#### Latest Official API (July 2026)

| Feature | Status | edgequake-llm |
|---------|--------|---------------|
| Mistral Medium 3.5 (flagship, 128B) | GA | ❌ Not in model lists |
| Mistral Small 4 | GA | ❌ |
| Codestral 2508 | GA | ❌ |
| `reasoning_effort` parameter | GA | ❌ Not mapped |
| Vision (native in Medium 3.5) | GA | ❌ **No vision** |
| `mistral-embed` (1024-dim) | GA | ✅ Supported |
| `mistral-embed-dim256-2510` | GA | ❌ Not listed |
| `mistral-embed-dim128-2510` | GA | ❌ Not listed |
| Image generation (agent tool) | Beta | ❌ Not supported |
| Magistral models | **Deprecated** | ⚠️ May be referenced |

**Ref**: https://docs.mistral.ai/studio-api/conversations/reasoning
**Ref**: https://mistral.ai/news/vibe-remote-agents-mistral-medium-3-5/

#### Gaps

1. **P1**: `reasoning_effort` not mapped (high/medium/low/none)
2. **P1**: Model catalog outdated — Medium 3.5 is the current flagship
3. **P2**: Variable-dimension embed models not listed (`mistral-embed-dim256`, `dim128`)
4. **P2**: Vision support exists for some models but needs Medium 3.5 validation
5. **INFO**: Magistral, Devstral deprecated → replaced by Medium 3.5
6. **INFO**: Audio/TTS/OCR capabilities already implemented (advanced feature)

---

### 2.8 xAI (Grok)

**File**: `src/providers/xai.rs`
**Official Docs**: https://docs.x.ai/

#### Current State

| Capability | Implemented | Details |
|-----------|-------------|---------|
| LLM Chat | ✅ | OpenAI-compatible `/v1/chat/completions` |
| Streaming | ✅ | SSE |
| Tool Calling | ✅ | Function tools |
| Vision | ✅ | Image input |
| Embeddings | ❌ | **xAI has no embeddings API** |
| Image Generation | ❌ | **Not implemented** (Grok Imagine exists) |
| Model Discovery | ✅ | `/v1/models` |

#### Latest Official API (July 2026)

| Feature | Status | edgequake-llm |
|---------|--------|---------------|
| Grok 4.3 (flagship) | GA | ❌ Not in model lists |
| Grok 4.20 Reasoning | GA | ❌ |
| Grok Build 0.1 (coding) | Early Access | ❌ |
| `grok-imagine-image-quality` | GA | ❌ **No image gen** |
| `grok-imagine-image` | GA | ❌ |
| `grok-imagine-video` | GA | ❌ |
| Reasoning effort (via model variants) | GA | ⚠️ Not explicitly configurable |
| 1M token context (Grok 4.20) | GA | ❌ Not reflected in max_context |
| Embeddings | **Not available** | N/A (correct) |

**Ref**: https://docs.x.ai/developers/model-capabilities/images/generation
**Ref**: https://x.ai/api

#### Gaps

1. **P1**: Model catalog outdated — Grok 4.3 / 4.20 are current
2. **P1**: No image generation — Grok Imagine API (`/v1/images/generations`) exists
3. **P2**: Context window not updated (1M for Grok 4.20)
4. **INFO**: Grok 3 / 4.1 models retired → auto-redirected to 4.3

---

### 2.9 Cohere

**File**: **DOES NOT EXIST** — No Cohere provider in edgequake-llm
**Official Docs**: https://docs.cohere.com/

#### Latest Official API (July 2026)

| Feature | Status | edgequake-llm |
|---------|--------|---------------|
| Command A+ (flagship, 218B MoE) | GA | ❌ **No provider** |
| Command A / A Reasoning / A Vision | GA | ❌ |
| Command R+ / R | GA | ❌ |
| Embed v4 (multimodal, 128k context) | GA | ❌ **No provider** |
| Rerank v4.0-pro / fast | GA | ❌ (have HTTP reranker but not Cohere-specific) |
| Chat API | GA | ❌ |
| Tool Use (multi-step) | GA | ❌ |
| Vision (Command A+) | GA | ❌ |
| RAG with citations | GA | ❌ |
| Image Generation | **Not available** | N/A |

**Ref**: https://docs.cohere.com/docs/models
**Ref**: https://docs.cohere.com/changelog/embed-multimodal-v4
**Ref**: https://cohere.com/blog/command-a-plus

#### Gaps

1. **P0**: No Cohere provider at all — critical gap for enterprise RAG workflows
2. **P0**: Embed v4 (multimodal, Matryoshka, 128k) is industry-leading
3. **P0**: Rerank v4 is the standard for search relevance
4. **P1**: Command A+ has vision, tool use, 48 languages
5. **P1**: Unique `input_type` (search_document / search_query) for embeddings

---

### 2.10 OpenRouter

**File**: `src/providers/openrouter.rs`
**Official Docs**: https://openrouter.ai/docs

#### Current State

| Capability | Implemented | Details |
|-----------|-------------|---------|
| LLM Chat | ✅ | OpenAI-compatible |
| Streaming | ✅ | SSE |
| Tool Calling | ✅ | Via OpenAI compat |
| Embeddings | ❌ | **Not implemented** |
| Image Generation | ❌ | **Not implemented** |
| Model Discovery | ✅ | `/api/v1/models` with caching |
| Fallback routing | ❌ | `models` array not supported |
| Provider preferences | ❌ | Not exposed |

#### Latest Official API (July 2026)

| Feature | Status | edgequake-llm |
|---------|--------|---------------|
| 400+ models, 70+ providers | GA | ✅ Discovery works |
| `~author/family-latest` aliases | GA | ❌ Not surfaced |
| OpenRouter Fusion | GA | ❌ |
| Plugins (web search, etc.) | GA | ❌ |
| Model fallback arrays | GA | ❌ |
| Provider routing preferences | GA | ❌ |

**Ref**: https://openrouter.ai/docs/guides/overview/models
**Ref**: https://openrouter.ai/docs/guides/routing/routers/latest-resolution

#### Gaps

1. **P2**: No embeddings — OpenRouter doesn't natively support embeddings
2. **P2**: No model fallback array support
3. **P2**: No `~latest` alias awareness
4. **P3**: No Fusion multi-model support

---

### 2.11 NVIDIA NIM

**File**: `src/providers/nvidia.rs`
**Official Docs**: https://docs.nvidia.com/nim/

#### Current State

| Capability | Implemented | Details |
|-----------|-------------|---------|
| LLM Chat | ✅ | OpenAI-compatible `/v1/chat/completions` |
| Streaming | ✅ | SSE |
| Tool Calling | ❌ | **Not implemented** |
| Embeddings | ❌ | **Not implemented** |
| Image Generation | ❌ | **Not implemented** |
| Model Discovery | ✅ | `/v1/models` |
| Auth | ✅ | API key header |

#### Latest Official API (July 2026)

| Feature | Status | edgequake-llm |
|---------|--------|---------------|
| `/v1/chat/completions` | GA | ✅ |
| `/v1/embeddings` | GA | ❌ **Missing** |
| `/v1/completions` | GA | ❌ |
| `/v1/responses` (OpenAI compat) | GA | ❌ |
| `/v1/messages` (Anthropic compat) | GA | ❌ |
| Image Generation (SD3.5, FLUX) | GA (Visual GenAI) | ❌ **Missing** |
| NeMo Retriever embeddings | GA | ❌ |
| `llama-nemotron-embed-1b-v2` | GA | ❌ |
| `nv-embedqa-e5-v5` | GA | ❌ |
| Reranking | GA | ❌ |
| Generative Scoring | GA (vLLM 0.20+) | ❌ |
| Tool calling in chat | GA | ❌ **Missing** |

**Ref**: https://docs.nvidia.com/nim/large-language-models/latest/reference/api-reference.html
**Ref**: https://docs.nvidia.com/nim/visual-genai/latest/api/openai-image-generation.html
**Ref**: https://docs.nvidia.com/nemo/microservices/latest/run-inference/nim-proxy/embeddings.html

#### Gaps

1. **P0**: No embeddings — NIM exposes `/v1/embeddings` (OpenAI-compatible)
2. **P1**: No tool calling — NIM supports it in `/v1/chat/completions`
3. **P1**: No image generation — Visual GenAI NIM has OpenAI-compatible image gen
4. **P2**: No reranking endpoint
5. **P2**: No Anthropic-compatible `/v1/messages` routing

---

### 2.12 Ollama

**File**: `src/providers/ollama.rs`
**Official Docs**: https://docs.ollama.com/api

#### Current State

| Capability | Implemented | Details |
|-----------|-------------|---------|
| LLM Chat | ✅ | Native `/api/chat` |
| Streaming | ✅ | NDJSON streaming |
| Tool Calling | ✅ | Via chat API |
| Embeddings | ✅ | Native `/api/embed` |
| Vision | ✅ | Base64 images in messages |
| Image Generation | ❌ | Experimental in Ollama |
| Model Management | ✅ | Pull, list, show |
| `reasoning_effort` | ✅ | For thinking models |

#### Latest Official API (July 2026)

| Feature | Status | edgequake-llm |
|---------|--------|---------------|
| `/api/embed` (batch, dimensions) | GA | ✅ |
| `/api/chat` with tools | GA | ✅ |
| OpenAI-compat `/v1/*` endpoints | GA | ❌ Not used (uses native API) |
| Image generation | Experimental | ❌ |
| Dimension reduction in embed | GA | ⚠️ Not exposed |
| L2-normalized embeddings | Default | ✅ |

**Ref**: https://docs.ollama.com/api/embed
**Ref**: https://github.com/ollama/ollama/blob/main/docs/api.md

#### Gaps

1. **P2**: Embed dimension reduction not exposed (`dimensions` field)
2. **P3**: Experimental image generation not tracked
3. **INFO**: Well-covered overall — minor incremental improvements

---

### 2.13 LM Studio

**File**: `src/providers/lmstudio.rs`
**Official Docs**: https://lmstudio.ai/docs/developer

#### Current State

| Capability | Implemented | Details |
|-----------|-------------|---------|
| LLM Chat | ✅ | OpenAI-compat + native v1 API |
| Streaming | ✅ | SSE + native streaming |
| Tool Calling | ✅ | OpenAI-compat tools |
| Embeddings | ✅ | `/v1/embeddings` |
| Vision | ✅ | Image input |
| Prefill Progress | ✅ | `PrefillProgress` in StreamChunk |
| Model Metadata | ✅ | Context window, quantization |
| Image Generation | ❌ | **Not available** in LM Studio |

#### Latest Official API (July 2026)

| Feature | Status | edgequake-llm |
|---------|--------|---------------|
| `/api/v1/chat` (native v1 API) | GA (v0.4.0+) | ✅ Supported |
| `/v1/responses` (OpenAI compat) | GA | ❌ Not used |
| `/v1/messages` (Anthropic compat) | GA | ❌ Not used |
| MCP via API | GA | ❌ |
| Stateful chats | GA | ⚠️ Via native API only |
| Model download/load/unload | GA | ❌ Not exposed |
| Auth tokens | GA | ❌ Not configured |

**Ref**: https://lmstudio.ai/docs/developer/rest
**Ref**: https://lmstudio.ai/docs/developer/openai-compat

#### Gaps

1. **P3**: Model download/load/unload management not exposed
2. **P3**: Auth token configuration not supported
3. **INFO**: Solid coverage for local inference use case

---

## 3. Cross-Reference Matrix

See [02-CROSS-REFERENCE-MATRIX.md](./02-CROSS-REFERENCE-MATRIX.md) for the full capability matrix.

---

## 4. Methodology & Sources

### Analysis Method

1. **Code-is-law**: Every provider `.rs` file was read in full. Capability claims are based on actual trait implementations (`LLMProvider`, `EmbeddingProvider`, `ImageGenProvider`), not comments or aspirational code.
2. **Official docs grounding**: Each provider was cross-referenced against its official API documentation as of 2026-07-04.
3. **Model catalog verification**: Current flagship models were verified against provider pricing pages and release notes.

### Primary Sources

| Provider | Primary Documentation URL |
|----------|--------------------------|
| OpenAI | https://developers.openai.com/api/docs |
| Azure OpenAI | https://learn.microsoft.com/en-us/azure/foundry/ |
| Anthropic | https://platform.claude.com/docs |
| Gemini | https://ai.google.dev/gemini-api/docs |
| Vertex AI | https://docs.cloud.google.com/vertex-ai/generative-ai/docs |
| AWS Bedrock | https://docs.aws.amazon.com/bedrock/latest/userguide/ |
| Mistral | https://docs.mistral.ai/ |
| xAI | https://docs.x.ai/ |
| Cohere | https://docs.cohere.com/ |
| OpenRouter | https://openrouter.ai/docs |
| NVIDIA NIM | https://docs.nvidia.com/nim/ |
| Ollama | https://docs.ollama.com/api |
| LM Studio | https://lmstudio.ai/docs/developer |
