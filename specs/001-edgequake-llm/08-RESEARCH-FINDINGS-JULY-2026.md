# 08 — Research Findings Synthesis (July 2026)

> **Date**: 2026-07-04  
> **Status**: FINAL  
> **Sources**: Live web research via 4 parallel research agents + codebase exploration  
> **Purpose**: Corrects and extends specs 02-07 with ground-truth July 2026 data

## Critical Corrections to Earlier Specs

The initial specs (02-06) were written using codebase analysis and estimated model data.
This document contains **ground-truth corrections** from official documentation research
conducted on July 4, 2026. All earlier specs should be read with these corrections applied.

---

## 1. OpenAI — Major Model Generation Shift

### CORRECTION: GPT-5.x is Current (Not GPT-4.1)

| Model | Context In | Max Out | Vision | Tools | Thinking | $/M In | $/M Out |
|-------|-----------|---------|--------|-------|----------|--------|---------|
| **gpt-5.5** | 1.05M | 32K | ✅ | ✅ | ❌ | 5.00 | 30.00 |
| **gpt-5.4** | 1.05M | 32K | ✅ | ✅ | ❌ | 2.50 | 15.00 |
| gpt-5.4-mini | 400K | 32K | ✅ | ✅ | ❌ | — | — |
| gpt-5.4-nano | 400K | 32K | ✅ | ✅ | ❌ | — | — |
| gpt-4.1 | 1M | 32K | ✅ | ✅ | ❌ | 2.00 | 8.00 |
| gpt-4.1-mini | 1M | 32K | ✅ | ✅ | ❌ | 0.40 | 1.60 |
| gpt-4.1-nano | 1M | 32K | ✅ | ✅ | ❌ | 0.10 | 0.40 |
| gpt-4o | 128K | 16K | ✅ | ✅ | ❌ | 2.50 | 10.00 |
| gpt-4o-mini | 128K | 16K | ✅ | ✅ | ❌ | 0.15 | 0.60 |
| o4-mini | 200K | 100K | ✅ | ✅ | ✅ | 1.10 | 4.40 |
| o3 | 200K | 100K | ✅ | ✅ | ✅ | 2.00 | 8.00 |
| o3-mini | 200K | 100K | ❌ | ✅ | ✅ | 1.10 | 4.40 |
| o1 | 200K | 100K | ✅ | ✅ | ✅ | 15.00 | 60.00 |

**Key findings:**
- **o-series (o1, o3, o4-mini) and GPT-4.x scheduled for deprecation Oct 23, 2026**
- **Responses API (`/v1/responses`)** is the new recommended interface; Chat Completions still supported
- `openai` npm SDK at v6.45.0
- Static registry should include both current and soon-to-deprecate models

### Discovery Strategy Update

- `/v1/models` still returns IDs without rich metadata — **HYBRID** strategy confirmed
- Must add deprecation awareness: `deprecated: true` for models past Oct 23, 2026

---

## 2. Anthropic — Fable 5 / Opus 4.8 Era

### CORRECTION: Far More Models Than Expected

| Model | API ID | Context | Max Out | Vision | Tools | Thinking | $/M In/Out |
|-------|--------|---------|---------|--------|-------|----------|-----------|
| **Claude Fable 5** | `claude-fable-5` | 1M | 128K | ✅ | ✅ | Adaptive (always on) | 10/50 |
| **Claude Mythos 5** | `claude-mythos-5` | 1M | 128K | ✅ | ✅ | Adaptive (always on) | — |
| **Claude Opus 4.8** | `claude-opus-4-8` | 1M | 128K | ✅ | ✅ | Adaptive only | 5/25 |
| **Claude Sonnet 5** | `claude-sonnet-5` | 1M | 128K | ✅ | ✅ | Adaptive (default on) | 3/15 |
| Claude Opus 4.7 | `claude-opus-4-7` | 1M | 128K | ✅ | ✅ | Adaptive only | — |
| Claude Opus 4.6 | `claude-opus-4-6` | 1M | 128K | ✅ | ✅ | Adaptive + Extended | — |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | 1M | 64K | ✅ | ✅ | Adaptive + Extended | — |
| Claude Sonnet 4.5 | `claude-sonnet-4-5-20250929` | 200K | 64K | ✅ | ✅ | Extended thinking | 3/15 |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | 200K | 64K | ✅ | ✅ | Extended (budget) | 1/5 |

**Critical findings:**
- **1M context window** for Fable 5, Opus 4.8, Sonnet 5 (not 200K as in current codebase)
- **128K max output** for latest models (not 8K or 32K as in current config)
- **Adaptive thinking** is a breaking change — Opus 4.7+ returns **400 error** if you send `thinking: {type: "enabled", budget_tokens: N}`. Must use `thinking: {type: "adaptive"}` with `effort` parameter
- **Effort levels**: `low`, `medium`, `high` (default), `xhigh`, `max`
- **`temperature`/`top_p`/`top_k` return 400** on Opus 4.7+ and Sonnet 5
- **Naming convention changed**: from 4.6 onward, dateless pinned IDs (e.g., `claude-opus-4-8` not `claude-opus-4-8-YYYYMMDD`)
- Claude Opus 4.0 and Sonnet 4.0 are **RETIRED** (Jun 15, 2026)

### Discovery Strategy Update — NOW RICH

The `/v1/models` endpoint returns a **structured `capabilities` object**:

```json
{
  "id": "claude-fable-5",
  "display_name": "Claude Fable 5",
  "max_input_tokens": 1048576,
  "max_tokens": 131072,
  "capabilities": {
    "batch": true,
    "citations": true,
    "code_execution": true,
    "computer_use": true,
    "image_input": true,
    "pdf_input": true,
    "structured_outputs": true,
    "thinking": { "adaptive": true, "enabled": false },
    "effort": { "low": true, "medium": true, "high": true, "xhigh": true, "max": true }
  }
}
```

**This changes the strategy from HYBRID to DYNAMIC** — Anthropic's `/v1/models` now returns everything we need. No static fallback necessary.

---

## 3. Google Gemini — 3.5 Flash is Current Frontier

### CORRECTION: Gemini 3.5 Flash is the Latest Stable Model

| Model | API ID | Context | Max Out | Vision | Tools | Thinking | $/M In/Out |
|-------|--------|---------|---------|--------|-------|----------|-----------|
| **Gemini 3.5 Flash** | `gemini-3.5-flash` | 1M | 65K | ✅ | ✅ | Levels | 1.50/9.00 |
| **Gemini 3.1 Pro** | `gemini-3.1-pro-preview` | 1M | 65K | ✅ | ✅ | Levels | 2.00/12.00 |
| Gemini 3.1 Flash-Lite | `gemini-3.1-flash-lite` | 1M | 65K | ✅ | ✅ | Levels | — |
| Gemini 2.5 Pro | `gemini-2.5-pro` | 1M | 65K | ✅ | ✅ | Budget | 1.25/10.00 |
| Gemini 2.5 Flash | `gemini-2.5-flash` | 1M | 65K | ✅ | ✅ | Budget | 0.30/2.50 |

**Key findings:**
- **Gemini 2.0 series SHUT DOWN** in June 2026
- `gemini-3-flash-preview` still in preview but `gemini-3.5-flash` is now GA stable
- Thinking levels for 3.x: `minimal`, `low`, `medium`, `high` (same as in current code)
- All support: audio input, video input, PDF, code execution, search grounding
- Embeddings: `gemini-embedding-2` (new), `gemini-embedding` (legacy)
- **Nano Banana** family for image generation: `gemini-3.1-flash-image`, `gemini-3-pro-image`

### Specialized Models

| Model | Purpose |
|-------|---------|
| `computer-use-preview` | Computer use agent |
| `gemini-deep-research-preview` | Deep research |
| `antigravity-agent-preview` | Agentic tasks |

### Discovery: Rich API Confirmed

```json
{
  "name": "models/gemini-3.5-flash",
  "displayName": "Gemini 3.5 Flash",
  "inputTokenLimit": 1048576,
  "outputTokenLimit": 65536,
  "supportedGenerationMethods": ["generateContent", "countTokens"],
  "thinking": true,
  "temperature": 1.0,
  "maxTemperature": 2.0
}
```

**DYNAMIC strategy confirmed** — the API returns everything needed.

---

## 4. Ollama — Now Has Capabilities Array

### CORRECTION: `/api/tags` Now Returns Capabilities

Since Ollama 0.30.0, the `/api/tags` response includes a **`capabilities` array**:

```json
{
  "models": [{
    "name": "gemma4:latest",
    "capabilities": ["completion", "vision", "tools", "thinking"]
  }]
}
```

Valid capability strings: `completion`, `vision`, `tools`, `embedding`, `thinking`, `insert`, `image`

**This significantly simplifies the Ollama discovery implementation** — no need to call `/api/show` per model just for basic capability detection.

`/api/show` still needed for: `num_ctx` (context length), full model_info metadata.

---

## 5. LM Studio — Rich Native API

### CORRECTION: v1 API Returns Structured Capabilities

LM Studio 0.4.0+ native API (`GET /api/v1/models`) returns:

```json
{
  "max_context_length": 262144,
  "capabilities": {
    "vision": true,
    "trained_for_tool_use": true,
    "reasoning": {
      "allowed_options": ["off", "on"],
      "default": "on"
    }
  },
  "loaded_instances": [{
    "config": { "context_length": 4096 }
  }]
}
```

**Key distinction**: `max_context_length` = training-time maximum vs `loaded_instances[].config.context_length` = runtime active context.

---

## 6. Mistral — Structured Capabilities Object

### CORRECTION: API Returns Rich Capabilities

```json
{
  "id": "mistral-medium-latest",
  "max_context_length": 32768,
  "capabilities": {
    "completion_chat": true,
    "completion_fim": false,
    "function_calling": true,
    "fine_tuning": false,
    "vision": true,
    "classification": false
  },
  "aliases": ["mistral-medium-2508"],
  "deprecation": null
}
```

**This changes Mistral from HYBRID to DYNAMIC** — the API returns enough metadata.

Notable new models: **Mistral Medium 3.5**, **Mistral Small 4**, **Voxtral TTS**, **OCR 4**, **Leanstral 1.5**

---

## 7. DeepSeek — V4 Models with 1M Context

### CORRECTION: DeepSeek V4 is Current

| Model | Context | Max Out | Thinking | $/M In/Out |
|-------|---------|---------|----------|-----------|
| deepseek-v4-flash | 1M | 384K | ✅ | 0.14/0.28 |
| deepseek-v4-pro | 1M | 384K | ✅ | 0.435/0.87 |

**Legacy aliases `deepseek-chat` and `deepseek-reasoner` deprecated July 24, 2026.**

Discovery API is minimal — returns only `id` and `owned_by`. Static enrichment required.

---

## 8. Together AI — Rich Discovery

Returns `type`, `context_length`, `pricing`, `organization`, `license` per model.
200+ models including DeepSeek V4 Pro, GPT-OSS-120B, Gemma 4, Qwen3.7-Plus.

---

## 9. Groq — Context Window in Discovery

Returns `context_window` and `active` fields. Latest models include `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3-32b`.

---

## Revised Discovery Strategy Matrix

Based on research findings, the strategy assignments change:

```
  REVISED Strategy Matrix (July 2026):

  ┌──────────────────┬──────────────┬──────────────────────────────────┐
  │ Provider         │ Strategy     │ Rationale (updated)               │
  ├──────────────────┼──────────────┼──────────────────────────────────┤
  │ OpenAI           │ HYBRID       │ /v1/models IDs only, no caps     │
  │ Anthropic        │ DYNAMIC  ▲   │ /v1/models now returns rich caps │
  │ Gemini           │ DYNAMIC      │ /v1beta/models comprehensive     │
  │ Ollama           │ DYNAMIC  ▲   │ /api/tags now has capabilities[] │
  │ LM Studio        │ DYNAMIC      │ /api/v1/models rich schema       │
  │ OpenRouter       │ DYNAMIC      │ /api/v1/models richest of all    │
  │ Mistral          │ DYNAMIC  ▲   │ /v1/models now has capabilities{}│
  │ NVIDIA NIM       │ HYBRID       │ /v1/models + static enrichment   │
  │ xAI              │ STATIC       │ No rich discovery API             │
  │ AWS Bedrock      │ HYBRID       │ AWS SDK + static enrichment      │
  │ DeepSeek         │ HYBRID   ▲   │ /models minimal, needs static    │
  │ Groq             │ HYBRID       │ /v1/models has context_window    │
  │ Together AI      │ DYNAMIC  ▲   │ /v1/models rich (type, ctx, $)   │
  │ Cohere           │ DYNAMIC  ▲   │ /v1/models has features[]        │
  │ HuggingFace      │ STATIC       │ No structured discovery          │
  │ VSCode Copilot   │ DYNAMIC      │ Copilot catalog API              │
  └──────────────────┴──────────────┴──────────────────────────────────┘

  ▲ = Changed from initial assessment based on research
```

---

## Impact on Implementation

### Breaking Changes to Detect in Anthropic Provider

1. **Adaptive thinking**: Opus 4.7+ and Sonnet 5 reject `thinking: {type: "enabled", budget_tokens: N}` with 400
2. **Sampling params**: `temperature`, `top_p`, `top_k` return 400 on Opus 4.7+/Sonnet 5
3. **Context window**: 1M tokens for latest models (was 200K)
4. **Max output**: 128K tokens for latest models (was 8K-64K)

### New Provider Capabilities to Add

1. **OpenAI Responses API** (`/v1/responses`) — new recommended interface
2. **Anthropic Adaptive Thinking** — `effort` parameter with levels
3. **Anthropic Beta Headers** — `fast-mode-2026-02-01`, `output-300k-2026-03-24`, etc.
4. **Gemini 3.x Thinking Levels** — already partially implemented
5. **DeepSeek V4** — 1M context, 384K output
6. **Ollama capabilities array** — replaces per-model `/api/show` calls for basic detection

### Azure OpenAI Findings (July 2026)

> **ADDED 2026-07-04**: Official data from learn.microsoft.com

1. **Model lineup massively expanded**: GPT-5.5, GPT-5.4 (pro/mini/nano), GPT-5.3 (chat/codex),
   GPT-5.2 (codex/chat), GPT-5.1 (chat/codex/codex-mini/codex-max), GPT-5 (pro/codex/mini/nano/chat)
2. **Responses API**: New v1 endpoint (`/openai/v1/`) — recommended over Chat Completions for new development
3. **API version**: Latest is `2025-06-01` for management plane; data plane models API still `2024-10-21`
4. **o-series reasoning models**: o4-mini, o3, o3-pro, o3-mini, o1, codex-mini — all with structured outputs
5. **Discovery limitation**: Data plane `GET /openai/models` returns capability flags but NO context lengths.
   Management plane `GET .../deployments` requires Azure AD token (not API key).
6. **Deployment-centric model**: Users create named deployments. Deployment name ≠ model ID.
   Discovery must map deployment → model → capabilities.
7. **Codebase gap**: `DEFAULT_API_VERSION` is `2024-10-21` (stale), context hardcoded to 128K for all models

Source: https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure

### AWS Bedrock Findings (July 2026)

> **ADDED 2026-07-04**: Official data from docs.aws.amazon.com/bedrock

1. **Claude 5.x on Bedrock**: Claude Sonnet 5 (2026-06-30), Claude Fable 5 (2026-06-09),
   Claude Opus 4.8 (2026-05-28), Claude Sonnet 4.6 (2026-02-17), Claude Opus 4.6 (2026-02-05)
2. **Simplified model IDs**: New Claude models use `anthropic.claude-sonnet-5` (no date/version suffix)
   while older ones still use `anthropic.claude-3-5-sonnet-20241022-v2:0` format
3. **Amazon Nova 2 series**: Nova 2 Lite, Nova 2 Sonic (with audio), Nova Premier (1M context, thinking)
4. **Meta Llama 4**: Maverick 17B (1M context, vision), Scout 17B (512K context, vision)
5. **OpenAI on Bedrock**: `gpt-oss-120b` and `gpt-oss-20b` (open-weight reasoning models)
6. **ListFoundationModels API**: Returns modelId, modalities, lifecycle, streaming — but NOT context
   lengths, tool support, or thinking support. Requires `aws_sdk_bedrock` (not runtime).
7. **Thinking type breaking change**: Claude 4.7+ requires `{type: "adaptive"}` — code sends `{type: "enabled"}`
8. **Context length issue**: `MODEL_RULES` maps `anthropic.claude-4` prefix to 200K,
   but Claude 4.6+/4.7 have 1M context. Prefix match is too broad.
9. **Bedrock Mantle**: New OpenAI-compatible endpoint `/models` for discovery (alternative to native API)
10. **100+ Marketplace models**: Beyond foundation models, Bedrock Marketplace has specialized models
    that are NOT covered by `ListFoundationModels`

Source: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html

### Static Registry Updates Required

The `model_config.rs::builtin_defaults()` needs these updates:
- Add GPT-5.5, GPT-5.4 family
- Add Claude Fable 5, Opus 4.8, Sonnet 5
- Add Gemini 3.5 Flash, remove 2.0 models
- Update context windows (many now 1M)
- Update max output tokens (Claude 128K, DeepSeek 384K)
- Mark deprecated models
- Add DeepSeek V4, Together, Groq, Cohere provider configs
- **Azure OpenAI**: Add GPT-5.x series with 400K context, o-series with 200K context
- **AWS Bedrock**: Update Claude 4.6+ to 1M context, add Nova 2, Llama 4, gpt-oss models
- **Bedrock MODEL_RULES**: Split `anthropic.claude-4` prefix into version-specific rules
