# 01 — Five-WHY Root Cause Analysis

> **Lens**: Product Owner  
> **Date**: 2026-07-04  
> **Status**: DRAFT

## Problem Statement

EdgeQuake LLM is a mature multi-provider Rust abstraction (v0.6.26) supporting 16+ providers. However, AI agent harnesses built on top of it cannot programmatically discover what models are available from a provider, nor can they query model capabilities (context window, vision, tool calling, thinking) to make intelligent routing decisions at runtime.

---

## Five-WHY Deep Dive

### WHY #1: Why can't agent harnesses auto-select the best model?

**Because** there is no unified API to query "what models does provider X support, and what can each model do?" The `LLMProvider` trait exposes `max_context_length()` and boolean capability methods (`supports_streaming`, `supports_function_calling`, `supports_json_mode`), but these are **instance methods** on an already-instantiated provider — you must already know the model to create the provider.

```
  Current Flow (broken for agents):

  ┌─────────┐    "Which model?"   ┌──────────────┐
  │  Agent   │ ──────────────────► │  ??? (gap)   │
  │ Harness  │                     └──────────────┘
  └─────────┘
       │
       │  Must hardcode model name
       ▼
  ┌──────────────────┐     ┌───────────────────┐
  │ ProviderFactory:: │ ──► │ OpenAIProvider     │
  │ create_with_model │     │ (already bound)    │
  └──────────────────┘     └───────────────────┘
```

### WHY #2: Why is there no unified discovery API?

**Because** the existing architecture was designed around **static configuration**. The `ModelCapabilities` struct exists in `model_config.rs` with rich fields (context_length, max_output_tokens, supports_vision, supports_function_calling, supports_thinking), but it's only used for **TOML-based static config** — not connected to the `LLMProvider` trait or factory.

The `ModelsConfig::builtin_defaults()` hardcodes model cards, but:
- It's **stale** (e.g., OpenAI models list GPT-4o but not GPT-4.1, o3, o4-mini)
- It's **incomplete** (Gemini, xAI, Bedrock providers have no model cards at all)
- It's **disconnected** from providers that CAN discover models dynamically

### WHY #3: Why are static configs stale and incomplete?

**Because** there's a **dual-source problem**: some providers (Ollama, LM Studio, OpenRouter, NVIDIA, Mistral) already implement dynamic model listing (`list_models()`, `list_models_cached()`), but each does it differently with incompatible return types. The static `ModelCard` config and the dynamic discovery exist in parallel universes.

```
  The Dual-Source Problem:

  ┌──────────────────────────┐      ┌─────────────────────────────┐
  │   Static ModelCard       │      │   Dynamic list_models()     │
  │   (model_config.rs)      │      │   (per-provider methods)    │
  ├──────────────────────────┤      ├─────────────────────────────┤
  │ ✓ OpenAI (2 models)     │      │ ✗ OpenAI (no list_models)   │
  │ ✓ Anthropic (3 models)  │      │ ✗ Anthropic (no list_models)│
  │ ✓ Ollama (6 models)     │      │ ✓ Ollama (list_models)      │
  │ ✗ Gemini (0 cards)      │      │ ✗ Gemini (no list_models)   │
  │ ✗ xAI (0 cards)         │      │ ✗ xAI (no list_models)      │
  │ ✗ Bedrock (0 cards)     │      │ ✗ Bedrock (no list_models)  │
  │ ✓ Mistral (5 models)    │      │ ✓ Mistral (list_models)     │
  │ ✓ NVIDIA (0 cards)      │      │ ✓ NVIDIA (list_models)      │
  │ ✗ OpenRouter (0 cards)  │      │ ✓ OpenRouter (list_models)  │
  │ ✗ LM Studio (2 cards)   │      │ ✓ LM Studio (native API)   │
  └──────────────────────────┘      └─────────────────────────────┘
           │                                    │
           └─── NOT CONNECTED ──────────────────┘
```

### WHY #4: Why do dynamic and static systems not connect?

**Because** there's no **common trait or interface** for model discovery. Each provider that supports dynamic listing defines its own:
- `OllamaProvider::list_models()` → `OllamaModelsResponse`
- `OpenRouterProvider::list_models_cached()` → `Vec<ModelInfo>`
- `NvidiaProvider::list_models()` → `NvidiaModelsResponse`
- `MistralProvider::list_models()` → `Vec<MistralModel>`
- `LMStudioProvider` → uses native `/api/v1/models` endpoint

Each has incompatible model metadata shapes, none normalize to `ModelCard`/`ModelCapabilities`.

### WHY #5: Why was no common trait designed from the start?

**Because** the project evolved organically — providers were added one at a time, each prioritizing **chat/streaming correctness** over discovery. The trait design (`LLMProvider`, `EmbeddingProvider`) focused on the **call path** (send messages, get responses) not the **meta path** (what can I send, what will I get back). This is the core architectural gap.

---

## Root Cause Summary

```
  ┌─────────────────────────────────────────────────────┐
  │           ROOT CAUSE DIAGRAM                         │
  │                                                      │
  │  Missing: ModelDiscoveryProvider trait                │
  │      │                                               │
  │      ├── No unified discovery API                    │
  │      │    ├── Dynamic providers: incompatible shapes  │
  │      │    └── Static providers: stale hardcoded data  │
  │      │                                               │
  │      ├── ModelCapabilities disconnected from trait    │
  │      │    ├── Exists in model_config.rs (rich)        │
  │      │    └── Not queryable via LLMProvider           │
  │      │                                               │
  │      └── Agent harnesses cannot:                     │
  │           ├── List available models per provider      │
  │           ├── Query capabilities before instantiation │
  │           ├── Auto-select best model for task         │
  │           └── Adapt to provider-specific limits       │
  └─────────────────────────────────────────────────────┘
```

---

## Impact Analysis

| Impact Area | Severity | Description |
|------------|----------|-------------|
| Agent Routing | **CRITICAL** | Agents cannot select models based on capabilities |
| Context Management | **HIGH** | No reliable way to know input/output token limits |
| Vision Pipeline | **HIGH** | No way to check if model supports images before sending |
| Tool Calling | **HIGH** | No way to verify tool support before sending tools |
| Thinking/Reasoning | **MEDIUM** | Cannot route reasoning tasks to thinking-capable models |
| Cost Optimization | **MEDIUM** | Cannot compare model costs programmatically |
| Provider Staleness | **HIGH** | Hardcoded model lists become outdated within weeks |

---

## Success Criteria

1. **Any provider** can be queried for available models without prior knowledge
2. **Every model** exposes a normalized `ModelCapabilities` (context, vision, tools, thinking)
3. **Dynamic providers** (Ollama, LM Studio, OpenRouter, NVIDIA, Mistral) use live API discovery
4. **Static providers** (OpenAI, Anthropic, Gemini, xAI, Bedrock) use a built-in registry updated to July 2026 models
5. **Zero breaking changes** to existing `LLMProvider`, `EmbeddingProvider` traits
6. **Single source of truth** — unified `ModelDiscoveryProvider` trait bridges static and dynamic
7. **Agent-ready** — harnesses can discover, filter, and select models programmatically
