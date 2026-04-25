# NVIDIA NIM Provider

> **First-principles specification for the NVIDIA NIM integration in edgequake-llm.**

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [API Reference](#api-reference)
4. [Authentication](#authentication)
5. [Model Catalog](#model-catalog)
6. [Dynamic Model Discovery](#dynamic-model-discovery)
7. [Special Parameters](#special-parameters)
8. [Edge Cases & Error Handling](#edge-cases--error-handling)
9. [Configuration](#configuration)
10. [Quick Start](#quick-start)

---

## Overview

NVIDIA NIM (NVIDIA Inference Microservices) provides a hosted inference platform at
`https://integrate.api.nvidia.com/v1` that exposes hundreds of AI models — including
NVIDIA's own Nemotron / NemoGuard families, Meta Llama, DeepSeek, Mistral, Qwen, and
more — through a single **OpenAI-compatible** REST API.

### Key Properties

| Property | Value |
|----------|-------|
| Base URL | `https://integrate.api.nvidia.com/v1` |
| Auth Header | `Authorization: Bearer <NVIDIA_API_KEY>` |
| Chat Endpoint | `POST /v1/chat/completions` |
| Model List Endpoint | `GET /v1/models` |
| Protocol | OpenAI-compatible SSE streaming |
| Free Tier | ~1 000 requests/model/month with API key |

### Why a Dedicated Provider (not just OpenAI-Compatible)?

While NVIDIA NIM is OpenAI-compatible, a dedicated provider adds:

1. **Static catalog** — typed model cards with context lengths, vision flags, and
   thinking-capable flags, giving users IntelliSense / autocomplete for model names.
2. **Dynamic listing** — `NvidiaProvider::list_models()` fetches live inventory from
   `GET /v1/models` and enriches it with free/paid status.
3. **Thinking / reasoning passthrough** — `CompletionOptions::reasoning_effort` is
    forwarded as the top-level `reasoning_effort` field for NVIDIA OpenAI-compatible
    chat requests.
4. **202 async polling support** — Some large NVIDIA requests return HTTP 202
    (`Accepted`) with an `NVCF-REQID` header. The provider automatically polls
    NVIDIA's status endpoint until completion, then returns a normal `LLMResponse`.
5. **Free model tracking** — `NvidiaModelInfo::is_free` reflects NVIDIA's rate-tier
   so users can filter for zero-cost experiments.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     NvidiaProvider architecture                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   User Request                                                               │
│        │                                                                     │
│        ▼                                                                     │
│  ┌──────────────────┐   chat / stream / tools                               │
│  │  NvidiaProvider  │ ─────────────────────────►  OpenAICompatibleProvider  │
│  │  (wrapper)       │                              POST /v1/chat/completions │
│  │                  │ ◄────────────────────────── (SSE + function calling)   │
│  │                  │                                                        │
│  │                  │   list_models()                                        │
│  │                  │ ─────────────────────────►  reqwest GET /v1/models    │
│  │                  │ ◄────────────────────────── NvidiaModelsResponse       │
│  │                  │                                                        │
│  │                  │   HTTP 202 (if queued)                                │
│  │                  │ ─────────────────────────► poll /v2/nvcf/pexec/status │
│  │                  │ ◄────────────────────────── until non-202              │
│  └──────────────────┘                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

The `NvidiaProvider` is a **thin wrapper** around `OpenAICompatibleProvider` — the
same battle-tested HTTP client used by xAI, Mistral, and others — so SSE streaming,
tool calling, vision, JSON mode, and retry logic are all inherited for free.

---

## API Reference

### Chat Completions

```
POST https://integrate.api.nvidia.com/v1/chat/completions
Authorization: Bearer <NVIDIA_API_KEY>
Content-Type: application/json
```

**Standard fields (OpenAI-compatible):**

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model ID, e.g. `"nvidia/llama-3.3-nemotron-super-49b-v1"` |
| `messages` | array | Role / content pairs |
| `temperature` | float | Sampling temperature (0–2) |
| `top_p` | float | Top-p nucleus sampling |
| `max_tokens` | int | Maximum output tokens |
| `stream` | bool | Enable SSE streaming |
| `tools` | array | Function / tool definitions |
| `tool_choice` | string/object | Tool selection strategy |

**NVIDIA-specific fields:**

| Field | Type | Models | Description |
|-------|------|--------|-------------|
| `reasoning_effort` | string | Thinking-capable models | Forwarded directly from `CompletionOptions::reasoning_effort` (commonly `"low"`, `"medium"`, `"high"`) |

### Model Listing

```
GET https://integrate.api.nvidia.com/v1/models
Authorization: Bearer <NVIDIA_API_KEY>
```

Returns an OpenAI-format models list (`object`, `data[]`) with model-level fields
(`id`, `object`, `created`, `owned_by`). `NvidiaProvider` enriches each model with an
additional local `is_free` flag derived from a static allowlist.

---

## Authentication

NVIDIA NIM uses a standard Bearer token:

```bash
export NVIDIA_API_KEY="nvapi-..."
```

Obtain a free API key at: <https://build.nvidia.com>

### Provider Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NVIDIA_API_KEY` | ✅ Yes | — | API key from build.nvidia.com |
| `NVIDIA_MODEL` | ❌ No | `nvidia/llama-3.3-nemotron-super-49b-v1` | Default chat model |
| `NVIDIA_BASE_URL` | ❌ No | `https://integrate.api.nvidia.com/v1` | Endpoint override (for self-hosted NIMs) |

---

## Model Catalog

### NVIDIA Nemotron Series (First-party)

| Model ID | Context | Thinking | Notes |
|----------|---------|----------|-------|
| `nvidia/llama-3.3-nemotron-super-49b-v1` | 128K | ✓ | Latest flagship (free tier) |
| `nvidia/llama-3.3-nemotron-super-49b-v1.5` | 128K | ✓ | v1.5 update |
| `nvidia/llama-3.1-nemotron-ultra-253b-v1` | 128K | ✓ | Largest Nemotron (paid) |
| `nvidia/llama-3.1-nemotron-nano-8b-v1` | 128K | ✓ | Compact (free tier) |
| `nvidia/llama-3.1-nemotron-nano-4b-v1_1` | 128K | ✓ | Nano 4B v1.1 (free) |
| `nvidia/nemotron-3-nano-30b-a3b` | 1M | ✓ | Mamba-Transformer MoE (free) |
| `nvidia/nemotron-3-super-120b-a12b` | 1M | ✓ | Super MoE (free) |

### DeepSeek Series (Reasoning)

| Model ID | Context | Thinking | Notes |
|----------|---------|----------|-------|
| `deepseek-ai/deepseek-v4-flash` | 64K | ✓ | Reasoning via `reasoning_effort` |
| `deepseek-ai/deepseek-v4-pro` | 64K | ✓ | Pro variant (paid) |
| `deepseek-ai/deepseek-v3.2` | 128K | — | Non-thinking |
| `deepseek-ai/deepseek-v3.1-terminus` | 128K | — | Terminus variant |
| `deepseek-ai/deepseek-r1` | 128K | ✓ | Original R1 reasoning model |

### Meta Llama Series

| Model ID | Context | Vision | Notes |
|----------|---------|--------|-------|
| `meta/llama-3.3-70b-instruct` | 128K | — | Latest Llama 3.3 (free tier) |
| `meta/llama-3.1-405b-instruct` | 128K | — | Large flagship (paid) |
| `meta/llama-3.1-70b-instruct` | 128K | — | 70B instruct (free tier) |
| `meta/llama-3.1-8b-instruct` | 128K | — | 8B instruct (free tier) |
| `meta/llama-3.2-3b-instruct` | 128K | — | 3B compact (free) |
| `meta/llama-3.2-1b-instruct` | 128K | — | 1B (free) |
| `meta/llama-4-maverick-17b-128e-instruct` | 1M | ✓ | Llama 4 multimodal |

### Microsoft Phi Series

| Model ID | Context | Notes |
|----------|---------|-------|
| `microsoft/phi-4-mini-instruct` | 128K | Phi-4 mini (free) |
| `microsoft/phi-4-mini-flash-reasoning` | 128K | Reasoning variant (free) |
| `microsoft/phi-3.5-mini` | 128K | Phi 3.5 Mini |
| `microsoft/phi-3-mini-128k-instruct` | 128K | Phi-3 mini |

### Mistral on NVIDIA

| Model ID | Context | Notes |
|----------|---------|-------|
| `mistralai/mistral-nemotron` | 128K | Mistral + NVIDIA optimization |
| `mistralai/mistral-small-24b-instruct` | 128K | Mistral Small 3 |
| `mistralai/mixtral-8x22b-instruct` | 65K | Mixtral MoE |
| `mistralai/mixtral-8x7b-instruct` | 32K | Mixtral 8x7B |
| `mistralai/mistral-7b-instruct-v0.3` | 32K | Mistral 7B v0.3 (free) |

### Qwen Series

| Model ID | Context | Notes |
|----------|---------|-------|
| `qwen/qwen2.5-7b-instruct` | 128K | Qwen 2.5 7B (free) |
| `qwen/qwen2.5-coder-32b-instruct` | 128K | Coding specialist |
| `qwen/qwen3-coder-480b-a35b-instruct` | 128K | Qwen3 Coder MoE |
| `qwen/qwq-32b` | 128K | Reasoning model |

### Moonshot Kimi Series

| Model ID | Context | Notes |
|----------|---------|-------|
| `moonshotai/kimi-k2-instruct` | 128K | Kimi K2 (free) |
| `moonshotai/kimi-k2-thinking` | 128K | Reasoning variant |

---

## Dynamic Model Discovery

The NVIDIA provider supports **live model listing** via:

```rust
let provider = NvidiaProvider::from_env()?;
let models = provider.list_models().await?;

for model in &models.data {
    println!("{} (owner={}, free={})", model.id, model.owned_by, model.is_free);
}
```

### Free vs. Paid Detection

NVIDIA's `/v1/models` endpoint does not expose a stable free/paid field in the
OpenAI-compatible schema. The provider sets `NvidiaModelInfo::is_free` by checking
whether each model ID appears in the static `NVIDIA_FREE_MODELS` allowlist.

Models in the **free tier** (as of April 2026, subject to change):
- All `nvidia/llama-3.1-nemotron-nano-*`
- `nvidia/nemotron-3-nano-30b-a3b`
- `nvidia/nemotron-3-super-120b-a12b`
- `meta/llama-3.1-8b-instruct`
- `meta/llama-3.1-70b-instruct`
- `meta/llama-3.2-1b-instruct`, `meta/llama-3.2-3b-instruct`
- `meta/llama-3.3-70b-instruct`
- `mistralai/mistral-7b-instruct-v0.3`
- `microsoft/phi-4-mini-instruct`
- `microsoft/phi-4-mini-flash-reasoning`
- `qwen/qwen2.5-7b-instruct`
- `moonshotai/kimi-k2-instruct`

> **Note**: Free tier = ~1 000 requests/month per model. Check
> [build.nvidia.com](https://build.nvidia.com) for the latest limits.

---

## Special Parameters

### Thinking / Reasoning (DeepSeek V4 Flash)

`NvidiaProvider` forwards `CompletionOptions::reasoning_effort` as a top-level
`reasoning_effort` request field:

```json
{
  "model": "deepseek-ai/deepseek-v4-flash",
  "messages": [{"role": "user", "content": "Solve: x² + 5x + 6 = 0"}],
  "reasoning_effort": "high",
  "max_tokens": 16384
}
```

The provider accepts this via `CompletionOptions::reasoning_effort`:

```rust
let opts = CompletionOptions {
    reasoning_effort: Some("high".to_string()),
    max_tokens: Some(16384),
    ..Default::default()
};
```

Common values: `"low"`, `"medium"`, `"high"`.

---

## Edge Cases & Error Handling

### 202 Async Response

Some long-running NVIDIA requests return HTTP **202 Accepted**. The request ID is
provided in the `NVCF-REQID` response header. `NvidiaProvider` handles this
automatically by polling:

```
GET https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{NVCF-REQID}
```

Polling continues until the response is no longer 202:
- `200`: parse and return a normal `LLMResponse`
- `4xx/5xx`: return `LlmError::ApiError`
- timeout window reached: return `LlmError::ApiError` with polling details

### Rate Limiting (429)

NVIDIA enforces per-model rate limits. The provider surfaces:
```
LlmError::RateLimitError("NVIDIA rate limit exceeded (429): ...")
```

The caller can use the `RetryExecutor` with exponential backoff.

### Model Not Available (404 / 422)

When a model is unavailable or the model ID is wrong, NVIDIA returns 422 or 404.
The provider converts these to `LlmError::InvalidRequest`.

### Empty API Key

The provider validates the key at construction time:
```
LlmError::ConfigError("NVIDIA_API_KEY is empty. Get your key from https://build.nvidia.com")
```

### Streaming with Thinking Models

DeepSeek V4 Flash emits `<think>...</think>` tokens inline in the stream. The provider
does **not** strip these — they appear in `StreamChunk::content`. Callers that want
to separate thinking from the final answer should parse the `<think>` tags themselves.

---

## Configuration

### Via Environment Variables (simplest)

```bash
export NVIDIA_API_KEY="nvapi-..."
export NVIDIA_MODEL="meta/llama-3.3-70b-instruct"  # optional
cargo run --example nvidia_chat
```

### Via models.toml

```toml
[[providers]]
name = "nvidia"
display_name = "NVIDIA NIM"
type = "openai_compatible"
api_key_env = "NVIDIA_API_KEY"
base_url = "https://integrate.api.nvidia.com/v1"
default_llm_model = "nvidia/llama-3.3-nemotron-super-49b-v1"

[[providers.models]]
name = "nvidia/llama-3.3-nemotron-super-49b-v1"
context_length = 131072
```

### Via ProviderFactory (runtime selection)

```rust
use edgequake_llm::{ProviderFactory, ProviderType};

std::env::set_var("NVIDIA_API_KEY", "nvapi-...");
let (llm, _embedding) = ProviderFactory::create(ProviderType::Nvidia)?;
```

### Auto-detection

The factory auto-detects NVIDIA when `NVIDIA_API_KEY` is set and no higher-priority
provider is configured:

```bash
export NVIDIA_API_KEY="nvapi-..."
# ProviderFactory::from_env() → NvidiaProvider
```

---

## Quick Start

```rust
use edgequake_llm::{NvidiaProvider, LLMProvider};
use edgequake_llm::traits::ChatMessage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = NvidiaProvider::from_env()?;
    
    let messages = vec![
        ChatMessage::user("Explain the Transformer architecture in 3 sentences."),
    ];
    
    let response = provider.chat(&messages, None).await?;
    println!("{}", response.content);
    
    Ok(())
}
```

### List Free Models

```rust
use edgequake_llm::NvidiaProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = NvidiaProvider::from_env()?;
    let models = provider.list_models().await?;
    
    println!("Free models available:");
    for model in models.data.iter().filter(|m| m.is_free) {
        println!("  - {}", model.id);
    }
    
    Ok(())
}
```

### Streaming with Reasoning

```rust
use edgequake_llm::{NvidiaProvider, LLMProvider};
use edgequake_llm::traits::{ChatMessage, CompletionOptions};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = NvidiaProvider::from_env()?
        .with_model("deepseek-ai/deepseek-v4-flash");
    
    let messages = vec![
        ChatMessage::user("Prove that sqrt(2) is irrational."),
    ];
    
    let opts = CompletionOptions {
        reasoning_effort: Some("high".to_string()),
        max_tokens: Some(4096),
        ..Default::default()
    };
    
    let mut stream = provider.chat_stream(&messages, Some(&opts)).await?;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(c) => print!("{}", c.content),
            Err(e) => eprintln!("Error: {}", e),
        }
    }
    
    Ok(())
}
```

---

## See Also

- [NVIDIA NIM API Reference](https://docs.api.nvidia.com/nim/reference)
- [NVIDIA Build Platform](https://build.nvidia.com)
- [Provider Families](../../provider-families.md)
- [OpenAI-Compatible Provider](../../../src/providers/openai_compatible.rs)
