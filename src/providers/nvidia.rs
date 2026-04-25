//! NVIDIA NIM Provider — First-class integration with NVIDIA's hosted inference platform.
//!
//! @implements FEAT-030: NVIDIA NIM provider (chat, streaming, tools, model listing)
//!
//! # Overview
//!
//! NVIDIA NIM (`integrate.api.nvidia.com`) exposes hundreds of models via an
//! OpenAI-compatible REST API, including NVIDIA's own Nemotron family, Meta Llama,
//! DeepSeek, Mistral, Qwen, and more.
//!
//! This provider is a **thin wrapper** around [`OpenAICompatibleProvider`] that adds:
//! - Static typed model catalog with context lengths, vision, and thinking flags
//! - Dynamic model discovery via `GET /v1/models` (`list_models()`)
//! - Free-tier model tracking (`NvidiaModelInfo::is_free`)
//! - `reasoning_effort` support for DeepSeek V4 Flash and other thinking models
//! - 202 async response detection (surfaces `requestId` in error message)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                     NvidiaProvider architecture                          │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │   User Request                                                           │
//! │        │                                                                 │
//! │        ▼                                                                 │
//! │  ┌──────────────────┐  chat / stream / tools                            │
//! │  │  NvidiaProvider  │ ──────────────────────► OpenAICompatibleProvider  │
//! │  │  (wrapper)       │                         POST /v1/chat/completions  │
//! │  │                  │ ◄─────────────────────── (SSE + function calling) │
//! │  │                  │                                                    │
//! │  │                  │  list_models()                                     │
//! │  │                  │ ──────────────────────► reqwest GET /v1/models    │
//! │  │                  │ ◄──────────────────────  NvidiaModelsResponse     │
//! │  └──────────────────┘                                                   │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Environment Variables
//!
//! | Variable | Required | Default | Description |
//! |----------|----------|---------|-------------|
//! | `NVIDIA_API_KEY` | ✅ Yes | — | API key from build.nvidia.com |
//! | `NVIDIA_MODEL` | ❌ No | `nvidia/llama-3.3-nemotron-super-49b-v1` | Default model |
//! | `NVIDIA_BASE_URL` | ❌ No | `https://integrate.api.nvidia.com/v1` | Endpoint override |
//!
//! # Quick Start
//!
//! ```bash
//! export NVIDIA_API_KEY=nvapi-...
//! cargo run --example nvidia_chat
//! ```
//!
//! ```rust,no_run
//! use edgequake_llm::{NvidiaProvider, LLMProvider};
//! use edgequake_llm::traits::ChatMessage;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = NvidiaProvider::from_env()?;
//! let messages = vec![ChatMessage::user("Explain the Transformer in 3 sentences.")];
//! let resp = provider.chat(&messages, None).await?;
//! println!("{}", resp.content);
//! # Ok(())
//! # }
//! ```
//!
//! # Available Chat Models (April 2026)
//!
//! | Model ID | Context | Vision | Thinking | Free |
//! |----------|---------|--------|----------|------|
//! | `nvidia/llama-3.3-nemotron-super-49b-v1` | 128K | — | ✓ | ✓ |
//! | `nvidia/nemotron-3-nano-30b-a3b` | 1M | — | ✓ | ✓ |
//! | `nvidia/nemotron-3-super-120b-a12b` | 1M | — | ✓ | ✓ |
//! | `nvidia/llama-3.1-nemotron-ultra-253b-v1` | 128K | — | ✓ | — |
//! | `deepseek-ai/deepseek-v4-flash` | 64K | — | ✓ | ✓ |
//! | `deepseek-ai/deepseek-r1` | 128K | — | ✓ | — |
//! | `meta/llama-3.3-70b-instruct` | 128K | — | — | ✓ |
//! | `meta/llama-4-maverick-17b-128e-instruct` | 1M | ✓ | — | ✓ |
//! | `microsoft/phi-4-mini-instruct` | 128K | — | — | ✓ |
//! | `qwen/qwq-32b` | 128K | — | ✓ | ✓ |
//! | `moonshotai/kimi-k2-instruct` | 128K | — | — | ✓ |

use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::debug;

use crate::error::{LlmError, Result};
use crate::model_config::{
    ModelCapabilities, ModelCard, ModelType, ProviderConfig, ProviderType as ConfigProviderType,
};
use crate::providers::openai_compatible::OpenAICompatibleProvider;
use crate::traits::StreamChunk;
use crate::traits::{
    ChatMessage, ChatRole, CompletionOptions, EmbeddingProvider, LLMProvider, LLMResponse,
    ToolChoice, ToolDefinition,
};

// ============================================================================
// Constants
// ============================================================================

/// NVIDIA NIM API base URL (includes /v1 prefix for OpenAI compatibility)
const NVIDIA_BASE_URL: &str = "https://integrate.api.nvidia.com/v1";

/// Default chat model — Nemotron Super 49B is the recommended free-tier flagship.
///
/// This model is in the free tier (~1 000 req/month) and supports thinking mode.
/// Updated: April 2026 (build.nvidia.com).
const NVIDIA_DEFAULT_MODEL: &str = "nvidia/llama-3.3-nemotron-super-49b-v1";

/// Provider display name
const NVIDIA_PROVIDER_NAME: &str = "nvidia";

/// HTTP timeout in seconds (5 minutes).
///
/// Reasoning models like DeepSeek V4 Flash with `reasoning_effort=max` can
/// stream for several minutes. 300s gives ample headroom.
const NVIDIA_TIMEOUT_SECS: u64 = 300;

// ============================================================================
// HTTP 202 Async-Inference Polling Constants
//
// NVIDIA Cloud Functions (NVCF) queues inference requests under high load.
// The initial POST /v1/chat/completions may return HTTP 202, with the
// opaque request ID in the `NVCF-REQID` response header.
//
// Polling:
//   GET https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{nvcf_reqid}
//   → 202: still pending, retry after NVIDIA_POLL_INTERVAL_MS
//   → 200: complete, parse body as OpenAI completion
//   → 4xx/5xx: error
//
// References:
//   LangChain NVIDIA: _common.py::_wait() / _wait_async()
//   NVCF docs: https://docs.nvidia.com/cloud-functions/user-guide/latest/
//              cloud-function/api.html#http-polling
// ============================================================================

/// NVCF async status polling base URL.
const NVIDIA_NVCF_STATUS_URL: &str = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status";

/// Response header that carries the NVCF request ID on HTTP 202 responses.
const NVIDIA_REQID_HEADER: &str = "NVCF-REQID";

/// Interval in milliseconds between polling attempts after a 202 response.
const NVIDIA_POLL_INTERVAL_MS: u64 = 500;

/// Maximum number of poll attempts before giving up.
///
/// 600 × 500 ms = 300 seconds (5 minutes).
const NVIDIA_MAX_POLL_ATTEMPTS: u32 = 600;

// ============================================================================
// Model Catalog
//
// Format: (id, display_name, context_length, supports_vision, supports_thinking, is_free)
//
// Context lengths are sourced from NVIDIA NIM API docs (April 2026).
// 1M = 1_000_000, 128K = 131_072, 64K = 65_536, 32K = 32_768.
//
// Free tier: ~1 000 requests/month/model with a free NVIDIA API key.
// Source: build.nvidia.com (April 2026).
// ============================================================================

/// NVIDIA NIM chat model catalog.
///
/// Tuple: `(id, display_name, context_length, vision, thinking, free_tier)`
const NVIDIA_CHAT_MODELS: &[(&str, &str, usize, bool, bool, bool)] = &[
    // ─────────────────────────────────────────────────────────────────────────
    // NVIDIA Nemotron family — first-party, NVIDIA-trained
    // ─────────────────────────────────────────────────────────────────────────
    (
        "nvidia/llama-3.3-nemotron-super-49b-v1",
        "Nemotron Super 49B v1 (128K, thinking, free)",
        131_072,
        false,
        true, // supports thinking
        true, // free tier
    ),
    (
        "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "Nemotron Super 49B v1.5 (128K, thinking, free)",
        131_072,
        false,
        true,
        true,
    ),
    (
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "Nemotron Ultra 253B v1 (128K, thinking)",
        131_072,
        false,
        true,
        false, // paid
    ),
    (
        "nvidia/llama-3.1-nemotron-nano-8b-v1",
        "Nemotron Nano 8B v1 (128K, thinking, free)",
        131_072,
        false,
        true,
        true,
    ),
    (
        "nvidia/llama-3.1-nemotron-nano-4b-v1_1",
        "Nemotron Nano 4B v1.1 (128K, thinking, free)",
        131_072,
        false,
        true,
        true,
    ),
    // ─────────────────────────────────────────────────────────────────────────
    // NVIDIA Nemotron 3 MoE series (hybrid Mamba-Transformer)
    // ─────────────────────────────────────────────────────────────────────────
    (
        "nvidia/nemotron-3-nano-30b-a3b",
        "Nemotron 3 Nano 30B-A3B MoE (1M, thinking, free)",
        1_000_000,
        false,
        true,
        true,
    ),
    (
        "nvidia/nemotron-3-super-120b-a12b",
        "Nemotron 3 Super 120B-A12B MoE (1M, thinking, free)",
        1_000_000,
        false,
        true,
        true,
    ),
    // ─────────────────────────────────────────────────────────────────────────
    // NVIDIA nemotron-mini / nano variants
    // ─────────────────────────────────────────────────────────────────────────
    (
        "nvidia/nemotron-mini-4b-instruct",
        "Nemotron Mini 4B Instruct (4K)",
        4_096,
        false,
        false,
        false,
    ),
    (
        "nvidia/nvidia-nemotron-nano-9b-v2",
        "Nemotron Nano 9B v2 (128K)",
        131_072,
        false,
        false,
        false,
    ),
    // ─────────────────────────────────────────────────────────────────────────
    // DeepSeek family (with reasoning_effort parameter)
    // ─────────────────────────────────────────────────────────────────────────
    (
        "deepseek-ai/deepseek-v4-flash",
        "DeepSeek V4 Flash (64K, thinking via reasoning_effort, free)",
        65_536,
        false,
        true, // supports reasoning_effort
        true,
    ),
    (
        "deepseek-ai/deepseek-v4-pro",
        "DeepSeek V4 Pro (64K, thinking via reasoning_effort)",
        65_536,
        false,
        true,
        false, // paid
    ),
    (
        "deepseek-ai/deepseek-v3.2",
        "DeepSeek V3.2 (128K)",
        131_072,
        false,
        false,
        false,
    ),
    (
        "deepseek-ai/deepseek-v3.1-terminus",
        "DeepSeek V3.1 Terminus (128K)",
        131_072,
        false,
        false,
        false,
    ),
    (
        "deepseek-ai/deepseek-r1",
        "DeepSeek R1 (128K, thinking)",
        131_072,
        false,
        true,
        false,
    ),
    // ─────────────────────────────────────────────────────────────────────────
    // Meta Llama family
    // ─────────────────────────────────────────────────────────────────────────
    (
        "meta/llama-3.3-70b-instruct",
        "Llama 3.3 70B Instruct (128K, free)",
        131_072,
        false,
        false,
        true,
    ),
    (
        "meta/llama-3.1-405b-instruct",
        "Llama 3.1 405B Instruct (128K)",
        131_072,
        false,
        false,
        false,
    ),
    (
        "meta/llama-3.1-70b-instruct",
        "Llama 3.1 70B Instruct (128K, free)",
        131_072,
        false,
        false,
        true,
    ),
    (
        "meta/llama-3.1-8b-instruct",
        "Llama 3.1 8B Instruct (128K, free)",
        131_072,
        false,
        false,
        true,
    ),
    (
        "meta/llama-3.2-3b-instruct",
        "Llama 3.2 3B Instruct (128K, free)",
        131_072,
        false,
        false,
        true,
    ),
    (
        "meta/llama-3.2-1b-instruct",
        "Llama 3.2 1B Instruct (128K, free)",
        131_072,
        false,
        false,
        true,
    ),
    (
        "meta/llama-4-maverick-17b-128e-instruct",
        "Llama 4 Maverick 17B 128E (1M, vision, free)",
        1_000_000,
        true, // multimodal
        false,
        true,
    ),
    (
        "meta/llama-3.2-11b-vision-instruct",
        "Llama 3.2 11B Vision (128K, vision)",
        131_072,
        true,
        false,
        false,
    ),
    (
        "meta/llama-3.2-90b-vision-instruct",
        "Llama 3.2 90B Vision (128K, vision)",
        131_072,
        true,
        false,
        false,
    ),
    // ─────────────────────────────────────────────────────────────────────────
    // Microsoft Phi family
    // ─────────────────────────────────────────────────────────────────────────
    (
        "microsoft/phi-4-mini-instruct",
        "Phi-4 Mini Instruct (128K, free)",
        131_072,
        false,
        false,
        true,
    ),
    (
        "microsoft/phi-4-mini-flash-reasoning",
        "Phi-4 Mini Flash Reasoning (128K, thinking, free)",
        131_072,
        false,
        true,
        true,
    ),
    (
        "microsoft/phi-4-multimodal-instruct",
        "Phi-4 Multimodal Instruct (128K, vision)",
        131_072,
        true,
        false,
        false,
    ),
    (
        "microsoft/phi-3.5-mini",
        "Phi-3.5 Mini Instruct (128K)",
        131_072,
        false,
        false,
        false,
    ),
    (
        "microsoft/phi-3.5-vision-instruct",
        "Phi-3.5 Vision Instruct (128K, vision)",
        131_072,
        true,
        false,
        false,
    ),
    (
        "microsoft/phi-3-mini-128k-instruct",
        "Phi-3 Mini 128K Instruct (128K)",
        131_072,
        false,
        false,
        false,
    ),
    (
        "microsoft/phi-3-mini-4k-instruct",
        "Phi-3 Mini 4K Instruct (4K)",
        4_096,
        false,
        false,
        false,
    ),
    (
        "microsoft/phi-3-small-128k-instruct",
        "Phi-3 Small 128K Instruct (128K)",
        131_072,
        false,
        false,
        false,
    ),
    (
        "microsoft/phi-3-medium-128k-instruct",
        "Phi-3 Medium 128K Instruct (128K)",
        131_072,
        false,
        false,
        false,
    ),
    // ─────────────────────────────────────────────────────────────────────────
    // Mistral on NVIDIA NIM
    // ─────────────────────────────────────────────────────────────────────────
    (
        "mistralai/mistral-nemotron",
        "Mistral Nemotron (128K)",
        131_072,
        false,
        false,
        false,
    ),
    (
        "mistralai/mistral-small-24b-instruct",
        "Mistral Small 24B Instruct (128K)",
        131_072,
        false,
        false,
        false,
    ),
    (
        "mistralai/mistral-large-2-instruct",
        "Mistral Large 2 Instruct (128K)",
        131_072,
        false,
        false,
        false,
    ),
    (
        "mistralai/mistral-7b-instruct-v0.3",
        "Mistral 7B Instruct v0.3 (32K, free)",
        32_768,
        false,
        false,
        true,
    ),
    (
        "mistralai/mixtral-8x7b-instruct",
        "Mixtral 8x7B Instruct (32K)",
        32_768,
        false,
        false,
        false,
    ),
    (
        "mistralai/mixtral-8x22b-instruct",
        "Mixtral 8x22B Instruct (65K)",
        65_536,
        false,
        false,
        false,
    ),
    (
        "mistralai/magistral-small-2506",
        "Magistral Small 2506 (128K, thinking)",
        131_072,
        false,
        true,
        false,
    ),
    // ─────────────────────────────────────────────────────────────────────────
    // Qwen family
    // ─────────────────────────────────────────────────────────────────────────
    (
        "qwen/qwen2.5-7b-instruct",
        "Qwen 2.5 7B Instruct (128K, free)",
        131_072,
        false,
        false,
        true,
    ),
    (
        "qwen/qwen2.5-coder-7b-instruct",
        "Qwen 2.5 Coder 7B Instruct (128K)",
        131_072,
        false,
        false,
        false,
    ),
    (
        "qwen/qwen2.5-coder-32b-instruct",
        "Qwen 2.5 Coder 32B Instruct (128K)",
        131_072,
        false,
        false,
        false,
    ),
    (
        "qwen/qwq-32b",
        "QwQ 32B (128K, thinking, free)",
        131_072,
        false,
        true,
        true,
    ),
    (
        "qwen/qwen3-coder-480b-a35b-instruct",
        "Qwen3 Coder 480B MoE (128K)",
        131_072,
        false,
        false,
        false,
    ),
    (
        "qwen/qwen3-next-80b-a3b-instruct",
        "Qwen3 Next 80B-A3B Instruct (128K)",
        131_072,
        false,
        false,
        false,
    ),
    (
        "qwen/qwen3-next-80b-a3b-thinking",
        "Qwen3 Next 80B-A3B Thinking (128K, thinking)",
        131_072,
        false,
        true,
        false,
    ),
    // ─────────────────────────────────────────────────────────────────────────
    // Moonshot Kimi family
    // ─────────────────────────────────────────────────────────────────────────
    (
        "moonshotai/kimi-k2-instruct",
        "Kimi K2 Instruct (128K, free)",
        131_072,
        false,
        false,
        true,
    ),
    (
        "moonshotai/kimi-k2-thinking",
        "Kimi K2 Thinking (128K, thinking)",
        131_072,
        false,
        true,
        false,
    ),
    (
        "moonshotai/kimi-k2-instruct-0905",
        "Kimi K2 Instruct 0905 (128K)",
        131_072,
        false,
        false,
        false,
    ),
    // ─────────────────────────────────────────────────────────────────────────
    // MiniMax family
    // ─────────────────────────────────────────────────────────────────────────
    (
        "minimaxai/minimax-m2.5",
        "MiniMax M2.5 (128K)",
        131_072,
        false,
        false,
        false,
    ),
    // ─────────────────────────────────────────────────────────────────────────
    // Other notable models
    // ─────────────────────────────────────────────────────────────────────────
    (
        "google/gemma-2-9b-it",
        "Gemma 2 9B IT (8K, free)",
        8_192,
        false,
        false,
        true,
    ),
    (
        "google/gemma-2-27b-it",
        "Gemma 2 27B IT (8K)",
        8_192,
        false,
        false,
        false,
    ),
    (
        "marin/marin-8b-instruct",
        "Marin 8B Instruct (128K, free)",
        131_072,
        false,
        false,
        true,
    ),
    (
        "databricks/dbrx-instruct",
        "DBRX Instruct (32K)",
        32_768,
        false,
        false,
        false,
    ),
    (
        "snowflake/arctic",
        "Snowflake Arctic (4K)",
        4_096,
        false,
        false,
        false,
    ),
    (
        "upstage/solar-10.7b-instruct",
        "Solar 10.7B Instruct (4K, free)",
        4_096,
        false,
        false,
        true,
    ),
    (
        "z-ai/glm4.7",
        "GLM-4.7 (128K)",
        131_072,
        false,
        false,
        false,
    ),
    (
        "openai/gpt-oss-20b",
        "OpenAI OSS 20B (128K)",
        131_072,
        false,
        false,
        false,
    ),
    (
        "openai/gpt-oss-120b",
        "OpenAI OSS 120B (128K)",
        131_072,
        false,
        false,
        false,
    ),
    (
        "bytedance/seed-oss-36b-instruct",
        "Seed OSS 36B Instruct (128K)",
        131_072,
        false,
        false,
        false,
    ),
    (
        "stepfun-ai/step-3-5-flash",
        "Step-3.5 Flash (128K)",
        131_072,
        false,
        false,
        false,
    ),
];

/// Known free-tier model IDs (used to enrich live `/v1/models` response).
///
/// NVIDIA does not expose a `free` flag in the models API. This static list
/// reflects the known free-tier models as of April 2026.
/// See: <https://build.nvidia.com> (filter by "Free" in the UI).
const NVIDIA_FREE_MODELS: &[&str] = &[
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nvidia/llama-3.1-nemotron-nano-8b-v1",
    "nvidia/llama-3.1-nemotron-nano-4b-v1_1",
    "nvidia/nemotron-3-nano-30b-a3b",
    "nvidia/nemotron-3-super-120b-a12b",
    "deepseek-ai/deepseek-v4-flash",
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.2-3b-instruct",
    "meta/llama-3.2-1b-instruct",
    "meta/llama-4-maverick-17b-128e-instruct",
    "microsoft/phi-4-mini-instruct",
    "microsoft/phi-4-mini-flash-reasoning",
    "mistralai/mistral-7b-instruct-v0.3",
    "qwen/qwen2.5-7b-instruct",
    "qwen/qwq-32b",
    "moonshotai/kimi-k2-instruct",
    "google/gemma-2-9b-it",
    "marin/marin-8b-instruct",
    "upstage/solar-10.7b-instruct",
];

// ============================================================================
// 202-Aware Non-Streaming Request / Response Types
//
// These local structs mirror the private types in openai_compatible.rs but are
// defined here so NvidiaProvider can build and parse requests directly with its
// own reqwest::Client — which lets us intercept HTTP 202 before the response
// body is consumed.
// ============================================================================

/// Multipart or plain-text message content (serde_json::Value lets us
/// serialise both formats without separate enums).
///
/// - Plain text  → `"string"`
/// - Vision/multipart → `[{"type":"text","text":"…"}, {"type":"image_url",…}]`
type NvidiaMsgContent = serde_json::Value;

/// Single message in the request body (OpenAI format).
#[derive(Debug, Serialize)]
struct NvidiaMessageReq {
    role: String,
    content: NvidiaMsgContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<NvidiaToolCallReq>>,
}

/// Tool call entry in an assistant message.
#[derive(Debug, Serialize)]
struct NvidiaToolCallReq {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: NvidiaFnCallReq,
}

/// Function call entry in a tool call.
#[derive(Debug, Serialize)]
struct NvidiaFnCallReq {
    name: String,
    arguments: String,
}

/// Full chat-completions request body sent to `/v1/chat/completions`.
#[derive(Debug, Serialize)]
struct NvidiaChatReq<'a> {
    model: &'a str,
    messages: Vec<NvidiaMessageReq>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    /// Always `false` for the 202-aware code path (streaming uses the inner provider).
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<NvidiaRespFormat>,
    /// `reasoning_effort` passthrough for DeepSeek and Nemotron thinking models.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
}

/// Response format specifier (for JSON mode).
#[derive(Debug, Serialize)]
struct NvidiaRespFormat {
    #[serde(rename = "type")]
    format_type: String,
}

// --------------- Response types -------------------------------------------

/// Minimal non-streaming completion response (OpenAI-compatible).
#[derive(Debug, Deserialize)]
struct NvidiaChatCompletion {
    #[serde(default)]
    id: Option<String>,
    /// Effective model name (may differ from the requested model when aliased).
    #[serde(default)]
    model: Option<String>,
    choices: Vec<NvidiaCompletionChoice>,
    #[serde(default)]
    usage: Option<NvidiaCompletionUsage>,
}

#[derive(Debug, Deserialize)]
struct NvidiaCompletionChoice {
    message: NvidiaCompletionMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct NvidiaCompletionMessage {
    /// Primary visible content.
    #[serde(default)]
    content: Option<String>,
    /// Internal reasoning from DeepSeek-style thinking models.
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<NvidiaToolCallResp>,
}

#[derive(Debug, Deserialize)]
struct NvidiaToolCallResp {
    id: String,
    #[serde(rename = "type", default)]
    call_type: String,
    function: NvidiaFnCallResp,
}

#[derive(Debug, Deserialize)]
struct NvidiaFnCallResp {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize, Default)]
struct NvidiaCompletionUsage {
    #[serde(default)]
    prompt_tokens: usize,
    #[serde(default)]
    completion_tokens: usize,
}

// ============================================================================
// Model Listing Response Structs
// ============================================================================

/// Response from `GET /v1/models` (OpenAI-compatible format).
#[derive(Debug, Deserialize)]
pub struct NvidiaModelsResponse {
    /// Always `"list"`.
    pub object: String,
    /// Array of model info objects.
    pub data: Vec<NvidiaModelInfo>,
}

/// Information about a single NVIDIA NIM model.
#[derive(Debug, Deserialize)]
pub struct NvidiaModelInfo {
    /// Model identifier, e.g. `"nvidia/llama-3.3-nemotron-super-49b-v1"`.
    pub id: String,
    /// Always `"model"`.
    pub object: String,
    /// Unix timestamp of model creation (may be absent).
    #[serde(default)]
    pub created: i64,
    /// Owning organization slug.
    #[serde(default)]
    pub owned_by: String,
    /// Whether this model is in the free tier.
    ///
    /// Not returned by the API; enriched from the static `NVIDIA_FREE_MODELS` list.
    #[serde(skip)]
    pub is_free: bool,
}

// ============================================================================
// NvidiaProvider
// ============================================================================

/// NVIDIA NIM provider — OpenAI-compatible inference platform.
///
/// Wraps [`OpenAICompatibleProvider`] to add NVIDIA-specific features:
/// - Static typed model catalog
/// - Dynamic `list_models()` via `GET /v1/models`
/// - Free-tier model tagging
/// - `reasoning_effort` / `chat_template_kwargs` passthrough
///
/// See the [module documentation](self) for full usage and configuration.
#[derive(Debug)]
pub struct NvidiaProvider {
    /// Inner OpenAI-compatible provider handling HTTP + SSE.
    inner: OpenAICompatibleProvider,
    /// Currently active chat model.
    model: String,
    /// API key (stored for native list_models request).
    api_key: String,
    /// Base URL (stored for native list_models request).
    base_url: String,
    /// Shared HTTP client for native requests (model listing).
    client: Client,
}

impl NvidiaProvider {
    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    /// Create a provider from environment variables.
    ///
    /// Reads:
    /// - `NVIDIA_API_KEY` (required)
    /// - `NVIDIA_MODEL` (optional)
    /// - `NVIDIA_BASE_URL` (optional)
    ///
    /// # Errors
    ///
    /// Returns `LlmError::ConfigError` if `NVIDIA_API_KEY` is not set or empty.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("NVIDIA_API_KEY").map_err(|_| {
            LlmError::ConfigError(
                "NVIDIA_API_KEY environment variable not set. \
                 Get your free API key from https://build.nvidia.com"
                    .to_string(),
            )
        })?;

        if api_key.is_empty() {
            return Err(LlmError::ConfigError(
                "NVIDIA_API_KEY is empty. Please set a valid API key from https://build.nvidia.com"
                    .to_string(),
            ));
        }

        let model =
            std::env::var("NVIDIA_MODEL").unwrap_or_else(|_| NVIDIA_DEFAULT_MODEL.to_string());
        let base_url =
            std::env::var("NVIDIA_BASE_URL").unwrap_or_else(|_| NVIDIA_BASE_URL.to_string());

        Self::new(api_key, model, Some(base_url))
    }

    /// Create a provider from a [`ProviderConfig`].
    ///
    /// Used by the factory when loading from `models.toml`.
    pub fn from_config(config: &ProviderConfig) -> Result<Self> {
        let api_key = if let Some(env_var) = &config.api_key_env {
            std::env::var(env_var).map_err(|_| {
                LlmError::ConfigError(format!(
                    "API key environment variable '{}' not set for NVIDIA provider.",
                    env_var
                ))
            })?
        } else {
            return Err(LlmError::ConfigError(
                "NVIDIA provider requires api_key_env to be set.".to_string(),
            ));
        };

        if api_key.is_empty() {
            return Err(LlmError::ConfigError(
                "NVIDIA API key is empty.".to_string(),
            ));
        }

        let model = config
            .default_llm_model
            .clone()
            .unwrap_or_else(|| NVIDIA_DEFAULT_MODEL.to_string());
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| NVIDIA_BASE_URL.to_string());

        Self::new(api_key, model, Some(base_url))
    }

    /// Create a provider with explicit configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - NVIDIA API key
    /// * `model` - Chat model ID (e.g., `"meta/llama-3.3-70b-instruct"`)
    /// * `base_url` - Optional custom base URL (defaults to NVIDIA NIM cloud)
    pub fn new(api_key: String, model: String, base_url: Option<String>) -> Result<Self> {
        let base_url = base_url.unwrap_or_else(|| NVIDIA_BASE_URL.to_string());

        let config = Self::build_provider_config(&api_key, &model, &base_url);
        let inner = OpenAICompatibleProvider::from_config(config)?;

        let client = Client::builder()
            .timeout(Duration::from_secs(NVIDIA_TIMEOUT_SECS))
            .build()
            .map_err(|e| LlmError::ConfigError(format!("Failed to build HTTP client: {}", e)))?;

        debug!(
            provider = NVIDIA_PROVIDER_NAME,
            model = %model,
            base_url = %base_url,
            "Created NVIDIA NIM provider"
        );

        Ok(Self {
            inner,
            model,
            api_key,
            base_url,
            client,
        })
    }

    // -------------------------------------------------------------------------
    // Builder methods
    // -------------------------------------------------------------------------

    /// Return a new provider configured for a different model.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use edgequake_llm::NvidiaProvider;
    /// let provider = NvidiaProvider::from_env()?
    ///     .with_model("deepseek-ai/deepseek-v4-flash");
    /// # Ok::<(), edgequake_llm::LlmError>(())
    /// ```
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self.inner = self.inner.with_model(model);
        self
    }

    // -------------------------------------------------------------------------
    // Model catalog helpers
    // -------------------------------------------------------------------------

    /// Context length for a given model ID.
    ///
    /// Falls back to 32 768 (32K) if the model is not in the static catalog.
    /// The fallback is intentionally conservative to avoid silent context overflow.
    pub fn context_length(model: &str) -> usize {
        NVIDIA_CHAT_MODELS
            .iter()
            .find(|(id, _, _, _, _, _)| *id == model)
            .map(|(_, _, ctx, _, _, _)| *ctx)
            .unwrap_or(32_768)
    }

    /// Whether a model supports vision / multimodal inputs.
    pub fn supports_vision(model: &str) -> bool {
        NVIDIA_CHAT_MODELS
            .iter()
            .find(|(id, _, _, _, _, _)| *id == model)
            .map(|(_, _, _, vision, _, _)| *vision)
            // Heuristic for unknown models: check if the name suggests vision
            .unwrap_or_else(|| {
                model.contains("vision")
                    || model.contains("vl")
                    || model.contains("multimodal")
                    || model.contains("maverick")
            })
    }

    /// Whether a model supports thinking / chain-of-thought (via `reasoning_effort`).
    pub fn supports_thinking(model: &str) -> bool {
        NVIDIA_CHAT_MODELS
            .iter()
            .find(|(id, _, _, _, _, _)| *id == model)
            .map(|(_, _, _, _, thinking, _)| *thinking)
            .unwrap_or(false)
    }

    /// Whether a model is in the free tier (~1 000 req/month).
    pub fn is_free_model(model: &str) -> bool {
        NVIDIA_FREE_MODELS.contains(&model)
    }

    /// Return a list of all statically-known chat models.
    ///
    /// Each entry is `(id, display_name, context_length, vision, thinking, free)`.
    pub fn available_models() -> Vec<(&'static str, &'static str, usize, bool, bool, bool)> {
        NVIDIA_CHAT_MODELS.to_vec()
    }

    /// Return only the free-tier models from the static catalog.
    pub fn free_models() -> Vec<(&'static str, &'static str, usize)> {
        NVIDIA_CHAT_MODELS
            .iter()
            .filter(|(_, _, _, _, _, free)| *free)
            .map(|(id, name, ctx, _, _, _)| (*id, *name, *ctx))
            .collect()
    }

    // -------------------------------------------------------------------------
    // Dynamic model listing (live API call)
    // -------------------------------------------------------------------------

    /// Fetch available models from the NVIDIA NIM API.
    ///
    /// Calls `GET {base_url}/models` and enriches each entry with `is_free`
    /// from the static free-tier allowlist.
    ///
    /// # Errors
    ///
    /// Returns error on network failure, non-200 response, or JSON parse failure.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use edgequake_llm::NvidiaProvider;
    /// # async fn example() -> Result<(), edgequake_llm::LlmError> {
    /// let provider = NvidiaProvider::from_env()?;
    /// let models = provider.list_models().await?;
    ///
    /// println!("Total models: {}", models.data.len());
    /// for m in models.data.iter().filter(|m| m.is_free) {
    ///     println!("  FREE: {}", m.id);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_models(&self) -> Result<NvidiaModelsResponse> {
        let url = format!("{}/models", self.base_url.trim_end_matches('/'));

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(format!("Failed to list NVIDIA models: {}", e)))?;

        let status = response.status();

        // Handle 202 (async pending) — unlikely for GET /models but guard it
        if status == reqwest::StatusCode::ACCEPTED {
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!(
                "NVIDIA returned 202 Accepted for model listing (unexpected): {}",
                body
            )));
        }

        let body = response.text().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to read NVIDIA models response: {}", e))
        })?;

        if !status.is_success() {
            return Err(LlmError::ApiError(format!(
                "NVIDIA models list failed ({status}): {body}"
            )));
        }

        let mut resp: NvidiaModelsResponse = serde_json::from_str(&body)
            .map_err(|e| LlmError::ApiError(format!("Failed to parse models response: {e}")))?;

        // Enrich with free-tier status
        for model in &mut resp.data {
            model.is_free = NVIDIA_FREE_MODELS.contains(&model.id.as_str());
        }

        Ok(resp)
    }

    // -------------------------------------------------------------------------
    // HTTP 202 async-inference support
    // -------------------------------------------------------------------------

    /// Convert a `ChatMessage` slice into the request message format.
    ///
    /// Handles:
    /// - Plain text messages (all roles)
    /// - Multipart vision messages (images converted to data-URI image_url parts)
    /// - Assistant messages with tool_calls
    /// - Tool-result messages (role = "tool", tool_call_id set)
    fn build_messages(messages: &[ChatMessage]) -> Vec<NvidiaMessageReq> {
        messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    ChatRole::System => "system",
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                    ChatRole::Tool | ChatRole::Function => "tool",
                };

                // Build multipart content when images are present (vision models).
                let content: serde_json::Value =
                    if msg.images.as_ref().is_some_and(|imgs| !imgs.is_empty()) {
                        let mut parts: Vec<serde_json::Value> = Vec::new();
                        if !msg.content.is_empty() {
                            parts.push(serde_json::json!({"type": "text", "text": &msg.content}));
                        }
                        if let Some(ref images) = msg.images {
                            for img in images {
                                let mut img_obj =
                                    serde_json::json!({"type": "image_url", "image_url": {"url": img.to_data_uri()}});
                                // Forward optional detail hint if present
                                if let Some(ref detail) = img.detail {
                                    img_obj["image_url"]["detail"] =
                                        serde_json::Value::String(detail.clone());
                                }
                                parts.push(img_obj);
                            }
                        }
                        serde_json::Value::Array(parts)
                    } else {
                        serde_json::Value::String(msg.content.clone())
                    };

                // Convert assistant tool_calls if present.
                let tool_calls = msg.tool_calls.as_ref().map(|tcs| {
                    tcs.iter()
                        .map(|tc| NvidiaToolCallReq {
                            id: tc.id.clone(),
                            call_type: tc.call_type.clone(),
                            function: NvidiaFnCallReq {
                                name: tc.function.name.clone(),
                                arguments: tc.function.arguments.clone(),
                            },
                        })
                        .collect::<Vec<_>>()
                });

                NvidiaMessageReq {
                    role: role.to_string(),
                    content,
                    name: msg.name.clone(),
                    tool_call_id: msg.tool_call_id.clone(),
                    tool_calls,
                }
            })
            .collect()
    }

    /// Serialise a `ToolChoice` enum into the JSON value NVIDIA expects.
    fn tool_choice_to_json(choice: &ToolChoice) -> serde_json::Value {
        match choice {
            ToolChoice::Auto(s) | ToolChoice::Required(s) => serde_json::json!(s),
            ToolChoice::Function { function, .. } => serde_json::json!({
                "type": "function",
                "function": {"name": function.name}
            }),
        }
    }

    /// Execute a **non-streaming** chat request with full HTTP 202 async-polling.
    ///
    /// This is the core low-level method used by `chat()`, `complete_with_options()`,
    /// and `chat_with_tools()`. Streaming paths continue to delegate to the inner
    /// `OpenAICompatibleProvider` because NVIDIA never returns 202 for SSE requests.
    ///
    /// # Flow
    ///
    /// ```text
    /// POST /v1/chat/completions
    ///        │
    ///        ├── HTTP 200 ──────────────────────────────► parse_completion_response()
    ///        │
    ///        └── HTTP 202 (NVCF queued)
    ///                 │
    ///                 │  Extract NVCF-REQID header
    ///                 │
    ///                 └── loop: GET /v2/nvcf/pexec/status/{id}
    ///                           ├── 202 → wait NVIDIA_POLL_INTERVAL_MS, retry
    ///                           ├── 200 → parse_completion_response()
    ///                           └── 4xx/5xx → LlmError::ApiError
    /// ```
    ///
    /// # Arguments
    ///
    /// * `messages` – Conversation turn(s).
    /// * `options` – Sampling parameters (temperature, max_tokens, etc.).
    /// * `tools` – Optional tool definitions and choice for function-calling.
    async fn execute_non_streaming_chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
        tools: Option<(&[ToolDefinition], Option<ToolChoice>)>,
    ) -> Result<LLMResponse> {
        let opts = options.cloned().unwrap_or_default();

        let use_json_mode = opts
            .response_format
            .as_ref()
            .map(|f| f == "json_object" || f == "json")
            .unwrap_or(false);

        // Normalise tools / tool_choice (only send tool_choice when tools present).
        let (api_tools, api_tool_choice) = match tools {
            Some((defs, choice)) if !defs.is_empty() => {
                let tc = choice.as_ref().map(Self::tool_choice_to_json);
                (Some(defs.to_vec()), tc)
            }
            _ => (None, None),
        };

        let request = NvidiaChatReq {
            model: &self.model,
            messages: Self::build_messages(messages),
            temperature: opts.temperature,
            top_p: opts.top_p,
            max_tokens: opts.max_tokens,
            stop: opts.stop.clone(),
            frequency_penalty: opts.frequency_penalty,
            presence_penalty: opts.presence_penalty,
            stream: false,
            tools: api_tools,
            tool_choice: api_tool_choice,
            response_format: if use_json_mode {
                Some(NvidiaRespFormat {
                    format_type: "json_object".to_string(),
                })
            } else {
                None
            },
            reasoning_effort: opts.reasoning_effort.clone(),
        };

        let chat_url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        debug!(
            provider = NVIDIA_PROVIDER_NAME,
            model = %self.model,
            url = %chat_url,
            "Sending NVIDIA non-streaming chat request"
        );

        let response = self
            .client
            .post(&chat_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(format!("NVIDIA chat request failed: {e}")))?;

        // Transparently resolve any HTTP 202 (async-queued) response via polling.
        let final_response = self.resolve_202(response).await?;
        self.parse_completion_response(final_response).await
    }

    /// Resolve a potential HTTP 202 response by polling the NVCF status endpoint.
    ///
    /// NVIDIA Cloud Functions returns **HTTP 202** when the inference request
    /// has been accepted but is still queued or processing.  The opaque request
    /// identifier is carried in the **`NVCF-REQID`** response header (not the body).
    ///
    /// Polling strategy (mirrors LangChain NVIDIA `_wait_async`):
    /// - Wait `NVIDIA_POLL_INTERVAL_MS` between each GET
    /// - Return immediately on any non-202 status (200 = done, 4xx/5xx = error)
    /// - Fail with `LlmError::ApiError` after `NVIDIA_MAX_POLL_ATTEMPTS`
    ///
    /// If the response is already 200 (or any other non-202 status), it is
    /// returned immediately without any polling.
    async fn resolve_202(&self, response: reqwest::Response) -> Result<reqwest::Response> {
        if response.status() != reqwest::StatusCode::ACCEPTED {
            return Ok(response);
        }

        // Extract the NVCF request identifier from the response header.
        let nvcf_reqid = response
            .headers()
            .get(NVIDIA_REQID_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .ok_or_else(|| {
                LlmError::ApiError(
                    "NVIDIA returned HTTP 202 (async-queued inference) but the required \
                     'NVCF-REQID' response header is missing — cannot poll for the result."
                        .to_string(),
                )
            })?;

        debug!(
            provider = NVIDIA_PROVIDER_NAME,
            nvcf_reqid = %nvcf_reqid,
            poll_interval_ms = NVIDIA_POLL_INTERVAL_MS,
            max_attempts = NVIDIA_MAX_POLL_ATTEMPTS,
            "NVIDIA returned HTTP 202 — starting async polling"
        );

        let poll_url = format!("{}/{}", NVIDIA_NVCF_STATUS_URL, nvcf_reqid);

        for attempt in 0..NVIDIA_MAX_POLL_ATTEMPTS {
            tokio::time::sleep(Duration::from_millis(NVIDIA_POLL_INTERVAL_MS)).await;

            let poll_resp = self
                .client
                .get(&poll_url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Accept", "application/json")
                .send()
                .await
                .map_err(|e| {
                    LlmError::NetworkError(format!(
                        "NVIDIA async poll #{attempt} failed for NVCF-REQID={nvcf_reqid}: {e}"
                    ))
                })?;

            if poll_resp.status() == reqwest::StatusCode::ACCEPTED {
                debug!(
                    attempt,
                    nvcf_reqid = %nvcf_reqid,
                    "NVIDIA async inference still pending"
                );
                continue;
            }

            // Any non-202 response (200 success, 4xx/5xx error) ends polling.
            debug!(
                attempt,
                nvcf_reqid = %nvcf_reqid,
                status = poll_resp.status().as_u16(),
                "NVIDIA async inference resolved"
            );
            return Ok(poll_resp);
        }

        Err(LlmError::ApiError(format!(
            "NVIDIA async inference timed out after {NVIDIA_MAX_POLL_ATTEMPTS} poll attempts \
             ({NVIDIA_POLL_INTERVAL_MS} ms interval, {:.0} s total) for NVCF-REQID='{nvcf_reqid}'",
            f64::from(NVIDIA_MAX_POLL_ATTEMPTS) * NVIDIA_POLL_INTERVAL_MS as f64 / 1000.0
        )))
    }

    /// Parse a final (non-202) HTTP response into an [`LLMResponse`].
    ///
    /// Handles:
    /// - HTTP 200: deserialise body as `NvidiaChatCompletion`
    /// - HTTP 4xx/5xx: extract structured error message or return raw body
    async fn parse_completion_response(&self, response: reqwest::Response) -> Result<LLMResponse> {
        let status = response.status();
        let body = response.text().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to read NVIDIA response body: {e}"))
        })?;

        if !status.is_success() {
            // Try to surface the structured error message from the body.
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&body) {
                if let Some(msg) = val.pointer("/error/message").and_then(|v| v.as_str()) {
                    return Err(LlmError::ApiError(format!(
                        "NVIDIA API error ({status}): {msg}"
                    )));
                }
            }
            return Err(LlmError::ApiError(format!(
                "NVIDIA API error ({status}): {}",
                &body[..1000.min(body.len())]
            )));
        }

        let completion: NvidiaChatCompletion = serde_json::from_str(&body).map_err(|e| {
            LlmError::ApiError(format!(
                "Failed to parse NVIDIA completion response: {e} | body preview: {}",
                &body[..500.min(body.len())]
            ))
        })?;

        let choice = completion
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::ApiError("NVIDIA response has no choices".to_string()))?;

        let content = choice.message.content.unwrap_or_default();
        let thinking = choice.message.reasoning_content.filter(|s| !s.is_empty());

        let (prompt_tokens, completion_tokens) = completion
            .usage
            .map(|u| (u.prompt_tokens, u.completion_tokens))
            .unwrap_or((0, 0));

        let model_name = completion.model.unwrap_or_else(|| self.model.clone());
        let finish_reason = choice.finish_reason.unwrap_or_else(|| "stop".to_string());

        // Convert tool calls to the canonical trait type.
        let tool_calls: Vec<crate::traits::ToolCall> = choice
            .message
            .tool_calls
            .into_iter()
            .map(|tc| crate::traits::ToolCall {
                id: tc.id,
                call_type: if tc.call_type.is_empty() {
                    "function".to_string()
                } else {
                    tc.call_type
                },
                function: crate::traits::FunctionCall {
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                },
                thought_signature: None,
            })
            .collect();

        let mut resp = LLMResponse::new(content, &model_name)
            .with_usage(prompt_tokens, completion_tokens)
            .with_finish_reason(finish_reason);

        if let Some(id) = completion.id {
            resp = resp.with_metadata("id", serde_json::Value::String(id));
        }

        if let Some(thinking_content) = thinking {
            resp = resp.with_thinking_content(thinking_content);
        }

        if !tool_calls.is_empty() {
            resp.tool_calls = tool_calls;
        }

        Ok(resp)
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /// Build the `ProviderConfig` passed to `OpenAICompatibleProvider`.
    fn build_provider_config(api_key: &str, model: &str, base_url: &str) -> ProviderConfig {
        let models: Vec<ModelCard> = NVIDIA_CHAT_MODELS
            .iter()
            .map(|(id, display, ctx, vision, thinking, _free)| ModelCard {
                name: id.to_string(),
                display_name: display.to_string(),
                model_type: ModelType::Llm,
                capabilities: ModelCapabilities {
                    context_length: *ctx,
                    supports_function_calling: true,
                    supports_json_mode: true,
                    supports_streaming: true,
                    supports_system_message: true,
                    supports_vision: *vision,
                    supports_thinking: *thinking,
                    ..Default::default()
                },
                ..Default::default()
            })
            .collect();

        ProviderConfig {
            name: NVIDIA_PROVIDER_NAME.to_string(),
            display_name: "NVIDIA NIM".to_string(),
            provider_type: ConfigProviderType::OpenAICompatible,
            api_key: Some(api_key.to_string()),
            api_key_env: Some("NVIDIA_API_KEY".to_string()),
            base_url: Some(base_url.to_string()),
            base_url_env: Some("NVIDIA_BASE_URL".to_string()),
            default_llm_model: Some(model.to_string()),
            default_embedding_model: None,
            models,
            headers: std::collections::HashMap::new(),
            enabled: true,
            timeout_seconds: NVIDIA_TIMEOUT_SECS,
            ..Default::default()
        }
    }
}

// ============================================================================
// LLMProvider Implementation (delegates to inner OpenAICompatibleProvider)
// ============================================================================

#[async_trait]
impl LLMProvider for NvidiaProvider {
    fn name(&self) -> &str {
        NVIDIA_PROVIDER_NAME
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_context_length(&self) -> usize {
        Self::context_length(&self.model)
    }

    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        self.complete_with_options(prompt, &CompletionOptions::default())
            .await
    }

    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<LLMResponse> {
        // Mirrors OpenAICompatibleProvider::complete_with_options — builds a
        // single-turn message list then routes through the 202-aware code path.
        let mut messages = Vec::new();
        if let Some(ref sys) = options.system_prompt {
            messages.push(ChatMessage::system(sys.clone()));
        }
        messages.push(ChatMessage::user(prompt));
        self.execute_non_streaming_chat(&messages, Some(options), None)
            .await
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        // Route through the 202-aware path instead of delegating to inner.
        self.execute_non_streaming_chat(messages, options, None)
            .await
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        // Non-streaming tool calls also route through the 202-aware path.
        self.execute_non_streaming_chat(messages, options, Some((tools, tool_choice)))
            .await
    }

    async fn chat_with_tools_stream(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        self.inner
            .chat_with_tools_stream(messages, tools, tool_choice, options)
            .await
    }

    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        self.inner.stream(prompt).await
    }

    fn supports_function_calling(&self) -> bool {
        self.inner.supports_function_calling()
    }

    fn supports_tool_streaming(&self) -> bool {
        self.inner.supports_tool_streaming()
    }
}

// ============================================================================
// EmbeddingProvider Implementation
//
// NVIDIA NIM does offer embedding models (e.g., nvidia/nv-embedqa-e5-v5),
// but they use a different endpoint path and input format.
// For the initial implementation, embeddings are intentionally unsupported
// via this provider. Users needing NVIDIA embeddings should use the
// OpenAICompatibleProvider directly with the appropriate embedding base URL.
// ============================================================================

#[async_trait]
impl EmbeddingProvider for NvidiaProvider {
    fn name(&self) -> &str {
        NVIDIA_PROVIDER_NAME
    }

    fn model(&self) -> &str {
        "none"
    }

    fn dimension(&self) -> usize {
        0
    }

    fn max_tokens(&self) -> usize {
        0
    }

    async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Err(LlmError::ConfigError(
            "NVIDIA NIM embeddings are not supported via NvidiaProvider in this release. \
             Use OpenAICompatibleProvider with base_url=https://integrate.api.nvidia.com/v1 \
             and an embedding model ID such as 'nvidia/nv-embedqa-e5-v5'."
                .to_string(),
        ))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Model catalog
    // -------------------------------------------------------------------------

    #[test]
    fn test_context_length_known_model() {
        assert_eq!(
            NvidiaProvider::context_length("meta/llama-3.1-8b-instruct"),
            131_072
        );
        assert_eq!(
            NvidiaProvider::context_length("nvidia/nemotron-3-nano-30b-a3b"),
            1_000_000
        );
        assert_eq!(
            NvidiaProvider::context_length("deepseek-ai/deepseek-v4-flash"),
            65_536
        );
    }

    #[test]
    fn test_context_length_unknown_model_fallback() {
        // Unknown models should fall back to 32K (conservative)
        assert_eq!(NvidiaProvider::context_length("unknown/model-xyz"), 32_768);
    }

    #[test]
    fn test_supports_vision() {
        assert!(NvidiaProvider::supports_vision(
            "meta/llama-4-maverick-17b-128e-instruct"
        ));
        assert!(NvidiaProvider::supports_vision(
            "meta/llama-3.2-11b-vision-instruct"
        ));
        assert!(!NvidiaProvider::supports_vision(
            "meta/llama-3.3-70b-instruct"
        ));
    }

    #[test]
    fn test_supports_thinking() {
        assert!(NvidiaProvider::supports_thinking(
            "nvidia/llama-3.3-nemotron-super-49b-v1"
        ));
        assert!(NvidiaProvider::supports_thinking(
            "deepseek-ai/deepseek-v4-flash"
        ));
        assert!(!NvidiaProvider::supports_thinking(
            "meta/llama-3.3-70b-instruct"
        ));
    }

    #[test]
    fn test_is_free_model() {
        assert!(NvidiaProvider::is_free_model("meta/llama-3.1-8b-instruct"));
        assert!(NvidiaProvider::is_free_model(
            "nvidia/llama-3.3-nemotron-super-49b-v1"
        ));
        assert!(!NvidiaProvider::is_free_model(
            "deepseek-ai/deepseek-v4-pro"
        ));
        assert!(!NvidiaProvider::is_free_model(
            "nvidia/llama-3.1-nemotron-ultra-253b-v1"
        ));
    }

    #[test]
    fn test_free_models_list() {
        let free = NvidiaProvider::free_models();
        // All known free models should be present
        assert!(!free.is_empty());
        let ids: Vec<&str> = free.iter().map(|(id, _, _)| *id).collect();
        assert!(ids.contains(&"meta/llama-3.1-8b-instruct"));
        assert!(ids.contains(&"qwen/qwq-32b"));
    }

    #[test]
    fn test_available_models_not_empty() {
        let models = NvidiaProvider::available_models();
        assert!(!models.is_empty());
        // Must have at least 20 models in the catalog
        assert!(models.len() >= 20);
    }

    #[test]
    fn test_catalog_integrity() {
        // Every model in NVIDIA_FREE_MODELS must exist in NVIDIA_CHAT_MODELS
        // (they may differ only during transitions; this catches typos)
        for free_id in NVIDIA_FREE_MODELS {
            let found = NVIDIA_CHAT_MODELS
                .iter()
                .any(|(id, _, _, _, _, _)| id == free_id);
            assert!(
                found,
                "Free model '{}' is not in NVIDIA_CHAT_MODELS catalog — add it or remove it from NVIDIA_FREE_MODELS",
                free_id
            );
        }
    }

    #[test]
    fn test_catalog_no_duplicate_ids() {
        let mut ids: Vec<&str> = NVIDIA_CHAT_MODELS
            .iter()
            .map(|(id, _, _, _, _, _)| *id)
            .collect();
        let original_len = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(
            ids.len(),
            original_len,
            "NVIDIA_CHAT_MODELS contains duplicate model IDs"
        );
    }

    // -------------------------------------------------------------------------
    // Provider construction (no network)
    // -------------------------------------------------------------------------

    #[test]
    fn test_from_env_missing_key() {
        // Temporarily clear the key if set
        let saved = std::env::var("NVIDIA_API_KEY").ok();
        std::env::remove_var("NVIDIA_API_KEY");

        let result = NvidiaProvider::from_env();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("NVIDIA_API_KEY"),
            "Error should mention NVIDIA_API_KEY, got: {}",
            err
        );

        // Restore
        if let Some(key) = saved {
            std::env::set_var("NVIDIA_API_KEY", key);
        }
    }

    #[test]
    fn test_from_env_empty_key() {
        let saved = std::env::var("NVIDIA_API_KEY").ok();
        std::env::set_var("NVIDIA_API_KEY", "");

        let result = NvidiaProvider::from_env();
        assert!(result.is_err());

        if let Some(key) = saved {
            std::env::set_var("NVIDIA_API_KEY", key);
        } else {
            std::env::remove_var("NVIDIA_API_KEY");
        }
    }

    #[test]
    fn test_new_creates_provider() {
        let provider = NvidiaProvider::new(
            "nvapi-test-key".to_string(),
            "meta/llama-3.3-70b-instruct".to_string(),
            None,
        );
        assert!(provider.is_ok());
        let p = provider.unwrap();
        assert_eq!(LLMProvider::name(&p), "nvidia");
        assert_eq!(LLMProvider::model(&p), "meta/llama-3.3-70b-instruct");
    }

    #[test]
    fn test_with_model_changes_model() {
        let provider = NvidiaProvider::new(
            "nvapi-test-key".to_string(),
            NVIDIA_DEFAULT_MODEL.to_string(),
            None,
        )
        .unwrap();
        let provider2 = provider.with_model("deepseek-ai/deepseek-v4-flash");
        assert_eq!(
            LLMProvider::model(&provider2),
            "deepseek-ai/deepseek-v4-flash"
        );
    }

    #[test]
    fn test_max_context_length() {
        let provider = NvidiaProvider::new(
            "nvapi-test-key".to_string(),
            "nvidia/nemotron-3-nano-30b-a3b".to_string(),
            None,
        )
        .unwrap();
        assert_eq!(provider.max_context_length(), 1_000_000);
    }

    #[test]
    fn test_provider_name() {
        let provider = NvidiaProvider::new(
            "nvapi-test-key".to_string(),
            NVIDIA_DEFAULT_MODEL.to_string(),
            None,
        )
        .unwrap();
        assert_eq!(LLMProvider::name(&provider), "nvidia");
    }

    #[test]
    fn test_embed_returns_error() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let provider = NvidiaProvider::new(
                "nvapi-test-key".to_string(),
                NVIDIA_DEFAULT_MODEL.to_string(),
                None,
            )
            .unwrap();
            let result = provider.embed(&["hello world".to_string()]).await;
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(err.contains("embeddings are not supported"), "Got: {}", err);
        });
    }

    // -------------------------------------------------------------------------
    // HTTP 202 async-polling constants and helpers
    // -------------------------------------------------------------------------

    #[test]
    fn test_polling_constants_are_sane() {
        use std::hint::black_box;

        let poll_interval_ms = black_box(NVIDIA_POLL_INTERVAL_MS);
        let max_poll_attempts = black_box(NVIDIA_MAX_POLL_ATTEMPTS);
        let poll_url = black_box(NVIDIA_NVCF_STATUS_URL);
        let reqid_header = black_box(NVIDIA_REQID_HEADER);

        // Interval must be positive and max_attempts must give at least 60 seconds
        assert!(poll_interval_ms > 0, "Poll interval must be > 0 ms");
        let total_ms = u64::from(max_poll_attempts) * poll_interval_ms;
        assert!(
            total_ms >= 60_000,
            "Total polling window {total_ms} ms is less than 60 s — increase limits"
        );
        assert!(!poll_url.is_empty(), "NVCF polling URL must not be empty");
        assert!(
            !reqid_header.is_empty(),
            "NVCF-REQID header name must not be empty"
        );
    }

    #[test]
    fn test_build_messages_plain_text() {
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("Hello, world!"),
        ];
        let reqs = NvidiaProvider::build_messages(&messages);
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0].role, "system");
        assert_eq!(
            reqs[0].content,
            serde_json::json!("You are a helpful assistant.")
        );
        assert_eq!(reqs[1].role, "user");
        assert_eq!(reqs[1].content, serde_json::json!("Hello, world!"));
        assert!(reqs[0].tool_calls.is_none());
        assert!(reqs[0].tool_call_id.is_none());
    }

    #[test]
    fn test_build_messages_tool_role() {
        use crate::traits::ChatRole;
        let mut tool_msg = ChatMessage::user("tool result here");
        tool_msg.role = ChatRole::Tool;
        tool_msg.tool_call_id = Some("call_abc".to_string());
        let reqs = NvidiaProvider::build_messages(&[tool_msg]);
        assert_eq!(reqs[0].role, "tool");
        assert_eq!(reqs[0].tool_call_id.as_deref(), Some("call_abc"));
    }

    #[test]
    fn test_build_messages_with_images() {
        use crate::traits::ImageData;
        let mut msg = ChatMessage::user("What is in this image?");
        msg.images = Some(vec![ImageData::new("aGVsbG8=", "image/png")]);
        let reqs = NvidiaProvider::build_messages(&[msg]);
        // Content should be an array (multipart)
        assert!(
            reqs[0].content.is_array(),
            "Expected multipart array, got: {:?}",
            reqs[0].content
        );
        let parts = reqs[0].content.as_array().unwrap();
        assert_eq!(parts.len(), 2, "Expected text + image_url part");
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image_url");
        let url = parts[1]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/png;base64,"), "URL: {url}");
    }

    #[test]
    fn test_tool_choice_to_json_auto() {
        let choice = ToolChoice::auto();
        let val = NvidiaProvider::tool_choice_to_json(&choice);
        assert_eq!(val, serde_json::json!("auto"));
    }

    #[test]
    fn test_tool_choice_to_json_none() {
        let choice = ToolChoice::none();
        let val = NvidiaProvider::tool_choice_to_json(&choice);
        assert_eq!(val, serde_json::json!("none"));
    }

    #[test]
    fn test_tool_choice_to_json_required() {
        let choice = ToolChoice::required();
        let val = NvidiaProvider::tool_choice_to_json(&choice);
        assert_eq!(val, serde_json::json!("required"));
    }

    #[test]
    fn test_tool_choice_to_json_function() {
        let choice = ToolChoice::function("get_weather");
        let val = NvidiaProvider::tool_choice_to_json(&choice);
        assert_eq!(val["type"], "function");
        assert_eq!(val["function"]["name"], "get_weather");
    }

    #[test]
    fn test_nvidia_chat_req_serialises_stream_false() {
        // Verify that the non-streaming request always sets stream=false
        let provider = NvidiaProvider::new(
            "nvapi-key".to_string(),
            "meta/llama-3.1-8b-instruct".to_string(),
            None,
        )
        .unwrap();
        let req = NvidiaChatReq {
            model: &provider.model,
            messages: vec![],
            temperature: Some(0.7),
            top_p: None,
            max_tokens: Some(512),
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            stream: false,
            tools: None,
            tool_choice: None,
            response_format: None,
            reasoning_effort: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["stream"], serde_json::json!(false));
        assert_eq!(json["model"], "meta/llama-3.1-8b-instruct");
        assert_eq!(json["max_tokens"], 512);
        // temperature is present
        assert!((json["temperature"].as_f64().unwrap() - 0.7).abs() < 0.001);
        // top_p should be absent (None → skip_serializing_if)
        assert!(json.get("top_p").is_none());
    }

    #[test]
    fn test_nvidia_chat_req_json_mode() {
        let req = NvidiaChatReq {
            model: "deepseek-ai/deepseek-v4-flash",
            messages: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            stream: false,
            tools: None,
            tool_choice: None,
            response_format: Some(NvidiaRespFormat {
                format_type: "json_object".to_string(),
            }),
            reasoning_effort: Some("high".to_string()),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["response_format"]["type"], "json_object");
        assert_eq!(json["reasoning_effort"], "high");
    }

    #[test]
    fn test_nvcf_reqid_header_name() {
        // The header name must match what NVIDIA sends in the 202 response.
        // LangChain reference: `response.headers.get("NVCF-REQID")`
        assert_eq!(NVIDIA_REQID_HEADER, "NVCF-REQID");
    }

    #[test]
    fn test_nvcf_polling_url_format() {
        // Verify the polling URL template contains the expected NVCF path
        assert!(
            NVIDIA_NVCF_STATUS_URL.contains("api.nvcf.nvidia.com"),
            "Polling URL should target api.nvcf.nvidia.com, got: {NVIDIA_NVCF_STATUS_URL}"
        );
        assert!(
            NVIDIA_NVCF_STATUS_URL.contains("pexec/status"),
            "Polling URL should contain 'pexec/status', got: {NVIDIA_NVCF_STATUS_URL}"
        );
        // Full constructed URL should look right
        let sample_id = "abc-123-def-456";
        let full_url = format!("{NVIDIA_NVCF_STATUS_URL}/{sample_id}");
        assert_eq!(
            full_url,
            format!("https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{sample_id}")
        );
    }
}
