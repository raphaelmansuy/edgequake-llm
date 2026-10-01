//! Provider prompt / KV-prefix cache policy (August 2026 SOTA).
//!
//! Two layers exist in EdgeQuake:
//! - **Response cache** (SPEC-103): skip the LLM call on an exact hit.
//! - **Provider KV / prompt cache** (this module): still generate, but reuse
//!   prefill KV for a stable prefix (cheaper input tokens + lower TTFT).
//!
//! # Per-provider policy
//!
//! | Provider | What EdgeQuake sends | Prefix rule |
//! |----------|----------------------|-------------|
//! | OpenAIProvider (Native) / Azure | `prompt_cache_key` + GPT-5.6 `prompt_cache_options`/`prompt_cache_breakpoint` | Constructor chooses the Chat Completions contract; 400 `error.param` learning; never host/model-name parsing |
//! | OpenAIProvider::compatible / Mistral / NVIDIA | `prompt_cache_key` | 64-token blocks on Mistral; cached input billed at 10% |
//! | Anthropic | `cache_control` on system (TTL `5m`/`1h`) | Max 4 breakpoints; pin TTL explicitly |
//! | OpenRouter | `cache_control` + `prompt_cache_key` + `session_id` | Router translates markers; `session_id` sticky-routes before the first hit |
//! | Bedrock Converse | `cachePoint` after system blocks | Converse protocol (not model-name sniffing); TTL `5m`/`1h` |
//! | Gemini 2.5+ | layout + optional `cachedContents` | Implicit min ~2k–4k; explicit cache is a different API |
//! | vLLM / Ollama | layout only | Engine prefix cache when tokens match; do not send GPT-5.6 fields |
//!
//! # OODA-17: Anthropic Prompt Caching
//!
//! This module also provides utilities for Anthropic `cache_control` markers
//! to reduce costs by 85-90% on repeated context.
//!
//! # Overview
//!
//! Prompt caching allows marking parts of the conversation as cacheable:
//! - System prompts (rarely change)
//! - Large file contexts (repeated across calls)
//! - Recent conversation history (conversation context)
//!
//! Cached tokens are served at 90% discount for subsequent requests.
//!
//! # Usage
//!
//! ```rust
//! use edgequake_llm::cache_prompt::{CachePromptConfig, apply_cache_control};
//! use edgequake_llm::traits::ChatMessage;
//!
//! let config = CachePromptConfig::default();
//! let mut messages = vec![
//!     ChatMessage::system("You are a helpful assistant"),
//!     ChatMessage::user("Large file content here..."),
//! ];
//!
//! apply_cache_control(&mut messages, &config);
//! // Now messages have cache_control set where appropriate
//! ```
//!
//! # See Also
//!
//! - [Anthropic Prompt Caching](https://docs.anthropic.com/claude/docs/prompt-caching)
//! - Aider reference: `base_coder.py`, `sendchat.py`

use crate::traits::{CacheControl, ChatMessage, ChatRole};
use serde::{Deserialize, Serialize};

/// Configuration for automatic prompt cache control marking.
///
/// # Fields
///
/// - `enabled`: Whether to apply cache control (default: true)
/// - `min_content_length`: Minimum message length to auto-cache (default: 1000)
/// - `cache_system_prompt`: Whether to cache system prompts (default: true)
/// - `cache_last_n_messages`: Number of recent user messages to cache (default: 3)
///
/// # Example
///
/// ```rust
/// use edgequake_llm::cache_prompt::CachePromptConfig;
///
/// // Use defaults
/// let config = CachePromptConfig::default();
///
/// // Custom configuration
/// let config = CachePromptConfig {
///     enabled: true,
///     min_content_length: 500,
///     cache_system_prompt: true,
///     cache_last_n_messages: 5,
///     cache_ttl: Some("1h".into()),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePromptConfig {
    /// Whether cache control marking is enabled.
    pub enabled: bool,

    /// Minimum content length (in characters) to auto-cache user messages.
    ///
    /// Messages shorter than this threshold are not automatically cached
    /// unless they are system prompts or in the last N messages.
    pub min_content_length: usize,

    /// Whether to cache the system prompt.
    ///
    /// System prompts rarely change and are excellent cache candidates.
    pub cache_system_prompt: bool,

    /// Number of recent user messages to cache.
    ///
    /// Caching recent messages helps with conversation context retention.
    pub cache_last_n_messages: usize,

    /// Anthropic cache TTL tier: `"5m"` (default) or `"1h"`.
    ///
    /// The 1h tier requires the `extended-cache-ttl-2025-04-11` beta header and
    /// costs more on cache writes but amortizes across long pauses between turns.
    #[serde(default)]
    pub cache_ttl: Option<String>,
}

impl Default for CachePromptConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_content_length: 1000,
            cache_system_prompt: true,
            cache_last_n_messages: 3,
            cache_ttl: None,
        }
    }
}

fn cache_marker_for_config(config: &CachePromptConfig) -> CacheControl {
    match config.cache_ttl.as_deref() {
        Some("1h") => CacheControl::ephemeral_ttl("1h"),
        Some("5m") => CacheControl::ephemeral_ttl("5m"),
        _ => CacheControl::ephemeral(),
    }
}

impl CachePromptConfig {
    /// Create a config with caching disabled.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create a config that only caches system prompts.
    pub fn system_only() -> Self {
        Self {
            enabled: true,
            min_content_length: usize::MAX,
            cache_system_prompt: true,
            cache_last_n_messages: 0,
            ..Default::default()
        }
    }

    /// Create an aggressive caching config.
    ///
    /// Caches more content for maximum cost reduction.
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            min_content_length: 100,
            cache_system_prompt: true,
            cache_last_n_messages: 10,
            ..Default::default()
        }
    }
}

/// Statistics about cache usage from an API response.
///
/// Anthropic returns cache statistics in the usage field of responses:
/// - `cache_read_input_tokens`: Tokens served from cache (90% cheaper)
/// - `cache_creation_input_tokens`: Tokens used to create the cache
///
/// # Cost Model
///
/// - Normal input tokens: $0.003 per 1K tokens
/// - Cached input tokens: $0.0003 per 1K tokens (90% discount)
/// - Cache creation has a small overhead but pays off after 2-3 uses
///
/// # Example
///
/// ```rust
/// use edgequake_llm::cache_prompt::CacheStats;
///
/// let stats = CacheStats {
///     input_tokens: 10000,
///     output_tokens: 1000,
///     cache_read_tokens: 8000,
///     cache_creation_tokens: 0,
/// };
///
/// println!("Cache hit rate: {:.0}%", stats.cache_hit_rate() * 100.0);
/// println!("Estimated savings: ${:.4}", stats.savings());
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total input tokens in the request.
    pub input_tokens: u64,

    /// Total output tokens in the response.
    pub output_tokens: u64,

    /// Input tokens served from cache.
    pub cache_read_tokens: u64,

    /// Tokens used to create new cache entries.
    pub cache_creation_tokens: u64,
}

impl CacheStats {
    /// Create new cache stats.
    pub fn new(
        input_tokens: u64,
        output_tokens: u64,
        cache_read_tokens: u64,
        cache_creation_tokens: u64,
    ) -> Self {
        Self {
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_creation_tokens,
        }
    }

    /// Calculate the cache hit rate as a fraction (0.0 to 1.0).
    ///
    /// Returns 0.0 if there are no input tokens.
    pub fn cache_hit_rate(&self) -> f64 {
        if self.input_tokens == 0 {
            0.0
        } else {
            self.cache_read_tokens as f64 / self.input_tokens as f64
        }
    }

    /// Estimate cost savings in dollars.
    ///
    /// Based on Anthropic Claude pricing (as of 2024):
    /// - Normal input: $0.003 per 1K tokens
    /// - Cached input: $0.0003 per 1K tokens
    ///
    /// Returns the difference between what would have been paid
    /// without caching vs with caching.
    pub fn savings(&self) -> f64 {
        const NORMAL_COST_PER_1K: f64 = 0.003;
        const CACHE_COST_PER_1K: f64 = 0.0003;

        // Cost without caching
        let normal_cost = self.input_tokens as f64 * NORMAL_COST_PER_1K / 1000.0;

        // Cost with caching
        let uncached_tokens = self.input_tokens.saturating_sub(self.cache_read_tokens);
        let cache_cost = self.cache_read_tokens as f64 * CACHE_COST_PER_1K / 1000.0
            + uncached_tokens as f64 * NORMAL_COST_PER_1K / 1000.0;

        normal_cost - cache_cost
    }

    /// Calculate the cost per call with current cache stats.
    pub fn cost_per_call(&self) -> f64 {
        const NORMAL_COST_PER_1K: f64 = 0.003;
        const CACHE_COST_PER_1K: f64 = 0.0003;
        const OUTPUT_COST_PER_1K: f64 = 0.015; // Claude output tokens

        let uncached_tokens = self.input_tokens.saturating_sub(self.cache_read_tokens);

        self.cache_read_tokens as f64 * CACHE_COST_PER_1K / 1000.0
            + uncached_tokens as f64 * NORMAL_COST_PER_1K / 1000.0
            + self.output_tokens as f64 * OUTPUT_COST_PER_1K / 1000.0
    }

    /// Check if caching was effective (hit rate > 50%).
    pub fn is_effective(&self) -> bool {
        self.cache_hit_rate() > 0.5
    }

    /// Merge stats from another request.
    pub fn merge(&mut self, other: &CacheStats) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_read_tokens += other.cache_read_tokens;
        self.cache_creation_tokens += other.cache_creation_tokens;
    }
}

/// Apply cache control to messages based on configuration.
///
/// This function marks messages with `cache_control` hints that providers
/// like Anthropic Claude can use to cache prompt prefixes.
///
/// # Cache Marking Strategy
///
/// 1. **System prompts**: Always cached (if `cache_system_prompt` is true)
/// 2. **Large user messages**: Cached if length > `min_content_length`
/// 3. **Recent user messages**: Last N user messages are cached
///
/// # Arguments
///
/// * `messages` - Mutable slice of messages to apply cache control to
/// * `config` - Configuration controlling which messages to cache
///
/// # Example
///
/// ```rust
/// use edgequake_llm::cache_prompt::{CachePromptConfig, apply_cache_control};
/// use edgequake_llm::traits::ChatMessage;
///
/// let config = CachePromptConfig::default();
/// let mut messages = vec![
///     ChatMessage::system("You are a helpful assistant"),
///     ChatMessage::user("Please analyze this file: ..."),
/// ];
///
/// apply_cache_control(&mut messages, &config);
///
/// assert!(messages[0].cache_control.is_some()); // System prompt cached
/// ```
pub fn apply_cache_control(messages: &mut [ChatMessage], config: &CachePromptConfig) {
    if !config.enabled {
        return;
    }

    // Track user message indices for last-N caching
    let user_indices: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| matches!(m.role, ChatRole::User))
        .map(|(i, _)| i)
        .collect();

    // Determine which indices should be cached as "last N"
    let last_n_start = user_indices
        .len()
        .saturating_sub(config.cache_last_n_messages);
    let last_n_indices: std::collections::HashSet<usize> =
        user_indices.into_iter().skip(last_n_start).collect();

    for (i, msg) in messages.iter_mut().enumerate() {
        let should_cache = match msg.role {
            ChatRole::System => config.cache_system_prompt,
            ChatRole::User => {
                // Cache if large content OR in last N user messages
                msg.content.len() >= config.min_content_length || last_n_indices.contains(&i)
            }
            _ => false, // Don't cache assistant/tool messages
        };

        if should_cache && msg.cache_control.is_none() {
            msg.cache_control = Some(cache_marker_for_config(config));
        }
    }
}

/// Parse cache statistics from an Anthropic API response.
///
/// Anthropic includes cache stats in the `usage` field:
/// ```json
/// {
///   "usage": {
///     "input_tokens": 10000,
///     "output_tokens": 500,
///     "cache_read_input_tokens": 8000,
///     "cache_creation_input_tokens": 0
///   }
/// }
/// ```
pub fn parse_cache_stats(usage: &serde_json::Value) -> CacheStats {
    CacheStats {
        input_tokens: usage["input_tokens"].as_u64().unwrap_or(0),
        output_tokens: usage["output_tokens"].as_u64().unwrap_or(0),
        cache_read_tokens: usage["cache_read_input_tokens"].as_u64().unwrap_or(0),
        cache_creation_tokens: usage["cache_creation_input_tokens"].as_u64().unwrap_or(0),
    }
}

/// How a provider reuses prompt prefixes / KV cache.
///
/// Selected by the **provider implementation**, never by parsing `provider.name()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptCachePolicy {
    /// Chat Completions `prompt_cache_key` (OpenAI, Azure, Mistral, NVIDIA, vLLM).
    OpenAiCompatible,
    /// Anthropic Claude: explicit `cache_control` breakpoints + TTL.
    Anthropic,
    /// OpenRouter: Anthropic-style `cache_control` plus `prompt_cache_key`/`session_id`.
    ///
    /// August 2026: OpenRouter translates `cache_control` ↔ `prompt_cache_breakpoint`
    /// per upstream. Sticky routing uses `session_id` (falls back to `prompt_cache_key`).
    OpenRouter,
    /// Gemini 2.5+: implicit prefix cache and/or `cachedContents`.
    Gemini,
    /// Ollama / llama.cpp: engine-side prefix reuse only when tokens match.
    LocalPrefix,
}

/// Which Chat Completions **contract** we are calling.
///
/// GPT-5.6 `prompt_cache_options` / `prompt_cache_breakpoint` are defined on the
/// official OpenAI and Azure OpenAI Chat Completions APIs. `OpenAIProvider` is
/// Native unless constructed with [`crate::OpenAIProvider::compatible`]. Compatible
/// servers (Mistral, Ollama, vLLM, …) must not receive those fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiChatApi {
    /// Default `api.openai.com` Chat Completions contract.
    Native,
    /// Azure OpenAI Chat Completions (same explicit-cache fields as native).
    Azure,
    /// OpenAI-compatible subset. `prompt_cache_key` only.
    Compatible,
}

impl OpenAiChatApi {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Native => "openai",
            Self::Azure => "azure",
            Self::Compatible => "openai-compatible",
        }
    }

    pub fn allows_explicit_breakpoints(self) -> bool {
        matches!(self, Self::Native | Self::Azure)
    }
}

/// Product flag `EDGEQUAKE_PROMPT_CACHE` (alias `EDGEQUAKE_PROVIDER_PROMPT_CACHE`).
///
/// Default **on**. Falsey: `0` / `false` / `off` / `no`.
pub fn env_flag_enabled_default_on(raw: Option<&str>) -> bool {
    !matches!(
        raw.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("0") | Some("false") | Some("off") | Some("no")
    )
}

/// Whether provider KV / prompt-cache policy is active (product default on).
pub fn provider_prompt_cache_enabled() -> bool {
    let raw = std::env::var("EDGEQUAKE_PROMPT_CACHE")
        .ok()
        .or_else(|| std::env::var("EDGEQUAKE_PROVIDER_PROMPT_CACHE").ok());
    env_flag_enabled_default_on(raw.as_deref())
}

/// Anthropic breakpoint TTL: `EDGEQUAKE_PROMPT_CACHE_TTL` = `5m` (default) or `1h`.
pub fn prompt_cache_ttl_from_env() -> Option<String> {
    match std::env::var("EDGEQUAKE_PROMPT_CACHE_TTL")
        .ok()
        .as_deref()
        .map(str::trim)
        .unwrap_or("5m")
    {
        "1h" | "1H" | "3600" => Some("1h".into()),
        _ => Some("5m".into()),
    }
}

fn sanitize_cache_key_part(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | ':') {
            out.push(c.to_ascii_lowercase());
        } else {
            out.push('-');
        }
    }
    out
}

/// Stable, non-secret prompt-cache routing key: `eq:{role}:{provider}:{model}`.
pub fn prompt_cache_key(role: &str, provider: &str, model: &str) -> String {
    let role = sanitize_cache_key_part(role.trim());
    let provider = sanitize_cache_key_part(provider.trim());
    let model = sanitize_cache_key_part(model.trim());
    let mut key = if role.is_empty() {
        format!("eq:{provider}:{model}")
    } else {
        format!("eq:{role}:{provider}:{model}")
    };
    if key.len() > 200 {
        key.truncate(200);
    }
    key
}

/// Apply provider KV-cache policy to a chat request (default-on).
///
/// `policy` is chosen by the provider crate, not inferred from a name string.
pub fn prepare_chat(
    messages: &[ChatMessage],
    options: Option<&crate::traits::CompletionOptions>,
    policy: PromptCachePolicy,
    key_provider: &str,
    model: &str,
) -> (Vec<ChatMessage>, crate::traits::CompletionOptions) {
    let mut options = options.cloned().unwrap_or_default();
    let mut messages = messages.to_vec();

    if !provider_prompt_cache_enabled() {
        options.prompt_cache_key = None;
        return (messages, options);
    }

    let wants_cache_control = matches!(
        policy,
        PromptCachePolicy::Anthropic | PromptCachePolicy::OpenRouter
    );
    let wants_key = matches!(
        policy,
        PromptCachePolicy::OpenAiCompatible | PromptCachePolicy::OpenRouter
    );

    if wants_cache_control {
        let mut cfg = CachePromptConfig::system_only();
        cfg.cache_ttl = prompt_cache_ttl_from_env();
        apply_cache_control(&mut messages, &cfg);
    }

    if wants_key && options.prompt_cache_key.is_none() {
        options.prompt_cache_key = Some(prompt_cache_key("", key_provider, model));
    }

    (messages, options)
}

fn explicit_breakpoint_memory() -> &'static std::sync::Mutex<std::collections::HashMap<String, bool>>
{
    static MAP: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, bool>>> =
        std::sync::OnceLock::new();
    MAP.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn explicit_breakpoint_memory_key(api: OpenAiChatApi, model: &str) -> String {
    format!("{}:{model}", api.as_str())
}

/// Process-lifetime memory of whether this (API, model) accepts GPT-5.6 explicit fields.
///
/// `Some(false)` is learned from the API (`prompt_cache_options` 400). Unknown → try.
pub fn remembered_explicit_breakpoint_support(api: OpenAiChatApi, model: &str) -> Option<bool> {
    let key = explicit_breakpoint_memory_key(api, model);
    explicit_breakpoint_memory()
        .lock()
        .ok()
        .and_then(|g| g.get(&key).copied())
}

/// Remember that this (API, model) rejected GPT-5.6 explicit fields.
///
/// Only `supported = false` is stored. A 200 does **not** prove the model is
/// GPT-5.6 — it only means the request was accepted. We keep retrying explicit
/// fields until a structured `error.param` 400 says otherwise.
pub fn remember_explicit_breakpoint_support(api: OpenAiChatApi, model: &str, supported: bool) {
    if supported {
        return;
    }
    let key = explicit_breakpoint_memory_key(api, model);
    if let Ok(mut g) = explicit_breakpoint_memory().lock() {
        g.insert(key, false);
    }
}

/// If `err` is an unsupported-explicit-cache 400, remember it and return true.
pub fn note_explicit_cache_rejection(
    api: OpenAiChatApi,
    model: &str,
    err: &crate::error::LlmError,
) -> bool {
    if is_unsupported_explicit_cache_error(err) {
        remember_explicit_breakpoint_support(api, model, false);
        true
    } else {
        false
    }
}

#[cfg(test)]
pub fn reset_explicit_breakpoint_memory_for_tests() {
    if let Ok(mut g) = explicit_breakpoint_memory().lock() {
        g.clear();
    }
}

/// Attempt GPT-5.6 explicit breakpoints unless this API/model already 400'd them.
pub fn should_attempt_explicit_breakpoints(api: OpenAiChatApi, model: &str) -> bool {
    provider_prompt_cache_enabled()
        && api.allows_explicit_breakpoints()
        && !matches!(
            remembered_explicit_breakpoint_support(api, model),
            Some(false)
        )
}

/// Build a Chat Completions body with explicit breakpoints, or `None` to use the typed request.
pub fn explicit_cache_request_body(
    request: &impl serde::Serialize,
    api: OpenAiChatApi,
    model: &str,
    stream: bool,
) -> std::result::Result<Option<serde_json::Value>, serde_json::Error> {
    if !should_attempt_explicit_breakpoints(api, model) {
        return Ok(None);
    }
    let mut body = serde_json::to_value(request)?;
    if stream {
        body["stream"] = serde_json::json!(true);
    }
    if apply_openai_explicit_prompt_cache(&mut body) {
        Ok(Some(body))
    } else {
        Ok(None)
    }
}

const EXPLICIT_CACHE_PARAMS: &[&str] = &["prompt_cache_options", "prompt_cache_breakpoint"];

fn json_error_param(value: &serde_json::Value) -> Option<&str> {
    value
        .pointer("/error/param")
        .and_then(|v| v.as_str())
        .or_else(|| value.get("param").and_then(|v| v.as_str()))
}

fn message_names_explicit_cache_param(msg: &str) -> bool {
    // async-openai ApiError Display: "... (param: prompt_cache_options) (code: ...)"
    for param in EXPLICIT_CACHE_PARAMS {
        let needle = format!("(param: {param})");
        if msg.contains(&needle) {
            return true;
        }
        let quoted = format!("\"param\":\"{param}\"");
        let quoted_sp = format!("\"param\": \"{param}\"");
        if msg.contains(&quoted) || msg.contains(&quoted_sp) {
            return true;
        }
    }
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(msg) {
        if let Some(p) = json_error_param(&v) {
            return EXPLICIT_CACHE_PARAMS.contains(&p);
        }
    }
    if let Some(idx) = msg.find('{') {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&msg[idx..]) {
            if let Some(p) = json_error_param(&v) {
                return EXPLICIT_CACHE_PARAMS.contains(&p);
            }
        }
    }
    false
}

/// True when the provider rejected GPT-5.6 explicit-cache fields.
///
/// Matches structured `error.param` / SDK `(param: …)` — not free-text mentions.
pub fn is_unsupported_explicit_cache_error(err: &crate::error::LlmError) -> bool {
    let msg = match err {
        crate::error::LlmError::InvalidRequest(m) | crate::error::LlmError::ApiError(m) => {
            m.as_str()
        }
        _ => return false,
    };
    message_names_explicit_cache_param(msg)
}

fn mark_text_part_breakpoint(part: &mut serde_json::Value) {
    if let Some(obj) = part.as_object_mut() {
        obj.insert(
            "prompt_cache_breakpoint".into(),
            serde_json::json!({ "mode": "explicit" }),
        );
    }
}

/// Attach an explicit breakpoint to the last leading `system`/`developer` message.
///
/// GPT-5.6 implicit mode breakpoints the *latest user message*, so extract-style
/// "stable system + changing chunk" never partial-matches without this marker.
pub fn apply_openai_explicit_prompt_cache(body: &mut serde_json::Value) -> bool {
    let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut()) else {
        return false;
    };

    let mut last_sys = None;
    for (i, msg) in messages.iter().enumerate() {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role == "system" || role == "developer" {
            last_sys = Some(i);
        } else {
            break;
        }
    }
    let Some(idx) = last_sys else {
        return false;
    };

    let Some(content) = messages[idx].get_mut("content") else {
        return false;
    };

    match content {
        serde_json::Value::String(text) => {
            let text = text.clone();
            *content = serde_json::json!([{
                "type": "text",
                "text": text,
                "prompt_cache_breakpoint": { "mode": "explicit" }
            }]);
        }
        serde_json::Value::Array(parts) => {
            let Some(part) = parts.iter_mut().rev().find(|p| {
                matches!(
                    p.get("type").and_then(|t| t.as_str()),
                    Some("text") | Some("input_text")
                ) || p.get("text").is_some()
            }) else {
                return false;
            };
            mark_text_part_breakpoint(part);
        }
        _ => return false,
    }

    body["prompt_cache_options"] = serde_json::json!({
        "mode": "explicit",
        "ttl": "30m"
    });
    true
}

/// GPT-5.6 Chat Completions reports writes under `usage.prompt_tokens_details.cache_write_tokens`.
pub fn cache_write_tokens_from_chat_usage(usage: &serde_json::Value) -> Option<usize> {
    usage
        .pointer("/prompt_tokens_details/cache_write_tokens")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CachePromptConfig::default();
        assert!(config.enabled);
        assert_eq!(config.min_content_length, 1000);
        assert!(config.cache_system_prompt);
        assert_eq!(config.cache_last_n_messages, 3);
    }

    #[test]
    fn test_disabled_config() {
        let config = CachePromptConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_system_only_config() {
        let config = CachePromptConfig::system_only();
        assert!(config.enabled);
        assert!(config.cache_system_prompt);
        assert_eq!(config.cache_last_n_messages, 0);
        assert_eq!(config.min_content_length, usize::MAX);
    }

    #[test]
    fn test_aggressive_config() {
        let config = CachePromptConfig::aggressive();
        assert!(config.enabled);
        assert_eq!(config.min_content_length, 100);
        assert_eq!(config.cache_last_n_messages, 10);
    }

    #[test]
    fn test_cache_control_disabled() {
        let config = CachePromptConfig::disabled();
        let mut messages = vec![
            ChatMessage::system("System prompt"),
            ChatMessage::user("User message"),
        ];

        apply_cache_control(&mut messages, &config);

        assert!(messages[0].cache_control.is_none());
        assert!(messages[1].cache_control.is_none());
    }

    #[test]
    fn test_cache_system_prompt() {
        let config = CachePromptConfig::default();
        let mut messages = vec![
            ChatMessage::system("You are a helpful assistant"),
            ChatMessage::user("Hello"),
        ];

        apply_cache_control(&mut messages, &config);

        assert!(messages[0].cache_control.is_some());
        assert_eq!(
            messages[0].cache_control.as_ref().unwrap().cache_type,
            "ephemeral"
        );
    }

    #[test]
    fn test_cache_ttl_1h_on_system_prompt() {
        let config = CachePromptConfig {
            cache_ttl: Some("1h".to_string()),
            ..Default::default()
        };
        let mut messages = vec![ChatMessage::system("Stable prefix")];
        apply_cache_control(&mut messages, &config);
        let cc = messages[0].cache_control.as_ref().unwrap();
        assert_eq!(cc.ttl.as_deref(), Some("1h"));
    }

    #[test]
    fn test_cache_large_messages() {
        let config = CachePromptConfig {
            min_content_length: 100,
            cache_last_n_messages: 0,
            ..Default::default()
        };

        let large_content = "x".repeat(150);
        let small_content = "y".repeat(50);

        let mut messages = vec![
            ChatMessage::system("System"),
            ChatMessage::user(&large_content),
            ChatMessage::user(&small_content),
        ];

        apply_cache_control(&mut messages, &config);

        // System should be cached
        assert!(messages[0].cache_control.is_some());
        // Large message should be cached
        assert!(messages[1].cache_control.is_some());
        // Small message should NOT be cached (last_n is 0)
        assert!(messages[2].cache_control.is_none());
    }

    #[test]
    fn test_cache_last_n_messages() {
        let config = CachePromptConfig {
            min_content_length: usize::MAX, // Disable size-based caching
            cache_last_n_messages: 2,
            cache_system_prompt: false,
            ..Default::default()
        };

        let mut messages = vec![
            ChatMessage::system("System"),
            ChatMessage::user("First"),
            ChatMessage::assistant("Response"),
            ChatMessage::user("Second"),
            ChatMessage::assistant("Response"),
            ChatMessage::user("Third"),
            ChatMessage::user("Fourth"),
        ];

        apply_cache_control(&mut messages, &config);

        // System not cached (disabled)
        assert!(messages[0].cache_control.is_none());
        // First two user messages not cached
        assert!(messages[1].cache_control.is_none());
        assert!(messages[3].cache_control.is_none());
        // Last two user messages cached
        assert!(messages[5].cache_control.is_some()); // Third
        assert!(messages[6].cache_control.is_some()); // Fourth
    }

    #[test]
    fn test_preserves_existing_cache_control() {
        let config = CachePromptConfig::default();
        let mut messages = vec![ChatMessage::system("System")];

        // Pre-set cache control
        messages[0].cache_control = Some(CacheControl::ephemeral());

        apply_cache_control(&mut messages, &config);

        // Should still have cache control
        assert!(messages[0].cache_control.is_some());
    }

    #[test]
    fn test_cache_hit_rate_zero_tokens() {
        let stats = CacheStats::default();
        assert_eq!(stats.cache_hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_hit_rate_full_cache() {
        let stats = CacheStats {
            input_tokens: 10000,
            output_tokens: 500,
            cache_read_tokens: 10000,
            cache_creation_tokens: 0,
        };
        assert_eq!(stats.cache_hit_rate(), 1.0);
    }

    #[test]
    fn test_cache_hit_rate_partial() {
        let stats = CacheStats {
            input_tokens: 10000,
            output_tokens: 500,
            cache_read_tokens: 8000,
            cache_creation_tokens: 0,
        };
        assert_eq!(stats.cache_hit_rate(), 0.8);
    }

    #[test]
    fn test_cache_savings() {
        let stats = CacheStats {
            input_tokens: 10000,
            output_tokens: 500,
            cache_read_tokens: 8000,
            cache_creation_tokens: 0,
        };

        let savings = stats.savings();

        // 8000 tokens saved at 90% discount = 8000 * 0.0027 / 1000 = $0.0216
        assert!(savings > 0.02);
        assert!(savings < 0.03);
    }

    #[test]
    fn test_cache_savings_no_cache() {
        let stats = CacheStats {
            input_tokens: 10000,
            output_tokens: 500,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        };

        assert_eq!(stats.savings(), 0.0);
    }

    #[test]
    fn test_is_effective() {
        let effective = CacheStats {
            input_tokens: 10000,
            cache_read_tokens: 6000,
            ..Default::default()
        };
        assert!(effective.is_effective());

        let ineffective = CacheStats {
            input_tokens: 10000,
            cache_read_tokens: 4000,
            ..Default::default()
        };
        assert!(!ineffective.is_effective());
    }

    #[test]
    fn test_merge_stats() {
        let mut stats1 = CacheStats {
            input_tokens: 1000,
            output_tokens: 100,
            cache_read_tokens: 500,
            cache_creation_tokens: 200,
        };

        let stats2 = CacheStats {
            input_tokens: 2000,
            output_tokens: 200,
            cache_read_tokens: 1000,
            cache_creation_tokens: 100,
        };

        stats1.merge(&stats2);

        assert_eq!(stats1.input_tokens, 3000);
        assert_eq!(stats1.output_tokens, 300);
        assert_eq!(stats1.cache_read_tokens, 1500);
        assert_eq!(stats1.cache_creation_tokens, 300);
    }

    #[test]
    fn test_parse_cache_stats() {
        let usage = serde_json::json!({
            "input_tokens": 10000,
            "output_tokens": 500,
            "cache_read_input_tokens": 8000,
            "cache_creation_input_tokens": 100
        });

        let stats = parse_cache_stats(&usage);

        assert_eq!(stats.input_tokens, 10000);
        assert_eq!(stats.output_tokens, 500);
        assert_eq!(stats.cache_read_tokens, 8000);
        assert_eq!(stats.cache_creation_tokens, 100);
    }

    #[test]
    fn test_parse_cache_stats_missing_fields() {
        let usage = serde_json::json!({
            "input_tokens": 5000,
            "output_tokens": 200
        });

        let stats = parse_cache_stats(&usage);

        assert_eq!(stats.input_tokens, 5000);
        assert_eq!(stats.output_tokens, 200);
        assert_eq!(stats.cache_read_tokens, 0);
        assert_eq!(stats.cache_creation_tokens, 0);
    }

    #[test]
    fn test_cost_per_call() {
        let stats = CacheStats {
            input_tokens: 10000,
            output_tokens: 1000,
            cache_read_tokens: 8000,
            cache_creation_tokens: 0,
        };

        let cost = stats.cost_per_call();

        // 8000 cached at $0.0003/1K = $0.0024
        // 2000 normal at $0.003/1K = $0.006
        // 1000 output at $0.015/1K = $0.015
        // Total: $0.0234
        assert!(cost > 0.02);
        assert!(cost < 0.03);
    }

    #[test]
    fn test_cache_stats_serialization() {
        let stats = CacheStats {
            input_tokens: 1000,
            output_tokens: 100,
            cache_read_tokens: 800,
            cache_creation_tokens: 50,
        };

        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: CacheStats = serde_json::from_str(&json).unwrap();

        assert_eq!(stats.input_tokens, deserialized.input_tokens);
        assert_eq!(stats.output_tokens, deserialized.output_tokens);
        assert_eq!(stats.cache_read_tokens, deserialized.cache_read_tokens);
        assert_eq!(
            stats.cache_creation_tokens,
            deserialized.cache_creation_tokens
        );
    }

    #[test]
    fn test_cache_stats_new_constructor() {
        let stats = CacheStats::new(5000, 500, 3000, 200);
        assert_eq!(stats.input_tokens, 5000);
        assert_eq!(stats.output_tokens, 500);
        assert_eq!(stats.cache_read_tokens, 3000);
        assert_eq!(stats.cache_creation_tokens, 200);
    }

    #[test]
    fn test_apply_cache_control_empty_messages() {
        let config = CachePromptConfig::default();
        let mut messages: Vec<ChatMessage> = vec![];
        apply_cache_control(&mut messages, &config);
        assert!(messages.is_empty());
    }

    #[test]
    fn test_apply_cache_control_only_assistant_messages() {
        let config = CachePromptConfig::default();
        let mut messages = vec![
            ChatMessage::assistant("I will help you"),
            ChatMessage::assistant("Here is the answer"),
        ];
        apply_cache_control(&mut messages, &config);
        // Assistant messages should never be cached
        assert!(messages[0].cache_control.is_none());
        assert!(messages[1].cache_control.is_none());
    }

    #[test]
    fn test_parse_cache_stats_empty_json() {
        let usage = serde_json::json!({});
        let stats = parse_cache_stats(&usage);
        assert_eq!(stats.input_tokens, 0);
        assert_eq!(stats.output_tokens, 0);
        assert_eq!(stats.cache_read_tokens, 0);
        assert_eq!(stats.cache_creation_tokens, 0);
    }

    #[test]
    fn test_is_effective_boundary_at_50_percent() {
        // Exactly 50% should NOT be effective (> 0.5 required)
        let stats = CacheStats {
            input_tokens: 10000,
            cache_read_tokens: 5000,
            ..Default::default()
        };
        assert!(!stats.is_effective());
    }

    #[test]
    fn test_cost_per_call_zero_tokens() {
        let stats = CacheStats::default();
        assert_eq!(stats.cost_per_call(), 0.0);
    }

    #[test]
    fn test_cost_per_call_all_cached() {
        let stats = CacheStats {
            input_tokens: 10000,
            output_tokens: 0,
            cache_read_tokens: 10000,
            cache_creation_tokens: 0,
        };
        let cost = stats.cost_per_call();
        // 10000 * 0.0003 / 1000 = $0.003
        assert!((cost - 0.003).abs() < 1e-10);
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = CachePromptConfig::aggressive();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: CachePromptConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.enabled, config.enabled);
        assert_eq!(deserialized.min_content_length, config.min_content_length);
        assert_eq!(deserialized.cache_system_prompt, config.cache_system_prompt);
        assert_eq!(
            deserialized.cache_last_n_messages,
            config.cache_last_n_messages
        );
    }

    #[test]
    fn test_savings_when_cache_read_exceeds_input() {
        // Edge case: cache_read_tokens > input_tokens should not panic
        let stats = CacheStats {
            input_tokens: 5000,
            output_tokens: 100,
            cache_read_tokens: 8000,
            cache_creation_tokens: 0,
        };
        // Should not panic due to saturating_sub
        let _ = stats.savings();
    }

    #[test]
    fn test_merge_into_default() {
        let mut stats = CacheStats::default();
        let other = CacheStats::new(100, 50, 80, 10);
        stats.merge(&other);
        assert_eq!(stats.input_tokens, 100);
        assert_eq!(stats.output_tokens, 50);
        assert_eq!(stats.cache_read_tokens, 80);
        assert_eq!(stats.cache_creation_tokens, 10);
    }

    #[test]
    fn test_apply_cache_control_single_user_with_last_n() {
        // With last_n_messages = 3 and only 1 user message, it should be cached
        let config = CachePromptConfig {
            min_content_length: usize::MAX,
            cache_last_n_messages: 3,
            cache_system_prompt: false,
            ..Default::default()
        };
        let mut messages = vec![ChatMessage::user("Short msg")];
        apply_cache_control(&mut messages, &config);
        assert!(messages[0].cache_control.is_some());
    }

    #[test]
    fn test_env_flag_default_on() {
        assert!(env_flag_enabled_default_on(None));
        assert!(env_flag_enabled_default_on(Some("")));
        assert!(env_flag_enabled_default_on(Some("1")));
        assert!(env_flag_enabled_default_on(Some("true")));
        assert!(!env_flag_enabled_default_on(Some("0")));
        assert!(!env_flag_enabled_default_on(Some("false")));
        assert!(!env_flag_enabled_default_on(Some("OFF")));
    }

    #[test]
    fn test_prompt_cache_key_sanitizes_and_scopes_role() {
        let key = prompt_cache_key("Extract", "Mistral", "mistral-small-latest");
        assert_eq!(key, "eq:extract:mistral:mistral-small-latest");
        let messy = prompt_cache_key("query", "OpenAI", "gpt-5.6 mini");
        assert_eq!(messy, "eq:query:openai:gpt-5.6-mini");
        assert!(!key.contains(' '));
    }

    #[test]
    fn test_openai_chat_api_contract_is_constructor_not_host() {
        assert!(OpenAiChatApi::Native.allows_explicit_breakpoints());
        assert!(OpenAiChatApi::Azure.allows_explicit_breakpoints());
        assert!(!OpenAiChatApi::Compatible.allows_explicit_breakpoints());
    }

    #[test]
    fn test_prepare_chat_openai_fills_key() {
        let messages = vec![
            ChatMessage::system("Stable extract instructions"),
            ChatMessage::user("unique chunk"),
        ];
        let (out, opts) = prepare_chat(
            &messages,
            None,
            PromptCachePolicy::OpenAiCompatible,
            "mistral",
            "mistral-small-latest",
        );
        assert!(opts.prompt_cache_key.as_deref().unwrap().starts_with("eq:"));
        assert!(out[0].cache_control.is_none());
    }

    #[test]
    fn test_prepare_chat_anthropic_marks_system_only() {
        let messages = vec![
            ChatMessage::system("Stable extract instructions"),
            ChatMessage::user("unique chunk"),
        ];
        let (out, _) = prepare_chat(
            &messages,
            None,
            PromptCachePolicy::Anthropic,
            "anthropic",
            "claude-sonnet-4-5",
        );
        assert!(out[0].cache_control.is_some());
        assert!(out[1].cache_control.is_none());
        assert_eq!(
            out[0].cache_control.as_ref().unwrap().ttl.as_deref(),
            Some("5m")
        );
    }

    #[test]
    fn test_prepare_chat_preserves_caller_key() {
        use crate::traits::CompletionOptions;
        let messages = vec![ChatMessage::system("sys")];
        let options = CompletionOptions {
            prompt_cache_key: Some("eq:extract:mistral:mistral-small-latest".into()),
            ..Default::default()
        };
        let (_, opts) = prepare_chat(
            &messages,
            Some(&options),
            PromptCachePolicy::OpenAiCompatible,
            "mistral",
            "other-model",
        );
        assert_eq!(
            opts.prompt_cache_key.as_deref(),
            Some("eq:extract:mistral:mistral-small-latest")
        );
    }

    #[test]
    fn test_explicit_breakpoints_follow_api_contract_not_model_name() {
        reset_explicit_breakpoint_memory_for_tests();
        assert!(should_attempt_explicit_breakpoints(
            OpenAiChatApi::Native,
            "any-deployment-name"
        ));
        assert!(should_attempt_explicit_breakpoints(
            OpenAiChatApi::Azure,
            "my-prod-chat"
        ));
        assert!(!should_attempt_explicit_breakpoints(
            OpenAiChatApi::Compatible,
            "gpt-5.6-sol"
        ));
        remember_explicit_breakpoint_support(OpenAiChatApi::Native, "gpt-5.4-mini", false);
        assert!(!should_attempt_explicit_breakpoints(
            OpenAiChatApi::Native,
            "gpt-5.4-mini"
        ));
        assert!(should_attempt_explicit_breakpoints(
            OpenAiChatApi::Native,
            "any-deployment-name"
        ));
    }

    #[test]
    fn test_prepare_chat_openrouter_marks_system_and_fills_key() {
        let messages = vec![
            ChatMessage::system("Stable extract instructions"),
            ChatMessage::user("unique chunk"),
        ];
        let (out, opts) = prepare_chat(
            &messages,
            None,
            PromptCachePolicy::OpenRouter,
            "openrouter",
            "anthropic/claude-sonnet-4",
        );
        assert!(out[0].cache_control.is_some());
        assert!(out[1].cache_control.is_none());
        assert!(opts
            .prompt_cache_key
            .as_deref()
            .unwrap()
            .starts_with("eq:openrouter:"));
    }

    #[test]
    fn test_unsupported_explicit_cache_error_matches_contract_fields() {
        assert!(is_unsupported_explicit_cache_error(
            &crate::error::LlmError::InvalidRequest(
                "invalid_request_error: Unknown parameter (param: prompt_cache_options)".into()
            )
        ));
        assert!(is_unsupported_explicit_cache_error(
            &crate::error::LlmError::ApiError(
                r#"Azure OpenAI error 400: {"error":{"message":"unknown","param":"prompt_cache_breakpoint","type":"invalid_request_error"}}"#.into()
            )
        ));
        assert!(is_unsupported_explicit_cache_error(
            &crate::error::LlmError::ApiError(
                r#"{"error":{"param":"prompt_cache_options","code":"unknown_parameter"}}"#.into()
            )
        ));
        assert!(!is_unsupported_explicit_cache_error(
            &crate::error::LlmError::InvalidRequest("temperature is invalid".into())
        ));
        assert!(
            !is_unsupported_explicit_cache_error(&crate::error::LlmError::ApiError(
                "see docs for prompt_cache_options when using GPT-5.6".into()
            )),
            "free-text mention must not trip the 400 retry"
        );
    }

    #[test]
    fn test_apply_openai_explicit_prompt_cache_marks_system_not_user() {
        let mut body = serde_json::json!({
            "model": "gpt-5.6",
            "prompt_cache_key": "eq:extract:openai:gpt-5.6",
            "messages": [
                { "role": "system", "content": "Stable extract instructions" },
                { "role": "user", "content": "chunk-17" }
            ]
        });
        assert!(apply_openai_explicit_prompt_cache(&mut body));
        assert_eq!(body["prompt_cache_options"]["mode"], "explicit");
        assert_eq!(body["prompt_cache_options"]["ttl"], "30m");
        assert_eq!(
            body["messages"][0]["content"][0]["prompt_cache_breakpoint"]["mode"],
            "explicit"
        );
        assert!(body["messages"][1]["content"].is_string());
    }

    #[test]
    fn test_apply_openai_explicit_prompt_cache_skips_user_only() {
        let mut body = serde_json::json!({
            "messages": [{ "role": "user", "content": "hello" }]
        });
        assert!(!apply_openai_explicit_prompt_cache(&mut body));
        assert!(body.get("prompt_cache_options").is_none());
    }

    #[test]
    fn test_cache_write_tokens_from_chat_usage() {
        let usage = serde_json::json!({
            "prompt_tokens": 2600,
            "prompt_tokens_details": {
                "cached_tokens": 2000,
                "cache_write_tokens": 400
            }
        });
        assert_eq!(cache_write_tokens_from_chat_usage(&usage), Some(400));
    }
}
