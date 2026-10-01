//! Ollama thinking capability resolver (SPEC-113 / #369).
//!
//! Capability SSOT: Ollama `capabilities` array (`"thinking"`), not model-name substrings.
//! Auto path omits `think` unless support is [`ThinkingSupport::Yes`] (or `legacy_name` mode).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use reqwest::Client;
use serde_json::Value;
use tracing::{debug, warn};

use crate::traits::CompletionOptions;

/// Whether the active Ollama model artifact supports the Thinking API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingSupport {
    Yes,
    No,
    Unknown,
}

impl ThinkingSupport {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Yes => "yes",
            Self::No => "no",
            Self::Unknown => "unknown",
        }
    }
}

/// Escape hatch for operators / rollback (env `EDGEQUAKE_OLLAMA_THINK_CAPABILITY`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkCapabilityMode {
    /// Probe `/api/show` (and tags warm); omit on Unknown.
    #[default]
    Auto,
    /// Never send `think`.
    ForceOff,
    /// Always send `think: true` (debug only).
    ForceOn,
    /// Pre-SPEC-113 name substring heuristic.
    LegacyName,
}

impl ThinkCapabilityMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::ForceOff => "force_off",
            Self::ForceOn => "force_on",
            Self::LegacyName => "legacy_name",
        }
    }
}

static NON_AUTO_MODE_LOG: Once = Once::new();

/// Parse `EDGEQUAKE_OLLAMA_THINK_CAPABILITY` (default `auto`).
pub fn think_capability_mode_from_env() -> ThinkCapabilityMode {
    let mode = match std::env::var("EDGEQUAKE_OLLAMA_THINK_CAPABILITY")
        .ok()
        .as_deref()
        .map(str::trim)
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        None | Some("") | Some("auto") => ThinkCapabilityMode::Auto,
        Some("force_off") | Some("off") | Some("0") | Some("false") => {
            ThinkCapabilityMode::ForceOff
        }
        Some("force_on") | Some("on") | Some("1") | Some("true") => ThinkCapabilityMode::ForceOn,
        Some("legacy_name") | Some("legacy") | Some("heuristic") => ThinkCapabilityMode::LegacyName,
        Some(other) => {
            warn!(
                mode = other,
                "unknown EDGEQUAKE_OLLAMA_THINK_CAPABILITY; using auto"
            );
            ThinkCapabilityMode::Auto
        }
    };
    if mode != ThinkCapabilityMode::Auto {
        NON_AUTO_MODE_LOG.call_once(|| {
            warn!(
                mode = mode.as_str(),
                "EDGEQUAKE_OLLAMA_THINK_CAPABILITY is non-default (SPEC-113 escape hatch)"
            );
        });
    }
    mode
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// TTL for capability cache entries (seconds). Env: `EDGEQUAKE_OLLAMA_CAPABILITY_TTL_SECS`.
pub fn capability_ttl_from_env() -> Duration {
    Duration::from_secs(env_u64("EDGEQUAKE_OLLAMA_CAPABILITY_TTL_SECS", 300))
}

/// Short TTL for [`ThinkingSupport::Unknown`] — must not suppress Auto think for a full probe TTL.
pub const UNKNOWN_CAPABILITY_TTL: Duration = Duration::from_secs(5);

fn ttl_for_support(support: ThinkingSupport, normal: Duration) -> Duration {
    match support {
        ThinkingSupport::Unknown => UNKNOWN_CAPABILITY_TTL,
        ThinkingSupport::Yes | ThinkingSupport::No => normal,
    }
}

/// Probe timeout (ms). Env: `EDGEQUAKE_OLLAMA_CAPABILITY_TIMEOUT_MS`.
pub fn capability_timeout_from_env() -> Duration {
    Duration::from_millis(env_u64("EDGEQUAKE_OLLAMA_CAPABILITY_TIMEOUT_MS", 2000))
}

/// DRY: `"thinking"` ∈ capabilities (case-sensitive per Ollama constant).
pub fn capabilities_include_thinking(caps: &[impl AsRef<str>]) -> bool {
    caps.iter().any(|c| c.as_ref() == "thinking")
}

/// Map a JSON `capabilities` array field to [`ThinkingSupport`].
///
/// - missing / non-array → [`ThinkingSupport::Unknown`]
/// - array containing `"thinking"` → [`ThinkingSupport::Yes`]
/// - array without `"thinking"` → [`ThinkingSupport::No`]
pub fn thinking_support_from_json_capabilities(value: &Value) -> ThinkingSupport {
    let Some(arr) = value.as_array() else {
        return ThinkingSupport::Unknown;
    };
    let caps: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
    if capabilities_include_thinking(&caps) {
        ThinkingSupport::Yes
    } else {
        ThinkingSupport::No
    }
}

/// Legacy name heuristic (SPEC-113 escape hatch only — not default SSOT).
pub fn is_thinking_model_legacy(model: &str) -> bool {
    let model_lower = model.to_lowercase();
    model_lower.contains("deepseek-r1")
        || model_lower.contains("qwen3")
        || model_lower.contains("qwq")
        || model_lower.contains("openthinker")
        || model_lower.contains("phi4-reasoning")
        || model_lower.contains("magistral")
        || model_lower.contains("cogito")
        || model_lower.contains("gpt-oss")
}

/// Pure sync mapper: effort + capability → Ollama wire `think` value.
pub fn map_think(
    model: &str,
    opts: &CompletionOptions,
    support: ThinkingSupport,
    mode: ThinkCapabilityMode,
) -> Option<Value> {
    // WHY Bool(false) not omit: Ollama enables thinking by default for
    // thinking-capable models when `think` is absent (docs + /api/generate).
    // Omitting the field therefore does *not* disable reasoning.
    if mode == ThinkCapabilityMode::ForceOff {
        return Some(Value::Bool(false));
    }
    if mode == ThinkCapabilityMode::ForceOn {
        return Some(Value::Bool(true));
    }

    let desired = opts.reasoning_effort.as_deref();
    let clamped = crate::reasoning_capabilities::clamp_reasoning_effort("ollama", model, desired);
    let level = match (clamped, desired) {
        (Some(c), _) => Some(c),
        (None, Some(d))
            if crate::reasoning_capabilities::capabilities("ollama", model).is_none() =>
        {
            Some(d.trim().to_ascii_lowercase())
        }
        _ => None,
    };

    if let Some(level) = level {
        let wire = match level.as_str() {
            "none" | "false" | "off" | "0" | "minimal" => Some(Value::Bool(false)),
            "true" | "on" | "1" => Some(Value::Bool(true)),
            "high" | "medium" | "low" | "max" => Some(Value::String(level.clone())),
            other if mode == ThinkCapabilityMode::LegacyName && is_thinking_model_legacy(model) => {
                debug!(level = other, "legacy_name: unknown effort → think true");
                Some(Value::Bool(true))
            }
            _ => None,
        };

        let wire = wire?;

        if mode == ThinkCapabilityMode::LegacyName {
            return Some(wire);
        }

        match support {
            ThinkingSupport::Yes => return Some(wire),
            ThinkingSupport::No | ThinkingSupport::Unknown => {
                // Explicit off is safe to send even when capability is unknown/no —
                // it cannot enable thinking. Explicit on/levels still require Yes.
                if wire == Value::Bool(false) {
                    return Some(wire);
                }
                warn!(
                    model,
                    support = support.as_str(),
                    effort = %level,
                    "omitting Ollama think: model is not thinking-capable (SPEC-113)"
                );
                return None;
            }
        }
    }

    // Auto (effort unset)
    if desired.is_some() {
        return None;
    }

    match mode {
        ThinkCapabilityMode::LegacyName => {
            if is_thinking_model_legacy(model) {
                Some(Value::Bool(true))
            } else {
                None
            }
        }
        ThinkCapabilityMode::Auto => {
            if support == ThinkingSupport::Yes {
                Some(Value::Bool(true))
            } else {
                None
            }
        }
        ThinkCapabilityMode::ForceOff | ThinkCapabilityMode::ForceOn => unreachable!(),
    }
}

#[derive(Debug)]
struct CacheEntry {
    support: ThinkingSupport,
    fetched_at: Instant,
}

type InflightKey = (String, String);
type InflightGuard = Arc<tokio::sync::Mutex<()>>;
type InflightMap = HashMap<InflightKey, InflightGuard>;

/// TTL cache keyed by `(host, model)` with singleflight probe.
#[derive(Debug, Default)]
pub struct OllamaCapabilityCache {
    entries: Mutex<HashMap<(String, String), CacheEntry>>,
    /// Inflight keys — holds the mutex while a probe runs (singleflight).
    inflight: Mutex<InflightMap>,
    /// Number of `/api/show` HTTP attempts (tests).
    show_requests: AtomicU64,
}

impl OllamaCapabilityCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn show_request_count(&self) -> u64 {
        self.show_requests.load(Ordering::Relaxed)
    }

    pub fn invalidate(&self, host: &str, model: &str) {
        let key = cache_key(host, model);
        if let Ok(mut g) = self.entries.lock() {
            g.remove(&key);
        }
    }

    pub fn clear(&self) {
        if let Ok(mut g) = self.entries.lock() {
            g.clear();
        }
    }

    fn get_fresh(&self, host: &str, model: &str, normal_ttl: Duration) -> Option<ThinkingSupport> {
        let key = cache_key(host, model);
        let g = self.entries.lock().ok()?;
        let entry = g.get(&key)?;
        let ttl = ttl_for_support(entry.support, normal_ttl);
        if entry.fetched_at.elapsed() <= ttl {
            Some(entry.support)
        } else {
            None
        }
    }

    fn insert(&self, host: &str, model: &str, support: ThinkingSupport) {
        // Yes/No are stable model facts. Unknown is "we don't know yet" — still inserted
        // but only with UNKNOWN_CAPABILITY_TTL (enforced in get_fresh).
        self.insert_at(host, model, support, Instant::now());
    }

    fn insert_at(&self, host: &str, model: &str, support: ThinkingSupport, fetched_at: Instant) {
        let key = cache_key(host, model);
        if let Ok(mut g) = self.entries.lock() {
            g.insert(
                key,
                CacheEntry {
                    support,
                    fetched_at,
                },
            );
        }
    }

    #[cfg(test)]
    fn insert_aged_for_test(
        &self,
        host: &str,
        model: &str,
        support: ThinkingSupport,
        age: Duration,
    ) {
        let fetched_at = Instant::now().checked_sub(age).unwrap_or_else(Instant::now);
        self.insert_at(host, model, support, fetched_at);
    }

    /// Seed cache from `/api/tags` model entries (show remains authoritative on miss).
    pub fn warm_from_tags_json(&self, host: &str, body: &Value) {
        let Some(models) = body.get("models").and_then(|m| m.as_array()) else {
            return;
        };
        for m in models {
            let Some(name) = m
                .get("name")
                .and_then(|v| v.as_str())
                .or_else(|| m.get("model").and_then(|v| v.as_str()))
            else {
                continue;
            };
            if name.is_empty() {
                continue;
            }
            let support = match m.get("capabilities") {
                None => continue, // old Ollama — leave uncached
                Some(caps) => thinking_support_from_json_capabilities(caps),
            };
            self.insert(host, name, support);
        }
    }
}

fn cache_key(host: &str, model: &str) -> (String, String) {
    (host.trim_end_matches('/').to_string(), model.to_string())
}

/// Fetches and caches thinking capability for Ollama models.
#[derive(Debug, Clone)]
pub struct OllamaCapabilityResolver {
    cache: Arc<OllamaCapabilityCache>,
    ttl: Duration,
    timeout: Duration,
}

impl OllamaCapabilityResolver {
    pub fn new(cache: Arc<OllamaCapabilityCache>) -> Self {
        Self {
            cache,
            ttl: capability_ttl_from_env(),
            timeout: capability_timeout_from_env(),
        }
    }

    pub fn with_ttl_timeout(
        cache: Arc<OllamaCapabilityCache>,
        ttl: Duration,
        timeout: Duration,
    ) -> Self {
        Self {
            cache,
            ttl,
            timeout,
        }
    }

    pub fn cache(&self) -> &Arc<OllamaCapabilityCache> {
        &self.cache
    }

    /// Resolve thinking support for `model` on `host` (prefers `/api/show`).
    pub async fn thinking_support(
        &self,
        client: &Client,
        host: &str,
        model: &str,
    ) -> ThinkingSupport {
        if let Some(cached) = self.cache.get_fresh(host, model, self.ttl) {
            return cached;
        }

        let key = cache_key(host, model);
        let gate = {
            let mut inflight = self
                .cache
                .inflight
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            inflight
                .entry(key.clone())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                .clone()
        };
        let _guard = gate.lock().await;

        // Re-check after acquiring singleflight lock.
        if let Some(cached) = self.cache.get_fresh(host, model, self.ttl) {
            return cached;
        }

        let support = self.fetch_show(client, host, model).await;
        // Cache all outcomes; Unknown uses short TTL (LAW-113-3 — do not lie for 300s).
        self.cache.insert(host, model, support);
        support
    }

    async fn fetch_show(&self, client: &Client, host: &str, model: &str) -> ThinkingSupport {
        self.cache.show_requests.fetch_add(1, Ordering::Relaxed);
        let url = format!("{}/api/show", host.trim_end_matches('/'));
        let body = serde_json::json!({ "model": model });
        let result = client
            .post(&url)
            .json(&body)
            .timeout(self.timeout)
            .send()
            .await;
        match result {
            Ok(resp) if resp.status().is_success() => match resp.json::<Value>().await {
                Ok(json) => match json.get("capabilities") {
                    Some(caps) => thinking_support_from_json_capabilities(caps),
                    None => ThinkingSupport::Unknown,
                },
                Err(e) => {
                    debug!(error = %e, "ollama /api/show JSON parse failed → Unknown");
                    ThinkingSupport::Unknown
                }
            },
            Ok(resp) => {
                debug!(status = %resp.status(), "ollama /api/show non-success → Unknown");
                ThinkingSupport::Unknown
            }
            Err(e) => {
                debug!(error = %e, "ollama /api/show failed → Unknown");
                ThinkingSupport::Unknown
            }
        }
    }

    /// Bulk-warm cache from GET `/api/tags`.
    pub async fn warm_from_tags(&self, client: &Client, host: &str) {
        let url = format!("{}/api/tags", host.trim_end_matches('/'));
        let Ok(resp) = client.get(&url).timeout(self.timeout).send().await else {
            return;
        };
        if !resp.status().is_success() {
            return;
        }
        let Ok(body) = resp.json::<Value>().await else {
            return;
        };
        self.cache.warm_from_tags_json(host, &body);
    }
}

/// Resolve wire `think` for a request (mode + probe + map).
pub async fn resolve_think_value(
    client: &Client,
    host: &str,
    model: &str,
    opts: &CompletionOptions,
    mode: ThinkCapabilityMode,
    resolver: &OllamaCapabilityResolver,
) -> Option<Value> {
    let support = match mode {
        ThinkCapabilityMode::ForceOff => ThinkingSupport::No,
        ThinkCapabilityMode::ForceOn => ThinkingSupport::Yes,
        ThinkCapabilityMode::LegacyName => {
            if is_thinking_model_legacy(model) {
                ThinkingSupport::Yes
            } else {
                ThinkingSupport::No
            }
        }
        ThinkCapabilityMode::Auto => resolver.thinking_support(client, host, model).await,
    };

    let think = map_think(model, opts, support, mode);
    let sent = match &think {
        None => "omit",
        Some(Value::Bool(true)) => "true",
        Some(Value::Bool(false)) => "false",
        Some(Value::String(s)) => s.as_str(),
        Some(_) => "other",
    };
    debug!(
        target: "ollama.think_decision",
        support = support.as_str(),
        effort = opts.reasoning_effort.as_deref().unwrap_or("auto"),
        mode = mode.as_str(),
        sent,
        "ollama think decision"
    );
    think
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::CompletionOptions;

    #[test]
    fn t113_01_capabilities_include_thinking_true() {
        assert!(capabilities_include_thinking(&["completion", "thinking"]));
    }

    #[test]
    fn t113_02_capabilities_include_thinking_false() {
        assert!(!capabilities_include_thinking(&["completion", "vision"]));
    }

    #[test]
    fn t113_03_auto_yes_sends_think() {
        let opts = CompletionOptions::default();
        let v = map_think(
            "anything",
            &opts,
            ThinkingSupport::Yes,
            ThinkCapabilityMode::Auto,
        );
        assert_eq!(v, Some(Value::Bool(true)));
    }

    #[test]
    fn t113_04_auto_no_omits() {
        let opts = CompletionOptions::default();
        assert!(map_think(
            "qwen3:8b",
            &opts,
            ThinkingSupport::No,
            ThinkCapabilityMode::Auto
        )
        .is_none());
    }

    #[test]
    fn t113_05_auto_unknown_omits() {
        let opts = CompletionOptions::default();
        assert!(map_think(
            "qwen3:8b",
            &opts,
            ThinkingSupport::Unknown,
            ThinkCapabilityMode::Auto
        )
        .is_none());
    }

    #[test]
    fn t113_06_qwen3_vl_name_no_support_omits() {
        let opts = CompletionOptions::default();
        assert!(map_think(
            "qwen3-vl:8b",
            &opts,
            ThinkingSupport::No,
            ThinkCapabilityMode::Auto
        )
        .is_none());
    }

    #[test]
    fn t113_07_alias_yes_allows_auto_think() {
        let opts = CompletionOptions::default();
        let v = map_think(
            "vl-instruct-8b",
            &opts,
            ThinkingSupport::Yes,
            ThinkCapabilityMode::Auto,
        );
        assert_eq!(v, Some(Value::Bool(true)));
    }

    #[test]
    fn t113_08_explicit_none_sends_think_false_when_yes() {
        let opts = CompletionOptions {
            reasoning_effort: Some("none".into()),
            ..Default::default()
        };
        assert_eq!(
            map_think(
                "qwen3:8b",
                &opts,
                ThinkingSupport::Yes,
                ThinkCapabilityMode::Auto
            ),
            Some(Value::Bool(false))
        );
    }

    #[test]
    fn t113_08b_force_off_sends_think_false() {
        let opts = CompletionOptions::default();
        assert_eq!(
            map_think(
                "qwen3.6:35b-a3b",
                &opts,
                ThinkingSupport::Yes,
                ThinkCapabilityMode::ForceOff
            ),
            Some(Value::Bool(false))
        );
    }

    #[test]
    fn t113_09_explicit_high_no_support_omits() {
        let opts = CompletionOptions {
            reasoning_effort: Some("high".into()),
            ..Default::default()
        };
        assert!(map_think(
            "qwen3-vl:8b",
            &opts,
            ThinkingSupport::No,
            ThinkCapabilityMode::Auto
        )
        .is_none());
    }

    #[test]
    fn t113_24_issue_fixture_name_qwen3_caps_no() {
        let opts = CompletionOptions::default();
        assert!(map_think(
            "qwen3-vl:latest",
            &opts,
            ThinkingSupport::No,
            ThinkCapabilityMode::Auto
        )
        .is_none());
    }

    #[test]
    fn t113_15_legacy_name_qwen3_vl_sends_think() {
        let opts = CompletionOptions::default();
        let v = map_think(
            "qwen3-vl:8b",
            &opts,
            ThinkingSupport::No,
            ThinkCapabilityMode::LegacyName,
        );
        assert_eq!(v, Some(Value::Bool(true)));
    }

    #[test]
    fn t113_explicit_high_yes_sends_level() {
        let opts = CompletionOptions {
            reasoning_effort: Some("high".into()),
            ..Default::default()
        };
        let v = map_think("m", &opts, ThinkingSupport::Yes, ThinkCapabilityMode::Auto).unwrap();
        assert_eq!(v, Value::String("high".into()));
    }

    #[test]
    fn thinking_support_from_json_fixtures() {
        let thinking = serde_json::json!(["completion", "tools", "thinking"]);
        assert_eq!(
            thinking_support_from_json_capabilities(&thinking),
            ThinkingSupport::Yes
        );
        let vl = serde_json::json!(["completion", "vision"]);
        assert_eq!(
            thinking_support_from_json_capabilities(&vl),
            ThinkingSupport::No
        );
        assert_eq!(
            thinking_support_from_json_capabilities(&Value::Null),
            ThinkingSupport::Unknown
        );
    }

    #[test]
    fn cache_hosts_isolated() {
        let cache = OllamaCapabilityCache::new();
        cache.insert("http://a:11434", "m", ThinkingSupport::Yes);
        cache.insert("http://b:11434", "m", ThinkingSupport::No);
        assert_eq!(
            cache.get_fresh("http://a:11434", "m", Duration::from_secs(60)),
            Some(ThinkingSupport::Yes)
        );
        assert_eq!(
            cache.get_fresh("http://b:11434", "m", Duration::from_secs(60)),
            Some(ThinkingSupport::No)
        );
    }

    #[test]
    fn unknown_cache_expires_before_normal_ttl() {
        let cache = OllamaCapabilityCache::new();
        let normal = Duration::from_secs(300);
        // Fresh Unknown is still served (stampede collapse).
        cache.insert("http://h", "m", ThinkingSupport::Unknown);
        assert_eq!(
            cache.get_fresh("http://h", "m", normal),
            Some(ThinkingSupport::Unknown)
        );
        // After UNKNOWN_CAPABILITY_TTL, Unknown must not suppress re-probe.
        cache.insert_aged_for_test(
            "http://h",
            "m",
            ThinkingSupport::Unknown,
            UNKNOWN_CAPABILITY_TTL + Duration::from_millis(50),
        );
        assert!(
            cache.get_fresh("http://h", "m", normal).is_none(),
            "Unknown must expire well before normal TTL"
        );
        // Yes still lives for normal TTL after the same age.
        cache.insert_aged_for_test(
            "http://h",
            "m2",
            ThinkingSupport::Yes,
            UNKNOWN_CAPABILITY_TTL + Duration::from_millis(50),
        );
        assert_eq!(
            cache.get_fresh("http://h", "m2", normal),
            Some(ThinkingSupport::Yes)
        );
    }
}
