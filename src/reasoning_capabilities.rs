//! Model reasoning-effort capability registry (SPEC-109).
//!
//! Capability SSOT for which `reasoning_effort` values a provider/model accepts.
//! Callers resolve a *desired* effort; this module clamps before wire send so
//! illegal values never produce HTTP 400s (e.g. `gpt-5-mini` rejects `none`).

/// Ordered product vocabulary (low → high).
pub const EFFORT_SCALE: &[&str] = &["none", "minimal", "low", "medium", "high", "xhigh", "max"];

/// Documented capabilities for a model that supports reasoning effort.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningCapabilities {
    /// Allowed effort strings (subset of [`EFFORT_SCALE`]).
    pub supported: &'static [&'static str],
    /// Informational provider default when the field is omitted.
    pub default_when_omitted: Option<&'static str>,
}

/// Effort vocabulary for Ollama models that are **live** thinking-capable (SPEC-113).
///
/// Call only after Ollama `capabilities` includes `"thinking"` (catalog / UI).
pub fn ollama_thinking_effort_vocab() -> ReasoningCapabilities {
    ReasoningCapabilities {
        supported: &["low", "medium", "high", "max"],
        default_when_omitted: None,
    }
}

/// Look up reasoning capabilities for a provider + model.
///
/// Returns `None` when the model is non-reasoning (field must be omitted).
pub fn capabilities(provider: &str, model: &str) -> Option<ReasoningCapabilities> {
    let p = provider.to_ascii_lowercase();
    let m = model.to_ascii_lowercase();

    // Provider-agnostic OpenAI-family model ids (native openai, azure, openai-compatible).
    if let Some(caps) = openai_family_capabilities(&m) {
        return Some(caps);
    }

    if p.contains("mistral") || m.contains("mistral") || m.contains("magistral") {
        return mistral_capabilities(&m);
    }

    if p.contains("anthropic") || p.contains("claude") || m.contains("claude") {
        return anthropic_capabilities(&m);
    }

    if p.contains("gemini") || m.contains("gemini") {
        if m.contains("gemini") {
            // Gemini 3.7 and 3.8 reject `minimal` (HTTP 400). 3.6 and 3.5 accept it.
            // Source: https://ai.google.dev/gemini-api/docs/generate-content/thinking
            if m.contains("gemini-3.7") || m.contains("gemini-3.8") {
                return Some(ReasoningCapabilities {
                    supported: &["low", "medium", "high"],
                    default_when_omitted: Some("medium"),
                });
            }
            // Map product effort → thinking; older Gemini chat models.
            return Some(ReasoningCapabilities {
                supported: &["none", "minimal", "low", "medium", "high"],
                default_when_omitted: None,
            });
        }
    }

    if p.contains("ollama") {
        // SPEC-113: name is not SSOT. Static registry always returns None for Ollama;
        // live `thinking` capability + [`ollama_thinking_effort_vocab`] gate eligibility.
        let _ = m;
        return None;
    }

    if p.contains("xai") || m.starts_with("grok") {
        // Non-reasoning build models omit the field.
        if m.contains("grok-build") {
            return None;
        }
        // Grok 4.6 / 4.7: low|medium|high|xhigh (cannot disable with none).
        // Source: https://docs.x.ai/developers/models
        if m.contains("grok-4.7") || m.contains("grok-4.6") {
            return Some(ReasoningCapabilities {
                supported: &["low", "medium", "high", "xhigh"],
                default_when_omitted: Some("high"),
            });
        }
        // Grok 4.5: low|medium|high (cannot disable).
        if m.contains("grok-4.5") {
            return Some(ReasoningCapabilities {
                supported: &["low", "medium", "high"],
                default_when_omitted: Some("high"),
            });
        }
        // Older Grok chat models (e.g. 4.3): allow none for legacy callers.
        return Some(ReasoningCapabilities {
            supported: &["none", "low", "medium", "high"],
            default_when_omitted: None,
        });
    }

    if p.contains("nvidia") || m.contains("deepseek") || m.contains("nemotron") {
        if m.contains("deepseek") || m.contains("nemotron") {
            return Some(ReasoningCapabilities {
                supported: &["low", "medium", "high", "max"],
                default_when_omitted: None,
            });
        }
    }

    // LM Studio / openai-compatible thinking models (heuristic).
    if p.contains("lmstudio")
        || p.contains("openai-compatible")
        || p.contains("openrouter")
        || p.contains("omlx")
        || p.contains("llamacpp")
        || p.contains("vllm")
        || p.contains("mlx")
    {
        if let Some(caps) = thinking_model_heuristic(&m) {
            return Some(caps);
        }
    }

    if p.contains("openrouter") {
        // OpenRouter translates `reasoning.effort` across many backends.
        return Some(ReasoningCapabilities {
            supported: &["none", "minimal", "low", "medium", "high", "xhigh", "max"],
            default_when_omitted: None,
        });
    }

    None
}

fn thinking_model_heuristic(m: &str) -> Option<ReasoningCapabilities> {
    if m.contains("qwen")
        || m.contains("deepseek")
        || m.contains("r1")
        || m.contains("think")
        || m.contains("reasoning")
        || m.contains("o1")
        || m.contains("o3")
        || m.contains("o4")
    {
        return Some(ReasoningCapabilities {
            supported: &["none", "low", "medium", "high", "max"],
            default_when_omitted: None,
        });
    }
    None
}

fn openai_family_capabilities(m: &str) -> Option<ReasoningCapabilities> {
    // Non-reasoning GPT-4.x / 4o
    if m.contains("gpt-4") && !m.contains("gpt-5") {
        return None;
    }

    // GPT-6 Luna: none…max, omitted default medium.
    // Chat Completions function calling only works at effort `none`.
    // Source: https://developers.openai.com/api/docs/models/gpt-6-luna
    if m.contains("gpt-6-luna") {
        return Some(ReasoningCapabilities {
            supported: &["none", "low", "medium", "high", "xhigh", "max"],
            default_when_omitted: Some("medium"),
        });
    }

    // GPT-6.1 Sol: no `none` or `minimal`. Omitted default is medium.
    // Chat Completions does not support tool calling on this model.
    // Source: https://developers.openai.com/api/docs/models/gpt-6.1-sol
    if m.contains("gpt-6.1") {
        return Some(ReasoningCapabilities {
            supported: &["low", "medium", "high", "xhigh", "max"],
            default_when_omitted: Some("medium"),
        });
    }

    // GPT-6 Astra: same effort set. The model page does not state an omitted default.
    // Source: https://developers.openai.com/api/docs/models/gpt-6-astra
    if m.contains("gpt-6") {
        return Some(ReasoningCapabilities {
            supported: &["low", "medium", "high", "xhigh", "max"],
            default_when_omitted: None,
        });
    }

    // gpt-5-mini / gpt-5-nano (2025-08): no `none`
    if m.contains("gpt-5-mini") || (m.contains("gpt-5-nano") && !m.contains("gpt-5.4")) {
        return Some(ReasoningCapabilities {
            supported: &["minimal", "low", "medium", "high"],
            default_when_omitted: Some("medium"),
        });
    }

    // gpt-5.4-mini / gpt-5.4-nano
    if m.contains("gpt-5.4-mini") || m.contains("gpt-5.4-nano") {
        return Some(ReasoningCapabilities {
            supported: &["none", "low", "medium", "high", "xhigh"],
            default_when_omitted: Some("none"),
        });
    }

    // gpt-5.6 family
    if m.contains("gpt-5.6") {
        return Some(ReasoningCapabilities {
            supported: &["none", "low", "medium", "high", "xhigh", "max"],
            default_when_omitted: Some("medium"),
        });
    }

    // gpt-5.4 / gpt-5.5 (full)
    if m.contains("gpt-5.4") || m.contains("gpt-5.5") {
        return Some(ReasoningCapabilities {
            supported: &["none", "low", "medium", "high", "xhigh"],
            default_when_omitted: Some("medium"),
        });
    }

    // gpt-5.1 / gpt-5.2
    if m.contains("gpt-5.1") || m.contains("gpt-5.2") {
        return Some(ReasoningCapabilities {
            supported: &["none", "low", "medium", "high"],
            default_when_omitted: Some("medium"),
        });
    }

    // Base gpt-5 (not mini/nano/5.x)
    if m.contains("gpt-5") {
        return Some(ReasoningCapabilities {
            supported: &["minimal", "low", "medium", "high"],
            default_when_omitted: Some("medium"),
        });
    }

    // o-series
    if m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("o4")
        || m.contains("o1-")
        || m.contains("o3-")
        || m.contains("o4-")
    {
        return Some(ReasoningCapabilities {
            supported: &["low", "medium", "high"],
            default_when_omitted: Some("medium"),
        });
    }

    None
}

fn mistral_capabilities(m: &str) -> Option<ReasoningCapabilities> {
    if m.contains("mistral-large") || m.contains("magistra") || m.contains("codestral") {
        return None; // API 3051 — omit always
    }
    if m.contains("mistral-small")
        || m.contains("mistral-medium-3")
        || m.contains("mistral-medium-250")
        || m.contains("medium-3-5")
        || m.contains("mistral-medium-latest")
        || m.contains("mistral-medium")
    {
        // Practical surface: high vs none (docs); accept common mid-tiers and clamp.
        return Some(ReasoningCapabilities {
            supported: &["none", "low", "medium", "high"],
            default_when_omitted: None,
        });
    }
    None
}

fn anthropic_capabilities(m: &str) -> Option<ReasoningCapabilities> {
    // Claude Opus 5.5: full ladder, omitted default is medium (Oct 2026 docs).
    if m.contains("opus-5-5") || m.contains("opus-5.5") {
        return Some(ReasoningCapabilities {
            supported: &["low", "medium", "high", "xhigh", "max"],
            default_when_omitted: Some("medium"),
        });
    }
    // Claude Opus 5 / Fable 5 / Mythos 5 / Opus 4.8 / Opus 4.7 / Sonnet 5(.5):
    // full ladder, omitted default high.
    if m.contains("opus-5")
        || m.contains("opus-4-8")
        || m.contains("opus-4.8")
        || m.contains("opus-4-7")
        || m.contains("opus-4.7")
        || m.contains("fable-5")
        || m.contains("mythos")
        || m.contains("sonnet-5")
        || m.contains("claude-sonnet-5")
    {
        return Some(ReasoningCapabilities {
            supported: &["low", "medium", "high", "xhigh", "max"],
            default_when_omitted: Some("high"),
        });
    }
    // Opus 4.6: max but no xhigh in older table; accept both for clamp-down.
    if m.contains("opus-4-6") || m.contains("opus-4.6") {
        return Some(ReasoningCapabilities {
            supported: &["low", "medium", "high", "max"],
            default_when_omitted: Some("high"),
        });
    }
    // Sonnet 4.6 / Opus 4.5 / generic Claude with effort support.
    if m.contains("sonnet-4-6")
        || m.contains("sonnet-4.6")
        || m.contains("opus-4-5")
        || m.contains("opus-4.5")
        || m.contains("claude")
        || m.contains("sonnet")
        || m.contains("opus")
    {
        return Some(ReasoningCapabilities {
            supported: &["low", "medium", "high", "max"],
            default_when_omitted: Some("high"),
        });
    }
    None
}

fn effort_index(effort: &str) -> Option<usize> {
    EFFORT_SCALE
        .iter()
        .position(|e| e.eq_ignore_ascii_case(effort.trim()))
}

/// Clamp a desired effort to a value the model accepts, or `None` to omit.
///
/// - Non-reasoning models → always `None`
/// - `desired == None` → `None` (Auto / omit)
/// - Exact match → keep
/// - Else nearest-lower on [`EFFORT_SCALE`]; if below all caps → lowest supported
pub fn clamp_reasoning_effort(
    provider: &str,
    model: &str,
    desired: Option<&str>,
) -> Option<String> {
    let caps = capabilities(provider, model)?;
    let desired = desired.map(str::trim).filter(|s| !s.is_empty())?;

    if caps
        .supported
        .iter()
        .any(|s| s.eq_ignore_ascii_case(desired))
    {
        return Some(desired.to_ascii_lowercase());
    }

    let Some(want_idx) = effort_index(desired) else {
        // Unknown token — do not guess upward
        return None;
    };

    // Nearest-lower: largest supported index that is <= desired
    let mut best: Option<(usize, &str)> = None;
    for &s in caps.supported {
        if let Some(idx) = effort_index(s) {
            if idx <= want_idx && best.map(|(bi, _)| idx > bi).unwrap_or(true) {
                best = Some((idx, s));
            }
        }
    }

    if let Some((_, s)) = best {
        return Some(s.to_string());
    }

    // Desired below all supported → lowest supported
    caps.supported.first().map(|s| (*s).to_string())
}

/// Lowest effort suitable for structured extraction / high-volume VLM.
pub fn lowest_for_structured_output(provider: &str, model: &str) -> Option<String> {
    let caps = capabilities(provider, model)?;
    for candidate in ["none", "minimal", "low"] {
        if caps.supported.iter().any(|s| *s == candidate) {
            return Some(candidate.to_string());
        }
    }
    caps.supported.first().map(|s| (*s).to_string())
}

/// Parse an effort string into async-openai's chat `ReasoningEffort`.
///
/// `"max"` maps to `Xhigh` (Chat Completions enum has no `Max`).
pub fn parse_openai_reasoning_effort(
    effort: &str,
) -> Option<async_openai::types::chat::ReasoningEffort> {
    use async_openai::types::chat::ReasoningEffort;
    match effort.trim().to_ascii_lowercase().as_str() {
        "none" => Some(ReasoningEffort::None),
        "minimal" => Some(ReasoningEffort::Minimal),
        "low" => Some(ReasoningEffort::Low),
        "medium" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        "xhigh" | "max" => Some(ReasoningEffort::Xhigh),
        _ => None,
    }
}

/// DRY: clamp `options.reasoning_effort` in place for a provider+model.
///
/// Sets the field to the clamped value, or clears it when the model must omit.
pub fn clamp_options_reasoning_effort(
    provider: &str,
    model: &str,
    options: &mut crate::traits::CompletionOptions,
) {
    let desired = options.reasoning_effort.as_deref();
    options.reasoning_effort = clamp_reasoning_effort(provider, model, desired);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt5_mini_none_clamps_to_minimal() {
        let got = clamp_reasoning_effort("openai", "gpt-5-mini", Some("none"));
        assert_eq!(got.as_deref(), Some("minimal"));
    }

    #[test]
    fn gpt54_nano_none_stays_none() {
        let got = clamp_reasoning_effort("openai", "gpt-5.4-nano", Some("none"));
        assert_eq!(got.as_deref(), Some("none"));
    }

    #[test]
    fn mistral_large_omits() {
        assert!(capabilities("mistral", "mistral-large-latest").is_none());
        assert!(clamp_reasoning_effort("mistral", "mistral-large-latest", Some("high")).is_none());
    }

    #[test]
    fn gpt41_omits() {
        assert!(capabilities("openai", "gpt-4.1-mini").is_none());
        assert!(clamp_reasoning_effort("openai", "gpt-4.1-mini", Some("low")).is_none());
    }

    #[test]
    fn lowest_structured_mini_is_minimal() {
        assert_eq!(
            lowest_for_structured_output("openai", "gpt-5-mini").as_deref(),
            Some("minimal")
        );
    }

    #[test]
    fn lowest_structured_nano54_is_none() {
        assert_eq!(
            lowest_for_structured_output("openai", "gpt-5.4-nano").as_deref(),
            Some("none")
        );
    }

    #[test]
    fn auto_omit_returns_none() {
        assert!(clamp_reasoning_effort("openai", "gpt-5-mini", None).is_none());
    }

    #[test]
    fn exact_match_preserved() {
        assert_eq!(
            clamp_reasoning_effort("openai", "gpt-5-mini", Some("low")).as_deref(),
            Some("low")
        );
    }

    #[test]
    fn xhigh_clamps_down_on_mini() {
        assert_eq!(
            clamp_reasoning_effort("openai", "gpt-5-mini", Some("xhigh")).as_deref(),
            Some("high")
        );
    }

    #[test]
    fn anthropic_sonnet5_supports_xhigh() {
        let caps = capabilities("anthropic", "claude-sonnet-5-5").expect("caps");
        assert!(caps.supported.contains(&"xhigh"));
        assert_eq!(caps.default_when_omitted, Some("high"));
        assert_eq!(
            clamp_reasoning_effort("anthropic", "claude-sonnet-5-5", Some("xhigh")).as_deref(),
            Some("xhigh")
        );
    }

    #[test]
    fn anthropic_opus55_default_effort_is_medium() {
        let caps = capabilities("anthropic", "claude-opus-5-5").expect("caps");
        assert_eq!(caps.default_when_omitted, Some("medium"));
    }

    #[test]
    fn gpt56_luna_supports_max_effort() {
        let caps = capabilities("openai", "gpt-5.6-luna").expect("caps");
        assert!(caps.supported.contains(&"max"));
        assert_eq!(
            clamp_reasoning_effort("openai", "gpt-5.6-terra", Some("max")).as_deref(),
            Some("max")
        );
    }

    #[test]
    fn gpt6_sol_rejects_none_luna_accepts_it() {
        assert_eq!(
            clamp_reasoning_effort("openai", "gpt-6.1-sol", Some("none")).as_deref(),
            Some("low")
        );
        assert_eq!(
            clamp_reasoning_effort("openai", "gpt-6-luna", Some("none")).as_deref(),
            Some("none")
        );
        let astra = capabilities("openai", "gpt-6-astra").expect("caps");
        assert!(astra.default_when_omitted.is_none());
        assert!(!astra.supported.contains(&"none"));
    }

    #[test]
    fn gemini36_keeps_minimal() {
        assert_eq!(
            clamp_reasoning_effort("gemini", "gemini-3.6-flash", Some("minimal")).as_deref(),
            Some("minimal")
        );
    }

    #[test]
    fn gemini38_rejects_minimal_clamps_to_low() {
        assert_eq!(
            clamp_reasoning_effort("gemini", "gemini-3.8-flash", Some("minimal")).as_deref(),
            Some("low")
        );
        assert_eq!(
            clamp_reasoning_effort("gemini", "gemini-3.8-flash", Some("none")).as_deref(),
            Some("low")
        );
    }

    #[test]
    fn grok47_supports_xhigh_rejects_none() {
        let caps = capabilities("xai", "grok-4.7").expect("caps");
        assert!(caps.supported.contains(&"xhigh"));
        assert!(!caps.supported.contains(&"none"));
        assert_eq!(
            clamp_reasoning_effort("xai", "grok-4.7", Some("none")).as_deref(),
            Some("low")
        );
    }

    #[test]
    fn xai_non_build_accepts_effort() {
        assert!(capabilities("xai", "grok-4.3").is_some());
        assert!(capabilities("xai", "grok-build").is_none());
    }

    #[test]
    fn clamp_options_mutates_in_place() {
        let mut opts = crate::traits::CompletionOptions {
            reasoning_effort: Some("none".into()),
            ..Default::default()
        };
        clamp_options_reasoning_effort("openai", "gpt-5-mini", &mut opts);
        assert_eq!(opts.reasoning_effort.as_deref(), Some("minimal"));
    }

    #[test]
    fn t113_18_ollama_static_registry_no_name_folklore() {
        assert!(capabilities("ollama", "qwen3-vl:8b").is_none());
        assert!(capabilities("ollama", "qwen3:8b").is_none());
        assert!(capabilities("ollama", "deepseek-r1:8b").is_none());
        let vocab = ollama_thinking_effort_vocab();
        assert!(vocab.supported.contains(&"high"));
    }
}
