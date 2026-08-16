//! SPEC-131 — fleet env knobs for omitting illegal LLM wire fields.
//!
//! WHY: Model parameter surfaces drift; operator policy (`omit`) outranks
//! hardcoded model substring catalogs (LAW-131-2 / LAW-131-4).

/// Environment variable: omit `temperature` on upstream requests.
pub const ENV_OMIT_TEMPERATURE: &str = "EDGEQUAKE_LLM_OMIT_TEMPERATURE";

/// Environment variable: omit `reasoning_effort` on upstream requests.
pub const ENV_OMIT_REASONING_EFFORT: &str = "EDGEQUAKE_LLM_OMIT_REASONING_EFFORT";

/// True when an env var is a truthy flag (`1`, `true`, `yes`, `on`; case-insensitive).
pub fn env_truthy(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let t = v.trim();
            t == "1"
                || t.eq_ignore_ascii_case("true")
                || t.eq_ignore_ascii_case("yes")
                || t.eq_ignore_ascii_case("on")
        }
        Err(_) => false,
    }
}

/// `EDGEQUAKE_LLM_OMIT_TEMPERATURE` — do not send temperature on the wire.
pub fn omit_temperature_from_env() -> bool {
    env_truthy(ENV_OMIT_TEMPERATURE)
}

/// `EDGEQUAKE_LLM_OMIT_REASONING_EFFORT` — do not send reasoning_effort on the wire.
pub fn omit_reasoning_effort_from_env() -> bool {
    env_truthy(ENV_OMIT_REASONING_EFFORT)
}

/// Force `None` when omit-reasoning-effort env is set (after SPEC-109 resolve).
pub fn apply_omit_reasoning_effort(effort: Option<String>) -> Option<String> {
    if omit_reasoning_effort_from_env() {
        None
    } else {
        effort
    }
}

/// Temperature to place on Chat Completions / Responses after env omit + ≈1.0 quirk.
///
/// - Omit-env → `None` (LAW-131-4)
/// - `Some(≈1.0)` → `None` (OpenAI default-only models reject explicit 1.0 on some paths)
/// - otherwise pass through
pub fn temperature_for_wire(temperature: Option<f32>) -> Option<f32> {
    if omit_temperature_from_env() {
        return None;
    }
    temperature.filter(|t| (*t - 1.0_f32).abs() > f32::EPSILON)
}

/// Reasoning effort for the wire after omit-env (does not clamp; callers clamp first).
pub fn reasoning_effort_for_wire(effort: Option<String>) -> Option<String> {
    apply_omit_reasoning_effort(effort)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clear_omit_env() {
        std::env::remove_var(ENV_OMIT_TEMPERATURE);
        std::env::remove_var(ENV_OMIT_REASONING_EFFORT);
    }

    #[test]
    fn env_truthy_accepts_common_flags() {
        clear_omit_env();
        std::env::set_var(ENV_OMIT_TEMPERATURE, "true");
        assert!(omit_temperature_from_env());
        std::env::set_var(ENV_OMIT_TEMPERATURE, "1");
        assert!(omit_temperature_from_env());
        std::env::set_var(ENV_OMIT_TEMPERATURE, "YES");
        assert!(omit_temperature_from_env());
        std::env::set_var(ENV_OMIT_TEMPERATURE, "0");
        assert!(!omit_temperature_from_env());
        clear_omit_env();
    }

    #[test]
    fn temperature_for_wire_omits_on_env_and_default_one() {
        clear_omit_env();
        assert_eq!(temperature_for_wire(Some(0.0)), Some(0.0));
        assert_eq!(temperature_for_wire(Some(1.0)), None);
        std::env::set_var(ENV_OMIT_TEMPERATURE, "true");
        assert_eq!(temperature_for_wire(Some(0.0)), None);
        clear_omit_env();
    }

    #[test]
    fn apply_omit_reasoning_effort_clears_when_env_set() {
        clear_omit_env();
        assert_eq!(
            apply_omit_reasoning_effort(Some("low".into())),
            Some("low".into())
        );
        std::env::set_var(ENV_OMIT_REASONING_EFFORT, "1");
        assert_eq!(apply_omit_reasoning_effort(Some("low".into())), None);
        clear_omit_env();
    }
}
