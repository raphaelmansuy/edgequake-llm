//! SPEC-131 — effective temperature resolution (product + wire policy).
//!
//! Lives in edgequake-llm so all crates (pipeline, api, pdf) share one SSOT
//! without cyclic deps (LAW-131-3).

use crate::omit_env::omit_temperature_from_env;

/// Heuristic gate: families that reject non-default temperature overrides.
pub fn model_requires_default_temperature(model: &str) -> bool {
    let normalized = model
        .trim()
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .to_ascii_lowercase();

    normalized.contains("gpt-5")
        || normalized.contains("gpt-4.1-nano")
        || normalized.contains("gpt-4.1-mini")
        || normalized.starts_with("o1")
        || normalized.starts_with("o3")
        || normalized.starts_with("o4")
}

/// Returns preferred temperature unless the model requires the provider default.
pub fn effective_temperature_for_model(model: &str, preferred_temperature: f32) -> Option<f32> {
    if model_requires_default_temperature(model) {
        None
    } else {
        Some(preferred_temperature)
    }
}

/// Env omit supersedes the model heuristic gate (LAW-131-2).
pub fn resolve_effective_temperature(model: &str, preferred_temperature: f32) -> Option<f32> {
    if omit_temperature_from_env() {
        return None;
    }
    effective_temperature_for_model(model, preferred_temperature)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::omit_env::ENV_OMIT_TEMPERATURE;

    fn clear_omit() {
        std::env::remove_var(ENV_OMIT_TEMPERATURE);
    }

    #[test]
    fn omit_env_forces_none_for_gemma() {
        clear_omit();
        std::env::set_var(ENV_OMIT_TEMPERATURE, "true");
        assert_eq!(
            resolve_effective_temperature("google.gemma-4-31b", 0.0),
            None
        );
        clear_omit();
    }

    #[test]
    fn gemma_without_omit_sends_preferred() {
        clear_omit();
        assert_eq!(
            resolve_effective_temperature("google.gemma-4-31b", 0.0),
            Some(0.0)
        );
    }

    #[test]
    fn gpt5_gate_still_omits_without_env() {
        clear_omit();
        assert_eq!(resolve_effective_temperature("gpt-5-nano", 0.0), None);
    }
}
