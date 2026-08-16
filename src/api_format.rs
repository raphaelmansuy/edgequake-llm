//! SPEC-131 — upstream LLM HTTP API format selection.
//!
//! Transport is configuration, not model identity (LAW-131-5).

use crate::error::{LlmError, Result};

/// Environment variable selecting Chat Completions vs Responses transport.
pub const ENV_API_FORMAT: &str = "EDGEQUAKE_LLM_API_FORMAT";

/// Upstream OpenAI-compatible request shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApiFormat {
    /// `POST …/chat/completions` (product default).
    #[default]
    ChatCompletions,
    /// `POST …/responses` (Bedrock Mantle GPT-5.6, Open Responses).
    Responses,
}

impl ApiFormat {
    /// Parse a format string (`chat_completions` | `responses`).
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "chat_completions" | "chat" | "chat-completions" => Ok(Self::ChatCompletions),
            "responses" | "response" => Ok(Self::Responses),
            other => Err(LlmError::ConfigError(format!(
                "Invalid {ENV_API_FORMAT}={other:?}; expected chat_completions|responses"
            ))),
        }
    }

    /// Read from env; unset → ChatCompletions. Invalid → error (fail loud).
    pub fn from_env() -> Result<Self> {
        match std::env::var(ENV_API_FORMAT) {
            Ok(v) if !v.trim().is_empty() => Self::parse(&v),
            _ => Ok(Self::ChatCompletions),
        }
    }

    /// Convenience: true when Responses transport is selected.
    pub fn is_responses(self) -> bool {
        matches!(self, Self::Responses)
    }
}

/// True when env selects Responses (invalid values treated as false for hot paths
/// that already validated at factory boot — prefer [`ApiFormat::from_env`] at start).
pub fn responses_api_from_env() -> bool {
    ApiFormat::from_env()
        .map(|f| f.is_responses())
        .unwrap_or(false)
}

/// Helper used by tests / Acc: treat empty as chat without error.
#[allow(dead_code)]
pub fn api_format_or_default() -> ApiFormat {
    ApiFormat::from_env().unwrap_or_default()
}

/// Unused placeholder removed — Responses selection uses [`ApiFormat::from_env`].
#[cfg(test)]
mod _pin {
    #[test]
    fn api_format_default_is_chat() {
        std::env::remove_var(super::ENV_API_FORMAT);
        assert_eq!(
            super::ApiFormat::from_env().unwrap(),
            super::ApiFormat::ChatCompletions
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults_and_aliases() {
        assert_eq!(
            ApiFormat::parse("chat_completions").unwrap(),
            ApiFormat::ChatCompletions
        );
        assert_eq!(ApiFormat::parse("responses").unwrap(), ApiFormat::Responses);
        assert!(ApiFormat::parse("foo").is_err());
    }

    #[test]
    fn from_env_default_chat() {
        std::env::remove_var(ENV_API_FORMAT);
        assert_eq!(ApiFormat::from_env().unwrap(), ApiFormat::ChatCompletions);
    }

    #[test]
    fn from_env_invalid_errors() {
        std::env::set_var(ENV_API_FORMAT, "not-a-format");
        assert!(ApiFormat::from_env().is_err());
        std::env::remove_var(ENV_API_FORMAT);
    }
}
