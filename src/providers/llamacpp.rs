//! llama.cpp `llama-server` — Metal GGUF OpenAI-compatible API.
//!
//! Default host: `http://127.0.0.1:8080`. Thin identity over
//! [`crate::providers::local_openai_common::LocalOpenAiProvider`].
//!
//! # Environment
//!
//! | Variable | Default | Description |
//! |----------|---------|-------------|
//! | `LLAMACPP_HOST` / `LLAMA_SERVER_HOST` / `LLAMACPP_BASE_URL` | `:8080` | Server base |
//! | `LLAMACPP_MODEL` | `default` | Chat model id |
//! | `LLAMACPP_API_KEY` | unset | Optional Bearer |
//! | `LLAMACPP_TIMEOUT_SECONDS` | `600` | HTTP timeout |

use crate::error::Result;
use crate::providers::local_openai_common::{
    LocalOpenAiIdentity, LocalOpenAiProvider, LocalOpenAiProviderBuilder, LocalOpenAiRuntimeConfig,
};

/// Default llama-server listen port.
pub const DEFAULT_LLAMACPP_PORT: u16 = 8080;

/// Default llama-server base URL.
pub const DEFAULT_LLAMACPP_HOST: &str = "http://127.0.0.1:8080";

/// Static identity for llama-server.
pub const LLAMACPP_IDENTITY: LocalOpenAiIdentity = LocalOpenAiIdentity {
    id: "llamacpp",
    display_name: "llama-server",
    default_host: DEFAULT_LLAMACPP_HOST,
    host_envs: &[
        "LLAMACPP_HOST",
        "LLAMA_SERVER_HOST",
        "LLAMACPP_BASE_URL",
        "LLAMA_SERVER_BASE_URL",
    ],
    key_envs: &["LLAMACPP_API_KEY", "LLAMA_SERVER_API_KEY"],
    model_env: "LLAMACPP_MODEL",
    embedding_model_env: "LLAMACPP_EMBEDDING_MODEL",
    timeout_env: "LLAMACPP_TIMEOUT_SECONDS",
    placeholder_key: "llamacpp",
    default_model: "default",
    default_timeout_secs: 600,
    default_context: 128_000,
    max_output_tokens: 4096,
};

/// Type alias — same shell as other local OpenAI servers.
pub type LlamaCppProvider = LocalOpenAiProvider;
/// Builder alias.
pub type LlamaCppProviderBuilder = LocalOpenAiProviderBuilder;
/// Runtime config alias.
pub type LlamaCppRuntimeConfig = LocalOpenAiRuntimeConfig;

/// Normalize host (strip `/v1`).
pub fn normalize_llamacpp_host(host: &str) -> String {
    crate::providers::local_openai_common::normalize_local_openai_host(host, DEFAULT_LLAMACPP_HOST)
}

pub fn resolve_llamacpp_runtime_config() -> LocalOpenAiRuntimeConfig {
    LLAMACPP_IDENTITY.resolve_runtime()
}

pub fn host_from_env() -> String {
    LLAMACPP_IDENTITY.host_from_env()
}

pub fn api_key_from_env() -> Option<String> {
    LLAMACPP_IDENTITY.api_key_from_env()
}

impl LlamaCppProviderExt for LocalOpenAiProvider {}

/// Convenience constructors scoped to llamacpp identity.
pub trait LlamaCppProviderExt {
    fn llamacpp_builder() -> LocalOpenAiProviderBuilder {
        LocalOpenAiProvider::builder(LLAMACPP_IDENTITY)
    }

    fn llamacpp_from_env() -> Result<LocalOpenAiProvider> {
        LocalOpenAiProvider::from_env(LLAMACPP_IDENTITY)
    }

    fn llamacpp_from_env_with_model(model: &str) -> Result<LocalOpenAiProvider> {
        LocalOpenAiProvider::from_env_with_model(LLAMACPP_IDENTITY, model)
    }
}

/// Build a llama-server provider from env.
pub fn from_env() -> Result<LocalOpenAiProvider> {
    LocalOpenAiProvider::from_env(LLAMACPP_IDENTITY)
}

/// Build with explicit model id.
pub fn from_env_with_model(model: &str) -> Result<LocalOpenAiProvider> {
    LocalOpenAiProvider::from_env_with_model(LLAMACPP_IDENTITY, model)
}

/// New builder with llamacpp defaults.
pub fn builder() -> LocalOpenAiProviderBuilder {
    LocalOpenAiProvider::builder(LLAMACPP_IDENTITY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::LLMProvider;

    #[test]
    fn defaults_and_name() {
        assert!(DEFAULT_LLAMACPP_HOST.contains("8080"));
        let p = builder().model("gguf-model").build().expect("build");
        assert_eq!(LLMProvider::name(&p), "llamacpp");
        assert_eq!(LLMProvider::model(&p), "gguf-model");
        assert_eq!(p.host(), DEFAULT_LLAMACPP_HOST);
    }

    #[test]
    fn normalize_strips_v1() {
        assert_eq!(
            normalize_llamacpp_host("http://127.0.0.1:8080/v1"),
            "http://127.0.0.1:8080"
        );
    }
}
