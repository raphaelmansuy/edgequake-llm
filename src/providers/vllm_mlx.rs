//! vLLM-MLX — MLX-backed vLLM-style local server (OpenAI-compatible).
//!
//! Default host: `http://127.0.0.1:8000` (may collide with MTPLX — use env / `/endpoint`).
//! Thin identity over [`crate::providers::local_openai_common::LocalOpenAiProvider`].
//!
//! # Environment
//!
//! | Variable | Default | Description |
//! |----------|---------|-------------|
//! | `VLLM_MLX_HOST` / `VLLM_MLX_BASE_URL` | `:8000` | Server base |
//! | `VLLM_MLX_MODEL` | `default` | Chat model id |
//! | `VLLM_MLX_API_KEY` | unset | Optional Bearer |
//! | `VLLM_MLX_TIMEOUT_SECONDS` | `600` | HTTP timeout |

use crate::error::Result;
use crate::providers::local_openai_common::{
    LocalOpenAiIdentity, LocalOpenAiProvider, LocalOpenAiProviderBuilder, LocalOpenAiRuntimeConfig,
};

pub const DEFAULT_VLLM_MLX_PORT: u16 = 8000;
pub const DEFAULT_VLLM_MLX_HOST: &str = "http://127.0.0.1:8000";

pub const VLLM_MLX_IDENTITY: LocalOpenAiIdentity = LocalOpenAiIdentity {
    id: "vllm-mlx",
    display_name: "vLLM-MLX",
    default_host: DEFAULT_VLLM_MLX_HOST,
    host_envs: &["VLLM_MLX_HOST", "VLLM_MLX_BASE_URL"],
    key_envs: &["VLLM_MLX_API_KEY"],
    model_env: "VLLM_MLX_MODEL",
    embedding_model_env: "VLLM_MLX_EMBEDDING_MODEL",
    timeout_env: "VLLM_MLX_TIMEOUT_SECONDS",
    placeholder_key: "vllm-mlx",
    default_model: "default",
    default_timeout_secs: 600,
    default_context: 128_000,
    max_output_tokens: 4096,
};

pub type VllmMlxProvider = LocalOpenAiProvider;
pub type VllmMlxProviderBuilder = LocalOpenAiProviderBuilder;
pub type VllmMlxRuntimeConfig = LocalOpenAiRuntimeConfig;

pub fn normalize_vllm_mlx_host(host: &str) -> String {
    crate::providers::local_openai_common::normalize_local_openai_host(host, DEFAULT_VLLM_MLX_HOST)
}

pub fn resolve_vllm_mlx_runtime_config() -> LocalOpenAiRuntimeConfig {
    VLLM_MLX_IDENTITY.resolve_runtime()
}

pub fn host_from_env() -> String {
    VLLM_MLX_IDENTITY.host_from_env()
}

pub fn api_key_from_env() -> Option<String> {
    VLLM_MLX_IDENTITY.api_key_from_env()
}

pub fn from_env() -> Result<LocalOpenAiProvider> {
    LocalOpenAiProvider::from_env(VLLM_MLX_IDENTITY)
}

pub fn from_env_with_model(model: &str) -> Result<LocalOpenAiProvider> {
    LocalOpenAiProvider::from_env_with_model(VLLM_MLX_IDENTITY, model)
}

pub fn builder() -> LocalOpenAiProviderBuilder {
    LocalOpenAiProvider::builder(VLLM_MLX_IDENTITY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::LLMProvider;

    #[test]
    fn defaults_and_name() {
        assert!(DEFAULT_VLLM_MLX_HOST.contains("8000"));
        let p = builder().model("mlx-qwen").build().expect("build");
        assert_eq!(LLMProvider::name(&p), "vllm-mlx");
        assert_eq!(p.host(), DEFAULT_VLLM_MLX_HOST);
    }
}
