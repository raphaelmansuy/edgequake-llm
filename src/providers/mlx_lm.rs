//! mlx-lm `mlx_lm.server` — official Apple MLX-LM OpenAI server.
//!
//! Default host: `http://127.0.0.1:8080` (collides with llama-server — use env / `/endpoint`).
//! Thin identity over [`crate::providers::local_openai_common::LocalOpenAiProvider`].
//!
//! # Environment
//!
//! | Variable | Default | Description |
//! |----------|---------|-------------|
//! | `MLX_LM_HOST` / `MLX_LM_BASE_URL` | `:8080` | Server base |
//! | `MLX_LM_MODEL` | `default` | Chat model id |
//! | `MLX_LM_API_KEY` | unset | Optional Bearer |
//! | `MLX_LM_TIMEOUT_SECONDS` | `600` | HTTP timeout |

use crate::error::Result;
use crate::providers::local_openai_common::{
    LocalOpenAiIdentity, LocalOpenAiProvider, LocalOpenAiProviderBuilder, LocalOpenAiRuntimeConfig,
};

pub const DEFAULT_MLX_LM_PORT: u16 = 8080;
pub const DEFAULT_MLX_LM_HOST: &str = "http://127.0.0.1:8080";

pub const MLX_LM_IDENTITY: LocalOpenAiIdentity = LocalOpenAiIdentity {
    id: "mlx-lm",
    display_name: "mlx-lm",
    default_host: DEFAULT_MLX_LM_HOST,
    host_envs: &["MLX_LM_HOST", "MLX_LM_BASE_URL", "MLXLM_HOST"],
    key_envs: &["MLX_LM_API_KEY", "MLXLM_API_KEY"],
    model_env: "MLX_LM_MODEL",
    embedding_model_env: "MLX_LM_EMBEDDING_MODEL",
    timeout_env: "MLX_LM_TIMEOUT_SECONDS",
    placeholder_key: "mlx-lm",
    default_model: "default",
    default_timeout_secs: 600,
    default_context: 128_000,
    max_output_tokens: 4096,
};

pub type MlxLmProvider = LocalOpenAiProvider;
pub type MlxLmProviderBuilder = LocalOpenAiProviderBuilder;
pub type MlxLmRuntimeConfig = LocalOpenAiRuntimeConfig;

pub fn normalize_mlx_lm_host(host: &str) -> String {
    crate::providers::local_openai_common::normalize_local_openai_host(host, DEFAULT_MLX_LM_HOST)
}

pub fn resolve_mlx_lm_runtime_config() -> LocalOpenAiRuntimeConfig {
    MLX_LM_IDENTITY.resolve_runtime()
}

pub fn host_from_env() -> String {
    MLX_LM_IDENTITY.host_from_env()
}

pub fn api_key_from_env() -> Option<String> {
    MLX_LM_IDENTITY.api_key_from_env()
}

pub fn from_env() -> Result<LocalOpenAiProvider> {
    LocalOpenAiProvider::from_env(MLX_LM_IDENTITY)
}

pub fn from_env_with_model(model: &str) -> Result<LocalOpenAiProvider> {
    LocalOpenAiProvider::from_env_with_model(MLX_LM_IDENTITY, model)
}

pub fn builder() -> LocalOpenAiProviderBuilder {
    LocalOpenAiProvider::builder(MLX_LM_IDENTITY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::LLMProvider;

    #[test]
    fn defaults_and_name() {
        assert!(DEFAULT_MLX_LM_HOST.contains("8080"));
        let p = builder()
            .model("mlx-community/Qwen")
            .build()
            .expect("build");
        assert_eq!(LLMProvider::name(&p), "mlx-lm");
        assert_eq!(p.host(), DEFAULT_MLX_LM_HOST);
    }
}
