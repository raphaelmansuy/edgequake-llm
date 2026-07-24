//! oMLX provider — Apple Silicon MLX inference (OpenAI-compatible).
//!
//! Thin wrapper over [`OpenAICompatibleProvider`] (DRY with LM Studio shape).
//! Default host: `http://127.0.0.1:9050` (oMLX menu-bar app default port).
//!
//! # Environment
//!
//! | Variable | Default | Description |
//! |----------|---------|-------------|
//! | `OMLX_HOST` / `OMLX_BASE_URL` | `http://127.0.0.1:9050` | Server base (with or without `/v1`) |
//! | `OMLX_MODEL` | `default` | Chat model id |
//! | `OMLX_API_KEY` | unset | Bearer token (also auto-read from `~/.omlx/settings.json`) |
//! | `OMLX_TIMEOUT_SECONDS` | `600` | HTTP timeout |
//! | `OMLX_EMBEDDING_MODEL` | same as chat | Embedding model id |
//!
//! When env vars are unset, host/port/api_key are resolved from
//! `~/.omlx/settings.json` (`server.port`, `auth.api_key`) so local menu-bar
//! installs work without extra configuration.

use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::Client;
use std::path::PathBuf;
use std::time::Duration;
use tracing::debug;

use crate::error::{LlmError, Result};
use crate::model_config::{ModelCapabilities, ModelCard, ModelType, ProviderConfig, ProviderType};
use crate::providers::openai_compatible::OpenAICompatibleProvider;
use crate::traits::{
    ChatMessage, CompletionOptions, EmbeddingProvider, LLMProvider, LLMResponse, StreamChunk,
    ToolChoice, ToolDefinition,
};

/// Default oMLX listen port (menu-bar app / Homebrew service).
pub const DEFAULT_OMLX_PORT: u16 = 9050;

/// Default oMLX listen address.
pub const DEFAULT_OMLX_HOST: &str = "http://127.0.0.1:9050";

const DEFAULT_OMLX_MODEL: &str = "default";
const DEFAULT_OMLX_TIMEOUT_SECS: u64 = 600;
const DEFAULT_OMLX_CONTEXT: usize = 128_000;
const DEFAULT_OMLX_MAX_OUTPUT: usize = 4096;

/// Resolved runtime connection parameters for oMLX.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OmlxRuntimeConfig {
    pub host: String,
    pub api_key: Option<String>,
    /// `model.default` / server default from settings when present.
    pub default_model: Option<String>,
}

/// Normalize host: trim whitespace and trailing `/v1` so callers can pass either form.
pub fn normalize_omlx_host(host: &str) -> String {
    let trimmed = host.trim().trim_end_matches('/');
    let without_v1 = match trimmed.strip_suffix("/v1") {
        Some(base) => base.trim_end_matches('/'),
        None => trimmed,
    };
    if without_v1.is_empty() {
        DEFAULT_OMLX_HOST.to_string()
    } else {
        without_v1.to_string()
    }
}

/// Path to the oMLX app settings file (`~/.omlx/settings.json`).
pub fn omlx_settings_path() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
        .map(|home| home.join(".omlx").join("settings.json"))
}

/// Read host/port/api_key from `~/.omlx/settings.json` when present.
pub fn load_omlx_settings_file() -> Option<(String, Option<String>, Option<String>)> {
    let path = omlx_settings_path()?;
    let raw = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;

    let host = value
        .pointer("/server/host")
        .and_then(|v| v.as_str())
        .unwrap_or("127.0.0.1");
    let port = value
        .pointer("/server/port")
        .and_then(|v| v.as_u64())
        .unwrap_or(u64::from(DEFAULT_OMLX_PORT)) as u16;
    let scheme = if port == 443 { "https" } else { "http" };
    let base = format!("{scheme}://{host}:{port}");

    let api_key = value
        .pointer("/auth/api_key")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);

    let default_model = value
        .pointer("/model/default_model")
        .or_else(|| value.pointer("/model/default"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);

    Some((base, api_key, default_model))
}

/// Resolve host + API key: env → `~/.omlx/settings.json` → compiled defaults.
pub fn resolve_omlx_runtime_config() -> OmlxRuntimeConfig {
    let settings = load_omlx_settings_file();

    let host = std::env::var("OMLX_HOST")
        .or_else(|_| std::env::var("OMLX_BASE_URL"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| settings.as_ref().map(|(h, _, _)| h.clone()))
        .unwrap_or_else(|| DEFAULT_OMLX_HOST.to_string());

    let api_key = std::env::var("OMLX_API_KEY")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| settings.as_ref().and_then(|(_, k, _)| k.clone()));

    let default_model = std::env::var("OMLX_MODEL")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| settings.and_then(|(_, _, m)| m));

    OmlxRuntimeConfig {
        host: normalize_omlx_host(&host),
        api_key,
        default_model,
    }
}

/// Resolve host from env / settings / default.
pub fn host_from_env() -> String {
    resolve_omlx_runtime_config().host
}

/// Resolve API key from env or `~/.omlx/settings.json` (never logged).
pub fn api_key_from_env() -> Option<String> {
    resolve_omlx_runtime_config().api_key
}

/// oMLX local MLX server (OpenAI-compatible API).
#[derive(Debug)]
pub struct OmlxProvider {
    inner: OpenAICompatibleProvider,
    host: String,
    client: Client,
    timeout_seconds: u64,
}

/// Builder for [`OmlxProvider`].
#[derive(Debug, Clone)]
pub struct OmlxProviderBuilder {
    host: String,
    model: String,
    embedding_model: String,
    api_key: Option<String>,
    max_context_length: usize,
    timeout_seconds: u64,
}

impl Default for OmlxProviderBuilder {
    fn default() -> Self {
        Self {
            host: DEFAULT_OMLX_HOST.to_string(),
            model: DEFAULT_OMLX_MODEL.to_string(),
            embedding_model: DEFAULT_OMLX_MODEL.to_string(),
            api_key: None,
            max_context_length: DEFAULT_OMLX_CONTEXT,
            timeout_seconds: DEFAULT_OMLX_TIMEOUT_SECS,
        }
    }
}

impl OmlxProviderBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn embedding_model(mut self, model: impl Into<String>) -> Self {
        self.embedding_model = model.into();
        self
    }

    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        let key = key.into();
        if key.trim().is_empty() {
            self.api_key = None;
        } else {
            self.api_key = Some(key);
        }
        self
    }

    pub fn max_context_length(mut self, length: usize) -> Self {
        self.max_context_length = length.max(1024);
        self
    }

    pub fn timeout_seconds(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds.max(30);
        self
    }

    pub fn build(self) -> Result<OmlxProvider> {
        let host = normalize_omlx_host(&self.host);
        let base_url = format!("{host}/v1");
        // oMLX may require an API key; OpenAI clients often want a non-empty placeholder.
        let api_key = self
            .api_key
            .or_else(|| std::env::var("OMLX_API_KEY").ok())
            .filter(|k| !k.trim().is_empty())
            .unwrap_or_else(|| "omlx".to_string());

        let config = ProviderConfig {
            name: "omlx".to_string(),
            display_name: "oMLX".to_string(),
            provider_type: ProviderType::OpenAICompatible,
            api_key_env: None,
            api_key: Some(api_key),
            base_url: Some(base_url),
            default_llm_model: Some(self.model.clone()),
            default_embedding_model: Some(self.embedding_model.clone()),
            timeout_seconds: self.timeout_seconds,
            models: vec![
                ModelCard {
                    name: self.model.clone(),
                    display_name: self.model.clone(),
                    model_type: ModelType::Llm,
                    capabilities: ModelCapabilities {
                        context_length: self.max_context_length,
                        supports_function_calling: true,
                        supports_streaming: true,
                        supports_json_mode: false,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ModelCard {
                    name: self.embedding_model.clone(),
                    display_name: self.embedding_model.clone(),
                    model_type: ModelType::Embedding,
                    capabilities: ModelCapabilities {
                        context_length: 8_192,
                        embedding_dimension: 768,
                        max_embedding_tokens: 8_192,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let inner = OpenAICompatibleProvider::from_config(config)?;
        let client = Client::builder()
            .timeout(Duration::from_secs(self.timeout_seconds))
            .no_proxy()
            .build()
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        Ok(OmlxProvider {
            inner,
            host,
            client,
            timeout_seconds: self.timeout_seconds,
        })
    }
}

impl OmlxProvider {
    pub fn builder() -> OmlxProviderBuilder {
        OmlxProviderBuilder::new()
    }

    pub fn from_env() -> Result<Self> {
        let cfg = resolve_omlx_runtime_config();
        let model = cfg
            .default_model
            .clone()
            .unwrap_or_else(|| DEFAULT_OMLX_MODEL.to_string());
        let embedding_model =
            std::env::var("OMLX_EMBEDDING_MODEL").unwrap_or_else(|_| model.clone());
        let timeout_seconds = std::env::var("OMLX_TIMEOUT_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_OMLX_TIMEOUT_SECS);
        let mut b = Self::builder()
            .host(cfg.host)
            .model(model)
            .embedding_model(embedding_model)
            .timeout_seconds(timeout_seconds);
        if let Some(key) = cfg.api_key {
            b = b.api_key(key);
        }
        b.build()
    }

    pub fn from_env_with_model(model: &str) -> Result<Self> {
        let cfg = resolve_omlx_runtime_config();
        let timeout_seconds = std::env::var("OMLX_TIMEOUT_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_OMLX_TIMEOUT_SECS);
        let mut b = Self::builder()
            .host(cfg.host)
            .model(model)
            .embedding_model(model)
            .timeout_seconds(timeout_seconds);
        if let Some(key) = cfg.api_key {
            b = b.api_key(key);
        }
        b.build()
    }

    pub fn with_application_context(
        mut self,
        ctx: crate::application_context::ApplicationContext,
    ) -> Self {
        self.inner = self.inner.with_application_context(ctx);
        self
    }

    pub fn host(&self) -> &str {
        &self.host
    }

    pub fn http_timeout_seconds(&self) -> u64 {
        self.timeout_seconds
    }

    /// GET `{host}/v1/models` — reachability + inventory.
    pub async fn health_check(&self) -> Result<()> {
        let models = self.list_models().await?;
        debug!(provider = "omlx", host = %self.host, count = models.len(), "oMLX health ok");
        Ok(())
    }

    /// List model ids from `GET /v1/models`.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/v1/models", self.host);
        let mut req = self.client.get(&url);
        // Prefer runtime key from env/settings (same path as chat).
        if let Some(key) = api_key_from_env() {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await.map_err(|e| {
            LlmError::NetworkError(format!(
                "oMLX unreachable at {} ({e}). Start oMLX (`omlx start`) or set OMLX_HOST.",
                self.host
            ))
        })?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            let hint = if status.as_u16() == 401 {
                " Set OMLX_API_KEY or ensure ~/.omlx/settings.json has auth.api_key."
            } else {
                ""
            };
            return Err(LlmError::ApiError(format!(
                "oMLX /v1/models returned {status}: {body}.{hint}"
            )));
        }
        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| LlmError::ApiError(format!("oMLX /v1/models invalid JSON: {e}")))?;
        Ok(parse_openai_models_list(&body))
    }
}

/// Parse OpenAI-shaped `{ "data": [ { "id": "..." } ] }`.
pub fn parse_openai_models_list(body: &serde_json::Value) -> Vec<String> {
    body.get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|id| id.as_str()).map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

#[async_trait]
impl LLMProvider for OmlxProvider {
    fn name(&self) -> &str {
        "omlx"
    }

    fn model(&self) -> &str {
        LLMProvider::model(&self.inner)
    }

    fn max_context_length(&self) -> usize {
        self.inner.max_context_length()
    }

    fn default_max_output_tokens(&self) -> Option<usize> {
        Some(DEFAULT_OMLX_MAX_OUTPUT)
    }

    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        self.inner.complete(prompt).await
    }

    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<LLMResponse> {
        self.inner.complete_with_options(prompt, options).await
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        self.inner.chat(messages, options).await
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: Option<ToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        self.inner
            .chat_with_tools(messages, tools, tool_choice, options)
            .await
    }

    fn supports_streaming(&self) -> bool {
        self.inner.supports_streaming()
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        false
    }

    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        self.inner.stream(prompt).await
    }

    fn supports_tool_streaming(&self) -> bool {
        self.inner.supports_tool_streaming()
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
}

#[async_trait]
impl EmbeddingProvider for OmlxProvider {
    fn name(&self) -> &str {
        "omlx"
    }

    fn model(&self) -> &str {
        EmbeddingProvider::model(&self.inner)
    }

    fn dimension(&self) -> usize {
        EmbeddingProvider::dimension(&self.inner)
    }

    fn max_tokens(&self) -> usize {
        EmbeddingProvider::max_tokens(&self.inner)
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.inner.embed(texts).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_v1_and_slash() {
        assert_eq!(
            normalize_omlx_host("http://127.0.0.1:9050/v1/"),
            "http://127.0.0.1:9050"
        );
        assert_eq!(
            normalize_omlx_host("http://127.0.0.1:9050"),
            "http://127.0.0.1:9050"
        );
        assert_eq!(normalize_omlx_host("  "), DEFAULT_OMLX_HOST);
        assert!(DEFAULT_OMLX_HOST.contains("9050"));
    }

    #[test]
    fn builder_defaults_to_9050() {
        let p = OmlxProvider::builder().model("x").build().expect("build");
        assert_eq!(p.host(), "http://127.0.0.1:9050");
    }

    #[test]
    fn parse_models_keeps_profiles() {
        let body = serde_json::json!({
            "object": "list",
            "data": [
                {"id": "qwen3-8b", "object": "model"},
                {"id": "qwen3-8b:thinking", "object": "model"},
                {"id": "mlx-community/Qwen-4bit", "object": "model"}
            ]
        });
        let ids = parse_openai_models_list(&body);
        assert_eq!(ids.len(), 3);
        assert!(ids.iter().any(|id| id == "qwen3-8b:thinking"));
        assert!(ids.iter().any(|id| id == "mlx-community/Qwen-4bit"));
    }

    #[test]
    fn builder_name_is_omlx() {
        let p = OmlxProvider::builder()
            .host("http://127.0.0.1:9050")
            .model("test-model")
            .build()
            .expect("build");
        assert_eq!(LLMProvider::name(&p), "omlx");
        assert_eq!(LLMProvider::model(&p), "test-model");
        assert_eq!(p.host(), "http://127.0.0.1:9050");
    }
}
