//! MTPLX provider — native MTP speculative decoding on Apple Silicon.
//!
//! Thin wrapper over [`OpenAICompatibleProvider`] (same shape as oMLX).
//! Default host: `http://127.0.0.1:8000` (CLI docs); settings.port overrides.
//!
//! # Environment
//!
//! | Variable | Default | Description |
//! |----------|---------|-------------|
//! | `MTPLX_HOST` / `MTPLX_BASE_URL` | settings or `:8000` | Server base |
//! | `MTPLX_MODEL` | settings `model` / `default` | Chat model id |
//! | `MTPLX_API_KEY` | unset | Optional Bearer |
//! | `MTPLX_TIMEOUT_SECONDS` | `600` | HTTP timeout |
//!
//! Settings: `~/Library/Application Support/MTPLX/settings.json` (macOS)
//! or `$MTPLX_SETTINGS` / `$XDG_CONFIG_HOME/mtplx/settings.json`.

use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::Client;
use std::path::PathBuf;
use std::time::Duration;
use tracing::debug;

use crate::error::{LlmError, Result};
use crate::model_config::{ModelCapabilities, ModelCard, ModelType, ProviderConfig, ProviderType};
use crate::providers::local_openai_common::{
    fetch_openai_model_ids, model_id_from_path, normalize_local_openai_host,
};
use crate::providers::openai_compatible::OpenAICompatibleProvider;
use crate::traits::{
    ChatMessage, CompletionOptions, EmbeddingProvider, LLMProvider, LLMResponse, StreamChunk,
    ToolChoice, ToolDefinition,
};

/// CLI / docs default port when settings are absent.
pub const DEFAULT_MTPLX_PORT: u16 = 8000;

/// Default MTPLX listen address.
pub const DEFAULT_MTPLX_HOST: &str = "http://127.0.0.1:8000";

const DEFAULT_MTPLX_MODEL: &str = "default";
const DEFAULT_MTPLX_TIMEOUT_SECS: u64 = 600;
const DEFAULT_MTPLX_CONTEXT: usize = 131_072;
const DEFAULT_MTPLX_MAX_OUTPUT: usize = 4096;

/// Resolved runtime connection parameters for MTPLX.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtplxRuntimeConfig {
    pub host: String,
    pub api_key: Option<String>,
    pub default_model: Option<String>,
    pub context_window: Option<usize>,
}

/// Normalize host (strip `/v1`).
pub fn normalize_mtplx_host(host: &str) -> String {
    normalize_local_openai_host(host, DEFAULT_MTPLX_HOST)
}

/// Settings file candidates (first existing wins).
pub fn mtplx_settings_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(p) = std::env::var("MTPLX_SETTINGS") {
        let t = p.trim();
        if !t.is_empty() {
            paths.push(PathBuf::from(t));
        }
    }
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        // macOS app support
        paths.push(
            home.join("Library")
                .join("Application Support")
                .join("MTPLX")
                .join("settings.json"),
        );
        paths.push(home.join(".config").join("mtplx").join("settings.json"));
    }
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from) {
        paths.push(xdg.join("mtplx").join("settings.json"));
    }
    if let Some(home) = std::env::var_os("USERPROFILE").map(PathBuf::from) {
        paths.push(
            home.join("AppData")
                .join("Roaming")
                .join("MTPLX")
                .join("settings.json"),
        );
    }
    paths
}

/// Parsed settings file fields: `(host, api_key, default_model, context_window)`.
pub type MtplxSettingsFile = (String, Option<String>, Option<String>, Option<usize>);

/// Read host/port/model/api_key from MTPLX settings when present.
pub fn load_mtplx_settings_file() -> Option<MtplxSettingsFile> {
    let path = mtplx_settings_paths().into_iter().find(|p| p.is_file())?;
    let raw = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;

    // Flat schema (live app): host/port at root; nested server.* also accepted.
    let host = value
        .get("host")
        .and_then(|v| v.as_str())
        .or_else(|| value.pointer("/server/host").and_then(|v| v.as_str()))
        .unwrap_or("127.0.0.1");
    let port = value
        .get("port")
        .and_then(|v| v.as_u64())
        .or_else(|| value.pointer("/server/port").and_then(|v| v.as_u64()))
        .unwrap_or(u64::from(DEFAULT_MTPLX_PORT)) as u16;
    let scheme = if port == 443 { "https" } else { "http" };
    let base = format!("{scheme}://{host}:{port}");

    let api_key = value
        .get("api_key")
        .and_then(|v| v.as_str())
        .or_else(|| value.pointer("/auth/api_key").and_then(|v| v.as_str()))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);

    let default_model = value
        .get("model")
        .and_then(|v| v.as_str())
        .map(model_id_from_path)
        .filter(|s| !s.is_empty());

    let context_window = value
        .get("context_window")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .filter(|n| *n > 0);

    Some((base, api_key, default_model, context_window))
}

/// Resolve host + API key: env → settings.json → defaults.
pub fn resolve_mtplx_runtime_config() -> MtplxRuntimeConfig {
    let settings = load_mtplx_settings_file();

    let host = std::env::var("MTPLX_HOST")
        .or_else(|_| std::env::var("MTPLX_BASE_URL"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| settings.as_ref().map(|(h, _, _, _)| h.clone()))
        .unwrap_or_else(|| DEFAULT_MTPLX_HOST.to_string());

    let api_key = std::env::var("MTPLX_API_KEY")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| settings.as_ref().and_then(|(_, k, _, _)| k.clone()));

    let default_model = std::env::var("MTPLX_MODEL")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| settings.as_ref().and_then(|(_, _, m, _)| m.clone()));

    let context_window = settings.and_then(|(_, _, _, c)| c);

    MtplxRuntimeConfig {
        host: normalize_mtplx_host(&host),
        api_key,
        default_model,
        context_window,
    }
}

pub fn host_from_env() -> String {
    resolve_mtplx_runtime_config().host
}

pub fn api_key_from_env() -> Option<String> {
    resolve_mtplx_runtime_config().api_key
}

/// List model ids from `~/.mtplx/models` when the API is down (offline inventory).
pub fn list_cached_model_ids() -> Vec<String> {
    let dir = std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|h| h.join(".mtplx").join("models"));
    let Some(dir) = dir.filter(|d| d.is_dir()) else {
        return Vec::new();
    };
    let mut ids = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                let name = entry.file_name().to_string_lossy().to_string();
                if !name.starts_with('.') {
                    ids.push(model_id_from_path(&name));
                }
            }
        }
    }
    ids.sort();
    ids.dedup();
    ids
}

/// MTPLX local MTP server (OpenAI-compatible API).
#[derive(Debug)]
pub struct MtplxProvider {
    inner: OpenAICompatibleProvider,
    host: String,
    client: Client,
    timeout_seconds: u64,
    api_key: Option<String>,
}

#[derive(Debug, Clone)]
pub struct MtplxProviderBuilder {
    host: String,
    model: String,
    embedding_model: String,
    api_key: Option<String>,
    max_context_length: usize,
    timeout_seconds: u64,
}

impl Default for MtplxProviderBuilder {
    fn default() -> Self {
        Self {
            host: DEFAULT_MTPLX_HOST.to_string(),
            model: DEFAULT_MTPLX_MODEL.to_string(),
            embedding_model: DEFAULT_MTPLX_MODEL.to_string(),
            api_key: None,
            max_context_length: DEFAULT_MTPLX_CONTEXT,
            timeout_seconds: DEFAULT_MTPLX_TIMEOUT_SECS,
        }
    }
}

impl MtplxProviderBuilder {
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

    pub fn build(self) -> Result<MtplxProvider> {
        let host = normalize_mtplx_host(&self.host);
        let base_url = format!("{host}/v1");
        let api_key = self
            .api_key
            .or_else(|| std::env::var("MTPLX_API_KEY").ok())
            .filter(|k| !k.trim().is_empty())
            .unwrap_or_else(|| "mtplx".to_string());

        let config = ProviderConfig {
            name: "mtplx".to_string(),
            display_name: "MTPLX".to_string(),
            provider_type: ProviderType::OpenAICompatible,
            api_key_env: None,
            api_key: Some(api_key.clone()),
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

        Ok(MtplxProvider {
            inner,
            host,
            client,
            timeout_seconds: self.timeout_seconds,
            api_key: if api_key == "mtplx" {
                api_key_from_env()
            } else {
                Some(api_key)
            },
        })
    }
}

impl MtplxProvider {
    pub fn builder() -> MtplxProviderBuilder {
        MtplxProviderBuilder::new()
    }

    pub fn from_env() -> Result<Self> {
        let cfg = resolve_mtplx_runtime_config();
        let model = cfg
            .default_model
            .clone()
            .unwrap_or_else(|| DEFAULT_MTPLX_MODEL.to_string());
        let timeout_seconds = std::env::var("MTPLX_TIMEOUT_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MTPLX_TIMEOUT_SECS);
        let mut b = Self::builder()
            .host(cfg.host)
            .model(model.clone())
            .embedding_model(model)
            .timeout_seconds(timeout_seconds);
        if let Some(ctx) = cfg.context_window {
            b = b.max_context_length(ctx);
        }
        if let Some(key) = cfg.api_key {
            b = b.api_key(key);
        }
        b.build()
    }

    pub fn from_env_with_model(model: &str) -> Result<Self> {
        let cfg = resolve_mtplx_runtime_config();
        let timeout_seconds = std::env::var("MTPLX_TIMEOUT_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MTPLX_TIMEOUT_SECS);
        let mut b = Self::builder()
            .host(cfg.host)
            .model(model)
            .embedding_model(model)
            .timeout_seconds(timeout_seconds);
        if let Some(ctx) = cfg.context_window {
            b = b.max_context_length(ctx);
        }
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

    pub async fn health_check(&self) -> Result<()> {
        let models = self.list_models().await?;
        debug!(
            provider = "mtplx",
            host = %self.host,
            count = models.len(),
            "MTPLX health ok"
        );
        Ok(())
    }

    pub async fn list_models(&self) -> Result<Vec<String>> {
        let key_owned = self.api_key.clone().or_else(api_key_from_env);
        match fetch_openai_model_ids(&self.client, &self.host, key_owned.as_deref(), "MTPLX").await
        {
            Ok(ids) if !ids.is_empty() => Ok(ids),
            Ok(_) => {
                let cached = list_cached_model_ids();
                Ok(cached)
            }
            Err(e) => {
                let cached = list_cached_model_ids();
                if cached.is_empty() {
                    Err(e)
                } else {
                    debug!(
                        provider = "mtplx",
                        count = cached.len(),
                        "API list failed; using ~/.mtplx/models cache"
                    );
                    Ok(cached)
                }
            }
        }
    }
}

#[async_trait]
impl LLMProvider for MtplxProvider {
    fn name(&self) -> &str {
        "mtplx"
    }

    fn model(&self) -> &str {
        LLMProvider::model(&self.inner)
    }

    fn max_context_length(&self) -> usize {
        self.inner.max_context_length()
    }

    fn default_max_output_tokens(&self) -> Option<usize> {
        Some(DEFAULT_MTPLX_MAX_OUTPUT)
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
impl EmbeddingProvider for MtplxProvider {
    fn name(&self) -> &str {
        "mtplx"
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
    fn normalize_and_defaults() {
        assert_eq!(
            normalize_mtplx_host("http://127.0.0.1:8002/v1"),
            "http://127.0.0.1:8002"
        );
        assert!(DEFAULT_MTPLX_HOST.contains("8000"));
    }

    #[test]
    fn builder_name_is_mtplx() {
        let p = MtplxProvider::builder()
            .host("http://127.0.0.1:8002")
            .model("test-model")
            .build()
            .expect("build");
        assert_eq!(LLMProvider::name(&p), "mtplx");
        assert_eq!(LLMProvider::model(&p), "test-model");
        assert_eq!(p.host(), "http://127.0.0.1:8002");
    }

    #[test]
    fn settings_paths_include_app_support() {
        let paths = mtplx_settings_paths();
        assert!(paths.iter().any(|p| p.to_string_lossy().contains("MTPLX")));
    }
}
