//! Shared helpers + thin server type for local OpenAI-compatible products.
//!
//! DRY: host normalization, `/v1/models` parsing, and the full provider shell
//! live once here. Product modules (llamacpp, vllm-mlx, mlx-lm, …) supply only
//! a [`LocalOpenAiIdentity`] — identity, env keys, defaults.
//!
//! SOLID:
//! - **S**: identity = product metadata; this module = HTTP OpenAI shell
//! - **O**: new Mac/local servers = one identity + factory registration
//! - **D**: callers depend on [`LocalOpenAiProvider`], not product-specific HTTP

use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::Client;
use std::time::Duration;
use tracing::debug;

use crate::error::{LlmError, Result};
use crate::model_config::{ModelCapabilities, ModelCard, ModelType, ProviderConfig, ProviderType};
use crate::providers::openai_compatible::OpenAICompatibleProvider;
use crate::traits::{
    ChatMessage, CompletionOptions, EmbeddingProvider, LLMProvider, LLMResponse, StreamChunk,
    ToolChoice, ToolDefinition,
};

// ─── Pure helpers ───────────────────────────────────────────────────────────

/// Normalize host: trim, strip trailing `/v1` (callers pass either form).
pub fn normalize_local_openai_host(host: &str, default_host: &str) -> String {
    let trimmed = host.trim().trim_end_matches('/');
    let without_v1 = match trimmed.strip_suffix("/v1") {
        Some(base) => base.trim_end_matches('/'),
        None => trimmed,
    };
    if without_v1.is_empty() {
        default_host.to_string()
    } else {
        without_v1.to_string()
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

/// GET `{host}/v1/models` with optional Bearer key.
pub async fn fetch_openai_model_ids(
    client: &Client,
    host: &str,
    api_key: Option<&str>,
    provider_label: &str,
) -> Result<Vec<String>> {
    let url = format!("{}/v1/models", host.trim_end_matches('/'));
    let mut req = client.get(&url);
    if let Some(key) = api_key.filter(|k| !k.trim().is_empty()) {
        req = req.bearer_auth(key.trim());
    }
    let resp = req.send().await.map_err(|e| {
        LlmError::NetworkError(format!(
            "{provider_label} unreachable at {host} ({e}). Start the local server or set the host env var."
        ))
    })?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        let hint = if status.as_u16() == 401 {
            " API key required — set the provider API key env var or settings file."
        } else {
            ""
        };
        return Err(LlmError::ApiError(format!(
            "{provider_label} /v1/models returned {status}: {body}.{hint}"
        )));
    }
    let body: serde_json::Value = resp.json().await.map_err(|e| {
        LlmError::ApiError(format!("{provider_label} /v1/models invalid JSON: {e}"))
    })?;
    Ok(parse_openai_models_list(&body))
}

/// Basename of a model path for catalog defaults (`…/Youssofal--Qwen` → id-ish name).
pub fn model_id_from_path(path: &str) -> String {
    let name = path
        .trim()
        .trim_end_matches('/')
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(path)
        .trim();
    // MTPLX dirs often use `--` instead of `/` between org and model.
    name.replace("--", "/")
}

/// First non-empty env among candidates.
pub fn first_env(keys: &[&str]) -> Option<String> {
    for k in keys {
        if let Ok(v) = std::env::var(k) {
            let t = v.trim();
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }
    None
}

// ─── Identity (product metadata) ────────────────────────────────────────────

/// Static identity for a local OpenAI-compatible product server.
///
/// Product modules define one of these; [`LocalOpenAiProvider`] does the rest.
#[derive(Debug, Clone, Copy)]
pub struct LocalOpenAiIdentity {
    /// Canonical id (`llamacpp`, `vllm-mlx`, `mlx-lm`).
    pub id: &'static str,
    /// Human label for catalogs / doctor.
    pub display_name: &'static str,
    /// Default base URL without `/v1`.
    pub default_host: &'static str,
    /// Env vars for host (first set wins).
    pub host_envs: &'static [&'static str],
    /// Env vars for optional API key.
    pub key_envs: &'static [&'static str],
    /// Env var for model override.
    pub model_env: &'static str,
    /// Env var for embedding model (falls back to chat model).
    pub embedding_model_env: &'static str,
    /// Env var for HTTP timeout seconds.
    pub timeout_env: &'static str,
    /// Placeholder Bearer when no key configured (OpenAI clients often want non-empty).
    pub placeholder_key: &'static str,
    pub default_model: &'static str,
    pub default_timeout_secs: u64,
    pub default_context: usize,
    pub max_output_tokens: usize,
}

/// Resolved host + optional key for a local identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalOpenAiRuntimeConfig {
    pub host: String,
    pub api_key: Option<String>,
    pub default_model: Option<String>,
}

impl LocalOpenAiIdentity {
    /// Env → compiled default (no product settings files).
    pub fn resolve_runtime(&self) -> LocalOpenAiRuntimeConfig {
        let host = first_env(self.host_envs).unwrap_or_else(|| self.default_host.to_string());
        let api_key = first_env(self.key_envs);
        let default_model = std::env::var(self.model_env)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        LocalOpenAiRuntimeConfig {
            host: normalize_local_openai_host(&host, self.default_host),
            api_key,
            default_model,
        }
    }

    pub fn host_from_env(&self) -> String {
        self.resolve_runtime().host
    }

    pub fn api_key_from_env(&self) -> Option<String> {
        self.resolve_runtime().api_key
    }
}

// ─── Provider shell ─────────────────────────────────────────────────────────

/// Thin OpenAI-compatible local server provider parameterized by identity.
#[derive(Debug)]
pub struct LocalOpenAiProvider {
    identity: LocalOpenAiIdentity,
    inner: OpenAICompatibleProvider,
    host: String,
    client: Client,
    timeout_seconds: u64,
    api_key: Option<String>,
}

/// Builder for [`LocalOpenAiProvider`].
#[derive(Debug, Clone)]
pub struct LocalOpenAiProviderBuilder {
    identity: LocalOpenAiIdentity,
    host: String,
    model: String,
    embedding_model: String,
    api_key: Option<String>,
    max_context_length: usize,
    timeout_seconds: u64,
}

impl LocalOpenAiProviderBuilder {
    pub fn new(identity: LocalOpenAiIdentity) -> Self {
        Self {
            identity,
            host: identity.default_host.to_string(),
            model: identity.default_model.to_string(),
            embedding_model: identity.default_model.to_string(),
            api_key: None,
            max_context_length: identity.default_context,
            timeout_seconds: identity.default_timeout_secs,
        }
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

    pub fn build(self) -> Result<LocalOpenAiProvider> {
        let host = normalize_local_openai_host(&self.host, self.identity.default_host);
        let base_url = format!("{host}/v1");
        let env_key = first_env(self.identity.key_envs);
        let api_key = self
            .api_key
            .or(env_key)
            .filter(|k| !k.trim().is_empty())
            .unwrap_or_else(|| self.identity.placeholder_key.to_string());

        let config = ProviderConfig {
            name: self.identity.id.to_string(),
            display_name: self.identity.display_name.to_string(),
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

        let stored_key = if api_key == self.identity.placeholder_key {
            first_env(self.identity.key_envs)
        } else {
            Some(api_key)
        };

        Ok(LocalOpenAiProvider {
            identity: self.identity,
            inner,
            host,
            client,
            timeout_seconds: self.timeout_seconds,
            api_key: stored_key,
        })
    }
}

impl LocalOpenAiProvider {
    pub fn builder(identity: LocalOpenAiIdentity) -> LocalOpenAiProviderBuilder {
        LocalOpenAiProviderBuilder::new(identity)
    }

    pub fn from_env(identity: LocalOpenAiIdentity) -> Result<Self> {
        let cfg = identity.resolve_runtime();
        let model = cfg
            .default_model
            .clone()
            .unwrap_or_else(|| identity.default_model.to_string());
        let embedding_model = std::env::var(identity.embedding_model_env)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| model.clone());
        let timeout_seconds = std::env::var(identity.timeout_env)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(identity.default_timeout_secs);
        let mut b = Self::builder(identity)
            .host(cfg.host)
            .model(model)
            .embedding_model(embedding_model)
            .timeout_seconds(timeout_seconds);
        if let Some(key) = cfg.api_key {
            b = b.api_key(key);
        }
        b.build()
    }

    pub fn from_env_with_model(identity: LocalOpenAiIdentity, model: &str) -> Result<Self> {
        let cfg = identity.resolve_runtime();
        let timeout_seconds = std::env::var(identity.timeout_env)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(identity.default_timeout_secs);
        let mut b = Self::builder(identity)
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

    pub fn identity(&self) -> LocalOpenAiIdentity {
        self.identity
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
            provider = self.identity.id,
            host = %self.host,
            count = models.len(),
            "local OpenAI-compatible health ok"
        );
        Ok(())
    }

    pub async fn list_models(&self) -> Result<Vec<String>> {
        let key = self
            .api_key
            .clone()
            .or_else(|| first_env(self.identity.key_envs));
        fetch_openai_model_ids(
            &self.client,
            &self.host,
            key.as_deref(),
            self.identity.display_name,
        )
        .await
    }
}

#[async_trait]
impl LLMProvider for LocalOpenAiProvider {
    fn name(&self) -> &str {
        self.identity.id
    }

    fn model(&self) -> &str {
        LLMProvider::model(&self.inner)
    }

    fn max_context_length(&self) -> usize {
        self.inner.max_context_length()
    }

    fn default_max_output_tokens(&self) -> Option<usize> {
        Some(self.identity.max_output_tokens)
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
impl EmbeddingProvider for LocalOpenAiProvider {
    fn name(&self) -> &str {
        self.identity.id
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

    const TEST_ID: LocalOpenAiIdentity = LocalOpenAiIdentity {
        id: "testlocal",
        display_name: "TestLocal",
        default_host: "http://127.0.0.1:9999",
        host_envs: &["TESTLOCAL_HOST"],
        key_envs: &["TESTLOCAL_API_KEY"],
        model_env: "TESTLOCAL_MODEL",
        embedding_model_env: "TESTLOCAL_EMBEDDING_MODEL",
        timeout_env: "TESTLOCAL_TIMEOUT_SECONDS",
        placeholder_key: "testlocal",
        default_model: "default",
        default_timeout_secs: 600,
        default_context: 128_000,
        max_output_tokens: 4096,
    };

    #[test]
    fn normalize_strips_v1() {
        assert_eq!(
            normalize_local_openai_host("http://127.0.0.1:8000/v1/", "http://x"),
            "http://127.0.0.1:8000"
        );
        assert_eq!(
            normalize_local_openai_host("  ", "http://def"),
            "http://def"
        );
    }

    #[test]
    fn parse_models() {
        let body = serde_json::json!({
            "data": [
                {"id": "a"},
                {"id": "org/model"}
            ]
        });
        assert_eq!(parse_openai_models_list(&body), vec!["a", "org/model"]);
    }

    #[test]
    fn path_to_model_id() {
        assert_eq!(
            model_id_from_path("/Users/x/.mtplx/models/Youssofal--Qwen3.6-27B"),
            "Youssofal/Qwen3.6-27B"
        );
    }

    #[test]
    fn builder_uses_identity() {
        let p = LocalOpenAiProvider::builder(TEST_ID)
            .model("m1")
            .build()
            .expect("build");
        assert_eq!(LLMProvider::name(&p), "testlocal");
        assert_eq!(LLMProvider::model(&p), "m1");
        assert_eq!(p.host(), "http://127.0.0.1:9999");
    }
}
