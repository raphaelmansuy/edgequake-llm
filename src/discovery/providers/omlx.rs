//! oMLX discovery — DYNAMIC strategy via OpenAI-compatible `/v1/models`.
//!
//! Resolves host + API key from env or `~/.omlx/settings.json` so menu-bar
//! installs work without exporting `OMLX_*` (API key is often required).

use async_trait::async_trait;
use chrono::Utc;

use crate::discovery::traits::ModelDiscoveryProvider;
use crate::discovery::types::{DiscoveredModel, DiscoverySource, DiscoveryStrategy};
use crate::model_config::{ModelCapabilities, ModelType};
use crate::providers::omlx::{
    parse_openai_models_list, resolve_omlx_runtime_config, DEFAULT_OMLX_HOST,
};

pub struct OmlxDiscovery {
    host: String,
    api_key: Option<String>,
}

impl Default for OmlxDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl OmlxDiscovery {
    pub fn new() -> Self {
        let cfg = resolve_omlx_runtime_config();
        Self {
            host: cfg.host,
            api_key: cfg.api_key,
        }
    }

    pub fn with_host(host: impl Into<String>) -> Self {
        let cfg = resolve_omlx_runtime_config();
        Self {
            host: crate::providers::omlx::normalize_omlx_host(&host.into()),
            api_key: cfg.api_key,
        }
    }

    async fn fetch_from_api(&self) -> Option<Vec<DiscoveredModel>> {
        let url = format!("{}/v1/models", self.host);
        let client = reqwest::Client::new();
        let mut req = client.get(&url).timeout(std::time::Duration::from_secs(5));
        if let Some(key) = self.api_key.as_deref() {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await.ok()?;
        if !resp.status().is_success() {
            tracing::debug!(
                status = %resp.status(),
                host = %self.host,
                has_key = self.api_key.is_some(),
                "oMLX /v1/models not successful"
            );
            return None;
        }
        let body: serde_json::Value = resp.json().await.ok()?;
        let now = Utc::now();
        let models = parse_openai_models_list(&body)
            .into_iter()
            .map(|id| DiscoveredModel {
                id: id.clone(),
                name: id,
                provider: "omlx".into(),
                context_length: 0,
                max_output_tokens: 0,
                capabilities: ModelCapabilities {
                    supports_function_calling: true,
                    supports_streaming: true,
                    supports_system_message: true,
                    ..Default::default()
                },
                source: DiscoverySource::DynamicApi,
                discovered_at: now,
                available: true,
                model_type: ModelType::Llm,
                ..Default::default()
            })
            .collect::<Vec<_>>();
        if models.is_empty() {
            None
        } else {
            Some(models)
        }
    }
}

#[async_trait]
impl ModelDiscoveryProvider for OmlxDiscovery {
    fn provider_id(&self) -> &str {
        "omlx"
    }

    fn discovery_strategy(&self) -> DiscoveryStrategy {
        DiscoveryStrategy::Dynamic
    }

    async fn discover_models(&self) -> crate::error::Result<Vec<DiscoveredModel>> {
        match self.fetch_from_api().await {
            Some(models) => Ok(models),
            None => {
                tracing::info!(
                    "oMLX unreachable or returned no models at {} (default {DEFAULT_OMLX_HOST}); empty list",
                    self.host
                );
                Ok(Vec::new())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_strategy_is_dynamic() {
        assert_eq!(
            OmlxDiscovery::new().discovery_strategy(),
            DiscoveryStrategy::Dynamic
        );
        assert_eq!(OmlxDiscovery::new().provider_id(), "omlx");
    }
}
