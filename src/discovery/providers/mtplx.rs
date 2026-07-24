//! MTPLX discovery — DYNAMIC strategy via OpenAI-compatible `/v1/models`.
//!
//! Falls back to `~/.mtplx/models` directory names when the API is offline
//! (labeled as available but source remains Dynamic when API succeeds).

use async_trait::async_trait;
use chrono::Utc;

use crate::discovery::traits::ModelDiscoveryProvider;
use crate::discovery::types::{DiscoveredModel, DiscoverySource, DiscoveryStrategy};
use crate::model_config::{ModelCapabilities, ModelType};
use crate::providers::local_openai_common::fetch_openai_model_ids;
use crate::providers::mtplx::{
    list_cached_model_ids, resolve_mtplx_runtime_config, DEFAULT_MTPLX_HOST,
};

pub struct MtplxDiscovery {
    host: String,
    api_key: Option<String>,
}

impl Default for MtplxDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl MtplxDiscovery {
    pub fn new() -> Self {
        let cfg = resolve_mtplx_runtime_config();
        Self {
            host: cfg.host,
            api_key: cfg.api_key,
        }
    }

    pub fn with_host(host: impl Into<String>) -> Self {
        let cfg = resolve_mtplx_runtime_config();
        Self {
            host: crate::providers::mtplx::normalize_mtplx_host(&host.into()),
            api_key: cfg.api_key,
        }
    }

    async fn fetch_live(&self) -> Option<Vec<String>> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .ok()?;
        fetch_openai_model_ids(&client, &self.host, self.api_key.as_deref(), "MTPLX")
            .await
            .ok()
            .filter(|ids| !ids.is_empty())
    }
}

#[async_trait]
impl ModelDiscoveryProvider for MtplxDiscovery {
    fn provider_id(&self) -> &str {
        "mtplx"
    }

    fn discovery_strategy(&self) -> DiscoveryStrategy {
        DiscoveryStrategy::Dynamic
    }

    async fn discover_models(&self) -> crate::error::Result<Vec<DiscoveredModel>> {
        let now = Utc::now();
        let (ids, source) = if let Some(live) = self.fetch_live().await {
            (live, DiscoverySource::DynamicApi)
        } else {
            let cached = list_cached_model_ids();
            if cached.is_empty() {
                tracing::info!(
                    "MTPLX unreachable at {} (default {DEFAULT_MTPLX_HOST}); empty list",
                    self.host
                );
                return Ok(Vec::new());
            }
            tracing::info!(
                count = cached.len(),
                "MTPLX API offline; using ~/.mtplx/models cache"
            );
            (cached, DiscoverySource::UserConfig)
        };

        Ok(ids
            .into_iter()
            .map(|id| DiscoveredModel {
                id: id.clone(),
                name: id,
                provider: "mtplx".into(),
                context_length: 0,
                max_output_tokens: 0,
                capabilities: ModelCapabilities {
                    supports_function_calling: true,
                    supports_streaming: true,
                    supports_system_message: true,
                    ..Default::default()
                },
                source: source.clone(),
                discovered_at: now,
                available: true,
                model_type: ModelType::Llm,
                ..Default::default()
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_strategy_is_dynamic() {
        assert_eq!(
            MtplxDiscovery::new().discovery_strategy(),
            DiscoveryStrategy::Dynamic
        );
        assert_eq!(MtplxDiscovery::new().provider_id(), "mtplx");
    }
}
