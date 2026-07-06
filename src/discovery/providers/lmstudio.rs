//! LM Studio discovery — DYNAMIC strategy.
//!
//! LM Studio 0.4.0+ native API returns structured capabilities
//! including vision, tool_use, reasoning, and context length.
//!
//! Source: <https://lmstudio.ai/docs/api>

use async_trait::async_trait;
use chrono::Utc;

use super::lmstudio_parse::parse_lmstudio_models_payload;
use crate::discovery::traits::ModelDiscoveryProvider;
use crate::discovery::types::{DiscoveredModel, DiscoveryStrategy};

pub struct LMStudioDiscovery {
    host: String,
}

impl Default for LMStudioDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl LMStudioDiscovery {
    pub fn new() -> Self {
        let host =
            std::env::var("LMSTUDIO_HOST").unwrap_or_else(|_| "http://localhost:1234".to_string());
        Self { host }
    }

    pub fn with_host(host: impl Into<String>) -> Self {
        Self { host: host.into() }
    }

    async fn fetch_url(&self, url: &str) -> Option<Vec<DiscoveredModel>> {
        let client = reqwest::Client::new();
        let resp = client
            .get(url)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
            .ok()?;

        if !resp.status().is_success() {
            return None;
        }

        let body: serde_json::Value = resp.json().await.ok()?;
        let models = parse_lmstudio_models_payload(&body, Utc::now());
        if models.is_empty() {
            None
        } else {
            Some(models)
        }
    }

    async fn fetch_from_api(&self) -> Option<Vec<DiscoveredModel>> {
        let base = self.host.trim_end_matches('/').trim_end_matches("/v1");

        if let Some(models) = self.fetch_url(&format!("{base}/api/v1/models")).await {
            return Some(models);
        }

        self.fetch_url(&format!("{base}/v1/models")).await
    }
}

#[async_trait]
impl ModelDiscoveryProvider for LMStudioDiscovery {
    fn provider_id(&self) -> &str {
        "lmstudio"
    }

    fn discovery_strategy(&self) -> DiscoveryStrategy {
        DiscoveryStrategy::Dynamic
    }

    async fn discover_models(&self) -> crate::error::Result<Vec<DiscoveredModel>> {
        match self.fetch_from_api().await {
            Some(models) => Ok(models),
            None => {
                tracing::info!(
                    "LM Studio unreachable or returned no models at {}, returning empty list",
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
            LMStudioDiscovery::new().discovery_strategy(),
            DiscoveryStrategy::Dynamic
        );
    }
}
