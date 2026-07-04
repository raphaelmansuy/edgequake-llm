//! OpenAI discovery — HYBRID strategy.
//!
//! `/v1/models` returns IDs but no capability metadata.
//! We use the static registry for capabilities and the API for availability.
//!
//! Source: https://platform.openai.com/docs/api-reference/models/list

use async_trait::async_trait;

use crate::discovery::registry::openai_models;
use crate::discovery::traits::ModelDiscoveryProvider;
use crate::discovery::types::{DiscoveredModel, DiscoverySource, DiscoveryStrategy};

pub struct OpenAIDiscovery {
    api_key: Option<String>,
}

impl Default for OpenAIDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAIDiscovery {
    pub fn new() -> Self {
        Self {
            api_key: std::env::var("OPENAI_API_KEY").ok(),
        }
    }

    pub fn with_api_key(api_key: String) -> Self {
        Self {
            api_key: Some(api_key),
        }
    }

    async fn fetch_model_ids(&self) -> Option<Vec<String>> {
        let key = self.api_key.as_ref()?;
        let client = reqwest::Client::new();
        let resp = client
            .get("https://api.openai.com/v1/models")
            .bearer_auth(key)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
            .ok()?;

        if !resp.status().is_success() {
            return None;
        }

        let body: serde_json::Value = resp.json().await.ok()?;
        let ids: Vec<String> = body["data"]
            .as_array()?
            .iter()
            .filter_map(|m| m["id"].as_str().map(String::from))
            .collect();
        Some(ids)
    }
}

#[async_trait]
impl ModelDiscoveryProvider for OpenAIDiscovery {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn discovery_strategy(&self) -> DiscoveryStrategy {
        DiscoveryStrategy::Hybrid
    }

    async fn discover_models(&self) -> crate::error::Result<Vec<DiscoveredModel>> {
        let mut registry = openai_models();
        let live_ids = self.fetch_model_ids().await.unwrap_or_default();

        if !live_ids.is_empty() {
            for model in &mut registry {
                model.available = live_ids.iter().any(|id| id == &model.id);
                model.source = DiscoverySource::Hybrid;
            }
        }
        Ok(registry)
    }
}
