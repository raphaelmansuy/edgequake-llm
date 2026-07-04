//! OpenAI-compatible discovery — TRY-DYNAMIC strategy.
//!
//! Attempts `GET {base_url}/models` for generic OpenAI-compatible endpoints
//! (Groq, Together, DeepSeek, custom). Falls back to empty list.

use async_trait::async_trait;
use chrono::Utc;

use crate::discovery::traits::ModelDiscoveryProvider;
use crate::discovery::types::{DiscoveredModel, DiscoverySource, DiscoveryStrategy};
use crate::model_config::{ModelCapabilities, ModelType};

pub struct OpenAICompatDiscovery {
    base_url: String,
    api_key: Option<String>,
    provider_name: String,
}

impl OpenAICompatDiscovery {
    pub fn new(base_url: impl Into<String>, provider_name: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: None,
            provider_name: provider_name.into(),
        }
    }

    pub fn with_api_key(mut self, key: String) -> Self {
        self.api_key = Some(key);
        self
    }

    async fn fetch_from_api(&self) -> Option<Vec<DiscoveredModel>> {
        let url = format!("{}/models", self.base_url.trim_end_matches('/'));
        let client = reqwest::Client::new();
        let mut req = client.get(&url).timeout(std::time::Duration::from_secs(10));
        if let Some(ref key) = self.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req.send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }

        let body: serde_json::Value = resp.json().await.ok()?;
        let data = body["data"].as_array()?;
        let now = Utc::now();

        let models: Vec<DiscoveredModel> = data
            .iter()
            .filter_map(|m| {
                let id = m["id"].as_str()?;
                if id.is_empty() {
                    return None;
                }
                let ctx = m["context_window"]
                    .as_u64()
                    .or_else(|| m["context_length"].as_u64())
                    .unwrap_or(0) as usize;

                Some(DiscoveredModel {
                    id: id.to_string(),
                    name: id.to_string(),
                    provider: self.provider_name.clone(),
                    context_length: ctx,
                    capabilities: ModelCapabilities {
                        context_length: ctx,
                        supports_streaming: true,
                        ..Default::default()
                    },
                    source: DiscoverySource::DynamicApi,
                    discovered_at: now,
                    available: true,
                    model_type: ModelType::Llm,
                    ..Default::default()
                })
            })
            .collect();
        Some(models)
    }
}

#[async_trait]
impl ModelDiscoveryProvider for OpenAICompatDiscovery {
    fn provider_id(&self) -> &str {
        &self.provider_name
    }

    fn discovery_strategy(&self) -> DiscoveryStrategy {
        DiscoveryStrategy::Dynamic
    }

    async fn discover_models(&self) -> crate::error::Result<Vec<DiscoveredModel>> {
        match self.fetch_from_api().await {
            Some(models) => Ok(models),
            None => {
                tracing::info!("OpenAI-compatible endpoint {} unavailable", self.base_url);
                Ok(Vec::new())
            }
        }
    }
}
