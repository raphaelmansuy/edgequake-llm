//! NVIDIA NIM discovery — HYBRID strategy.
//!
//! Static catalog provides capability flags (vision, thinking);
//! dynamic `/v1/models` confirms availability and discovers new models.
//!
//! Source: https://docs.api.nvidia.com

use async_trait::async_trait;
use chrono::Utc;

use crate::discovery::traits::ModelDiscoveryProvider;
use crate::discovery::types::{DiscoveredModel, DiscoverySource, DiscoveryStrategy};
use crate::model_config::{ModelCapabilities, ModelType};

pub struct NvidiaDiscovery {
    api_key: Option<String>,
}

impl Default for NvidiaDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl NvidiaDiscovery {
    pub fn new() -> Self {
        Self {
            api_key: std::env::var("NVIDIA_API_KEY").ok(),
        }
    }

    async fn fetch_model_ids(&self) -> Option<Vec<String>> {
        let key = self.api_key.as_ref()?;
        let client = reqwest::Client::new();
        let resp = client
            .get("https://integrate.api.nvidia.com/v1/models")
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

    fn static_catalog() -> Vec<DiscoveredModel> {
        let now = Utc::now();
        vec![
            DiscoveredModel {
                id: "meta/llama-3.3-70b-instruct".into(),
                name: "Llama 3.3 70B Instruct".into(),
                provider: "nvidia".into(),
                context_length: 131_072,
                max_output_tokens: 4_096,
                capabilities: ModelCapabilities {
                    context_length: 131_072,
                    max_output_tokens: 4_096,
                    supports_function_calling: true,
                    supports_streaming: true,
                    supports_system_message: true,
                    ..Default::default()
                },
                source: DiscoverySource::StaticRegistry,
                discovered_at: now,
                available: true,
                model_type: ModelType::Llm,
                ..Default::default()
            },
            DiscoveredModel {
                id: "deepseek-ai/deepseek-r1".into(),
                name: "DeepSeek R1".into(),
                provider: "nvidia".into(),
                context_length: 65_536,
                max_output_tokens: 8_192,
                capabilities: ModelCapabilities {
                    context_length: 65_536,
                    max_output_tokens: 8_192,
                    supports_thinking: true,
                    supports_streaming: true,
                    supports_system_message: true,
                    ..Default::default()
                },
                source: DiscoverySource::StaticRegistry,
                discovered_at: now,
                available: true,
                model_type: ModelType::Llm,
                tags: vec!["reasoning".into()],
                ..Default::default()
            },
        ]
    }
}

#[async_trait]
impl ModelDiscoveryProvider for NvidiaDiscovery {
    fn provider_id(&self) -> &str {
        "nvidia"
    }

    fn discovery_strategy(&self) -> DiscoveryStrategy {
        DiscoveryStrategy::Hybrid
    }

    async fn discover_models(&self) -> crate::error::Result<Vec<DiscoveredModel>> {
        let mut catalog = Self::static_catalog();
        if let Some(live_ids) = self.fetch_model_ids().await {
            for model in &mut catalog {
                model.available = live_ids.iter().any(|id| id == &model.id);
                model.source = DiscoverySource::Hybrid;
            }
        }
        Ok(catalog)
    }
}
