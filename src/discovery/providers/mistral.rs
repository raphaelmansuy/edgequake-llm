//! Mistral discovery — DYNAMIC strategy.
//!
//! `/v1/models` returns structured capabilities including
//! function_calling, vision, context length, and aliases.
//!
//! Source: <https://docs.mistral.ai/api/#tag/models>

use async_trait::async_trait;
use chrono::Utc;

use crate::discovery::registry::mistral_models;
use crate::discovery::traits::ModelDiscoveryProvider;
use crate::discovery::types::{DiscoveredModel, DiscoverySource, DiscoveryStrategy};
use crate::model_config::{ModelCapabilities, ModelType};

pub struct MistralDiscovery {
    api_key: Option<String>,
}

impl Default for MistralDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl MistralDiscovery {
    pub fn new() -> Self {
        Self {
            api_key: std::env::var("MISTRAL_API_KEY").ok(),
        }
    }

    async fn fetch_from_api(&self) -> Option<Vec<DiscoveredModel>> {
        let key = self.api_key.as_ref()?;
        let client = reqwest::Client::new();
        let resp = client
            .get("https://api.mistral.ai/v1/models")
            .bearer_auth(key)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
            .ok()?;

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
                let ctx = m["max_context_length"].as_u64().unwrap_or(0) as usize;
                let caps = &m["capabilities"];
                let vision = caps["vision"].as_bool().unwrap_or(false);
                let tools = caps["function_calling"].as_bool().unwrap_or(false);
                let is_chat = caps["completion_chat"].as_bool().unwrap_or(true);
                let is_embed = !is_chat;

                let deprecated = m["deprecation"].is_string() || m["deprecation"].is_object();

                Some(DiscoveredModel {
                    id: id.to_string(),
                    name: id.to_string(),
                    provider: "mistral".into(),
                    context_length: ctx,
                    max_output_tokens: 0,
                    capabilities: ModelCapabilities {
                        context_length: ctx,
                        supports_vision: vision,
                        supports_function_calling: tools,
                        supports_streaming: true,
                        supports_system_message: true,
                        ..Default::default()
                    },
                    source: DiscoverySource::DynamicApi,
                    discovered_at: now,
                    available: true,
                    model_type: if is_embed {
                        ModelType::Embedding
                    } else {
                        ModelType::Llm
                    },
                    deprecated,
                    ..Default::default()
                })
            })
            .collect();
        Some(models)
    }
}

#[async_trait]
impl ModelDiscoveryProvider for MistralDiscovery {
    fn provider_id(&self) -> &str {
        "mistral"
    }

    fn discovery_strategy(&self) -> DiscoveryStrategy {
        DiscoveryStrategy::Dynamic
    }

    async fn discover_models(&self) -> crate::error::Result<Vec<DiscoveredModel>> {
        match self.fetch_from_api().await {
            Some(models) if !models.is_empty() => Ok(models),
            _ => {
                tracing::info!("Mistral API unavailable, using static registry");
                Ok(mistral_models())
            }
        }
    }
}
