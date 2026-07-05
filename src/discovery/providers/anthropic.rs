//! Anthropic discovery — DYNAMIC strategy.
//!
//! `/v1/models` now returns rich capabilities including context length,
//! max_tokens, and a structured `capabilities` object.
//!
//! Source: <https://docs.anthropic.com/en/api/models>

use async_trait::async_trait;
use chrono::Utc;

use crate::discovery::registry::anthropic_models;
use crate::discovery::traits::ModelDiscoveryProvider;
use crate::discovery::types::{DiscoveredModel, DiscoverySource, DiscoveryStrategy};
use crate::model_config::{ModelCapabilities, ModelType};

pub struct AnthropicDiscovery {
    api_key: Option<String>,
}

impl Default for AnthropicDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl AnthropicDiscovery {
    pub fn new() -> Self {
        Self {
            api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
        }
    }

    pub fn with_api_key(api_key: String) -> Self {
        Self {
            api_key: Some(api_key),
        }
    }

    async fn fetch_from_api(&self) -> Option<Vec<DiscoveredModel>> {
        let key = self.api_key.as_ref()?;
        let client = reqwest::Client::new();
        let resp = client
            .get("https://api.anthropic.com/v1/models")
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
            .ok()?;

        if !resp.status().is_success() {
            tracing::warn!(
                status = %resp.status(),
                "Anthropic /v1/models returned non-200"
            );
            return None;
        }

        let body: serde_json::Value = resp.json().await.ok()?;
        let data = body["data"].as_array()?;

        let now = Utc::now();
        let models: Vec<DiscoveredModel> = data
            .iter()
            .filter_map(|m| {
                let id = m["id"].as_str()?.to_string();
                let display_name = m["display_name"].as_str().unwrap_or(&id).to_string();
                let context_length = m["max_input_tokens"].as_u64().unwrap_or(200_000) as usize;
                let max_output = m["max_tokens"].as_u64().unwrap_or(8_192) as usize;

                let caps = &m["capabilities"];
                let supports_vision = caps["image_input"].as_bool().unwrap_or(false);
                let supports_thinking = caps
                    .get("thinking")
                    .map(|t| {
                        t["adaptive"].as_bool().unwrap_or(false)
                            || t["enabled"].as_bool().unwrap_or(false)
                    })
                    .unwrap_or(false);

                Some(DiscoveredModel {
                    id,
                    name: display_name,
                    provider: "anthropic".into(),
                    context_length,
                    max_output_tokens: max_output,
                    capabilities: ModelCapabilities {
                        context_length,
                        max_output_tokens: max_output,
                        supports_vision,
                        supports_function_calling: true,
                        supports_json_mode: true,
                        supports_streaming: true,
                        supports_thinking,
                        supports_system_message: true,
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
impl ModelDiscoveryProvider for AnthropicDiscovery {
    fn provider_id(&self) -> &str {
        "anthropic"
    }

    fn discovery_strategy(&self) -> DiscoveryStrategy {
        DiscoveryStrategy::Dynamic
    }

    async fn discover_models(&self) -> crate::error::Result<Vec<DiscoveredModel>> {
        match self.fetch_from_api().await {
            Some(models) if !models.is_empty() => Ok(models),
            _ => {
                tracing::info!("Anthropic API unavailable, using static registry");
                Ok(anthropic_models())
            }
        }
    }
}
