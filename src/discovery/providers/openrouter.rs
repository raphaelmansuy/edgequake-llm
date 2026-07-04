//! OpenRouter discovery — DYNAMIC strategy.
//!
//! OpenRouter's `/api/v1/models` is the richest discovery API of all providers,
//! returning context_length, pricing, modality, and supported_parameters.
//!
//! Source: https://openrouter.ai/docs/models

use async_trait::async_trait;
use chrono::Utc;

use crate::discovery::traits::ModelDiscoveryProvider;
use crate::discovery::types::{DiscoveredModel, DiscoverySource, DiscoveryStrategy};
use crate::model_config::{ModelCapabilities, ModelType};

pub struct OpenRouterDiscovery {
    api_key: Option<String>,
}

impl Default for OpenRouterDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenRouterDiscovery {
    pub fn new() -> Self {
        Self {
            api_key: std::env::var("OPENROUTER_API_KEY").ok(),
        }
    }

    async fn fetch_from_api(&self) -> Option<Vec<DiscoveredModel>> {
        let client = reqwest::Client::new();
        let mut req = client
            .get("https://openrouter.ai/api/v1/models")
            .timeout(std::time::Duration::from_secs(15));

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
                let name = m["name"].as_str().unwrap_or(id);
                let ctx = m["context_length"].as_u64().unwrap_or(0) as usize;
                let max_out = m["top_provider"]["max_completion_tokens"]
                    .as_u64()
                    .unwrap_or(4096) as usize;

                let modalities: Vec<String> = m["architecture"]["input_modalities"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                let supported_params: Vec<String> = m["supported_parameters"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                let supports_vision = modalities.iter().any(|m| m == "image");
                let supports_tools = supported_params.iter().any(|p| p == "tools");
                let supports_thinking = supported_params.iter().any(|p| p == "reasoning");

                let cost_in = m["pricing"]["prompt"]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .map(|p| p * 1_000_000.0);
                let cost_out = m["pricing"]["completion"]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .map(|p| p * 1_000_000.0);

                Some(DiscoveredModel {
                    id: id.to_string(),
                    name: name.to_string(),
                    provider: "openrouter".into(),
                    context_length: ctx,
                    max_output_tokens: max_out,
                    capabilities: ModelCapabilities {
                        context_length: ctx,
                        max_output_tokens: max_out,
                        supports_vision,
                        supports_function_calling: supports_tools,
                        supports_thinking,
                        supports_streaming: true,
                        supports_system_message: true,
                        ..Default::default()
                    },
                    source: DiscoverySource::DynamicApi,
                    discovered_at: now,
                    available: true,
                    cost_per_m_input: cost_in,
                    cost_per_m_output: cost_out,
                    model_type: ModelType::Llm,
                    ..Default::default()
                })
            })
            .collect();
        Some(models)
    }
}

#[async_trait]
impl ModelDiscoveryProvider for OpenRouterDiscovery {
    fn provider_id(&self) -> &str {
        "openrouter"
    }

    fn discovery_strategy(&self) -> DiscoveryStrategy {
        DiscoveryStrategy::Dynamic
    }

    async fn discover_models(&self) -> crate::error::Result<Vec<DiscoveredModel>> {
        match self.fetch_from_api().await {
            Some(models) => Ok(models),
            None => {
                tracing::info!("OpenRouter API unavailable, returning empty list");
                Ok(Vec::new())
            }
        }
    }
}
