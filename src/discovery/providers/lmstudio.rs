//! LM Studio discovery — DYNAMIC strategy.
//!
//! LM Studio 0.4.0+ native API returns structured capabilities
//! including vision, tool_use, reasoning, and context length.
//!
//! Source: https://lmstudio.ai/docs/api

use async_trait::async_trait;
use chrono::Utc;

use crate::discovery::traits::ModelDiscoveryProvider;
use crate::discovery::types::{DiscoveredModel, DiscoverySource, DiscoveryStrategy};
use crate::model_config::{ModelCapabilities, ModelType};

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

    async fn fetch_from_api(&self) -> Option<Vec<DiscoveredModel>> {
        let base = self.host.trim_end_matches('/');
        let url = format!("{}/api/v1/models", base);
        let client = reqwest::Client::new();
        let resp = client
            .get(&url)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
            .ok()?;

        if !resp.status().is_success() {
            return None;
        }

        let body: serde_json::Value = resp.json().await.ok()?;
        let data = body["data"].as_array().or_else(|| body.as_array())?;

        let now = Utc::now();
        let models: Vec<DiscoveredModel> = data
            .iter()
            .filter_map(|m| {
                let id = m["id"].as_str().or_else(|| m["path"].as_str())?;
                if id.is_empty() {
                    return None;
                }

                let ctx = m["max_context_length"].as_u64().unwrap_or(0) as usize;
                let caps = &m["capabilities"];
                let vision = caps["vision"].as_bool().unwrap_or(false);
                let tools = caps["trained_for_tool_use"].as_bool().unwrap_or(false);
                let thinking = caps
                    .get("reasoning")
                    .and_then(|r| r.get("default"))
                    .and_then(|d| d.as_str())
                    .map(|s| s == "on")
                    .unwrap_or(false);

                Some(DiscoveredModel {
                    id: id.to_string(),
                    name: id.to_string(),
                    provider: "lmstudio".into(),
                    context_length: ctx,
                    max_output_tokens: 0,
                    capabilities: ModelCapabilities {
                        context_length: ctx,
                        supports_vision: vision,
                        supports_function_calling: tools,
                        supports_thinking: thinking,
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
            })
            .collect();
        Some(models)
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
                    "LM Studio unreachable at {}, returning empty list",
                    self.host
                );
                Ok(Vec::new())
            }
        }
    }
}
