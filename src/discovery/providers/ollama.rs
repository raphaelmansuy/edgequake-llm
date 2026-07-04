//! Ollama discovery — DYNAMIC strategy.
//!
//! Since Ollama 0.30.0, `/api/tags` returns a `capabilities` array
//! per model: ["completion", "vision", "tools", "thinking", "embedding"].
//!
//! Source: https://github.com/ollama/ollama/blob/main/docs/api.md

use async_trait::async_trait;
use chrono::Utc;

use crate::discovery::traits::ModelDiscoveryProvider;
use crate::discovery::types::{DiscoveredModel, DiscoverySource, DiscoveryStrategy};
use crate::model_config::{ModelCapabilities, ModelType};

pub struct OllamaDiscovery {
    host: String,
}

impl Default for OllamaDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl OllamaDiscovery {
    pub fn new() -> Self {
        let host =
            std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
        Self { host }
    }

    pub fn with_host(host: impl Into<String>) -> Self {
        Self { host: host.into() }
    }

    async fn fetch_from_api(&self) -> Option<Vec<DiscoveredModel>> {
        let url = format!("{}/api/tags", self.host.trim_end_matches('/'));
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
        let models_arr = body["models"].as_array()?;

        let now = Utc::now();
        let models: Vec<DiscoveredModel> = models_arr
            .iter()
            .filter_map(|m| {
                let name = m["name"].as_str().or_else(|| m["model"].as_str())?;
                if name.is_empty() {
                    return None;
                }

                let caps: Vec<String> = m["capabilities"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                let is_embedding = caps.iter().any(|c| c == "embedding");
                let supports_vision = caps.iter().any(|c| c == "vision");
                let supports_tools = caps.iter().any(|c| c == "tools");
                let supports_thinking = caps.iter().any(|c| c == "thinking");

                Some(DiscoveredModel {
                    id: name.to_string(),
                    name: name.to_string(),
                    provider: "ollama".into(),
                    context_length: 0, // Requires /api/show per model for num_ctx
                    max_output_tokens: 0,
                    capabilities: ModelCapabilities {
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
                    model_type: if is_embedding {
                        ModelType::Embedding
                    } else {
                        ModelType::Llm
                    },
                    ..Default::default()
                })
            })
            .collect();
        Some(models)
    }
}

#[async_trait]
impl ModelDiscoveryProvider for OllamaDiscovery {
    fn provider_id(&self) -> &str {
        "ollama"
    }

    fn discovery_strategy(&self) -> DiscoveryStrategy {
        DiscoveryStrategy::Dynamic
    }

    async fn discover_models(&self) -> crate::error::Result<Vec<DiscoveredModel>> {
        match self.fetch_from_api().await {
            Some(models) => Ok(models),
            None => {
                tracing::info!("Ollama unreachable at {}, returning empty list", self.host);
                Ok(Vec::new())
            }
        }
    }
}
