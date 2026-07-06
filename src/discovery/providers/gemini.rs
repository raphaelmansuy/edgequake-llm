//! Gemini (Google AI) discovery — DYNAMIC strategy.
//!
//! `GET /v1beta/models` returns inputTokenLimit, outputTokenLimit,
//! supportedGenerationMethods, and thinking: bool.
//!
//! Source: <https://ai.google.dev/gemini-api/docs/models>

use async_trait::async_trait;
use chrono::Utc;

use super::gemini_model_parse::parse_gemini_models_response;
use crate::discovery::registry::gemini_models;
use crate::discovery::traits::ModelDiscoveryProvider;
use crate::discovery::types::{DiscoveredModel, DiscoverySource, DiscoveryStrategy};

pub struct GeminiDiscovery {
    api_key: Option<String>,
}

impl Default for GeminiDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl GeminiDiscovery {
    pub fn new() -> Self {
        Self {
            api_key: std::env::var("GEMINI_API_KEY")
                .or_else(|_| std::env::var("GOOGLE_API_KEY"))
                .ok(),
        }
    }

    pub fn with_api_key(api_key: String) -> Self {
        Self {
            api_key: Some(api_key),
        }
    }

    async fn fetch_from_api(&self) -> Option<Vec<DiscoveredModel>> {
        let key = self.api_key.as_ref()?;
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models?key={}",
            key
        );
        let client = reqwest::Client::new();
        let resp = client
            .get(&url)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
            .ok()?;

        if !resp.status().is_success() {
            return None;
        }

        let body: serde_json::Value = resp.json().await.ok()?;
        let models =
            parse_gemini_models_response(&body, "gemini", DiscoverySource::DynamicApi, Utc::now());
        if models.is_empty() {
            None
        } else {
            Some(models)
        }
    }
}

#[async_trait]
impl ModelDiscoveryProvider for GeminiDiscovery {
    fn provider_id(&self) -> &str {
        "gemini"
    }

    fn discovery_strategy(&self) -> DiscoveryStrategy {
        DiscoveryStrategy::Dynamic
    }

    async fn discover_models(&self) -> crate::error::Result<Vec<DiscoveredModel>> {
        match self.fetch_from_api().await {
            Some(models) if !models.is_empty() => Ok(models),
            _ => {
                tracing::info!("Gemini API unavailable, using static registry");
                Ok(gemini_models())
            }
        }
    }
}
