//! Vertex AI discovery — HYBRID strategy.
//!
//! Primary: unified Gemini API on Vertex (`GET /v1beta/models`).
//! Fallback: static registry re-tagged as `provider: "vertexai"`.
//!
//! Source: specs/001-edgequake-llm/04-PROVIDER-DISCOVERY-APPROACHES.md §3

use async_trait::async_trait;
use chrono::Utc;
use std::collections::HashSet;

use super::gemini_model_parse::parse_gemini_models_response;
use super::google_vertex_auth::{resolve_vertex_discovery_credentials, VertexDiscoveryCredentials};
use crate::discovery::registry::vertexai_models;
use crate::discovery::traits::ModelDiscoveryProvider;
use crate::discovery::types::{DiscoveredModel, DiscoverySource, DiscoveryStrategy};

pub struct VertexAIDiscovery {
    credentials: Option<VertexDiscoveryCredentials>,
}

impl Default for VertexAIDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl VertexAIDiscovery {
    pub fn new() -> Self {
        Self {
            credentials: resolve_vertex_discovery_credentials(),
        }
    }

    pub fn with_credentials(
        project_id: impl Into<String>,
        region: impl Into<String>,
        access_token: impl Into<String>,
    ) -> Self {
        Self {
            credentials: Some(VertexDiscoveryCredentials {
                project_id: project_id.into(),
                region: region.into(),
                access_token: access_token.into(),
            }),
        }
    }

    fn discovery_hosts(region: &str) -> Vec<String> {
        if region == "global" {
            vec!["aiplatform.googleapis.com".to_string()]
        } else {
            vec![
                format!("{region}-aiplatform.googleapis.com"),
                "aiplatform.googleapis.com".to_string(),
            ]
        }
    }

    async fn fetch_host_models(&self, host: &str, token: &str) -> Option<Vec<DiscoveredModel>> {
        let url = format!("https://{host}/v1beta/models");
        let client = reqwest::Client::new();
        let resp = client
            .get(&url)
            .header("Authorization", format!("Bearer {token}"))
            .timeout(std::time::Duration::from_secs(15))
            .send()
            .await
            .ok()?;

        if !resp.status().is_success() {
            return None;
        }

        let body: serde_json::Value = resp.json().await.ok()?;
        let models = parse_gemini_models_response(
            &body,
            "vertexai",
            DiscoverySource::DynamicApi,
            Utc::now(),
        );
        if models.is_empty() {
            None
        } else {
            Some(models)
        }
    }

    async fn fetch_live_models(&self) -> Option<Vec<DiscoveredModel>> {
        let creds = self.credentials.as_ref()?;
        let mut seen = HashSet::new();
        let mut out = Vec::new();

        for host in Self::discovery_hosts(&creds.region) {
            if let Some(mut models) = self.fetch_host_models(&host, &creds.access_token).await {
                for model in models.drain(..) {
                    if seen.insert(model.id.clone()) {
                        out.push(model);
                    }
                }
            }
        }

        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }
}

#[async_trait]
impl ModelDiscoveryProvider for VertexAIDiscovery {
    fn provider_id(&self) -> &str {
        "vertexai"
    }

    fn discovery_strategy(&self) -> DiscoveryStrategy {
        DiscoveryStrategy::Hybrid
    }

    async fn discover_models(&self) -> crate::error::Result<Vec<DiscoveredModel>> {
        if let Some(models) = self.fetch_live_models().await {
            if !models.is_empty() {
                return Ok(models);
            }
        }

        tracing::info!("Vertex AI live discovery unavailable, using static registry");
        Ok(vertexai_models())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::types::CapabilityFilter;

    #[test]
    fn static_fallback_tags_vertexai_provider() {
        let models = vertexai_models();
        assert!(!models.is_empty());
        assert!(models.iter().all(|m| m.provider == "vertexai"));
    }

    #[test]
    fn static_filter_matches_vertexai_provider() {
        let filter = CapabilityFilter {
            provider: Some("vertexai".to_string()),
            ..Default::default()
        };
        let matches: Vec<_> = vertexai_models()
            .into_iter()
            .filter(|m| filter.matches(m))
            .collect();
        assert!(!matches.is_empty());
        assert!(matches.iter().any(|m| m.id.contains("gemini")));
    }

    #[test]
    fn dedupe_hosts_prefers_first_seen() {
        let hosts = VertexAIDiscovery::discovery_hosts("us-central1");
        assert_eq!(hosts.len(), 2);
        assert!(hosts[0].contains("us-central1"));
    }
}
