//! xAI Grok discovery — STATIC strategy.
//!
//! xAI's API (`GET /v1/models`) returns minimal metadata,
//! so we rely on the static registry with cited documentation.
//!
//! Source: <https://docs.x.ai/docs/models>

use async_trait::async_trait;

use crate::discovery::registry::xai_models;
use crate::discovery::traits::ModelDiscoveryProvider;
use crate::discovery::types::{DiscoveredModel, DiscoveryStrategy};

pub struct XAIDiscovery;

impl Default for XAIDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl XAIDiscovery {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ModelDiscoveryProvider for XAIDiscovery {
    fn provider_id(&self) -> &str {
        "xai"
    }

    fn discovery_strategy(&self) -> DiscoveryStrategy {
        DiscoveryStrategy::Static
    }

    async fn discover_models(&self) -> crate::error::Result<Vec<DiscoveredModel>> {
        Ok(xai_models())
    }
}
