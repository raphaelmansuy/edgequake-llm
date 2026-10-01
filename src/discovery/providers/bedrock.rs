//! AWS Bedrock discovery — HYBRID strategy (feature-gated).
//!
//! Static registry for known Bedrock models. Dynamic ListFoundationModels
//! requires `aws_sdk_bedrock` (not just runtime) which is not yet a dependency.
//!
//! Source: <https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html>
//! Verified: 2026-10-01

use async_trait::async_trait;
use chrono::Utc;

use crate::discovery::traits::ModelDiscoveryProvider;
use crate::discovery::types::{DiscoveredModel, DiscoverySource, DiscoveryStrategy};
use crate::model_config::{ModelCapabilities, ModelType};

pub struct BedrockDiscovery;

impl Default for BedrockDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl BedrockDiscovery {
    pub fn new() -> Self {
        Self
    }

    fn static_catalog() -> Vec<DiscoveredModel> {
        let now = Utc::now();
        vec![
            DiscoveredModel {
                id: "anthropic.claude-sonnet-5-5".into(),
                name: "Claude Sonnet 5.5 (Bedrock)".into(),
                provider: "bedrock".into(),
                context_length: 1_000_000,
                max_output_tokens: 128_000,
                capabilities: ModelCapabilities {
                    context_length: 1_000_000,
                    max_output_tokens: 128_000,
                    supports_vision: true,
                    supports_function_calling: true,
                    supports_streaming: true,
                    supports_thinking: true,
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
                id: "anthropic.claude-opus-5-5".into(),
                name: "Claude Opus 5.5 (Bedrock)".into(),
                provider: "bedrock".into(),
                context_length: 1_000_000,
                max_output_tokens: 128_000,
                capabilities: ModelCapabilities {
                    context_length: 1_000_000,
                    max_output_tokens: 128_000,
                    supports_vision: true,
                    supports_function_calling: true,
                    supports_streaming: true,
                    supports_thinking: true,
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
                id: "anthropic.claude-fable-5-1".into(),
                name: "Claude Fable 5.1 (Bedrock)".into(),
                provider: "bedrock".into(),
                context_length: 1_000_000,
                max_output_tokens: 128_000,
                capabilities: ModelCapabilities {
                    context_length: 1_000_000,
                    max_output_tokens: 128_000,
                    supports_vision: true,
                    supports_function_calling: true,
                    supports_streaming: true,
                    supports_thinking: true,
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
                id: "xai.grok-4.7".into(),
                name: "Grok 4.7 (Bedrock)".into(),
                provider: "bedrock".into(),
                context_length: 500_000,
                max_output_tokens: 32_768,
                capabilities: ModelCapabilities {
                    context_length: 500_000,
                    max_output_tokens: 32_768,
                    supports_vision: true,
                    supports_function_calling: true,
                    supports_streaming: true,
                    supports_thinking: true,
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
                id: "anthropic.claude-sonnet-5".into(),
                name: "Claude Sonnet 5 (Bedrock)".into(),
                provider: "bedrock".into(),
                context_length: 1_048_576,
                max_output_tokens: 131_072,
                capabilities: ModelCapabilities {
                    context_length: 1_048_576,
                    max_output_tokens: 131_072,
                    supports_vision: true,
                    supports_function_calling: true,
                    supports_streaming: true,
                    supports_thinking: true,
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
                id: "anthropic.claude-fable-5".into(),
                name: "Claude Fable 5 (Bedrock)".into(),
                provider: "bedrock".into(),
                context_length: 1_048_576,
                max_output_tokens: 131_072,
                capabilities: ModelCapabilities {
                    context_length: 1_048_576,
                    max_output_tokens: 131_072,
                    supports_vision: true,
                    supports_function_calling: true,
                    supports_streaming: true,
                    supports_thinking: true,
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
                id: "anthropic.claude-opus-4-8".into(),
                name: "Claude Opus 4.8 (Bedrock)".into(),
                provider: "bedrock".into(),
                context_length: 1_048_576,
                max_output_tokens: 131_072,
                capabilities: ModelCapabilities {
                    context_length: 1_048_576,
                    max_output_tokens: 131_072,
                    supports_vision: true,
                    supports_function_calling: true,
                    supports_streaming: true,
                    supports_thinking: true,
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
                id: "amazon.nova-premier-v1:0".into(),
                name: "Amazon Nova Premier (Bedrock)".into(),
                provider: "bedrock".into(),
                context_length: 1_048_576,
                max_output_tokens: 0,
                capabilities: ModelCapabilities {
                    context_length: 1_048_576,
                    supports_vision: true,
                    supports_function_calling: true,
                    supports_streaming: true,
                    supports_thinking: true,
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
            DiscoveredModel {
                id: "meta.llama-4-maverick-17b-instruct-v1:0".into(),
                name: "Llama 4 Maverick 17B (Bedrock)".into(),
                provider: "bedrock".into(),
                context_length: 1_048_576,
                max_output_tokens: 16_384,
                capabilities: ModelCapabilities {
                    context_length: 1_048_576,
                    max_output_tokens: 16_384,
                    supports_vision: true,
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
        ]
    }
}

#[async_trait]
impl ModelDiscoveryProvider for BedrockDiscovery {
    fn provider_id(&self) -> &str {
        "bedrock"
    }

    fn discovery_strategy(&self) -> DiscoveryStrategy {
        DiscoveryStrategy::Hybrid
    }

    async fn discover_models(&self) -> crate::error::Result<Vec<DiscoveredModel>> {
        Ok(Self::static_catalog())
    }
}
