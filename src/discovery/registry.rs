//! Static model registry — built-in model data with source citations.
//!
//! Every entry MUST include:
//! 1. Official doc URL in a comment
//! 2. Date when the entry was last verified
//! 3. Explicit values for ALL capability fields (no defaults, no guessing)
//!
//! Unknown values are represented as `0` or `false`, never guessed.
//! This module contains ZERO heuristics or name-pattern inference.

use chrono::Utc;

use crate::model_config::{ModelCapabilities, ModelType};

use super::types::{DiscoveredModel, DiscoverySource};

/// Return a capability set that explicitly signals "unknown".
///
/// EVERY field that we don't know is 0/false — we never guess.
/// The caller sees `context_length: 0` and must handle accordingly.
pub fn unknown_model(provider: &str, model_id: &str) -> DiscoveredModel {
    tracing::warn!(
        provider = provider,
        model = model_id,
        "Model not found in discovery registry; returning unknown capabilities"
    );
    DiscoveredModel {
        id: model_id.to_string(),
        name: model_id.to_string(),
        provider: provider.to_string(),
        source: DiscoverySource::Unknown,
        available: true,
        ..Default::default()
    }
}

// ============================================================================
// OpenAI Models
// Source: https://platform.openai.com/docs/models
// Verified: 2026-07-04
// ============================================================================

pub fn openai_models() -> Vec<DiscoveredModel> {
    let now = Utc::now();
    vec![
        // Source: https://platform.openai.com/docs/models/gpt-5.5
        DiscoveredModel {
            id: "gpt-5.5".into(),
            name: "GPT-5.5".into(),
            provider: "openai".into(),
            context_length: 1_048_576,
            max_output_tokens: 32_768,
            capabilities: ModelCapabilities {
                context_length: 1_048_576,
                max_output_tokens: 32_768,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(5.00),
            cost_per_m_output: Some(30.00),
            model_type: ModelType::Llm,
            tags: vec!["flagship".into()],
            deprecated: false,
        },
        // Source: https://platform.openai.com/docs/models/gpt-5.4
        DiscoveredModel {
            id: "gpt-5.4".into(),
            name: "GPT-5.4".into(),
            provider: "openai".into(),
            context_length: 1_048_576,
            max_output_tokens: 32_768,
            capabilities: ModelCapabilities {
                context_length: 1_048_576,
                max_output_tokens: 32_768,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(2.50),
            cost_per_m_output: Some(15.00),
            model_type: ModelType::Llm,
            tags: vec![],
            deprecated: false,
        },
        // Source: https://platform.openai.com/docs/models/gpt-4.1
        DiscoveredModel {
            id: "gpt-4.1".into(),
            name: "GPT-4.1".into(),
            provider: "openai".into(),
            context_length: 1_048_576,
            max_output_tokens: 32_768,
            capabilities: ModelCapabilities {
                context_length: 1_048_576,
                max_output_tokens: 32_768,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(2.00),
            cost_per_m_output: Some(8.00),
            model_type: ModelType::Llm,
            tags: vec!["coding".into()],
            deprecated: false,
        },
        // Source: https://platform.openai.com/docs/models/gpt-4.1-mini
        DiscoveredModel {
            id: "gpt-4.1-mini".into(),
            name: "GPT-4.1 Mini".into(),
            provider: "openai".into(),
            context_length: 1_048_576,
            max_output_tokens: 32_768,
            capabilities: ModelCapabilities {
                context_length: 1_048_576,
                max_output_tokens: 32_768,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(0.40),
            cost_per_m_output: Some(1.60),
            model_type: ModelType::Llm,
            tags: vec!["fast".into()],
            deprecated: false,
        },
        // Source: https://developers.openai.com/api/docs/models/gpt-4.1-nano
        // Status: DEPRECATED as of July 2026 (succeeded by GPT-5 mini)
        DiscoveredModel {
            id: "gpt-4.1-nano".into(),
            name: "GPT-4.1 Nano (Deprecated)".into(),
            provider: "openai".into(),
            context_length: 1_048_576,
            max_output_tokens: 32_768,
            capabilities: ModelCapabilities {
                context_length: 1_048_576,
                max_output_tokens: 32_768,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(0.10),
            cost_per_m_output: Some(0.40),
            model_type: ModelType::Llm,
            tags: vec!["fast".into(), "affordable".into()],
            deprecated: true,
        },
        // Source: https://platform.openai.com/docs/models/gpt-4o
        DiscoveredModel {
            id: "gpt-4o".into(),
            name: "GPT-4o".into(),
            provider: "openai".into(),
            context_length: 128_000,
            max_output_tokens: 16_384,
            capabilities: ModelCapabilities {
                context_length: 128_000,
                max_output_tokens: 16_384,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(2.50),
            cost_per_m_output: Some(10.00),
            model_type: ModelType::Llm,
            tags: vec![],
            deprecated: false,
        },
        // Source: https://platform.openai.com/docs/models/gpt-4o-mini
        DiscoveredModel {
            id: "gpt-4o-mini".into(),
            name: "GPT-4o Mini".into(),
            provider: "openai".into(),
            context_length: 128_000,
            max_output_tokens: 16_384,
            capabilities: ModelCapabilities {
                context_length: 128_000,
                max_output_tokens: 16_384,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(0.15),
            cost_per_m_output: Some(0.60),
            model_type: ModelType::Llm,
            tags: vec!["fast".into(), "affordable".into()],
            deprecated: false,
        },
        // Source: https://platform.openai.com/docs/models/o3
        DiscoveredModel {
            id: "o3".into(),
            name: "o3".into(),
            provider: "openai".into(),
            context_length: 200_000,
            max_output_tokens: 100_000,
            capabilities: ModelCapabilities {
                context_length: 200_000,
                max_output_tokens: 100_000,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(2.00),
            cost_per_m_output: Some(8.00),
            model_type: ModelType::Llm,
            tags: vec!["reasoning".into()],
            deprecated: false,
        },
        // Source: https://developers.openai.com/api/docs/models/o4-mini
        // Status: DEPRECATED as of July 2026 (succeeded by GPT-5 mini)
        DiscoveredModel {
            id: "o4-mini".into(),
            name: "o4-mini (Deprecated)".into(),
            provider: "openai".into(),
            context_length: 200_000,
            max_output_tokens: 100_000,
            capabilities: ModelCapabilities {
                context_length: 200_000,
                max_output_tokens: 100_000,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(1.10),
            cost_per_m_output: Some(4.40),
            model_type: ModelType::Llm,
            tags: vec!["reasoning".into(), "affordable".into()],
            deprecated: true,
        },
        // Source: https://developers.openai.com/api/docs/models/gpt-5
        DiscoveredModel {
            id: "gpt-5".into(),
            name: "GPT-5".into(),
            provider: "openai".into(),
            context_length: 1_048_576,
            max_output_tokens: 32_768,
            capabilities: ModelCapabilities {
                context_length: 1_048_576,
                max_output_tokens: 32_768,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(2.00),
            cost_per_m_output: Some(8.00),
            model_type: ModelType::Llm,
            tags: vec!["reasoning".into()],
            deprecated: false,
        },
        // Source: https://developers.openai.com/api/docs/models/gpt-5-mini
        DiscoveredModel {
            id: "gpt-5-mini".into(),
            name: "GPT-5 Mini".into(),
            provider: "openai".into(),
            context_length: 1_048_576,
            max_output_tokens: 32_768,
            capabilities: ModelCapabilities {
                context_length: 1_048_576,
                max_output_tokens: 32_768,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(0.40),
            cost_per_m_output: Some(1.60),
            model_type: ModelType::Llm,
            tags: vec!["reasoning".into(), "fast".into()],
            deprecated: false,
        },
        // Embeddings
        // Source: https://platform.openai.com/docs/models/text-embedding-3-small
        DiscoveredModel {
            id: "text-embedding-3-small".into(),
            name: "Embedding 3 Small".into(),
            provider: "openai".into(),
            context_length: 8_191,
            max_output_tokens: 0,
            capabilities: ModelCapabilities {
                embedding_dimension: 1536,
                max_embedding_tokens: 8191,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(0.02),
            cost_per_m_output: None,
            model_type: ModelType::Embedding,
            tags: vec!["embedding".into()],
            deprecated: false,
        },
        DiscoveredModel {
            id: "text-embedding-3-large".into(),
            name: "Embedding 3 Large".into(),
            provider: "openai".into(),
            context_length: 8_191,
            max_output_tokens: 0,
            capabilities: ModelCapabilities {
                embedding_dimension: 3072,
                max_embedding_tokens: 8191,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(0.13),
            cost_per_m_output: None,
            model_type: ModelType::Embedding,
            tags: vec!["embedding".into()],
            deprecated: false,
        },
    ]
}

// ============================================================================
// Anthropic Claude Models
// Source: https://docs.anthropic.com/en/docs/about-claude/models
// Verified: 2026-07-04
// ============================================================================

pub fn anthropic_models() -> Vec<DiscoveredModel> {
    let now = Utc::now();
    vec![
        DiscoveredModel {
            id: "claude-fable-5".into(),
            name: "Claude Fable 5".into(),
            provider: "anthropic".into(),
            context_length: 1_048_576,
            max_output_tokens: 131_072,
            capabilities: ModelCapabilities {
                context_length: 1_048_576,
                max_output_tokens: 131_072,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(10.00),
            cost_per_m_output: Some(50.00),
            model_type: ModelType::Llm,
            tags: vec!["flagship".into(), "reasoning".into()],
            deprecated: false,
        },
        DiscoveredModel {
            id: "claude-opus-4-8".into(),
            name: "Claude Opus 4.8".into(),
            provider: "anthropic".into(),
            context_length: 1_048_576,
            max_output_tokens: 131_072,
            capabilities: ModelCapabilities {
                context_length: 1_048_576,
                max_output_tokens: 131_072,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(5.00),
            cost_per_m_output: Some(25.00),
            model_type: ModelType::Llm,
            tags: vec!["reasoning".into()],
            deprecated: false,
        },
        DiscoveredModel {
            id: "claude-sonnet-5".into(),
            name: "Claude Sonnet 5".into(),
            provider: "anthropic".into(),
            context_length: 1_048_576,
            max_output_tokens: 131_072,
            capabilities: ModelCapabilities {
                context_length: 1_048_576,
                max_output_tokens: 131_072,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(3.00),
            cost_per_m_output: Some(15.00),
            model_type: ModelType::Llm,
            tags: vec!["coding".into(), "balanced".into()],
            deprecated: false,
        },
        DiscoveredModel {
            id: "claude-opus-4-7".into(),
            name: "Claude Opus 4.7".into(),
            provider: "anthropic".into(),
            context_length: 1_048_576,
            max_output_tokens: 131_072,
            capabilities: ModelCapabilities {
                context_length: 1_048_576,
                max_output_tokens: 131_072,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: None,
            cost_per_m_output: None,
            model_type: ModelType::Llm,
            tags: vec!["reasoning".into()],
            deprecated: false,
        },
        DiscoveredModel {
            id: "claude-opus-4-6".into(),
            name: "Claude Opus 4.6".into(),
            provider: "anthropic".into(),
            context_length: 1_048_576,
            max_output_tokens: 131_072,
            capabilities: ModelCapabilities {
                context_length: 1_048_576,
                max_output_tokens: 131_072,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: None,
            cost_per_m_output: None,
            model_type: ModelType::Llm,
            tags: vec![],
            deprecated: false,
        },
        DiscoveredModel {
            id: "claude-sonnet-4-6".into(),
            name: "Claude Sonnet 4.6".into(),
            provider: "anthropic".into(),
            context_length: 1_048_576,
            max_output_tokens: 65_536,
            capabilities: ModelCapabilities {
                context_length: 1_048_576,
                max_output_tokens: 65_536,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: None,
            cost_per_m_output: None,
            model_type: ModelType::Llm,
            tags: vec![],
            deprecated: false,
        },
        DiscoveredModel {
            id: "claude-sonnet-4-5-20250929".into(),
            name: "Claude Sonnet 4.5".into(),
            provider: "anthropic".into(),
            context_length: 200_000,
            max_output_tokens: 65_536,
            capabilities: ModelCapabilities {
                context_length: 200_000,
                max_output_tokens: 65_536,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(3.00),
            cost_per_m_output: Some(15.00),
            model_type: ModelType::Llm,
            tags: vec![],
            deprecated: false,
        },
        DiscoveredModel {
            id: "claude-3-5-sonnet-20241022".into(),
            name: "Claude 3.5 Sonnet".into(),
            provider: "anthropic".into(),
            context_length: 200_000,
            max_output_tokens: 8_192,
            capabilities: ModelCapabilities {
                context_length: 200_000,
                max_output_tokens: 8_192,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(3.00),
            cost_per_m_output: Some(15.00),
            model_type: ModelType::Llm,
            tags: vec![],
            deprecated: false,
        },
        // Source: https://platform.claude.com/docs/en/about-claude/models/overview
        DiscoveredModel {
            id: "claude-haiku-4-5".into(),
            name: "Claude Haiku 4.5".into(),
            provider: "anthropic".into(),
            context_length: 200_000,
            max_output_tokens: 65_536,
            capabilities: ModelCapabilities {
                context_length: 200_000,
                max_output_tokens: 65_536,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(1.00),
            cost_per_m_output: Some(5.00),
            model_type: ModelType::Llm,
            tags: vec!["fast".into(), "affordable".into()],
            deprecated: false,
        },
        DiscoveredModel {
            id: "claude-3-5-haiku-20241022".into(),
            name: "Claude 3.5 Haiku (Legacy)".into(),
            provider: "anthropic".into(),
            context_length: 200_000,
            max_output_tokens: 8_192,
            capabilities: ModelCapabilities {
                context_length: 200_000,
                max_output_tokens: 8_192,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(0.80),
            cost_per_m_output: Some(4.00),
            model_type: ModelType::Llm,
            tags: vec!["fast".into(), "affordable".into()],
            deprecated: false,
        },
    ]
}

// ============================================================================
// Google Gemini Models
// Source: https://ai.google.dev/gemini-api/docs/models
// Verified: 2026-07-04
// ============================================================================

pub fn gemini_models() -> Vec<DiscoveredModel> {
    let now = Utc::now();
    vec![
        DiscoveredModel {
            id: "gemini-3.5-flash".into(),
            name: "Gemini 3.5 Flash".into(),
            provider: "gemini".into(),
            context_length: 1_048_576,
            max_output_tokens: 65_536,
            capabilities: ModelCapabilities {
                context_length: 1_048_576,
                max_output_tokens: 65_536,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(1.50),
            cost_per_m_output: Some(9.00),
            model_type: ModelType::Llm,
            tags: vec!["fast".into()],
            deprecated: false,
        },
        DiscoveredModel {
            id: "gemini-2.5-pro".into(),
            name: "Gemini 2.5 Pro".into(),
            provider: "gemini".into(),
            context_length: 1_048_576,
            max_output_tokens: 65_536,
            capabilities: ModelCapabilities {
                context_length: 1_048_576,
                max_output_tokens: 65_536,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(1.25),
            cost_per_m_output: Some(10.00),
            model_type: ModelType::Llm,
            tags: vec!["reasoning".into()],
            deprecated: false,
        },
        DiscoveredModel {
            id: "gemini-2.5-flash".into(),
            name: "Gemini 2.5 Flash".into(),
            provider: "gemini".into(),
            context_length: 1_048_576,
            max_output_tokens: 65_536,
            capabilities: ModelCapabilities {
                context_length: 1_048_576,
                max_output_tokens: 65_536,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(0.30),
            cost_per_m_output: Some(2.50),
            model_type: ModelType::Llm,
            tags: vec!["fast".into(), "affordable".into()],
            deprecated: false,
        },
        DiscoveredModel {
            id: "gemini-3.1-pro-preview".into(),
            name: "Gemini 3.1 Pro".into(),
            provider: "gemini".into(),
            context_length: 1_048_576,
            max_output_tokens: 65_536,
            capabilities: ModelCapabilities {
                context_length: 1_048_576,
                max_output_tokens: 65_536,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(2.00),
            cost_per_m_output: Some(12.00),
            model_type: ModelType::Llm,
            tags: vec!["reasoning".into()],
            deprecated: false,
        },
    ]
}

// ============================================================================
// Mistral Models
// Source: https://docs.mistral.ai/models/overview
// Verified: 2026-07-04
// ============================================================================

pub fn mistral_models() -> Vec<DiscoveredModel> {
    let now = Utc::now();
    vec![
        DiscoveredModel {
            id: "mistral-large-latest".into(),
            name: "Mistral Large 3".into(),
            provider: "mistral".into(),
            context_length: 262_144,
            max_output_tokens: 32_768,
            capabilities: ModelCapabilities {
                context_length: 262_144,
                max_output_tokens: 32_768,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(0.50),
            cost_per_m_output: Some(1.50),
            model_type: ModelType::Llm,
            tags: vec!["flagship".into()],
            deprecated: false,
        },
        DiscoveredModel {
            id: "mistral-small-latest".into(),
            name: "Mistral Small 4".into(),
            provider: "mistral".into(),
            context_length: 262_144,
            max_output_tokens: 32_768,
            capabilities: ModelCapabilities {
                context_length: 262_144,
                max_output_tokens: 32_768,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(0.15),
            cost_per_m_output: Some(0.60),
            model_type: ModelType::Llm,
            tags: vec!["fast".into(), "affordable".into()],
            deprecated: false,
        },
        DiscoveredModel {
            id: "codestral-latest".into(),
            name: "Codestral".into(),
            provider: "mistral".into(),
            context_length: 262_144,
            max_output_tokens: 32_768,
            capabilities: ModelCapabilities {
                context_length: 262_144,
                max_output_tokens: 32_768,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(0.30),
            cost_per_m_output: Some(0.90),
            model_type: ModelType::Llm,
            tags: vec!["coding".into()],
            deprecated: false,
        },
        DiscoveredModel {
            id: "mistral-medium-latest".into(),
            name: "Mistral Medium 3.5".into(),
            provider: "mistral".into(),
            context_length: 262_144,
            max_output_tokens: 32_768,
            capabilities: ModelCapabilities {
                context_length: 262_144,
                max_output_tokens: 32_768,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(1.50),
            cost_per_m_output: Some(7.50),
            model_type: ModelType::Llm,
            tags: vec!["reasoning".into(), "agentic".into()],
            deprecated: false,
        },
        DiscoveredModel {
            id: "magistral-medium-latest".into(),
            name: "Magistral Medium".into(),
            provider: "mistral".into(),
            context_length: 131_072,
            max_output_tokens: 40_960,
            capabilities: ModelCapabilities {
                context_length: 131_072,
                max_output_tokens: 40_960,
                supports_function_calling: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(2.00),
            cost_per_m_output: Some(5.00),
            model_type: ModelType::Llm,
            tags: vec!["reasoning".into()],
            deprecated: false,
        },
        DiscoveredModel {
            id: "pixtral-large-latest".into(),
            name: "Pixtral Large".into(),
            provider: "mistral".into(),
            context_length: 131_072,
            max_output_tokens: 32_768,
            capabilities: ModelCapabilities {
                context_length: 131_072,
                max_output_tokens: 32_768,
                supports_vision: true,
                supports_function_calling: true,
                supports_streaming: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(2.00),
            cost_per_m_output: Some(6.00),
            model_type: ModelType::Llm,
            tags: vec!["vision".into()],
            deprecated: false,
        },
    ]
}

// ============================================================================
// xAI Grok Models
// Source: https://docs.x.ai/developers/models/grok-4-0709
// Verified: 2026-07-04
// Note: All old models (grok-3, grok-4, grok-4.20, etc.) were retired
//       on May 15, 2026 and redirect to grok-4.3.
//       See: https://docs.x.ai/developers/migration/may-15-retirement
// ============================================================================

pub fn xai_models() -> Vec<DiscoveredModel> {
    let now = Utc::now();
    vec![
        // Grok 4.3 — the ONLY active model as of June 15, 2026.
        // All legacy slugs (grok-3, grok-4, grok-4.20, etc.) redirect here.
        // Source: https://docs.x.ai/developers/models/grok-4-0709
        DiscoveredModel {
            id: "grok-4.3".into(),
            name: "Grok 4.3".into(),
            provider: "xai".into(),
            context_length: 1_000_000,
            max_output_tokens: 32_768,
            capabilities: ModelCapabilities {
                context_length: 1_000_000,
                max_output_tokens: 32_768,
                supports_vision: true,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(1.25),
            cost_per_m_output: Some(2.50),
            model_type: ModelType::Llm,
            tags: vec!["flagship".into(), "reasoning".into()],
            deprecated: false,
        },
        // Grok Build 0.1 — fast coding model (early access).
        // Source: https://x.ai/api
        DiscoveredModel {
            id: "grok-build-0.1".into(),
            name: "Grok Build 0.1".into(),
            provider: "xai".into(),
            context_length: 262_144,
            max_output_tokens: 32_768,
            capabilities: ModelCapabilities {
                context_length: 262_144,
                max_output_tokens: 32_768,
                supports_function_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                supports_system_message: true,
                ..Default::default()
            },
            source: DiscoverySource::StaticRegistry,
            discovered_at: now,
            available: true,
            cost_per_m_input: Some(1.00),
            cost_per_m_output: Some(2.00),
            model_type: ModelType::Llm,
            tags: vec!["coding".into()],
            deprecated: false,
        },
    ]
}

/// Get all static registry models across all providers.
pub fn all_static_models() -> Vec<DiscoveredModel> {
    let mut all = Vec::with_capacity(64);
    all.extend(openai_models());
    all.extend(anthropic_models());
    all.extend(gemini_models());
    all.extend(mistral_models());
    all.extend(xai_models());
    all
}

/// Look up a model in the static registry by provider and model ID.
pub fn static_lookup(provider: &str, model_id: &str) -> Option<DiscoveredModel> {
    let models = match provider {
        "openai" => openai_models(),
        "anthropic" => anthropic_models(),
        "gemini" | "vertexai" => gemini_models(),
        "mistral" => mistral_models(),
        "xai" => xai_models(),
        _ => return None,
    };
    models.into_iter().find(|m| m.id == model_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_registry_not_empty() {
        let models = openai_models();
        assert!(!models.is_empty());
        assert!(models.iter().all(|m| m.provider == "openai"));
    }

    #[test]
    fn test_anthropic_registry_not_empty() {
        let models = anthropic_models();
        assert!(!models.is_empty());
        assert!(models.iter().all(|m| m.provider == "anthropic"));
    }

    #[test]
    fn test_gemini_registry_not_empty() {
        let models = gemini_models();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_all_static_models_no_empty_ids() {
        for model in all_static_models() {
            assert!(!model.id.is_empty(), "Model has empty ID");
            assert!(
                !model.provider.is_empty(),
                "Model {} has empty provider",
                model.id
            );
            assert!(
                model.source == DiscoverySource::StaticRegistry,
                "Model {} has wrong source",
                model.id
            );
        }
    }

    #[test]
    fn test_static_lookup_found() {
        assert!(static_lookup("openai", "gpt-4.1").is_some());
        assert!(static_lookup("anthropic", "claude-fable-5").is_some());
    }

    #[test]
    fn test_static_lookup_not_found() {
        assert!(static_lookup("openai", "nonexistent").is_none());
        assert!(static_lookup("unknown_provider", "any").is_none());
    }

    #[test]
    fn test_unknown_model_returns_zero_capabilities() {
        let m = unknown_model("test", "mystery-model");
        assert_eq!(m.context_length, 0);
        assert_eq!(m.max_output_tokens, 0);
        assert_eq!(m.source, DiscoverySource::Unknown);
        assert!(!m.capabilities.supports_vision);
    }

    #[test]
    fn test_codestral_context_is_256k() {
        let codestral = static_lookup("mistral", "codestral-latest").unwrap();
        assert_eq!(codestral.context_length, 262_144);
    }

    #[test]
    fn test_claude_fable_5_context_is_1m() {
        let fable = static_lookup("anthropic", "claude-fable-5").unwrap();
        assert_eq!(fable.context_length, 1_048_576);
        assert_eq!(fable.max_output_tokens, 131_072);
    }

    #[test]
    fn test_xai_grok43_is_flagship() {
        let grok = static_lookup("xai", "grok-4.3").unwrap();
        assert_eq!(grok.context_length, 1_000_000);
        assert!(grok.capabilities.supports_vision);
        assert!(grok.capabilities.supports_thinking);
        assert!(grok.capabilities.supports_function_calling);
        assert!(!grok.deprecated);
    }

    #[test]
    fn test_openai_deprecated_models() {
        let nano = static_lookup("openai", "gpt-4.1-nano").unwrap();
        assert!(nano.deprecated);
        let o4 = static_lookup("openai", "o4-mini").unwrap();
        assert!(o4.deprecated);
    }

    #[test]
    fn test_openai_gpt5_models() {
        let gpt5 = static_lookup("openai", "gpt-5").unwrap();
        assert!(!gpt5.deprecated);
        assert!(gpt5.capabilities.supports_thinking);
        let gpt5mini = static_lookup("openai", "gpt-5-mini").unwrap();
        assert!(!gpt5mini.deprecated);
    }

    #[test]
    fn test_anthropic_haiku_45() {
        let haiku = static_lookup("anthropic", "claude-haiku-4-5").unwrap();
        assert_eq!(haiku.context_length, 200_000);
        assert_eq!(haiku.max_output_tokens, 65_536);
    }

    #[test]
    fn test_mistral_large_context_256k() {
        let large = static_lookup("mistral", "mistral-large-latest").unwrap();
        assert_eq!(large.context_length, 262_144);
        assert!(large.capabilities.supports_vision);
    }

    #[test]
    fn test_mistral_medium_35() {
        let medium = static_lookup("mistral", "mistral-medium-latest").unwrap();
        assert_eq!(medium.context_length, 262_144);
        assert!(medium.capabilities.supports_thinking);
        assert!(medium.capabilities.supports_vision);
    }
}
