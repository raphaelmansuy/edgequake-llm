//! Shared LM Studio `/api/v1/models` and `/v1/models` payload parsing.

use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::discovery::types::{DiscoveredModel, DiscoverySource};
use crate::model_config::{ModelCapabilities, ModelType};

/// Parse native (`models[]`) or OpenAI-compatible (`data[]`) LM Studio model list JSON.
pub fn parse_lmstudio_models_payload(
    body: &Value,
    discovered_at: DateTime<Utc>,
) -> Vec<DiscoveredModel> {
    let items = body["models"]
        .as_array()
        .or_else(|| body["data"].as_array())
        .or_else(|| body.as_array());

    let Some(items) = items else {
        return Vec::new();
    };

    items
        .iter()
        .filter_map(|entry| parse_lmstudio_entry(entry, discovered_at))
        .collect()
}

fn parse_lmstudio_entry(entry: &Value, discovered_at: DateTime<Utc>) -> Option<DiscoveredModel> {
    let id = entry["key"]
        .as_str()
        .or_else(|| entry["id"].as_str())
        .or_else(|| entry["path"].as_str())
        .filter(|s| !s.is_empty())?;

    let display_name = entry["display_name"]
        .as_str()
        .or_else(|| entry["name"].as_str())
        .unwrap_or(id);

    let ctx = entry["max_context_length"]
        .as_u64()
        .or_else(|| entry["context_length"].as_u64())
        .unwrap_or(0) as usize;

    let caps = &entry["capabilities"];
    let vision = caps["vision"].as_bool().unwrap_or(false);
    let tools = caps["trained_for_tool_use"].as_bool().unwrap_or(false);
    let thinking = caps
        .get("reasoning")
        .and_then(|r| r.get("default"))
        .and_then(|d| d.as_str())
        .map(|s| s == "on")
        .unwrap_or(false);

    let model_type = if vision {
        ModelType::Multimodal
    } else {
        ModelType::Llm
    };

    Some(DiscoveredModel {
        id: id.to_string(),
        name: display_name.to_string(),
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
        discovered_at,
        available: true,
        model_type,
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::types::DiscoverySource;

    #[test]
    fn parse_native_models_array() {
        let body: Value = serde_json::json!({
            "models": [{
                "key": "gemma-4-12b-coder",
                "display_name": "Gemma4 Coding",
                "max_context_length": 131072,
                "capabilities": {
                    "vision": false,
                    "trained_for_tool_use": true,
                    "reasoning": { "default": "off" }
                }
            }]
        });
        let models = parse_lmstudio_models_payload(&body, Utc::now());
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "gemma-4-12b-coder");
        assert_eq!(models[0].name, "Gemma4 Coding");
        assert_eq!(models[0].provider, "lmstudio");
        assert_eq!(models[0].source, DiscoverySource::DynamicApi);
        assert!(models[0].capabilities.supports_function_calling);
    }

    #[test]
    fn parse_openai_compatible_data_array() {
        let body: Value = serde_json::json!({
            "data": [{ "id": "google/gemma-4-26b", "object": "model" }]
        });
        let models = parse_lmstudio_models_payload(&body, Utc::now());
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "google/gemma-4-26b");
    }

    #[test]
    fn parse_vision_sets_multimodal() {
        let body: Value = serde_json::json!({
            "models": [{
                "key": "vision-model",
                "capabilities": { "vision": true }
            }]
        });
        let models = parse_lmstudio_models_payload(&body, Utc::now());
        assert_eq!(models[0].model_type, ModelType::Multimodal);
        assert!(models[0].capabilities.supports_vision);
    }

    #[test]
    fn empty_object_returns_empty() {
        assert!(parse_lmstudio_models_payload(&serde_json::json!({}), Utc::now()).is_empty());
    }
}
