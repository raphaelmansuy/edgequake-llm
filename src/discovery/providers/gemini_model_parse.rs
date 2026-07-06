//! Shared parsing for Google Gemini `Model{}` JSON (Google AI + Vertex unified API).

use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::discovery::types::{DiscoveredModel, DiscoverySource};
use crate::model_config::{ModelCapabilities, ModelType};

/// Parse a `models[]` array from Gemini / Vertex `v1beta/models` JSON.
pub fn parse_gemini_models_array(
    models_arr: &[Value],
    provider: &str,
    source: DiscoverySource,
    discovered_at: DateTime<Utc>,
) -> Vec<DiscoveredModel> {
    models_arr
        .iter()
        .filter_map(|m| parse_gemini_model_entry(m, provider, source.clone(), discovered_at))
        .collect()
}

/// Parse top-level `{ "models": [...] }` discovery response.
pub fn parse_gemini_models_response(
    body: &Value,
    provider: &str,
    source: DiscoverySource,
    discovered_at: DateTime<Utc>,
) -> Vec<DiscoveredModel> {
    body["models"]
        .as_array()
        .map(|arr| parse_gemini_models_array(arr, provider, source, discovered_at))
        .unwrap_or_default()
}

fn parse_gemini_model_entry(
    m: &Value,
    provider: &str,
    source: DiscoverySource,
    discovered_at: DateTime<Utc>,
) -> Option<DiscoveredModel> {
    let full_name = m["name"].as_str()?;
    let id = full_name.strip_prefix("models/").unwrap_or(full_name);
    let display = m["displayName"].as_str().unwrap_or(id);
    let ctx = m["inputTokenLimit"].as_u64().unwrap_or(0) as usize;
    let max_out = m["outputTokenLimit"].as_u64().unwrap_or(0) as usize;
    let thinking = m["thinking"].as_bool().unwrap_or(false);

    let methods: Vec<String> = m["supportedGenerationMethods"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let is_embedding = methods.iter().any(|m| m == "embedContent")
        && !methods.iter().any(|m| m == "generateContent");

    if ctx == 0 && !is_embedding {
        return None;
    }

    Some(DiscoveredModel {
        id: id.to_string(),
        name: display.to_string(),
        provider: provider.to_string(),
        context_length: ctx,
        max_output_tokens: max_out,
        capabilities: ModelCapabilities {
            context_length: ctx,
            max_output_tokens: max_out,
            supports_vision: true,
            supports_function_calling: methods.iter().any(|m| m == "generateContent"),
            supports_json_mode: true,
            supports_streaming: true,
            supports_thinking: thinking,
            supports_system_message: true,
            ..Default::default()
        },
        source,
        discovered_at,
        available: true,
        model_type: if is_embedding {
            ModelType::Embedding
        } else {
            ModelType::Llm
        },
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::types::DiscoverySource;

    #[test]
    fn parse_vertex_model_with_thinking() {
        let body: Value = serde_json::json!({
            "models": [{
                "name": "models/gemini-2.5-flash",
                "displayName": "Gemini 2.5 Flash",
                "inputTokenLimit": 1048576,
                "outputTokenLimit": 65536,
                "thinking": true,
                "supportedGenerationMethods": ["generateContent", "countTokens"]
            }]
        });
        let now = Utc::now();
        let models =
            parse_gemini_models_response(&body, "vertexai", DiscoverySource::DynamicApi, now);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].provider, "vertexai");
        assert_eq!(models[0].id, "gemini-2.5-flash");
        assert!(models[0].capabilities.supports_thinking);
        assert_eq!(models[0].source, DiscoverySource::DynamicApi);
    }

    #[test]
    fn parse_embedding_only_model() {
        let body: Value = serde_json::json!({
            "models": [{
                "name": "models/text-embedding-004",
                "displayName": "Text Embedding 004",
                "inputTokenLimit": 2048,
                "outputTokenLimit": 1,
                "supportedGenerationMethods": ["embedContent"]
            }]
        });
        let models = parse_gemini_models_response(
            &body,
            "vertexai",
            DiscoverySource::DynamicApi,
            Utc::now(),
        );
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].model_type, ModelType::Embedding);
    }
}
