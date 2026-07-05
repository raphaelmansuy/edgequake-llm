//! End-to-end tests for the provider upgrade plan (specs/002-upgrade-support).
//!
//! Tests are organized by feature:
//! - FEAT-040: Cohere provider (LLM + Embeddings)
//! - FEAT-042: NVIDIA embeddings
//! - FEAT-043: OpenAI image generation
//! - FEAT-044: Model catalog updates
//! - FEAT-051: Mistral reasoning_effort
//! - FEAT-054: xAI image generation
//! - FEAT-055: Gemini embedding-2 model catalog
//!
//! Live API tests are `#[ignore]`-gated. Unit tests run without keys.

use edgequake_llm::discovery::{all_static_models, cohere_models, nvidia_models, static_lookup};
use edgequake_llm::model_config::ModelType;
use edgequake_llm::ProviderType;

// ============================================================================
// FEAT-040: Cohere Provider
// ============================================================================

#[test]
fn test_cohere_provider_type_from_str() {
    assert_eq!(ProviderType::from_str("cohere"), Some(ProviderType::Cohere));
    assert_eq!(
        ProviderType::from_str("cohere-ai"),
        Some(ProviderType::Cohere)
    );
}

#[test]
fn test_cohere_static_models_present() {
    let models = cohere_models();
    assert!(
        models.len() >= 3,
        "Expected at least 3 Cohere models, got {}",
        models.len()
    );

    let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
    assert!(
        ids.contains(&"command-a-plus-05-2026"),
        "Missing Command A Plus"
    );
    assert!(ids.contains(&"embed-v4.0"), "Missing Embed v4.0");
}

#[test]
fn test_cohere_static_lookup() {
    let model = static_lookup("cohere", "command-a-plus-05-2026");
    assert!(model.is_some(), "Cohere Command A+ not in static registry");
    let model = model.unwrap();
    assert_eq!(model.context_length, 256_000);
    assert!(model.capabilities.supports_function_calling);
    assert!(model.capabilities.supports_streaming);
}

#[test]
fn test_cohere_embed_model_in_registry() {
    let model = static_lookup("cohere", "embed-v4.0");
    assert!(model.is_some(), "Cohere embed-v4.0 not in static registry");
    let model = model.unwrap();
    assert_eq!(model.model_type, ModelType::Embedding);
}

#[test]
#[ignore = "requires COHERE_API_KEY"]
fn test_cohere_provider_from_env() {
    let provider = edgequake_llm::CohereProvider::from_env();
    assert!(
        provider.is_ok(),
        "Failed to create Cohere provider: {:?}",
        provider.err()
    );

    let provider = provider.unwrap();
    assert_eq!(edgequake_llm::LLMProvider::name(&provider), "cohere");
}

#[tokio::test]
#[ignore = "requires COHERE_API_KEY"]
async fn test_cohere_chat_completion() {
    use edgequake_llm::{ChatMessage, ChatRole, LLMProvider};

    let provider = edgequake_llm::CohereProvider::from_env().expect("COHERE_API_KEY required");
    let messages = vec![ChatMessage {
        role: ChatRole::User,
        content: "Say exactly: hello world".to_string(),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        cache_control: None,
        images: None,
    }];
    let response = provider.chat(&messages, None).await;
    assert!(response.is_ok(), "Chat failed: {:?}", response.err());
    let response = response.unwrap();
    assert!(
        !response.content.is_empty(),
        "Response content should not be empty"
    );
}

#[tokio::test]
#[ignore = "requires COHERE_API_KEY"]
async fn test_cohere_embeddings() {
    use edgequake_llm::EmbeddingProvider;

    let provider = edgequake_llm::CohereProvider::from_env().expect("COHERE_API_KEY required");
    let texts = vec!["Hello world".to_string(), "Goodbye world".to_string()];
    let result = provider.embed(&texts).await;
    assert!(result.is_ok(), "Embed failed: {:?}", result.err());
    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 2);
    assert!(
        !embeddings[0].is_empty(),
        "Embedding vector should not be empty"
    );
}

// ============================================================================
// FEAT-042: NVIDIA Embeddings
// ============================================================================

#[test]
fn test_nvidia_static_models_present() {
    let models = nvidia_models();
    assert!(
        !models.is_empty(),
        "NVIDIA static models should not be empty"
    );

    let has_embed = models.iter().any(|m| m.model_type == ModelType::Embedding);
    assert!(has_embed, "NVIDIA should have at least one embedding model");
}

#[test]
fn test_nvidia_static_lookup() {
    let model = static_lookup("nvidia", "nvidia/llama-3.3-nemotron-super-49b-v1");
    assert!(model.is_some(), "Nemotron Super 49B not in static registry");
    let model = model.unwrap();
    assert!(model.capabilities.supports_function_calling);
    assert!(model.capabilities.supports_vision);
}

#[test]
fn test_nvidia_embed_model_lookup() {
    let model = static_lookup("nvidia", "nvidia/llama-nemotron-embed-1b-v2");
    assert!(model.is_some(), "Nemotron Embed not in static registry");
    let model = model.unwrap();
    assert_eq!(model.model_type, ModelType::Embedding);
}

// ============================================================================
// FEAT-044: Model Catalog Updates
// ============================================================================

#[test]
fn test_openai_gpt55_in_registry() {
    let model = static_lookup("openai", "gpt-5.5");
    assert!(model.is_some(), "GPT-5.5 should be in static registry");
    let model = model.unwrap();
    assert_eq!(model.context_length, 1_048_576);
    assert!(model.capabilities.supports_vision);
    assert!(!model.deprecated);
}

#[test]
fn test_openai_gpt54_in_registry() {
    let model = static_lookup("openai", "gpt-5.4");
    assert!(model.is_some(), "GPT-5.4 should be in static registry");
    let model = model.unwrap();
    assert_eq!(model.context_length, 1_048_576);
    assert!(!model.deprecated);
}

#[test]
fn test_openai_gpt41_in_registry() {
    let model = static_lookup("openai", "gpt-4.1");
    assert!(model.is_some(), "GPT-4.1 should be in registry");
    let model = model.unwrap();
    assert!(!model.deprecated);
}

#[test]
fn test_all_static_models_include_new_providers() {
    let all = all_static_models();
    let providers: std::collections::HashSet<&str> =
        all.iter().map(|m| m.provider.as_str()).collect();

    assert!(
        providers.contains("cohere"),
        "all_static_models should include cohere"
    );
    assert!(
        providers.contains("nvidia"),
        "all_static_models should include nvidia"
    );
}

// ============================================================================
// Provider type round-trip tests
// ============================================================================

#[test]
fn test_provider_type_cohere_variant() {
    let pt = ProviderType::from_str("cohere");
    assert_eq!(pt, Some(ProviderType::Cohere));
}

#[test]
fn test_provider_type_nvidia_variant() {
    let pt = ProviderType::from_str("nvidia");
    assert_eq!(pt, Some(ProviderType::Nvidia));
    let pt = ProviderType::from_str("nvidia-nim");
    assert_eq!(pt, Some(ProviderType::Nvidia));
    let pt = ProviderType::from_str("nim");
    assert_eq!(pt, Some(ProviderType::Nvidia));
}
