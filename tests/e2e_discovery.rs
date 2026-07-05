//! End-to-end tests for the Model Discovery system.
//!
//! These tests verify the full discovery pipeline:
//! - Static registry correctness
//! - Service construction and provider registration
//! - Capability filtering
//! - Caching behavior
//! - Concurrent access safety
//!
//! Tests that require live API keys are `#[ignore]`-gated.

use edgequake_llm::discovery::{
    all_static_models, static_lookup, unknown_model, CapabilityFilter, DiscoveredModel,
    DiscoverySource, ModelDiscoveryService,
};
use edgequake_llm::model_config::ModelType;
use std::time::Duration;

// ============================================================================
// Static Registry Tests
// ============================================================================

#[test]
fn test_static_registry_has_all_major_providers() {
    let models = all_static_models();
    let providers: Vec<&str> = models.iter().map(|m| m.provider.as_str()).collect();

    assert!(providers.contains(&"openai"), "Missing OpenAI models");
    assert!(providers.contains(&"anthropic"), "Missing Anthropic models");
    assert!(providers.contains(&"gemini"), "Missing Gemini models");
    assert!(providers.contains(&"mistral"), "Missing Mistral models");
    assert!(providers.contains(&"xai"), "Missing xAI models");
    assert!(providers.contains(&"cohere"), "Missing Cohere models");
    assert!(providers.contains(&"nvidia"), "Missing NVIDIA models");
}

#[test]
fn test_static_registry_model_count() {
    let models = all_static_models();
    assert!(
        models.len() >= 30,
        "Expected at least 30 models, got {}",
        models.len()
    );
}

#[test]
fn test_static_registry_no_empty_fields() {
    for model in all_static_models() {
        assert!(!model.id.is_empty(), "Model has empty ID");
        assert!(!model.name.is_empty(), "Model {} has empty name", model.id);
        assert!(
            !model.provider.is_empty(),
            "Model {} has empty provider",
            model.id
        );
        assert_eq!(
            model.source,
            DiscoverySource::StaticRegistry,
            "Model {} has wrong source",
            model.id
        );
    }
}

#[test]
fn test_static_registry_context_lengths_positive() {
    for model in all_static_models() {
        if model.model_type == ModelType::Llm {
            assert!(
                model.context_length > 0,
                "LLM model {} has zero context length",
                model.id
            );
            assert!(
                model.max_output_tokens > 0,
                "LLM model {} has zero max output tokens",
                model.id
            );
        }
    }
}

#[test]
fn test_static_lookup_known_models() {
    let cases = vec![
        ("openai", "gpt-5.5"),
        ("openai", "gpt-4.1"),
        ("openai", "o3"),
        ("anthropic", "claude-fable-5"),
        ("anthropic", "claude-opus-4-8"),
        ("anthropic", "claude-sonnet-5"),
        ("gemini", "gemini-2.5-flash"),
        ("gemini", "gemini-2.5-pro"),
        ("mistral", "codestral-latest"),
        ("xai", "grok-4.3"),
        ("cohere", "command-a-plus-05-2026"),
        ("nvidia", "nvidia/llama-3.3-nemotron-super-49b-v1"),
    ];

    for (provider, model_id) in cases {
        assert!(
            static_lookup(provider, model_id).is_some(),
            "static_lookup({}, {}) returned None",
            provider,
            model_id
        );
    }
}

#[test]
fn test_static_lookup_vertexai_shares_gemini_models() {
    let gemini = static_lookup("gemini", "gemini-2.5-flash");
    let vertex = static_lookup("vertexai", "gemini-2.5-flash");
    assert!(gemini.is_some());
    assert!(vertex.is_some());
    assert_eq!(
        gemini.unwrap().context_length,
        vertex.unwrap().context_length
    );
}

#[test]
fn test_unknown_model_returns_zero_capabilities() {
    let model = unknown_model("test-provider", "nonexistent-model");
    assert_eq!(model.context_length, 0);
    assert_eq!(model.max_output_tokens, 0);
    assert!(!model.capabilities.supports_vision);
    assert!(!model.capabilities.supports_function_calling);
    assert!(!model.capabilities.supports_thinking);
    assert_eq!(model.source, DiscoverySource::Unknown);
}

// ============================================================================
// CapabilityFilter Tests
// ============================================================================

fn make_model(
    id: &str,
    provider: &str,
    ctx: usize,
    vision: bool,
    tools: bool,
    thinking: bool,
) -> DiscoveredModel {
    use edgequake_llm::model_config::ModelCapabilities;

    DiscoveredModel {
        id: id.into(),
        name: id.into(),
        provider: provider.into(),
        context_length: ctx,
        max_output_tokens: 4096,
        capabilities: ModelCapabilities {
            supports_vision: vision,
            supports_function_calling: tools,
            supports_thinking: thinking,
            supports_streaming: true,
            supports_json_mode: true,
            ..Default::default()
        },
        cost_per_m_input: Some(2.0),
        cost_per_m_output: Some(8.0),
        ..Default::default()
    }
}

#[test]
fn test_filter_empty_matches_everything() {
    let filter = CapabilityFilter::default();
    let model = make_model("any", "any", 128_000, false, false, false);
    assert!(filter.matches(&model));
}

#[test]
fn test_filter_combined_capabilities() {
    let filter = CapabilityFilter {
        requires_vision: Some(true),
        requires_tools: Some(true),
        requires_thinking: Some(true),
        min_context_length: Some(500_000),
        ..Default::default()
    };

    let matching = make_model("big", "anthropic", 1_000_000, true, true, true);
    assert!(filter.matches(&matching));

    let too_small = make_model("small", "anthropic", 200_000, true, true, true);
    assert!(!filter.matches(&too_small));

    let no_vision = make_model("blind", "anthropic", 1_000_000, false, true, true);
    assert!(!filter.matches(&no_vision));
}

#[test]
fn test_filter_cost_ceiling() {
    let filter = CapabilityFilter {
        max_cost_per_m_input: Some(1.0),
        ..Default::default()
    };

    let cheap = {
        let mut m = make_model("cheap", "openai", 128_000, false, false, false);
        m.cost_per_m_input = Some(0.15);
        m
    };
    assert!(filter.matches(&cheap));

    let expensive = make_model("expensive", "openai", 128_000, true, true, false);
    assert!(!filter.matches(&expensive));
}

#[test]
fn test_filter_exclude_deprecated() {
    let filter = CapabilityFilter {
        exclude_deprecated: Some(true),
        ..Default::default()
    };

    let mut model = make_model("old", "openai", 128_000, true, true, false);
    model.deprecated = true;
    assert!(!filter.matches(&model));

    model.deprecated = false;
    assert!(filter.matches(&model));
}

#[test]
fn test_filter_by_provider() {
    let filter = CapabilityFilter {
        provider: Some("anthropic".into()),
        ..Default::default()
    };

    let claude = make_model("claude", "anthropic", 200_000, true, true, true);
    assert!(filter.matches(&claude));

    let gpt = make_model("gpt", "openai", 200_000, true, true, false);
    assert!(!filter.matches(&gpt));
}

#[test]
fn test_filter_by_model_type() {
    let filter = CapabilityFilter {
        model_type: Some(ModelType::Embedding),
        ..Default::default()
    };

    let mut embedding = make_model("embed", "openai", 8191, false, false, false);
    embedding.model_type = ModelType::Embedding;
    assert!(filter.matches(&embedding));

    let llm = make_model("llm", "openai", 128_000, true, true, false);
    assert!(!filter.matches(&llm));
}

// ============================================================================
// Service Tests (no live API keys needed)
// ============================================================================

#[tokio::test]
async fn test_service_default_registers_all_providers() {
    let service = ModelDiscoveryService::new();
    let ids = service.provider_ids();

    assert!(ids.contains(&"openai"));
    assert!(ids.contains(&"anthropic"));
    assert!(ids.contains(&"gemini"));
    assert!(ids.contains(&"ollama"));
    assert!(ids.contains(&"lmstudio"));
    assert!(ids.contains(&"openrouter"));
    assert!(ids.contains(&"mistral"));
    assert!(ids.contains(&"nvidia"));
    assert!(ids.contains(&"xai"));
}

#[tokio::test]
async fn test_service_builder_no_defaults() {
    let service = ModelDiscoveryService::builder().no_defaults().build();
    assert!(service.provider_ids().is_empty());
}

#[tokio::test]
async fn test_service_builder_custom_ttl() {
    let service = ModelDiscoveryService::builder()
        .cache_ttl(Duration::from_secs(30))
        .no_defaults()
        .build();
    assert!(service.provider_ids().is_empty());
}

#[tokio::test]
async fn test_service_xai_static_discovery() {
    let service = ModelDiscoveryService::builder()
        .no_defaults()
        .provider(Box::new(
            edgequake_llm::discovery::providers::xai::XAIDiscovery::new(),
        ))
        .build();

    let models = service.discover_all().await.unwrap();
    assert!(
        !models.is_empty(),
        "xAI static discovery returned no models"
    );
    assert!(
        models.iter().all(|m| m.provider == "xai"),
        "All xAI models should have provider 'xai'"
    );
    assert!(
        models.iter().any(|m| m.id == "grok-4.3"),
        "xAI should include grok-4.3"
    );
}

#[tokio::test]
async fn test_service_find_reasoning_models_from_static() {
    let service = ModelDiscoveryService::builder()
        .no_defaults()
        .provider(Box::new(
            edgequake_llm::discovery::providers::xai::XAIDiscovery::new(),
        ))
        .build();

    let filter = CapabilityFilter {
        requires_thinking: Some(true),
        ..Default::default()
    };

    let models = service.find_models(&filter).await.unwrap();
    assert!(
        !models.is_empty(),
        "Should find at least one reasoning model from xAI"
    );
}

#[tokio::test]
async fn test_service_cache_invalidation() {
    let service = ModelDiscoveryService::builder()
        .no_defaults()
        .provider(Box::new(
            edgequake_llm::discovery::providers::xai::XAIDiscovery::new(),
        ))
        .build();

    let first = service.discover_all().await.unwrap();
    assert!(!first.is_empty());

    service.invalidate_cache("xai").await;

    let second = service.discover_all().await.unwrap();
    assert_eq!(first.len(), second.len());
}

#[tokio::test]
async fn test_service_invalidate_all_caches() {
    let service = ModelDiscoveryService::builder()
        .no_defaults()
        .provider(Box::new(
            edgequake_llm::discovery::providers::xai::XAIDiscovery::new(),
        ))
        .build();

    let _ = service.discover_all().await.unwrap();
    service.invalidate_all_caches().await;

    let models = service.discover_all().await.unwrap();
    assert!(!models.is_empty());
}

#[tokio::test]
async fn test_service_get_model_found() {
    let service = ModelDiscoveryService::builder()
        .no_defaults()
        .provider(Box::new(
            edgequake_llm::discovery::providers::xai::XAIDiscovery::new(),
        ))
        .build();

    let model = service.get_model("xai", "grok-4.3").await.unwrap();
    assert!(model.is_some());
    let m = model.unwrap();
    assert_eq!(m.id, "grok-4.3");
    assert_eq!(m.provider, "xai");
    assert_eq!(m.context_length, 1_000_000);
}

#[tokio::test]
async fn test_service_get_model_not_found() {
    let service = ModelDiscoveryService::builder()
        .no_defaults()
        .provider(Box::new(
            edgequake_llm::discovery::providers::xai::XAIDiscovery::new(),
        ))
        .build();

    let model = service.get_model("xai", "nonexistent-model").await.unwrap();
    assert!(model.is_none());
}

#[tokio::test]
async fn test_service_discover_for_unregistered_provider() {
    let service = ModelDiscoveryService::builder().no_defaults().build();
    let models = service.discover_for_provider("nonexistent").await.unwrap();
    assert!(models.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_service_concurrent_access() {
    use std::sync::Arc;

    let service = Arc::new(
        ModelDiscoveryService::builder()
            .no_defaults()
            .provider(Box::new(
                edgequake_llm::discovery::providers::xai::XAIDiscovery::new(),
            ))
            .build(),
    );

    let mut handles = Vec::new();
    for _ in 0..10 {
        let svc = Arc::clone(&service);
        handles.push(tokio::spawn(
            async move { svc.discover_all().await.unwrap() },
        ));
    }

    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    for models in &results {
        assert_eq!(
            models.len(),
            results[0].len(),
            "Concurrent access should return consistent results"
        );
    }
}

// ============================================================================
// Provider Catalog & Static Capability Search
// ============================================================================

#[test]
fn test_provider_factory_list_providers_matches_catalog() {
    use edgequake_llm::{ProviderCatalog, ProviderFactory};

    assert_eq!(
        ProviderFactory::list_providers(),
        ProviderCatalog::list_llm_providers()
    );
    assert!(ProviderFactory::list_providers().contains(&"cohere"));
    assert!(ProviderFactory::list_providers().contains(&"nvidia"));
}

#[test]
fn test_find_static_models_by_capability() {
    use edgequake_llm::{
        find_static_models, CapabilityFilter, ModelCapability, ModelDiscoveryService,
    };

    let filter = CapabilityFilter::default()
        .requiring(ModelCapability::Thinking)
        .with_provider("anthropic")
        .excluding_deprecated();

    let models = find_static_models(&filter);
    assert!(!models.is_empty());
    assert!(models.iter().all(|m| m.provider == "anthropic"));
    assert!(models.iter().all(|m| m.capabilities.supports_thinking));

    let service = ModelDiscoveryService::new();
    assert_eq!(service.find_models_static(&filter).len(), models.len());
}

#[test]
fn test_imagegen_factory_lists_providers() {
    use edgequake_llm::ImageGenFactory;

    let providers = ImageGenFactory::list_providers();
    assert!(providers.contains(&"fal-ai"));
    assert!(providers.contains(&"openai"));
}

// ============================================================================
// Model Name Search
// ============================================================================

#[test]
fn test_search_static_models_by_substring() {
    use edgequake_llm::{search_static_models, ModelSearchQuery};

    let query = ModelSearchQuery::new("claude-opus");
    let hits = search_static_models(&query);
    assert!(!hits.is_empty());
    assert!(hits.iter().all(|h| h.model.id.contains("claude-opus")));
}

#[test]
fn test_search_static_models_fuzzy_typo() {
    use edgequake_llm::{search_static_models, ModelSearchQuery};

    let query = ModelSearchQuery::new("gpt41").fuzzy(true);
    let hits = search_static_models(&query);
    assert!(hits.iter().any(|h| h.model.id == "gpt-4.1"));
}

#[test]
fn test_search_qualified_provider_model() {
    use edgequake_llm::{search_static_models, ModelSearchQuery};

    let query = ModelSearchQuery::new("openai/gpt-4.1");
    let hits = search_static_models(&query);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].model.provider, "openai");
    assert_eq!(hits[0].model.id, "gpt-4.1");
}

#[test]
fn test_static_lookup_by_name() {
    use edgequake_llm::static_lookup_by_name;

    assert!(static_lookup_by_name("openai", "gpt-4.1").is_some());
    assert!(static_lookup_by_name("openai", "GPT-4.1").is_some());
}

#[test]
fn test_search_static_by_input_and_output_length() {
    use edgequake_llm::{search_static_models, ModelSearchQuery};

    let query = ModelSearchQuery::new("claude")
        .with_min_context_length(500_000)
        .with_min_output_tokens(32_768);
    let hits = search_static_models(&query);
    assert!(!hits.is_empty());
    assert!(hits.iter().all(|h| h.model.context_length >= 500_000));
    assert!(hits.iter().all(|h| h.model.max_output_tokens >= 32_768));
}

// ============================================================================
// Live API Discovery Tests (require API keys)
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_live_openai_discovery() {
    let _ = dotenvy::dotenv();
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    let service = ModelDiscoveryService::new();
    let models = service.discover_for_provider("openai").await.unwrap();
    assert!(!models.is_empty(), "OpenAI should discover models");
    assert!(
        models.iter().all(|m| m.provider == "openai"),
        "All models should be from openai"
    );
}

#[tokio::test]
#[ignore = "Requires ANTHROPIC_API_KEY"]
async fn test_live_anthropic_discovery() {
    let _ = dotenvy::dotenv();
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        return;
    }

    let service = ModelDiscoveryService::new();
    let models = service.discover_for_provider("anthropic").await.unwrap();
    assert!(!models.is_empty(), "Anthropic should discover models");
}

#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY or GOOGLE_API_KEY"]
async fn test_live_gemini_discovery() {
    let _ = dotenvy::dotenv();
    if std::env::var("GEMINI_API_KEY").is_err() && std::env::var("GOOGLE_API_KEY").is_err() {
        return;
    }

    let service = ModelDiscoveryService::new();
    let models = service.discover_for_provider("gemini").await.unwrap();
    assert!(!models.is_empty(), "Gemini should discover models");
}

#[tokio::test]
#[ignore = "Requires MISTRAL_API_KEY"]
async fn test_live_mistral_discovery() {
    let _ = dotenvy::dotenv();
    if std::env::var("MISTRAL_API_KEY").is_err() {
        return;
    }

    let service = ModelDiscoveryService::new();
    let models = service.discover_for_provider("mistral").await.unwrap();
    assert!(!models.is_empty(), "Mistral should discover models");
}
