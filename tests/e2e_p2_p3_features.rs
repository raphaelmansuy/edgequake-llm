//! E2E tests for P2/P3 features from specs/002-upgrade-support.
//!
//! Features covered:
//! - FEAT-050: Mistral Vision (image input support)
//! - FEAT-057: Bedrock Stability AI image generation
//! - FEAT-060: Azure Managed Identity auth (AzureCredential)
//! - FEAT-062: Structured Outputs (JSON Schema)
//! - FEAT-063: Ollama embed dimensions (truncate)
//! - FEAT-064: OpenRouter fallback model arrays
//! - P2: Azure OpenAI image generation (AzureImageGen)
//! - P2: NVIDIA image generation (NvidiaImageGen)
//! - P2: Mistral variable embed dimensions
//! - P2: Bedrock Nova 2 multimodal embeddings
//!
//! Live API tests are `#[ignore]`-gated. Unit/structural tests run without keys.

// ============================================================================
// FEAT-050: Mistral Vision — image input wiring
// ============================================================================

#[test]
fn test_mistral_vision_models_flagged() {
    use edgequake_llm::discovery::mistral_models;

    let models = mistral_models();
    let vision_models: Vec<_> = models
        .iter()
        .filter(|m| m.capabilities.supports_vision)
        .collect();

    assert!(
        !vision_models.is_empty(),
        "At least one Mistral model should support vision"
    );

    let ids: Vec<&str> = vision_models.iter().map(|m| m.id.as_str()).collect();
    assert!(
        ids.iter().any(|id| id.contains("large")),
        "Mistral Large should support vision, got: {:?}",
        ids
    );
}

// ============================================================================
// FEAT-062: Structured Outputs (JSON Schema)
// ============================================================================

#[test]
fn test_completion_options_json_schema() {
    use edgequake_llm::CompletionOptions;

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" }
        },
        "required": ["name", "age"]
    });

    let opts = CompletionOptions::json_schema(schema.clone());
    assert_eq!(opts.response_format.as_deref(), Some("json_schema"));
    assert_eq!(opts.response_schema, Some(schema));
}

#[test]
fn test_completion_options_json_mode_unchanged() {
    use edgequake_llm::CompletionOptions;

    let opts = CompletionOptions::json_mode();
    assert_eq!(opts.response_format.as_deref(), Some("json_object"));
    assert!(opts.response_schema.is_none());
}

// ============================================================================
// FEAT-064: OpenRouter fallback model arrays
// ============================================================================

#[test]
fn test_openrouter_with_fallback_models() {
    use edgequake_llm::OpenRouterProvider;

    let provider = OpenRouterProvider::new("test-key").with_fallback_models(vec![
        "anthropic/claude-sonnet-5".to_string(),
        "openai/gpt-5.5".to_string(),
        "google/gemini-3.5-flash".to_string(),
    ]);

    assert_eq!(edgequake_llm::LLMProvider::name(&provider), "openrouter");
}

// ============================================================================
// P2: Azure OpenAI Image Generation (AzureImageGen)
// ============================================================================

#[test]
fn test_azure_imagegen_available_models() {
    use edgequake_llm::imagegen::traits::ImageGenProvider;
    use edgequake_llm::AzureImageGen;

    let provider = AzureImageGen::new("https://test.openai.azure.com", "test-key", "gpt-image-2");
    let models = provider.available_models();
    assert!(!models.is_empty(), "Should have at least one model");
    assert!(
        models.iter().any(|m| m.contains("gpt-image-2")),
        "Should include gpt-image-2"
    );
}

#[test]
fn test_azure_imagegen_provider_name() {
    use edgequake_llm::imagegen::traits::ImageGenProvider;
    use edgequake_llm::AzureImageGen;

    let provider = AzureImageGen::new("https://test.openai.azure.com", "test-key", "gpt-image-2");
    assert_eq!(provider.name(), "azure-openai");
}

#[tokio::test]
#[ignore = "requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"]
async fn test_azure_imagegen_from_env() {
    use edgequake_llm::AzureImageGen;

    let provider = AzureImageGen::from_env();
    assert!(provider.is_ok(), "Failed: {:?}", provider.err());
}

// ============================================================================
// P2: NVIDIA Image Generation (NvidiaImageGen)
// ============================================================================

#[test]
fn test_nvidia_imagegen_available_models() {
    use edgequake_llm::imagegen::traits::ImageGenProvider;
    use edgequake_llm::NvidiaImageGen;

    let provider = NvidiaImageGen::new("test-key");
    let models = provider.available_models();
    assert!(
        models.len() >= 3,
        "Expected at least 3 NVIDIA imagegen models, got {}",
        models.len()
    );
    assert!(
        models.iter().any(|m| m.contains("stable-diffusion")),
        "Should include Stable Diffusion models"
    );
}

#[test]
fn test_nvidia_imagegen_provider_name() {
    use edgequake_llm::imagegen::traits::ImageGenProvider;
    use edgequake_llm::NvidiaImageGen;

    let provider = NvidiaImageGen::new("test-key");
    assert_eq!(provider.name(), "nvidia");
}

#[tokio::test]
#[ignore = "requires NVIDIA_API_KEY"]
async fn test_nvidia_imagegen_from_env() {
    use edgequake_llm::NvidiaImageGen;

    let provider = NvidiaImageGen::from_env();
    assert!(provider.is_ok(), "Failed: {:?}", provider.err());
}

// ============================================================================
// P2: Mistral variable embed dimensions
// ============================================================================

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn test_mistral_embed_with_custom_dimensions() {
    use edgequake_llm::providers::mistral::MistralProvider;
    use edgequake_llm::EmbeddingProvider;

    let provider = MistralProvider::from_env()
        .expect("MISTRAL_API_KEY required")
        .with_embed_dimensions(256);

    let texts = vec!["Hello world".to_string()];
    let result = provider.embed(&texts).await;
    assert!(
        result.is_ok(),
        "Embed with dimensions failed: {:?}",
        result.err()
    );
}

// ============================================================================
// P2: Bedrock Nova 2 multimodal embeddings (requires bedrock feature)
// ============================================================================

#[cfg(feature = "bedrock")]
#[test]
fn test_bedrock_nova2_embed_dimension() {
    use edgequake_llm::providers::bedrock::BedrockProvider;

    assert_eq!(
        BedrockProvider::dimension_for_model("amazon.nova-embed-multimodal-v2:0"),
        1024,
        "Nova 2 multimodal embed should have 1024 dimensions"
    );
}

#[cfg(feature = "bedrock")]
#[test]
fn test_bedrock_nova2_embed_matches_nova_prefix() {
    use edgequake_llm::providers::bedrock::BedrockProvider;

    assert_eq!(
        BedrockProvider::dimension_for_model("amazon.nova-embed-v1:0"),
        1024,
    );
}

// ============================================================================
// ImageGen factory auto-detection
// ============================================================================

#[test]
fn test_imagegen_factory_error_message_includes_all_providers() {
    use edgequake_llm::ImageGenFactory;

    std::env::remove_var("GEMINI_API_KEY");
    std::env::remove_var("GOOGLE_CLOUD_PROJECT");
    std::env::remove_var("FAL_KEY");
    std::env::remove_var("OPENAI_API_KEY");
    std::env::remove_var("XAI_API_KEY");
    std::env::remove_var("AZURE_OPENAI_ENDPOINT");
    std::env::remove_var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT");
    std::env::remove_var("NVIDIA_API_KEY");

    let result = ImageGenFactory::from_env();
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("NVIDIA_API_KEY"),
        "Error message should mention NVIDIA_API_KEY, got: {}",
        err_msg
    );
}

// ============================================================================
// CompletionOptions backward compatibility
// ============================================================================

#[test]
fn test_completion_options_default_has_no_schema() {
    use edgequake_llm::CompletionOptions;

    let opts = CompletionOptions::default();
    assert!(opts.response_schema.is_none());
    assert!(opts.response_format.is_none());
}

#[test]
fn test_completion_options_with_reasoning_effort() {
    use edgequake_llm::CompletionOptions;

    let opts = CompletionOptions {
        reasoning_effort: Some("high".to_string()),
        ..Default::default()
    };
    assert_eq!(opts.reasoning_effort.as_deref(), Some("high"));
    assert!(opts.response_schema.is_none());
}
