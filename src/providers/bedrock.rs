//! AWS Bedrock Runtime provider via the Converse API.
//!
//! This provider implements the `LLMProvider` trait using the AWS Bedrock Runtime
//! Converse API, which provides a model-agnostic interface for chat completions
//! across all Bedrock-hosted models (Anthropic Claude, Amazon Nova, Meta Llama,
//! Mistral, Cohere, etc.).
//!
//! # Feature Gate
//!
//! This module is only available when the `bedrock` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! edgequake-llm = { version = "0.2", features = ["bedrock"] }
//! ```
//!
//! # Environment Variables
//!
//! - `AWS_BEDROCK_MODEL`: Model ID (default: `amazon.nova-lite-v1:0`). Bare model
//!   IDs are automatically resolved to inference profile IDs based on the region
//!   (e.g., `eu.amazon.nova-lite-v1:0` in `eu-west-1`).
//! - `AWS_REGION` / `AWS_DEFAULT_REGION`: AWS region (default: `us-east-1`)
//! - Standard AWS credential chain (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
//!   `AWS_SESSION_TOKEN`, `AWS_PROFILE`, IAM roles, etc.)
//!
//! # Inference Profiles
//!
//! Modern Bedrock models require cross-region inference profile IDs instead of
//! bare model IDs. This provider automatically resolves bare model IDs (e.g.,
//! `amazon.nova-lite-v1:0`) to the appropriate inference profile based on the
//! configured AWS region:
//!
//! | Region prefix | Resolved prefix | Example |
//! |---|---|---|
//! | `us-*` | `us.` | `us.amazon.nova-lite-v1:0` |
//! | `eu-*` | `eu.` | `eu.amazon.nova-lite-v1:0` |
//! | `ap-*` | `ap.` | `ap.amazon.nova-lite-v1:0` |
//!
//! You can also pass a fully-qualified inference profile ID (e.g.,
//! `us.anthropic.claude-sonnet-4-20250514-v1:0`) or an ARN directly — these
//! are used as-is without modification.
//!
//! # Example
//!
//! ```rust,ignore
//! use edgequake_llm::BedrockProvider;
//! use edgequake_llm::traits::{ChatMessage, LLMProvider};
//!
//! let provider = BedrockProvider::from_env().await?;
//! let messages = vec![ChatMessage::user("Hello!")];
//! let response = provider.chat(&messages, None).await?;
//! println!("{}", response.content);
//! ```

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_config::SdkConfig;
use aws_sdk_bedrockruntime::error::SdkError;
use aws_sdk_bedrockruntime::operation::converse::ConverseError;
use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError;
use aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError;
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::types::{
    ContentBlock, ConversationRole, ConverseOutput, ImageBlock, ImageFormat, ImageSource,
    InferenceConfiguration, Message, StopReason, SystemContentBlock, Tool,
    ToolChoice as BedrockToolChoice, ToolConfiguration, ToolInputSchema, ToolSpecification,
    ToolUseBlock,
};
use aws_sdk_bedrockruntime::Client;
use aws_smithy_types::Document;
use base64::Engine as _;
use futures::stream::BoxStream;
use tracing::{debug, instrument};

use crate::error::{LlmError, Result};
use crate::traits::{
    ChatMessage, ChatRole, CompletionOptions, EmbeddingProvider, LLMProvider, LLMResponse,
    ToolCall as EdgequakeToolCall, ToolChoice as EdgequakeToolChoice,
    ToolDefinition as EdgequakeToolDefinition,
};

// ============================================================================
// Typed Error Conversions  (G2 — ADR-001 §6.3)
//
// Convert AWS SDK service errors into the appropriate LlmError variant so that
// callers can distinguish throttling, validation errors, and auth failures.
// All conversions are internal to this module.
// ============================================================================

impl From<SdkError<ConverseError>> for LlmError {
    fn from(err: SdkError<ConverseError>) -> Self {
        match err.as_service_error() {
            Some(se) => match se {
                ConverseError::ThrottlingException(e) => {
                    LlmError::RateLimited(e.message().unwrap_or("throttled").to_string())
                }
                ConverseError::ModelTimeoutException(_) => LlmError::Timeout,
                ConverseError::ValidationException(e) => {
                    LlmError::InvalidRequest(e.message().unwrap_or("validation error").to_string())
                }
                ConverseError::AccessDeniedException(e) => {
                    LlmError::AuthError(e.message().unwrap_or("access denied").to_string())
                }
                other => LlmError::ProviderError(format!(
                    "Bedrock Converse: {} — {}",
                    other.meta().code().unwrap_or("Unknown"),
                    other.meta().message().unwrap_or("no message"),
                )),
            },
            None => LlmError::ProviderError(format!("Bedrock Converse SDK error: {err:?}")),
        }
    }
}

impl From<SdkError<ConverseStreamError>> for LlmError {
    fn from(err: SdkError<ConverseStreamError>) -> Self {
        match err.as_service_error() {
            Some(se) => match se {
                ConverseStreamError::ThrottlingException(e) => {
                    LlmError::RateLimited(e.message().unwrap_or("throttled").to_string())
                }
                ConverseStreamError::ValidationException(e) => {
                    LlmError::InvalidRequest(e.message().unwrap_or("validation error").to_string())
                }
                other => LlmError::ProviderError(format!(
                    "Bedrock ConverseStream: {} — {}",
                    other.meta().code().unwrap_or("Unknown"),
                    other.meta().message().unwrap_or("no message"),
                )),
            },
            None => LlmError::ProviderError(format!("Bedrock ConverseStream SDK error: {err:?}")),
        }
    }
}

impl From<SdkError<InvokeModelError>> for LlmError {
    fn from(err: SdkError<InvokeModelError>) -> Self {
        match err.as_service_error() {
            Some(se) => match se {
                InvokeModelError::ThrottlingException(e) => {
                    LlmError::RateLimited(e.message().unwrap_or("throttled").to_string())
                }
                InvokeModelError::ValidationException(e) => {
                    LlmError::InvalidRequest(e.message().unwrap_or("validation error").to_string())
                }
                other => LlmError::ProviderError(format!(
                    "Bedrock InvokeModel: {} — {}",
                    other.meta().code().unwrap_or("Unknown"),
                    other.meta().message().unwrap_or("no message"),
                )),
            },
            None => LlmError::ProviderError(format!("Bedrock InvokeModel SDK error: {err:?}")),
        }
    }
}

impl From<aws_sdk_bedrockruntime::error::BuildError> for LlmError {
    fn from(e: aws_sdk_bedrockruntime::error::BuildError) -> Self {
        LlmError::InvalidRequest(format!("Bedrock builder error: {e}"))
    }
}

// ============================================================================
// Constants
// ============================================================================

/// Default Bedrock model (Amazon Nova Lite — works across all regions without
/// geo-restrictions).
const DEFAULT_MODEL: &str = "amazon.nova-lite-v1:0";

/// Default AWS region for Bedrock
const DEFAULT_REGION: &str = "us-east-1";

/// Default max context length (Nova Lite = 300k tokens)
const DEFAULT_MAX_CONTEXT: usize = 300_000;

/// Default Bedrock embedding model (Amazon Titan Embed Text v2 — works across
/// all regions without geo-restrictions, 1024-dimensional vectors).
const DEFAULT_EMBEDDING_MODEL: &str = "amazon.titan-embed-text-v2:0";

/// Default embedding dimension for Titan Embed Text v2
const DEFAULT_EMBEDDING_DIMENSION: usize = 1024;

/// Default max tokens for embedding input (Titan Embed v2 = 8192 tokens)
const DEFAULT_EMBEDDING_MAX_TOKENS: usize = 8192;

const GEOGRAPHY_PREFIXES: &[&str] = &["us.", "eu.", "ap.", "ca.", "sa.", "me.", "af."];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmbeddingRequestShape {
    SingleInputText,
    CohereBatch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ModelRule {
    prefixes: &'static [&'static str],
    context_length: usize,
    requires_inference_profile: bool,
}

impl ModelRule {
    const fn new(
        prefixes: &'static [&'static str],
        context_length: usize,
        requires_inference_profile: bool,
    ) -> Self {
        Self {
            prefixes,
            context_length,
            requires_inference_profile,
        }
    }

    fn matches(self, normalized_model: &str) -> bool {
        self.prefixes
            .iter()
            .any(|prefix| normalized_model.starts_with(prefix))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EmbeddingModelRule {
    prefixes: &'static [&'static str],
    dimension: usize,
    max_tokens: usize,
    request_shape: EmbeddingRequestShape,
}

impl EmbeddingModelRule {
    const fn new(
        prefixes: &'static [&'static str],
        dimension: usize,
        max_tokens: usize,
        request_shape: EmbeddingRequestShape,
    ) -> Self {
        Self {
            prefixes,
            dimension,
            max_tokens,
            request_shape,
        }
    }

    fn matches(self, normalized_model: &str) -> bool {
        self.prefixes
            .iter()
            .any(|prefix| normalized_model.starts_with(prefix))
    }
}

const MODEL_RULES: &[ModelRule] = &[
    ModelRule::new(&["amazon.nova-"], 300_000, true),
    ModelRule::new(
        &[
            "anthropic.claude-3",
            "anthropic.claude-4",
            "anthropic.claude-sonnet-4",
            "anthropic.claude-opus-4",
            "anthropic.claude-haiku-4",
        ],
        200_000,
        true,
    ),
    ModelRule::new(&["anthropic.claude-2"], 100_000, false),
    ModelRule::new(&["mistral.devstral-"], 256_000, false),
    ModelRule::new(&["minimax."], 1_000_000, false),
    ModelRule::new(&["qwen.qwen2-5-", "qwen.qwen3-"], 131_072, false),
    ModelRule::new(
        &[
            "meta.llama",
            "cohere.command-",
            "deepseek.",
            "mistral.pixtral-",
            "mistral.magistral-",
            "writer.",
            "google.gemma-",
            "nvidia.nemotron-",
            "zai.glm-",
            "openai.gpt-oss-",
        ],
        128_000,
        false,
    ),
    ModelRule::new(&["mistral."], 32_000, false),
];

const EMBEDDING_MODEL_RULES: &[EmbeddingModelRule] = &[
    EmbeddingModelRule::new(
        &["amazon.titan-embed-text-v2:0", "amazon.nova-embed-"],
        1024,
        8192,
        EmbeddingRequestShape::SingleInputText,
    ),
    EmbeddingModelRule::new(
        &[
            "amazon.titan-embed-text-v1",
            "amazon.titan-embed-g1-text-02",
        ],
        1536,
        512,
        EmbeddingRequestShape::SingleInputText,
    ),
    EmbeddingModelRule::new(
        &["amazon.titan-embed-image-"],
        1024,
        DEFAULT_EMBEDDING_MAX_TOKENS,
        EmbeddingRequestShape::SingleInputText,
    ),
    EmbeddingModelRule::new(
        &["cohere.embed-english-v3", "cohere.embed-multilingual-v3"],
        1024,
        2048,
        EmbeddingRequestShape::CohereBatch,
    ),
    EmbeddingModelRule::new(
        &["cohere.embed-v4:0"],
        1536,
        2048,
        EmbeddingRequestShape::CohereBatch,
    ),
    EmbeddingModelRule::new(
        &["twelvelabs.marengo-"],
        1024,
        DEFAULT_EMBEDDING_MAX_TOKENS,
        EmbeddingRequestShape::SingleInputText,
    ),
];

#[derive(Debug)]
struct PreparedConverseRequest {
    resolved_model: String,
    bedrock_messages: Vec<Message>,
    system_blocks: Vec<SystemContentBlock>,
    inference_config: Option<InferenceConfiguration>,
    additional_fields: Option<Document>,
}

// ============================================================================
// BedrockProvider
// ============================================================================

/// AWS Bedrock Runtime LLM provider using the Converse API.
///
/// Uses the model-agnostic Converse API which works with all Bedrock models
/// without requiring model-specific payload formatting.
///
/// # Inference Profiles
///
/// Modern Bedrock models require cross-region inference profile IDs
/// (e.g., `us.amazon.nova-lite-v1:0` instead of `amazon.nova-lite-v1:0`).
/// The provider automatically resolves bare model IDs to inference profile
/// IDs based on the configured region:
///
/// | Region prefix | Geography prefix |
/// |---|---|
/// | `us-*` | `us.` |
/// | `eu-*` | `eu.` |
/// | `ap-*` | `ap.` |
/// | `ca-*` | `ca.` |
/// | `sa-*` | `sa.` |
/// | `me-*` | `me.` |
/// | `af-*` | `af.` |
///
/// You can also pass a fully-qualified inference profile ID directly
/// (e.g., `us.anthropic.claude-sonnet-4-20250514-v1:0`) or an ARN.
#[derive(Debug, Clone)]
pub struct BedrockProvider {
    client: Client,
    /// Stored SDK config so `with_retry_config()` can rebuild the client.
    sdk_config: SdkConfig,
    /// The model identifier as provided by the user (bare or prefixed).
    model: String,
    /// The AWS region code (e.g., `us-east-1`, `eu-west-1`).
    region: String,
    max_context_length: usize,
    /// The embedding model ID (e.g., `amazon.titan-embed-text-v2:0`).
    embedding_model: String,
    /// The embedding vector dimension.
    embedding_dimension: usize,
}

impl BedrockProvider {
    /// Create a new Bedrock provider from an existing AWS SDK config.
    ///
    /// # Arguments
    ///
    /// * `sdk_config` - Pre-configured AWS SDK config
    /// * `model` - Bedrock model ID (e.g., `amazon.nova-lite-v1:0`) or inference
    ///   profile ID (e.g., `us.amazon.nova-lite-v1:0`). Bare model IDs are
    ///   automatically resolved to inference profile IDs based on the region.
    pub fn new(sdk_config: &SdkConfig, model: impl Into<String>) -> Self {
        let model = model.into();
        let region = sdk_config
            .region()
            .map(|r| r.to_string())
            .unwrap_or_else(|| DEFAULT_REGION.to_string());
        let max_context_length = Self::context_length_for_model(&model);
        Self {
            client: Client::new(sdk_config),
            sdk_config: sdk_config.clone(),
            model,
            region,
            max_context_length,
            embedding_model: DEFAULT_EMBEDDING_MODEL.to_string(),
            embedding_dimension: DEFAULT_EMBEDDING_DIMENSION,
        }
    }

    /// Create a provider from environment variables (async).
    ///
    /// Uses the standard AWS credential chain and reads:
    /// - `AWS_BEDROCK_MODEL` for the model ID
    /// - `AWS_REGION` / `AWS_DEFAULT_REGION` for the region
    pub async fn from_env() -> Result<Self> {
        let region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|_| DEFAULT_REGION.to_string());

        let model =
            std::env::var("AWS_BEDROCK_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

        let embedding_model = std::env::var("AWS_BEDROCK_EMBEDDING_MODEL")
            .unwrap_or_else(|_| DEFAULT_EMBEDDING_MODEL.to_string());

        let embedding_dimension = Self::dimension_for_model(&embedding_model);

        // G1 (ADR-001 §6.2): use BehaviorVersion::latest() instead of from_env()
        // to ensure the most correct SDK defaults are applied.
        let sdk_config = aws_config::defaults(BehaviorVersion::latest())
            .region(aws_config::Region::new(region.clone()))
            .load()
            .await;

        let max_context_length = Self::context_length_for_model(&model);

        Ok(Self {
            client: Client::new(&sdk_config),
            sdk_config,
            model,
            region,
            max_context_length,
            embedding_model,
            embedding_dimension,
        })
    }

    /// Override the AWS SDK retry configuration.
    ///
    /// The SDK applies 3 retries with adaptive backoff by default.  Use this
    /// to increase retries for bursty workloads or reduce them for latency-
    /// sensitive paths.
    ///
    /// # Example
    /// ```rust,ignore
    /// use aws_config::retry::RetryConfig;
    ///
    /// let provider = BedrockProvider::from_env().await?
    ///     .with_retry_config(RetryConfig::adaptive().with_max_attempts(5));
    /// ```
    pub fn with_retry_config(mut self, config: aws_config::retry::RetryConfig) -> Self {
        let bedrock_config = aws_sdk_bedrockruntime::config::Builder::from(&self.sdk_config)
            .retry_config(config)
            .build();
        self.client = Client::from_conf(bedrock_config);
        self
    }

    /// Set a custom model ID.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        let model = model.into();
        self.max_context_length = Self::context_length_for_model(&model);
        self.model = model;
        self
    }

    /// Set a custom max context length.
    pub fn with_max_context_length(mut self, length: usize) -> Self {
        self.max_context_length = length;
        self
    }

    /// Set a custom embedding model.
    ///
    /// # Supported Embedding Models
    ///
    /// | Model ID | Provider | Dimensions |
    /// |---|---|---|
    /// | `amazon.titan-embed-text-v2:0` | Amazon | 1024 |
    /// | `amazon.titan-embed-text-v1` | Amazon | 1536 |
    /// | `amazon.titan-embed-g1-text-02` | Amazon | 1536 |
    /// | `cohere.embed-english-v3` | Cohere | 1024 |
    /// | `cohere.embed-multilingual-v3` | Cohere | 1024 |
    /// | `cohere.embed-v4:0` | Cohere | 1536 |
    pub fn with_embedding_model(mut self, model: impl Into<String>) -> Self {
        let model = model.into();
        self.embedding_dimension = Self::dimension_for_model(&model);
        self.embedding_model = model;
        self
    }

    /// Set a custom embedding dimension (overrides auto-detection).
    pub fn with_embedding_dimension(mut self, dimension: usize) -> Self {
        self.embedding_dimension = dimension;
        self
    }

    /// Return the configured AWS region code (e.g., `us-east-1`).
    pub fn region(&self) -> &str {
        &self.region
    }

    /// Resolve a bare model ID to an inference profile ID based on the region.
    ///
    /// Modern Bedrock models require cross-region inference profile IDs
    /// (e.g., `us.amazon.nova-lite-v1:0` instead of `amazon.nova-lite-v1:0`).
    /// If the model already has a geography prefix (`us.`, `eu.`, `ap.`, etc.),
    /// an ARN, or a `global.` prefix, it is returned unchanged.
    ///
    /// Geography mapping:
    /// - `us-*` → `us.`
    /// - `eu-*` → `eu.`
    /// - `ap-*` → `ap.`
    /// - `ca-*` → `ca.`
    /// - `sa-*` → `sa.`
    /// - `me-*` → `me.`
    /// - `af-*` → `af.`
    fn resolve_model_id(&self) -> String {
        Self::resolve_model_id_for_region(&self.model, &self.region)
    }

    fn strip_model_qualifier(model: &str) -> &str {
        if model.starts_with("arn:") {
            return model;
        }

        if let Some(stripped) = model.strip_prefix("global.") {
            return stripped;
        }

        GEOGRAPHY_PREFIXES
            .iter()
            .find_map(|prefix| model.strip_prefix(prefix))
            .unwrap_or(model)
    }

    fn model_rule(model: &str) -> Option<ModelRule> {
        let normalized_model = Self::strip_model_qualifier(model).to_ascii_lowercase();
        MODEL_RULES
            .iter()
            .copied()
            .find(|rule| rule.matches(&normalized_model))
    }

    fn embedding_model_rule(model: &str) -> Option<EmbeddingModelRule> {
        let normalized_model = Self::strip_model_qualifier(model).to_ascii_lowercase();
        EMBEDDING_MODEL_RULES
            .iter()
            .copied()
            .find(|rule| rule.matches(&normalized_model))
    }

    fn geography_prefix_for_region(region: &str) -> Option<&'static str> {
        match region.split('-').next().unwrap_or_default() {
            "us" => Some("us"),
            "eu" => Some("eu"),
            "ap" => Some("ap"),
            "ca" => Some("ca"),
            "sa" => Some("sa"),
            "me" => Some("me"),
            "af" => Some("af"),
            _ => None,
        }
    }

    /// Static helper for model ID resolution (also used in tests).
    ///
    /// Only adds a geographic prefix (e.g. `eu.`, `us.`) for model families
    /// with explicitly-known inference-profile support. For all other models,
    /// the bare model ID is returned so Converse targets the regional model
    /// directly instead of guessing a profile identifier.
    fn resolve_model_id_for_region(model: &str, region: &str) -> String {
        if model.starts_with("arn:")
            || model.starts_with("global.")
            || GEOGRAPHY_PREFIXES
                .iter()
                .any(|prefix| model.starts_with(prefix))
        {
            return model.to_string();
        }

        let Some(rule) = Self::model_rule(model) else {
            return model.to_string();
        };

        if !rule.requires_inference_profile {
            return model.to_string();
        }

        Self::geography_prefix_for_region(region)
            .map(|prefix| format!("{prefix}.{model}"))
            .unwrap_or_else(|| model.to_string())
    }

    /// Estimate context length from model ID.
    fn context_length_for_model(model: &str) -> usize {
        Self::model_rule(model)
            .map(|rule| rule.context_length)
            .unwrap_or(DEFAULT_MAX_CONTEXT)
    }

    /// Determine embedding dimension from embedding model ID.
    pub fn dimension_for_model(model: &str) -> usize {
        Self::embedding_model_rule(model)
            .map(|rule| rule.dimension)
            .unwrap_or(DEFAULT_EMBEDDING_DIMENSION)
    }

    /// Determine max input tokens for an embedding model.
    fn embedding_max_tokens_for_model(model: &str) -> usize {
        Self::embedding_model_rule(model)
            .map(|rule| rule.max_tokens)
            .unwrap_or(DEFAULT_EMBEDDING_MAX_TOKENS)
    }

    // ========================================================================
    // Embedding Helpers
    // ========================================================================

    /// Build the JSON request body for an embedding model.
    ///
    /// Different embedding models on Bedrock require different JSON schemas:
    /// - **Titan Embed**: `{"inputText": "..."}`
    /// - **Cohere Embed**: `{"texts": ["..."], "input_type": "search_query"}`
    fn build_embedding_request(model: &str, texts: &[String]) -> Result<Vec<u8>> {
        let request_shape = Self::embedding_model_rule(model)
            .map(|rule| rule.request_shape)
            .unwrap_or(EmbeddingRequestShape::SingleInputText);

        match request_shape {
            EmbeddingRequestShape::SingleInputText => {
                if texts.len() != 1 {
                    return Err(LlmError::InvalidRequest(format!(
                        "Embedding model '{model}' requires single-text requests; batch splitting must happen at the caller"
                    )));
                }

                let body = serde_json::json!({
                    "inputText": texts[0]
                });
                serde_json::to_vec(&body).map_err(|e| {
                    LlmError::InvalidRequest(format!("Failed to serialize embedding body: {e}"))
                })
            }
            EmbeddingRequestShape::CohereBatch => {
                let body = serde_json::json!({
                    "texts": texts,
                    "input_type": "search_query"
                });
                serde_json::to_vec(&body).map_err(|e| {
                    LlmError::InvalidRequest(format!("Failed to serialize embedding body: {e}"))
                })
            }
        }
    }

    fn parse_embedding_vector(value: &serde_json::Value, path: &str) -> Result<Vec<f32>> {
        let values = value.as_array().ok_or_else(|| {
            LlmError::ProviderError(format!("Expected embedding array at '{path}'"))
        })?;

        values
            .iter()
            .enumerate()
            .map(|(index, item)| {
                item.as_f64().map(|value| value as f32).ok_or_else(|| {
                    LlmError::ProviderError(format!(
                        "Embedding value at '{path}[{index}]' is not numeric"
                    ))
                })
            })
            .collect()
    }

    /// Parse the embedding response from an embedding model.
    ///
    /// - **Titan Embed**: `{"embedding": [f32...], "inputTextTokenCount": N}`
    /// - **Cohere Embed**: `{"embeddings": [[f32...], ...], ...}`
    fn parse_embedding_response(model: &str, response_bytes: &[u8]) -> Result<Vec<Vec<f32>>> {
        let json: serde_json::Value = serde_json::from_slice(response_bytes).map_err(|e| {
            LlmError::ProviderError(format!("Failed to parse embedding response: {e}"))
        })?;

        let request_shape = Self::embedding_model_rule(model)
            .map(|rule| rule.request_shape)
            .unwrap_or(EmbeddingRequestShape::SingleInputText);

        match request_shape {
            EmbeddingRequestShape::SingleInputText => {
                let embedding = json.get("embedding").ok_or_else(|| {
                    LlmError::ProviderError("Missing 'embedding' array in response".to_string())
                })?;
                Ok(vec![Self::parse_embedding_vector(embedding, "embedding")?])
            }
            EmbeddingRequestShape::CohereBatch => {
                let embeddings_value = json.get("embeddings").ok_or_else(|| {
                    LlmError::ProviderError("Missing 'embeddings' in Cohere response".to_string())
                })?;

                let arrays = if let Some(object) = embeddings_value.as_object() {
                    object.get("float").ok_or_else(|| {
                        LlmError::ProviderError(
                            "Missing 'float' array in Cohere embeddings response".to_string(),
                        )
                    })?
                } else {
                    embeddings_value
                };

                let vectors = arrays.as_array().ok_or_else(|| {
                    LlmError::ProviderError(
                        "Expected 'embeddings' to be an array in Cohere response".to_string(),
                    )
                })?;

                vectors
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        Self::parse_embedding_vector(value, &format!("embeddings[{index}]"))
                    })
                    .collect()
            }
        }
    }

    /// Check if a model ID is a Cohere embedding model (supports batch).
    fn is_cohere_embedding(model: &str) -> bool {
        matches!(
            Self::embedding_model_rule(model).map(|rule| rule.request_shape),
            Some(EmbeddingRequestShape::CohereBatch)
        )
    }

    // ========================================================================
    // Document Conversion Helpers
    // ========================================================================

    /// Convert a `serde_json::Value` to an `aws_smithy_types::Document`.
    ///
    /// AWS Smithy `Document` does not implement serde traits, so we need
    /// manual conversion between JSON and Document representations.
    fn json_to_document(value: &serde_json::Value) -> Document {
        match value {
            serde_json::Value::Null => Document::Null,
            serde_json::Value::Bool(b) => Document::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(u) = n.as_u64() {
                    Document::Number(aws_smithy_types::Number::PosInt(u))
                } else if let Some(i) = n.as_i64() {
                    Document::Number(aws_smithy_types::Number::NegInt(i))
                } else if let Some(f) = n.as_f64() {
                    Document::Number(aws_smithy_types::Number::Float(f))
                } else {
                    Document::Null
                }
            }
            serde_json::Value::String(s) => Document::String(s.clone()),
            serde_json::Value::Array(arr) => {
                Document::Array(arr.iter().map(Self::json_to_document).collect())
            }
            serde_json::Value::Object(obj) => Document::Object(
                obj.iter()
                    .map(|(k, v)| (k.clone(), Self::json_to_document(v)))
                    .collect(),
            ),
        }
    }

    /// Convert an `aws_smithy_types::Document` to a `serde_json::Value`.
    fn document_to_json(doc: &Document) -> serde_json::Value {
        match doc {
            Document::Null => serde_json::Value::Null,
            Document::Bool(b) => serde_json::Value::Bool(*b),
            Document::Number(n) => match n {
                aws_smithy_types::Number::PosInt(u) => serde_json::json!(*u),
                aws_smithy_types::Number::NegInt(i) => serde_json::json!(*i),
                aws_smithy_types::Number::Float(f) => serde_json::json!(*f),
            },
            Document::String(s) => serde_json::Value::String(s.clone()),
            Document::Array(arr) => {
                serde_json::Value::Array(arr.iter().map(Self::document_to_json).collect())
            }
            Document::Object(obj) => serde_json::Value::Object(
                obj.iter()
                    .map(|(k, v)| (k.clone(), Self::document_to_json(v)))
                    .collect(),
            ),
        }
    }

    // ========================================================================
    // Message Conversion Helpers
    // ========================================================================

    fn build_message(role: ConversationRole, content_blocks: Vec<ContentBlock>) -> Result<Message> {
        let mut builder = Message::builder().role(role);
        for block in content_blocks {
            builder = builder.content(block);
        }
        builder.build().map_err(Into::into)
    }

    fn image_format_for_mime_type(mime_type: &str) -> Result<ImageFormat> {
        match mime_type {
            "image/jpeg" => Ok(ImageFormat::Jpeg),
            "image/png" => Ok(ImageFormat::Png),
            "image/gif" => Ok(ImageFormat::Gif),
            "image/webp" => Ok(ImageFormat::Webp),
            other => Err(LlmError::InvalidRequest(format!(
                "Unsupported image MIME type for Bedrock: {other}"
            ))),
        }
    }

    fn convert_user_message(msg: &ChatMessage) -> Result<Message> {
        let mut content_blocks = Vec::new();

        if !msg.content.is_empty() {
            content_blocks.push(ContentBlock::Text(msg.content.clone()));
        }

        if let Some(images) = &msg.images {
            for image in images {
                if image.is_url() {
                    return Err(LlmError::InvalidRequest(
                        "Bedrock Converse does not support URL-based images; provide base64-encoded image data instead."
                            .to_string(),
                    ));
                }

                let format = Self::image_format_for_mime_type(&image.mime_type)?;
                let raw_bytes = base64::engine::general_purpose::STANDARD
                    .decode(&image.data)
                    .map_err(|e| {
                        LlmError::InvalidRequest(format!("Failed to decode base64 image data: {e}"))
                    })?;
                let image_block = ImageBlock::builder()
                    .format(format)
                    .source(ImageSource::Bytes(Blob::new(raw_bytes)))
                    .build()?;
                content_blocks.push(ContentBlock::Image(image_block));
            }
        }

        if content_blocks.is_empty() {
            content_blocks.push(ContentBlock::Text(String::new()));
        }

        Self::build_message(ConversationRole::User, content_blocks)
    }

    fn convert_assistant_message(
        msg: &ChatMessage,
        tool_name_aliases: Option<&HashMap<String, String>>,
    ) -> Result<Message> {
        let mut content_blocks = Vec::new();

        if !msg.content.is_empty() {
            content_blocks.push(ContentBlock::Text(msg.content.clone()));
        }

        if let Some(tool_calls) = &msg.tool_calls {
            for tool_call in tool_calls {
                let input =
                    serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments)
                        .map(|value| Self::json_to_document(&value))
                        .unwrap_or_else(|_| Document::String(tool_call.function.arguments.clone()));

                let tool_name =
                    Self::resolve_bedrock_tool_name(&tool_call.function.name, tool_name_aliases);

                let tool_use = ToolUseBlock::builder()
                    .tool_use_id(&tool_call.id)
                    .name(tool_name)
                    .input(input)
                    .build()?;
                content_blocks.push(ContentBlock::ToolUse(tool_use));
            }
        }

        if content_blocks.is_empty() {
            return Err(LlmError::InvalidRequest(
                "Assistant messages for Bedrock must contain text or tool calls".to_string(),
            ));
        }

        Self::build_message(ConversationRole::Assistant, content_blocks)
    }

    fn convert_tool_result_message(msg: &ChatMessage) -> Result<Message> {
        let tool_call_id = msg.tool_call_id.as_deref().ok_or_else(|| {
            LlmError::InvalidRequest(
                "Tool and function messages for Bedrock require a tool_call_id".to_string(),
            )
        })?;

        let tool_result = aws_sdk_bedrockruntime::types::ToolResultBlock::builder()
            .tool_use_id(tool_call_id)
            .content(aws_sdk_bedrockruntime::types::ToolResultContentBlock::Text(
                msg.content.clone(),
            ))
            .build()?;

        Self::build_message(
            ConversationRole::User,
            vec![ContentBlock::ToolResult(tool_result)],
        )
    }

    /// Convert edgequake ChatMessages into Bedrock Messages + optional system blocks.
    #[cfg(test)]
    fn convert_messages(
        messages: &[ChatMessage],
        system_prompt: Option<&str>,
    ) -> Result<(Vec<Message>, Vec<SystemContentBlock>)> {
        Self::convert_messages_with_aliases(messages, system_prompt, None)
    }

    fn convert_messages_with_aliases(
        messages: &[ChatMessage],
        system_prompt: Option<&str>,
        tool_name_aliases: Option<&HashMap<String, String>>,
    ) -> Result<(Vec<Message>, Vec<SystemContentBlock>)> {
        let mut bedrock_messages: Vec<Message> = Vec::new();
        let mut system_blocks: Vec<SystemContentBlock> = Vec::new();

        // Add explicit system prompt if provided via CompletionOptions
        if let Some(sys) = system_prompt {
            if !sys.is_empty() {
                system_blocks.push(SystemContentBlock::Text(sys.to_string()));
            }
        }

        for msg in messages {
            match msg.role {
                ChatRole::System => {
                    if !msg.content.is_empty() {
                        system_blocks.push(SystemContentBlock::Text(msg.content.clone()));
                    }
                }
                ChatRole::User => bedrock_messages.push(Self::convert_user_message(msg)?),
                ChatRole::Assistant => {
                    bedrock_messages.push(Self::convert_assistant_message(msg, tool_name_aliases)?)
                }
                ChatRole::Tool | ChatRole::Function => {
                    bedrock_messages.push(Self::convert_tool_result_message(msg)?);
                }
            }
        }

        Ok((bedrock_messages, system_blocks))
    }

    /// Build `InferenceConfiguration` from `CompletionOptions`.
    fn short_hash(input: &str) -> String {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        input.hash(&mut hasher);
        format!("{:08x}", hasher.finish())
    }

    fn sanitize_tool_name(name: &str) -> String {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return "tool".to_string();
        }

        let normalized_source = trimmed
            .split_once('(')
            .map(|(prefix, _)| prefix.trim())
            .filter(|prefix| {
                !prefix.is_empty()
                    && prefix
                        .chars()
                        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
            })
            .unwrap_or(trimmed);

        let mut sanitized = String::with_capacity(normalized_source.len().min(64));
        let mut prev_was_sep = false;

        for ch in normalized_source.chars() {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                sanitized.push(ch);
                prev_was_sep = false;
            } else if !prev_was_sep && !sanitized.is_empty() {
                sanitized.push('_');
                prev_was_sep = true;
            }
        }

        let sanitized = sanitized.trim_matches(['_', '-']).to_string();
        let mut candidate = if sanitized.is_empty() {
            "tool".to_string()
        } else {
            sanitized
        };

        if candidate.len() > 64 {
            let hash = Self::short_hash(trimmed);
            let max_prefix_len = 64usize.saturating_sub(hash.len() + 1);
            let mut prefix = candidate.chars().take(max_prefix_len).collect::<String>();
            prefix = prefix.trim_matches(['_', '-']).to_string();
            if prefix.is_empty() {
                prefix = "tool".to_string();
            }
            candidate = format!("{prefix}-{hash}");
        }

        candidate
    }

    fn uniquify_tool_name(base: &str, original: &str, used: &mut HashSet<String>) -> String {
        if used.insert(base.to_string()) {
            return base.to_string();
        }

        let hash = Self::short_hash(original);
        let max_prefix_len = 64usize.saturating_sub(hash.len() + 1);
        let mut prefix = base.chars().take(max_prefix_len).collect::<String>();
        prefix = prefix.trim_matches(['_', '-']).to_string();
        let unique = if prefix.is_empty() {
            format!("tool-{hash}")
        } else {
            format!("{prefix}-{hash}")
        };
        used.insert(unique.clone());
        unique
    }

    fn build_tool_name_aliases(tools: &[EdgequakeToolDefinition]) -> HashMap<String, String> {
        let mut aliases = HashMap::new();
        let mut used = HashSet::new();

        for tool in tools {
            let base = Self::sanitize_tool_name(&tool.function.name);
            let alias = Self::uniquify_tool_name(&base, &tool.function.name, &mut used);
            aliases.insert(tool.function.name.clone(), alias);
        }

        aliases
    }

    fn resolve_bedrock_tool_name(
        name: &str,
        tool_name_aliases: Option<&HashMap<String, String>>,
    ) -> String {
        tool_name_aliases
            .and_then(|aliases| aliases.get(name).cloned())
            .unwrap_or_else(|| Self::sanitize_tool_name(name))
    }

    fn build_inference_config(
        options: Option<&CompletionOptions>,
    ) -> Option<InferenceConfiguration> {
        let opts = options?;
        let mut builder = InferenceConfiguration::builder();
        let mut has_config = false;

        if let Some(max_tokens) = opts.max_tokens {
            builder = builder.max_tokens(max_tokens as i32);
            has_config = true;
        }
        if let Some(temperature) = opts.temperature {
            builder = builder.temperature(temperature);
            has_config = true;
        }
        if let Some(top_p) = opts.top_p {
            builder = builder.top_p(top_p);
            has_config = true;
        }
        if let Some(ref stops) = opts.stop {
            for s in stops {
                builder = builder.stop_sequences(s.clone());
            }
            has_config = true;
        }

        if has_config {
            Some(builder.build())
        } else {
            None
        }
    }

    /// Convert edgequake tool definitions to Bedrock ToolConfiguration.
    #[cfg(test)]
    fn build_tool_config(
        tools: &[EdgequakeToolDefinition],
        tool_choice: Option<&EdgequakeToolChoice>,
    ) -> Result<Option<ToolConfiguration>> {
        let aliases = Self::build_tool_name_aliases(tools);
        Self::build_tool_config_with_aliases(tools, tool_choice, &aliases)
    }

    fn build_tool_config_with_aliases(
        tools: &[EdgequakeToolDefinition],
        tool_choice: Option<&EdgequakeToolChoice>,
        tool_name_aliases: &HashMap<String, String>,
    ) -> Result<Option<ToolConfiguration>> {
        if tools.is_empty() {
            return Ok(None);
        }

        let mut bedrock_tools = Vec::new();
        for tool in tools {
            // Bedrock uses Claude under the hood — apply the same schema
            // sanitization as the Anthropic provider: strip unsupported
            // constraint keywords and ensure additionalProperties: false.
            let sanitized_params = super::schema_utils::ensure_additional_properties_false(
                super::schema_utils::strip_keys_recursive(
                    tool.function.parameters.clone(),
                    &[
                        "minimum",
                        "maximum",
                        "minLength",
                        "maxLength",
                        "minItems",
                        "maxItems",
                    ],
                ),
            );
            let schema_doc = Self::json_to_document(&sanitized_params);
            let tool_name =
                Self::resolve_bedrock_tool_name(&tool.function.name, Some(tool_name_aliases));

            let spec = ToolSpecification::builder()
                .name(tool_name)
                .description(&tool.function.description)
                .input_schema(ToolInputSchema::Json(schema_doc))
                .build()?;
            bedrock_tools.push(Tool::ToolSpec(spec));
        }

        let mut config_builder = ToolConfiguration::builder();
        for tool in bedrock_tools {
            config_builder = config_builder.tools(tool);
        }

        // Map tool_choice
        if let Some(choice) = tool_choice {
            let bedrock_choice = match choice {
                // "none" means disable tool calling — omit tool_config entirely
                EdgequakeToolChoice::Auto(s) if s == "none" => {
                    return Ok(None);
                }
                EdgequakeToolChoice::Auto(_) => BedrockToolChoice::Auto(
                    aws_sdk_bedrockruntime::types::AutoToolChoice::builder().build(),
                ),
                EdgequakeToolChoice::Required(_) => BedrockToolChoice::Any(
                    aws_sdk_bedrockruntime::types::AnyToolChoice::builder().build(),
                ),
                EdgequakeToolChoice::Function { function, .. } => {
                    let tool_name =
                        Self::resolve_bedrock_tool_name(&function.name, Some(tool_name_aliases));
                    BedrockToolChoice::Tool(
                        aws_sdk_bedrockruntime::types::SpecificToolChoice::builder()
                            .name(tool_name)
                            .build()?,
                    )
                }
            };
            config_builder = config_builder.tool_choice(bedrock_choice);
        }

        let config = config_builder.build()?;
        Ok(Some(config))
    }

    fn build_thinking_request_fields(options: Option<&CompletionOptions>) -> Option<Document> {
        let budget = options.and_then(|opts| opts.thinking_budget_tokens)?;
        Some(Document::Object(std::collections::HashMap::from([(
            "thinking".to_string(),
            Document::Object(std::collections::HashMap::from([
                ("type".to_string(), Document::String("enabled".to_string())),
                (
                    "budget_tokens".to_string(),
                    Document::Number(aws_smithy_types::Number::PosInt(u64::from(budget))),
                ),
            ])),
        )])))
    }

    fn prepare_converse_request(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
        tool_name_aliases: Option<&HashMap<String, String>>,
    ) -> Result<PreparedConverseRequest> {
        let system_prompt = options.and_then(|opts| opts.system_prompt.as_deref());
        let (bedrock_messages, system_blocks) =
            Self::convert_messages_with_aliases(messages, system_prompt, tool_name_aliases)?;

        Ok(PreparedConverseRequest {
            resolved_model: self.resolve_model_id(),
            bedrock_messages,
            system_blocks,
            inference_config: Self::build_inference_config(options),
            additional_fields: Self::build_thinking_request_fields(options),
        })
    }

    /// Map Bedrock StopReason to edgequake finish_reason string.
    fn map_stop_reason(reason: &StopReason) -> String {
        match reason {
            StopReason::EndTurn => "stop".to_string(),
            StopReason::MaxTokens => "length".to_string(),
            StopReason::StopSequence => "stop".to_string(),
            StopReason::ToolUse => "tool_calls".to_string(),
            StopReason::ContentFiltered => "content_filter".to_string(),
            StopReason::GuardrailIntervened => "content_filter".to_string(),
            _ => "stop".to_string(),
        }
    }

    fn extract_reasoning_content(
        reasoning: &aws_sdk_bedrockruntime::types::ReasoningContentBlock,
    ) -> Option<String> {
        match reasoning {
            aws_sdk_bedrockruntime::types::ReasoningContentBlock::ReasoningText(text) => {
                Some(text.text().to_string())
            }
            aws_sdk_bedrockruntime::types::ReasoningContentBlock::RedactedContent(_) => None,
            _ => None,
        }
    }

    /// Extract text content, tool calls, and thinking content from Bedrock ConverseOutput.
    ///
    /// Returns `(content_text, tool_calls, thinking_content)`.
    #[cfg(test)]
    fn extract_content(
        output: &ConverseOutput,
    ) -> (String, Vec<EdgequakeToolCall>, Option<String>) {
        Self::extract_content_with_aliases(output, None)
    }

    fn extract_content_with_aliases(
        output: &ConverseOutput,
        reverse_tool_name_aliases: Option<&HashMap<String, String>>,
    ) -> (String, Vec<EdgequakeToolCall>, Option<String>) {
        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();
        let mut thinking_parts = Vec::new();

        if let ConverseOutput::Message(msg) = output {
            for block in msg.content() {
                match block {
                    ContentBlock::Text(text) => {
                        text_parts.push(text.clone());
                    }
                    ContentBlock::ToolUse(tool_use) => {
                        // Convert Document input to JSON string for the tool call arguments
                        let arguments_json = Self::document_to_json(&tool_use.input);
                        let arguments_str =
                            serde_json::to_string(&arguments_json).unwrap_or_default();

                        let tool_name = reverse_tool_name_aliases
                            .and_then(|aliases| aliases.get(&tool_use.name).cloned())
                            .unwrap_or_else(|| tool_use.name.clone());

                        tool_calls.push(EdgequakeToolCall {
                            id: tool_use.tool_use_id.clone(),
                            call_type: "function".to_string(),
                            function: crate::traits::FunctionCall {
                                name: tool_name,
                                arguments: arguments_str,
                            },
                            thought_signature: None,
                        });
                    }
                    ContentBlock::ReasoningContent(reasoning) => {
                        if let Some(text) = Self::extract_reasoning_content(reasoning) {
                            thinking_parts.push(text);
                        }
                    }
                    _ => {}
                }
            }
        }

        let thinking_content = if thinking_parts.is_empty() {
            None
        } else {
            Some(thinking_parts.join(""))
        };

        (text_parts.join(""), tool_calls, thinking_content)
    }

    fn build_llm_response(
        response: aws_sdk_bedrockruntime::operation::converse::ConverseOutput,
        resolved_model: String,
        reverse_tool_name_aliases: Option<&HashMap<String, String>>,
    ) -> Result<LLMResponse> {
        let output = response
            .output()
            .ok_or_else(|| LlmError::ProviderError("Bedrock returned no output".to_string()))?;
        let (content, tool_calls, thinking_content) =
            Self::extract_content_with_aliases(output, reverse_tool_name_aliases);

        let (prompt_tokens, completion_tokens, total_tokens) = response
            .usage()
            .map(|usage| {
                let input = usage.input_tokens() as usize;
                let output = usage.output_tokens() as usize;
                (input, output, input + output)
            })
            .unwrap_or((0, 0, 0));

        Ok(LLMResponse {
            content,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            model: resolved_model,
            finish_reason: Some(Self::map_stop_reason(response.stop_reason())),
            tool_calls,
            metadata: HashMap::new(),
            cache_hit_tokens: None,
            cache_write_tokens: None,
            thinking_tokens: None,
            thinking_content,
        })
    }
}

// ============================================================================
// LLMProvider Implementation
// ============================================================================

#[async_trait]
impl LLMProvider for BedrockProvider {
    fn name(&self) -> &str {
        "bedrock"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_context_length(&self) -> usize {
        self.max_context_length
    }

    #[instrument(skip(self, prompt), fields(provider = "bedrock", model = %self.model))]
    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        let messages = vec![ChatMessage::user(prompt)];
        self.chat(&messages, None).await
    }

    #[instrument(skip(self, prompt, options), fields(provider = "bedrock", model = %self.model))]
    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<LLMResponse> {
        let messages = vec![ChatMessage::user(prompt)];
        self.chat(&messages, Some(options)).await
    }

    #[instrument(skip(self, messages, options), fields(provider = "bedrock", model = %self.model))]
    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let prepared = self.prepare_converse_request(messages, options, None)?;
        let mut request = self.client.converse().model_id(&prepared.resolved_model);

        for msg in prepared.bedrock_messages {
            request = request.messages(msg);
        }
        for block in prepared.system_blocks {
            request = request.system(block);
        }
        if let Some(config) = prepared.inference_config {
            request = request.inference_config(config);
        }
        debug!(
            "Sending Bedrock Converse request for model: {} (resolved: {})",
            self.model, prepared.resolved_model
        );

        if let Some(additional_fields) = prepared.additional_fields {
            request = request.additional_model_request_fields(additional_fields);
        }

        let response = request.send().await?;
        Self::build_llm_response(response, prepared.resolved_model, None)
    }

    #[instrument(skip(self, messages, tools, tool_choice, options), fields(provider = "bedrock", model = %self.model))]
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[EdgequakeToolDefinition],
        tool_choice: Option<EdgequakeToolChoice>,
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let tool_name_aliases = Self::build_tool_name_aliases(tools);
        let reverse_tool_name_aliases: HashMap<String, String> = tool_name_aliases
            .iter()
            .map(|(original, alias)| (alias.clone(), original.clone()))
            .collect();

        let prepared =
            self.prepare_converse_request(messages, options, Some(&tool_name_aliases))?;
        let mut request = self.client.converse().model_id(&prepared.resolved_model);

        for msg in prepared.bedrock_messages {
            request = request.messages(msg);
        }
        for block in prepared.system_blocks {
            request = request.system(block);
        }
        if let Some(config) = prepared.inference_config {
            request = request.inference_config(config);
        }

        if let Some(tool_config) =
            Self::build_tool_config_with_aliases(tools, tool_choice.as_ref(), &tool_name_aliases)?
        {
            request = request.tool_config(tool_config);
        }

        debug!(
            "Sending Bedrock Converse request with {} tools for model: {} (resolved: {})",
            tools.len(),
            self.model,
            prepared.resolved_model
        );

        if let Some(additional_fields) = prepared.additional_fields {
            request = request.additional_model_request_fields(additional_fields);
        }

        let response = request.send().await?;
        Self::build_llm_response(
            response,
            prepared.resolved_model,
            Some(&reverse_tool_name_aliases),
        )
    }

    #[instrument(skip(self, prompt), fields(provider = "bedrock", model = %self.model))]
    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        let messages = vec![ChatMessage::user(prompt)];
        let prepared = self.prepare_converse_request(&messages, None, None)?;

        let mut request = self
            .client
            .converse_stream()
            .model_id(&prepared.resolved_model);

        for msg in prepared.bedrock_messages {
            request = request.messages(msg);
        }
        for block in prepared.system_blocks {
            request = request.system(block);
        }

        debug!(
            "Sending Bedrock ConverseStream request for model: {} (resolved: {})",
            self.model, prepared.resolved_model
        );

        // G2: typed ? for initial send
        let response = request.send().await?;

        // G3 + G4 (ADR-001 §6.5): SDK-idiomatic recv() loop with as_text() accessor.
        use aws_sdk_bedrockruntime::types::{
            error::ConverseStreamOutputError, ConverseStreamOutput as CSO,
        };
        use futures::stream;

        let mapped_stream = stream::unfold(response.stream, |mut rx| async move {
            loop {
                match rx.recv().await {
                    Ok(Some(cso)) => match cso {
                        CSO::ContentBlockDelta(delta_event) => {
                            if let Some(delta) = delta_event.delta() {
                                // G4: Use SDK accessor as_text() instead of pattern match
                                if let Ok(text) = delta.as_text() {
                                    return Some((Ok(text.to_string()), rx));
                                }
                            }
                            // Non-text delta (e.g. tool-use blob) — skip
                        }
                        CSO::MessageStop(_) => return None,
                        _ => {
                            // MessageStart, ContentBlockStart, ContentBlockStop, Metadata
                        }
                    },
                    Ok(None) => return None,
                    Err(e) => {
                        // Typed stream error extraction (G2)
                        let msg = e
                            .as_service_error()
                            .map(|se| match se {
                                ConverseStreamOutputError::ValidationException(v) => {
                                    v.message().unwrap_or("validation error").to_string()
                                }
                                ConverseStreamOutputError::ThrottlingException(t) => {
                                    t.message().unwrap_or("throttled").to_string()
                                }
                                other => other
                                    .meta()
                                    .message()
                                    .unwrap_or("unknown stream error")
                                    .to_string(),
                            })
                            .unwrap_or_else(|| "Bedrock stream recv error".to_string());
                        return Some((Err(LlmError::ProviderError(msg)), rx));
                    }
                }
            }
        });

        Ok(Box::pin(mapped_stream))
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_tool_streaming(&self) -> bool {
        false // Tool streaming via Bedrock ConverseStream is complex; deferred
    }

    fn supports_json_mode(&self) -> bool {
        false
    }

    fn supports_function_calling(&self) -> bool {
        // Most Bedrock models support tool use via the Converse API
        true
    }
}

// ============================================================================
// EmbeddingProvider Implementation
// ============================================================================

#[async_trait]
impl EmbeddingProvider for BedrockProvider {
    fn name(&self) -> &str {
        "bedrock"
    }

    /// Returns the embedding model ID (not the LLM model).
    #[allow(clippy::misnamed_getters)]
    fn model(&self) -> &str {
        &self.embedding_model
    }

    fn dimension(&self) -> usize {
        self.embedding_dimension
    }

    fn max_tokens(&self) -> usize {
        Self::embedding_max_tokens_for_model(&self.embedding_model)
    }

    #[instrument(skip(self, texts), fields(provider = "bedrock", embedding_model = %self.embedding_model))]
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Cohere embedding models support batch natively (up to 96 texts)
        if Self::is_cohere_embedding(&self.embedding_model) {
            // Batch all texts in one request (Cohere supports up to 96 texts)
            let chunks: Vec<&[String]> = texts.chunks(96).collect();
            let mut all_embeddings = Vec::with_capacity(texts.len());

            for chunk in chunks {
                let body = Self::build_embedding_request(&self.embedding_model, chunk)?;
                let response = self
                    .client
                    .invoke_model()
                    .model_id(&self.embedding_model)
                    .content_type("application/json")
                    .accept("application/json")
                    .body(Blob::new(body))
                    .send()
                    .await?;

                let response_bytes = response.body().as_ref();
                let mut embeddings =
                    Self::parse_embedding_response(&self.embedding_model, response_bytes)?;
                all_embeddings.append(&mut embeddings);
            }

            debug!(
                "Generated {} Cohere embeddings ({} dims)",
                all_embeddings.len(),
                self.embedding_dimension
            );
            Ok(all_embeddings)
        } else {
            // Titan/Nova: one text per request
            let mut all_embeddings = Vec::with_capacity(texts.len());

            for text in texts {
                let body = Self::build_embedding_request(
                    &self.embedding_model,
                    std::slice::from_ref(text),
                )?;
                let response = self
                    .client
                    .invoke_model()
                    .model_id(&self.embedding_model)
                    .content_type("application/json")
                    .accept("application/json")
                    .body(Blob::new(body))
                    .send()
                    .await?;

                let response_bytes = response.body().as_ref();
                let mut embeddings =
                    Self::parse_embedding_response(&self.embedding_model, response_bytes)?;
                all_embeddings.append(&mut embeddings);
            }

            debug!(
                "Generated {} Titan/Nova embeddings ({} dims)",
                all_embeddings.len(),
                self.embedding_dimension
            );
            Ok(all_embeddings)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_length_claude3() {
        assert_eq!(
            BedrockProvider::context_length_for_model("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            200_000
        );
        assert_eq!(
            BedrockProvider::context_length_for_model("anthropic.claude-4-sonnet-20250514-v1:0"),
            200_000
        );
    }

    #[test]
    fn test_context_length_claude2() {
        assert_eq!(
            BedrockProvider::context_length_for_model("anthropic.claude-2"),
            100_000
        );
    }

    #[test]
    fn test_context_length_nova() {
        assert_eq!(
            BedrockProvider::context_length_for_model("amazon.nova-pro-v1:0"),
            300_000
        );
    }

    #[test]
    fn test_context_length_llama() {
        assert_eq!(
            BedrockProvider::context_length_for_model("meta.llama3-70b-instruct-v1:0"),
            128_000
        );
    }

    #[test]
    fn test_context_length_mistral() {
        assert_eq!(
            BedrockProvider::context_length_for_model("mistral.mistral-large-2407-v1:0"),
            32_000
        );
    }

    #[test]
    fn test_context_length_cohere() {
        assert_eq!(
            BedrockProvider::context_length_for_model("cohere.command-r-plus-v1:0"),
            128_000
        );
    }

    #[test]
    fn test_context_length_deepseek() {
        assert_eq!(
            BedrockProvider::context_length_for_model("deepseek.r1-v1:0"),
            128_000
        );
    }

    #[test]
    fn test_context_length_qwen() {
        assert_eq!(
            BedrockProvider::context_length_for_model("qwen.qwen2-5-72b-instruct-v1:0"),
            131_072
        );
    }

    #[test]
    fn test_context_length_writer() {
        assert_eq!(
            BedrockProvider::context_length_for_model("writer.palmyra-x-004-v1:0"),
            128_000
        );
    }

    #[test]
    fn test_context_length_default() {
        assert_eq!(
            BedrockProvider::context_length_for_model("some-unknown-model"),
            DEFAULT_MAX_CONTEXT
        );
    }

    #[test]
    fn test_stop_reason_mapping() {
        assert_eq!(
            BedrockProvider::map_stop_reason(&StopReason::EndTurn),
            "stop"
        );
        assert_eq!(
            BedrockProvider::map_stop_reason(&StopReason::MaxTokens),
            "length"
        );
        assert_eq!(
            BedrockProvider::map_stop_reason(&StopReason::StopSequence),
            "stop"
        );
        assert_eq!(
            BedrockProvider::map_stop_reason(&StopReason::ToolUse),
            "tool_calls"
        );
        assert_eq!(
            BedrockProvider::map_stop_reason(&StopReason::ContentFiltered),
            "content_filter"
        );
        assert_eq!(
            BedrockProvider::map_stop_reason(&StopReason::GuardrailIntervened),
            "content_filter"
        );
    }

    #[test]
    fn test_build_inference_config_none() {
        assert!(BedrockProvider::build_inference_config(None).is_none());
    }

    #[test]
    fn test_build_inference_config_with_options() {
        let opts = CompletionOptions {
            max_tokens: Some(1024),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stop: Some(vec!["END".to_string()]),
            ..Default::default()
        };
        let config = BedrockProvider::build_inference_config(Some(&opts));
        assert!(config.is_some());
    }

    #[test]
    fn test_build_inference_config_empty_options() {
        let opts = CompletionOptions::default();
        let config = BedrockProvider::build_inference_config(Some(&opts));
        assert!(config.is_none());
    }

    #[test]
    fn test_convert_messages_system() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
        ];
        let (bedrock_msgs, system_blocks) =
            BedrockProvider::convert_messages(&messages, None).unwrap();
        // System message goes into system_blocks, user message into bedrock_msgs
        assert_eq!(system_blocks.len(), 1);
        assert_eq!(bedrock_msgs.len(), 1);
    }

    #[test]
    fn test_convert_messages_with_system_prompt_option() {
        let messages = vec![ChatMessage::user("Hello")];
        let (bedrock_msgs, system_blocks) =
            BedrockProvider::convert_messages(&messages, Some("Be concise")).unwrap();
        assert_eq!(system_blocks.len(), 1);
        assert_eq!(bedrock_msgs.len(), 1);
    }

    #[test]
    fn test_convert_messages_empty_system_prompt_ignored() {
        let messages = vec![ChatMessage::user("Hello")];
        let (_, system_blocks) = BedrockProvider::convert_messages(&messages, Some("")).unwrap();
        assert_eq!(system_blocks.len(), 0);
    }

    #[test]
    fn test_convert_messages_tool_result() {
        let messages = vec![ChatMessage::tool_result("call_123", "Result data")];
        let (bedrock_msgs, system_blocks) =
            BedrockProvider::convert_messages(&messages, None).unwrap();
        assert_eq!(system_blocks.len(), 0);
        assert_eq!(bedrock_msgs.len(), 1);
        // Tool results are sent as user messages in Bedrock
    }

    #[test]
    fn test_convert_messages_multiple_system_blocks() {
        let messages = vec![
            ChatMessage::system("System 1"),
            ChatMessage::system("System 2"),
            ChatMessage::user("Hello"),
        ];
        let (bedrock_msgs, system_blocks) =
            BedrockProvider::convert_messages(&messages, Some("Prefix system")).unwrap();
        // 1 from options + 2 from messages = 3 system blocks
        assert_eq!(system_blocks.len(), 3);
        assert_eq!(bedrock_msgs.len(), 1);
    }

    #[test]
    fn test_convert_messages_user_and_assistant() {
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
            ChatMessage::user("How are you?"),
        ];
        let (bedrock_msgs, system_blocks) =
            BedrockProvider::convert_messages(&messages, None).unwrap();
        assert_eq!(system_blocks.len(), 0);
        assert_eq!(bedrock_msgs.len(), 3);
    }

    #[test]
    fn test_convert_messages_empty_assistant_rejected() {
        let messages = vec![ChatMessage::assistant("")];
        let result = BedrockProvider::convert_messages(&messages, None);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LlmError::InvalidRequest(_)));
    }

    #[test]
    fn test_convert_messages_tool_result_requires_tool_call_id() {
        let message = ChatMessage {
            role: ChatRole::Tool,
            content: "tool output".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            cache_control: None,
            images: None,
        };
        let result = BedrockProvider::convert_messages(&[message], None);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LlmError::InvalidRequest(_)));
    }

    #[test]
    fn test_json_to_document_null() {
        let doc = BedrockProvider::json_to_document(&serde_json::Value::Null);
        assert!(matches!(doc, Document::Null));
    }

    #[test]
    fn test_json_to_document_bool() {
        let doc = BedrockProvider::json_to_document(&serde_json::json!(true));
        assert!(matches!(doc, Document::Bool(true)));
    }

    #[test]
    fn test_json_to_document_string() {
        let doc = BedrockProvider::json_to_document(&serde_json::json!("hello"));
        assert!(matches!(doc, Document::String(s) if s == "hello"));
    }

    #[test]
    fn test_json_to_document_number() {
        let doc = BedrockProvider::json_to_document(&serde_json::json!(42));
        assert!(matches!(
            doc,
            Document::Number(aws_smithy_types::Number::PosInt(42))
        ));
    }

    #[test]
    fn test_json_to_document_negative_number() {
        let doc = BedrockProvider::json_to_document(&serde_json::json!(-5));
        assert!(matches!(
            doc,
            Document::Number(aws_smithy_types::Number::NegInt(-5))
        ));
    }

    #[test]
    fn test_json_to_document_float() {
        let doc = BedrockProvider::json_to_document(&serde_json::json!(1.125));
        if let Document::Number(aws_smithy_types::Number::Float(f)) = doc {
            assert!((f - 1.125).abs() < f64::EPSILON);
        } else {
            panic!("Expected float document");
        }
    }

    #[test]
    fn test_json_to_document_array() {
        let doc = BedrockProvider::json_to_document(&serde_json::json!([1, "two", null]));
        if let Document::Array(arr) = doc {
            assert_eq!(arr.len(), 3);
        } else {
            panic!("Expected array document");
        }
    }

    #[test]
    fn test_json_to_document_object() {
        let doc = BedrockProvider::json_to_document(&serde_json::json!({"key": "value"}));
        if let Document::Object(obj) = doc {
            assert_eq!(obj.len(), 1);
            assert!(obj.contains_key("key"));
        } else {
            panic!("Expected object document");
        }
    }

    #[test]
    fn test_document_to_json_roundtrip() {
        let original = serde_json::json!({
            "name": "test",
            "age": 30,
            "active": true,
            "tags": ["a", "b"],
            "nested": {"x": 1.5}
        });
        let doc = BedrockProvider::json_to_document(&original);
        let recovered = BedrockProvider::document_to_json(&doc);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_extract_content_text_only() {
        let msg = Message::builder()
            .role(ConversationRole::Assistant)
            .content(ContentBlock::Text("Hello world".to_string()))
            .build()
            .unwrap();
        let output = ConverseOutput::Message(msg);
        let (text, tool_calls, thinking) = BedrockProvider::extract_content(&output);
        assert_eq!(text, "Hello world");
        assert!(tool_calls.is_empty());
        assert!(thinking.is_none());
    }

    #[test]
    fn test_extract_content_multiple_text_blocks() {
        let msg = Message::builder()
            .role(ConversationRole::Assistant)
            .content(ContentBlock::Text("Hello ".to_string()))
            .content(ContentBlock::Text("world".to_string()))
            .build()
            .unwrap();
        let output = ConverseOutput::Message(msg);
        let (text, _, thinking) = BedrockProvider::extract_content(&output);
        assert_eq!(text, "Hello world");
        assert!(thinking.is_none());
    }

    #[test]
    fn test_extract_content_with_tool_use() {
        let tool_use = ToolUseBlock::builder()
            .tool_use_id("call_123")
            .name("get_weather")
            .input(Document::Object(
                vec![("city".to_string(), Document::String("Paris".to_string()))]
                    .into_iter()
                    .collect(),
            ))
            .build()
            .unwrap();

        let msg = Message::builder()
            .role(ConversationRole::Assistant)
            .content(ContentBlock::Text("Let me check the weather.".to_string()))
            .content(ContentBlock::ToolUse(tool_use))
            .build()
            .unwrap();

        let output = ConverseOutput::Message(msg);
        let (text, tool_calls, thinking) = BedrockProvider::extract_content(&output);
        assert_eq!(text, "Let me check the weather.");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_123");
        assert_eq!(tool_calls[0].call_type, "function");
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert!(tool_calls[0].function.arguments.contains("Paris"));
        assert!(thinking.is_none());
    }

    #[test]
    fn test_extract_content_with_reasoning_text() {
        let reasoning = aws_sdk_bedrockruntime::types::ReasoningTextBlock::builder()
            .text("Let me reason this through.")
            .signature("sig-123")
            .build()
            .unwrap();
        let msg = Message::builder()
            .role(ConversationRole::Assistant)
            .content(ContentBlock::ReasoningContent(
                aws_sdk_bedrockruntime::types::ReasoningContentBlock::ReasoningText(reasoning),
            ))
            .content(ContentBlock::Text("Final answer".to_string()))
            .build()
            .unwrap();

        let output = ConverseOutput::Message(msg);
        let (text, tool_calls, thinking) = BedrockProvider::extract_content(&output);
        assert_eq!(text, "Final answer");
        assert!(tool_calls.is_empty());
        assert_eq!(thinking, Some("Let me reason this through.".to_string()));
    }

    #[test]
    fn test_extract_content_with_tool_name_alias_reverse_mapping() {
        let tool_use = ToolUseBlock::builder()
            .tool_use_id("call_456")
            .name("calculate")
            .input(Document::Object(
                vec![(
                    "expression".to_string(),
                    Document::String("7 * 8".to_string()),
                )]
                .into_iter()
                .collect(),
            ))
            .build()
            .unwrap();

        let msg = Message::builder()
            .role(ConversationRole::Assistant)
            .content(ContentBlock::ToolUse(tool_use))
            .build()
            .unwrap();

        let output = ConverseOutput::Message(msg);
        let aliases = HashMap::from([(
            "calculate".to_string(),
            "calculate(expression=\"7 * 8\")".to_string(),
        )]);
        let (_, tool_calls, _) =
            BedrockProvider::extract_content_with_aliases(&output, Some(&aliases));

        assert_eq!(tool_calls.len(), 1);
        assert_eq!(
            tool_calls[0].function.name,
            "calculate(expression=\"7 * 8\")"
        );
    }

    #[test]
    fn test_build_tool_config_empty_tools() {
        let result = BedrockProvider::build_tool_config(&[], None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_build_tool_config_auto_none_returns_none() {
        let tools = vec![EdgequakeToolDefinition::function(
            "test_fn",
            "A test function",
            serde_json::json!({"type": "object", "properties": {}}),
        )];
        let choice = EdgequakeToolChoice::none();
        let result = BedrockProvider::build_tool_config(&tools, Some(&choice)).unwrap();
        assert!(
            result.is_none(),
            "tool_choice='none' should omit tool config"
        );
    }

    #[test]
    fn test_build_tool_config_with_tools() {
        let tools = vec![EdgequakeToolDefinition::function(
            "search",
            "Search the web",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }),
        )];
        let result = BedrockProvider::build_tool_config(&tools, None).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_sanitize_tool_name_strips_inline_call_syntax() {
        let sanitized =
            BedrockProvider::sanitize_tool_name("click_on_link(url=\"https://example.com\")");
        assert_eq!(sanitized, "click_on_link");
    }

    #[test]
    fn test_sanitize_tool_name_enforces_bedrock_constraints() {
        let original = "tool with spaces and symbols !@#$%^&*() and a very very very long suffix that exceeds the provider limit";
        let sanitized = BedrockProvider::sanitize_tool_name(original);
        assert!(
            sanitized.len() <= 64,
            "sanitized name must fit Bedrock's 64-char limit"
        );
        assert!(
            sanitized
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-'),
            "sanitized name must satisfy Bedrock's identifier regex"
        );
    }

    #[test]
    fn test_build_tool_config_accepts_malformed_tool_names() {
        let tools = vec![EdgequakeToolDefinition::function(
            "click_on_link(url=\"https://blog.x.com/engineering/en_us/topics/open-source\")",
            "Click a browser link",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string"}
                },
                "required": ["url"]
            }),
        )];
        let result = BedrockProvider::build_tool_config(&tools, None).unwrap();
        assert!(
            result.is_some(),
            "Bedrock tool config should sanitize malformed names"
        );
    }

    #[test]
    fn test_convert_messages_accepts_assistant_tool_calls_with_malformed_names() {
        let message = ChatMessage {
            role: ChatRole::Assistant,
            content: String::new(),
            name: None,
            tool_calls: Some(vec![EdgequakeToolCall {
                id: "call_123".to_string(),
                call_type: "function".to_string(),
                function: crate::traits::FunctionCall {
                    name: "click_on_link(url=\"https://example.com\")".to_string(),
                    arguments: "{\"url\":\"https://example.com\"}".to_string(),
                },
                thought_signature: None,
            }]),
            tool_call_id: None,
            cache_control: None,
            images: None,
        };

        let result = BedrockProvider::convert_messages(&[message], None);
        assert!(
            result.is_ok(),
            "assistant tool history should be sanitized for Bedrock"
        );
    }

    #[test]
    fn test_provider_name_and_model() {
        // We can't easily construct a BedrockProvider without AWS config in tests,
        // but we can test the static helper methods which don't require a client.
        assert_eq!(
            BedrockProvider::context_length_for_model("anthropic.claude-3-5-haiku-20241022-v1:0"),
            200_000
        );
    }

    #[test]
    fn test_with_model_updates_context() {
        // Verify that context_length_for_model returns different values for different models
        let claude =
            BedrockProvider::context_length_for_model("anthropic.claude-3-5-sonnet-20241022-v2:0");
        let nova = BedrockProvider::context_length_for_model("amazon.nova-pro-v1:0");
        let llama = BedrockProvider::context_length_for_model("meta.llama3-70b-instruct-v1:0");
        assert_ne!(claude, nova);
        assert_ne!(nova, llama);
    }

    // ====================================================================
    // Inference Profile Resolution Tests
    // ====================================================================

    #[test]
    fn test_resolve_model_id_bare_us_region() {
        assert_eq!(
            BedrockProvider::resolve_model_id_for_region("amazon.nova-lite-v1:0", "us-east-1"),
            "us.amazon.nova-lite-v1:0"
        );
    }

    #[test]
    fn test_resolve_model_id_bare_eu_region() {
        assert_eq!(
            BedrockProvider::resolve_model_id_for_region("amazon.nova-lite-v1:0", "eu-west-1"),
            "eu.amazon.nova-lite-v1:0"
        );
    }

    #[test]
    fn test_resolve_model_id_bare_ap_region() {
        assert_eq!(
            BedrockProvider::resolve_model_id_for_region(
                "anthropic.claude-3-haiku-20240307-v1:0",
                "ap-southeast-1"
            ),
            "ap.anthropic.claude-3-haiku-20240307-v1:0"
        );
    }

    #[test]
    fn test_resolve_model_id_already_prefixed_us() {
        assert_eq!(
            BedrockProvider::resolve_model_id_for_region("us.amazon.nova-lite-v1:0", "eu-west-1"),
            "us.amazon.nova-lite-v1:0"
        );
    }

    #[test]
    fn test_resolve_model_id_already_prefixed_eu() {
        assert_eq!(
            BedrockProvider::resolve_model_id_for_region("eu.amazon.nova-lite-v1:0", "us-east-1"),
            "eu.amazon.nova-lite-v1:0"
        );
    }

    #[test]
    fn test_resolve_model_id_already_prefixed_global() {
        assert_eq!(
            BedrockProvider::resolve_model_id_for_region(
                "global.anthropic.claude-sonnet-4-20250514-v1:0",
                "us-east-1"
            ),
            "global.anthropic.claude-sonnet-4-20250514-v1:0"
        );
    }

    #[test]
    fn test_resolve_model_id_arn_passthrough() {
        let arn = "arn:aws:bedrock:us-east-1:123456789:inference-profile/my-profile";
        assert_eq!(
            BedrockProvider::resolve_model_id_for_region(arn, "eu-west-1"),
            arn
        );
    }

    #[test]
    fn test_resolve_model_id_other_geographies() {
        assert_eq!(
            BedrockProvider::resolve_model_id_for_region("amazon.nova-lite-v1:0", "ca-central-1"),
            "ca.amazon.nova-lite-v1:0"
        );
        assert_eq!(
            BedrockProvider::resolve_model_id_for_region("amazon.nova-lite-v1:0", "sa-east-1"),
            "sa.amazon.nova-lite-v1:0"
        );
        assert_eq!(
            BedrockProvider::resolve_model_id_for_region("amazon.nova-lite-v1:0", "me-south-1"),
            "me.amazon.nova-lite-v1:0"
        );
        assert_eq!(
            BedrockProvider::resolve_model_id_for_region("amazon.nova-lite-v1:0", "af-south-1"),
            "af.amazon.nova-lite-v1:0"
        );
    }

    #[test]
    fn test_resolve_model_id_no_prefix_for_non_profile_models() {
        // Models without cross-region inference profiles should NOT get a prefix
        assert_eq!(
            BedrockProvider::resolve_model_id_for_region(
                "mistral.mistral-large-2402-v1:0",
                "eu-west-1"
            ),
            "mistral.mistral-large-2402-v1:0"
        );
        assert_eq!(
            BedrockProvider::resolve_model_id_for_region("google.gemma-3-27b-it", "eu-west-1"),
            "google.gemma-3-27b-it"
        );
        assert_eq!(
            BedrockProvider::resolve_model_id_for_region("qwen.qwen3-32b-v1:0", "us-east-1"),
            "qwen.qwen3-32b-v1:0"
        );
        assert_eq!(
            BedrockProvider::resolve_model_id_for_region(
                "nvidia.nemotron-nano-12b-v2",
                "us-east-1"
            ),
            "nvidia.nemotron-nano-12b-v2"
        );
        assert_eq!(
            BedrockProvider::resolve_model_id_for_region("minimax.minimax-m2", "eu-west-1"),
            "minimax.minimax-m2"
        );
    }

    #[test]
    fn test_resolve_model_id_unknown_region_does_not_guess_prefix() {
        assert_eq!(
            BedrockProvider::resolve_model_id_for_region("amazon.nova-lite-v1:0", "cn-north-1"),
            "amazon.nova-lite-v1:0"
        );
    }

    // ====================================================================
    // Embedding Dimension Detection Tests
    // ====================================================================

    #[test]
    fn test_dimension_titan_embed_v2() {
        assert_eq!(
            BedrockProvider::dimension_for_model("amazon.titan-embed-text-v2:0"),
            1024
        );
    }

    #[test]
    fn test_dimension_titan_embed_v1() {
        assert_eq!(
            BedrockProvider::dimension_for_model("amazon.titan-embed-text-v1"),
            1536
        );
    }

    #[test]
    fn test_dimension_titan_embed_g1() {
        assert_eq!(
            BedrockProvider::dimension_for_model("amazon.titan-embed-g1-text-02"),
            1536
        );
    }

    #[test]
    fn test_dimension_cohere_embed_v4() {
        assert_eq!(
            BedrockProvider::dimension_for_model("cohere.embed-v4:0"),
            1536
        );
    }

    #[test]
    fn test_dimension_cohere_embed_v3() {
        assert_eq!(
            BedrockProvider::dimension_for_model("cohere.embed-english-v3"),
            1024
        );
        assert_eq!(
            BedrockProvider::dimension_for_model("cohere.embed-multilingual-v3"),
            1024
        );
    }

    #[test]
    fn test_dimension_nova_embed() {
        assert_eq!(
            BedrockProvider::dimension_for_model("amazon.nova-embed-v1:0"),
            1024
        );
    }

    #[test]
    fn test_dimension_unknown_defaults() {
        assert_eq!(
            BedrockProvider::dimension_for_model("some-unknown-embed-model"),
            DEFAULT_EMBEDDING_DIMENSION
        );
    }

    // ====================================================================
    // Embedding Request/Response Parsing Tests
    // ====================================================================

    #[test]
    fn test_build_embedding_request_titan() {
        let body = BedrockProvider::build_embedding_request(
            "amazon.titan-embed-text-v2:0",
            &["Hello world".to_string()],
        )
        .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["inputText"], "Hello world");
    }

    #[test]
    fn test_build_embedding_request_cohere() {
        let body = BedrockProvider::build_embedding_request(
            "cohere.embed-english-v3",
            &["Hello".to_string(), "World".to_string()],
        )
        .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["texts"], serde_json::json!(["Hello", "World"]));
        assert_eq!(json["input_type"], "search_query");
    }

    #[test]
    fn test_build_embedding_request_titan_rejects_batch() {
        let result = BedrockProvider::build_embedding_request(
            "amazon.titan-embed-text-v2:0",
            &["Hello".to_string(), "World".to_string()],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_embedding_response_titan() {
        let response = serde_json::json!({
            "embedding": [0.1, 0.2, 0.3],
            "inputTextTokenCount": 3
        });
        let bytes = serde_json::to_vec(&response).unwrap();
        let result =
            BedrockProvider::parse_embedding_response("amazon.titan-embed-text-v2:0", &bytes)
                .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 3);
        assert!((result[0][0] - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_parse_embedding_response_cohere_list() {
        let response = serde_json::json!({
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "id": "test",
            "response_type": "embeddings_floats"
        });
        let bytes = serde_json::to_vec(&response).unwrap();
        let result =
            BedrockProvider::parse_embedding_response("cohere.embed-english-v3", &bytes).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 2);
        assert_eq!(result[1].len(), 2);
    }

    #[test]
    fn test_parse_embedding_response_cohere_dict() {
        let response = serde_json::json!({
            "embeddings": {"float": [[0.5, 0.6, 0.7]]},
            "id": "test",
            "response_type": "embeddings_by_type"
        });
        let bytes = serde_json::to_vec(&response).unwrap();
        let result =
            BedrockProvider::parse_embedding_response("cohere.embed-v4:0", &bytes).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 3);
        assert!((result[0][0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_parse_embedding_response_rejects_non_numeric_value() {
        let response = serde_json::json!({
            "embedding": [0.1, "bad", 0.3]
        });
        let bytes = serde_json::to_vec(&response).unwrap();
        let result =
            BedrockProvider::parse_embedding_response("amazon.titan-embed-text-v2:0", &bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_embedding_response_rejects_invalid_cohere_dict() {
        let response = serde_json::json!({
            "embeddings": {"int8": [[1, 2, 3]]}
        });
        let bytes = serde_json::to_vec(&response).unwrap();
        let result = BedrockProvider::parse_embedding_response("cohere.embed-v4:0", &bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_cohere_embedding() {
        assert!(BedrockProvider::is_cohere_embedding(
            "cohere.embed-english-v3"
        ));
        assert!(BedrockProvider::is_cohere_embedding("cohere.embed-v4:0"));
        assert!(!BedrockProvider::is_cohere_embedding(
            "amazon.titan-embed-text-v2:0"
        ));
        assert!(!BedrockProvider::is_cohere_embedding(
            "cohere.command-r-plus-v1:0"
        ));
    }

    #[test]
    fn test_embedding_max_tokens() {
        assert_eq!(
            BedrockProvider::embedding_max_tokens_for_model("amazon.titan-embed-text-v2:0"),
            8192
        );
        assert_eq!(
            BedrockProvider::embedding_max_tokens_for_model("amazon.titan-embed-text-v1"),
            512
        );
        assert_eq!(
            BedrockProvider::embedding_max_tokens_for_model("cohere.embed-english-v3"),
            2048
        );
    }

    // ====================================================================
    // New unit tests — ADR-001 §7.2
    // ====================================================================

    // G7: Magistral series is ON_DEMAND only — must NOT receive a geo prefix
    // (verified via `aws bedrock list-foundation-models` 2026-04-04)
    #[test]
    fn test_resolve_model_id_magistral_no_prefix_us() {
        assert_eq!(
            BedrockProvider::resolve_model_id_for_region(
                "mistral.magistral-small-2509",
                "us-east-1"
            ),
            "mistral.magistral-small-2509",
            "magistral is ON_DEMAND only in Bedrock catalog; adding a prefix breaks the call"
        );
    }

    #[test]
    fn test_resolve_model_id_magistral_no_prefix_eu() {
        assert_eq!(
            BedrockProvider::resolve_model_id_for_region(
                "mistral.magistral-medium-2506",
                "eu-west-1"
            ),
            "mistral.magistral-medium-2506",
            "magistral-medium is ON_DEMAND only; must not receive eu. prefix"
        );
    }

    // thinking_budget_tokens field on CompletionOptions
    #[test]
    fn test_thinking_budget_tokens_default_none() {
        let opts = CompletionOptions::default();
        assert!(
            opts.thinking_budget_tokens.is_none(),
            "thinking_budget_tokens should be None by default"
        );
    }

    #[test]
    fn test_thinking_budget_tokens_some() {
        let opts = CompletionOptions {
            thinking_budget_tokens: Some(5_000),
            ..Default::default()
        };
        assert_eq!(opts.thinking_budget_tokens, Some(5_000));
    }

    // json_to_document roundtrip covers Document::Number PosInt branch
    // (same path used for thinking budget serialisation)
    #[test]
    fn test_json_to_document_number_pos_int() {
        let val = serde_json::json!({"budget_tokens": 8192u64});
        let doc = BedrockProvider::json_to_document(&val);
        if let Document::Object(map) = doc {
            match map.get("budget_tokens") {
                Some(Document::Number(aws_smithy_types::Number::PosInt(n))) => {
                    assert_eq!(*n, 8192u64);
                }
                other => panic!("Expected PosInt(8192), got {other:?}"),
            }
        } else {
            panic!("Expected Document::Object");
        }
    }

    // G5: URL images must be rejected with InvalidRequest
    #[test]
    fn test_convert_messages_url_image_rejected() {
        use crate::traits::ImageData;

        let img = ImageData {
            data: "https://example.com/photo.jpg".to_string(),
            mime_type: "image/jpeg".to_string(),
            detail: None,
        };
        let msg = ChatMessage {
            role: ChatRole::User,
            content: "Look at this".to_string(),
            images: Some(vec![img]),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            cache_control: None,
        };
        let result = BedrockProvider::convert_messages(&[msg], None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, LlmError::InvalidRequest(_)),
            "Expected InvalidRequest, got {err:?}"
        );
    }

    // G5: Unsupported MIME type must be rejected with InvalidRequest
    #[test]
    fn test_convert_messages_unsupported_mime_rejected() {
        use crate::traits::ImageData;

        let img = ImageData {
            data: "aGVsbG8=".to_string(), // valid base64 content
            mime_type: "image/bmp".to_string(),
            detail: None,
        };
        let msg = ChatMessage {
            role: ChatRole::User,
            content: "Look at this".to_string(),
            images: Some(vec![img]),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            cache_control: None,
        };
        let result = BedrockProvider::convert_messages(&[msg], None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, LlmError::InvalidRequest(_)),
            "Expected InvalidRequest for unsupported MIME, got {err:?}"
        );
    }

    // G5: Valid base64 PNG image should convert without error
    #[test]
    fn test_convert_messages_valid_png_image() {
        use crate::traits::ImageData;

        // Minimal 1×1 red PNG encoded in base64
        let png_b64 =
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg==";
        let img = ImageData {
            data: png_b64.to_string(),
            mime_type: "image/png".to_string(),
            detail: None,
        };
        let msg = ChatMessage {
            role: ChatRole::User,
            content: "Describe this image.".to_string(),
            images: Some(vec![img]),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            cache_control: None,
        };
        let result = BedrockProvider::convert_messages(&[msg], None);
        assert!(result.is_ok(), "Expected Ok for valid PNG image");
        let (messages, _system) = result.unwrap();
        assert_eq!(messages.len(), 1);
        // The message should have 2 content blocks: text + image
        let content = messages[0].content();
        assert_eq!(content.len(), 2, "Expected text block + image block");
        assert!(
            matches!(content[0], ContentBlock::Text(_)),
            "First block should be text"
        );
        assert!(
            matches!(content[1], ContentBlock::Image(_)),
            "Second block should be image"
        );
    }

    #[test]
    fn test_extract_content_text_without_reasoning_still_returns_none() {
        let msg = Message::builder()
            .role(ConversationRole::Assistant)
            .content(ContentBlock::Text("Final answer".to_string()))
            .build()
            .unwrap();
        let output = ConverseOutput::Message(msg);
        let (_text, _tools, thinking) = BedrockProvider::extract_content(&output);
        assert!(thinking.is_none());
    }
}
