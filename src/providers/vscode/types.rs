//! Type definitions for VSCode Copilot API.
//!
//! These types match the OpenAI-compatible API format used by the Copilot API.
//!
//! # Design Rationale
//!
//! WHY: We define explicit types rather than using `serde_json::Value` because:
//! 1. **Type Safety** - Compile-time verification of request/response structure
//! 2. **Documentation** - Types serve as living documentation of the API
//! 3. **IDE Support** - Autocomplete and type hints for developers
//! 4. **Validation** - Invalid data fails at deserialization, not at runtime
//!
//! # Type Categories
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    API Type Hierarchy                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                   │
//! │  Request Types (sent to API)                                     │
//! │  ├── ChatCompletionRequest   → /chat/completions                │
//! │  │   ├── RequestMessage      → role + content + tools           │
//! │  │   ├── RequestTool         → function definitions             │
//! │  │   └── ResponseFormat      → JSON mode                        │
//! │  └── EmbeddingRequest        → /embeddings                      │
//! │                                                                   │
//! │  Response Types (received from API)                              │
//! │  ├── ChatCompletionResponse  → Non-streaming response           │
//! │  │   ├── ResponseChoice      → message + finish_reason          │
//! │  │   └── Usage               → token counts                     │
//! │  ├── ChatCompletionChunk     → Streaming response               │
//! │  │   └── ChunkChoice         → delta content                    │
//! │  ├── EmbeddingResponse       → Embedding vectors                │
//! │  └── ModelsResponse          → Available models                 │
//! │                                                                   │
//! │  Shared Types                                                     │
//! │  ├── ResponseToolCall        → Tool calls in responses          │
//! │  ├── Model                   → Model metadata                   │
//! │  └── ModelCapabilities       → Limits and features              │
//! │                                                                   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Serialization
//!
//! All types use `serde` for JSON serialization with these conventions:
//! - Optional fields use `#[serde(skip_serializing_if = "Option::is_none")]`
//! - Default values use `#[serde(default)]`
//! - Renamed fields use `#[serde(rename = "...")]`
//!
//! # OpenAI Compatibility
//!
//! These types are compatible with OpenAI's API format, which is also used by:
//! - GitHub Copilot API
//! - Azure OpenAI
//! - Many open-source LLM servers (vLLM, Ollama, etc.)

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Chat completion request (OpenAI-compatible format).
#[derive(Debug, Clone, Serialize, Default)]
pub struct ChatCompletionRequest {
    /// Array of messages in the conversation.
    pub messages: Vec<RequestMessage>,

    /// Model identifier (e.g., "gpt-4o-mini", "gpt-4o").
    pub model: String,

    /// Sampling temperature (0.0-2.0). Higher = more random.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Nucleus sampling (top_p). Alternative to temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,

    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    /// Enable streaming responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Frequency penalty (-2.0 to 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Presence penalty (-2.0 to 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Response format specification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// Tools available for the model to call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<RequestTool>>,

    /// How the model should select tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<JsonValue>,

    /// Whether to allow parallel tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}

/// Tool definition for the API request.
#[derive(Debug, Clone, Serialize)]
pub struct RequestTool {
    /// Type of tool (always "function").
    #[serde(rename = "type")]
    pub tool_type: String,

    /// Function definition.
    pub function: RequestFunction,
}

/// Function definition for tool calling.
#[derive(Debug, Clone, Serialize)]
pub struct RequestFunction {
    /// Name of the function.
    pub name: String,

    /// Description of what the function does.
    pub description: String,

    /// JSON Schema for the function parameters.
    pub parameters: JsonValue,

    /// Whether to use strict mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

// ============================================================================
// Image Support Types (OODA-55)
// ============================================================================
//
// VS Code Copilot uses OpenAI-compatible format for images:
//
// Text only:                        With images:
// ┌─────────────────────────┐      ┌─────────────────────────────────┐
// │ content: "Hello"        │      │ content: [                      │
// └─────────────────────────┘      │   {type: "text", text: "..."},  │
//                                  │   {type: "image_url",           │
//                                  │    image_url: {url: "data:..."}}│
//                                  │ ]                               │
//                                  └─────────────────────────────────┘
//
// WHY: Serde untagged allows backward-compatible serialization
// ============================================================================

/// Request content that can be text or multipart (OODA-55).
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(untagged)]
pub enum RequestContent {
    /// Simple text content (backward compatible)
    Text(String),
    /// Multipart content with text and images
    Parts(Vec<ContentPart>),
}

/// Content part for multipart messages (OODA-55).
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content part
    #[serde(rename = "text")]
    Text { text: String },
    /// Image URL content part
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlContent },
}

/// Image URL content (OODA-55).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ImageUrlContent {
    /// Data URI or URL of the image
    pub url: String,
    /// Detail level: "auto", "low", or "high"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// A single message in the conversation.
#[derive(Debug, Clone, Serialize)]
pub struct RequestMessage {
    /// Role of the message sender.
    pub role: String,

    /// Content of the message (text or multipart with images).
    /// OODA-55: Changed from `Option<String>` to `Option<RequestContent>`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<RequestContent>,

    /// Optional name for the sender.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Tool calls made by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ResponseToolCall>>,

    /// Tool call ID (for tool role messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// Cache control hint (for Anthropic Claude via VSCode proxy).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<RequestCacheControl>,
}

/// Cache control hint for prompt caching (Anthropic Claude).
#[derive(Debug, Clone, Serialize)]
pub struct RequestCacheControl {
    /// Cache type (e.g., "ephemeral").
    #[serde(rename = "type")]
    pub cache_type: String,
}

/// Response format specification.
#[derive(Debug, Clone, Serialize)]
pub struct ResponseFormat {
    /// Format type: "text" or "json_object".
    #[serde(rename = "type")]
    pub format_type: String,
}

/// Chat completion response (non-streaming).
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ChatCompletionResponse {
    /// Unique identifier for the completion.
    pub id: String,

    /// Object type (always "chat.completion").
    #[serde(default)]
    pub object: Option<String>,

    /// Unix timestamp of creation.
    #[serde(default)]
    pub created: Option<u64>,

    /// Model used for generation.
    pub model: String,

    /// Array of completion choices.
    pub choices: Vec<Choice>,

    /// Token usage statistics.
    pub usage: Option<Usage>,

    /// Extra fields we don't use but need for deserialization compatibility.
    #[serde(flatten)]
    pub extra: Option<serde_json::Value>,
}

/// A single completion choice.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Choice {
    /// Index of this choice (optional - Anthropic models omit this field entirely).
    #[serde(default)]
    pub index: Option<usize>,

    /// The generated message.
    pub message: ResponseMessage,

    /// Reason for completion stop.
    pub finish_reason: Option<String>,

    /// Extra fields like content_filter_results.
    #[serde(flatten)]
    pub extra: Option<serde_json::Value>,
}

/// Response message from the assistant.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ResponseMessage {
    /// Role (usually "assistant").
    pub role: String,

    /// Generated content.
    pub content: Option<String>,

    /// Tool calls made by the assistant.
    #[serde(default)]
    pub tool_calls: Option<Vec<ResponseToolCall>>,

    /// Extra fields like padding.
    #[serde(flatten)]
    pub extra: Option<serde_json::Value>,
}

/// Tool call in the response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseToolCall {
    /// Unique identifier for this tool call.
    pub id: String,

    /// Type of tool (always "function").
    #[serde(rename = "type")]
    pub call_type: String,

    /// Function call details.
    pub function: ResponseFunctionCall,
}

/// Function call details in a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFunctionCall {
    /// Name of the function.
    pub name: String,

    /// JSON-encoded arguments.
    pub arguments: String,
}

/// Prompt token details including cache statistics (OODA-24).
///
/// WHY: KV cache hits are 10x cheaper. Tracking cached_tokens enables:
/// - Cache hit rate monitoring
/// - Prompt structure optimization
/// - Accurate cost calculations
#[derive(Debug, Clone, Deserialize, Default)]
pub struct PromptTokensDetails {
    /// Number of tokens served from KV cache.
    /// When present, indicates the prompt prefix was cached.
    #[serde(default)]
    pub cached_tokens: Option<usize>,
}

/// Token usage statistics.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Usage {
    /// Tokens in the prompt.
    pub prompt_tokens: usize,

    /// Tokens in the completion.
    pub completion_tokens: usize,

    /// Total tokens used.
    pub total_tokens: usize,

    /// Breakdown of prompt tokens for cache tracking (OODA-24).
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,

    /// Extra fields like completion_tokens_details.
    #[serde(flatten)]
    pub extra: Option<serde_json::Value>,
}

/// Streaming chunk.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ChatCompletionChunk {
    /// Unique identifier.
    pub id: String,

    /// Object type (always "chat.completion.chunk").
    #[serde(default)]
    pub object: Option<String>,

    /// Unix timestamp.
    #[serde(default)]
    pub created: Option<u64>,

    /// Model used.
    #[serde(default)]
    pub model: Option<String>,

    /// Array of delta choices.
    pub choices: Vec<ChunkChoice>,

    /// Optional usage stats (usually in last chunk).
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// A single chunk choice.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ChunkChoice {
    /// Index of this choice (optional - Anthropic models omit this field entirely).
    #[serde(default)]
    pub index: Option<usize>,

    /// Delta content.
    pub delta: Delta,

    /// Finish reason (if complete).
    pub finish_reason: Option<String>,
}

/// Delta content in a streaming chunk.
///
/// OODA-05: Added tool_calls field for streaming tool call support.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Delta {
    /// Role (present in first chunk).
    pub role: Option<String>,

    /// Content delta.
    pub content: Option<String>,

    /// Tool calls delta (OODA-05).
    ///
    /// During streaming, tool calls arrive incrementally:
    /// 1. First chunk: index + id + function.name
    /// 2. Subsequent chunks: function.arguments (partial JSON)
    #[serde(default)]
    pub tool_calls: Option<Vec<DeltaToolCall>>,
}

/// Tool call delta in streaming responses (OODA-05).
///
/// Tool calls stream incrementally:
/// ```text
/// Chunk 1: {index: 0, id: "call_abc", function: {name: "write_file"}}
/// Chunk 2: {index: 0, function: {arguments: "{\"path\":"}}
/// Chunk 3: {index: 0, function: {arguments: " \"./test.py\""}}
/// ...
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct DeltaToolCall {
    /// Tool call index (for parallel calls).
    pub index: usize,

    /// Tool call ID (present in first chunk for this tool).
    #[serde(default)]
    pub id: Option<String>,

    /// Tool type (always "function").
    #[serde(rename = "type", default)]
    pub tool_type: Option<String>,

    /// Function details (partial).
    #[serde(default)]
    pub function: Option<DeltaFunction>,
}

/// Partial function data in streaming (OODA-05).
#[derive(Debug, Clone, Deserialize)]
pub struct DeltaFunction {
    /// Function name (present in first chunk).
    #[serde(default)]
    pub name: Option<String>,

    /// Function arguments (partial JSON, accumulates across chunks).
    #[serde(default)]
    pub arguments: Option<String>,
}

// ============================================================================
// Models API Types
// ============================================================================

/// Models list response.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ModelsResponse {
    /// Array of available models.
    pub data: Vec<Model>,

    /// Object type.
    #[serde(default)]
    pub object: Option<String>,
}

/// Model limits configuration.
#[derive(Debug, Clone, Default, Deserialize)]
#[allow(dead_code)]
pub struct ModelLimits {
    /// Maximum context window tokens.
    #[serde(default)]
    pub max_context_window_tokens: Option<usize>,

    /// Maximum output tokens.
    #[serde(default)]
    pub max_output_tokens: Option<usize>,

    /// Maximum prompt tokens.
    #[serde(default)]
    pub max_prompt_tokens: Option<usize>,

    /// Maximum inputs (for embeddings).
    #[serde(default)]
    pub max_inputs: Option<usize>,
}

/// Model supported features.
#[derive(Debug, Clone, Default, Deserialize)]
#[allow(dead_code)]
pub struct ModelSupports {
    /// Whether tool/function calls are supported.
    #[serde(default)]
    pub tool_calls: Option<bool>,

    /// Whether parallel tool calls are supported.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,

    /// Whether custom dimensions are supported (for embeddings).
    #[serde(default)]
    pub dimensions: Option<bool>,
}

/// Model capabilities.
#[derive(Debug, Clone, Default, Deserialize)]
#[allow(dead_code)]
pub struct ModelCapabilities {
    /// Model family (e.g., "gpt-4o", "gpt-4").
    #[serde(default)]
    pub family: Option<String>,

    /// Model limits.
    #[serde(default)]
    pub limits: ModelLimits,

    /// Object type.
    #[serde(default)]
    pub object: Option<String>,

    /// Supported features.
    #[serde(default)]
    pub supports: ModelSupports,

    /// Tokenizer name.
    #[serde(default)]
    pub tokenizer: Option<String>,

    /// Model type (e.g., "chat", "embeddings").
    #[serde(default, rename = "type")]
    pub model_type: Option<String>,
}

/// Model policy information.
#[derive(Debug, Clone, Default, Deserialize)]
#[allow(dead_code)]
pub struct ModelPolicy {
    /// Policy state.
    #[serde(default)]
    pub state: Option<String>,

    /// Terms of use.
    #[serde(default)]
    pub terms: Option<String>,
}

/// A single model from the Copilot API.
///
/// This struct matches the full model schema from GitHub Copilot API,
/// including capabilities, limits, and supported features.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Model {
    /// Model identifier (e.g., "gpt-4o", "gpt-4o-mini").
    pub id: String,

    /// Object type (usually "model").
    #[serde(default)]
    pub object: Option<String>,

    /// Human-readable name.
    #[serde(default)]
    pub name: Option<String>,

    /// Model vendor (e.g., "azure-openai").
    #[serde(default)]
    pub vendor: Option<String>,

    /// Model version.
    #[serde(default)]
    pub version: Option<String>,

    /// Model capabilities including limits and supported features.
    #[serde(default)]
    pub capabilities: Option<ModelCapabilities>,

    /// Whether model appears in model picker.
    #[serde(default)]
    pub model_picker_enabled: Option<bool>,

    /// Whether this is a preview model.
    #[serde(default)]
    pub preview: Option<bool>,

    /// Model policy.
    #[serde(default)]
    pub policy: Option<ModelPolicy>,

    // Legacy fields for backward compatibility
    /// Creation timestamp (legacy OpenAI format).
    #[serde(default)]
    pub created: Option<u64>,

    /// Owner of the model (legacy OpenAI format).
    #[serde(default)]
    pub owned_by: Option<String>,
}

impl Model {
    /// Get the maximum context window tokens for this model.
    pub fn max_context_tokens(&self) -> Option<usize> {
        self.capabilities
            .as_ref()
            .and_then(|c| c.limits.max_context_window_tokens)
    }

    /// Get the maximum output tokens for this model.
    pub fn max_output_tokens(&self) -> Option<usize> {
        self.capabilities
            .as_ref()
            .and_then(|c| c.limits.max_output_tokens)
    }

    /// Check if this model supports tool calls.
    pub fn supports_tools(&self) -> bool {
        self.capabilities
            .as_ref()
            .and_then(|c| c.supports.tool_calls)
            .unwrap_or(false)
    }
}

// ============================================================================
// Embeddings API Types
// ============================================================================

/// Input for embedding request - can be a single string or array.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// Single text input.
    Single(String),
    /// Multiple text inputs.
    Multiple(Vec<String>),
}

impl From<String> for EmbeddingInput {
    fn from(s: String) -> Self {
        EmbeddingInput::Single(s)
    }
}

impl From<&str> for EmbeddingInput {
    fn from(s: &str) -> Self {
        EmbeddingInput::Single(s.to_string())
    }
}

impl From<Vec<String>> for EmbeddingInput {
    fn from(v: Vec<String>) -> Self {
        EmbeddingInput::Multiple(v)
    }
}

/// Embedding request.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingRequest {
    /// Input text(s) to embed.
    pub input: EmbeddingInput,

    /// Model to use for embeddings.
    pub model: String,
}

impl EmbeddingRequest {
    /// Create a new embedding request for a single input.
    pub fn new(input: impl Into<EmbeddingInput>, model: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            model: model.into(),
        }
    }
}

/// A single embedding result.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Embedding {
    /// Object type (always "embedding").
    pub object: String,

    /// The embedding vector.
    pub embedding: Vec<f32>,

    /// Index of this embedding in the request.
    pub index: usize,
}

/// Usage statistics for embedding request.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct EmbeddingUsage {
    /// Tokens in the prompt.
    pub prompt_tokens: usize,

    /// Total tokens used.
    pub total_tokens: usize,
}

/// Embedding response.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct EmbeddingResponse {
    /// Object type (always "list").
    pub object: String,

    /// Array of embeddings.
    pub data: Vec<Embedding>,

    /// Model used for embeddings.
    pub model: String,

    /// Usage statistics.
    pub usage: EmbeddingUsage,
}

impl EmbeddingResponse {
    /// Get the first embedding vector, if available.
    pub fn first_embedding(&self) -> Option<&Vec<f32>> {
        self.data.first().map(|e| &e.embedding)
    }

    /// Get all embedding vectors.
    pub fn embeddings(&self) -> Vec<&Vec<f32>> {
        self.data.iter().map(|e| &e.embedding).collect()
    }

    /// Get the embedding dimension.
    pub fn dimension(&self) -> Option<usize> {
        self.data.first().map(|e| e.embedding.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_input_from_string() {
        let input: EmbeddingInput = "hello world".into();
        match input {
            EmbeddingInput::Single(s) => assert_eq!(s, "hello world"),
            _ => panic!("Expected Single variant"),
        }
    }

    #[test]
    fn test_embedding_input_from_owned_string() {
        let input: EmbeddingInput = String::from("hello world").into();
        match input {
            EmbeddingInput::Single(s) => assert_eq!(s, "hello world"),
            _ => panic!("Expected Single variant"),
        }
    }

    #[test]
    fn test_embedding_input_from_vec() {
        let texts = vec!["hello".to_string(), "world".to_string()];
        let input: EmbeddingInput = texts.into();
        match input {
            EmbeddingInput::Multiple(v) => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[0], "hello");
                assert_eq!(v[1], "world");
            }
            _ => panic!("Expected Multiple variant"),
        }
    }

    #[test]
    fn test_embedding_request_serialization() {
        let request = EmbeddingRequest::new("test input", "text-embedding-3-small");
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"input\":\"test input\""));
        assert!(json.contains("\"model\":\"text-embedding-3-small\""));
    }

    #[test]
    fn test_embedding_request_multiple_inputs() {
        let input = EmbeddingInput::Multiple(vec!["one".to_string(), "two".to_string()]);
        let request = EmbeddingRequest::new(input, "model");
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("[\"one\",\"two\"]"));
    }

    #[test]
    fn test_embedding_response_deserialization() {
        let json = r#"{
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0
                }
            ],
            "model": "text-embedding-3-small",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }"#;

        let response: EmbeddingResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.object, "list");
        assert_eq!(response.model, "text-embedding-3-small");
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].embedding, vec![0.1, 0.2, 0.3]);
        assert_eq!(response.usage.prompt_tokens, 5);
    }

    #[test]
    fn test_embedding_response_helpers() {
        let json = r#"{
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [1.0, 2.0], "index": 0},
                {"object": "embedding", "embedding": [3.0, 4.0], "index": 1}
            ],
            "model": "test",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        }"#;

        let response: EmbeddingResponse = serde_json::from_str(json).unwrap();

        // Test first_embedding
        let first = response.first_embedding().unwrap();
        assert_eq!(*first, vec![1.0, 2.0]);

        // Test dimension
        assert_eq!(response.dimension(), Some(2));

        // Test embeddings
        let all = response.embeddings();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_model_deserialization_full() {
        let json = r#"{
            "id": "gpt-4o",
            "object": "model",
            "name": "GPT-4o",
            "vendor": "azure-openai",
            "version": "2024-08-06",
            "capabilities": {
                "family": "gpt-4o",
                "limits": {
                    "max_context_window_tokens": 128000,
                    "max_output_tokens": 16384
                },
                "supports": {
                    "tool_calls": true,
                    "parallel_tool_calls": true
                },
                "tokenizer": "cl100k_base",
                "type": "chat"
            },
            "model_picker_enabled": true,
            "preview": false
        }"#;

        let model: Model = serde_json::from_str(json).unwrap();
        assert_eq!(model.id, "gpt-4o");
        assert_eq!(model.name, Some("GPT-4o".to_string()));
        assert_eq!(model.vendor, Some("azure-openai".to_string()));
        assert!(model.supports_tools());
        assert_eq!(model.max_context_tokens(), Some(128000));
        assert_eq!(model.max_output_tokens(), Some(16384));
    }

    #[test]
    fn test_model_deserialization_minimal() {
        // Test with minimal fields (OpenAI format)
        let json = r#"{
            "id": "gpt-4",
            "object": "model",
            "created": 1687882410,
            "owned_by": "openai"
        }"#;

        let model: Model = serde_json::from_str(json).unwrap();
        assert_eq!(model.id, "gpt-4");
        assert_eq!(model.created, Some(1687882410));
        assert_eq!(model.owned_by, Some("openai".to_string()));
        assert!(!model.supports_tools()); // No capabilities = no tools
    }

    #[test]
    fn test_model_limits() {
        let limits = ModelLimits {
            max_context_window_tokens: Some(128000),
            max_output_tokens: Some(4096),
            max_prompt_tokens: None,
            max_inputs: None,
        };

        assert_eq!(limits.max_context_window_tokens, Some(128000));
        assert_eq!(limits.max_output_tokens, Some(4096));
    }

    #[test]
    fn test_models_response_deserialization() {
        let json = r#"{
            "data": [
                {"id": "gpt-4o", "object": "model"},
                {"id": "gpt-4o-mini", "object": "model"}
            ]
        }"#;

        let response: ModelsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.data.len(), 2);
        assert_eq!(response.data[0].id, "gpt-4o");
        assert_eq!(response.data[1].id, "gpt-4o-mini");
    }

    // ========================================================================
    // Chat Completion Response Fixture Tests
    // ========================================================================

    #[test]
    fn test_chat_completion_response_deserialization() {
        let json = r#"{
            "id": "chatcmpl-ABC123",
            "object": "chat.completion",
            "created": 1699876543,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20
            }
        }"#;

        let response: ChatCompletionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "chatcmpl-ABC123");
        assert_eq!(response.object, Some("chat.completion".to_string()));
        assert_eq!(response.created, Some(1699876543));
        assert_eq!(response.model, "gpt-4o-mini");
        assert_eq!(response.choices.len(), 1);

        let choice = &response.choices[0];
        assert_eq!(choice.index, Some(0));
        assert_eq!(choice.message.role, "assistant");
        assert_eq!(
            choice.message.content,
            Some("Hello! How can I help you today?".to_string())
        );
        assert_eq!(choice.finish_reason, Some("stop".to_string()));

        let usage = response.usage.as_ref().unwrap();
        assert_eq!(usage.prompt_tokens, 12);
        assert_eq!(usage.completion_tokens, 8);
        assert_eq!(usage.total_tokens, 20);
    }

    #[test]
    fn test_chat_completion_response_with_tool_calls() {
        let json = r#"{
            "id": "chatcmpl-tool123",
            "object": "chat.completion",
            "created": 1699876543,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{\"location\": \"San Francisco\"}"
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 25,
                "total_tokens": 75
            }
        }"#;

        let response: ChatCompletionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "chatcmpl-tool123");

        let choice = &response.choices[0];
        assert_eq!(choice.finish_reason, Some("tool_calls".to_string()));
        assert!(choice.message.content.is_none());

        let tool_calls = choice.message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_abc123");
        assert_eq!(tool_calls[0].call_type, "function");
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert!(tool_calls[0].function.arguments.contains("San Francisco"));
    }

    #[test]
    fn test_chat_completion_response_multiple_choices() {
        let json = r#"{
            "id": "chatcmpl-multi",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Option A"},
                    "finish_reason": "stop"
                },
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "Option B"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        }"#;

        let response: ChatCompletionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.choices.len(), 2);
        assert_eq!(
            response.choices[0].message.content,
            Some("Option A".to_string())
        );
        assert_eq!(
            response.choices[1].message.content,
            Some("Option B".to_string())
        );
    }

    #[test]
    fn test_chat_completion_response_with_extra_fields() {
        // Test that we handle extra fields from Copilot API gracefully
        let json = r#"{
            "id": "chatcmpl-extra",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                    "content_filter_results": {
                        "hate": {"filtered": false}
                    }
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            "system_fingerprint": "fp_abc123"
        }"#;

        let response: ChatCompletionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "chatcmpl-extra");
        assert_eq!(
            response.choices[0].message.content,
            Some("Hello".to_string())
        );
        // Extra fields should be captured but not cause deserialization to fail
    }

    // ========================================================================
    // Streaming Chunk Fixture Tests
    // ========================================================================

    #[test]
    fn test_streaming_chunk_first_chunk() {
        // First chunk typically contains role
        let json = r#"{
            "id": "chatcmpl-stream123",
            "object": "chat.completion.chunk",
            "created": 1699876543,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": ""
                    },
                    "finish_reason": null
                }
            ]
        }"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.id, "chatcmpl-stream123");
        assert_eq!(chunk.object, Some("chat.completion.chunk".to_string()));
        assert_eq!(chunk.model, Some("gpt-4o".to_string()));

        let choice = &chunk.choices[0];
        assert_eq!(choice.delta.role, Some("assistant".to_string()));
        assert_eq!(choice.delta.content, Some("".to_string()));
        assert!(choice.finish_reason.is_none());
    }

    #[test]
    fn test_streaming_chunk_content_delta() {
        // Middle chunks contain content
        let json = r#"{
            "id": "chatcmpl-stream123",
            "object": "chat.completion.chunk",
            "created": 1699876543,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "Hello"
                    },
                    "finish_reason": null
                }
            ]
        }"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();
        let choice = &chunk.choices[0];
        assert!(choice.delta.role.is_none());
        assert_eq!(choice.delta.content, Some("Hello".to_string()));
    }

    #[test]
    fn test_streaming_chunk_final_chunk() {
        // Final chunk has finish_reason and possibly usage
        let json = r#"{
            "id": "chatcmpl-stream123",
            "object": "chat.completion.chunk",
            "created": 1699876543,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 50,
                "total_tokens": 70
            }
        }"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();
        let choice = &chunk.choices[0];
        assert!(choice.delta.content.is_none());
        assert_eq!(choice.finish_reason, Some("stop".to_string()));

        let usage = chunk.usage.as_ref().unwrap();
        assert_eq!(usage.total_tokens, 70);
    }

    #[test]
    fn test_streaming_chunk_empty_delta() {
        // Sometimes delta is completely empty
        let json = r#"{
            "id": "chatcmpl-stream123",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": null
                }
            ]
        }"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();
        let choice = &chunk.choices[0];
        assert!(choice.delta.role.is_none());
        assert!(choice.delta.content.is_none());
    }

    // ========================================================================
    // Request Serialization Tests
    // ========================================================================

    #[test]
    fn test_chat_completion_request_minimal() {
        let request = ChatCompletionRequest {
            messages: vec![RequestMessage {
                role: "user".to_string(),
                content: Some(RequestContent::Text("Hello".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                cache_control: None,
            }],
            model: "gpt-4o".to_string(),
            ..Default::default()
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"Hello\""));
        assert!(json.contains("\"model\":\"gpt-4o\""));
        // Optional fields should not appear
        assert!(!json.contains("temperature"));
        assert!(!json.contains("max_tokens"));
    }

    #[test]
    fn test_chat_completion_request_with_options() {
        let request = ChatCompletionRequest {
            messages: vec![RequestMessage {
                role: "user".to_string(),
                content: Some(RequestContent::Text("Hello".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                cache_control: None,
            }],
            model: "gpt-4o".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(1000),
            stream: Some(true),
            ..Default::default()
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"temperature\":0.7"));
        assert!(json.contains("\"max_tokens\":1000"));
        assert!(json.contains("\"stream\":true"));
    }

    #[test]
    fn test_chat_completion_request_with_tools() {
        let request = ChatCompletionRequest {
            messages: vec![RequestMessage {
                role: "user".to_string(),
                content: Some(RequestContent::Text("What's the weather?".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                cache_control: None,
            }],
            model: "gpt-4o".to_string(),
            tools: Some(vec![RequestTool {
                tool_type: "function".to_string(),
                function: RequestFunction {
                    name: "get_weather".to_string(),
                    description: "Get weather info".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }),
                    strict: Some(true),
                },
            }]),
            ..Default::default()
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"tools\""));
        assert!(json.contains("\"get_weather\""));
        assert!(json.contains("\"strict\":true"));
    }

    #[test]
    fn test_response_format_serialization() {
        let format = ResponseFormat {
            format_type: "json_object".to_string(),
        };

        let json = serde_json::to_string(&format).unwrap();
        assert!(json.contains("\"type\":\"json_object\""));
    }

    // ========================================================================
    // Tool Call Serialization/Deserialization Tests
    // ========================================================================

    #[test]
    fn test_tool_call_roundtrip() {
        let tool_call = ResponseToolCall {
            id: "call_123".to_string(),
            call_type: "function".to_string(),
            function: ResponseFunctionCall {
                name: "test_func".to_string(),
                arguments: "{\"arg\": \"value\"}".to_string(),
            },
        };

        // Serialize
        let json = serde_json::to_string(&tool_call).unwrap();
        assert!(json.contains("\"id\":\"call_123\""));
        assert!(json.contains("\"type\":\"function\""));

        // Deserialize
        let parsed: ResponseToolCall = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "call_123");
        assert_eq!(parsed.function.name, "test_func");
    }

    // ========================================================================
    // Model Capabilities Edge Cases
    // ========================================================================

    #[test]
    fn test_model_with_embedding_capabilities() {
        let json = r#"{
            "id": "text-embedding-3-small",
            "object": "model",
            "name": "Text Embedding 3 Small",
            "capabilities": {
                "family": "text-embedding-3",
                "limits": {
                    "max_inputs": 2048
                },
                "supports": {
                    "dimensions": true
                },
                "type": "embeddings"
            }
        }"#;

        let model: Model = serde_json::from_str(json).unwrap();
        assert_eq!(model.id, "text-embedding-3-small");

        let caps = model.capabilities.as_ref().unwrap();
        assert_eq!(caps.model_type, Some("embeddings".to_string()));
        assert_eq!(caps.limits.max_inputs, Some(2048));
        assert_eq!(caps.supports.dimensions, Some(true));
    }

    #[test]
    fn test_model_with_policy() {
        let json = r#"{
            "id": "claude-3.5-sonnet",
            "object": "model",
            "policy": {
                "state": "active",
                "terms": "https://example.com/terms"
            }
        }"#;

        let model: Model = serde_json::from_str(json).unwrap();
        let policy = model.policy.as_ref().unwrap();
        assert_eq!(policy.state, Some("active".to_string()));
        assert!(policy.terms.as_ref().unwrap().contains("terms"));
    }

    // ========================================================================
    // Request JSON Format Tests (Iteration 30)
    // ========================================================================
    // WHY: These tests verify the exact JSON structure matches GitHub Copilot API
    // expectations. The API is sensitive to field order and presence of null values.

    #[test]
    fn test_chat_request_json_format() {
        // WHY: Verify minimal request produces clean JSON without null fields
        // The Copilot API rejects requests with unexpected null values
        let request = ChatCompletionRequest {
            messages: vec![RequestMessage {
                role: "user".to_string(),
                content: Some(RequestContent::Text("Test".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                cache_control: None,
            }],
            model: "gpt-4o".to_string(),
            ..Default::default()
        };

        let json = serde_json::to_string(&request).unwrap();

        // Verify required fields are present
        assert!(json.contains("\"messages\""));
        assert!(json.contains("\"model\":\"gpt-4o\""));

        // Verify None fields are not serialized (skip_serializing_if)
        // This is critical for Copilot API compatibility
        assert!(!json.contains("\"temperature\":null"));
        assert!(!json.contains("\"max_tokens\":null"));
        assert!(!json.contains("\"tools\":null"));
        assert!(!json.contains("\"stop\":null"));
    }

    #[test]
    fn test_chat_request_with_tools_format() {
        // WHY: Verify tool-calling requests include proper "type" field structure
        // The Copilot API requires tools to have type: "function" explicitly
        let request = ChatCompletionRequest {
            messages: vec![RequestMessage {
                role: "user".to_string(),
                content: Some(RequestContent::Text("Search for files".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                cache_control: None,
            }],
            model: "gpt-4o".to_string(),
            tools: Some(vec![RequestTool {
                tool_type: "function".to_string(),
                function: RequestFunction {
                    name: "file_search".to_string(),
                    description: "Search for files in workspace".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": { "type": "string" }
                        }
                    }),
                    strict: None,
                },
            }]),
            ..Default::default()
        };

        let json = serde_json::to_string(&request).unwrap();

        // Verify tool structure matches Copilot API expectation
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("\"name\":\"file_search\""));
        assert!(json.contains("\"description\":\"Search for files in workspace\""));
        assert!(json.contains("\"parameters\""));

        // Verify strict field is omitted when None
        assert!(!json.contains("\"strict\":null"));
    }

    // ========================================================================
    // Response Edge Case Tests (Iteration 31)
    // ========================================================================
    // WHY: These tests verify graceful handling of edge cases that occur
    // in production but aren't covered by typical happy-path tests.
    // The Copilot API can return unusual but valid responses.

    #[test]
    fn test_response_null_content_no_tools() {
        // WHY: The API can return null content without tool_calls when:
        // 1. Content filtering removes all output
        // 2. The model decides not to respond
        // 3. Truncation occurs at an unfortunate boundary
        // We must handle this gracefully without panics.
        let json = r#"{
            "id": "chatcmpl-edge1",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": null},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5}
        }"#;

        let response: ChatCompletionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "chatcmpl-edge1");
        assert_eq!(response.choices.len(), 1);

        // Verify content is None and no panic occurs
        assert!(response.choices[0].message.content.is_none());
        assert!(response.choices[0].message.tool_calls.is_none());

        // Usage should still be present
        let usage = response.usage.as_ref().unwrap();
        assert_eq!(usage.completion_tokens, 0);
    }

    #[test]
    fn test_response_empty_choices() {
        // WHY: Empty choices can occur when:
        // 1. All choices are filtered by content policy
        // 2. Error conditions that don't set HTTP error status
        // 3. Edge cases in model configuration
        // Code must not assume choices[0] exists.
        let json = r#"{
            "id": "chatcmpl-edge2",
            "model": "gpt-4o",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}
        }"#;

        let response: ChatCompletionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "chatcmpl-edge2");

        // Verify empty choices doesn't cause panic
        assert!(response.choices.is_empty());
        assert_eq!(response.choices.len(), 0);

        // Verify first() returns None instead of panic
        assert!(response.choices.is_empty());
    }

    #[test]
    fn test_response_minimal_fields() {
        // WHY: Some providers return minimal responses without optional fields.
        // Our types use Option<T> and skip_serializing_if for compatibility.
        // This test verifies minimal response parsing succeeds.
        let json = r#"{
            "id": "chatcmpl-minimal",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop"
            }]
        }"#;

        let response: ChatCompletionResponse = serde_json::from_str(json).unwrap();

        // Verify required fields present
        assert_eq!(response.id, "chatcmpl-minimal");
        assert_eq!(response.model, "gpt-4o-mini");
        assert_eq!(response.choices.len(), 1);

        // Verify optional fields are None (not panic)
        assert!(response.usage.is_none());
        assert!(response.object.is_none());
        assert!(response.created.is_none());
    }

    // ========================================================================
    // Anthropic Response Format Tests (OODA-07)
    // ========================================================================
    // WHY: These tests verify handling of Anthropic/Claude model responses
    // which omit the `index` field entirely and may split content/tool_calls
    // across multiple choices.

    #[test]
    fn test_response_missing_index_anthropic() {
        // WHY: Anthropic models don't include index field in responses
        // Our Choice struct must handle this with Option<usize>
        let json = r#"{
            "id": "msg_claude_01",
            "model": "claude-haiku-4.5",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello! I can help you with that."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            }
        }"#;

        let response: ChatCompletionResponse = serde_json::from_str(json).unwrap();

        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].index, None); // Index should be None
        assert_eq!(
            response.choices[0].message.content,
            Some("Hello! I can help you with that.".to_string())
        );
        assert_eq!(response.choices[0].finish_reason, Some("stop".to_string()));
    }

    #[test]
    fn test_response_anthropic_split_choices() {
        // WHY: Claude models via Copilot API return TWO choices:
        // - Choice 1: Contains only content (thinking)
        // - Choice 2: Contains only tool_calls
        // This tests that deserialization succeeds (normalization tested separately in client)
        let json = r#"{
            "id": "msg_haiku_split",
            "model": "claude-haiku-4.5",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "I'll examine the file to understand its structure",
                        "role": "assistant"
                    }
                },
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "function": {
                                "arguments": "{\"path\":\"demo/game.js\"}",
                                "name": "read_file"
                            },
                            "id": "toolu_01ABC123",
                            "type": "function"
                        }]
                    }
                }
            ],
            "created": 1768984171,
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }"#;

        let response: ChatCompletionResponse = serde_json::from_str(json).unwrap();

        // Should successfully deserialize both choices
        assert_eq!(response.choices.len(), 2);

        // First choice has content only
        assert!(response.choices[0].message.content.is_some());
        assert!(response.choices[0].message.tool_calls.is_none());

        // Second choice has tool_calls only
        assert!(response.choices[1].message.content.is_none());
        assert!(response.choices[1].message.tool_calls.is_some());
        assert_eq!(
            response.choices[1]
                .message
                .tool_calls
                .as_ref()
                .unwrap()
                .len(),
            1
        );
    }

    // ========================================================================
    // Tool Result Message Tests (Iteration 34)
    // ========================================================================
    // WHY: Tool results complete the tool-calling cycle. The format must
    // exactly match Copilot API expectations for proper round-trip.

    #[test]
    fn test_tool_result_message_format() {
        // WHY: Tool result messages have role="tool" and include tool_call_id
        // This format is required for the model to correlate results with calls
        let message = RequestMessage {
            role: "tool".to_string(),
            content: Some(RequestContent::Text("File found: src/main.rs".to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: Some("call_abc123".to_string()),
            cache_control: None,
        };

        let json = serde_json::to_string(&message).unwrap();

        // Verify required fields
        assert!(json.contains("\"role\":\"tool\""));
        assert!(json.contains("\"tool_call_id\":\"call_abc123\""));
        assert!(json.contains("File found: src/main.rs"));

        // Verify None fields are omitted (skip_serializing_if)
        assert!(!json.contains("\"name\":null"));
        assert!(!json.contains("\"tool_calls\":null"));
    }

    #[test]
    fn test_tool_result_id_preserved() {
        // WHY: tool_call_id must match exactly for model correlation
        // Even slight differences break the tool calling flow
        let original_id = "call_xYz789AbC";

        let message = RequestMessage {
            role: "tool".to_string(),
            content: Some(RequestContent::Text("Result data".to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: Some(original_id.to_string()),
            cache_control: None,
        };

        // Serialize
        let json = serde_json::to_string(&message).unwrap();

        // ID must appear exactly in JSON
        assert!(
            json.contains(&format!("\"tool_call_id\":\"{}\"", original_id)),
            "tool_call_id must be preserved exactly in serialization"
        );

        // Verify ID appears only once
        assert_eq!(
            json.matches(original_id).count(),
            1,
            "tool_call_id should appear exactly once"
        );
    }

    #[test]
    fn test_tool_result_with_unicode() {
        // WHY: Tool results may contain unicode (file contents, search results)
        // Serialization must handle unicode without corruption
        let unicode_content = "搜索结果: 找到 3 个文件 🎯";

        let message = RequestMessage {
            role: "tool".to_string(),
            content: Some(RequestContent::Text(unicode_content.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: Some("call_search".to_string()),
            cache_control: None,
        };

        let json = serde_json::to_string(&message).unwrap();

        // Unicode should be encoded in JSON (may be escaped or direct)
        // Either way, parsing back should give original
        let _ = serde_json::from_str::<serde_json::Value>(&json).unwrap();

        // OODA-55: Content is now wrapped in RequestContent::Text
        // The JSON structure is just the string value directly for Text variant
        assert!(
            json.contains(unicode_content),
            "Unicode content must be preserved in JSON"
        );
    }
}
