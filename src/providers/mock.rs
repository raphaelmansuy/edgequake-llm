//! Mock LLM and Embedding provider for testing.
//!
//! # OODA-45: E2E Testing Infrastructure
//! 
//! This module provides deterministic mock providers for E2E testing:
//! - MockProvider: Basic LLM mock with queue-based responses
//! - MockAgentProvider: Advanced mock with tool call support for React agent testing
//!
//! ## Architecture
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Mock Provider System                         │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  MockProvider (basic)     MockAgentProvider (advanced)          │
//! │  ├── add_response()       ├── add_response()                    │
//! │  └── complete()           ├── add_tool_response()               │
//! │                           ├── chat_with_tools()                 │
//! │                           └── chat_with_tools_stream()          │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use async_trait::async_trait;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::error::Result;
use crate::traits::{
    ChatMessage, CompletionOptions, EmbeddingProvider, LLMProvider, LLMResponse,
    StreamChunk, ToolCall, ToolChoice, ToolDefinition,
};

/// Mock LLM provider for testing.
#[derive(Debug, Clone)]
pub struct MockProvider {
    responses: Arc<Mutex<Vec<String>>>,
    embeddings: Arc<Mutex<Vec<Vec<f32>>>>,
}

/// Advanced mock provider for React agent E2E testing.
/// 
/// Supports deterministic tool calling responses for testing agent workflows.
/// 
/// # Example
/// ```
/// use edgequake_llm::providers::MockAgentProvider;
/// use edgequake_llm::traits::ToolCall;
/// 
/// let provider = MockAgentProvider::new();
/// // Add a response with tool calls
/// provider.add_tool_response_sync("I'll create that file", vec![
///     ToolCall {
///         id: "call_1".to_string(),
///         call_type: "function".to_string(),
///         function: edgequake_llm::traits::FunctionCall {
///             name: "write_file".to_string(),
///             arguments: r#"{"path": "test.txt", "content": "hello"}"#.to_string(),
///         },
///     }
/// ]);
/// ```
#[derive(Debug, Clone)]
pub struct MockAgentProvider {
    /// Queue of responses (content + tool calls)
    responses: Arc<Mutex<Vec<MockResponse>>>,
    /// Number of responses consumed
    call_count: Arc<AtomicUsize>,
    /// Model name for testing model-specific behavior
    model_name: String,
}

/// A mock response containing content and tool calls.
#[derive(Debug, Clone)]
pub struct MockResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

impl MockProvider {
    /// Create a new mock provider with default responses.
    pub fn new() -> Self {
        Self {
            responses: Arc::new(Mutex::new(Vec::new())),
            embeddings: Arc::new(Mutex::new(vec![
                vec![0.1; 1536], // Default 1536-dim embedding
            ])),
        }
    }

    /// Add a response to the queue.
    pub async fn add_response(&self, response: impl Into<String>) {
        self.responses.lock().await.push(response.into());
    }

    /// Add an embedding to the queue.
    pub async fn add_embedding(&self, embedding: Vec<f32>) {
        self.embeddings.lock().await.push(embedding);
    }
}

impl Default for MockProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProvider for MockProvider {
    fn name(&self) -> &str {
        "mock"
    }

    fn model(&self) -> &str {
        "mock-model"
    }

    fn max_context_length(&self) -> usize {
        4096
    }

    async fn complete(&self, _prompt: &str) -> Result<LLMResponse> {
        let mut responses = self.responses.lock().await;
        let content = if responses.is_empty() {
            "Mock response".to_string()
        } else {
            responses.remove(0)
        };

        Ok(LLMResponse::new(content, "mock-model"))
    }

    async fn complete_with_options(
        &self,
        prompt: &str,
        _options: &crate::traits::CompletionOptions,
    ) -> Result<LLMResponse> {
        self.complete(prompt).await
    }

    async fn chat(
        &self,
        _messages: &[crate::traits::ChatMessage],
        _options: Option<&crate::traits::CompletionOptions>,
    ) -> Result<LLMResponse> {
        self.complete("").await
    }

    async fn stream(
        &self,
        prompt: &str,
    ) -> Result<futures::stream::BoxStream<'static, Result<String>>> {
        use futures::StreamExt;
        let response = self.complete(prompt).await?;
        let stream = futures::stream::iter(vec![Ok(response.content)]);
        Ok(stream.boxed())
    }
}

#[async_trait]
impl EmbeddingProvider for MockProvider {
    fn name(&self) -> &str {
        "mock"
    }

    fn model(&self) -> &str {
        "mock-embedding"
    }

    fn dimension(&self) -> usize {
        1536
    }

    fn max_tokens(&self) -> usize {
        512
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for _ in texts {
            let mut embeddings = self.embeddings.lock().await;
            let emb = if embeddings.is_empty() {
                vec![0.1; 1536]
            } else {
                embeddings.remove(0)
            };
            results.push(emb);
        }
        Ok(results)
    }
}

// ============================================================================
// MockAgentProvider - Advanced E2E Testing Provider
// ============================================================================

impl MockAgentProvider {
    /// Create a new mock agent provider.
    pub fn new() -> Self {
        Self {
            responses: Arc::new(Mutex::new(Vec::new())),
            call_count: Arc::new(AtomicUsize::new(0)),
            model_name: "mock-agent".to_string(),
        }
    }

    /// Create with a specific model name for testing model-specific behavior.
    pub fn with_model(model_name: impl Into<String>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(Vec::new())),
            call_count: Arc::new(AtomicUsize::new(0)),
            model_name: model_name.into(),
        }
    }

    /// Add a text-only response (no tool calls).
    pub async fn add_response(&self, content: impl Into<String>) {
        self.responses.lock().await.push(MockResponse {
            content: content.into(),
            tool_calls: vec![],
        });
    }

    /// Add a response with tool calls.
    pub async fn add_tool_response(&self, content: impl Into<String>, tool_calls: Vec<ToolCall>) {
        self.responses.lock().await.push(MockResponse {
            content: content.into(),
            tool_calls,
        });
    }

    /// Synchronous version of add_response for test setup.
    pub fn add_response_sync(&self, content: impl Into<String>) {
        // Use try_lock for synchronous contexts - should always succeed in test setup
        if let Ok(mut responses) = self.responses.try_lock() {
            responses.push(MockResponse {
                content: content.into(),
                tool_calls: vec![],
            });
        }
    }

    /// Synchronous version of add_tool_response for test setup.
    pub fn add_tool_response_sync(&self, content: impl Into<String>, tool_calls: Vec<ToolCall>) {
        if let Ok(mut responses) = self.responses.try_lock() {
            responses.push(MockResponse {
                content: content.into(),
                tool_calls,
            });
        }
    }

    /// Get the number of responses consumed.
    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }

    /// Check if all queued responses have been consumed.
    pub async fn is_exhausted(&self) -> bool {
        self.responses.lock().await.is_empty()
    }

    /// Get next response from queue.
    async fn next_response(&self) -> MockResponse {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let mut responses = self.responses.lock().await;
        if responses.is_empty() {
            // Default: task_complete with success message
            MockResponse {
                content: "Task completed successfully.".to_string(),
                tool_calls: vec![ToolCall {
                    id: format!("call_{}", self.call_count.load(Ordering::SeqCst)),
                    call_type: "function".to_string(),
                    function: crate::traits::FunctionCall {
                        name: "task_complete".to_string(),
                        arguments: r#"{"result": "success"}"#.to_string(),
                    },
                }],
            }
        } else {
            responses.remove(0)
        }
    }
}

impl Default for MockAgentProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProvider for MockAgentProvider {
    fn name(&self) -> &str {
        "mock-agent"
    }

    fn model(&self) -> &str {
        &self.model_name
    }

    fn max_context_length(&self) -> usize {
        128_000 // Simulate large context window
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_tool_streaming(&self) -> bool {
        true
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    async fn complete(&self, _prompt: &str) -> Result<LLMResponse> {
        let response = self.next_response().await;
        Ok(LLMResponse::new(response.content, &self.model_name))
    }

    async fn complete_with_options(
        &self,
        prompt: &str,
        _options: &CompletionOptions,
    ) -> Result<LLMResponse> {
        self.complete(prompt).await
    }

    async fn chat(
        &self,
        _messages: &[ChatMessage],
        _options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        self.complete("").await
    }

    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: &[ToolDefinition],
        _tool_choice: Option<ToolChoice>,
        _options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let mock_response = self.next_response().await;
        let mut response = LLMResponse::new(mock_response.content, &self.model_name);
        response.tool_calls = mock_response.tool_calls;
        Ok(response)
    }

    async fn chat_with_tools_stream(
        &self,
        _messages: &[ChatMessage],
        _tools: &[ToolDefinition],
        _tool_choice: Option<ToolChoice>,
        _options: Option<&CompletionOptions>,
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamChunk>>> {
        use futures::StreamExt;

        let mock_response = self.next_response().await;
        
        // Build stream chunks that simulate real streaming behavior
        // StreamChunk is an enum with Content, ToolCallDelta, and Finished variants
        let mut chunks = Vec::new();
        
        // Content chunk (if any)
        if !mock_response.content.is_empty() {
            chunks.push(Ok(StreamChunk::Content(mock_response.content.clone())));
        }
        
        // Tool call chunks (if any) - emit one delta per tool call
        for (index, tool_call) in mock_response.tool_calls.iter().enumerate() {
            chunks.push(Ok(StreamChunk::ToolCallDelta {
                index,
                id: Some(tool_call.id.clone()),
                function_name: Some(tool_call.function.name.clone()),
                function_arguments: Some(tool_call.function.arguments.clone()),
            }));
        }
        
        // Final chunk with finish reason
        chunks.push(Ok(StreamChunk::Finished {
            reason: "stop".to_string(),
            ttft_ms: None,
        }));
        
        let stream = futures::stream::iter(chunks);
        Ok(stream.boxed())
    }

    async fn stream(
        &self,
        prompt: &str,
    ) -> Result<futures::stream::BoxStream<'static, Result<String>>> {
        use futures::StreamExt;
        let response = self.complete(prompt).await?;
        let stream = futures::stream::iter(vec![Ok(response.content)]);
        Ok(stream.boxed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::FunctionCall;

    #[tokio::test]
    async fn test_mock_provider() {
        let provider = MockProvider::new();
        provider.add_response("This is a mock response.").await;

        // Test LLM
        let response = provider.complete("test").await.unwrap();
        assert_eq!(response.content, "This is a mock response.");

        // Test embedding
        let embedding = provider.embed_one("test").await.unwrap();
        assert_eq!(embedding.len(), 1536);
    }

    #[tokio::test]
    async fn test_custom_responses() {
        let provider = MockProvider::new();
        provider.add_response("Custom response").await;

        let response = provider.complete("test").await.unwrap();
        assert_eq!(response.content, "Custom response");
    }

    // ========================================================================
    // MockAgentProvider Tests
    // ========================================================================

    #[tokio::test]
    async fn test_mock_agent_provider_basic() {
        let provider = MockAgentProvider::new();
        provider.add_response("Hello from mock agent").await;

        let response = provider.complete("test").await.unwrap();
        assert_eq!(response.content, "Hello from mock agent");
        assert_eq!(provider.call_count(), 1);
    }

    #[tokio::test]
    async fn test_mock_agent_provider_with_tools() {
        let provider = MockAgentProvider::new();
        
        // Add response with tool call
        provider.add_tool_response(
            "I'll create that file for you.",
            vec![ToolCall {
                id: "call_1".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "write_file".to_string(),
                    arguments: r#"{"path": "test.txt", "content": "hello world"}"#.to_string(),
                },
            }],
        ).await;

        let response = provider.chat_with_tools(
            &[ChatMessage::user("Create test.txt")],
            &[],
            None,
            None,
        ).await.unwrap();

        assert!(response.content.contains("create that file"));
        assert!(!response.tool_calls.is_empty());
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].function.name, "write_file");
    }

    #[tokio::test]
    async fn test_mock_agent_provider_stream() {
        use futures::StreamExt;

        let provider = MockAgentProvider::new();
        provider.add_tool_response(
            "Creating file...",
            vec![ToolCall {
                id: "call_1".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "write_file".to_string(),
                    arguments: r#"{"path": "test.txt", "content": "hello"}"#.to_string(),
                },
            }],
        ).await;

        let mut stream = provider.chat_with_tools_stream(
            &[ChatMessage::user("Create file")],
            &[],
            None,
            None,
        ).await.unwrap();

        let mut content = String::new();
        let mut tool_call_count = 0;
        let mut finish_reason = None;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.unwrap();
            match chunk {
                StreamChunk::Content(delta) => content.push_str(&delta),
                StreamChunk::ThinkingContent { text, .. } => content.push_str(&text),
                StreamChunk::ToolCallDelta { .. } => tool_call_count += 1,
                StreamChunk::Finished { reason, .. } => finish_reason = Some(reason),
            }
        }

        assert_eq!(content, "Creating file...");
        assert_eq!(tool_call_count, 1);
        assert_eq!(finish_reason, Some("stop".to_string()));
    }

    #[tokio::test]
    async fn test_mock_agent_default_task_complete() {
        // When queue is empty, should return task_complete
        let provider = MockAgentProvider::new();
        
        let response = provider.chat_with_tools(
            &[ChatMessage::user("Do something")],
            &[],
            None,
            None,
        ).await.unwrap();

        assert!(!response.tool_calls.is_empty());
        assert_eq!(response.tool_calls[0].function.name, "task_complete");
    }

    #[tokio::test]
    async fn test_mock_agent_sync_setup() {
        let provider = MockAgentProvider::new();
        
        // Use sync methods for test setup
        provider.add_response_sync("Sync response 1");
        provider.add_tool_response_sync("With tools", vec![ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: r#"{"path": "test.txt"}"#.to_string(),
            },
        }]);

        let r1 = provider.complete("test").await.unwrap();
        assert_eq!(r1.content, "Sync response 1");

        let r2 = provider.chat_with_tools(&[], &[], None, None).await.unwrap();
        assert_eq!(r2.content, "With tools");
        assert!(!r2.tool_calls.is_empty());
    }

    #[tokio::test]
    async fn test_mock_agent_model_specific() {
        let provider = MockAgentProvider::with_model("claude-sonnet-4-20250514");
        assert_eq!(provider.model(), "claude-sonnet-4-20250514");
        
        let provider2 = MockAgentProvider::with_model("gpt-4-turbo");
        assert_eq!(provider2.model(), "gpt-4-turbo");
    }
}
