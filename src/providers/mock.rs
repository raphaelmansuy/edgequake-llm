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

    // ---- Iteration 25: Additional mock provider tests ----

    #[test]
    fn test_mock_provider_default() {
        let p = MockProvider::default();
        assert_eq!(LLMProvider::name(&p), "mock");
        assert_eq!(LLMProvider::model(&p), "mock-model");
    }

    #[tokio::test]
    async fn test_mock_provider_default_response_when_empty() {
        let p = MockProvider::new();
        // No responses queued → returns "Mock response"
        let resp = p.complete("anything").await.unwrap();
        assert_eq!(resp.content, "Mock response");
    }

    #[tokio::test]
    async fn test_mock_provider_multiple_responses_fifo() {
        let p = MockProvider::new();
        p.add_response("first").await;
        p.add_response("second").await;
        p.add_response("third").await;

        assert_eq!(p.complete("").await.unwrap().content, "first");
        assert_eq!(p.complete("").await.unwrap().content, "second");
        assert_eq!(p.complete("").await.unwrap().content, "third");
        // Queue empty → default
        assert_eq!(p.complete("").await.unwrap().content, "Mock response");
    }

    #[tokio::test]
    async fn test_mock_provider_custom_embedding() {
        let p = MockProvider::new();
        // MockProvider::new() starts with one default [0.1; 1536] embedding.
        // Consume the default first.
        let _ = p.embed(&["consume_default".to_string()]).await.unwrap();
        // Now add custom and it will be next
        p.add_embedding(vec![1.0, 2.0, 3.0]).await;

        let embs = p.embed(&["hello".to_string()]).await.unwrap();
        assert_eq!(embs[0], vec![1.0, 2.0, 3.0]);
    }

    #[tokio::test]
    async fn test_mock_provider_embed_multiple_texts() {
        let p = MockProvider::new();
        // Default embedding is used when queue is empty
        let embs = p
            .embed(&["a".to_string(), "b".to_string()])
            .await
            .unwrap();
        assert_eq!(embs.len(), 2);
        assert_eq!(embs[0].len(), 1536);
        assert_eq!(embs[1].len(), 1536);
    }

    #[tokio::test]
    async fn test_mock_provider_embedding_provider_trait() {
        let p = MockProvider::new();
        assert_eq!(EmbeddingProvider::name(&p), "mock");
        assert_eq!(EmbeddingProvider::model(&p), "mock-embedding");
        assert_eq!(p.dimension(), 1536);
        assert_eq!(EmbeddingProvider::max_tokens(&p), 512);
    }

    #[tokio::test]
    async fn test_mock_provider_max_context_length() {
        let p = MockProvider::new();
        assert_eq!(p.max_context_length(), 4096);
    }

    #[tokio::test]
    async fn test_mock_provider_chat_delegation() {
        let p = MockProvider::new();
        p.add_response("chat response").await;
        let resp = p
            .chat(&[ChatMessage::user("hi")], None)
            .await
            .unwrap();
        assert_eq!(resp.content, "chat response");
    }

    #[tokio::test]
    async fn test_mock_provider_complete_with_options() {
        let p = MockProvider::new();
        p.add_response("opts response").await;
        let opts = CompletionOptions::with_temperature(0.5);
        let resp = p.complete_with_options("prompt", &opts).await.unwrap();
        assert_eq!(resp.content, "opts response");
    }

    #[tokio::test]
    async fn test_mock_provider_stream() {
        use futures::StreamExt;
        let p = MockProvider::new();
        p.add_response("streamed").await;
        let mut stream = p.stream("test").await.unwrap();
        let chunk = stream.next().await.unwrap().unwrap();
        assert_eq!(chunk, "streamed");
    }

    #[tokio::test]
    async fn test_mock_agent_default_impl() {
        let p = MockAgentProvider::default();
        assert_eq!(LLMProvider::name(&p), "mock-agent");
        assert_eq!(p.model(), "mock-agent");
    }

    #[tokio::test]
    async fn test_mock_agent_supports_traits() {
        let p = MockAgentProvider::new();
        assert!(p.supports_streaming());
        assert!(p.supports_tool_streaming());
        assert!(p.supports_function_calling());
        assert_eq!(p.max_context_length(), 128_000);
    }

    #[tokio::test]
    async fn test_mock_agent_call_count_tracking() {
        let p = MockAgentProvider::new();
        assert_eq!(p.call_count(), 0);

        p.add_response("a").await;
        p.add_response("b").await;
        p.complete("").await.unwrap();
        assert_eq!(p.call_count(), 1);
        p.complete("").await.unwrap();
        assert_eq!(p.call_count(), 2);
    }

    #[tokio::test]
    async fn test_mock_agent_is_exhausted() {
        let p = MockAgentProvider::new();
        assert!(p.is_exhausted().await);

        p.add_response("one").await;
        assert!(!p.is_exhausted().await);

        p.complete("").await.unwrap();
        assert!(p.is_exhausted().await);
    }

    #[tokio::test]
    async fn test_mock_agent_chat_delegation() {
        let p = MockAgentProvider::new();
        p.add_response("agent chat").await;
        let resp = p
            .chat(&[ChatMessage::user("hi")], None)
            .await
            .unwrap();
        assert_eq!(resp.content, "agent chat");
    }

    #[tokio::test]
    async fn test_mock_agent_complete_with_options() {
        let p = MockAgentProvider::new();
        p.add_response("agent opts").await;
        let opts = CompletionOptions::default();
        let resp = p.complete_with_options("prompt", &opts).await.unwrap();
        assert_eq!(resp.content, "agent opts");
    }
}
