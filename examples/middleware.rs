//! Middleware example
//!
//! Demonstrates the middleware system for cross-cutting concerns.
//!
//! Run with: cargo run --example middleware
//!
//! This example shows:
//! - Creating a middleware stack
//! - Using built-in logging and metrics middleware
//! - Creating custom middleware
//! - Processing LLM requests through middleware

use async_trait::async_trait;
use edgequake_llm::{
    ChatMessage, ChatRole, LLMMiddleware, LLMMiddlewareStack, LLMRequest, LLMResponse,
    LogLevel, LoggingLLMMiddleware, MetricsLLMMiddleware, Result,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🔗 EdgeQuake LLM - Middleware Example\n");
    println!("{}", "─".repeat(60));

    // Create a middleware stack
    let mut stack = LLMMiddlewareStack::new();

    // Add built-in logging middleware with debug level
    let logging = Arc::new(LoggingLLMMiddleware::with_level(LogLevel::Debug));
    stack.add(logging);

    // Add metrics middleware
    let metrics = Arc::new(MetricsLLMMiddleware::new());
    stack.add(metrics.clone());

    // Add custom validation middleware
    let validator = Arc::new(ValidationMiddleware::new(100)); // min 100 chars
    stack.add(validator.clone());

    // Add custom audit middleware
    let audit = Arc::new(AuditMiddleware::new());
    stack.add(audit.clone());

    println!("\n📦 Middleware Stack ({} middlewares):", stack.len());
    println!("   1. LoggingLLMMiddleware (Debug level)");
    println!("   2. MetricsLLMMiddleware (Token/time tracking)");
    println!("   3. ValidationMiddleware (Custom - min message length)");
    println!("   4. AuditMiddleware (Custom - request logging)");

    // Simulate some LLM requests
    println!("\n📤 Processing requests...\n");

    // Request 1: Valid request
    let request1 = create_sample_request(
        "You are a helpful assistant that explains complex topics simply.",
        "Explain how LLM middleware works in a production system.",
    );
    process_request(&stack, &request1, 1).await?;

    // Request 2: Another valid request
    let request2 = create_sample_request(
        "You are a code reviewer.",
        "Review this middleware implementation for potential improvements and thread safety.",
    );
    process_request(&stack, &request2, 2).await?;

    // Request 3: Short message (will fail validation)
    let request3 = create_sample_request("Assistant", "Hi");
    let result = process_request(&stack, &request3, 3).await;
    if result.is_err() {
        println!("   ⚠️  Request 3 rejected by middleware: {:?}", result.err());
    }

    // Display metrics summary
    let summary = metrics.get_summary();
    println!("\n📊 Metrics Summary:");
    println!("   Total Requests: {}", summary.total_requests);
    println!("   Total Tokens: {}", summary.total_tokens);
    println!("   Prompt Tokens: {}", summary.prompt_tokens);
    println!("   Completion Tokens: {}", summary.completion_tokens);
    println!("   Total Time: {}ms", summary.total_time_ms);
    if summary.total_requests > 0 {
        println!(
            "   Avg Latency: {:.1}ms",
            summary.total_time_ms as f64 / summary.total_requests as f64
        );
    }

    // Display audit log
    println!("\n📜 Audit Log:");
    for entry in audit.get_entries() {
        println!("   • {}", entry);
    }

    println!("\n{}", "─".repeat(60));
    println!("✨ Middleware enables clean separation of cross-cutting concerns!");

    Ok(())
}

fn create_sample_request(system: &str, user: &str) -> LLMRequest {
    let messages = vec![
        ChatMessage {
            role: ChatRole::System,
            content: system.to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
            cache_control: None,
        },
        ChatMessage {
            role: ChatRole::User,
            content: user.to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
            cache_control: None,
        },
    ];
    LLMRequest::new(messages, "openai", "gpt-4o")
}

async fn process_request(
    stack: &LLMMiddlewareStack,
    request: &LLMRequest,
    num: u32,
) -> Result<()> {
    println!("   Request {}: {} messages to {}", num, request.message_count(), request.model);

    // Execute before hooks
    stack.before(request).await?;

    // Simulate LLM response (in real usage, call provider.chat())
    let response = simulate_response();

    // Execute after hooks
    stack.after(request, &response, 150).await?;

    println!("   ✅ Request {} processed ({} tokens)\n", num, response.total_tokens);
    Ok(())
}

fn simulate_response() -> LLMResponse {
    LLMResponse {
        content: "This is a simulated response from the LLM provider.".to_string(),
        prompt_tokens: 50,
        completion_tokens: 25,
        total_tokens: 75,
        model: "gpt-4o".to_string(),
        finish_reason: Some("stop".to_string()),
        tool_calls: vec![],
        metadata: HashMap::new(),
        cache_hit_tokens: None,
        thinking_tokens: None,
        thinking_content: None,
    }
}

// ============================================================================
// Custom Middleware: Validation
// ============================================================================

/// Custom middleware that validates request parameters.
struct ValidationMiddleware {
    min_message_length: usize,
    rejected_count: AtomicU32,
}

impl ValidationMiddleware {
    fn new(min_message_length: usize) -> Self {
        Self {
            min_message_length,
            rejected_count: AtomicU32::new(0),
        }
    }

    #[allow(dead_code)]
    fn rejected_count(&self) -> u32 {
        self.rejected_count.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl LLMMiddleware for ValidationMiddleware {
    fn name(&self) -> &str {
        "validation"
    }

    async fn before(&self, request: &LLMRequest) -> Result<()> {
        // Check total content length
        let total_length: usize = request.messages.iter().map(|m| m.content.len()).sum();

        if total_length < self.min_message_length {
            self.rejected_count.fetch_add(1, Ordering::Relaxed);
            return Err(edgequake_llm::LlmError::InvalidRequest(format!(
                "Message content too short: {} chars (min: {})",
                total_length, self.min_message_length
            )));
        }

        Ok(())
    }
}

// ============================================================================
// Custom Middleware: Audit
// ============================================================================

/// Custom middleware that logs all requests for auditing.
struct AuditMiddleware {
    entries: std::sync::Mutex<Vec<String>>,
}

impl AuditMiddleware {
    fn new() -> Self {
        Self {
            entries: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn get_entries(&self) -> Vec<String> {
        self.entries.lock().unwrap().clone()
    }
}

#[async_trait]
impl LLMMiddleware for AuditMiddleware {
    fn name(&self) -> &str {
        "audit"
    }

    async fn before(&self, request: &LLMRequest) -> Result<()> {
        let entry = format!(
            "REQUEST: {} messages to {}/{}",
            request.message_count(),
            request.provider,
            request.model
        );
        self.entries.lock().unwrap().push(entry);
        Ok(())
    }

    async fn after(
        &self,
        request: &LLMRequest,
        response: &LLMResponse,
        duration_ms: u64,
    ) -> Result<()> {
        let entry = format!(
            "RESPONSE: {} tokens in {}ms from {}",
            response.total_tokens, duration_ms, request.model
        );
        self.entries.lock().unwrap().push(entry);
        Ok(())
    }
}
