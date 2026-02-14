//! Retry Handling example
//!
//! Demonstrates error handling and automatic retry strategies.
//!
//! Run with: cargo run --example retry_handling
//!
//! This example shows:
//! - Different retry strategies (exponential backoff, wait-and-retry)
//! - Handling transient vs permanent errors
//! - Custom retry configurations
//! - Error categorization and recovery

use edgequake_llm::{LlmError, RetryExecutor, RetryStrategy};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 EdgeQuake LLM - Retry Handling Example\n");
    println!("{}", "─".repeat(60));

    // Demonstrate different retry strategies
    demonstrate_retry_strategies();
    
    // Demonstrate retry executor with simulated operations
    demonstrate_retry_executor().await;
    
    // Demonstrate error categorization
    demonstrate_error_handling();

    println!("\n{}", "─".repeat(60));
    println!("✨ Proper retry handling ensures resilient LLM applications!");
    
    Ok(())
}

fn demonstrate_retry_strategies() {
    println!("\n📋 Retry Strategies:\n");

    // 1. Network backoff - for transient network errors
    let network = RetryStrategy::network_backoff();
    println!("1. Network Backoff:");
    println!("   For: Network timeouts, connection errors");
    if let RetryStrategy::ExponentialBackoff { base_delay, max_delay, max_attempts } = network {
        println!("   Base delay: {:?}", base_delay);
        println!("   Max delay: {:?}", max_delay);
        println!("   Max attempts: {}", max_attempts);
    }

    // 2. Server backoff - for 5xx server errors
    let server = RetryStrategy::server_backoff();
    println!("\n2. Server Backoff:");
    println!("   For: Server errors (500, 502, 503)");
    if let RetryStrategy::ExponentialBackoff { base_delay, max_delay, max_attempts } = server {
        println!("   Base delay: {:?}", base_delay);
        println!("   Max delay: {:?}", max_delay);
        println!("   Max attempts: {}", max_attempts);
    }

    // 3. Wait-and-retry - for rate limits
    let rate_limit = RetryStrategy::WaitAndRetry {
        wait: Duration::from_secs(60),
    };
    println!("\n3. Wait-and-Retry:");
    println!("   For: Rate limit errors (429)");
    if let RetryStrategy::WaitAndRetry { wait } = rate_limit {
        println!("   Wait duration: {:?}", wait);
    }

    // 4. No retry - for permanent errors
    let no_retry = RetryStrategy::NoRetry;
    println!("\n4. No Retry:");
    println!("   For: Auth errors, invalid requests, model not found");
    println!("   Should retry: {}", no_retry.should_retry());

    // 5. Reduce context - for token limit errors
    let _reduce = RetryStrategy::ReduceContext;
    println!("\n5. Reduce Context:");
    println!("   For: Token limit exceeded");
    println!("   Action: Truncate messages and retry");
}

async fn demonstrate_retry_executor() {
    println!("\n🔄 Retry Executor Demo:\n");

    let executor = RetryExecutor::new();
    let attempt_count = Arc::new(AtomicU32::new(0));

    // Simulate an operation that fails twice then succeeds
    let count = attempt_count.clone();
    let strategy = RetryStrategy::ExponentialBackoff {
        base_delay: Duration::from_millis(50), // Short for demo
        max_delay: Duration::from_millis(200),
        max_attempts: 5,
    };

    println!("   Executing operation with exponential backoff...");
    println!("   (Simulating 2 failures before success)\n");

    let result = executor
        .execute(&strategy, || {
            let count = count.clone();
            async move {
                let attempt = count.fetch_add(1, Ordering::SeqCst) + 1;
                println!("   Attempt {}", attempt);
                
                if attempt < 3 {
                    // Simulate transient error
                    Err(LlmError::NetworkError("Connection reset".to_string()))
                } else {
                    Ok("Success on attempt 3!")
                }
            }
        })
        .await;

    match result {
        Ok(msg) => println!("\n   ✅ {}", msg),
        Err(e) => println!("\n   ❌ Failed after all retries: {}", e),
    }

    // Demonstrate silent executor
    println!("\n   Silent executor (no logging):");
    let silent_executor = RetryExecutor::silent();
    let count2 = Arc::new(AtomicU32::new(0));
    let count_inner = count2.clone();
    
    let result = silent_executor
        .execute(&RetryStrategy::network_backoff(), || {
            let count = count_inner.clone();
            async move {
                let attempt = count.fetch_add(1, Ordering::SeqCst) + 1;
                if attempt < 2 {
                    Err(LlmError::Timeout)
                } else {
                    Ok("Silent success")
                }
            }
        })
        .await;
    
    println!("   Attempts: {}, Result: {:?}", count2.load(Ordering::SeqCst), result.is_ok());
}

fn demonstrate_error_handling() {
    println!("\n⚠️  Error-to-Strategy Mapping:\n");

    // Show how different errors map to strategies
    let errors = vec![
        (LlmError::NetworkError("timeout".into()), "NetworkError"),
        (LlmError::RateLimited("too many requests".into()), "RateLimited"),
        (LlmError::AuthError("invalid key".into()), "AuthError"),
        (LlmError::TokenLimitExceeded { max: 4096, got: 5000 }, "TokenLimitExceeded"),
        (LlmError::ModelNotFound("gpt-99".into()), "ModelNotFound"),
    ];

    for (error, name) in errors {
        let strategy = error.retry_strategy();
        let recoverable = error.is_recoverable();
        
        println!("   {}:", name);
        println!("      Strategy: {:?}", strategy_name(&strategy));
        println!("      Recoverable: {}", recoverable);
        println!();
    }
}

fn strategy_name(strategy: &RetryStrategy) -> &'static str {
    match strategy {
        RetryStrategy::ExponentialBackoff { .. } => "ExponentialBackoff",
        RetryStrategy::WaitAndRetry { .. } => "WaitAndRetry",
        RetryStrategy::ReduceContext => "ReduceContext",
        RetryStrategy::NoRetry => "NoRetry",
    }
}
