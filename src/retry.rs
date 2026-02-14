//! Retry executor for LLM operations with exponential backoff.
//!
//! This module provides a retry executor that applies the appropriate
//! retry strategy based on the error type.
//!
//! # Usage
//!
//! ```ignore
//! use edgequake_llm::retry::RetryExecutor;
//! use edgequake_llm::error::RetryStrategy;
//!
//! let executor = RetryExecutor::new();
//! let result = executor.execute(
//!     &RetryStrategy::network_backoff(),
//!     || async { provider.complete(messages, options).await },
//! ).await;
//! ```
//!
//! @implements specs/improve-tools/006-error-handling.md
//! @iteration OODA-11

use crate::error::{LlmError, RetryStrategy};
use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, info, warn};

/// Executor for retry logic with configurable backoff strategies.
///
/// The executor wraps async operations and automatically retries them
/// according to the specified retry strategy.
#[derive(Debug, Default)]
pub struct RetryExecutor {
    /// Optional callback for logging retry attempts.
    log_retries: bool,
}

impl RetryExecutor {
    /// Create a new retry executor.
    pub fn new() -> Self {
        Self { log_retries: true }
    }

    /// Create a retry executor without logging.
    pub fn silent() -> Self {
        Self { log_retries: false }
    }

    /// Execute an async operation with automatic retry based on strategy.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The retry strategy to use
    /// * `operation` - Async closure that performs the operation
    ///
    /// # Returns
    ///
    /// The result of the operation, or the last error if all retries fail.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let executor = RetryExecutor::new();
    /// let result = executor.execute(
    ///     &RetryStrategy::network_backoff(),
    ///     || async { make_api_call().await },
    /// ).await;
    /// ```
    pub async fn execute<F, Fut, T>(
        &self,
        strategy: &RetryStrategy,
        mut operation: F,
    ) -> Result<T, LlmError>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, LlmError>>,
    {
        match strategy {
            RetryStrategy::NoRetry => operation().await,

            RetryStrategy::WaitAndRetry { wait } => {
                self.execute_wait_and_retry(*wait, operation).await
            }

            RetryStrategy::ExponentialBackoff {
                base_delay,
                max_delay,
                max_attempts,
            } => {
                self.execute_exponential_backoff(*base_delay, *max_delay, *max_attempts, operation)
                    .await
            }

            RetryStrategy::ReduceContext => {
                // We can't automatically reduce context here.
                // The caller should catch ReduceContext strategy and handle it.
                // Just attempt once.
                operation().await
            }
        }
    }

    /// Execute with wait-and-retry strategy.
    async fn execute_wait_and_retry<F, Fut, T>(
        &self,
        wait: Duration,
        mut operation: F,
    ) -> Result<T, LlmError>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, LlmError>>,
    {
        match operation().await {
            Ok(v) => Ok(v),
            Err(e) => {
                if self.log_retries {
                    warn!("Operation failed, waiting {:?} before retry: {}", wait, e);
                }
                sleep(wait).await;
                operation().await
            }
        }
    }

    /// Execute with exponential backoff strategy.
    async fn execute_exponential_backoff<F, Fut, T>(
        &self,
        base_delay: Duration,
        max_delay: Duration,
        max_attempts: u32,
        mut operation: F,
    ) -> Result<T, LlmError>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, LlmError>>,
    {
        let mut delay = base_delay;
        let mut attempts = 0;

        loop {
            attempts += 1;

            match operation().await {
                Ok(v) => {
                    if attempts > 1 && self.log_retries {
                        info!("Operation succeeded after {} attempts", attempts);
                    }
                    return Ok(v);
                }
                Err(e) => {
                    if attempts >= max_attempts {
                        if self.log_retries {
                            warn!(
                                "Operation failed after {} attempts, giving up: {}",
                                attempts, e
                            );
                        }
                        return Err(e);
                    }

                    // Check if the error itself suggests we shouldn't retry
                    let error_strategy = e.retry_strategy();
                    if matches!(error_strategy, RetryStrategy::NoRetry) {
                        if self.log_retries {
                            debug!("Error is non-retryable, stopping: {}", e);
                        }
                        return Err(e);
                    }

                    if self.log_retries {
                        warn!(
                            "Attempt {}/{} failed, retrying in {:?}: {}",
                            attempts, max_attempts, delay, e
                        );
                    }

                    sleep(delay).await;
                    delay = (delay * 2).min(max_delay);
                }
            }
        }
    }

    /// Execute an operation with automatic strategy detection.
    ///
    /// This variant automatically determines the retry strategy from
    /// the first error that occurs.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let executor = RetryExecutor::new();
    /// let result = executor.execute_auto(|| async {
    ///     make_api_call().await
    /// }).await;
    /// ```
    pub async fn execute_auto<F, Fut, T>(&self, mut operation: F) -> Result<T, LlmError>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, LlmError>>,
    {
        match operation().await {
            Ok(v) => Ok(v),
            Err(e) => {
                let strategy = e.retry_strategy();

                if !strategy.should_retry() {
                    return Err(e);
                }

                if self.log_retries {
                    debug!("First attempt failed, using strategy {:?}: {}", strategy, e);
                }

                // Sleep before next attempt based on strategy
                match &strategy {
                    RetryStrategy::WaitAndRetry { wait } => {
                        sleep(*wait).await;
                        operation().await
                    }
                    RetryStrategy::ExponentialBackoff {
                        base_delay,
                        max_delay,
                        max_attempts,
                    } => {
                        sleep(*base_delay).await;
                        // Continue with remaining attempts
                        self.execute_exponential_backoff(
                            *base_delay * 2, // Already waited base_delay
                            *max_delay,
                            max_attempts - 1, // Already used one attempt
                            operation,
                        )
                        .await
                    }
                    RetryStrategy::ReduceContext | RetryStrategy::NoRetry => {
                        // Shouldn't reach here due to should_retry check
                        Err(e)
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_no_retry_succeeds() {
        let executor = RetryExecutor::silent();
        let result = executor
            .execute(&RetryStrategy::NoRetry, || async { Ok::<_, LlmError>(42) })
            .await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_no_retry_fails_immediately() {
        let executor = RetryExecutor::silent();
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        let result = executor
            .execute(&RetryStrategy::NoRetry, || {
                let count = call_count_clone.clone();
                async move {
                    count.fetch_add(1, Ordering::SeqCst);
                    Err::<i32, _>(LlmError::AuthError("bad key".to_string()))
                }
            })
            .await;

        assert!(result.is_err());
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_exponential_backoff_retries() {
        let executor = RetryExecutor::silent();
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        let result = executor
            .execute(
                &RetryStrategy::ExponentialBackoff {
                    base_delay: Duration::from_millis(1),
                    max_delay: Duration::from_millis(10),
                    max_attempts: 3,
                },
                || {
                    let count = call_count_clone.clone();
                    async move {
                        let attempts = count.fetch_add(1, Ordering::SeqCst) + 1;
                        if attempts < 3 {
                            Err(LlmError::NetworkError("failed".to_string()))
                        } else {
                            Ok(42)
                        }
                    }
                },
            )
            .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_exponential_backoff_gives_up() {
        let executor = RetryExecutor::silent();
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        let result = executor
            .execute(
                &RetryStrategy::ExponentialBackoff {
                    base_delay: Duration::from_millis(1),
                    max_delay: Duration::from_millis(10),
                    max_attempts: 3,
                },
                || {
                    let count = call_count_clone.clone();
                    async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        Err::<i32, _>(LlmError::NetworkError("always fails".to_string()))
                    }
                },
            )
            .await;

        assert!(result.is_err());
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_wait_and_retry() {
        let executor = RetryExecutor::silent();
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        let result = executor
            .execute(
                &RetryStrategy::WaitAndRetry {
                    wait: Duration::from_millis(1),
                },
                || {
                    let count = call_count_clone.clone();
                    async move {
                        let attempts = count.fetch_add(1, Ordering::SeqCst) + 1;
                        if attempts < 2 {
                            Err(LlmError::RateLimited("wait".to_string()))
                        } else {
                            Ok(42)
                        }
                    }
                },
            )
            .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_stops_on_permanent_error() {
        let executor = RetryExecutor::silent();
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        let result = executor
            .execute(
                &RetryStrategy::ExponentialBackoff {
                    base_delay: Duration::from_millis(1),
                    max_delay: Duration::from_millis(10),
                    max_attempts: 5,
                },
                || {
                    let count = call_count_clone.clone();
                    async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        // Return a non-retryable error
                        Err::<i32, _>(LlmError::AuthError("invalid".to_string()))
                    }
                },
            )
            .await;

        assert!(result.is_err());
        // Should stop after first attempt since AuthError is non-retryable
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }
}
