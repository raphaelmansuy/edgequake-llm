//! LLM error types with retry strategies.
//!
//! # Error Handling Philosophy
//!
//! Errors should be:
//! 1. **Actionable**: Tell the user what to do, not just what went wrong
//! 2. **Specific**: Include relevant context (model name, token counts, etc.)
//! 3. **Recoverable**: Distinguish transient errors (retry) from permanent ones
//!
//! # Common Errors and Solutions
//!
//! | Error | Cause | Solution |
//! |-------|-------|----------|
//! | `AuthError` | Invalid/expired API key | Check `OPENAI_API_KEY` env var |
//! | `RateLimited` | Too many requests | Wait for `retry_after` seconds |
//! | `TokenLimitExceeded` | Input too long | Reduce chunk size or context |
//! | `ModelNotFound` | Invalid model name | Use `gpt-4o-mini` or `gpt-3.5-turbo` |
//! | `Timeout` | Network slow | Increase timeout or retry |
//!
//! # Retry Strategies
//!
//! Each error type has an associated retry strategy:
//! - `ExponentialBackoff`: For transient network/server errors
//! - `WaitAndRetry`: For rate limiting (wait specified duration)
//! - `ReduceContext`: For token limit errors (caller should reduce input)
//! - `NoRetry`: For permanent errors (auth, invalid request)
//!
//! @implements specs/improve-tools/006-error-handling.md
//! @iteration OODA-11

use std::time::Duration;
use thiserror::Error;

/// Result type for LLM operations.
pub type Result<T> = std::result::Result<T, LlmError>;

// ============================================================================
// Retry Strategy
// ============================================================================

/// Strategy for retrying failed LLM operations.
///
/// Each error type maps to an appropriate retry strategy based on
/// whether the error is transient (retry) or permanent (no retry).
#[derive(Debug, Clone, PartialEq)]
pub enum RetryStrategy {
    /// Retry with exponential backoff (for transient errors).
    ExponentialBackoff {
        /// Initial delay before first retry.
        base_delay: Duration,
        /// Maximum delay between retries.
        max_delay: Duration,
        /// Maximum number of retry attempts.
        max_attempts: u32,
    },

    /// Wait for a specific duration then retry once (for rate limits).
    WaitAndRetry {
        /// Duration to wait before retrying.
        wait: Duration,
    },

    /// Do not retry, but caller should reduce context size and try again.
    ReduceContext,

    /// Do not retry at all (permanent error).
    NoRetry,
}

impl RetryStrategy {
    /// Create a standard exponential backoff strategy for network errors.
    pub fn network_backoff() -> Self {
        Self::ExponentialBackoff {
            base_delay: Duration::from_millis(125),
            max_delay: Duration::from_secs(30),
            max_attempts: 5,
        }
    }

    /// Create a standard exponential backoff strategy for server errors.
    pub fn server_backoff() -> Self {
        Self::ExponentialBackoff {
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            max_attempts: 3,
        }
    }

    /// Check if this strategy allows retrying.
    pub fn should_retry(&self) -> bool {
        !matches!(self, Self::NoRetry)
    }
}

// ============================================================================
// LLM Error Types
// ============================================================================

/// Errors that can occur in LLM operations.
#[derive(Debug, Error)]
pub enum LlmError {
    /// API error from the provider.
    #[error("API error: {0}")]
    ApiError(String),

    /// Rate limit exceeded.
    #[error("Rate limit exceeded: {0}")]
    RateLimited(String),

    /// Invalid request parameters.
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Authentication error.
    #[error("Authentication error: {0}")]
    AuthError(String),

    /// Token limit exceeded.
    #[error("Token limit exceeded: max {max}, got {got}")]
    TokenLimitExceeded { max: usize, got: usize },

    /// Model not found.
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Network error.
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Provider-specific error.
    #[error("Provider error: {0}")]
    ProviderError(String),

    /// Timeout error.
    #[error("Request timed out")]
    Timeout,

    /// Feature not supported.
    #[error("Not supported: {0}")]
    NotSupported(String),

    /// Unknown error.
    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl From<reqwest::Error> for LlmError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            LlmError::Timeout
        } else if err.is_connect() {
            LlmError::NetworkError(format!("Connection failed: {}", err))
        } else {
            LlmError::NetworkError(err.to_string())
        }
    }
}

impl From<async_openai::error::OpenAIError> for LlmError {
    fn from(err: async_openai::error::OpenAIError) -> Self {
        match err {
            async_openai::error::OpenAIError::ApiError(api_err) => {
                let message = api_err.message.clone();
                if message.contains("rate limit") || message.contains("Rate limit") {
                    LlmError::RateLimited(message)
                } else if message.contains("authentication") || message.contains("invalid_api_key")
                {
                    LlmError::AuthError(message)
                } else if message.contains("model") && message.contains("not found") {
                    LlmError::ModelNotFound(message)
                } else {
                    LlmError::ApiError(message)
                }
            }
            async_openai::error::OpenAIError::Reqwest(req_err) => LlmError::from(req_err),
            async_openai::error::OpenAIError::JSONDeserialize(json_err) => {
                LlmError::SerializationError(json_err)
            }
            _ => LlmError::ProviderError(err.to_string()),
        }
    }
}

// ============================================================================
// Retry Strategy Methods
// ============================================================================

impl LlmError {
    /// Get the appropriate retry strategy for this error.
    ///
    /// # Returns
    ///
    /// - `ExponentialBackoff` for transient network/server errors
    /// - `WaitAndRetry` for rate limiting
    /// - `ReduceContext` for token limit errors
    /// - `NoRetry` for permanent errors (auth, invalid request, etc.)
    ///
    /// # Example
    ///
    /// ```
    /// use edgequake_llm::{LlmError, RetryStrategy};
    ///
    /// let error = LlmError::NetworkError("connection failed".to_string());
    /// let strategy = error.retry_strategy();
    /// assert!(strategy.should_retry());
    /// ```
    pub fn retry_strategy(&self) -> RetryStrategy {
        match self {
            // Transient network errors - aggressive retry
            Self::NetworkError(_) | Self::Timeout => RetryStrategy::network_backoff(),

            // Rate limiting - wait the specified duration
            Self::RateLimited(_) => RetryStrategy::WaitAndRetry {
                wait: Duration::from_secs(60),
            },

            // Server errors - moderate retry
            Self::ApiError(msg)
                if msg.contains("500") || msg.contains("502") || msg.contains("503") =>
            {
                RetryStrategy::server_backoff()
            }
            Self::ProviderError(_) => RetryStrategy::server_backoff(),

            // Token limit - caller should reduce context
            Self::TokenLimitExceeded { .. } => RetryStrategy::ReduceContext,

            // Permanent errors - no retry
            Self::AuthError(_)
            | Self::InvalidRequest(_)
            | Self::ModelNotFound(_)
            | Self::ConfigError(_)
            | Self::NotSupported(_) => RetryStrategy::NoRetry,

            // Default for other errors - conservative retry
            Self::ApiError(_) | Self::SerializationError(_) | Self::Unknown(_) => {
                RetryStrategy::ExponentialBackoff {
                    base_delay: Duration::from_secs(1),
                    max_delay: Duration::from_secs(30),
                    max_attempts: 2,
                }
            }
        }
    }

    /// Get a user-friendly description of the error with suggested action.
    ///
    /// # Example
    ///
    /// ```
    /// use edgequake_llm::LlmError;
    ///
    /// let error = LlmError::AuthError("invalid key".to_string());
    /// assert!(error.user_description().contains("API key"));
    /// ```
    pub fn user_description(&self) -> String {
        match self {
            Self::NetworkError(_) => {
                "Unable to connect to the API. Check your internet connection.".to_string()
            }
            Self::Timeout => "Request timed out. The server may be overloaded.".to_string(),
            Self::RateLimited(_) => "Rate limited by the API. Waiting before retry...".to_string(),
            Self::TokenLimitExceeded { max, got } => {
                format!(
                    "Context too large ({}/{} tokens). Reducing context and retrying...",
                    got, max
                )
            }
            Self::AuthError(_) => {
                "Authentication failed. Please check your API key is valid and not expired."
                    .to_string()
            }
            Self::ModelNotFound(model) => {
                format!(
                    "Model '{}' not found. Use a supported model like 'gpt-4o-mini'.",
                    model
                )
            }
            Self::InvalidRequest(msg) => {
                format!("Invalid request: {}. Check your parameters.", msg)
            }
            Self::ConfigError(msg) => format!("Configuration error: {}.", msg),
            Self::NotSupported(feature) => {
                format!("Feature '{}' is not supported by this provider.", feature)
            }
            Self::ApiError(_) | Self::ProviderError(_) => {
                "API server error. Retrying...".to_string()
            }
            Self::SerializationError(_) => {
                "Failed to parse API response. This may be a temporary issue.".to_string()
            }
            Self::Unknown(msg) => format!("An unexpected error occurred: {}", msg),
        }
    }

    /// Check if this error is recoverable (can be retried).
    pub fn is_recoverable(&self) -> bool {
        self.retry_strategy().should_retry()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_error_display() {
        let error = LlmError::ApiError("something went wrong".to_string());
        assert_eq!(error.to_string(), "API error: something went wrong");

        let error = LlmError::RateLimited("too many requests".to_string());
        assert_eq!(error.to_string(), "Rate limit exceeded: too many requests");

        let error = LlmError::InvalidRequest("bad params".to_string());
        assert_eq!(error.to_string(), "Invalid request: bad params");
    }

    #[test]
    fn test_llm_error_auth() {
        let error = LlmError::AuthError("invalid key".to_string());
        assert_eq!(error.to_string(), "Authentication error: invalid key");
    }

    #[test]
    fn test_llm_error_token_limit() {
        let error = LlmError::TokenLimitExceeded {
            max: 4096,
            got: 5000,
        };
        assert_eq!(
            error.to_string(),
            "Token limit exceeded: max 4096, got 5000"
        );
    }

    #[test]
    fn test_llm_error_model_not_found() {
        let error = LlmError::ModelNotFound("gpt-5-turbo".to_string());
        assert_eq!(error.to_string(), "Model not found: gpt-5-turbo");
    }

    #[test]
    fn test_llm_error_network() {
        let error = LlmError::NetworkError("connection refused".to_string());
        assert_eq!(error.to_string(), "Network error: connection refused");
    }

    #[test]
    fn test_llm_error_config() {
        let error = LlmError::ConfigError("missing api key".to_string());
        assert_eq!(error.to_string(), "Configuration error: missing api key");
    }

    #[test]
    fn test_llm_error_provider() {
        let error = LlmError::ProviderError("openai specific error".to_string());
        assert_eq!(error.to_string(), "Provider error: openai specific error");
    }

    #[test]
    fn test_llm_error_timeout() {
        let error = LlmError::Timeout;
        assert_eq!(error.to_string(), "Request timed out");
    }

    #[test]
    fn test_llm_error_not_supported() {
        let error = LlmError::NotSupported("function calling".to_string());
        assert_eq!(error.to_string(), "Not supported: function calling");
    }

    #[test]
    fn test_llm_error_unknown() {
        let error = LlmError::Unknown("mystery error".to_string());
        assert_eq!(error.to_string(), "Unknown error: mystery error");
    }

    #[test]
    fn test_llm_error_debug() {
        let error = LlmError::ApiError("test".to_string());
        let debug = format!("{:?}", error);
        assert!(debug.contains("ApiError"));
        assert!(debug.contains("test"));
    }

    #[test]
    fn test_llm_error_from_serde_json() {
        let json_str = "not json at all";
        let json_err: serde_json::Error =
            serde_json::from_str::<serde_json::Value>(json_str).unwrap_err();
        let llm_err: LlmError = json_err.into();
        assert!(matches!(llm_err, LlmError::SerializationError(_)));
    }

    // ========================================================================
    // Retry Strategy Tests
    // ========================================================================

    #[test]
    fn test_network_error_retry_strategy() {
        let error = LlmError::NetworkError("connection failed".to_string());
        let strategy = error.retry_strategy();

        match strategy {
            RetryStrategy::ExponentialBackoff { max_attempts, .. } => {
                assert_eq!(max_attempts, 5);
            }
            _ => panic!("Expected ExponentialBackoff for network error"),
        }
        assert!(strategy.should_retry());
        assert!(error.is_recoverable());
    }

    #[test]
    fn test_timeout_retry_strategy() {
        let error = LlmError::Timeout;
        let strategy = error.retry_strategy();

        assert!(matches!(strategy, RetryStrategy::ExponentialBackoff { .. }));
        assert!(strategy.should_retry());
    }

    #[test]
    fn test_rate_limited_retry_strategy() {
        let error = LlmError::RateLimited("too many requests".to_string());
        let strategy = error.retry_strategy();

        match strategy {
            RetryStrategy::WaitAndRetry { wait } => {
                assert_eq!(wait, Duration::from_secs(60));
            }
            _ => panic!("Expected WaitAndRetry for rate limit"),
        }
        assert!(strategy.should_retry());
    }

    #[test]
    fn test_token_limit_reduce_context_strategy() {
        let error = LlmError::TokenLimitExceeded {
            max: 4096,
            got: 5000,
        };
        let strategy = error.retry_strategy();

        assert!(matches!(strategy, RetryStrategy::ReduceContext));
        assert!(strategy.should_retry());
    }

    #[test]
    fn test_auth_error_no_retry() {
        let error = LlmError::AuthError("invalid key".to_string());
        let strategy = error.retry_strategy();

        assert!(matches!(strategy, RetryStrategy::NoRetry));
        assert!(!strategy.should_retry());
        assert!(!error.is_recoverable());
    }

    #[test]
    fn test_invalid_request_no_retry() {
        let error = LlmError::InvalidRequest("bad params".to_string());
        assert!(matches!(error.retry_strategy(), RetryStrategy::NoRetry));
    }

    #[test]
    fn test_model_not_found_no_retry() {
        let error = LlmError::ModelNotFound("gpt-5".to_string());
        assert!(matches!(error.retry_strategy(), RetryStrategy::NoRetry));
    }

    #[test]
    fn test_user_description_network() {
        let error = LlmError::NetworkError("connection refused".to_string());
        let desc = error.user_description();
        assert!(desc.contains("internet connection"));
    }

    #[test]
    fn test_user_description_auth() {
        let error = LlmError::AuthError("invalid".to_string());
        let desc = error.user_description();
        assert!(desc.contains("API key"));
    }

    #[test]
    fn test_user_description_token_limit() {
        let error = LlmError::TokenLimitExceeded {
            max: 4096,
            got: 5000,
        };
        let desc = error.user_description();
        assert!(desc.contains("5000/4096"));
        assert!(desc.contains("Reducing"));
    }

    #[test]
    fn test_retry_strategy_equality() {
        let s1 = RetryStrategy::network_backoff();
        let s2 = RetryStrategy::network_backoff();
        assert_eq!(s1, s2);

        let s3 = RetryStrategy::NoRetry;
        assert_ne!(s1, s3);
    }

    // ========================================================================
    // user_description() coverage for all variants
    // ========================================================================

    #[test]
    fn test_user_description_timeout() {
        let error = LlmError::Timeout;
        let desc = error.user_description();
        assert!(desc.contains("timed out"));
    }

    #[test]
    fn test_user_description_rate_limited() {
        let error = LlmError::RateLimited("slow down".to_string());
        let desc = error.user_description();
        assert!(desc.contains("Rate limited"));
    }

    #[test]
    fn test_user_description_model_not_found() {
        let error = LlmError::ModelNotFound("gpt-5".to_string());
        let desc = error.user_description();
        assert!(desc.contains("gpt-5"));
        assert!(desc.contains("not found"));
    }

    #[test]
    fn test_user_description_not_supported() {
        let error = LlmError::NotSupported("streaming".to_string());
        let desc = error.user_description();
        assert!(desc.contains("streaming"));
        assert!(desc.contains("not supported"));
    }

    #[test]
    fn test_user_description_unknown() {
        let error = LlmError::Unknown("mystery".to_string());
        let desc = error.user_description();
        assert!(desc.contains("mystery"));
    }

    #[test]
    fn test_user_description_api_error() {
        let error = LlmError::ApiError("server crashed".to_string());
        let desc = error.user_description();
        assert!(desc.contains("Retrying"));
    }

    #[test]
    fn test_user_description_provider_error() {
        let error = LlmError::ProviderError("internal failure".to_string());
        let desc = error.user_description();
        assert!(desc.contains("Retrying"));
    }

    #[test]
    fn test_user_description_serialization() {
        let json_err = serde_json::from_str::<serde_json::Value>("bad").unwrap_err();
        let error = LlmError::SerializationError(json_err);
        let desc = error.user_description();
        assert!(desc.contains("parse"));
    }

    #[test]
    fn test_user_description_config() {
        let error = LlmError::ConfigError("missing field".to_string());
        let desc = error.user_description();
        assert!(desc.contains("Configuration"));
    }

    #[test]
    fn test_user_description_invalid_request() {
        let error = LlmError::InvalidRequest("empty prompt".to_string());
        let desc = error.user_description();
        assert!(desc.contains("empty prompt"));
    }

    // ========================================================================
    // retry_strategy() remaining branches
    // ========================================================================

    #[test]
    fn test_api_error_500_server_backoff() {
        let error = LlmError::ApiError("HTTP 500 internal server error".to_string());
        let strategy = error.retry_strategy();
        match strategy {
            RetryStrategy::ExponentialBackoff { max_attempts, .. } => {
                assert_eq!(max_attempts, 3); // server_backoff has 3 attempts
            }
            _ => panic!("Expected ExponentialBackoff for 500 error"),
        }
    }

    #[test]
    fn test_api_error_502_server_backoff() {
        let error = LlmError::ApiError("502 bad gateway".to_string());
        assert!(matches!(
            error.retry_strategy(),
            RetryStrategy::ExponentialBackoff { .. }
        ));
    }

    #[test]
    fn test_api_error_503_server_backoff() {
        let error = LlmError::ApiError("503 service unavailable".to_string());
        assert!(matches!(
            error.retry_strategy(),
            RetryStrategy::ExponentialBackoff { .. }
        ));
    }

    #[test]
    fn test_provider_error_server_backoff() {
        let error = LlmError::ProviderError("internal issue".to_string());
        let strategy = error.retry_strategy();
        match strategy {
            RetryStrategy::ExponentialBackoff {
                base_delay,
                max_delay,
                max_attempts,
            } => {
                assert_eq!(base_delay, Duration::from_secs(1));
                assert_eq!(max_delay, Duration::from_secs(60));
                assert_eq!(max_attempts, 3);
            }
            _ => panic!("Expected server_backoff for ProviderError"),
        }
    }

    #[test]
    fn test_unknown_error_retry_strategy() {
        let error = LlmError::Unknown("something".to_string());
        let strategy = error.retry_strategy();
        match strategy {
            RetryStrategy::ExponentialBackoff { max_attempts, .. } => {
                assert_eq!(max_attempts, 2);
            }
            _ => panic!("Expected ExponentialBackoff for Unknown"),
        }
    }

    #[test]
    fn test_serialization_error_retry_strategy() {
        let json_err = serde_json::from_str::<serde_json::Value>("bad").unwrap_err();
        let error = LlmError::SerializationError(json_err);
        let strategy = error.retry_strategy();
        assert!(matches!(strategy, RetryStrategy::ExponentialBackoff { .. }));
    }

    #[test]
    fn test_api_error_non_5xx_retry_strategy() {
        let error = LlmError::ApiError("generic error".to_string());
        let strategy = error.retry_strategy();
        match strategy {
            RetryStrategy::ExponentialBackoff { max_attempts, .. } => {
                assert_eq!(max_attempts, 2);
            }
            _ => panic!("Expected ExponentialBackoff for generic ApiError"),
        }
    }

    #[test]
    fn test_config_error_no_retry() {
        let error = LlmError::ConfigError("bad config".to_string());
        assert!(matches!(error.retry_strategy(), RetryStrategy::NoRetry));
        assert!(!error.is_recoverable());
    }

    #[test]
    fn test_not_supported_no_retry() {
        let error = LlmError::NotSupported("embeddings".to_string());
        assert!(matches!(error.retry_strategy(), RetryStrategy::NoRetry));
        assert!(!error.is_recoverable());
    }

    // ========================================================================
    // RetryStrategy constructor verification
    // ========================================================================

    #[test]
    fn test_server_backoff_values() {
        let strategy = RetryStrategy::server_backoff();
        match strategy {
            RetryStrategy::ExponentialBackoff {
                base_delay,
                max_delay,
                max_attempts,
            } => {
                assert_eq!(base_delay, Duration::from_secs(1));
                assert_eq!(max_delay, Duration::from_secs(60));
                assert_eq!(max_attempts, 3);
            }
            _ => panic!("Expected ExponentialBackoff"),
        }
    }

    #[test]
    fn test_network_backoff_values() {
        let strategy = RetryStrategy::network_backoff();
        match strategy {
            RetryStrategy::ExponentialBackoff {
                base_delay,
                max_delay,
                max_attempts,
            } => {
                assert_eq!(base_delay, Duration::from_millis(125));
                assert_eq!(max_delay, Duration::from_secs(30));
                assert_eq!(max_attempts, 5);
            }
            _ => panic!("Expected ExponentialBackoff"),
        }
    }

    #[test]
    fn test_reduce_context_should_retry() {
        let strategy = RetryStrategy::ReduceContext;
        assert!(strategy.should_retry());
    }

    #[test]
    fn test_wait_and_retry_should_retry() {
        let strategy = RetryStrategy::WaitAndRetry {
            wait: Duration::from_secs(1),
        };
        assert!(strategy.should_retry());
    }

    // ========================================================================
    // is_recoverable coverage
    // ========================================================================

    #[test]
    fn test_is_recoverable_network() {
        assert!(LlmError::NetworkError("fail".to_string()).is_recoverable());
    }

    #[test]
    fn test_is_recoverable_timeout() {
        assert!(LlmError::Timeout.is_recoverable());
    }

    #[test]
    fn test_is_recoverable_rate_limited() {
        assert!(LlmError::RateLimited("wait".to_string()).is_recoverable());
    }

    #[test]
    fn test_is_not_recoverable_invalid_request() {
        assert!(!LlmError::InvalidRequest("bad".to_string()).is_recoverable());
    }

    #[test]
    fn test_is_not_recoverable_model_not_found() {
        assert!(!LlmError::ModelNotFound("x".to_string()).is_recoverable());
    }
}
