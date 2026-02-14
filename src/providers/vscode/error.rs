//! Error types for VSCode Copilot provider.
//!
//! # Design Rationale
//!
//! WHY: We use a dedicated `VsCodeError` type rather than the generic `LlmError` because:
//! 1. VSCode/Copilot has specific error conditions (proxy unavailable, rate limiting)
//! 2. Error messages can include actionable hints (e.g., "Is copilot-api running?")
//! 3. Conversion to `LlmError` is automatic via `From` trait
//!
//! # Error Categories
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    VsCodeError Types                             │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                   │
//! │  Initialization Errors                                           │
//! │  ├── ClientInit      → HTTP client TLS/config issues            │
//! │  └── ProxyUnavailable → Proxy server not running                │
//! │                                                                   │
//! │  Runtime Errors                                                   │
//! │  ├── Network         → DNS, timeout, connection refused         │
//! │  ├── Authentication  → Invalid/expired token                    │
//! │  ├── RateLimited     → 429 Too Many Requests                    │
//! │  ├── InvalidRequest  → 400 Bad Request                          │
//! │  └── ServiceUnavailable → 503 Service Unavailable               │
//! │                                                                   │
//! │  Response Errors                                                  │
//! │  ├── ApiError        → Generic API error (500, etc.)            │
//! │  ├── Decode          → JSON deserialization failed              │
//! │  └── Stream          → SSE parsing error                        │
//! │                                                                   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Error Recovery
//!
//! | Error | Retryable | Action |
//! |-------|-----------|--------|
//! | `RateLimited` | Yes | Wait with exponential backoff |
//! | `Network` | Yes | Retry after delay |
//! | `ServiceUnavailable` | Yes | Retry after delay |
//! | `Authentication` | No | Re-authenticate |
//! | `InvalidRequest` | No | Fix request parameters |
//! | `ClientInit` | No | Fix configuration |

use thiserror::Error;

pub type Result<T> = std::result::Result<T, VsCodeError>;

/// VSCode Copilot provider errors.
#[derive(Error, Debug)]
pub enum VsCodeError {
    /// Failed to initialize HTTP client.
    #[error("Failed to initialize client: {0}")]
    ClientInit(String),

    /// Proxy server is unavailable or not responding.
    #[error("Proxy unavailable: {0}. Is copilot-api running on localhost:4141?")]
    ProxyUnavailable(String),

    /// Network communication error.
    #[error("Network error: {0}")]
    Network(String),

    /// Authentication or authorization failed.
    #[error("Authentication failed: {0}")]
    Authentication(String),

    /// Rate limit exceeded.
    #[error("Rate limited. Try again later.")]
    RateLimited,

    /// Invalid request format or parameters.
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Service temporarily unavailable.
    #[error("Service unavailable")]
    ServiceUnavailable,

    /// Generic API error.
    #[error("API error: {0}")]
    ApiError(String),

    /// Failed to decode response.
    #[error("Failed to decode response: {0}")]
    Decode(String),

    /// Streaming error.
    #[error("Stream error: {0}")]
    Stream(String),
}

impl VsCodeError {
    /// Returns true if this error is retryable.
    ///
    /// # WHY
    ///
    /// Consumers of this API need to know which errors warrant retry attempts
    /// versus which errors indicate permanent failures. This method encapsulates
    /// that knowledge so callers don't need to match on error variants.
    ///
    /// # Retryable Errors
    ///
    /// - `Network`: Temporary connectivity issues (DNS, timeout, connection refused)
    /// - `RateLimited`: 429 response - will succeed after backoff
    /// - `ServiceUnavailable`: 503 response - server temporarily down
    ///
    /// # Non-Retryable Errors
    ///
    /// - `ClientInit`: Configuration issue - won't resolve with retry
    /// - `ProxyUnavailable`: Proxy needs to be started
    /// - `Authentication`: Token invalid - need new token
    /// - `InvalidRequest`: Request parameters wrong - fix before retry
    /// - `ApiError`: Permanent server-side failure
    /// - `Decode`: Response format issue - server bug or version mismatch
    /// - `Stream`: SSE parsing error - unlikely to resolve
    ///
    /// # Example
    ///
    /// ```rust
    /// use edgequake_llm::providers::vscode::VsCodeError;
    ///
    /// let err = VsCodeError::RateLimited;
    /// if err.is_retryable() {
    ///     // Apply exponential backoff and retry
    /// }
    /// ```
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            VsCodeError::Network(_) | VsCodeError::RateLimited | VsCodeError::ServiceUnavailable
        )
    }
}

// Convert VsCodeError to LlmError
impl From<VsCodeError> for crate::error::LlmError {
    fn from(err: VsCodeError) -> Self {
        match err {
            VsCodeError::ClientInit(msg) => Self::ConfigError(msg),
            VsCodeError::ProxyUnavailable(msg) => Self::NetworkError(msg),
            VsCodeError::Network(msg) => Self::NetworkError(msg),
            VsCodeError::Authentication(msg) => Self::AuthError(msg),
            VsCodeError::RateLimited => Self::RateLimited("Rate limit exceeded".to_string()),
            VsCodeError::InvalidRequest(msg) => Self::InvalidRequest(msg),
            VsCodeError::ServiceUnavailable => {
                Self::NetworkError("Service unavailable".to_string())
            }
            VsCodeError::ApiError(msg) => Self::ApiError(msg),
            VsCodeError::Decode(msg) => Self::ApiError(format!("Decode: {}", msg)),
            VsCodeError::Stream(msg) => Self::ApiError(format!("Stream: {}", msg)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LlmError;

    // ========================================================================
    // Display Trait Tests - Verify Error Messages
    // ========================================================================

    #[test]
    fn test_vscode_error_display_client_init() {
        let err = VsCodeError::ClientInit("TLS handshake failed".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Failed to initialize client"));
        assert!(msg.contains("TLS handshake failed"));
    }

    #[test]
    fn test_vscode_error_display_proxy_unavailable() {
        let err = VsCodeError::ProxyUnavailable("connection refused".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Proxy unavailable"));
        assert!(msg.contains("connection refused"));
        assert!(msg.contains("localhost:4141")); // Helpful hint
    }

    #[test]
    fn test_vscode_error_display_network() {
        let err = VsCodeError::Network("timeout after 30s".to_string());
        assert_eq!(err.to_string(), "Network error: timeout after 30s");
    }

    #[test]
    fn test_vscode_error_display_authentication() {
        let err = VsCodeError::Authentication("token expired".to_string());
        assert_eq!(err.to_string(), "Authentication failed: token expired");
    }

    #[test]
    fn test_vscode_error_display_rate_limited() {
        let err = VsCodeError::RateLimited;
        assert_eq!(err.to_string(), "Rate limited. Try again later.");
    }

    #[test]
    fn test_vscode_error_display_service_unavailable() {
        let err = VsCodeError::ServiceUnavailable;
        assert_eq!(err.to_string(), "Service unavailable");
    }

    // ========================================================================
    // From<VsCodeError> for LlmError Conversion Tests
    // ========================================================================

    #[test]
    fn test_conversion_client_init_to_config_error() {
        let vscode_err = VsCodeError::ClientInit("init failed".to_string());
        let llm_err: LlmError = vscode_err.into();

        match llm_err {
            LlmError::ConfigError(msg) => assert_eq!(msg, "init failed"),
            other => panic!("Expected ConfigError, got {:?}", other),
        }
    }

    #[test]
    fn test_conversion_proxy_unavailable_to_network_error() {
        let vscode_err = VsCodeError::ProxyUnavailable("refused".to_string());
        let llm_err: LlmError = vscode_err.into();

        match llm_err {
            LlmError::NetworkError(msg) => assert_eq!(msg, "refused"),
            other => panic!("Expected NetworkError, got {:?}", other),
        }
    }

    #[test]
    fn test_conversion_network_to_network_error() {
        let vscode_err = VsCodeError::Network("dns lookup failed".to_string());
        let llm_err: LlmError = vscode_err.into();

        match llm_err {
            LlmError::NetworkError(msg) => assert_eq!(msg, "dns lookup failed"),
            other => panic!("Expected NetworkError, got {:?}", other),
        }
    }

    #[test]
    fn test_conversion_authentication_to_auth_error() {
        let vscode_err = VsCodeError::Authentication("invalid token".to_string());
        let llm_err: LlmError = vscode_err.into();

        match llm_err {
            LlmError::AuthError(msg) => assert_eq!(msg, "invalid token"),
            other => panic!("Expected AuthError, got {:?}", other),
        }
    }

    #[test]
    fn test_conversion_rate_limited() {
        let vscode_err = VsCodeError::RateLimited;
        let llm_err: LlmError = vscode_err.into();

        match llm_err {
            LlmError::RateLimited(msg) => assert!(msg.contains("Rate limit")),
            other => panic!("Expected RateLimited, got {:?}", other),
        }
    }

    #[test]
    fn test_conversion_invalid_request() {
        let vscode_err = VsCodeError::InvalidRequest("missing model".to_string());
        let llm_err: LlmError = vscode_err.into();

        match llm_err {
            LlmError::InvalidRequest(msg) => assert_eq!(msg, "missing model"),
            other => panic!("Expected InvalidRequest, got {:?}", other),
        }
    }

    #[test]
    fn test_conversion_service_unavailable() {
        let vscode_err = VsCodeError::ServiceUnavailable;
        let llm_err: LlmError = vscode_err.into();

        match llm_err {
            LlmError::NetworkError(msg) => assert!(msg.contains("unavailable")),
            other => panic!("Expected NetworkError, got {:?}", other),
        }
    }

    #[test]
    fn test_conversion_api_error() {
        let vscode_err = VsCodeError::ApiError("internal server error".to_string());
        let llm_err: LlmError = vscode_err.into();

        match llm_err {
            LlmError::ApiError(msg) => assert_eq!(msg, "internal server error"),
            other => panic!("Expected ApiError, got {:?}", other),
        }
    }

    #[test]
    fn test_conversion_decode_error() {
        let vscode_err = VsCodeError::Decode("invalid JSON".to_string());
        let llm_err: LlmError = vscode_err.into();

        match llm_err {
            LlmError::ApiError(msg) => {
                assert!(msg.contains("Decode"));
                assert!(msg.contains("invalid JSON"));
            }
            other => panic!("Expected ApiError, got {:?}", other),
        }
    }

    #[test]
    fn test_conversion_stream_error() {
        let vscode_err = VsCodeError::Stream("connection reset".to_string());
        let llm_err: LlmError = vscode_err.into();

        match llm_err {
            LlmError::ApiError(msg) => {
                assert!(msg.contains("Stream"));
                assert!(msg.contains("connection reset"));
            }
            other => panic!("Expected ApiError, got {:?}", other),
        }
    }

    // ========================================================================
    // is_retryable() Tests
    // WHY: Verify correct categorization of retryable vs non-retryable errors
    // ========================================================================

    #[test]
    fn test_is_retryable_network_error() {
        // WHY: Network errors are temporary and should be retried
        let err = VsCodeError::Network("connection timeout".to_string());
        assert!(err.is_retryable(), "Network errors should be retryable");
    }

    #[test]
    fn test_is_retryable_rate_limited() {
        // WHY: 429 response means we should wait and retry
        let err = VsCodeError::RateLimited;
        assert!(
            err.is_retryable(),
            "Rate limited errors should be retryable"
        );
    }

    #[test]
    fn test_is_retryable_service_unavailable() {
        // WHY: 503 means server is temporarily down
        let err = VsCodeError::ServiceUnavailable;
        assert!(
            err.is_retryable(),
            "Service unavailable should be retryable"
        );
    }

    #[test]
    fn test_is_not_retryable_auth_error() {
        // WHY: Auth errors need new credentials, not retry
        let err = VsCodeError::Authentication("token expired".to_string());
        assert!(!err.is_retryable(), "Auth errors should not be retryable");
    }

    #[test]
    fn test_is_not_retryable_invalid_request() {
        // WHY: Invalid request needs to be fixed, not retried
        let err = VsCodeError::InvalidRequest("missing model".to_string());
        assert!(
            !err.is_retryable(),
            "Invalid request should not be retryable"
        );
    }

    #[test]
    fn test_is_not_retryable_client_init() {
        // WHY: Client init errors are configuration issues
        let err = VsCodeError::ClientInit("TLS failed".to_string());
        assert!(
            !err.is_retryable(),
            "Client init errors should not be retryable"
        );
    }

    #[test]
    fn test_is_not_retryable_api_error() {
        // WHY: Generic API errors (500) are typically permanent
        let err = VsCodeError::ApiError("internal error".to_string());
        assert!(!err.is_retryable(), "API errors should not be retryable");
    }

    #[test]
    fn test_is_not_retryable_decode_error() {
        // WHY: Decode errors indicate server response format issues
        let err = VsCodeError::Decode("invalid JSON".to_string());
        assert!(!err.is_retryable(), "Decode errors should not be retryable");
    }
}
