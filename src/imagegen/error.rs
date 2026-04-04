//! Errors for image generation providers.

use std::time::Duration;

use thiserror::Error;

/// Result type for image generation operations.
pub type Result<T> = std::result::Result<T, ImageGenError>;

/// Errors returned by image generation providers.
#[derive(Debug, Error)]
pub enum ImageGenError {
    #[error("authentication error: {0}")]
    AuthError(String),

    #[error("configuration error: {0}")]
    ConfigError(String),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("invalid response: {0}")]
    InvalidResponse(String),

    #[error("provider error: {0}")]
    ProviderError(String),

    #[error("network error: {0}")]
    NetworkError(String),

    #[error("content filtered: {reason}")]
    ContentFiltered { reason: String },

    #[error("rate limited")]
    RateLimited { retry_after: Option<Duration> },

    #[error("request timed out after {elapsed_secs}s")]
    Timeout { elapsed_secs: u64 },

    #[error("not supported: {0}")]
    NotSupported(String),

    #[error("serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

impl From<reqwest::Error> for ImageGenError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            Self::Timeout { elapsed_secs: 0 }
        } else if err.is_connect() {
            Self::NetworkError(format!("connection failed: {err}"))
        } else {
            Self::NetworkError(err.to_string())
        }
    }
}

impl From<base64::DecodeError> for ImageGenError {
    fn from(err: base64::DecodeError) -> Self {
        Self::InvalidResponse(format!("invalid base64 image payload: {err}"))
    }
}
