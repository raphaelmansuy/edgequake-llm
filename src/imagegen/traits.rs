//! Shared image generation trait.

use async_trait::async_trait;

use crate::imagegen::error::Result;
use crate::imagegen::types::{ImageGenRequest, ImageGenResponse};

/// Provider-agnostic image generation capability.
#[async_trait]
pub trait ImageGenProvider: Send + Sync {
    /// Stable provider identifier used in logs and metrics.
    fn name(&self) -> &str;

    /// Default model used by this provider.
    fn default_model(&self) -> &str;

    /// Generate one or more images from a request.
    async fn generate(&self, request: &ImageGenRequest) -> Result<ImageGenResponse>;

    /// Models exposed by this provider.
    fn available_models(&self) -> Vec<&str> {
        vec![self.default_model()]
    }
}
