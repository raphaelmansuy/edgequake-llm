use async_trait::async_trait;
use base64::Engine;

use crate::imagegen::error::Result;
use crate::imagegen::traits::ImageGenProvider;
use crate::imagegen::types::{GeneratedImage, ImageGenData, ImageGenRequest, ImageGenResponse};

const ONE_BY_ONE_PNG: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+XcZ0AAAAASUVORK5CYII=";

/// In-memory provider used by tests.
#[derive(Debug, Clone, Default)]
pub struct MockImageGenProvider {
    model: String,
}

#[async_trait]
impl ImageGenProvider for MockImageGenProvider {
    fn name(&self) -> &str {
        "mock-imagegen"
    }

    fn default_model(&self) -> &str {
        if self.model.is_empty() {
            "mock-image-model"
        } else {
            &self.model
        }
    }

    async fn generate(&self, request: &ImageGenRequest) -> Result<ImageGenResponse> {
        let bytes = base64::engine::general_purpose::STANDARD.decode(ONE_BY_ONE_PNG)?;
        Ok(ImageGenResponse {
            images: vec![GeneratedImage {
                data: ImageGenData::Bytes(bytes),
                width: 1,
                height: 1,
                mime_type: "image/png".to_string(),
                seed: request.options.seed,
            }],
            provider: self.name().to_string(),
            model: request
                .model
                .clone()
                .unwrap_or_else(|| self.default_model().to_string()),
            latency_ms: 0,
            enhanced_prompt: Some(request.prompt.clone()),
        })
    }
}
