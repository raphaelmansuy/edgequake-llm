//! Factory helpers for image generation providers.

use std::sync::Arc;

use crate::imagegen::error::{ImageGenError, Result};
use crate::imagegen::providers::{
    FalImageGen, GeminiImageGenProvider, MockImageGenProvider, VertexAIImageGen,
};
use crate::imagegen::traits::ImageGenProvider;

/// Factory for constructing image generation providers.
pub struct ImageGenFactory;

impl ImageGenFactory {
    /// Create the most appropriate provider from the current environment.
    pub fn from_env() -> Result<Arc<dyn ImageGenProvider>> {
        if std::env::var("GEMINI_API_KEY").is_ok() {
            return Ok(Arc::new(GeminiImageGenProvider::from_env()?));
        }

        if std::env::var("GOOGLE_CLOUD_PROJECT").is_ok() {
            return Ok(Arc::new(GeminiImageGenProvider::from_env_vertex_ai()?));
        }

        if std::env::var("FAL_KEY").is_ok() {
            return Ok(Arc::new(FalImageGen::from_env()?));
        }

        Err(ImageGenError::ConfigError(
            "no image generation credentials found; set GEMINI_API_KEY, GOOGLE_CLOUD_PROJECT, or FAL_KEY".to_string(),
        ))
    }

    /// Create a Gemini image provider.
    pub fn gemini_from_env() -> Result<GeminiImageGenProvider> {
        GeminiImageGenProvider::from_env()
    }

    /// Create a Gemini image provider forced to Vertex AI backend.
    pub fn gemini_vertex_from_env() -> Result<GeminiImageGenProvider> {
        GeminiImageGenProvider::from_env_vertex_ai()
    }

    /// Create a Vertex Imagen provider.
    pub fn vertex_imagen_from_env() -> Result<VertexAIImageGen> {
        VertexAIImageGen::from_env()
    }

    /// Create a FAL provider.
    pub fn fal_from_env() -> Result<FalImageGen> {
        FalImageGen::from_env()
    }

    /// Create a mock provider.
    pub fn mock() -> MockImageGenProvider {
        MockImageGenProvider::default()
    }
}
