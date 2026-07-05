//! Factory helpers for image generation providers.

use std::sync::Arc;

use crate::imagegen::error::{ImageGenError, Result};
use crate::imagegen::providers::{
    AzureImageGen, FalImageGen, GeminiImageGenProvider, MockImageGenProvider, NvidiaImageGen,
    OpenAIImageGen, VertexAIImageGen, XAIImageGen,
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

        if std::env::var("OPENAI_API_KEY").is_ok() {
            return Ok(Arc::new(OpenAIImageGen::from_env()?));
        }

        if std::env::var("XAI_API_KEY").is_ok() {
            return Ok(Arc::new(XAIImageGen::from_env()?));
        }

        if std::env::var("AZURE_OPENAI_ENDPOINT").is_ok()
            || std::env::var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT").is_ok()
        {
            return Ok(Arc::new(AzureImageGen::from_env()?));
        }

        if std::env::var("NVIDIA_API_KEY").is_ok() {
            return Ok(Arc::new(NvidiaImageGen::from_env()?));
        }

        Err(ImageGenError::ConfigError(
            "no image generation credentials found; set GEMINI_API_KEY, GOOGLE_CLOUD_PROJECT, FAL_KEY, OPENAI_API_KEY, XAI_API_KEY, AZURE_OPENAI_ENDPOINT, or NVIDIA_API_KEY".to_string(),
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

    /// Create an OpenAI image provider.
    pub fn openai_from_env() -> Result<OpenAIImageGen> {
        OpenAIImageGen::from_env()
    }

    /// Create an xAI Grok Imagine provider.
    pub fn xai_from_env() -> Result<XAIImageGen> {
        XAIImageGen::from_env()
    }

    /// Create an Azure OpenAI image provider.
    pub fn azure_from_env() -> Result<AzureImageGen> {
        AzureImageGen::from_env()
    }

    /// Create an NVIDIA NIM image provider.
    pub fn nvidia_from_env() -> Result<NvidiaImageGen> {
        NvidiaImageGen::from_env()
    }

    /// Create a mock provider.
    pub fn mock() -> MockImageGenProvider {
        MockImageGenProvider::default()
    }

    /// List canonical image generation provider IDs.
    pub fn list_providers() -> Vec<&'static str> {
        crate::provider_catalog::ProviderCatalog::list_imagegen_providers()
    }
}
