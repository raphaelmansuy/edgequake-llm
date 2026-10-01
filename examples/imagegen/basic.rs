//! Basic image generation example.
//!
//! Demonstrates the provider-agnostic image generation pipeline:
//! 1. Auto-detect credentials from the environment
//! 2. Build a request with a text prompt
//! 3. Generate images and inspect the response
//!
//! Run with: cargo run --example imagegen_basic
//!
//! Set one of the following API keys to enable a provider:
//! - GEMINI_API_KEY      → Gemini Imagen
//! - GOOGLE_CLOUD_PROJECT → Vertex AI Imagen
//! - FAL_KEY             → FAL.ai
//! - OPENAI_API_KEY      → gpt-image-2.5-flare
//! - XAI_API_KEY         → grok-imagine-image-2.0
//! - AZURE_OPENAI_ENDPOINT → Azure OpenAI image deployment
//! - NVIDIA_API_KEY      → NVIDIA NIM

use edgequake_llm::{ImageGenData, ImageGenFactory, ImageGenOptions, ImageGenRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ImageGenFactory::from_env()?;

    println!("Provider : {}", provider.name());
    println!("Model    : {}", provider.default_model());
    println!("Available: {:?}", provider.available_models());
    println!();

    // --- Simple request ---------------------------------------------------
    let request = ImageGenRequest::new("A serene mountain landscape at sunset");
    let response = provider.generate(&request).await?;

    println!(
        "Generated {} image(s) in {}ms",
        response.images.len(),
        response.latency_ms
    );
    for (i, img) in response.images.iter().enumerate() {
        let size_info = match &img.data {
            ImageGenData::Bytes(bytes) => format!("{} bytes", bytes.len()),
            ImageGenData::Url(url) => format!("url: {url}"),
        };
        println!(
            "  Image {}: {}x{} {} ({})",
            i + 1,
            img.width,
            img.height,
            img.mime_type,
            size_info,
        );
    }
    if let Some(enhanced) = &response.enhanced_prompt {
        println!("  Enhanced prompt: {enhanced}");
    }
    println!();

    // --- Request with options ---------------------------------------------
    let options = ImageGenOptions {
        aspect_ratio: Some(edgequake_llm::AspectRatio::Landscape169),
        output_format: Some(edgequake_llm::ImageFormat::Png),
        ..Default::default()
    };
    let request =
        ImageGenRequest::new("Futuristic cityscape with neon lights").with_options(options);

    println!("Generating with options (16:9, PNG)...");
    let response = provider.generate(&request).await?;

    println!(
        "Generated {} image(s) in {}ms",
        response.images.len(),
        response.latency_ms
    );
    for (i, img) in response.images.iter().enumerate() {
        let size_info = match &img.data {
            ImageGenData::Bytes(bytes) => format!("{} bytes", bytes.len()),
            ImageGenData::Url(url) => format!("url: {url}"),
        };
        println!(
            "  Image {}: {}x{} {} ({})",
            i + 1,
            img.width,
            img.height,
            img.mime_type,
            size_info,
        );
    }

    Ok(())
}
