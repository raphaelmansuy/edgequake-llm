//! Vision/Multimodal example
//!
//! Demonstrates image analysis using vision-capable LLM providers.
//!
//! Run with: cargo run --example vision
//! Requires: OPENAI_API_KEY environment variable (GPT-4V or newer)
//!
//! This example shows:
//! - Loading images as base64
//! - Creating multimodal messages with images
//! - Analyzing images with vision models
//! - Using detail level for quality/cost trade-offs

use base64::Engine;
use edgequake_llm::{ChatMessage, CompletionOptions, ImageData, LLMProvider, OpenAIProvider};
use std::fs;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    // Initialize provider - using gpt-4o which has vision capabilities
    let provider = OpenAIProvider::new(&api_key).with_model("gpt-4o");

    println!("👁️  EdgeQuake LLM - Vision/Multimodal Example\n");
    println!("Provider: {} | Model: {}", provider.name(), provider.model());
    println!("{}", "─".repeat(60));

    // Create a sample image (small colored rectangle for demonstration)
    let image_data = create_sample_image();
    
    println!("\n📷 Image: Sample colored rectangle (100x50 pixels)");
    println!("   Format: PNG | Size: {} bytes (base64)", image_data.len());

    // Create image with high detail for best analysis
    let image = ImageData::new(image_data, "image/png").with_detail("auto");

    // Create multimodal message
    let messages = vec![
        ChatMessage::system("You are a helpful image analysis assistant. Describe images concisely."),
        ChatMessage::user_with_images(
            "What do you see in this image? Describe the colors and shapes.",
            vec![image],
        ),
    ];

    println!("\n📤 Sending image to {} for analysis...\n", provider.model());

    // Options for the completion
    let options = CompletionOptions {
        max_tokens: Some(500),
        temperature: Some(0.3), // Lower temperature for more factual descriptions
        ..Default::default()
    };

    // Get analysis
    let response = provider.chat(&messages, Some(&options)).await?;

    println!("🔍 Analysis:\n");
    println!("{}", response.content);
    println!("\n{}", "─".repeat(60));
    println!(
        "📊 Tokens: {} prompt + {} completion = {} total",
        response.prompt_tokens, response.completion_tokens, response.total_tokens
    );

    // Demonstrate detail levels
    println!("\n💡 Detail Level Options:");
    println!("   • auto - Let model decide (default)");
    println!("   • low  - 512px max, faster, ~85 tokens per image");
    println!("   • high - 2048px max, detailed, ~170-1105 tokens");

    // Demonstrate loading from file (if exists)
    demonstrate_file_loading();

    Ok(())
}

/// Create a simple sample image (colored rectangle) as base64.
/// In real applications, you would load images from files or URLs.
fn create_sample_image() -> String {
    // Create a minimal PNG with colored pixels
    // This is a simple 4x4 pixel PNG for demonstration
    // In practice, use the image crate or load from files
    
    // Pre-encoded small PNG (4x4 red square)
    // This avoids external dependencies for the example
    let png_bytes: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
        0x00, 0x00, 0x00, 0x0D, // IHDR length
        0x49, 0x48, 0x44, 0x52, // IHDR
        0x00, 0x00, 0x00, 0x04, // width: 4
        0x00, 0x00, 0x00, 0x04, // height: 4
        0x08, 0x02, // bit depth: 8, color type: RGB
        0x00, 0x00, 0x00, // compression, filter, interlace
        0x26, 0x93, 0x09, 0x29, // CRC
        0x00, 0x00, 0x00, 0x1C, // IDAT length
        0x49, 0x44, 0x41, 0x54, // IDAT
        0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0xC1, // compressed data
        0xFF, 0x19, 0x18, 0x00, 0x00, 0x18, 0x60, 0x60,
        0x60, 0x00, 0x00, 0x00, 0x31, 0x00, 0x01,
        0xAA, 0x5A, 0xB6, 0x47, // CRC
        0x00, 0x00, 0x00, 0x00, // IEND length
        0x49, 0x45, 0x4E, 0x44, // IEND
        0xAE, 0x42, 0x60, 0x82, // CRC
    ];
    
    base64::engine::general_purpose::STANDARD.encode(png_bytes)
}

/// Demonstrate loading an image from a file path.
fn demonstrate_file_loading() {
    println!("\n📁 Loading Images from Files:");
    println!("```rust");
    println!("use std::fs;");
    println!("use base64::Engine;");
    println!("");
    println!("fn load_image_from_file(path: &str) -> Result<ImageData, std::io::Error> {{");
    println!("    let bytes = fs::read(path)?;");
    println!("    let base64 = base64::engine::general_purpose::STANDARD.encode(&bytes);");
    println!("    ");
    println!("    // Detect MIME type from extension");
    println!("    let mime_type = match Path::new(path).extension().and_then(|e| e.to_str()) {{");
    println!("        Some(\"png\") => \"image/png\",");
    println!("        Some(\"jpg\") | Some(\"jpeg\") => \"image/jpeg\",");
    println!("        Some(\"gif\") => \"image/gif\",");
    println!("        Some(\"webp\") => \"image/webp\",");
    println!("        _ => \"image/png\", // default");
    println!("    }};");
    println!("    ");
    println!("    Ok(ImageData::new(base64, mime_type))");
    println!("}}");
    println!("```");
}

/// Load an image from a file path (utility function).
#[allow(dead_code)]
fn load_image_from_file(path: &str) -> Result<ImageData, std::io::Error> {
    let bytes = fs::read(path)?;
    let base64_data = base64::engine::general_purpose::STANDARD.encode(&bytes);
    
    // Detect MIME type from extension
    let mime_type = match Path::new(path).extension().and_then(|e| e.to_str()) {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        _ => "image/png",
    };
    
    Ok(ImageData::new(base64_data, mime_type))
}
