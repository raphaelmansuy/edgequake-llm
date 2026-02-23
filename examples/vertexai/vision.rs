//! VertexAI vision examples — multimodal image understanding.
//!
//! Run: cargo run --example vertexai_vision
//! Requires:
//!   - GOOGLE_CLOUD_PROJECT
//!   - Authenticated via `gcloud auth login` (or GOOGLE_ACCESS_TOKEN)

use edgequake_llm::providers::gemini::GeminiProvider;
use edgequake_llm::traits::{ChatMessage, CompletionOptions, LLMProvider};
use edgequake_llm::ImageData;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = GeminiProvider::from_env_vertex_ai()?;
    println!(
        "VertexAI Vision Examples — model: {}\n",
        LLMProvider::model(&provider)
    );

    // ────────────────────────────────────────────────────────────── 1 ──
    // Basic vision — describe a tiny image
    // ────────────────────────────────────────────────────────────── 1 ──
    println!("=== 1. Basic image description ===");

    // Minimal 1x1 red pixel PNG (base64)
    let red_pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";
    let image = ImageData::new(red_pixel, "image/png");
    let messages = vec![ChatMessage::user_with_images(
        "What color is this image? Reply with just the color name.",
        vec![image],
    )];
    let resp = provider.chat(&messages, None).await?;
    println!("Single pixel: {}\n", resp.content);

    // ────────────────────────────────────────────────────────────── 2 ──
    // Multiple images — compare and contrast
    // ────────────────────────────────────────────────────────────── 2 ──
    println!("=== 2. Multiple images comparison ===");

    // 1x1 blue pixel
    let blue_pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPj/HwADBwIAMCbHYQAAAABJRU5ErkJggg==";
    let img_red = ImageData::new(red_pixel, "image/png");
    let img_blue = ImageData::new(blue_pixel, "image/png");

    let messages = vec![ChatMessage::user_with_images(
        "I'm showing you two images. Describe the main difference between them. Be brief.",
        vec![img_red, img_blue],
    )];
    let resp = provider.chat(&messages, None).await?;
    println!("Comparison: {}\n", resp.content);

    // ────────────────────────────────────────────────────────────── 3 ──
    // Vision + JSON structured output
    // ────────────────────────────────────────────────────────────── 3 ──
    println!("=== 3. Vision + JSON structured output ===");

    let image = ImageData::new(red_pixel, "image/png");
    let messages = vec![ChatMessage::user_with_images(
        "Analyze this image. Return JSON with keys: dominant_color, width_guess, height_guess, description.",
        vec![image],
    )];
    let json_opts = CompletionOptions::json_mode();
    let resp = provider.chat(&messages, Some(&json_opts)).await?;
    println!("JSON: {}\n", resp.content);

    // ────────────────────────────────────────────────────────────── 4 ──
    // Vision with system prompt
    // ────────────────────────────────────────────────────────────── 4 ──
    println!("=== 4. Vision with system prompt ===");

    let image = ImageData::new(red_pixel, "image/png");
    let messages = vec![
        ChatMessage::system(
            "You are an art critic. Describe images using flowery, poetic language.",
        ),
        ChatMessage::user_with_images("Describe this artwork.", vec![image]),
    ];
    let resp = provider
        .chat(
            &messages,
            Some(&CompletionOptions {
                max_tokens: Some(200),
                ..Default::default()
            }),
        )
        .await?;
    println!("Art critic: {}\n", resp.content);

    // ────────────────────────────────────────────────────────────── 5 ──
    // JPEG image (inline tiny JPEG)
    // ────────────────────────────────────────────────────────────── 5 ──
    println!("=== 5. JPEG image ===");

    // Minimal 1x1 white JPEG
    let white_jpeg = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8AKwA//9k=";
    let image = ImageData::new(white_jpeg, "image/jpeg");
    let messages = vec![ChatMessage::user_with_images(
        "What can you tell me about this image?",
        vec![image],
    )];
    let resp = provider.chat(&messages, None).await?;
    println!("JPEG: {}\n", resp.content);

    // ────────────────────────────────────────────────────────────── 6 ──
    // Vision with custom model (flash for faster inference)
    // ────────────────────────────────────────────────────────────── 6 ──
    println!("=== 6. Vision with explicit model ===");

    let flash_provider = GeminiProvider::from_env_vertex_ai()?.with_model("gemini-2.5-flash");
    let image = ImageData::new(red_pixel, "image/png");
    let messages = vec![ChatMessage::user_with_images(
        "Describe this image in exactly 5 words.",
        vec![image],
    )];
    let resp = flash_provider
        .chat(
            &messages,
            Some(&CompletionOptions {
                max_tokens: Some(50),
                ..Default::default()
            }),
        )
        .await?;
    println!(
        "Flash model ({}): {}\n",
        LLMProvider::model(&flash_provider),
        resp.content
    );

    println!("=== Done! ===");
    Ok(())
}
