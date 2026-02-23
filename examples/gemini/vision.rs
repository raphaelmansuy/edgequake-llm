//! Gemini vision / multimodal example
//!
//! Run: cargo run --example gemini_vision
//! Requires: GEMINI_API_KEY
//!
//! Shows:
//!   1. Base64-encoded image analysis
//!   2. Multiple images in one message
//!   3. Vision + JSON structured output (classification)
//!   4. Image comparison

use base64::{engine::general_purpose::STANDARD, Engine as _};
use edgequake_llm::providers::gemini::GeminiProvider;
use edgequake_llm::traits::{ChatMessage, CompletionOptions, ImageData, LLMProvider};

// Reliable public JPEGs — Azure sample dataset
const LANDMARK_URL: &str =
    "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files\
/master/ComputerVision/Images/landmark.jpg";

const TEXT_IMG_URL: &str =
    "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files\
/master/ComputerVision/Images/printed_text.jpg";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = GeminiProvider::from_env()?;
    println!(
        "Provider: {} | Model: {}",
        LLMProvider::name(&provider),
        LLMProvider::model(&provider),
    );
    println!("{}", "-".repeat(60));

    let short = CompletionOptions {
        max_tokens: Some(200),
        ..Default::default()
    };

    // ── 1. Base64-encoded image ──────────────────────────────────────────
    println!("\n=== 1. Base64-encoded image ===");
    {
        // Download image and encode as base64
        let bytes = reqwest::get(LANDMARK_URL).await?.bytes().await?;
        let b64 = STANDARD.encode(&bytes);
        let img = ImageData::new(b64, "image/jpeg");

        let resp = provider
            .chat(
                &[
                    ChatMessage::system("You are a concise image analyst. One sentence max."),
                    ChatMessage::user_with_images("What landmark is shown?", vec![img]),
                ],
                Some(&short),
            )
            .await?;
        println!("Response: {}", resp.content);
        println!(
            "Tokens: {}p + {}c = {}t",
            resp.prompt_tokens, resp.completion_tokens, resp.total_tokens
        );
    }

    // ── 2. Multiple images in one message ─────────────────────────────────
    println!("\n=== 2. Multiple images ===");
    {
        let bytes1 = reqwest::get(LANDMARK_URL).await?.bytes().await?;
        let bytes2 = reqwest::get(TEXT_IMG_URL).await?.bytes().await?;

        let img1 = ImageData::new(STANDARD.encode(&bytes1), "image/jpeg");
        let img2 = ImageData::new(STANDARD.encode(&bytes2), "image/jpeg");

        let resp = provider
            .chat(
                &[
                    ChatMessage::system("You are a concise image analyst."),
                    ChatMessage::user_with_images(
                        "Two images follow. In one sentence each, what does each show?",
                        vec![img1, img2],
                    ),
                ],
                Some(&short),
            )
            .await?;
        println!("Response: {}", resp.content);
    }

    // ── 3. Vision + JSON structured output ─────────────────────────────────
    println!("\n=== 3. Vision + JSON classification ===");
    {
        let bytes = reqwest::get(LANDMARK_URL).await?.bytes().await?;
        let b64 = STANDARD.encode(&bytes);
        let img = ImageData::new(b64, "image/jpeg");

        let json_opts = CompletionOptions {
            max_tokens: Some(300),
            response_format: Some("json_object".to_string()),
            ..Default::default()
        };

        let resp = provider
            .chat(
                &[
                    ChatMessage::system(
                        "You classify images. Return JSON with: \
                         { \"category\": string, \"subject\": string, \"confidence\": number (0-1) }",
                    ),
                    ChatMessage::user_with_images("Classify this image.", vec![img]),
                ],
                Some(&json_opts),
            )
            .await?;
        println!("JSON classification: {}", resp.content);

        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&resp.content) {
            println!(
                "  Category: {}, Subject: {}, Confidence: {}",
                parsed["category"], parsed["subject"], parsed["confidence"]
            );
        }
    }

    // ── 4. Inline 1x1 PNG for quick testing ──────────────────────────────
    println!("\n=== 4. Inline tiny PNG (color detection) ===");
    {
        // Minimal 1x1 red-ish pixel PNG
        let tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
        let img = ImageData::new(tiny_png, "image/png");

        let resp = provider
            .chat(
                &[ChatMessage::user_with_images(
                    "What color is this single-pixel image? Just say the color name.",
                    vec![img],
                )],
                Some(&short),
            )
            .await?;
        println!("Color: {}", resp.content);
    }

    println!("\n=== Done! Vision features demonstrated. ===");
    Ok(())
}
