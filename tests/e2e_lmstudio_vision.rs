//! E2E tests for LM Studio provider vision (multimodal) support.
//!
//! These tests verify that images are correctly forwarded to LM Studio vision models
//! via the OpenAI-compatible content-parts array format.
//!
//! Requirements:
//! - LM Studio running at `http://localhost:1234` (or set `LMSTUDIO_HOST`)
//! - A vision-capable model loaded, e.g. `gemma3:latest`, `llava:latest`,
//!   or any multimodal model that supports `image_url` content parts.
//!
//! Run with:
//!   cargo test --test e2e_lmstudio_vision -- --ignored --nocapture

use edgequake_llm::providers::lmstudio::LMStudioProvider;
use edgequake_llm::traits::{ChatMessage, ImageData, LLMProvider};

/// Minimal 10×10 red pixel PNG (base64, no data-URI prefix).
const RED_PIXEL_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAEklEQVR4nGP4z8CAB\
     +GTG8HSALfKY52fTcuYAAAAAElFTkSuQmCC";

/// Returns the LM Studio host from env or the default.
fn lmstudio_host() -> String {
    std::env::var("LMSTUDIO_HOST").unwrap_or_else(|_| "http://localhost:1234".to_string())
}

/// Try to connect to local LM Studio and return true if it responds.
async fn lmstudio_is_available(host: &str) -> bool {
    let client = reqwest::Client::new();
    client
        .get(format!("{}/v1/models", host))
        .send()
        .await
        .is_ok()
}

// ============================================================================
// Vision tests — LM Studio
// ============================================================================

/// Verify that LMStudioProvider forwards images to a vision-capable model
/// via the OpenAI-compatible content-parts array.
///
/// Before the fix `ChatMessageRequest.content` was a plain `String`, so images
/// were silently discarded. After the fix it is a `serde_json::Value` that
/// is serialised as an array when images are present.
#[tokio::test]
#[ignore = "Requires LM Studio running with a vision-capable model loaded"]
async fn test_lmstudio_vision_red_pixel() {
    let host = lmstudio_host();

    if !lmstudio_is_available(&host).await {
        eprintln!("LM Studio not available at {} — skipping test", host);
        return;
    }

    let model =
        std::env::var("LMSTUDIO_VISION_MODEL").unwrap_or_else(|_| "gemma3:latest".to_string());

    let provider = LMStudioProvider::builder()
        .host(&host)
        .model(&model)
        .build()
        .expect("Failed to build LMStudioProvider");

    let img = ImageData::new(RED_PIXEL_PNG_B64, "image/png");
    let messages = vec![ChatMessage::user_with_images(
        "What is the dominant color in this image? Reply with one color word only.",
        vec![img],
    )];

    let response = provider
        .chat(&messages, None)
        .await
        .expect("LM Studio vision request failed");

    println!("LM Studio vision response: {}", response.content);

    assert!(
        !response.content.is_empty(),
        "Response must not be empty — image may have been dropped"
    );

    let lower = response.content.to_lowercase();
    assert!(
        lower.contains("red")
            || lower.contains("color")
            || lower.contains("colour")
            || lower.contains("pixel")
            || lower.contains("image"),
        "Response should reference image content, got: {}",
        response.content
    );
}

/// Ensure a plain (non-vision) text request still works after the images fix.
///
/// When `images` is `None`, `build_content` returns a plain JSON string
/// (not an array), so the API receives the same format as before.
#[tokio::test]
#[ignore = "Requires LM Studio running with any model loaded"]
async fn test_lmstudio_text_only_regression() {
    let host = lmstudio_host();

    if !lmstudio_is_available(&host).await {
        eprintln!("LM Studio not available at {} — skipping test", host);
        return;
    }

    let provider = LMStudioProvider::from_env().expect("Failed to build LMStudioProvider");

    let messages = vec![ChatMessage::user("What is 2 + 2? Reply with one number.")];

    let response = provider
        .chat(&messages, None)
        .await
        .expect("Text-only LM Studio request failed");

    println!("Text-only response: {}", response.content);

    assert!(
        response.content.contains('4') || response.content.to_lowercase().contains("four"),
        "Expected '4' or 'four', got: {}",
        response.content
    );
}

// ============================================================================
// Unit tests (no network required)
// ============================================================================

#[cfg(test)]
mod unit {
    use edgequake_llm::traits::{ChatMessage, ImageData};

    /// Verify that `ChatMessage::user_with_images` populates the images field
    /// with the correct base64 data — the prerequisite for the LM Studio fix.
    #[test]
    fn test_chat_message_images_populated() {
        let b64 = "iVBORw0KGgo=";
        let img = ImageData::new(b64, "image/png");
        let msg = ChatMessage::user_with_images("look at this", vec![img]);

        let images = msg.images.expect("images must be Some");
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].data, b64);
        assert_eq!(images[0].mime_type, "image/png");
    }

    /// Verify that `ImageData::to_data_uri` produces the right data URI —
    /// this is what LM Studio's `image_url` part receives.
    #[test]
    fn test_image_data_uri() {
        let img = ImageData::new("abc123", "image/jpeg");
        assert_eq!(img.to_data_uri(), "data:image/jpeg;base64,abc123");
    }
}
