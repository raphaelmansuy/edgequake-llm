//! Vision / Multimodal example — run: cargo run --example openai_vision
//! Requires: OPENAI_API_KEY  (uses gpt-4o-mini by default)
//!
//! Shows:
//!   1. URL image  (fastest — model fetches server-side, no upload)
//!   2. Base64 image (for private/local images)
//!   3. Multiple images in one message
//!   4. Detail level comparison (low vs high)
//!   5. Vision + JSON structured output (classification)

use base64::{engine::general_purpose::STANDARD, Engine as _};
use edgequake_llm::{ChatMessage, CompletionOptions, ImageData, LLMProvider, OpenAIProvider};

// Reliable public JPEGs — Azure sample dataset, no rate-limits, no content filters.
const LANDMARK_URL: &str =
    "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files\
/master/ComputerVision/Images/landmark.jpg";

const TEXT_IMG_URL: &str =
    "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files\
/master/ComputerVision/Images/printed_text.jpg";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    // gpt-4o-mini: vision-capable, cheap. Change to "gpt-4o" for higher quality.
    let provider = OpenAIProvider::new(&api_key).with_model("gpt-4o-mini");
    println!(
        "Provider: {} | Model: {}",
        provider.name(),
        provider.model()
    );
    println!("{}", "-".repeat(60));

    let short = CompletionOptions {
        max_tokens: Some(120),
        ..Default::default()
    };

    // ── 1. URL image ────────────────────────────────────────────────────────
    // The URL is passed directly; the model fetches the image server-side.
    println!("\n=== 1. URL image ===");
    {
        let img = ImageData::from_url(LANDMARK_URL).with_detail("low");
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
            "Tokens: {}p + {}c = {}",
            resp.prompt_tokens, resp.completion_tokens, resp.total_tokens
        );
    }

    // ── 2. Base64-encoded image ──────────────────────────────────────────────
    // Download the bytes, base64-encode them, pass inline.
    // Use this for private/local images that the API cannot fetch by URL.
    println!("\n=== 2. Base64-encoded image ===");
    {
        let bytes = reqwest::get(LANDMARK_URL).await?.bytes().await?;
        let b64 = STANDARD.encode(&bytes);
        let img = ImageData::new(b64, "image/jpeg").with_detail("low");
        let resp = provider
            .chat(
                &[
                    ChatMessage::system("You are a concise image analyst. One sentence max."),
                    ChatMessage::user_with_images(
                        "Describe this image in one sentence.",
                        vec![img],
                    ),
                ],
                Some(&short),
            )
            .await?;
        println!("Response: {}", resp.content);
    }

    // ── 3. Multiple images in one message ─────────────────────────────────
    println!("\n=== 3. Multiple images in one message ===");
    {
        let img1 = ImageData::from_url(LANDMARK_URL).with_detail("low");
        let img2 = ImageData::from_url(TEXT_IMG_URL).with_detail("low");
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

    // ── 4. Detail level comparison ─────────────────────────────────────────
    // low  ~  85 tokens/image — fast and cheap for general scenes
    // high ~ 170-1105 tokens  — needed for text, charts, fine detail
    println!("\n=== 4. Detail level comparison ===");
    println!("  low  ~ 85 tokens/image  — fast, general scene understanding");
    println!("  high ~ 170-1105 tokens  — required for text, charts, fine detail");
    {
        for detail in ["low", "high"] {
            let img = ImageData::from_url(TEXT_IMG_URL).with_detail(detail);
            let resp = provider
                .chat(
                    &[
                        ChatMessage::system("You are an OCR assistant."),
                        ChatMessage::user_with_images("What text do you see?", vec![img]),
                    ],
                    Some(&CompletionOptions {
                        max_tokens: Some(150),
                        ..Default::default()
                    }),
                )
                .await?;
            println!(
                "  [detail={}] {} | {}p+{}c tokens",
                detail,
                resp.content.lines().next().unwrap_or(""),
                resp.prompt_tokens,
                resp.completion_tokens
            );
        }
    }

    // ── 5. Vision + JSON structured output ────────────────────────────────
    // Combine a vision model with response_format=json_object for typed output.
    println!("\n=== 5. Vision + JSON structured classification ===");
    {
        let img = ImageData::from_url(LANDMARK_URL).with_detail("low");
        let resp = provider
            .chat(
                &[
                    ChatMessage::system(
                        concat!(
                            "You are an image classifier. ",
                            "Reply ONLY with valid JSON: ",
                            r#"{"category":"landmark|nature|person|text|other","confidence":"high|medium|low","description":"one sentence"}"#,
                        ),
                    ),
                    ChatMessage::user_with_images("Classify this image.", vec![img]),
                ],
                Some(&CompletionOptions {
                    max_tokens: Some(100),
                    response_format: Some("json_object".to_string()),
                    ..Default::default()
                }),
            )
            .await?;
        println!("Response: {}", resp.content);
    }

    println!("\n{}", "-".repeat(60));
    println!("Vision example complete!");
    Ok(())
}
