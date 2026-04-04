//! End-to-end tests for image generation providers.
//!
//! Run the Vertex AI Nano Banana test with:
//!
//! ```bash
//! cargo test --test e2e_imagegen test_vertexai_nano_banana_single_image -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use edgequake_llm::{
    AspectRatio, GeminiImageGenProvider, ImageGenData, ImageGenOptions, ImageGenProvider,
    ImageGenRequest, ImageResolution,
};

fn target_tmp_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp")
}

fn output_extension(mime_type: &str) -> &'static str {
    match mime_type {
        "image/png" => "png",
        "image/webp" => "webp",
        _ => "jpg",
    }
}

/// Cheapest Vertex AI Nano Banana model as of the current spec set.
///
/// WHY: This validates the real `generateContent` Vertex path for image generation
/// without using a more expensive preview/pro model.
#[tokio::test]
#[ignore = "Requires GOOGLE_CLOUD_PROJECT and Vertex AI credentials"]
async fn test_vertexai_nano_banana_single_image() {
    let provider = match GeminiImageGenProvider::from_env_vertex_ai() {
        Ok(provider) => provider.with_model(
            std::env::var("VERTEXAI_IMAGEGEN_MODEL")
                .unwrap_or_else(|_| "gemini-2.5-flash-image".to_string()),
        ),
        Err(err) => {
            eprintln!("Skipping Vertex AI imagegen test: {err}");
            return;
        }
    };

    let request = ImageGenRequest::new(
        "A minimalist black and white line drawing of a single banana on a plain white background.",
    )
    .with_options(ImageGenOptions {
        count: Some(1),
        aspect_ratio: Some(AspectRatio::Square),
        resolution: Some(ImageResolution::OneK),
        ..Default::default()
    });

    let response = provider.generate(&request).await.unwrap();
    assert_eq!(response.images.len(), 1, "expected exactly one image");

    let image = &response.images[0];
    let bytes = match &image.data {
        ImageGenData::Bytes(bytes) => bytes,
        ImageGenData::Url(url) => panic!("expected bytes from Vertex AI, got URL: {url}"),
    };

    assert!(
        !bytes.is_empty(),
        "expected non-empty image bytes from Vertex AI Nano Banana"
    );
    assert!(
        image.mime_type.starts_with("image/"),
        "unexpected mime type: {}",
        image.mime_type
    );

    std::fs::create_dir_all(target_tmp_dir()).unwrap();
    let output_path = target_tmp_dir().join(format!(
        "vertexai_nano_banana_e2e.{}",
        output_extension(&image.mime_type)
    ));
    std::fs::write(&output_path, bytes).unwrap();

    println!("Model: {}", response.model);
    println!("Provider: {}", response.provider);
    println!("Latency: {} ms", response.latency_ms);
    println!("Saved output: {}", output_path.display());
}
