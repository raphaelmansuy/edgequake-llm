//! OpenAI image generation provider.
//!
//! Default is `gpt-image-2.5-flare` (everyday quality). `gpt-image-2.5-sunburst`
//! is the higher-quality edit model. `gpt-image-2` remains available.

use std::time::Instant;

use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use tracing::{debug, warn};

use crate::imagegen::error::{ImageGenError, Result};
use crate::imagegen::traits::ImageGenProvider;
use crate::imagegen::types::{
    AspectRatio, GeneratedImage, ImageFormat, ImageGenData, ImageGenRequest, ImageGenResponse,
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "gpt-image-2.5-flare";

/// OpenAI image generation provider using GPT Image 2.5.
#[derive(Debug, Clone)]
pub struct OpenAIImageGen {
    api_key: String,
    base_url: String,
    http_client: Client,
}

#[derive(Debug, Deserialize)]
struct OpenAIImageResponse {
    data: Vec<OpenAIImageItem>,
}

#[derive(Debug, Deserialize)]
struct OpenAIImageItem {
    #[serde(default)]
    b64_json: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    revised_prompt: Option<String>,
}

impl OpenAIImageGen {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            http_client: Client::new(),
        }
    }

    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| ImageGenError::ConfigError("OPENAI_API_KEY must be set".to_string()))?;
        Ok(Self::new(api_key))
    }

    fn active_model<'a>(&'a self, request: &'a ImageGenRequest) -> &'a str {
        request.model.as_deref().unwrap_or(DEFAULT_MODEL)
    }

    fn endpoint_url(&self) -> String {
        format!("{}/images/generations", self.base_url)
    }

    fn map_aspect_ratio_to_size(ratio: AspectRatio) -> &'static str {
        match ratio {
            AspectRatio::Auto | AspectRatio::Square | AspectRatio::SquareHd => "1024x1024",
            AspectRatio::Landscape169
            | AspectRatio::Ultrawide
            | AspectRatio::Extreme41
            | AspectRatio::Extreme81 => "1792x1024",
            AspectRatio::Portrait169 | AspectRatio::Extreme14 | AspectRatio::Extreme18 => {
                "1024x1792"
            }
            AspectRatio::Landscape43 | AspectRatio::Frame54 | AspectRatio::Print32 => "1536x1024",
            AspectRatio::Portrait43 | AspectRatio::Frame45 | AspectRatio::Print23 => "1024x1536",
        }
    }

    fn resolve_size(&self, request: &ImageGenRequest) -> &'static str {
        let ratio = request.options.aspect_ratio_or_default();
        Self::map_aspect_ratio_to_size(ratio)
    }

    fn parse_size_dimensions(size: &str) -> (u32, u32) {
        let parts: Vec<&str> = size.split('x').collect();
        if parts.len() == 2 {
            let w = parts[0].parse().unwrap_or(1024);
            let h = parts[1].parse().unwrap_or(1024);
            (w, h)
        } else {
            (1024, 1024)
        }
    }

    fn build_request_body(&self, request: &ImageGenRequest) -> serde_json::Value {
        let model = self.active_model(request);
        let n = request.options.count_or_default();
        let size = self.resolve_size(request);

        let quality = request
            .options
            .extra
            .get("quality")
            .and_then(|v| v.as_str())
            .unwrap_or("auto");

        let mut body = json!({
            "model": model,
            "prompt": request.prompt,
            "n": n,
            "size": size,
            "quality": quality,
        });

        // GPT Image 2.5 uses `output_format`. Earlier models use `response_format`.
        // Source: https://developers.openai.com/api/docs/guides/image-prompting
        if model.contains("gpt-image-2.5") {
            let format = match request.options.output_format {
                Some(ImageFormat::Jpeg) => "jpeg",
                Some(ImageFormat::Webp) => "webp",
                Some(ImageFormat::Png) | None => "png",
            };
            body["output_format"] = json!(format);
        } else {
            body["response_format"] = json!("b64_json");
        }

        body
    }

    fn parse_response(
        &self,
        response: OpenAIImageResponse,
        size: &str,
    ) -> Result<(Vec<GeneratedImage>, Option<String>)> {
        let (width, height) = Self::parse_size_dimensions(size);
        let mut images = Vec::new();
        let mut enhanced_prompt = None;

        for item in response.data {
            if enhanced_prompt.is_none() {
                enhanced_prompt = item.revised_prompt;
            }

            let data = if let Some(b64) = item.b64_json {
                let bytes = BASE64
                    .decode(&b64)
                    .map_err(|e| ImageGenError::InvalidResponse(format!("base64 decode: {e}")))?;
                ImageGenData::Bytes(bytes)
            } else if let Some(url) = item.url {
                ImageGenData::Url(url)
            } else {
                return Err(ImageGenError::InvalidResponse(
                    "response item has neither b64_json nor url".to_string(),
                ));
            };

            images.push(GeneratedImage {
                data,
                width,
                height,
                mime_type: "image/png".to_string(),
                seed: None,
            });
        }

        Ok((images, enhanced_prompt))
    }
}

#[async_trait]
impl ImageGenProvider for OpenAIImageGen {
    fn name(&self) -> &str {
        "openai"
    }

    fn default_model(&self) -> &str {
        DEFAULT_MODEL
    }

    fn available_models(&self) -> Vec<&str> {
        vec![
            "gpt-image-2.5-flare",
            "gpt-image-2.5-sunburst",
            "gpt-image-2",
            "gpt-image-1",
            "gpt-image-1-mini",
        ]
    }

    async fn generate(&self, request: &ImageGenRequest) -> Result<ImageGenResponse> {
        if request.prompt.trim().is_empty() {
            return Err(ImageGenError::InvalidRequest(
                "prompt must not be empty".to_string(),
            ));
        }

        let model = self.active_model(request).to_string();
        let size = self.resolve_size(request);
        let body = self.build_request_body(request);

        debug!(provider = "openai", model = %model, "generating image");

        let started = Instant::now();
        let response = self
            .http_client
            .post(self.endpoint_url())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        let response_text = response.text().await?;

        if !status.is_success() {
            warn!(provider = "openai", status = %status, "image generation failed");
            return Err(match status.as_u16() {
                400 => ImageGenError::InvalidRequest(response_text),
                401 | 403 => ImageGenError::AuthError(response_text),
                429 => ImageGenError::RateLimited { retry_after: None },
                _ => ImageGenError::ProviderError(format!(
                    "HTTP {}: {}",
                    status.as_u16(),
                    response_text
                )),
            });
        }

        let latency_ms = started.elapsed().as_millis() as u64;
        let payload: OpenAIImageResponse = serde_json::from_str(&response_text)?;

        if payload.data.is_empty() {
            return Err(ImageGenError::InvalidResponse(
                "OpenAI returned empty image data".to_string(),
            ));
        }

        let (images, enhanced_prompt) = self.parse_response(payload, size)?;

        Ok(ImageGenResponse {
            images,
            provider: self.name().to_string(),
            model,
            latency_ms,
            enhanced_prompt,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imagegen::types::{AspectRatio, ImageGenOptions, ImageGenRequest};

    #[test]
    fn test_build_request_body_default() {
        let provider = OpenAIImageGen::new("test-key");
        let request = ImageGenRequest::new("A sunset over mountains");
        let body = provider.build_request_body(&request);

        assert_eq!(body["model"], "gpt-image-2.5-flare");
        assert_eq!(body["prompt"], "A sunset over mountains");
        assert_eq!(body["n"], 1);
        assert_eq!(body["size"], "1024x1024");
        assert_eq!(body["quality"], "auto");
        assert_eq!(body["output_format"], "png");
        assert!(body.get("response_format").is_none());
    }

    #[test]
    fn test_gpt_image_2_keeps_response_format() {
        let provider = OpenAIImageGen::new("test-key");
        let request = ImageGenRequest::new("A sunset").with_model("gpt-image-2");
        let body = provider.build_request_body(&request);
        assert_eq!(body["response_format"], "b64_json");
        assert!(body.get("output_format").is_none());
    }

    #[test]
    fn test_build_request_body_landscape() {
        let provider = OpenAIImageGen::new("test-key");
        let request = ImageGenRequest::new("panoramic view").with_options(ImageGenOptions {
            aspect_ratio: Some(AspectRatio::Landscape169),
            count: Some(2),
            ..Default::default()
        });
        let body = provider.build_request_body(&request);

        assert_eq!(body["size"], "1792x1024");
        assert_eq!(body["n"], 2);
    }

    #[test]
    fn test_build_request_body_portrait() {
        let provider = OpenAIImageGen::new("test-key");
        let request = ImageGenRequest::new("tall building").with_options(ImageGenOptions {
            aspect_ratio: Some(AspectRatio::Portrait169),
            ..Default::default()
        });
        let body = provider.build_request_body(&request);

        assert_eq!(body["size"], "1024x1792");
    }

    #[test]
    fn test_build_request_body_with_model_override() {
        let provider = OpenAIImageGen::new("test-key");
        let request = ImageGenRequest::new("a cat").with_model("gpt-image-1");
        let body = provider.build_request_body(&request);

        assert_eq!(body["model"], "gpt-image-1");
    }

    #[test]
    fn test_size_mapping_all_ratios() {
        assert_eq!(
            OpenAIImageGen::map_aspect_ratio_to_size(AspectRatio::Square),
            "1024x1024"
        );
        assert_eq!(
            OpenAIImageGen::map_aspect_ratio_to_size(AspectRatio::Landscape43),
            "1536x1024"
        );
        assert_eq!(
            OpenAIImageGen::map_aspect_ratio_to_size(AspectRatio::Portrait43),
            "1024x1536"
        );
        assert_eq!(
            OpenAIImageGen::map_aspect_ratio_to_size(AspectRatio::Frame54),
            "1536x1024"
        );
        assert_eq!(
            OpenAIImageGen::map_aspect_ratio_to_size(AspectRatio::Frame45),
            "1024x1536"
        );
    }

    #[test]
    fn test_parse_size_dimensions() {
        assert_eq!(
            OpenAIImageGen::parse_size_dimensions("1024x1024"),
            (1024, 1024)
        );
        assert_eq!(
            OpenAIImageGen::parse_size_dimensions("1792x1024"),
            (1792, 1024)
        );
        assert_eq!(
            OpenAIImageGen::parse_size_dimensions("1024x1792"),
            (1024, 1792)
        );
    }

    #[test]
    fn test_parse_response_b64() {
        let provider = OpenAIImageGen::new("test-key");
        let raw = OpenAIImageResponse {
            data: vec![OpenAIImageItem {
                b64_json: Some(BASE64.encode(b"fake-image-bytes")),
                url: None,
                revised_prompt: Some("enhanced prompt".to_string()),
            }],
        };
        let (images, enhanced) = provider.parse_response(raw, "1024x1024").unwrap();

        assert_eq!(images.len(), 1);
        assert_eq!(images[0].width, 1024);
        assert_eq!(images[0].height, 1024);
        assert_eq!(
            images[0].data,
            ImageGenData::Bytes(b"fake-image-bytes".to_vec())
        );
        assert_eq!(enhanced, Some("enhanced prompt".to_string()));
    }

    #[test]
    fn test_available_models() {
        let provider = OpenAIImageGen::new("key");
        let models = provider.available_models();
        assert!(models.contains(&"gpt-image-2.5-flare"));
        assert!(models.contains(&"gpt-image-2.5-sunburst"));
        assert!(models.contains(&"gpt-image-2"));
        assert!(models.contains(&"gpt-image-1"));
        assert!(models.contains(&"gpt-image-1-mini"));
    }
}
