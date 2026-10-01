//! xAI Grok Imagine image generation provider.

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
    AspectRatio, GeneratedImage, ImageGenData, ImageGenRequest, ImageGenResponse,
};

const DEFAULT_BASE_URL: &str = "https://api.x.ai/v1";
const DEFAULT_MODEL: &str = "grok-imagine-image-2.0";

/// xAI Grok Imagine image generation provider.
#[derive(Debug, Clone)]
pub struct XAIImageGen {
    api_key: String,
    base_url: String,
    http_client: Client,
}

#[derive(Debug, Deserialize)]
struct XAIImageResponse {
    data: Vec<XAIImageItem>,
}

#[derive(Debug, Deserialize)]
struct XAIImageItem {
    #[serde(default)]
    b64_json: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    revised_prompt: Option<String>,
}

impl XAIImageGen {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            http_client: Client::new(),
        }
    }

    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("XAI_API_KEY")
            .map_err(|_| ImageGenError::ConfigError("XAI_API_KEY must be set".to_string()))?;
        Ok(Self::new(api_key))
    }

    fn active_model<'a>(&'a self, request: &'a ImageGenRequest) -> &'a str {
        request.model.as_deref().unwrap_or(DEFAULT_MODEL)
    }

    fn endpoint_url(&self) -> String {
        format!("{}/images/generations", self.base_url)
    }

    fn map_aspect_ratio(ratio: AspectRatio) -> &'static str {
        match ratio {
            AspectRatio::Auto => "auto",
            AspectRatio::Square | AspectRatio::SquareHd => "1:1",
            AspectRatio::Landscape43 | AspectRatio::Frame54 => "4:3",
            AspectRatio::Landscape169 => "16:9",
            AspectRatio::Ultrawide | AspectRatio::Extreme41 | AspectRatio::Extreme81 => "2:1",
            AspectRatio::Portrait43 | AspectRatio::Frame45 => "3:4",
            AspectRatio::Portrait169 => "9:16",
            AspectRatio::Print32 => "3:2",
            AspectRatio::Print23 => "2:3",
            AspectRatio::Extreme14 | AspectRatio::Extreme18 => "1:2",
        }
    }

    fn dimensions_for_ratio(ratio: AspectRatio) -> (u32, u32) {
        match ratio {
            AspectRatio::Auto | AspectRatio::Square | AspectRatio::SquareHd => (1024, 1024),
            AspectRatio::Landscape43 | AspectRatio::Frame54 => (1024, 768),
            AspectRatio::Landscape169 => (1024, 576),
            AspectRatio::Ultrawide | AspectRatio::Extreme41 | AspectRatio::Extreme81 => (1024, 512),
            AspectRatio::Portrait43 | AspectRatio::Frame45 => (768, 1024),
            AspectRatio::Portrait169 => (576, 1024),
            AspectRatio::Print32 => (1024, 683),
            AspectRatio::Print23 => (683, 1024),
            AspectRatio::Extreme14 | AspectRatio::Extreme18 => (512, 1024),
        }
    }

    fn build_request_body(&self, request: &ImageGenRequest) -> serde_json::Value {
        let model = self.active_model(request);
        let n = request.options.count_or_default();
        let ratio = request.options.aspect_ratio_or_default();
        let aspect_ratio_str = Self::map_aspect_ratio(ratio);

        let response_format = request
            .options
            .extra
            .get("response_format")
            .and_then(|v| v.as_str())
            .unwrap_or("b64_json");

        json!({
            "model": model,
            "prompt": request.prompt,
            "n": n,
            "response_format": response_format,
            "aspect_ratio": aspect_ratio_str
        })
    }

    fn parse_response(
        &self,
        response: XAIImageResponse,
        ratio: AspectRatio,
    ) -> Result<(Vec<GeneratedImage>, Option<String>)> {
        let (width, height) = Self::dimensions_for_ratio(ratio);
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
impl ImageGenProvider for XAIImageGen {
    fn name(&self) -> &str {
        "xai"
    }

    fn default_model(&self) -> &str {
        DEFAULT_MODEL
    }

    fn available_models(&self) -> Vec<&str> {
        vec![
            "grok-imagine-image-2.0",
            "grok-imagine-image-quality",
            "grok-imagine-image",
        ]
    }

    async fn generate(&self, request: &ImageGenRequest) -> Result<ImageGenResponse> {
        if request.prompt.trim().is_empty() {
            return Err(ImageGenError::InvalidRequest(
                "prompt must not be empty".to_string(),
            ));
        }

        let model = self.active_model(request).to_string();
        let ratio = request.options.aspect_ratio_or_default();
        let body = self.build_request_body(request);

        debug!(provider = "xai", model = %model, "generating image");

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
            warn!(provider = "xai", status = %status, "image generation failed");
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
        let payload: XAIImageResponse = serde_json::from_str(&response_text)?;

        if payload.data.is_empty() {
            return Err(ImageGenError::InvalidResponse(
                "xAI returned empty image data".to_string(),
            ));
        }

        let (images, enhanced_prompt) = self.parse_response(payload, ratio)?;

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
        let provider = XAIImageGen::new("test-key");
        let request = ImageGenRequest::new("A collage of London landmarks");
        let body = provider.build_request_body(&request);

        assert_eq!(body["model"], "grok-imagine-image-2.0");
        assert_eq!(body["prompt"], "A collage of London landmarks");
        assert_eq!(body["n"], 1);
        assert_eq!(body["response_format"], "b64_json");
        assert_eq!(body["aspect_ratio"], "auto");
    }

    #[test]
    fn test_build_request_body_square() {
        let provider = XAIImageGen::new("test-key");
        let request = ImageGenRequest::new("a cat").with_options(ImageGenOptions {
            aspect_ratio: Some(AspectRatio::Square),
            ..Default::default()
        });
        let body = provider.build_request_body(&request);

        assert_eq!(body["aspect_ratio"], "1:1");
    }

    #[test]
    fn test_build_request_body_landscape() {
        let provider = XAIImageGen::new("test-key");
        let request = ImageGenRequest::new("panoramic view").with_options(ImageGenOptions {
            aspect_ratio: Some(AspectRatio::Landscape169),
            count: Some(3),
            ..Default::default()
        });
        let body = provider.build_request_body(&request);

        assert_eq!(body["aspect_ratio"], "16:9");
        assert_eq!(body["n"], 3);
    }

    #[test]
    fn test_build_request_body_portrait() {
        let provider = XAIImageGen::new("test-key");
        let request = ImageGenRequest::new("tall tower").with_options(ImageGenOptions {
            aspect_ratio: Some(AspectRatio::Portrait169),
            ..Default::default()
        });
        let body = provider.build_request_body(&request);

        assert_eq!(body["aspect_ratio"], "9:16");
    }

    #[test]
    fn test_build_request_body_with_model_override() {
        let provider = XAIImageGen::new("test-key");
        let request = ImageGenRequest::new("a dog").with_model("grok-imagine-image");
        let body = provider.build_request_body(&request);

        assert_eq!(body["model"], "grok-imagine-image");
    }

    #[test]
    fn test_aspect_ratio_mapping() {
        assert_eq!(XAIImageGen::map_aspect_ratio(AspectRatio::Auto), "auto");
        assert_eq!(XAIImageGen::map_aspect_ratio(AspectRatio::Square), "1:1");
        assert_eq!(
            XAIImageGen::map_aspect_ratio(AspectRatio::Landscape43),
            "4:3"
        );
        assert_eq!(
            XAIImageGen::map_aspect_ratio(AspectRatio::Landscape169),
            "16:9"
        );
        assert_eq!(
            XAIImageGen::map_aspect_ratio(AspectRatio::Portrait43),
            "3:4"
        );
        assert_eq!(
            XAIImageGen::map_aspect_ratio(AspectRatio::Portrait169),
            "9:16"
        );
        assert_eq!(XAIImageGen::map_aspect_ratio(AspectRatio::Print32), "3:2");
        assert_eq!(XAIImageGen::map_aspect_ratio(AspectRatio::Print23), "2:3");
        assert_eq!(XAIImageGen::map_aspect_ratio(AspectRatio::Ultrawide), "2:1");
        assert_eq!(XAIImageGen::map_aspect_ratio(AspectRatio::Extreme14), "1:2");
    }

    #[test]
    fn test_parse_response_b64() {
        let provider = XAIImageGen::new("test-key");
        let raw = XAIImageResponse {
            data: vec![XAIImageItem {
                b64_json: Some(BASE64.encode(b"fake-grok-image")),
                url: None,
                revised_prompt: Some("better prompt".to_string()),
            }],
        };
        let (images, enhanced) = provider.parse_response(raw, AspectRatio::Square).unwrap();

        assert_eq!(images.len(), 1);
        assert_eq!(images[0].width, 1024);
        assert_eq!(images[0].height, 1024);
        assert_eq!(
            images[0].data,
            ImageGenData::Bytes(b"fake-grok-image".to_vec())
        );
        assert_eq!(enhanced, Some("better prompt".to_string()));
    }

    #[test]
    fn test_parse_response_url() {
        let provider = XAIImageGen::new("test-key");
        let raw = XAIImageResponse {
            data: vec![XAIImageItem {
                b64_json: None,
                url: Some("https://example.com/image.png".to_string()),
                revised_prompt: None,
            }],
        };
        let (images, enhanced) = provider
            .parse_response(raw, AspectRatio::Landscape169)
            .unwrap();

        assert_eq!(images.len(), 1);
        assert_eq!(images[0].width, 1024);
        assert_eq!(images[0].height, 576);
        assert_eq!(
            images[0].data,
            ImageGenData::Url("https://example.com/image.png".to_string())
        );
        assert_eq!(enhanced, None);
    }

    #[test]
    fn test_available_models() {
        let provider = XAIImageGen::new("key");
        let models = provider.available_models();
        assert!(models.contains(&"grok-imagine-image-2.0"));
        assert!(models.contains(&"grok-imagine-image-quality"));
        assert!(models.contains(&"grok-imagine-image"));
    }
}
