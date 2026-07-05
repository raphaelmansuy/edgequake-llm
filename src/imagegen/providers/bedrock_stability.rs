//! Bedrock Stability AI image generation provider.
//!
//! Requires the `bedrock` feature flag. Uses the AWS SDK `InvokeModel` API
//! to call Stability AI models hosted on Amazon Bedrock.

#![allow(unused_imports)]

#[cfg(feature = "bedrock")]
use std::time::Instant;

#[cfg(feature = "bedrock")]
use async_trait::async_trait;
#[cfg(feature = "bedrock")]
use aws_sdk_bedrockruntime::primitives::Blob;
#[cfg(feature = "bedrock")]
use aws_sdk_bedrockruntime::Client;
#[cfg(feature = "bedrock")]
use base64::engine::general_purpose::STANDARD as BASE64;
#[cfg(feature = "bedrock")]
use base64::Engine;
#[cfg(feature = "bedrock")]
use serde::Deserialize;
#[cfg(feature = "bedrock")]
use serde_json::json;
#[cfg(feature = "bedrock")]
use tracing::{debug, warn};

#[cfg(feature = "bedrock")]
use crate::imagegen::error::{ImageGenError, Result};
#[cfg(feature = "bedrock")]
use crate::imagegen::traits::ImageGenProvider;
#[cfg(feature = "bedrock")]
use crate::imagegen::types::{
    AspectRatio, GeneratedImage, ImageGenData, ImageGenRequest, ImageGenResponse,
};

#[cfg(feature = "bedrock")]
const DEFAULT_MODEL_ID: &str = "stability.stable-image-core-v1:1";

/// Bedrock Stability AI image generation provider.
///
/// Invokes Stability AI models through the Amazon Bedrock `InvokeModel` API.
/// Authentication is handled by the standard AWS credential chain.
#[cfg(feature = "bedrock")]
#[derive(Debug, Clone)]
pub struct BedrockStabilityImageGen {
    client: Client,
    model_id: String,
    region: String,
}

#[cfg(feature = "bedrock")]
#[derive(Debug, Deserialize)]
struct StabilityResponse {
    images: Vec<String>,
    #[serde(default)]
    seeds: Vec<u64>,
    #[serde(default)]
    finish_reasons: Vec<String>,
}

#[cfg(feature = "bedrock")]
impl BedrockStabilityImageGen {
    /// Create a new provider from an existing AWS SDK config.
    pub fn new(sdk_config: &aws_config::SdkConfig) -> Self {
        let region = sdk_config
            .region()
            .map(|r| r.to_string())
            .unwrap_or_else(|| "us-east-1".to_string());

        Self {
            client: Client::new(sdk_config),
            model_id: DEFAULT_MODEL_ID.to_string(),
            region,
        }
    }

    /// Create a provider using the default AWS credential chain.
    pub async fn from_env() -> Result<Self> {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        Ok(Self::new(&config))
    }

    /// Override the model id.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model_id = model.into();
        self
    }

    fn active_model<'a>(&'a self, request: &'a ImageGenRequest) -> &'a str {
        request.model.as_deref().unwrap_or(&self.model_id)
    }

    fn map_aspect_ratio(ratio: AspectRatio) -> &'static str {
        match ratio {
            AspectRatio::Auto | AspectRatio::Square | AspectRatio::SquareHd => "1:1",
            AspectRatio::Landscape43 => "4:3",
            AspectRatio::Portrait43 => "3:4",
            AspectRatio::Landscape169 => "16:9",
            AspectRatio::Portrait169 => "9:16",
            AspectRatio::Print32 => "3:2",
            AspectRatio::Print23 => "2:3",
            AspectRatio::Frame54 => "5:4",
            AspectRatio::Frame45 => "4:5",
            AspectRatio::Ultrawide | AspectRatio::Extreme41 | AspectRatio::Extreme81 => "21:9",
            AspectRatio::Extreme14 | AspectRatio::Extreme18 => "9:16",
        }
    }

    fn dimensions_for_ratio(ratio: AspectRatio) -> (u32, u32) {
        match ratio {
            AspectRatio::Auto | AspectRatio::Square | AspectRatio::SquareHd => (1024, 1024),
            AspectRatio::Landscape43 | AspectRatio::Frame54 => (1024, 768),
            AspectRatio::Portrait43 | AspectRatio::Frame45 => (768, 1024),
            AspectRatio::Landscape169 => (1024, 576),
            AspectRatio::Portrait169 => (576, 1024),
            AspectRatio::Print32 => (1024, 683),
            AspectRatio::Print23 => (683, 1024),
            AspectRatio::Ultrawide | AspectRatio::Extreme41 | AspectRatio::Extreme81 => (1024, 439),
            AspectRatio::Extreme14 | AspectRatio::Extreme18 => (576, 1024),
        }
    }

    fn build_request_body(&self, request: &ImageGenRequest) -> serde_json::Value {
        let ratio = request.options.aspect_ratio_or_default();
        let aspect_ratio_str = Self::map_aspect_ratio(ratio);
        let seed = request.options.seed.unwrap_or(0);

        let mut body = json!({
            "prompt": request.prompt,
            "mode": "text-to-image",
            "aspect_ratio": aspect_ratio_str,
            "output_format": "png",
            "seed": seed
        });

        if let Some(ref neg) = request.options.negative_prompt {
            body["negative_prompt"] = serde_json::Value::String(neg.clone());
        }

        body
    }
}

#[cfg(feature = "bedrock")]
#[async_trait]
impl ImageGenProvider for BedrockStabilityImageGen {
    fn name(&self) -> &str {
        "bedrock-stability"
    }

    fn default_model(&self) -> &str {
        DEFAULT_MODEL_ID
    }

    fn available_models(&self) -> Vec<&str> {
        vec![
            "stability.stable-image-core-v1:1",
            "stability.stable-image-ultra-v1:0",
            "stability.sd3-5-large-v1:0",
        ]
    }

    async fn generate(&self, request: &ImageGenRequest) -> Result<ImageGenResponse> {
        if request.prompt.trim().is_empty() {
            return Err(ImageGenError::InvalidRequest(
                "prompt must not be empty".to_string(),
            ));
        }

        let model_id = self.active_model(request).to_string();
        let ratio = request.options.aspect_ratio_or_default();
        let body = self.build_request_body(request);
        let body_bytes = serde_json::to_vec(&body)?;

        debug!(
            provider = "bedrock-stability",
            model = %model_id,
            region = %self.region,
            "generating image"
        );

        let started = Instant::now();

        let response = self
            .client
            .invoke_model()
            .model_id(&model_id)
            .content_type("application/json")
            .body(Blob::new(body_bytes))
            .send()
            .await
            .map_err(|e| ImageGenError::ProviderError(format!("Bedrock InvokeModel: {e}")))?;

        let latency_ms = started.elapsed().as_millis() as u64;

        let response_bytes = response.body().as_ref();
        let payload: StabilityResponse = serde_json::from_slice(response_bytes)?;

        for (i, reason) in payload.finish_reasons.iter().enumerate() {
            if reason != "SUCCESS" {
                warn!(
                    provider = "bedrock-stability",
                    index = i,
                    finish_reason = %reason,
                    "non-success finish reason"
                );
                return Err(ImageGenError::ContentFiltered {
                    reason: format!("image {i} finish_reason: {reason}"),
                });
            }
        }

        if payload.images.is_empty() {
            return Err(ImageGenError::InvalidResponse(
                "Bedrock Stability returned no images".to_string(),
            ));
        }

        let (width, height) = Self::dimensions_for_ratio(ratio);
        let mut images = Vec::with_capacity(payload.images.len());

        for (i, b64) in payload.images.iter().enumerate() {
            let bytes = BASE64
                .decode(b64)
                .map_err(|e| ImageGenError::InvalidResponse(format!("base64 decode: {e}")))?;

            let seed = payload.seeds.get(i).copied();

            images.push(GeneratedImage {
                data: ImageGenData::Bytes(bytes),
                width,
                height,
                mime_type: "image/png".to_string(),
                seed,
            });
        }

        Ok(ImageGenResponse {
            images,
            provider: self.name().to_string(),
            model: model_id,
            latency_ms,
            enhanced_prompt: None,
        })
    }
}

#[cfg(test)]
#[cfg(feature = "bedrock")]
mod tests {
    use super::*;
    use crate::imagegen::types::{AspectRatio, ImageGenOptions, ImageGenRequest};

    #[test]
    fn test_aspect_ratio_mapping() {
        assert_eq!(
            BedrockStabilityImageGen::map_aspect_ratio(AspectRatio::Auto),
            "1:1"
        );
        assert_eq!(
            BedrockStabilityImageGen::map_aspect_ratio(AspectRatio::Square),
            "1:1"
        );
        assert_eq!(
            BedrockStabilityImageGen::map_aspect_ratio(AspectRatio::Landscape43),
            "4:3"
        );
        assert_eq!(
            BedrockStabilityImageGen::map_aspect_ratio(AspectRatio::Portrait43),
            "3:4"
        );
        assert_eq!(
            BedrockStabilityImageGen::map_aspect_ratio(AspectRatio::Landscape169),
            "16:9"
        );
        assert_eq!(
            BedrockStabilityImageGen::map_aspect_ratio(AspectRatio::Portrait169),
            "9:16"
        );
        assert_eq!(
            BedrockStabilityImageGen::map_aspect_ratio(AspectRatio::Print32),
            "3:2"
        );
        assert_eq!(
            BedrockStabilityImageGen::map_aspect_ratio(AspectRatio::Print23),
            "2:3"
        );
        assert_eq!(
            BedrockStabilityImageGen::map_aspect_ratio(AspectRatio::Frame54),
            "5:4"
        );
        assert_eq!(
            BedrockStabilityImageGen::map_aspect_ratio(AspectRatio::Frame45),
            "4:5"
        );
        assert_eq!(
            BedrockStabilityImageGen::map_aspect_ratio(AspectRatio::Ultrawide),
            "21:9"
        );
    }

    #[test]
    fn test_dimensions_for_ratio() {
        assert_eq!(
            BedrockStabilityImageGen::dimensions_for_ratio(AspectRatio::Square),
            (1024, 1024)
        );
        assert_eq!(
            BedrockStabilityImageGen::dimensions_for_ratio(AspectRatio::Landscape43),
            (1024, 768)
        );
        assert_eq!(
            BedrockStabilityImageGen::dimensions_for_ratio(AspectRatio::Portrait169),
            (576, 1024)
        );
    }

    #[test]
    fn test_build_request_body_default() {
        let config = aws_config::SdkConfig::builder()
            .region(aws_config::Region::new("us-east-1"))
            .build();
        let provider = BedrockStabilityImageGen::new(&config);
        let request = ImageGenRequest::new("A beautiful sunset");
        let body = provider.build_request_body(&request);

        assert_eq!(body["prompt"], "A beautiful sunset");
        assert_eq!(body["mode"], "text-to-image");
        assert_eq!(body["aspect_ratio"], "1:1");
        assert_eq!(body["output_format"], "png");
        assert_eq!(body["seed"], 0);
        assert!(body.get("negative_prompt").is_none());
    }

    #[test]
    fn test_build_request_body_with_options() {
        let config = aws_config::SdkConfig::builder()
            .region(aws_config::Region::new("us-west-2"))
            .build();
        let provider = BedrockStabilityImageGen::new(&config);
        let request = ImageGenRequest::new("A mountain scene").with_options(ImageGenOptions {
            aspect_ratio: Some(AspectRatio::Landscape169),
            seed: Some(42),
            negative_prompt: Some("blurry, low quality".to_string()),
            ..Default::default()
        });
        let body = provider.build_request_body(&request);

        assert_eq!(body["aspect_ratio"], "16:9");
        assert_eq!(body["seed"], 42);
        assert_eq!(body["negative_prompt"], "blurry, low quality");
    }

    #[test]
    fn test_with_model() {
        let config = aws_config::SdkConfig::builder()
            .region(aws_config::Region::new("us-east-1"))
            .build();
        let provider =
            BedrockStabilityImageGen::new(&config).with_model("stability.stable-image-ultra-v1:0");
        assert_eq!(provider.model_id, "stability.stable-image-ultra-v1:0");
    }

    #[test]
    fn test_available_models() {
        let config = aws_config::SdkConfig::builder()
            .region(aws_config::Region::new("us-east-1"))
            .build();
        let provider = BedrockStabilityImageGen::new(&config);
        let models = provider.available_models();
        assert!(models.contains(&"stability.stable-image-core-v1:1"));
        assert!(models.contains(&"stability.stable-image-ultra-v1:0"));
        assert!(models.contains(&"stability.sd3-5-large-v1:0"));
    }

    #[test]
    fn test_name() {
        let config = aws_config::SdkConfig::builder()
            .region(aws_config::Region::new("us-east-1"))
            .build();
        let provider = BedrockStabilityImageGen::new(&config);
        assert_eq!(provider.name(), "bedrock-stability");
    }

    #[test]
    fn test_default_model() {
        let config = aws_config::SdkConfig::builder()
            .region(aws_config::Region::new("us-east-1"))
            .build();
        let provider = BedrockStabilityImageGen::new(&config);
        assert_eq!(provider.default_model(), "stability.stable-image-core-v1:1");
    }

    #[test]
    fn test_active_model_default() {
        let config = aws_config::SdkConfig::builder()
            .region(aws_config::Region::new("us-east-1"))
            .build();
        let provider = BedrockStabilityImageGen::new(&config);
        let request = ImageGenRequest::new("test");
        assert_eq!(
            provider.active_model(&request),
            "stability.stable-image-core-v1:1"
        );
    }

    #[test]
    fn test_active_model_override() {
        let config = aws_config::SdkConfig::builder()
            .region(aws_config::Region::new("us-east-1"))
            .build();
        let provider = BedrockStabilityImageGen::new(&config);
        let request = ImageGenRequest::new("test").with_model("stability.stable-image-ultra-v1:0");
        assert_eq!(
            provider.active_model(&request),
            "stability.stable-image-ultra-v1:0"
        );
    }
}
