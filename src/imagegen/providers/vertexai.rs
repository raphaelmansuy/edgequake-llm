use std::time::Instant;

use async_trait::async_trait;
use base64::Engine;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::imagegen::error::{ImageGenError, Result};
use crate::imagegen::providers::gcp::{access_token_from_env_or_gcloud, env_region};
use crate::imagegen::traits::ImageGenProvider;
use crate::imagegen::types::{
    GeneratedImage, ImageGenData, ImageGenRequest, ImageGenResponse, SafetyLevel,
};

const DEFAULT_VERTEX_IMAGEN_MODEL: &str = "imagen-4.0-generate-001";

#[derive(Debug, Clone)]
pub struct VertexAIImageGen {
    client: Client,
    project_id: String,
    region: String,
    access_token: String,
    model: String,
}

#[derive(Debug, Serialize)]
struct VertexImagenRequest {
    instances: Vec<Value>,
    parameters: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexImagenResponse {
    #[serde(default)]
    predictions: Vec<VertexImagenPrediction>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexImagenPrediction {
    #[serde(default)]
    bytes_base64_encoded: Option<String>,
    #[serde(default)]
    mime_type: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
}

#[derive(Debug, Deserialize)]
struct VertexErrorEnvelope {
    error: VertexErrorBody,
}

#[derive(Debug, Deserialize)]
struct VertexErrorBody {
    message: String,
}

impl VertexAIImageGen {
    pub fn new(
        project_id: impl Into<String>,
        region: impl Into<String>,
        access_token: impl Into<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            project_id: project_id.into(),
            region: region.into(),
            access_token: access_token.into(),
            model: DEFAULT_VERTEX_IMAGEN_MODEL.to_string(),
        }
    }

    pub fn from_env() -> Result<Self> {
        let project_id = std::env::var("GOOGLE_CLOUD_PROJECT").map_err(|_| {
            ImageGenError::ConfigError("Vertex Imagen requires GOOGLE_CLOUD_PROJECT".to_string())
        })?;
        let access_token = access_token_from_env_or_gcloud()?;
        let model = std::env::var("IMAGEGEN_MODEL")
            .unwrap_or_else(|_| DEFAULT_VERTEX_IMAGEN_MODEL.to_string());
        Ok(Self::new(project_id, env_region(), access_token).with_model(model))
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    fn active_model<'a>(&'a self, request: &'a ImageGenRequest) -> &'a str {
        request.model.as_deref().unwrap_or(&self.model)
    }

    fn host(&self) -> String {
        if self.region == "global" {
            "aiplatform.googleapis.com".to_string()
        } else {
            format!("{}-aiplatform.googleapis.com", self.region)
        }
    }

    fn build_url(&self, model: &str) -> String {
        format!(
            "https://{}/v1/projects/{}/locations/{}/publishers/google/models/{}:predict",
            self.host(),
            self.project_id,
            self.region,
            model
        )
    }

    fn validate_request(&self, request: &ImageGenRequest) -> Result<()> {
        if request.prompt.trim().is_empty() {
            return Err(ImageGenError::InvalidRequest(
                "prompt must not be empty".to_string(),
            ));
        }

        let count = request.options.count_or_default();
        if !(1..=4).contains(&count) {
            return Err(ImageGenError::InvalidRequest(
                "Vertex Imagen sample count must be between 1 and 4".to_string(),
            ));
        }

        if request.options.width.is_some() || request.options.height.is_some() {
            return Err(ImageGenError::InvalidRequest(
                "Vertex Imagen does not support explicit width/height; use aspect_ratio"
                    .to_string(),
            ));
        }

        Ok(())
    }

    fn build_request(&self, request: &ImageGenRequest) -> VertexImagenRequest {
        let options = &request.options;
        let output_format = options.output_format_or_default();
        let mut parameters = json!({
            "sampleCount": options.count_or_default(),
            "aspectRatio": options.aspect_ratio_or_default().as_vertex_str(),
            "enhancePrompt": options.enhance_prompt.unwrap_or(false),
            "safetySetting": match options.safety_level_or_default() {
                SafetyLevel::BlockNone => "block_none",
                SafetyLevel::BlockLow => "block_low_and_above",
                SafetyLevel::BlockMedium => "block_medium_and_above",
                SafetyLevel::BlockHigh => "block_only_high",
            },
            "outputOptions": {
                "mimeType": output_format.mime_type(),
            }
        });

        if let Some(seed) = options.seed {
            parameters["seed"] = json!(seed);
        }
        if let Some(negative_prompt) = &options.negative_prompt {
            parameters["negativePrompt"] = json!(negative_prompt);
        }
        if let Some(guidance_scale) = options.guidance_scale {
            parameters["guidanceScale"] = json!(guidance_scale);
        }
        if let Some(style) = options.extra.get("style") {
            parameters["sampleImageStyle"] = style.clone();
        }
        if let Some(watermark) = options.extra.get("watermark") {
            parameters["addWatermark"] = watermark.clone();
        }
        if let Some(person_generation) = options.extra.get("person_gen") {
            parameters["personGeneration"] = person_generation.clone();
        }
        if let Some(language) = options.extra.get("language") {
            parameters["language"] = language.clone();
        }

        VertexImagenRequest {
            instances: vec![json!({ "prompt": request.prompt })],
            parameters,
        }
    }
}

#[async_trait]
impl ImageGenProvider for VertexAIImageGen {
    fn name(&self) -> &str {
        "vertexai-imagen"
    }

    fn default_model(&self) -> &str {
        &self.model
    }

    fn available_models(&self) -> Vec<&str> {
        vec![
            "imagen-4.0-ultra-generate-001",
            "imagen-4.0-generate-001",
            "imagen-4.0-fast-generate-001",
            "imagen-3.0-generate-002",
            "imagen-3.0-generate-001",
            "imagen-3.0-fast-generate-001",
        ]
    }

    async fn generate(&self, request: &ImageGenRequest) -> Result<ImageGenResponse> {
        self.validate_request(request)?;
        let model = self.active_model(request).to_string();
        let started_at = Instant::now();

        let response = self
            .client
            .post(self.build_url(&model))
            .bearer_auth(&self.access_token)
            .json(&self.build_request(request))
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;
        if !status.is_success() {
            let parsed = serde_json::from_str::<VertexErrorEnvelope>(&body).ok();
            let message = parsed.map(|payload| payload.error.message).unwrap_or(body);
            return Err(match status.as_u16() {
                400 => ImageGenError::InvalidRequest(message),
                401 | 403 => ImageGenError::AuthError(message),
                429 => ImageGenError::RateLimited { retry_after: None },
                500..=599 => ImageGenError::ProviderError(message),
                _ => ImageGenError::ProviderError(message),
            });
        }

        let payload: VertexImagenResponse = serde_json::from_str(&body)?;
        if payload.predictions.is_empty() {
            return Err(ImageGenError::ContentFiltered {
                reason: "Vertex Imagen returned no predictions".to_string(),
            });
        }

        let (width, height) = request
            .options
            .aspect_ratio_or_default()
            .default_dimensions();
        let mut enhanced_prompt = None;
        let mut images = Vec::new();
        for prediction in payload.predictions {
            let bytes_base64_encoded = prediction.bytes_base64_encoded.ok_or_else(|| {
                ImageGenError::InvalidResponse(
                    "Vertex Imagen prediction missing bytesBase64Encoded".to_string(),
                )
            })?;
            if enhanced_prompt.is_none() {
                enhanced_prompt = prediction.prompt.clone();
            }
            images.push(GeneratedImage {
                data: ImageGenData::Bytes(
                    base64::engine::general_purpose::STANDARD.decode(bytes_base64_encoded)?,
                ),
                width,
                height,
                mime_type: prediction.mime_type.unwrap_or_else(|| {
                    request
                        .options
                        .output_format_or_default()
                        .mime_type()
                        .to_string()
                }),
                seed: request.options.seed,
            });
        }

        Ok(ImageGenResponse {
            images,
            provider: self.name().to_string(),
            model,
            latency_ms: started_at.elapsed().as_millis() as u64,
            enhanced_prompt,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::VertexAIImageGen;
    use crate::imagegen::types::{AspectRatio, ImageGenOptions, ImageGenRequest};

    #[test]
    fn test_build_request_maps_options() {
        let provider = VertexAIImageGen::new("proj", "us-central1", "token");
        let request = ImageGenRequest::new("test").with_options(ImageGenOptions {
            aspect_ratio: Some(AspectRatio::Landscape169),
            seed: Some(42),
            ..Default::default()
        });
        let body = provider.build_request(&request);
        assert_eq!(body.parameters["aspectRatio"], "16:9");
        assert_eq!(body.parameters["seed"], 42);
    }
}
