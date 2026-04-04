use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::time::sleep;

use crate::imagegen::error::{ImageGenError, Result};
use crate::imagegen::traits::ImageGenProvider;
use crate::imagegen::types::{
    GeneratedImage, ImageGenData, ImageGenRequest, ImageGenResponse, SafetyLevel,
};

const DEFAULT_FAL_MODEL: &str = "fal-ai/flux/dev";
const DEFAULT_TIMEOUT_SECS: u64 = 300;
const DEFAULT_POLL_INTERVAL_MS: u64 = 500;
const MAX_POLL_INTERVAL_MS: u64 = 5_000;

#[derive(Debug, Clone)]
pub struct FalImageGen {
    client: Client,
    api_key: String,
    model: String,
    timeout_secs: u64,
    poll_interval_ms: u64,
}

#[derive(Debug, Deserialize)]
struct FalSubmitResponse {
    request_id: String,
}

#[derive(Debug, Deserialize)]
struct FalStatusResponse {
    status: String,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FalResultResponse {
    #[serde(default)]
    images: Vec<FalImage>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    has_nsfw_concepts: Vec<bool>,
    #[serde(default)]
    timings: Option<FalTimings>,
}

#[derive(Debug, Deserialize)]
struct FalImage {
    url: String,
    #[serde(default)]
    width: u32,
    #[serde(default)]
    height: u32,
    #[serde(default)]
    content_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FalTimings {
    #[serde(default)]
    inference: Option<f64>,
}

impl FalImageGen {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            model: DEFAULT_FAL_MODEL.to_string(),
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            poll_interval_ms: DEFAULT_POLL_INTERVAL_MS,
        }
    }

    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("FAL_KEY")
            .map_err(|_| ImageGenError::ConfigError("FAL_KEY must be set".to_string()))?;
        let model =
            std::env::var("IMAGEGEN_FAL_MODEL").unwrap_or_else(|_| DEFAULT_FAL_MODEL.to_string());
        let timeout_secs = std::env::var("IMAGEGEN_FAL_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_TIMEOUT_SECS);
        let poll_interval_ms = std::env::var("IMAGEGEN_FAL_POLL_INTERVAL")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_POLL_INTERVAL_MS);
        Ok(Self::new(api_key)
            .with_model(model)
            .with_timeout_secs(timeout_secs)
            .with_poll_interval_ms(poll_interval_ms))
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    pub fn with_poll_interval_ms(mut self, poll_interval_ms: u64) -> Self {
        self.poll_interval_ms = poll_interval_ms;
        self
    }

    fn active_model<'a>(&'a self, request: &'a ImageGenRequest) -> &'a str {
        request.model.as_deref().unwrap_or(&self.model)
    }

    fn submit_url(&self, endpoint: &str) -> String {
        format!("https://queue.fal.run/{endpoint}")
    }

    fn status_url(&self, endpoint: &str, request_id: &str) -> String {
        format!("https://queue.fal.run/{endpoint}/requests/{request_id}/status")
    }

    fn result_url(&self, endpoint: &str, request_id: &str) -> String {
        format!("https://queue.fal.run/{endpoint}/requests/{request_id}")
    }

    fn build_request_body(&self, request: &ImageGenRequest, endpoint: &str) -> Value {
        let options = &request.options;
        let image_size = match (options.width, options.height) {
            (Some(width), Some(height)) => json!({ "width": width, "height": height }),
            _ => json!(options.aspect_ratio_or_default().as_fal_str()),
        };

        let default_steps = if endpoint.contains("schnell") {
            4
        } else if endpoint.contains("pro") {
            40
        } else {
            28
        };

        let steps = options
            .extra
            .get("steps")
            .and_then(|value| value.as_u64())
            .unwrap_or(default_steps);
        let acceleration = options
            .extra
            .get("acceleration")
            .cloned()
            .unwrap_or_else(|| json!("none"));

        json!({
            "prompt": request.prompt,
            "image_size": image_size,
            "num_inference_steps": steps,
            "guidance_scale": options.guidance_scale.unwrap_or(3.5),
            "num_images": options.count_or_default(),
            "enable_safety_checker": !matches!(options.safety_level_or_default(), SafetyLevel::BlockNone),
            "output_format": options.output_format_or_default().as_fal_str(),
            "acceleration": acceleration,
            "seed": options.seed,
        })
    }

    async fn submit(&self, endpoint: &str, body: &Value) -> Result<String> {
        let response = self
            .client
            .post(self.submit_url(endpoint))
            .header("Authorization", format!("Key {}", self.api_key))
            .json(body)
            .send()
            .await?;
        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(match status.as_u16() {
                400 | 404 => ImageGenError::InvalidRequest(body),
                401 | 403 => ImageGenError::AuthError(body),
                429 => ImageGenError::RateLimited { retry_after: None },
                _ => ImageGenError::ProviderError(body),
            });
        }

        let payload: FalSubmitResponse = serde_json::from_str(&body)?;
        Ok(payload.request_id)
    }

    async fn poll_until_complete(&self, endpoint: &str, request_id: &str) -> Result<()> {
        let started_at = Instant::now();
        let mut interval_ms = self.poll_interval_ms;

        loop {
            if started_at.elapsed() > Duration::from_secs(self.timeout_secs) {
                return Err(ImageGenError::Timeout {
                    elapsed_secs: self.timeout_secs,
                });
            }

            let response = self
                .client
                .get(self.status_url(endpoint, request_id))
                .header("Authorization", format!("Key {}", self.api_key))
                .send()
                .await?;
            let status = response.status();
            let body = response.text().await?;
            if !status.is_success() {
                return Err(ImageGenError::ProviderError(body));
            }
            let payload: FalStatusResponse = serde_json::from_str(&body)?;
            match payload.status.as_str() {
                "COMPLETED" => return Ok(()),
                "FAILED" => {
                    return Err(ImageGenError::ProviderError(
                        payload
                            .error
                            .unwrap_or_else(|| "FAL queue job failed".to_string()),
                    ))
                }
                _ => {
                    sleep(Duration::from_millis(interval_ms)).await;
                    interval_ms = (interval_ms * 2).min(MAX_POLL_INTERVAL_MS);
                }
            }
        }
    }

    async fn fetch_result(&self, endpoint: &str, request_id: &str) -> Result<FalResultResponse> {
        let response = self
            .client
            .get(self.result_url(endpoint, request_id))
            .header("Authorization", format!("Key {}", self.api_key))
            .send()
            .await?;
        let status = response.status();
        let body = response.text().await?;
        if !status.is_success() {
            return Err(ImageGenError::ProviderError(body));
        }
        Ok(serde_json::from_str(&body)?)
    }
}

#[async_trait]
impl ImageGenProvider for FalImageGen {
    fn name(&self) -> &str {
        "fal-ai"
    }

    fn default_model(&self) -> &str {
        &self.model
    }

    fn available_models(&self) -> Vec<&str> {
        vec![
            "fal-ai/flux/dev",
            "fal-ai/flux/schnell",
            "fal-ai/flux/pro",
            "fal-ai/stable-diffusion-xl",
            "fal-ai/aura-flow",
            "fal-ai/hyper-sdxl",
        ]
    }

    async fn generate(&self, request: &ImageGenRequest) -> Result<ImageGenResponse> {
        if request.prompt.trim().is_empty() {
            return Err(ImageGenError::InvalidRequest(
                "prompt must not be empty".to_string(),
            ));
        }

        let endpoint = self.active_model(request).to_string();
        let submit_body = self.build_request_body(request, &endpoint);
        let request_id = self.submit(&endpoint, &submit_body).await?;
        self.poll_until_complete(&endpoint, &request_id).await?;
        let payload = self.fetch_result(&endpoint, &request_id).await?;

        if payload.has_nsfw_concepts.iter().any(|value| *value) {
            return Err(ImageGenError::ContentFiltered {
                reason: "FAL safety checker flagged the generated output".to_string(),
            });
        }

        let latency_ms = payload
            .timings
            .and_then(|timings| timings.inference)
            .map(|seconds| (seconds * 1000.0) as u64)
            .unwrap_or_default();
        let images = payload
            .images
            .into_iter()
            .map(|image| GeneratedImage {
                data: ImageGenData::Url(image.url),
                width: image.width,
                height: image.height,
                mime_type: image.content_type.unwrap_or_else(|| {
                    request
                        .options
                        .output_format_or_default()
                        .mime_type()
                        .to_string()
                }),
                seed: payload.seed,
            })
            .collect::<Vec<_>>();

        if images.is_empty() {
            return Err(ImageGenError::InvalidResponse(
                "FAL result did not include any images".to_string(),
            ));
        }

        Ok(ImageGenResponse {
            images,
            provider: self.name().to_string(),
            model: endpoint,
            latency_ms,
            enhanced_prompt: payload.prompt,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::FalImageGen;
    use crate::imagegen::types::{AspectRatio, ImageGenOptions, ImageGenRequest};

    #[test]
    fn test_build_request_body_uses_ratio_and_seed() {
        let provider = FalImageGen::new("key");
        let request = ImageGenRequest::new("test").with_options(ImageGenOptions {
            aspect_ratio: Some(AspectRatio::Landscape169),
            seed: Some(42),
            ..Default::default()
        });
        let body = provider.build_request_body(&request, "fal-ai/flux/dev");
        assert_eq!(body["image_size"], "landscape_16_9");
        assert_eq!(body["seed"], 42);
    }
}
