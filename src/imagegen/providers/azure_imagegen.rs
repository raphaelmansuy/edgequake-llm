//! Azure OpenAI image generation provider (gpt-image-2).
//!
//! Uses the same image generation API as OpenAI, but with Azure-specific
//! authentication (`api-key` header) and deployment-based endpoints.

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

const DEFAULT_DEPLOYMENT: &str = "gpt-image-2";
/// Empty = Foundry `/openai/v1/images/generations` (no dated api-version).
/// Set `AZURE_OPENAI_API_VERSION` for legacy deployment URLs.
const DEFAULT_API_VERSION: &str = "";

/// Azure OpenAI image generation provider.
///
/// Authenticates via the `api-key` header (Azure AD tokens are not yet
/// supported by this implementation).
#[derive(Debug, Clone)]
pub struct AzureImageGen {
    api_key: String,
    endpoint: String,
    deployment_name: String,
    /// Empty string selects Foundry v1 (`/openai/v1/...`).
    api_version: String,
    http_client: Client,
}

#[derive(Debug, Deserialize)]
struct AzureImageResponse {
    data: Vec<AzureImageItem>,
}

#[derive(Debug, Deserialize)]
struct AzureImageItem {
    #[serde(default)]
    b64_json: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    revised_prompt: Option<String>,
}

impl AzureImageGen {
    pub fn new(
        endpoint: impl Into<String>,
        api_key: impl Into<String>,
        deployment_name: impl Into<String>,
    ) -> Self {
        Self {
            api_key: api_key.into(),
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            deployment_name: deployment_name.into(),
            api_version: DEFAULT_API_VERSION.to_string(),
            http_client: Client::new(),
        }
    }

    pub fn from_env() -> Result<Self> {
        let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT")
            .or_else(|_| std::env::var("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT"))
            .map_err(|_| {
                ImageGenError::ConfigError(
                    "AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_CONTENTGEN_API_ENDPOINT must be set"
                        .to_string(),
                )
            })?;

        let api_key = std::env::var("AZURE_OPENAI_API_KEY")
            .or_else(|_| std::env::var("AZURE_OPENAI_CONTENTGEN_API_KEY"))
            .map_err(|_| {
                ImageGenError::ConfigError(
                    "AZURE_OPENAI_API_KEY or AZURE_OPENAI_CONTENTGEN_API_KEY must be set"
                        .to_string(),
                )
            })?;

        let deployment = std::env::var("AZURE_OPENAI_IMAGE_DEPLOYMENT")
            .unwrap_or_else(|_| DEFAULT_DEPLOYMENT.to_string());

        // Only override when the env var is set — empty default = Foundry v1.
        let mut provider = Self::new(endpoint, api_key, deployment);
        if let Ok(api_version) = std::env::var("AZURE_OPENAI_API_VERSION") {
            provider.api_version = api_version;
        }
        Ok(provider)
    }

    fn active_model<'a>(&'a self, request: &'a ImageGenRequest) -> &'a str {
        request.model.as_deref().unwrap_or(&self.deployment_name)
    }

    fn endpoint_url(&self) -> String {
        if self.api_version.is_empty() || self.api_version.eq_ignore_ascii_case("v1") {
            format!("{}/openai/v1/images/generations", self.endpoint)
        } else {
            format!(
                "{}/openai/deployments/{}/images/generations?api-version={}",
                self.endpoint, self.deployment_name, self.api_version
            )
        }
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
        let n = request.options.count_or_default();
        let size = self.resolve_size(request);

        let quality = request
            .options
            .extra
            .get("quality")
            .and_then(|v| v.as_str())
            .unwrap_or("auto");

        let mut body = json!({
            "prompt": request.prompt,
            "n": n,
            "size": size,
            "quality": quality,
            "response_format": "b64_json"
        });

        // Foundry v1 has no deployment in the URL — send model in the body.
        // Legacy dated deployment URLs omit model (deployment is path-scoped).
        if self.api_version.is_empty() || self.api_version.eq_ignore_ascii_case("v1") {
            body["model"] = json!(self.active_model(request));
        }

        body
    }

    fn parse_response(
        &self,
        response: AzureImageResponse,
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
impl ImageGenProvider for AzureImageGen {
    fn name(&self) -> &str {
        "azure-openai"
    }

    fn default_model(&self) -> &str {
        DEFAULT_DEPLOYMENT
    }

    fn available_models(&self) -> Vec<&str> {
        vec!["gpt-image-2"]
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

        debug!(provider = "azure-openai", model = %model, "generating image");

        let started = Instant::now();
        let response = self
            .http_client
            .post(self.endpoint_url())
            .header("api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        let response_text = response.text().await?;

        if !status.is_success() {
            warn!(provider = "azure-openai", status = %status, "image generation failed");
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
        let payload: AzureImageResponse = serde_json::from_str(&response_text)?;

        if payload.data.is_empty() {
            return Err(ImageGenError::InvalidResponse(
                "Azure OpenAI returned empty image data".to_string(),
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

    fn test_provider() -> AzureImageGen {
        AzureImageGen::new(
            "https://myresource.openai.azure.com",
            "test-key",
            "gpt-image-2",
        )
    }

    #[test]
    fn test_endpoint_url_foundry_v1_default() {
        let provider = test_provider();
        assert_eq!(
            provider.endpoint_url(),
            "https://myresource.openai.azure.com/openai/v1/images/generations"
        );
    }

    #[test]
    fn test_endpoint_url_legacy_api_version() {
        let mut provider = test_provider();
        provider.api_version = "2024-10-01-preview".to_string();
        assert_eq!(
            provider.endpoint_url(),
            "https://myresource.openai.azure.com/openai/deployments/gpt-image-2/images/generations?api-version=2024-10-01-preview"
        );
    }

    #[test]
    fn test_endpoint_url_trailing_slash() {
        let provider =
            AzureImageGen::new("https://myresource.openai.azure.com/", "key", "gpt-image-2");
        assert!(
            !provider.endpoint_url().contains("//openai"),
            "trailing slash on endpoint should be stripped"
        );
    }

    #[test]
    fn test_build_request_body_default() {
        let provider = test_provider();
        let request = ImageGenRequest::new("A sunset over mountains");
        let body = provider.build_request_body(&request);

        assert_eq!(body["prompt"], "A sunset over mountains");
        assert_eq!(body["n"], 1);
        assert_eq!(body["size"], "1024x1024");
        assert_eq!(body["quality"], "auto");
        assert_eq!(body["response_format"], "b64_json");
        // Foundry v1 default: model is required in the body.
        assert_eq!(body["model"], "gpt-image-2");
    }

    #[test]
    fn test_build_request_body_legacy_omits_model() {
        let mut provider = test_provider();
        provider.api_version = "2024-10-01-preview".to_string();
        let request = ImageGenRequest::new("A sunset over mountains");
        let body = provider.build_request_body(&request);
        assert!(
            body.get("model").is_none(),
            "legacy deployment URL scopes model in the path"
        );
    }

    #[test]
    fn test_build_request_body_landscape() {
        let provider = test_provider();
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
        let provider = test_provider();
        let request = ImageGenRequest::new("tall building").with_options(ImageGenOptions {
            aspect_ratio: Some(AspectRatio::Portrait169),
            ..Default::default()
        });
        let body = provider.build_request_body(&request);

        assert_eq!(body["size"], "1024x1792");
    }

    #[test]
    fn test_size_mapping_all_ratios() {
        assert_eq!(
            AzureImageGen::map_aspect_ratio_to_size(AspectRatio::Square),
            "1024x1024"
        );
        assert_eq!(
            AzureImageGen::map_aspect_ratio_to_size(AspectRatio::Landscape43),
            "1536x1024"
        );
        assert_eq!(
            AzureImageGen::map_aspect_ratio_to_size(AspectRatio::Portrait43),
            "1024x1536"
        );
        assert_eq!(
            AzureImageGen::map_aspect_ratio_to_size(AspectRatio::Frame54),
            "1536x1024"
        );
        assert_eq!(
            AzureImageGen::map_aspect_ratio_to_size(AspectRatio::Frame45),
            "1024x1536"
        );
    }

    #[test]
    fn test_parse_size_dimensions() {
        assert_eq!(
            AzureImageGen::parse_size_dimensions("1024x1024"),
            (1024, 1024)
        );
        assert_eq!(
            AzureImageGen::parse_size_dimensions("1792x1024"),
            (1792, 1024)
        );
        assert_eq!(
            AzureImageGen::parse_size_dimensions("1024x1792"),
            (1024, 1792)
        );
    }

    #[test]
    fn test_parse_response_b64() {
        let provider = test_provider();
        let raw = AzureImageResponse {
            data: vec![AzureImageItem {
                b64_json: Some(BASE64.encode(b"fake-azure-image")),
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
            ImageGenData::Bytes(b"fake-azure-image".to_vec())
        );
        assert_eq!(enhanced, Some("enhanced prompt".to_string()));
    }

    #[test]
    fn test_parse_response_url() {
        let provider = test_provider();
        let raw = AzureImageResponse {
            data: vec![AzureImageItem {
                b64_json: None,
                url: Some("https://blob.core.windows.net/img.png".to_string()),
                revised_prompt: None,
            }],
        };
        let (images, enhanced) = provider.parse_response(raw, "1792x1024").unwrap();

        assert_eq!(images.len(), 1);
        assert_eq!(images[0].width, 1792);
        assert_eq!(images[0].height, 1024);
        assert_eq!(
            images[0].data,
            ImageGenData::Url("https://blob.core.windows.net/img.png".to_string())
        );
        assert_eq!(enhanced, None);
    }

    #[test]
    fn test_parse_response_empty() {
        let provider = test_provider();
        let raw = AzureImageResponse { data: vec![] };
        let (images, enhanced) = provider.parse_response(raw, "1024x1024").unwrap();
        assert!(images.is_empty());
        assert_eq!(enhanced, None);
    }

    #[test]
    fn test_parse_response_no_data() {
        let provider = test_provider();
        let raw = AzureImageResponse {
            data: vec![AzureImageItem {
                b64_json: None,
                url: None,
                revised_prompt: None,
            }],
        };
        let err = provider.parse_response(raw, "1024x1024").unwrap_err();
        assert!(err.to_string().contains("neither b64_json nor url"));
    }

    #[test]
    fn test_available_models() {
        let provider = test_provider();
        let models = provider.available_models();
        assert!(models.contains(&"gpt-image-2"));
    }

    #[test]
    fn test_name() {
        let provider = test_provider();
        assert_eq!(provider.name(), "azure-openai");
    }
}
