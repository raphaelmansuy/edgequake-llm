use std::path::Path;
use std::time::Instant;

use async_trait::async_trait;
use base64::Engine;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::imagegen::error::{ImageGenError, Result};
use crate::imagegen::providers::gcp::{access_token_from_env_or_gcloud, env_region};
use crate::imagegen::traits::ImageGenProvider;
use crate::imagegen::types::{
    AspectRatio, GeneratedImage, ImageFormat, ImageGenData, ImageGenRequest, ImageGenResponse,
};

const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_GEMINI_IMAGE_MODEL: &str = "gemini-2.5-flash-image";

#[derive(Debug, Clone)]
enum GeminiImageEndpoint {
    GoogleAI {
        api_key: String,
    },
    VertexAI {
        project_id: String,
        region: String,
        access_token: String,
    },
}

/// Gemini image generation provider supporting both Google AI and Vertex AI endpoints.
#[derive(Debug, Clone)]
pub struct GeminiImageGenProvider {
    client: Client,
    endpoint: GeminiImageEndpoint,
    model: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerateRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiContent {
    parts: Vec<GeminiPart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inline_data: Option<GeminiBlob>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiBlob {
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    response_modalities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_config: Option<GeminiImageConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GeminiThinkingConfig>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiImageConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_size: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiThinkingConfig {
    thinking_level: String,
    include_thoughts: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerateResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    prompt_feedback: Option<GeminiPromptFeedback>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    #[serde(default)]
    content: Option<GeminiResponseContent>,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponseContent {
    #[serde(default)]
    parts: Vec<GeminiResponsePart>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponsePart {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    inline_data: Option<GeminiResponseBlob>,
    #[serde(default)]
    thought: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponseBlob {
    mime_type: String,
    data: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiPromptFeedback {
    #[serde(default)]
    block_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiErrorEnvelope {
    error: GeminiErrorBody,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiErrorBody {
    #[allow(dead_code)]
    code: i32,
    message: String,
    #[allow(dead_code)]
    status: String,
}

impl GeminiImageGenProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            endpoint: GeminiImageEndpoint::GoogleAI {
                api_key: api_key.into(),
            },
            model: DEFAULT_GEMINI_IMAGE_MODEL.to_string(),
        }
    }

    pub fn vertex_ai(
        project_id: impl Into<String>,
        region: impl Into<String>,
        access_token: impl Into<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            endpoint: GeminiImageEndpoint::VertexAI {
                project_id: project_id.into(),
                region: region.into(),
                access_token: access_token.into(),
            },
            model: DEFAULT_GEMINI_IMAGE_MODEL.to_string(),
        }
    }

    pub fn from_env() -> Result<Self> {
        if let Ok(api_key) = std::env::var("GEMINI_API_KEY") {
            return Ok(Self::new(api_key));
        }

        Self::from_env_vertex_ai()
    }

    pub fn from_env_vertex_ai() -> Result<Self> {
        let project_id = std::env::var("GOOGLE_CLOUD_PROJECT").map_err(|_| {
            ImageGenError::ConfigError(
                "Vertex AI image generation requires GOOGLE_CLOUD_PROJECT".to_string(),
            )
        })?;

        let access_token = access_token_from_env_or_gcloud()?;
        Ok(Self::vertex_ai(project_id, env_region(), access_token))
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    fn active_model<'a>(&'a self, request: &'a ImageGenRequest) -> &'a str {
        request.model.as_deref().unwrap_or(&self.model)
    }

    fn vertex_host(region: &str) -> String {
        if region == "global" {
            "aiplatform.googleapis.com".to_string()
        } else {
            format!("{region}-aiplatform.googleapis.com")
        }
    }

    fn requires_global_region(model: &str) -> bool {
        model.contains("3.1-") || model.contains("3-pro") || model.ends_with("-preview")
    }

    fn build_url(&self, model: &str) -> String {
        match &self.endpoint {
            GeminiImageEndpoint::GoogleAI { api_key } => {
                format!("{GEMINI_API_BASE}/models/{model}:generateContent?key={api_key}")
            }
            GeminiImageEndpoint::VertexAI {
                project_id, region, ..
            } => {
                let effective_region = if Self::requires_global_region(model) {
                    "global"
                } else {
                    region.as_str()
                };
                let host = Self::vertex_host(effective_region);
                format!(
                    "https://{host}/v1/projects/{project_id}/locations/{effective_region}/publishers/google/models/{model}:generateContent"
                )
            }
        }
    }

    fn validate_request(&self, request: &ImageGenRequest, model: &str) -> Result<()> {
        if request.prompt.trim().is_empty() {
            return Err(ImageGenError::InvalidRequest(
                "prompt must not be empty".to_string(),
            ));
        }

        if request.options.count_or_default() != 1 {
            return Err(ImageGenError::NotSupported(
                "Gemini image generation currently supports exactly one output image per request"
                    .to_string(),
            ));
        }

        let aspect = request.options.aspect_ratio_or_default();
        let extreme_ratio = matches!(
            aspect,
            AspectRatio::Extreme41
                | AspectRatio::Extreme14
                | AspectRatio::Extreme81
                | AspectRatio::Extreme18
        );
        if extreme_ratio && model != "gemini-3.1-flash-image-preview" {
            return Err(ImageGenError::InvalidRequest(format!(
                "aspect ratio {} requires gemini-3.1-flash-image-preview",
                aspect.as_gemini_str()
            )));
        }

        Ok(())
    }

    async fn build_parts(&self, request: &ImageGenRequest) -> Result<Vec<GeminiPart>> {
        let mut parts = Vec::new();
        for reference in &request.options.reference_images {
            let (mime_type, data) = self.load_reference_image(reference).await?;
            parts.push(GeminiPart {
                text: None,
                inline_data: Some(GeminiBlob { mime_type, data }),
            });
        }
        parts.push(GeminiPart {
            text: Some(request.prompt.clone()),
            inline_data: None,
        });
        Ok(parts)
    }

    async fn load_reference_image(&self, reference: &str) -> Result<(String, String)> {
        if let Some((mime_type, data)) = parse_data_uri(reference) {
            return Ok((mime_type, data));
        }

        if reference.starts_with("http://") || reference.starts_with("https://") {
            let response = self.client.get(reference).send().await?;
            let headers = response.headers().clone();
            let bytes = response.bytes().await?;
            let mime_type = headers
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.split(';').next())
                .unwrap_or_else(|| infer_mime_type(reference))
                .to_string();
            let data = base64::engine::general_purpose::STANDARD.encode(bytes);
            return Ok((mime_type, data));
        }

        let path = Path::new(reference);
        if path.exists() {
            let bytes = tokio::fs::read(path).await.map_err(|err| {
                ImageGenError::InvalidRequest(format!("failed to read {reference}: {err}"))
            })?;
            let data = base64::engine::general_purpose::STANDARD.encode(bytes);
            return Ok((infer_mime_type(reference).to_string(), data));
        }

        Err(ImageGenError::InvalidRequest(format!(
            "unsupported reference image source: {reference}"
        )))
    }

    async fn parse_response(
        &self,
        request: &ImageGenRequest,
        model: &str,
        response: reqwest::Response,
        started_at: Instant,
    ) -> Result<ImageGenResponse> {
        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            let parsed = serde_json::from_str::<GeminiErrorEnvelope>(&body).ok();
            let message = parsed
                .map(|payload| payload.error.message)
                .unwrap_or(body.clone());
            return Err(match status.as_u16() {
                400 => ImageGenError::InvalidRequest(message),
                401 => ImageGenError::AuthError(message),
                403 => ImageGenError::AuthError(message),
                429 => ImageGenError::RateLimited { retry_after: None },
                500..=599 => ImageGenError::ProviderError(message),
                _ => ImageGenError::ProviderError(message),
            });
        }

        let payload: GeminiGenerateResponse = serde_json::from_str(&body)?;
        if payload.candidates.is_empty() {
            if let Some(feedback) = payload.prompt_feedback {
                if let Some(reason) = feedback.block_reason {
                    return Err(ImageGenError::ContentFiltered { reason });
                }
            }
            return Err(ImageGenError::InvalidResponse(
                "Gemini returned no candidates".to_string(),
            ));
        }

        let mut images = Vec::new();
        let mut text_fragments = Vec::new();
        let (width, height) = request
            .options
            .aspect_ratio_or_default()
            .default_dimensions();

        for candidate in payload.candidates {
            if let Some(content) = candidate.content {
                for part in content.parts {
                    if part.thought.unwrap_or(false) {
                        continue;
                    }
                    if let Some(text) = part.text {
                        text_fragments.push(text);
                    }
                    if let Some(blob) = part.inline_data {
                        let bytes = base64::engine::general_purpose::STANDARD.decode(blob.data)?;
                        images.push(GeneratedImage {
                            data: ImageGenData::Bytes(bytes),
                            width,
                            height,
                            mime_type: blob.mime_type,
                            seed: request.options.seed,
                        });
                    }
                }
            }
        }

        if images.is_empty() {
            return Err(ImageGenError::InvalidResponse(
                "Gemini response did not include an image".to_string(),
            ));
        }

        Ok(ImageGenResponse {
            images,
            provider: self.name().to_string(),
            model: model.to_string(),
            latency_ms: started_at.elapsed().as_millis() as u64,
            enhanced_prompt: if text_fragments.is_empty() {
                None
            } else {
                Some(text_fragments.join(""))
            },
        })
    }
}

#[async_trait]
impl ImageGenProvider for GeminiImageGenProvider {
    fn name(&self) -> &str {
        match self.endpoint {
            GeminiImageEndpoint::GoogleAI { .. } => "gemini-image",
            GeminiImageEndpoint::VertexAI { .. } => "vertexai-gemini-image",
        }
    }

    fn default_model(&self) -> &str {
        &self.model
    }

    fn available_models(&self) -> Vec<&str> {
        vec![
            "gemini-2.5-flash-image",
            "gemini-3.1-flash-image-preview",
            "gemini-3-pro-image-preview",
        ]
    }

    async fn generate(&self, request: &ImageGenRequest) -> Result<ImageGenResponse> {
        let model = self.active_model(request).to_string();
        self.validate_request(request, &model)?;

        let started_at = Instant::now();
        let response_modalities = vec!["IMAGE".to_string()];
        let aspect_ratio = request
            .options
            .aspect_ratio
            .filter(|ratio| *ratio != AspectRatio::Auto);
        let body = GeminiGenerateRequest {
            contents: vec![GeminiContent {
                parts: self.build_parts(request).await?,
                role: Some("user".to_string()),
            }],
            generation_config: Some(GeminiGenerationConfig {
                response_modalities,
                image_config: Some(GeminiImageConfig {
                    aspect_ratio: aspect_ratio.map(|ratio| ratio.as_gemini_str().to_string()),
                    image_size: Some(
                        request
                            .options
                            .resolution_or_default()
                            .as_gemini_str()
                            .to_string(),
                    ),
                }),
                thinking_config: request.options.thinking_level.map(|thinking| {
                    GeminiThinkingConfig {
                        thinking_level: thinking.as_gemini_api_label().to_string(),
                        include_thoughts: false,
                    }
                }),
            }),
            tools: request
                .options
                .enable_web_search
                .filter(|enabled| *enabled)
                .map(|_| vec![json!({ "googleSearch": {} })]),
        };

        let mut builder = self.client.post(self.build_url(&model)).json(&body);
        if let GeminiImageEndpoint::VertexAI { access_token, .. } = &self.endpoint {
            builder = builder.bearer_auth(access_token);
        }
        let response = builder.send().await?;
        self.parse_response(request, &model, response, started_at)
            .await
    }
}

fn parse_data_uri(value: &str) -> Option<(String, String)> {
    let rest = value.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    let mime_type = meta.strip_suffix(";base64").unwrap_or(meta).to_string();
    Some((mime_type, data.to_string()))
}

fn infer_mime_type(path_or_url: &str) -> &'static str {
    let path = path_or_url.split('?').next().unwrap_or(path_or_url);
    match Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => "image/png",
        Some("webp") => "image/webp",
        Some("gif") => "image/gif",
        _ => ImageFormat::Jpeg.mime_type(),
    }
}

#[cfg(test)]
mod tests {
    use super::{infer_mime_type, parse_data_uri, GeminiImageGenProvider};

    #[test]
    fn test_parse_data_uri() {
        let parsed = parse_data_uri("data:image/png;base64,abc123").unwrap();
        assert_eq!(parsed.0, "image/png");
        assert_eq!(parsed.1, "abc123");
    }

    #[test]
    fn test_infer_mime_type() {
        assert_eq!(infer_mime_type("https://example.com/a.webp"), "image/webp");
        assert_eq!(infer_mime_type("/tmp/a.png"), "image/png");
        assert_eq!(infer_mime_type("/tmp/a.jpg"), "image/jpeg");
    }

    #[test]
    fn test_vertex_url_for_global_preview_models() {
        let provider = GeminiImageGenProvider::vertex_ai("proj", "europe-west1", "token")
            .with_model("gemini-3.1-flash-image-preview");
        let url = provider.build_url("gemini-3.1-flash-image-preview");
        assert!(url.contains("/locations/global/"));
        assert!(url.contains("aiplatform.googleapis.com"));
    }
}
