//! Shared types for image generation.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Named aspect ratio presets shared across providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AspectRatio {
    #[default]
    Auto,
    Square,
    SquareHd,
    Landscape43,
    Landscape169,
    Ultrawide,
    Portrait43,
    Portrait169,
    Frame54,
    Frame45,
    Print32,
    Print23,
    Extreme41,
    Extreme14,
    Extreme81,
    Extreme18,
}

impl AspectRatio {
    /// Default pixel size used when the provider does not return explicit dimensions.
    pub fn default_dimensions(self) -> (u32, u32) {
        match self {
            Self::Auto => (1024, 1024),
            Self::Square => (1024, 1024),
            Self::SquareHd => (2048, 2048),
            Self::Landscape43 => (1024, 768),
            Self::Landscape169 => (1280, 720),
            Self::Ultrawide => (2560, 1080),
            Self::Portrait43 => (768, 1024),
            Self::Portrait169 => (720, 1280),
            Self::Frame54 => (1280, 1024),
            Self::Frame45 => (1024, 1280),
            Self::Print32 => (1152, 768),
            Self::Print23 => (768, 1152),
            Self::Extreme41 => (2048, 512),
            Self::Extreme14 => (512, 2048),
            Self::Extreme81 => (3072, 384),
            Self::Extreme18 => (384, 3072),
        }
    }

    /// Vertex Imagen ratio name.
    pub fn as_vertex_str(self) -> &'static str {
        match self {
            Self::Auto | Self::Square | Self::SquareHd => "1:1",
            Self::Landscape43 => "4:3",
            Self::Landscape169 | Self::Ultrawide => "16:9",
            Self::Portrait43 => "3:4",
            Self::Portrait169 => "9:16",
            Self::Frame54 => "5:4",
            Self::Frame45 => "4:5",
            Self::Print32 => "3:2",
            Self::Print23 => "2:3",
            Self::Extreme41 => "4:1",
            Self::Extreme14 => "1:4",
            Self::Extreme81 => "8:1",
            Self::Extreme18 => "1:8",
        }
    }

    /// FAL image size preset.
    pub fn as_fal_str(self) -> &'static str {
        match self {
            Self::Auto | Self::Square => "square",
            Self::SquareHd => "square_hd",
            Self::Landscape43 | Self::Frame54 | Self::Print32 => "landscape_4_3",
            Self::Landscape169 | Self::Ultrawide | Self::Extreme41 | Self::Extreme81 => {
                "landscape_16_9"
            }
            Self::Portrait43 | Self::Frame45 | Self::Print23 => "portrait_4_3",
            Self::Portrait169 | Self::Extreme14 | Self::Extreme18 => "portrait_16_9",
        }
    }

    /// Gemini image ratio name.
    pub fn as_gemini_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Square | Self::SquareHd => "1:1",
            Self::Landscape43 => "4:3",
            Self::Landscape169 => "16:9",
            Self::Ultrawide => "21:9",
            Self::Portrait43 => "3:4",
            Self::Portrait169 => "9:16",
            Self::Frame54 => "5:4",
            Self::Frame45 => "4:5",
            Self::Print32 => "3:2",
            Self::Print23 => "2:3",
            Self::Extreme41 => "4:1",
            Self::Extreme14 => "1:4",
            Self::Extreme81 => "8:1",
            Self::Extreme18 => "1:8",
        }
    }
}

/// Output image format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
    #[default]
    Jpeg,
    Png,
    Webp,
}

impl ImageFormat {
    pub fn mime_type(self) -> &'static str {
        match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
            Self::Webp => "image/webp",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::Jpeg => "jpg",
            Self::Png => "png",
            Self::Webp => "webp",
        }
    }

    pub fn as_fal_str(self) -> &'static str {
        match self {
            Self::Jpeg => "jpeg",
            Self::Png => "png",
            Self::Webp => "webp",
        }
    }
}

/// Provider-agnostic content safety level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SafetyLevel {
    BlockNone,
    BlockLow,
    #[default]
    BlockMedium,
    BlockHigh,
}

/// Gemini image size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ImageResolution {
    Half,
    #[default]
    OneK,
    TwoK,
    FourK,
}

impl ImageResolution {
    pub fn as_gemini_str(self) -> &'static str {
        match self {
            Self::Half => "512",
            Self::OneK => "1K",
            Self::TwoK => "2K",
            Self::FourK => "4K",
        }
    }

    pub fn as_nano_banana_str(self) -> &'static str {
        match self {
            Self::Half => "0.5K",
            Self::OneK => "1K",
            Self::TwoK => "2K",
            Self::FourK => "4K",
        }
    }
}

/// Gemini reasoning depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingLevel {
    #[default]
    Minimal,
    High,
}

impl ThinkingLevel {
    pub fn as_gemini_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::High => "high",
        }
    }

    pub fn as_gemini_api_label(self) -> &'static str {
        match self {
            Self::Minimal => "Minimal",
            Self::High => "High",
        }
    }
}

/// Shared provider options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImageGenOptions {
    pub count: Option<u8>,
    pub aspect_ratio: Option<AspectRatio>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub seed: Option<u64>,
    pub negative_prompt: Option<String>,
    pub guidance_scale: Option<f32>,
    pub output_format: Option<ImageFormat>,
    pub safety_level: Option<SafetyLevel>,
    pub enhance_prompt: Option<bool>,
    pub resolution: Option<ImageResolution>,
    pub enable_web_search: Option<bool>,
    pub thinking_level: Option<ThinkingLevel>,
    #[serde(default)]
    pub reference_images: Vec<String>,
    #[serde(default)]
    pub extra: HashMap<String, JsonValue>,
}

impl ImageGenOptions {
    pub fn aspect_ratio_or_default(&self) -> AspectRatio {
        self.aspect_ratio.unwrap_or_default()
    }

    pub fn count_or_default(&self) -> u8 {
        self.count.unwrap_or(1)
    }

    pub fn output_format_or_default(&self) -> ImageFormat {
        self.output_format.unwrap_or_default()
    }

    pub fn safety_level_or_default(&self) -> SafetyLevel {
        self.safety_level.unwrap_or_default()
    }

    pub fn resolution_or_default(&self) -> ImageResolution {
        self.resolution.unwrap_or_default()
    }
}

/// Image generation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenRequest {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default)]
    pub options: ImageGenOptions,
}

impl ImageGenRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            model: None,
            options: ImageGenOptions::default(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_options(mut self, options: ImageGenOptions) -> Self {
        self.options = options;
        self
    }
}

/// Generated image payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageGenData {
    Bytes(Vec<u8>),
    Url(String),
}

/// Single generated image.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedImage {
    pub data: ImageGenData,
    pub width: u32,
    pub height: u32,
    pub mime_type: String,
    pub seed: Option<u64>,
}

/// Image generation response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenResponse {
    pub images: Vec<GeneratedImage>,
    pub provider: String,
    pub model: String,
    pub latency_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enhanced_prompt: Option<String>,
}
