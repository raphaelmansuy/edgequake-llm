//! Image generation provider implementations.

mod gcp;

pub mod azure_imagegen;
#[cfg(feature = "bedrock")]
pub mod bedrock_stability;
pub mod fal;
pub mod gemini;
pub mod mock;
pub mod nvidia_imagegen;
pub mod openai_imagegen;
pub mod vertexai;
pub mod xai_imagegen;

pub use azure_imagegen::AzureImageGen;
#[cfg(feature = "bedrock")]
pub use bedrock_stability::BedrockStabilityImageGen;
pub use fal::FalImageGen;
pub use gemini::GeminiImageGenProvider;
pub use mock::MockImageGenProvider;
pub use nvidia_imagegen::NvidiaImageGen;
pub use openai_imagegen::OpenAIImageGen;
pub use vertexai::VertexAIImageGen;
pub use xai_imagegen::XAIImageGen;
