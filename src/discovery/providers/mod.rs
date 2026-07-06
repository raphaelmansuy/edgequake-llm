//! Per-provider discovery implementations.
//!
//! Each provider maps its native API response or static catalog
//! into `Vec<DiscoveredModel>`. Zero heuristics — capabilities
//! come from API responses, cited docs, or explicit "unknown".

pub mod anthropic;
#[cfg(feature = "bedrock")]
pub mod bedrock;
pub mod gemini;
pub mod gemini_model_parse;
pub mod google_vertex_auth;
pub mod lmstudio;
pub mod lmstudio_parse;
pub mod mistral;
pub mod nvidia;
pub mod ollama;
pub mod openai;
pub mod openai_compat;
pub mod openrouter;
pub mod vertexai;
pub mod xai;
