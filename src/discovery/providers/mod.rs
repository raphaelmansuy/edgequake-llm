//! Per-provider discovery implementations.
//!
//! Each provider maps its native API response or static catalog
//! into `Vec<DiscoveredModel>`. Zero heuristics — capabilities
//! come from API responses, cited docs, or explicit "unknown".

pub mod anthropic;
#[cfg(feature = "bedrock")]
pub mod bedrock;
pub mod gemini;
pub mod lmstudio;
pub mod mistral;
pub mod nvidia;
pub mod ollama;
pub mod openai;
pub mod openai_compat;
pub mod openrouter;
pub mod xai;
