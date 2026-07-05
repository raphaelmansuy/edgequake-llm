//! Unified provider catalog — single source of truth for provider metadata.
//!
//! # Design
//!
//! - **SRP**: Catalog describes providers; [`crate::factory::ProviderFactory`] creates them.
//! - **DRY**: All list/resolve APIs delegate here (Rust, Python, docs).
//! - **OCP**: New providers add one [`ProviderDescriptor`] entry.

use crate::factory::ProviderType;

/// Feature flags exposed by a provider integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProviderFeatures {
    /// Chat / completion API.
    pub chat: bool,
    /// Text embedding API.
    pub embedding: bool,
    /// Model discovery (static, dynamic, or hybrid).
    pub model_discovery: bool,
    /// Image generation API.
    pub image_generation: bool,
}

/// Metadata for one provider integration.
#[derive(Debug, Clone, Copy)]
pub struct ProviderDescriptor {
    /// Canonical lowercase ID (e.g. `"openai"`).
    pub id: &'static str,
    /// Alternate names accepted by [`ProviderCatalog::resolve_id`].
    pub aliases: &'static [&'static str],
    /// Supported surfaces.
    pub features: ProviderFeatures,
    /// Factory enum variant when this provider is constructible via [`ProviderType`].
    pub provider_type: Option<ProviderType>,
}

/// Read-only catalog of all built-in provider integrations.
pub struct ProviderCatalog;

impl ProviderCatalog {
    /// All registered provider descriptors (chat, embed-only, and image-only).
    pub fn all() -> &'static [ProviderDescriptor] {
        ALL_DESCRIPTORS
    }

    /// Canonical IDs for providers with chat/completion support.
    pub fn list_llm_providers() -> Vec<&'static str> {
        Self::all()
            .iter()
            .filter(|d| d.features.chat)
            .map(|d| d.id)
            .collect()
    }

    /// Canonical IDs for providers with embedding support.
    pub fn list_embedding_providers() -> Vec<&'static str> {
        Self::all()
            .iter()
            .filter(|d| d.features.embedding)
            .map(|d| d.id)
            .collect()
    }

    /// Canonical IDs for providers registered in the model discovery service.
    pub fn list_discovery_providers() -> Vec<&'static str> {
        Self::all()
            .iter()
            .filter(|d| d.features.model_discovery)
            .map(|d| d.id)
            .collect()
    }

    /// Canonical IDs for image generation providers.
    pub fn list_imagegen_providers() -> Vec<&'static str> {
        Self::all()
            .iter()
            .filter(|d| d.features.image_generation)
            .map(|d| d.id)
            .collect()
    }

    /// Resolve a user-supplied name (ID or alias) to a canonical provider ID.
    pub fn resolve_id(input: &str) -> Option<&'static str> {
        let needle = input.to_lowercase();
        Self::all().iter().find_map(|d| {
            if d.id.eq_ignore_ascii_case(&needle) {
                return Some(d.id);
            }
            d.aliases
                .iter()
                .find(|a| a.eq_ignore_ascii_case(&needle))
                .map(|_| d.id)
        })
    }

    /// Look up a descriptor by canonical ID.
    pub fn get(id: &str) -> Option<&'static ProviderDescriptor> {
        Self::all().iter().find(|d| d.id.eq_ignore_ascii_case(id))
    }
}

const CHAT_EMBED_DISCOVERY: ProviderFeatures = ProviderFeatures {
    chat: true,
    embedding: true,
    model_discovery: true,
    image_generation: false,
};

const CHAT_EMBED: ProviderFeatures = ProviderFeatures {
    chat: true,
    embedding: true,
    model_discovery: false,
    image_generation: false,
};

const CHAT_EMBED_IMAGE: ProviderFeatures = ProviderFeatures {
    chat: true,
    embedding: true,
    model_discovery: false,
    image_generation: true,
};

const CHAT_EMBED_DISCOVERY_IMAGE: ProviderFeatures = ProviderFeatures {
    chat: true,
    embedding: true,
    model_discovery: true,
    image_generation: true,
};

const CHAT_ONLY_DISCOVERY: ProviderFeatures = ProviderFeatures {
    chat: true,
    embedding: false,
    model_discovery: true,
    image_generation: false,
};

const CHAT_ONLY: ProviderFeatures = ProviderFeatures {
    chat: true,
    embedding: false,
    model_discovery: false,
    image_generation: false,
};

const EMBED_ONLY: ProviderFeatures = ProviderFeatures {
    chat: false,
    embedding: true,
    model_discovery: false,
    image_generation: false,
};

const IMAGE_ONLY: ProviderFeatures = ProviderFeatures {
    chat: false,
    embedding: false,
    model_discovery: false,
    image_generation: true,
};

static ALL_DESCRIPTORS: &[ProviderDescriptor] = &[
    ProviderDescriptor {
        id: "openai",
        aliases: &[],
        features: ProviderFeatures {
            chat: true,
            embedding: true,
            model_discovery: true,
            image_generation: true,
        },
        provider_type: Some(ProviderType::OpenAI),
    },
    ProviderDescriptor {
        id: "anthropic",
        aliases: &["claude"],
        features: CHAT_ONLY_DISCOVERY,
        provider_type: Some(ProviderType::Anthropic),
    },
    ProviderDescriptor {
        id: "gemini",
        aliases: &["google"],
        features: CHAT_EMBED_DISCOVERY_IMAGE,
        provider_type: Some(ProviderType::Gemini),
    },
    ProviderDescriptor {
        id: "vertexai",
        aliases: &["vertex"],
        features: CHAT_EMBED_IMAGE,
        provider_type: Some(ProviderType::VertexAI),
    },
    ProviderDescriptor {
        id: "openrouter",
        aliases: &["open-router"],
        features: CHAT_ONLY_DISCOVERY,
        provider_type: Some(ProviderType::OpenRouter),
    },
    ProviderDescriptor {
        id: "xai",
        aliases: &["grok"],
        features: ProviderFeatures {
            chat: true,
            embedding: false,
            model_discovery: true,
            image_generation: true,
        },
        provider_type: Some(ProviderType::XAI),
    },
    ProviderDescriptor {
        id: "huggingface",
        aliases: &["hf", "hugging-face", "hugging_face"],
        features: CHAT_ONLY,
        provider_type: Some(ProviderType::HuggingFace),
    },
    ProviderDescriptor {
        id: "openai-compatible",
        aliases: &["openai_compatible", "openaicompatible", "compatible"],
        features: CHAT_EMBED,
        provider_type: Some(ProviderType::OpenAICompatible),
    },
    ProviderDescriptor {
        id: "ollama",
        aliases: &[],
        features: CHAT_EMBED_DISCOVERY,
        provider_type: Some(ProviderType::Ollama),
    },
    ProviderDescriptor {
        id: "lmstudio",
        aliases: &["lm-studio", "lm_studio"],
        features: CHAT_EMBED_DISCOVERY,
        provider_type: Some(ProviderType::LMStudio),
    },
    ProviderDescriptor {
        id: "vscode-copilot",
        aliases: &["vscode", "copilot"],
        features: CHAT_EMBED,
        provider_type: Some(ProviderType::VsCodeCopilot),
    },
    ProviderDescriptor {
        id: "mistral",
        aliases: &["mistral-ai", "mistralai"],
        features: CHAT_EMBED_DISCOVERY,
        provider_type: Some(ProviderType::Mistral),
    },
    ProviderDescriptor {
        id: "azure",
        aliases: &["azure-openai", "azure_openai", "azureopenai"],
        features: CHAT_EMBED_IMAGE,
        provider_type: Some(ProviderType::AzureOpenAI),
    },
    ProviderDescriptor {
        id: "nvidia",
        aliases: &["nvidia-nim", "nim"],
        features: ProviderFeatures {
            chat: true,
            embedding: true,
            model_discovery: true,
            image_generation: true,
        },
        provider_type: Some(ProviderType::Nvidia),
    },
    ProviderDescriptor {
        id: "cohere",
        aliases: &["cohere-ai"],
        features: CHAT_EMBED,
        provider_type: Some(ProviderType::Cohere),
    },
    ProviderDescriptor {
        id: "mock",
        aliases: &[],
        features: CHAT_EMBED,
        provider_type: Some(ProviderType::Mock),
    },
    ProviderDescriptor {
        id: "jina",
        aliases: &[],
        features: EMBED_ONLY,
        provider_type: None,
    },
    #[cfg(feature = "bedrock")]
    ProviderDescriptor {
        id: "bedrock",
        aliases: &["aws-bedrock", "aws_bedrock"],
        features: ProviderFeatures {
            chat: true,
            embedding: true,
            model_discovery: true,
            image_generation: false,
        },
        provider_type: Some(ProviderType::Bedrock),
    },
    // Image-generation-only runtime surfaces (distinct from chat factory IDs)
    ProviderDescriptor {
        id: "gemini-image",
        aliases: &[],
        features: IMAGE_ONLY,
        provider_type: None,
    },
    ProviderDescriptor {
        id: "vertexai-gemini-image",
        aliases: &[],
        features: IMAGE_ONLY,
        provider_type: None,
    },
    ProviderDescriptor {
        id: "vertexai-imagen",
        aliases: &[],
        features: IMAGE_ONLY,
        provider_type: None,
    },
    ProviderDescriptor {
        id: "fal-ai",
        aliases: &["fal"],
        features: IMAGE_ONLY,
        provider_type: None,
    },
    #[cfg(feature = "bedrock")]
    ProviderDescriptor {
        id: "bedrock-stability",
        aliases: &[],
        features: IMAGE_ONLY,
        provider_type: None,
    },
    ProviderDescriptor {
        id: "mock-imagegen",
        aliases: &[],
        features: IMAGE_ONLY,
        provider_type: None,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_llm_includes_major_providers() {
        let ids = ProviderCatalog::list_llm_providers();
        for expected in [
            "openai",
            "anthropic",
            "gemini",
            "mistral",
            "nvidia",
            "cohere",
        ] {
            assert!(ids.contains(&expected), "missing LLM provider: {expected}");
        }
    }

    #[test]
    fn list_discovery_matches_service_defaults() {
        let ids = ProviderCatalog::list_discovery_providers();
        for expected in [
            "openai",
            "anthropic",
            "gemini",
            "ollama",
            "lmstudio",
            "openrouter",
            "mistral",
            "nvidia",
            "xai",
        ] {
            assert!(
                ids.contains(&expected),
                "missing discovery provider: {expected}"
            );
        }
    }

    #[test]
    fn resolve_aliases() {
        assert_eq!(ProviderCatalog::resolve_id("claude"), Some("anthropic"));
        assert_eq!(ProviderCatalog::resolve_id("AZURE"), Some("azure"));
        assert_eq!(ProviderCatalog::resolve_id("lm-studio"), Some("lmstudio"));
        assert_eq!(ProviderCatalog::resolve_id("unknown-xyz"), None);
    }

    #[test]
    fn jina_is_embed_only() {
        let jina = ProviderCatalog::get("jina").unwrap();
        assert!(!jina.features.chat);
        assert!(jina.features.embedding);
    }
}
