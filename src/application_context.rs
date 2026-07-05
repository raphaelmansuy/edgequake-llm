//! Application identity and attribution context for LLM provider calls.
//!
//! Single semantic type at the SDK boundary; provider-specific HTTP headers and
//! body fields are resolved via [`crate::http::attribution`].

use std::collections::HashMap;

use crate::error::{LlmError, Result};

/// Provider family for attribution resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributionProviderKind {
    OpenAI,
    AzureOpenAI,
    Anthropic,
    Gemini,
    VertexAI,
    OpenRouter,
    OpenAICompatible,
    Mistral,
    Nvidia,
    Cohere,
    Bedrock,
    XAI,
    HuggingFace,
    LMStudio,
    Ollama,
    VsCodeCopilot,
    Mock,
}

/// How strictly attribution must propagate to the upstream provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttributionPolicy {
    /// Propagate what the provider supports; emit warnings for gaps.
    #[default]
    BestEffort,
    /// Fail when `app_id` is set but cannot reach the provider.
    RequireAppId,
    /// Disable attribution injection (testing).
    Disabled,
}

/// Non-fatal attribution issues surfaced to callers and tracing spans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttributionWarning {
    /// OpenRouter requires `HTTP-Referer` for app leaderboard attribution.
    OpenRouterMissingReferer,
    /// Localhost referer without a display title.
    OpenRouterLocalhostMissingTitle,
    /// Custom header name or value was invalid and skipped.
    InvalidHeader { name: String, reason: String },
    /// Provider does not support application attribution headers.
    ProviderUnsupported { provider: String },
    /// Approaching Azure custom header passthrough limit (~10).
    AzureHeaderBudget { count: usize },
}

/// Caller-supplied identity for attribution and request correlation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ApplicationContext {
    /// Stable application identifier (slug). Primary attribution key.
    pub app_id: Option<String>,
    /// Human-readable name for provider dashboards.
    pub app_name: Option<String>,
    /// Public URL of the application (OpenRouter `HTTP-Referer`).
    pub app_url: Option<String>,
    /// Multi-tenant partition.
    pub tenant_id: Option<String>,
    /// Per-request correlation ID (OpenAI `X-Client-Request-Id`, etc.).
    pub request_id: Option<String>,
    /// End-user identifier for provider safety/billing fields (OpenAI `user`, Azure `end_user_id`).
    pub end_user_id: Option<String>,
    /// Additional passthrough headers (`traceparent`, vendor-specific).
    pub extra_headers: HashMap<String, String>,
}

impl ApplicationContext {
    /// Create an empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse canonical ingress headers into a validated context.
    pub fn from_ingress_headers(headers: &HashMap<String, String>) -> Result<Self> {
        let mut ctx = Self::new();
        if let Some(v) = headers.get("x-edgequake-app-id") {
            ctx.app_id = Some(sanitize_app_id(v)?);
        }
        if let Some(v) = headers.get("x-edgequake-app-name") {
            ctx.app_name = Some(sanitize_app_name(v)?);
        }
        if let Some(v) = headers.get("x-edgequake-app-url") {
            ctx.app_url = Some(sanitize_app_url(v)?);
        }
        if let Some(v) = headers.get("x-edgequake-tenant-id") {
            ctx.tenant_id = Some(sanitize_tenant_id(v)?);
        }
        if let Some(v) = headers.get("x-edgequake-request-id") {
            ctx.request_id = Some(sanitize_request_id(v)?);
        }
        for (k, v) in headers {
            let lower = k.to_ascii_lowercase();
            if lower.starts_with("x-edgequake-") {
                continue;
            }
            if lower == "traceparent" || lower == "tracestate" || lower == "baggage" {
                ctx.extra_headers.insert(k.clone(), v.clone());
            }
        }
        Ok(ctx)
    }

    /// Load defaults from environment (`EDGEQUAKE_APP_*`).
    pub fn from_env() -> Self {
        let mut ctx = Self::new();
        if let Ok(v) = std::env::var("EDGEQUAKE_APP_ID") {
            if let Ok(id) = sanitize_app_id(&v) {
                ctx.app_id = Some(id);
            }
        }
        if let Ok(v) = std::env::var("EDGEQUAKE_APP_NAME") {
            if let Ok(name) = sanitize_app_name(&v) {
                ctx.app_name = Some(name);
            }
        }
        if let Ok(v) = std::env::var("EDGEQUAKE_APP_URL") {
            if let Ok(url) = sanitize_app_url(&v) {
                ctx.app_url = Some(url);
            }
        }
        if let Ok(v) = std::env::var("EDGEQUAKE_TENANT_ID") {
            if let Ok(tid) = sanitize_tenant_id(&v) {
                ctx.tenant_id = Some(tid);
            }
        }
        ctx
    }

    /// Merge `other` into `self`. Per-field and extra_headers from `other` win.
    pub fn merge(&mut self, other: ApplicationContext) {
        if other.app_id.is_some() {
            self.app_id = other.app_id;
        }
        if other.app_name.is_some() {
            self.app_name = other.app_name;
        }
        if other.app_url.is_some() {
            self.app_url = other.app_url;
        }
        if other.tenant_id.is_some() {
            self.tenant_id = other.tenant_id;
        }
        if other.request_id.is_some() {
            self.request_id = other.request_id;
        }
        if other.end_user_id.is_some() {
            self.end_user_id = other.end_user_id;
        }
        self.extra_headers.extend(other.extra_headers);
    }

    /// Whether any attribution or correlation field is set.
    pub fn is_empty(&self) -> bool {
        self.app_id.is_none()
            && self.app_name.is_none()
            && self.app_url.is_none()
            && self.tenant_id.is_none()
            && self.request_id.is_none()
            && self.end_user_id.is_none()
            && self.extra_headers.is_empty()
    }

    /// Whether application-level attribution (not just request_id) is present.
    pub fn has_app_attribution(&self) -> bool {
        self.app_id.is_some() || self.app_name.is_some() || self.app_url.is_some()
    }

    /// Build from optional app id (Python SDK convenience).
    pub fn with_app_id(app_id: impl Into<String>) -> Result<Self> {
        Ok(Self {
            app_id: Some(sanitize_app_id(&app_id.into())?),
            ..Default::default()
        })
    }
}

/// Builder for `ApplicationContext` with validation on build.
#[derive(Debug, Default)]
pub struct ApplicationContextBuilder {
    inner: ApplicationContext,
}

impl ApplicationContextBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn app_id(mut self, value: impl Into<String>) -> Self {
        self.inner.app_id = Some(value.into());
        self
    }

    pub fn app_name(mut self, value: impl Into<String>) -> Self {
        self.inner.app_name = Some(value.into());
        self
    }

    pub fn app_url(mut self, value: impl Into<String>) -> Self {
        self.inner.app_url = Some(value.into());
        self
    }

    pub fn tenant_id(mut self, value: impl Into<String>) -> Self {
        self.inner.tenant_id = Some(value.into());
        self
    }

    pub fn request_id(mut self, value: impl Into<String>) -> Self {
        self.inner.request_id = Some(value.into());
        self
    }

    pub fn end_user_id(mut self, value: impl Into<String>) -> Self {
        self.inner.end_user_id = Some(value.into());
        self
    }

    pub fn extra_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.inner.extra_headers.insert(key.into(), value.into());
        self
    }

    pub fn build(self) -> Result<ApplicationContext> {
        let mut ctx = self.inner;
        if let Some(ref id) = ctx.app_id {
            ctx.app_id = Some(sanitize_app_id(id)?);
        }
        if let Some(ref name) = ctx.app_name {
            ctx.app_name = Some(sanitize_app_name(name)?);
        }
        if let Some(ref url) = ctx.app_url {
            ctx.app_url = Some(sanitize_app_url(url)?);
        }
        if let Some(ref tid) = ctx.tenant_id {
            ctx.tenant_id = Some(sanitize_tenant_id(tid)?);
        }
        if let Some(ref rid) = ctx.request_id {
            ctx.request_id = Some(sanitize_request_id(rid)?);
        }
        Ok(ctx)
    }
}

// ---------------------------------------------------------------------------
// Sanitizers
// ---------------------------------------------------------------------------

/// Validate and normalize an application ID slug.
pub fn sanitize_app_id(value: &str) -> Result<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(LlmError::InvalidRequest("app_id must not be empty".into()));
    }
    if trimmed.len() > 128 {
        return Err(LlmError::InvalidRequest(
            "app_id must be at most 128 characters".into(),
        ));
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("sk-") || lower.starts_with("bearer") {
        return Err(LlmError::InvalidRequest(
            "app_id must not resemble an API key or credential".into(),
        ));
    }
    if !trimmed
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
    {
        return Err(LlmError::InvalidRequest(
            "app_id may only contain ASCII letters, digits, '.', '_', and '-'".into(),
        ));
    }
    if !trimmed
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphanumeric())
    {
        return Err(LlmError::InvalidRequest(
            "app_id must start with an ASCII letter or digit".into(),
        ));
    }
    Ok(trimmed.to_string())
}

pub fn sanitize_app_name(value: &str) -> Result<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(LlmError::InvalidRequest(
            "app_name must not be empty".into(),
        ));
    }
    if trimmed.len() > 256 {
        return Err(LlmError::InvalidRequest(
            "app_name must be at most 256 characters".into(),
        ));
    }
    if !trimmed.is_ascii() {
        return Err(LlmError::InvalidRequest(
            "app_name must be ASCII only".into(),
        ));
    }
    Ok(trimmed.to_string())
}

pub fn sanitize_app_url(value: &str) -> Result<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(LlmError::InvalidRequest("app_url must not be empty".into()));
    }
    if trimmed.len() > 2048 {
        return Err(LlmError::InvalidRequest(
            "app_url must be at most 2048 characters".into(),
        ));
    }
    if !trimmed.starts_with("http://") && !trimmed.starts_with("https://") {
        return Err(LlmError::InvalidRequest(
            "app_url must start with http:// or https://".into(),
        ));
    }
    Ok(trimmed.to_string())
}

pub fn sanitize_tenant_id(value: &str) -> Result<String> {
    sanitize_app_id(value)
}

pub fn sanitize_request_id(value: &str) -> Result<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(LlmError::InvalidRequest(
            "request_id must not be empty".into(),
        ));
    }
    if trimmed.len() > 512 {
        return Err(LlmError::InvalidRequest(
            "request_id must be at most 512 characters (OpenAI limit)".into(),
        ));
    }
    if !trimmed.is_ascii() {
        return Err(LlmError::InvalidRequest(
            "request_id must be ASCII only".into(),
        ));
    }
    Ok(trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn sanitize_app_id_accepts_valid_slug() {
        assert_eq!(
            sanitize_app_id("my-app_v2.backend").unwrap(),
            "my-app_v2.backend"
        );
    }

    #[test]
    fn sanitize_app_id_rejects_api_key_like() {
        assert!(sanitize_app_id("sk-abc123").is_err());
        assert!(sanitize_app_id("Bearer-token").is_err());
    }

    #[test]
    fn sanitize_app_id_rejects_invalid_chars() {
        assert!(sanitize_app_id("app with spaces").is_err());
    }

    #[test]
    fn sanitize_request_id_rejects_non_ascii() {
        assert!(sanitize_request_id("req-日本語").is_err());
    }

    #[test]
    fn sanitize_request_id_truncates_policy_at_512() {
        let long = "a".repeat(513);
        assert!(sanitize_request_id(&long).is_err());
        assert!(sanitize_request_id(&"a".repeat(512)).is_ok());
    }

    #[test]
    fn builder_validates_on_build() {
        let ctx = ApplicationContextBuilder::new()
            .app_id("valid-app")
            .request_id("req-1")
            .build()
            .unwrap();
        assert_eq!(ctx.app_id.as_deref(), Some("valid-app"));
    }

    #[test]
    fn merge_prefers_other_values() {
        let mut base = ApplicationContext {
            app_id: Some("old".into()),
            ..Default::default()
        };
        base.merge(ApplicationContext {
            app_id: Some("new".into()),
            request_id: Some("r1".into()),
            ..Default::default()
        });
        assert_eq!(base.app_id.as_deref(), Some("new"));
        assert_eq!(base.request_id.as_deref(), Some("r1"));
    }

    #[test]
    fn from_ingress_headers_parses_canonical_keys() {
        let mut headers = HashMap::new();
        headers.insert("x-edgequake-app-id".into(), "workspace-api".into());
        headers.insert("traceparent".into(), "00-abc-def-01".into());
        let ctx = ApplicationContext::from_ingress_headers(&headers).unwrap();
        assert_eq!(ctx.app_id.as_deref(), Some("workspace-api"));
        assert_eq!(
            ctx.extra_headers.get("traceparent").map(String::as_str),
            Some("00-abc-def-01")
        );
    }

    #[test]
    fn has_app_attribution() {
        assert!(!ApplicationContext::default().has_app_attribution());
        assert!(ApplicationContext {
            app_id: Some("x".into()),
            ..Default::default()
        }
        .has_app_attribution());
    }

    #[test]
    fn sanitize_app_id_rejects_empty() {
        assert!(sanitize_app_id("").is_err());
        assert!(sanitize_app_id("   ").is_err());
    }

    #[test]
    fn sanitize_app_id_rejects_leading_invalid() {
        assert!(sanitize_app_id("-bad").is_err());
        assert!(sanitize_app_id(".bad").is_err());
    }

    #[test]
    fn sanitize_app_url_requires_parseable_url() {
        assert!(sanitize_app_url("https://app.example.com").is_ok());
        assert!(sanitize_app_url("not-a-url").is_err());
    }

    #[test]
    fn sanitize_tenant_id_accepts_alphanumeric() {
        assert_eq!(
            sanitize_tenant_id("tenant-1_prod").unwrap(),
            "tenant-1_prod"
        );
    }

    #[test]
    #[serial]
    fn from_env_reads_edgequake_app_id() {
        std::env::set_var("EDGEQUAKE_APP_ID", "env-app");
        let ctx = ApplicationContext::from_env();
        assert_eq!(ctx.app_id.as_deref(), Some("env-app"));
        std::env::remove_var("EDGEQUAKE_APP_ID");
    }

    #[test]
    #[serial]
    fn from_env_ignores_invalid_app_id() {
        std::env::remove_var("EDGEQUAKE_APP_ID");
        std::env::set_var("EDGEQUAKE_APP_ID", "sk-secret");
        let ctx = ApplicationContext::from_env();
        assert!(ctx.app_id.is_none());
        std::env::remove_var("EDGEQUAKE_APP_ID");
    }

    #[test]
    fn is_empty_context() {
        assert!(ApplicationContext::default().is_empty());
        assert!(!ApplicationContext {
            request_id: Some("r".into()),
            ..Default::default()
        }
        .is_empty());
    }

    #[test]
    fn builder_rejects_empty_app_id() {
        let err = ApplicationContextBuilder::new().app_id("").build();
        assert!(err.is_err());
    }

    #[test]
    fn merge_ingress_preserves_traceparent() {
        let mut headers = HashMap::new();
        headers.insert("x-edgequake-app-id".into(), "app".into());
        headers.insert("traceparent".into(), "00-abcd".into());
        headers.insert("Authorization".into(), "Bearer x".into());
        let ctx = ApplicationContext::from_ingress_headers(&headers).unwrap();
        assert_eq!(ctx.app_id.as_deref(), Some("app"));
        assert_eq!(
            ctx.extra_headers.get("traceparent").map(String::as_str),
            Some("00-abcd")
        );
        assert!(!ctx.extra_headers.contains_key("Authorization"));
    }
}
