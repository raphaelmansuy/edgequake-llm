//! Application attribution resolution — maps `ApplicationContext` to provider-canonical headers/body fields.
//!
//! Single source of truth for reserved-header filtering (DRY across providers).

use std::collections::HashMap;

use crate::application_context::{ApplicationContext, AttributionWarning};

pub use crate::application_context::AttributionProviderKind;

/// Resolved outbound attribution for one provider call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolvedAttribution {
    pub headers: HashMap<String, String>,
    pub body_fields: HashMap<String, String>,
    pub warnings: Vec<AttributionWarning>,
}

impl ResolvedAttribution {
    pub fn merge_into_headers(&self, target: &mut HashMap<String, String>) {
        for (k, v) in &self.headers {
            if !is_header_reserved(k, AttributionProviderKind::OpenAICompatible) {
                target.insert(k.clone(), v.clone());
            }
        }
    }
}

/// Whether a header must not be set by callers (auth / protocol).
pub fn is_header_reserved(name: &str, kind: AttributionProviderKind) -> bool {
    let lower = name.to_ascii_lowercase();
    const BASE: &[&str] = &[
        "authorization",
        "content-type",
        "content-length",
        "host",
        "user-agent",
    ];
    if BASE.contains(&lower.as_str()) {
        return true;
    }
    match kind {
        AttributionProviderKind::Anthropic => matches!(
            lower.as_str(),
            "x-api-key" | "anthropic-version" | "anthropic-beta"
        ),
        AttributionProviderKind::Gemini | AttributionProviderKind::VertexAI => {
            lower == "x-goog-api-key"
        }
        _ => false,
    }
}

/// Filter and insert extra headers; returns warnings for skipped entries.
pub fn merge_extra_headers(
    target: &mut HashMap<String, String>,
    extra: &HashMap<String, String>,
    kind: AttributionProviderKind,
) -> Vec<AttributionWarning> {
    let mut warnings = Vec::new();
    for (k, v) in extra {
        if is_header_reserved(k, kind) {
            continue;
        }
        if !is_valid_header_name(k) {
            warnings.push(AttributionWarning::InvalidHeader {
                name: k.clone(),
                reason: "invalid header name".into(),
            });
            continue;
        }
        if !v.is_ascii() {
            warnings.push(AttributionWarning::InvalidHeader {
                name: k.clone(),
                reason: "header value must be ASCII".into(),
            });
            continue;
        }
        target.insert(k.clone(), v.clone());
    }
    warnings
}

fn is_valid_header_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Resolve attribution for a provider from application context.
pub fn resolve_attribution(
    kind: AttributionProviderKind,
    ctx: &ApplicationContext,
) -> ResolvedAttribution {
    if ctx.is_empty() {
        return ResolvedAttribution::default();
    }

    let mut resolved = ResolvedAttribution::default();

    match kind {
        AttributionProviderKind::OpenAI | AttributionProviderKind::OpenAICompatible => {
            resolve_openai_family(ctx, &mut resolved);
        }
        AttributionProviderKind::AzureOpenAI => {
            resolve_azure(ctx, &mut resolved);
        }
        AttributionProviderKind::Anthropic => {
            resolve_passthrough(ctx, &mut resolved);
        }
        AttributionProviderKind::Gemini | AttributionProviderKind::VertexAI => {
            resolve_gemini(ctx, &mut resolved);
        }
        AttributionProviderKind::OpenRouter => {
            resolve_openrouter(ctx, &mut resolved);
        }
        AttributionProviderKind::Mistral => {
            resolve_mistral(ctx, &mut resolved);
        }
        AttributionProviderKind::Cohere => {
            resolve_cohere(ctx, &mut resolved);
        }
        AttributionProviderKind::Nvidia => {
            resolve_nvidia(ctx, &mut resolved);
        }
        AttributionProviderKind::Bedrock => {
            resolve_bedrock(ctx, &mut resolved);
        }
        AttributionProviderKind::XAI
        | AttributionProviderKind::HuggingFace
        | AttributionProviderKind::LMStudio
        | AttributionProviderKind::Omlx
        | AttributionProviderKind::Mtplx
        | AttributionProviderKind::LlamaCpp
        | AttributionProviderKind::VllmMlx
        | AttributionProviderKind::MlxLm
        | AttributionProviderKind::Ollama => {
            resolve_openai_family(ctx, &mut resolved);
        }
        AttributionProviderKind::VsCodeCopilot => {
            resolved
                .warnings
                .push(AttributionWarning::ProviderUnsupported {
                    provider: "vscode-copilot".into(),
                });
        }
        AttributionProviderKind::Mock => {}
    }

    resolved.warnings.extend(merge_extra_headers(
        &mut resolved.headers,
        &ctx.extra_headers,
        kind,
    ));

    inject_trace_context(&mut resolved.headers);
    inject_baggage_to_provider_headers(&mut resolved.headers);

    resolved
}

/// Whether W3C trace context injection is enabled (`EDGEQUAKE_OTEL_INJECT_TRACE_CONTEXT`).
///
/// Defaults to `true` when the `otel` feature is enabled (spec 003 §6).
pub fn otel_trace_injection_enabled() -> bool {
    match std::env::var("EDGEQUAKE_OTEL_INJECT_TRACE_CONTEXT") {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => cfg!(feature = "otel"),
        },
        Err(_) => cfg!(feature = "otel"),
    }
}

/// Whether `app_id` / `tenant_id` are copied to OTEL baggage (`EDGEQUAKE_OTEL_PROMOTE_APP_TO_BAGGAGE`).
pub fn otel_baggage_promotion_enabled() -> bool {
    std::env::var("EDGEQUAKE_OTEL_PROMOTE_APP_TO_BAGGAGE")
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Whether W3C `baggage` header is forwarded to upstream providers (dangerous; default off).
pub fn otel_propagate_baggage_to_providers() -> bool {
    std::env::var("EDGEQUAKE_PROPAGATE_BAGGAGE_TO_PROVIDERS")
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Map `LLMProvider::name()` to attribution resolver kind.
pub fn attribution_kind_from_provider_name(name: &str) -> AttributionProviderKind {
    match name {
        "openai" => AttributionProviderKind::OpenAI,
        "azure" | "azure-openai" => AttributionProviderKind::AzureOpenAI,
        "anthropic" => AttributionProviderKind::Anthropic,
        "gemini" => AttributionProviderKind::Gemini,
        "vertexai" | "vertex-ai" => AttributionProviderKind::VertexAI,
        "openrouter" => AttributionProviderKind::OpenRouter,
        "mistral" => AttributionProviderKind::Mistral,
        "nvidia" => AttributionProviderKind::Nvidia,
        "cohere" => AttributionProviderKind::Cohere,
        "bedrock" | "aws-bedrock" => AttributionProviderKind::Bedrock,
        "xai" | "x-ai" => AttributionProviderKind::XAI,
        "huggingface" | "hf" => AttributionProviderKind::HuggingFace,
        "lmstudio" | "lm-studio" => AttributionProviderKind::LMStudio,
        "omlx" | "o-mlx" | "o_mlx" => AttributionProviderKind::Omlx,
        "mtplx" | "mtp-lx" | "mtp_lx" | "mtpl-x" => AttributionProviderKind::Mtplx,
        "llamacpp" | "llama-server" | "llama.cpp" | "llamacpp-server" => {
            AttributionProviderKind::LlamaCpp
        }
        "vllm-mlx" | "vllm_mlx" | "vllmmx" => AttributionProviderKind::VllmMlx,
        "mlx-lm" | "mlx_lm" | "mlxlm" => AttributionProviderKind::MlxLm,
        "ollama" => AttributionProviderKind::Ollama,
        "vscode-copilot" | "copilot" => AttributionProviderKind::VsCodeCopilot,
        "mock" => AttributionProviderKind::Mock,
        _ => AttributionProviderKind::OpenAICompatible,
    }
}

/// Log attribution warnings at provider build time (DRY helper for all providers).
pub fn log_attribution_warnings(provider: &str, warnings: &[AttributionWarning]) {
    for w in warnings {
        tracing::warn!(provider, ?w, "application attribution warning");
    }
}

/// Emit span events documenting attribution resolution (spec 003 §7).
///
/// Call from [`TracingProvider`](crate::providers::TracingProvider) on each LLM span.
pub fn record_attribution_span_events(
    kind: AttributionProviderKind,
    ctx: &ApplicationContext,
    provider_name: &str,
) {
    if ctx.is_empty() {
        return;
    }

    if !supports_attribution(kind) && ctx.has_app_attribution() {
        tracing::event!(
            target: "edgequake.attribution",
            tracing::Level::INFO,
            event.name = "edgequake.attribution.unsupported",
            provider = provider_name,
            "Application attribution unsupported for provider"
        );
        return;
    }

    let resolved = resolve_attribution(kind, ctx);

    if !resolved.headers.is_empty() || !resolved.body_fields.is_empty() {
        let header_keys: Vec<&str> = resolved.headers.keys().map(String::as_str).collect();
        let body_keys: Vec<&str> = resolved.body_fields.keys().map(String::as_str).collect();
        tracing::event!(
            target: "edgequake.attribution",
            tracing::Level::INFO,
            event.name = "edgequake.attribution.resolved",
            provider = provider_name,
            attribution.header_keys = ?header_keys,
            attribution.body_keys = ?body_keys,
            "Application attribution resolved"
        );
    }

    for w in &resolved.warnings {
        tracing::event!(
            target: "edgequake.attribution",
            tracing::Level::WARN,
            event.name = "edgequake.attribution.warning",
            provider = provider_name,
            warning = ?w,
            "Application attribution warning"
        );
    }
}

/// Promote `app_id` and `tenant_id` to OTEL baggage on the current context (allowlist only).
pub fn promote_application_context_to_baggage(ctx: &ApplicationContext) {
    if !otel_baggage_promotion_enabled() || ctx.is_empty() {
        return;
    }

    #[cfg(feature = "otel")]
    {
        use opentelemetry::baggage::BaggageExt;
        use opentelemetry::Context;
        use opentelemetry::KeyValue;

        let mut kvs = Vec::new();
        if let Some(ref id) = ctx.app_id {
            kvs.push(KeyValue::new("app_id", id.clone()));
        }
        if let Some(ref tid) = ctx.tenant_id {
            kvs.push(KeyValue::new("tenant_id", tid.clone()));
        }
        if !kvs.is_empty() {
            Context::current_with_baggage(kvs);
        }
    }

    #[cfg(not(feature = "otel"))]
    let _ = ctx;
}

/// Inject active span `traceparent` / `tracestate` into outbound headers when enabled.
///
/// Requires the `otel` crate feature. No-op when disabled or no active span context.
pub fn inject_trace_context(headers: &mut HashMap<String, String>) {
    if !otel_trace_injection_enabled() {
        return;
    }

    #[cfg(feature = "otel")]
    {
        use opentelemetry::global;
        use opentelemetry::propagation::Injector;
        use tracing_opentelemetry::OpenTelemetrySpanExt;

        struct HeaderInjector<'a>(&'a mut HashMap<String, String>);

        impl Injector for HeaderInjector<'_> {
            fn set(&mut self, key: &str, value: String) {
                if is_header_reserved(key, AttributionProviderKind::OpenAICompatible) {
                    return;
                }
                self.0.insert(key.to_string(), value);
            }
        }

        let cx = tracing::Span::current().context();
        global::get_text_map_propagator(|propagator| {
            propagator.inject_context(&cx, &mut HeaderInjector(headers));
        });
    }

    #[cfg(not(feature = "otel"))]
    let _ = headers;
}

/// Forward W3C baggage to provider HTTP when explicitly enabled (default off).
fn inject_baggage_to_provider_headers(headers: &mut HashMap<String, String>) {
    if !otel_propagate_baggage_to_providers() {
        return;
    }

    #[cfg(feature = "otel")]
    {
        use opentelemetry::global;
        use opentelemetry::propagation::Injector;

        struct HeaderInjector<'a>(&'a mut HashMap<String, String>);

        impl Injector for HeaderInjector<'_> {
            fn set(&mut self, key: &str, value: String) {
                if key.eq_ignore_ascii_case("baggage")
                    && !is_header_reserved(key, AttributionProviderKind::OpenAICompatible)
                {
                    self.0.insert(key.to_string(), value);
                }
            }
        }

        global::get_text_map_propagator(|propagator| {
            propagator.inject_context(
                &opentelemetry::Context::current(),
                &mut HeaderInjector(headers),
            );
        });
    }

    #[cfg(not(feature = "otel"))]
    let _ = headers;
}

fn resolve_openai_family(ctx: &ApplicationContext, out: &mut ResolvedAttribution) {
    if let Some(ref rid) = ctx.request_id {
        out.headers
            .insert("X-Client-Request-Id".into(), rid.clone());
    }
    if let Some(ref uid) = ctx.end_user_id {
        out.body_fields.insert("user".into(), uid.clone());
    }
}

fn resolve_azure(ctx: &ApplicationContext, out: &mut ResolvedAttribution) {
    let app_label = ctx.app_name.clone().or_else(|| ctx.app_id.clone());
    if let Some(label) = app_label {
        out.body_fields.insert("application_name".into(), label);
    }
    if let Some(ref uid) = ctx.end_user_id {
        out.body_fields.insert("end_user_id".into(), uid.clone());
    }
    if let Some(ref tid) = ctx.tenant_id {
        out.body_fields
            .insert("end_user_tenant_id".into(), tid.clone());
    }
    if let Some(ref rid) = ctx.request_id {
        out.headers
            .insert("x-ms-client-request-id".into(), rid.clone());
    }
    let header_count = out.headers.len() + ctx.extra_headers.len();
    if header_count >= 8 {
        out.warnings.push(AttributionWarning::AzureHeaderBudget {
            count: header_count,
        });
    }
}

fn resolve_passthrough(ctx: &ApplicationContext, out: &mut ResolvedAttribution) {
    if let Some(ref rid) = ctx.request_id {
        out.headers.insert("x-request-id".into(), rid.clone());
    }
    if let Some(ref id) = ctx.app_id {
        out.headers.insert("x-application-id".into(), id.clone());
    }
}

fn resolve_gemini(ctx: &ApplicationContext, out: &mut ResolvedAttribution) {
    if let Some(ref id) = ctx.app_id {
        out.headers
            .insert("x-goog-api-client".into(), format!("edgequake-app/{}", id));
    } else if let Some(ref name) = ctx.app_name {
        let slug = name
            .to_ascii_lowercase()
            .replace(|c: char| !c.is_ascii_alphanumeric(), "-");
        out.headers.insert(
            "x-goog-api-client".into(),
            format!("edgequake-app/{}", slug),
        );
    }
    if let Some(ref rid) = ctx.request_id {
        out.headers.insert("x-request-id".into(), rid.clone());
    }
}

fn resolve_openrouter(ctx: &ApplicationContext, out: &mut ResolvedAttribution) {
    let url = ctx.app_url.clone();
    let title = ctx.app_name.clone().or_else(|| ctx.app_id.clone());

    if let Some(ref u) = url {
        out.headers.insert("HTTP-Referer".into(), u.clone());
    } else if ctx.has_app_attribution() {
        out.warnings
            .push(AttributionWarning::OpenRouterMissingReferer);
    }

    if let Some(ref t) = title {
        out.headers.insert("X-OpenRouter-Title".into(), t.clone());
        out.headers.insert("X-Title".into(), t.clone());
    }

    if url.as_deref().is_some_and(|u| u.contains("localhost")) && title.is_none() {
        out.warnings
            .push(AttributionWarning::OpenRouterLocalhostMissingTitle);
    }
}

fn resolve_mistral(ctx: &ApplicationContext, out: &mut ResolvedAttribution) {
    let client_name = ctx.app_name.clone().or_else(|| ctx.app_id.clone());
    if let Some(name) = client_name {
        out.headers.insert("X-Client-Name".into(), name);
    }
    resolve_openai_family(ctx, out);
}

fn resolve_cohere(ctx: &ApplicationContext, out: &mut ResolvedAttribution) {
    let client_name = ctx.app_name.clone().or_else(|| ctx.app_id.clone());
    if let Some(name) = client_name {
        out.headers.insert("X-Client-Name".into(), name);
    }
    if let Some(ref rid) = ctx.request_id {
        out.headers.insert("x-request-id".into(), rid.clone());
    }
}

fn resolve_nvidia(ctx: &ApplicationContext, out: &mut ResolvedAttribution) {
    if let Some(ref rid) = ctx.request_id {
        out.headers.insert("X-Request-Id".into(), rid.clone());
    } else if let Some(ref id) = ctx.app_id {
        out.headers.insert(
            "X-Request-Id".into(),
            format!("{}/{}", id, uuid::Uuid::new_v4()),
        );
    }
}

fn resolve_bedrock(ctx: &ApplicationContext, out: &mut ResolvedAttribution) {
    if let Some(ref id) = ctx.app_id {
        out.body_fields.insert("app".into(), id.clone());
    }
    if let Some(ref name) = ctx.app_name {
        out.body_fields.insert("app_name".into(), name.clone());
    }
    if let Some(ref tid) = ctx.tenant_id {
        out.body_fields.insert("tenant_id".into(), tid.clone());
    }
}

/// Append Gemini client token without clobbering an existing value.
pub fn append_goog_api_client(existing: Option<&str>, token: &str) -> String {
    match existing {
        Some(prev) if !prev.is_empty() => format!("{} {}", prev.trim(), token),
        _ => token.to_string(),
    }
}

/// Whether this provider kind supports HTTP/body attribution propagation.
pub fn supports_attribution(kind: AttributionProviderKind) -> bool {
    !matches!(kind, AttributionProviderKind::VsCodeCopilot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application_context::ApplicationContext;
    use serial_test::serial;

    #[test]
    #[serial]
    fn reserved_headers_include_auth() {
        assert!(is_header_reserved(
            "Authorization",
            AttributionProviderKind::OpenAI
        ));
        assert!(is_header_reserved(
            "x-api-key",
            AttributionProviderKind::Anthropic
        ));
    }

    #[test]
    fn openai_resolves_client_request_id() {
        let ctx = ApplicationContext {
            request_id: Some("req-123".into()),
            ..Default::default()
        };
        let r = resolve_attribution(AttributionProviderKind::OpenAI, &ctx);
        assert_eq!(
            r.headers.get("X-Client-Request-Id").map(String::as_str),
            Some("req-123")
        );
    }

    #[test]
    fn openrouter_requires_referer_warning() {
        let ctx = ApplicationContext {
            app_id: Some("myapp".into()),
            ..Default::default()
        };
        let r = resolve_attribution(AttributionProviderKind::OpenRouter, &ctx);
        assert!(r
            .warnings
            .contains(&AttributionWarning::OpenRouterMissingReferer));
    }

    #[test]
    fn openrouter_sets_referer_and_title() {
        let ctx = ApplicationContext {
            app_url: Some("https://app.example.com".into()),
            app_name: Some("My App".into()),
            ..Default::default()
        };
        let r = resolve_attribution(AttributionProviderKind::OpenRouter, &ctx);
        assert_eq!(
            r.headers.get("HTTP-Referer").map(String::as_str),
            Some("https://app.example.com")
        );
        assert_eq!(
            r.headers.get("X-OpenRouter-Title").map(String::as_str),
            Some("My App")
        );
    }

    #[test]
    fn gemini_appends_goog_client_token() {
        let ctx = ApplicationContext {
            app_id: Some("backend".into()),
            ..Default::default()
        };
        let r = resolve_attribution(AttributionProviderKind::Gemini, &ctx);
        assert_eq!(
            r.headers.get("x-goog-api-client").map(String::as_str),
            Some("edgequake-app/backend")
        );
    }

    #[test]
    fn append_goog_api_client_merges() {
        assert_eq!(
            append_goog_api_client(Some("gl-rust/1.0"), "edgequake-app/x"),
            "gl-rust/1.0 edgequake-app/x"
        );
    }

    #[test]
    fn bedrock_resolves_metadata_fields() {
        let ctx = ApplicationContext {
            app_id: Some("crm".into()),
            tenant_id: Some("tenant-1".into()),
            ..Default::default()
        };
        let r = resolve_attribution(AttributionProviderKind::Bedrock, &ctx);
        assert_eq!(r.body_fields.get("app").map(String::as_str), Some("crm"));
        assert_eq!(
            r.body_fields.get("tenant_id").map(String::as_str),
            Some("tenant-1")
        );
    }

    #[test]
    fn cohere_sets_x_client_name() {
        let ctx = ApplicationContext {
            app_name: Some("my-project".into()),
            ..Default::default()
        };
        let r = resolve_attribution(AttributionProviderKind::Cohere, &ctx);
        assert_eq!(
            r.headers.get("X-Client-Name").map(String::as_str),
            Some("my-project")
        );
    }

    #[test]
    fn vscode_copilot_unsupported() {
        let ctx = ApplicationContext {
            app_id: Some("x".into()),
            ..Default::default()
        };
        let r = resolve_attribution(AttributionProviderKind::VsCodeCopilot, &ctx);
        assert!(matches!(
            r.warnings.first(),
            Some(AttributionWarning::ProviderUnsupported { .. })
        ));
    }

    #[test]
    fn extra_headers_drop_authorization() {
        let mut extra = HashMap::new();
        extra.insert("Authorization".into(), "Bearer bad".into());
        extra.insert("x-custom".into(), "ok".into());
        let mut target = HashMap::new();
        merge_extra_headers(&mut target, &extra, AttributionProviderKind::OpenAI);
        assert!(!target.contains_key("Authorization"));
        assert_eq!(target.get("x-custom").map(String::as_str), Some("ok"));
    }

    #[test]
    #[serial]
    fn otel_trace_injection_enabled_by_default_with_otel_feature() {
        std::env::remove_var("EDGEQUAKE_OTEL_INJECT_TRACE_CONTEXT");
        assert_eq!(otel_trace_injection_enabled(), cfg!(feature = "otel"));
    }

    #[test]
    #[serial]
    fn otel_trace_injection_can_be_disabled() {
        std::env::set_var("EDGEQUAKE_OTEL_INJECT_TRACE_CONTEXT", "false");
        assert!(!otel_trace_injection_enabled());
        std::env::remove_var("EDGEQUAKE_OTEL_INJECT_TRACE_CONTEXT");
    }

    #[test]
    #[serial]
    fn otel_baggage_promotion_disabled_by_default() {
        std::env::remove_var("EDGEQUAKE_OTEL_PROMOTE_APP_TO_BAGGAGE");
        assert!(!otel_baggage_promotion_enabled());
    }

    #[test]
    fn attribution_kind_from_provider_name_maps_known_providers() {
        assert_eq!(
            attribution_kind_from_provider_name("openai"),
            AttributionProviderKind::OpenAI
        );
        assert_eq!(
            attribution_kind_from_provider_name("ollama"),
            AttributionProviderKind::Ollama
        );
        assert_eq!(
            attribution_kind_from_provider_name("vscode-copilot"),
            AttributionProviderKind::VsCodeCopilot
        );
        assert_eq!(
            attribution_kind_from_provider_name("custom-vendor"),
            AttributionProviderKind::OpenAICompatible
        );
    }

    #[test]
    fn record_attribution_span_events_no_panic_for_openai() {
        let ctx = ApplicationContext {
            app_id: Some("app".into()),
            request_id: Some("req-1".into()),
            ..Default::default()
        };
        record_attribution_span_events(AttributionProviderKind::OpenAI, &ctx, "openai");
    }

    #[test]
    fn record_attribution_span_events_unsupported_provider() {
        let ctx = ApplicationContext {
            app_id: Some("app".into()),
            ..Default::default()
        };
        record_attribution_span_events(
            AttributionProviderKind::VsCodeCopilot,
            &ctx,
            "vscode-copilot",
        );
    }

    #[test]
    fn log_attribution_warnings_emits_without_panic() {
        log_attribution_warnings(
            "openrouter",
            &[AttributionWarning::OpenRouterMissingReferer],
        );
    }

    #[test]
    #[serial]
    fn inject_trace_context_noop_when_disabled() {
        std::env::set_var("EDGEQUAKE_OTEL_INJECT_TRACE_CONTEXT", "false");
        let mut headers = HashMap::new();
        inject_trace_context(&mut headers);
        assert!(headers.is_empty());
        std::env::remove_var("EDGEQUAKE_OTEL_INJECT_TRACE_CONTEXT");
    }
}
