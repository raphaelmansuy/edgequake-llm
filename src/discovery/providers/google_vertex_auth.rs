//! Vertex AI credential resolution for discovery (sync, env + gcloud).

use std::process::Command;

/// Resolved Vertex AI credentials for discovery HTTP calls.
#[derive(Debug, Clone)]
pub struct VertexDiscoveryCredentials {
    pub project_id: String,
    pub region: String,
    pub access_token: String,
}

/// Resolve credentials from env and optional gcloud CLI (same order as `GeminiProvider::from_env_vertex_ai`).
pub fn resolve_vertex_discovery_credentials() -> Option<VertexDiscoveryCredentials> {
    let project_id = std::env::var("GOOGLE_CLOUD_PROJECT").ok()?;
    let region = std::env::var("GOOGLE_CLOUD_REGION").unwrap_or_else(|_| "us-central1".to_string());

    let access_token = std::env::var("GOOGLE_ACCESS_TOKEN")
        .ok()
        .or_else(|| run_gcloud_token_cmd(&["auth", "print-access-token"]).ok())
        .or_else(|| {
            run_gcloud_token_cmd(&["auth", "application-default", "print-access-token"]).ok()
        })?;

    Some(VertexDiscoveryCredentials {
        project_id,
        region,
        access_token,
    })
}

fn run_gcloud_token_cmd(args: &[&str]) -> Result<String, String> {
    let output = Command::new("gcloud")
        .args(args)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if token.is_empty() {
        return Err("empty token".into());
    }
    Ok(token)
}
