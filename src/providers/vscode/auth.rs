//! GitHub device code authentication flow for Copilot.
//!
//! Implements the OAuth device code flow to authenticate with GitHub
//! and obtain access tokens for GitHub Copilot.
//!
//! # Device Code Flow
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                  GitHub OAuth Device Code Flow                   │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                   │
//! │  ┌────────────────┐                                              │
//! │  │ 1. Request     │  POST /login/device/code                     │
//! │  │    Device Code │  → Returns user_code + device_code           │
//! │  └───────┬────────┘                                              │
//! │          │                                                        │
//! │          ▼                                                        │
//! │  ┌────────────────┐                                              │
//! │  │ 2. Display URL │  User visits: https://github.com/login/device│
//! │  │    to User     │  User enters: ABC-123 (user_code)            │
//! │  └───────┬────────┘                                              │
//! │          │                                                        │
//! │          ▼                                                        │
//! │  ┌────────────────┐    ┌──────────────────┐                      │
//! │  │ 3. Poll for    │───▶│ authorization_   │ (waiting)            │
//! │  │    Access Token│    │ pending          │                      │
//! │  │                │◀───┴──────────────────┘                      │
//! │  │   (every N sec)│    ┌──────────────────┐                      │
//! │  │                │───▶│ access_token     │ (success!)           │
//! │  └───────┬────────┘    └──────────────────┘                      │
//! │          │                                                        │
//! │          ▼                                                        │
//! │  ┌────────────────┐                                              │
//! │  │ 4. Store Token │  Save to ~/.config/gh/github_token.json      │
//! │  │    for Later   │  Token used for Copilot API auth             │
//! │  └────────────────┘                                              │
//! │                                                                   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Security Notes
//!
//! - Client ID is public (Iv1.b507a08c87ecfe98) - this is by design for device flow
//! - User must authorize in browser - no secrets embedded
//! - Tokens stored locally with restrictive permissions
//! - Scopes: `read:user` - minimal permissions needed
//!
//! # Error Handling
//!
//! Polling responses:
//! - `authorization_pending` - User hasn't authorized yet, keep polling
//! - `slow_down` - Polling too fast, increase interval
//! - `expired_token` - Device code expired, restart flow
//! - `access_denied` - User denied authorization

use anyhow::{Context, Result};
use serde::Deserialize;
use std::time::Duration;

const GITHUB_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
const GITHUB_SCOPES: &str = "read:user";
const DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
const ACCESS_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";

/// Response from device code request.
#[derive(Debug, Clone, Deserialize)]
pub struct DeviceCodeResponse {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub expires_in: u64,
    pub interval: u64,
}

/// Response from access token polling.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AccessTokenResponse {
    Success {
        access_token: String,
        token_type: String,
        scope: String,
    },
    Pending {
        error: String,
    },
}

/// GitHub authentication manager.
pub struct GitHubAuth {
    client: reqwest::Client,
}

impl GitHubAuth {
    /// Create a new GitHub auth manager.
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self { client })
    }

    /// Request a device code from GitHub.
    pub async fn request_device_code(&self) -> Result<DeviceCodeResponse> {
        let response = self
            .client
            .post(DEVICE_CODE_URL)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "client_id": GITHUB_CLIENT_ID,
                "scope": GITHUB_SCOPES,
            }))
            .send()
            .await
            .context("Failed to request device code")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Device code request failed: {} - {}", status, body);
        }

        response
            .json()
            .await
            .context("Failed to parse device code response")
    }

    /// Poll for access token after user authorization.
    pub async fn poll_access_token(&self, device_code: &str, interval: u64) -> Result<String> {
        let poll_interval = Duration::from_secs(interval);

        loop {
            tokio::time::sleep(poll_interval).await;

            let response = self
                .client
                .post(ACCESS_TOKEN_URL)
                .header("Accept", "application/json")
                .header("Content-Type", "application/json")
                .json(&serde_json::json!({
                    "client_id": GITHUB_CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                }))
                .send()
                .await
                .context("Failed to poll for access token")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                anyhow::bail!("Access token poll failed: {} - {}", status, body);
            }

            let result: AccessTokenResponse = response
                .json()
                .await
                .context("Failed to parse access token response")?;

            match result {
                AccessTokenResponse::Success { access_token, .. } => {
                    return Ok(access_token);
                }
                AccessTokenResponse::Pending { error } => {
                    if error == "authorization_pending" {
                        // Continue polling
                        continue;
                    } else if error == "slow_down" {
                        // Slow down polling
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        continue;
                    } else {
                        anyhow::bail!("Authorization failed: {}", error);
                    }
                }
            }
        }
    }

    /// Complete device code flow: request code and poll for token.
    pub async fn device_code_flow<F>(&self, on_code: F) -> Result<String>
    where
        F: FnOnce(&DeviceCodeResponse),
    {
        // Request device code
        let device_code_response = self.request_device_code().await?;

        // Notify callback with user code
        on_code(&device_code_response);

        // Poll for access token
        self.poll_access_token(
            &device_code_response.device_code,
            device_code_response.interval,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires user interaction
    async fn test_device_code_flow() {
        let auth = GitHubAuth::new().unwrap();

        let token = auth
            .device_code_flow(|code| {
                println!("\nPlease visit: {}", code.verification_uri);
                println!("And enter code: {}", code.user_code);
                println!("\nWaiting for authorization...\n");
            })
            .await
            .unwrap();

        println!("Got token: {}...", &token[..20]);
        assert!(!token.is_empty());
    }

    // ========================================================================
    // Response Parsing Tests (No Network Required)
    // ========================================================================

    #[test]
    fn test_device_code_response_deserialization() {
        let json = r#"{
            "device_code": "abc123",
            "user_code": "ABCD-1234",
            "verification_uri": "https://github.com/login/device",
            "expires_in": 900,
            "interval": 5
        }"#;

        let response: DeviceCodeResponse = serde_json::from_str(json).unwrap();

        assert_eq!(response.device_code, "abc123");
        assert_eq!(response.user_code, "ABCD-1234");
        assert_eq!(response.verification_uri, "https://github.com/login/device");
        assert_eq!(response.expires_in, 900);
        assert_eq!(response.interval, 5);
    }

    #[test]
    fn test_device_code_response_with_extra_fields() {
        // GitHub may return additional fields
        let json = r#"{
            "device_code": "abc123",
            "user_code": "EFGH-5678",
            "verification_uri": "https://github.com/login/device",
            "expires_in": 600,
            "interval": 10,
            "verification_uri_complete": "https://github.com/login/device?user_code=EFGH-5678"
        }"#;

        // Should parse without error despite extra field
        let response: DeviceCodeResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.user_code, "EFGH-5678");
    }

    #[test]
    fn test_access_token_success_response() {
        let json = r#"{
            "access_token": "gho_test_token_abc123",
            "token_type": "bearer",
            "scope": "read:user"
        }"#;

        let response: AccessTokenResponse = serde_json::from_str(json).unwrap();

        match response {
            AccessTokenResponse::Success {
                access_token,
                token_type,
                scope,
            } => {
                assert_eq!(access_token, "gho_test_token_abc123");
                assert_eq!(token_type, "bearer");
                assert_eq!(scope, "read:user");
            }
            AccessTokenResponse::Pending { .. } => {
                panic!("Expected Success variant");
            }
        }
    }

    #[test]
    fn test_access_token_pending_authorization() {
        let json = r#"{
            "error": "authorization_pending"
        }"#;

        let response: AccessTokenResponse = serde_json::from_str(json).unwrap();

        match response {
            AccessTokenResponse::Pending { error } => {
                assert_eq!(error, "authorization_pending");
            }
            AccessTokenResponse::Success { .. } => {
                panic!("Expected Pending variant");
            }
        }
    }

    #[test]
    fn test_access_token_slow_down_error() {
        let json = r#"{
            "error": "slow_down"
        }"#;

        let response: AccessTokenResponse = serde_json::from_str(json).unwrap();

        match response {
            AccessTokenResponse::Pending { error } => {
                assert_eq!(error, "slow_down");
            }
            AccessTokenResponse::Success { .. } => {
                panic!("Expected Pending variant");
            }
        }
    }

    #[test]
    fn test_access_token_expired_token_error() {
        let json = r#"{
            "error": "expired_token"
        }"#;

        let response: AccessTokenResponse = serde_json::from_str(json).unwrap();

        match response {
            AccessTokenResponse::Pending { error } => {
                assert_eq!(error, "expired_token");
            }
            AccessTokenResponse::Success { .. } => {
                panic!("Expected Pending variant");
            }
        }
    }

    #[test]
    fn test_access_token_access_denied_error() {
        let json = r#"{
            "error": "access_denied"
        }"#;

        let response: AccessTokenResponse = serde_json::from_str(json).unwrap();

        match response {
            AccessTokenResponse::Pending { error } => {
                assert_eq!(error, "access_denied");
            }
            AccessTokenResponse::Success { .. } => {
                panic!("Expected Pending variant");
            }
        }
    }

    // ========================================================================
    // Client Creation Tests
    // ========================================================================

    #[test]
    fn test_github_auth_creation() {
        let auth = GitHubAuth::new();
        assert!(auth.is_ok());
    }

    // ========================================================================
    // Constants Tests
    // ========================================================================

    #[test]
    fn test_github_client_id_format() {
        // Client ID should be in Iv1. format
        assert!(GITHUB_CLIENT_ID.starts_with("Iv1."));
        assert!(GITHUB_CLIENT_ID.len() > 5);
    }

    #[test]
    fn test_device_code_url_is_https() {
        assert!(DEVICE_CODE_URL.starts_with("https://"));
        assert!(DEVICE_CODE_URL.contains("github.com"));
    }

    #[test]
    fn test_access_token_url_is_https() {
        assert!(ACCESS_TOKEN_URL.starts_with("https://"));
        assert!(ACCESS_TOKEN_URL.contains("github.com"));
    }
}
