//! Token storage and management for GitHub Copilot.
//!
//! Handles storing, loading, and refreshing GitHub and Copilot tokens.
//!
//! # Token Lifecycle
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────┐
//! │                    TOKEN LIFECYCLE                             │
//! ├───────────────────────────────────────────────────────────────┤
//! │                                                                │
//! │  1. Initial Auth (one-time via device code):                  │
//! │                                                                │
//! │     User ──device code──▶ GitHub ──grants──▶ GitHub Token     │
//! │                                              (persisted to    │
//! │                                               disk)           │
//! │                                                                │
//! │  2. Token Exchange (automatic):                               │
//! │                                                                │
//! │     GitHub Token ──GET /copilot_internal/v2/token──▶          │
//! │                         Copilot Token (15min expiry)          │
//! │                                                                │
//! │  3. Token Refresh (automatic before each request):            │
//! │                                                                │
//! │     IF (now >= expires_at - 60s):                             │
//! │         Fetch new Copilot Token                               │
//! │         Persist to disk                                       │
//! │     END                                                        │
//! │                                                                │
//! └───────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Token Storage Locations
//!
//! - **macOS/Linux**: `~/.config/edgequake/copilot/`
//! - **Windows**: `%APPDATA%\edgequake\copilot\`
//!
//! Files:
//! - `github_token.json` - Long-lived GitHub OAuth token
//! - `copilot_token.json` - Short-lived Copilot API token
//!
//! # Refresh Buffer
//!
//! Tokens are refreshed 60 seconds before expiry to avoid race conditions
//! where a token expires mid-request.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::fs;
use tracing::debug;

const COPILOT_TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";
const TOKEN_REFRESH_BUFFER: u64 = 60; // Refresh 60 seconds before expiry

/// GitHub access token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubToken {
    pub access_token: String,
    pub created_at: u64,
}

/// Copilot access token with expiry information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopilotToken {
    pub token: String,
    pub expires_at: u64,
    pub refresh_in: u64,
    pub organization_list: Option<Vec<String>>,
}

/// Token manager for GitHub and Copilot tokens.
#[derive(Clone)]
pub struct TokenManager {
    config_dir: PathBuf,
    client: reqwest::Client,
}

impl TokenManager {
    /// Create a new token manager.
    pub fn new() -> Result<Self> {
        let config_dir = Self::get_config_dir()?;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self { config_dir, client })
    }

    /// Get the configuration directory for token storage.
    fn get_config_dir() -> Result<PathBuf> {
        let base_dir = dirs::config_dir().context("Failed to get config directory")?;

        let config_dir = base_dir.join("edgequake").join("copilot");
        Ok(config_dir)
    }

    /// Get the path to the GitHub token file.
    fn github_token_path(&self) -> PathBuf {
        self.config_dir.join("github_token.json")
    }

    /// Get the path to the Copilot token file.
    fn copilot_token_path(&self) -> PathBuf {
        self.config_dir.join("copilot_token.json")
    }

    /// Ensure the config directory exists.
    async fn ensure_config_dir(&self) -> Result<()> {
        fs::create_dir_all(&self.config_dir)
            .await
            .context("Failed to create config directory")?;
        Ok(())
    }

    /// Save GitHub token to disk.
    pub async fn save_github_token(&self, access_token: String) -> Result<()> {
        self.ensure_config_dir().await?;

        let token = GitHubToken {
            access_token,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        let json =
            serde_json::to_string_pretty(&token).context("Failed to serialize GitHub token")?;

        fs::write(self.github_token_path(), json)
            .await
            .context("Failed to write GitHub token")?;

        Ok(())
    }

    /// Load GitHub token from disk.
    pub async fn load_github_token(&self) -> Result<GitHubToken> {
        let json = fs::read_to_string(self.github_token_path())
            .await
            .context("Failed to read GitHub token")?;

        serde_json::from_str(&json).context("Failed to parse GitHub token")
    }

    /// Save Copilot token to disk.
    pub async fn save_copilot_token(&self, token: CopilotToken) -> Result<()> {
        self.ensure_config_dir().await?;

        let json =
            serde_json::to_string_pretty(&token).context("Failed to serialize Copilot token")?;

        fs::write(self.copilot_token_path(), json)
            .await
            .context("Failed to write Copilot token")?;

        Ok(())
    }

    /// Load Copilot token from disk.
    pub async fn load_copilot_token(&self) -> Result<CopilotToken> {
        let json = fs::read_to_string(self.copilot_token_path())
            .await
            .context("Failed to read Copilot token")?;

        serde_json::from_str(&json).context("Failed to parse Copilot token")
    }

    /// Check if Copilot token needs refresh.
    pub fn needs_refresh(&self, token: &CopilotToken) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let refresh_at = token.expires_at.saturating_sub(TOKEN_REFRESH_BUFFER);
        now >= refresh_at
    }

    /// Fetch a new Copilot token using GitHub token.
    pub async fn fetch_copilot_token(&self, github_token: &str) -> Result<CopilotToken> {
        let response = self
            .client
            .get(COPILOT_TOKEN_URL)
            .header("Accept", "application/json")
            .header("Authorization", format!("Bearer {}", github_token))
            .header("User-Agent", "GitHubCopilot/0.26.7")
            .send()
            .await
            .context("Failed to fetch Copilot token")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Copilot token request failed: {} - {}", status, body);
        }

        #[derive(Deserialize)]
        struct CopilotTokenResponse {
            token: String,
            expires_at: u64,
            refresh_in: Option<u64>,
            organization_list: Option<Vec<String>>,
        }

        let resp: CopilotTokenResponse = response
            .json()
            .await
            .context("Failed to parse Copilot token response")?;

        Ok(CopilotToken {
            token: resp.token,
            expires_at: resp.expires_at,
            refresh_in: resp.refresh_in.unwrap_or(900), // Default 15 minutes
            organization_list: resp.organization_list,
        })
    }

    /// Validate a Copilot token by making a test API call.
    /// Returns Ok(true) if valid, Ok(false) if invalid/expired, Err on network issues.
    pub async fn validate_copilot_token(&self, token: &str) -> Result<bool> {
        let client = reqwest::Client::new();
        let response = client
            .get("https://api.githubcopilot.com/models")
            .header("Authorization", format!("Bearer {}", token))
            .header("Editor-Version", "vscode/1.85.0")
            .header("Editor-Plugin-Version", "copilot/1.155.0")
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await?;

        Ok(response.status().is_success())
    }

    /// Get a valid Copilot token, refreshing if necessary and validating with API call.
    pub async fn get_valid_copilot_token(&self) -> Result<String> {
        // Try to load existing Copilot token
        if let Ok(copilot_token) = self.load_copilot_token().await {
            if !self.needs_refresh(&copilot_token) {
                // Token not expired by timestamp, but validate it actually works
                debug!("Validating cached Copilot token with API call...");
                if self
                    .validate_copilot_token(&copilot_token.token)
                    .await
                    .unwrap_or(false)
                {
                    debug!("Cached Copilot token is valid");
                    return Ok(copilot_token.token);
                }
                debug!("Cached Copilot token failed validation, will refresh");
            } else {
                debug!(
                    "Copilot token expired or expiring soon (within {}s), refreshing...",
                    TOKEN_REFRESH_BUFFER
                );
            }
        } else {
            debug!("No cached Copilot token found, fetching fresh token");
        }

        // Need to refresh - get GitHub token
        let github_token = self.load_github_token().await?;

        // Fetch new Copilot token
        debug!("Requesting fresh Copilot token from GitHub API");
        let copilot_token = self.fetch_copilot_token(&github_token.access_token).await?;
        let token_value = copilot_token.token.clone();

        // Validate the new token
        if !self
            .validate_copilot_token(&token_value)
            .await
            .unwrap_or(false)
        {
            return Err(anyhow::anyhow!(
                "Fetched token is invalid. Your GitHub account may not have Copilot access."
            ));
        }

        // Save for next time
        self.save_copilot_token(copilot_token.clone()).await?;
        debug!(
            "Successfully refreshed and saved Copilot token (expires at: {})",
            copilot_token.expires_at
        );

        Ok(token_value)
    }

    /// Clear all stored tokens.
    pub async fn clear_tokens(&self) -> Result<()> {
        let _ = fs::remove_file(self.github_token_path()).await;
        let _ = fs::remove_file(self.copilot_token_path()).await;
        Ok(())
    }

    /// Check if GitHub token exists.
    pub async fn has_github_token(&self) -> bool {
        self.github_token_path().exists()
    }

    /// Check if Copilot token exists.
    pub async fn has_copilot_token(&self) -> bool {
        self.copilot_token_path().exists()
    }

    /// Try to load GitHub token from VS Code Copilot's hosts.json as fallback.
    /// Returns None if file doesn't exist or cannot be parsed.
    pub async fn try_load_vscode_github_token(&self) -> Option<String> {
        let vscode_hosts_path = dirs::config_dir()?
            .join("github-copilot")
            .join("hosts.json");

        if !vscode_hosts_path.exists() {
            return None;
        }

        let contents = fs::read_to_string(&vscode_hosts_path).await.ok()?;

        #[derive(Deserialize)]
        struct HostsJson {
            #[serde(rename = "github.com")]
            github_com: Option<GithubComEntry>,
        }

        #[derive(Deserialize)]
        struct GithubComEntry {
            oauth_token: String,
        }

        let hosts: HostsJson = serde_json::from_str(&contents).ok()?;
        Some(hosts.github_com?.oauth_token)
    }

    /// Import GitHub token from VS Code Copilot configuration.
    /// Returns true if token was successfully imported.
    pub async fn import_vscode_token(&self) -> Result<bool> {
        if let Some(token) = self.try_load_vscode_github_token().await {
            self.save_github_token(token).await?;
            debug!("Successfully imported GitHub token from VS Code Copilot");
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_token_manager_creation() {
        let manager = TokenManager::new().unwrap();
        assert!(manager.config_dir.to_string_lossy().contains("edgequake"));
    }

    #[tokio::test]
    async fn test_config_dir_creation() {
        let manager = TokenManager::new().unwrap();
        manager.ensure_config_dir().await.unwrap();
        assert!(manager.config_dir.exists());
    }

    // ========================================================================
    // Token Refresh Logic Tests
    // ========================================================================

    #[test]
    fn test_needs_refresh_false_when_valid() {
        let manager = TokenManager::new().unwrap();

        // Token expires in 2 hours (well beyond the 60-second buffer)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let token = CopilotToken {
            token: "test_token".to_string(),
            expires_at: now + 7200, // 2 hours from now
            refresh_in: 900,
            organization_list: None,
        };

        assert!(!manager.needs_refresh(&token));
    }

    #[test]
    fn test_needs_refresh_true_when_expired() {
        let manager = TokenManager::new().unwrap();

        // Token expired 1 hour ago
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let token = CopilotToken {
            token: "test_token".to_string(),
            expires_at: now.saturating_sub(3600), // 1 hour ago
            refresh_in: 900,
            organization_list: None,
        };

        assert!(manager.needs_refresh(&token));
    }

    #[test]
    fn test_needs_refresh_true_within_buffer() {
        let manager = TokenManager::new().unwrap();

        // Token expires in 30 seconds (within the 60-second buffer)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let token = CopilotToken {
            token: "test_token".to_string(),
            expires_at: now + 30, // 30 seconds from now
            refresh_in: 900,
            organization_list: None,
        };

        assert!(manager.needs_refresh(&token));
    }

    #[test]
    fn test_needs_refresh_false_just_outside_buffer() {
        let manager = TokenManager::new().unwrap();

        // Token expires in 120 seconds (outside the 60-second buffer)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let token = CopilotToken {
            token: "test_token".to_string(),
            expires_at: now + 120, // 2 minutes from now
            refresh_in: 900,
            organization_list: None,
        };

        assert!(!manager.needs_refresh(&token));
    }

    // ========================================================================
    // Token Serialization Tests
    // ========================================================================

    #[test]
    fn test_github_token_serialization_roundtrip() {
        let token = GitHubToken {
            access_token: "gho_test_token_12345".to_string(),
            created_at: 1699876543,
        };

        let json = serde_json::to_string(&token).unwrap();
        let parsed: GitHubToken = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.access_token, token.access_token);
        assert_eq!(parsed.created_at, token.created_at);
    }

    #[test]
    fn test_copilot_token_serialization_roundtrip() {
        let token = CopilotToken {
            token: "tid=test;exp=1234567890;sku=copilot".to_string(),
            expires_at: 1699876543,
            refresh_in: 900,
            organization_list: Some(vec!["org1".to_string(), "org2".to_string()]),
        };

        let json = serde_json::to_string(&token).unwrap();
        let parsed: CopilotToken = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.token, token.token);
        assert_eq!(parsed.expires_at, token.expires_at);
        assert_eq!(parsed.refresh_in, token.refresh_in);
        assert_eq!(parsed.organization_list, token.organization_list);
    }

    #[test]
    fn test_copilot_token_without_org_list() {
        let token = CopilotToken {
            token: "test_token".to_string(),
            expires_at: 1699876543,
            refresh_in: 900,
            organization_list: None,
        };

        let json = serde_json::to_string(&token).unwrap();
        let parsed: CopilotToken = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.organization_list, None);
    }

    #[test]
    fn test_github_token_json_format() {
        let token = GitHubToken {
            access_token: "test123".to_string(),
            created_at: 1699876543,
        };

        let json = serde_json::to_string(&token).unwrap();

        assert!(json.contains("access_token"));
        assert!(json.contains("test123"));
        assert!(json.contains("created_at"));
        assert!(json.contains("1699876543"));
    }

    // ========================================================================
    // Path Tests
    // ========================================================================

    #[test]
    fn test_token_paths_are_distinct() {
        let manager = TokenManager::new().unwrap();

        let github_path = manager.github_token_path();
        let copilot_path = manager.copilot_token_path();

        assert_ne!(github_path, copilot_path);
        assert!(github_path.to_string_lossy().contains("github"));
        assert!(copilot_path.to_string_lossy().contains("copilot"));
    }

    #[test]
    fn test_paths_under_config_dir() {
        let manager = TokenManager::new().unwrap();

        let github_path = manager.github_token_path();
        let copilot_path = manager.copilot_token_path();

        assert!(github_path.starts_with(&manager.config_dir));
        assert!(copilot_path.starts_with(&manager.config_dir));
    }
}
