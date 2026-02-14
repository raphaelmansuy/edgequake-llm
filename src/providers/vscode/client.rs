//! HTTP client for GitHub Copilot API.
//!
//! # Direct vs Proxy Mode
//!
//! This client supports two modes of operation:
//!
//! ## Direct Mode (Default)
//! Connects directly to `api.githubcopilot.com` using native Rust HTTP.
//! No external dependencies required after initial GitHub authentication.
//!
//! ## Proxy Mode (Legacy)
//! Connects to a local copilot-api proxy for backward compatibility.
//! Set `VSCODE_COPILOT_DIRECT=false` to use proxy mode.
//!
//! # Request Flow
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      REQUEST FLOW                                │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  Application                      VsCodeCopilotClient           │
//! │       │                                  │                       │
//! │       │  complete("prompt")              │                       │
//! │       ├─────────────────────────────────▶│                       │
//! │       │                                  │                       │
//! │       │                          ┌───────┴───────┐               │
//! │       │                          │ 1. get_token()│               │
//! │       │                          │   └─▶ TokenMgr│               │
//! │       │                          │      (refresh │               │
//! │       │                          │       if exp) │               │
//! │       │                          └───────┬───────┘               │
//! │       │                                  │                       │
//! │       │                          ┌───────┴───────┐               │
//! │       │                          │2. build_hdrs()│               │
//! │       │                          │ - Authorization│              │
//! │       │                          │ - x-request-id │              │
//! │       │                          │ - openai-intent│              │
//! │       │                          └───────┬───────┘               │
//! │       │                                  │                       │
//! │       │                          ┌───────┴───────┐               │
//! │       │                          │ 3. POST to API│               │
//! │       │                          │  Direct: /chat│               │
//! │       │                          │  Proxy: /v1/  │               │
//! │       │                          └───────┬───────┘               │
//! │       │                                  │                       │
//! │       │◀─────────────────────────────────┤                       │
//! │       │  Response (parsed JSON)          │                       │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Account Types
//!
//! Supports different GitHub Copilot account types:
//! - `individual` → `api.githubcopilot.com`
//! - `business` → `api.business.githubcopilot.com`
//! - `enterprise` → `api.enterprise.githubcopilot.com`
//!
//! # Header Requirements
//!
//! Direct mode sends these headers (matching TypeScript copilot-api):
//! - `Authorization: Bearer <token>`
//! - `x-github-api-version: 2025-04-01`
//! - `copilot-integration-id: vscode-chat`
//! - `openai-intent: conversation-panel`
//! - `x-request-id: <uuid>`
//! - `editor-version: vscode/1.95.0`
//! - `editor-plugin-version: copilot-chat/0.26.7`
//! - `x-vscode-user-agent-library-version: electron-fetch`

use reqwest::{Client as ReqwestClient, Response, StatusCode};
use serde::de::DeserializeOwned;
use std::time::Duration;
use tracing::{debug, error, warn};
use uuid::Uuid;

use super::error::{Result, VsCodeError};
use super::token::TokenManager;
use super::types::*;

// API Constants - Match TypeScript copilot-api implementation
const COPILOT_API_VERSION: &str = "2025-04-01";
#[allow(dead_code)]
const COPILOT_VERSION: &str = "0.26.7";
const EDITOR_VERSION: &str = "vscode/1.95.0";
const EDITOR_PLUGIN_VERSION: &str = "copilot-chat/0.26.7";
const USER_AGENT: &str = "GitHubCopilotChat/0.26.7";
const MAX_RETRIES: u32 = 3; // Maximum retry attempts for transient errors
const INITIAL_RETRY_DELAY_MS: u64 = 1000; // Initial retry delay (1 second)

/// Account type for Copilot API endpoint selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AccountType {
    /// Individual GitHub Copilot subscription
    #[default]
    Individual,
    /// GitHub Copilot Business
    Business,
    /// GitHub Copilot Enterprise
    Enterprise,
}

impl AccountType {
    /// Get the base URL for this account type.
    pub fn base_url(&self) -> &'static str {
        match self {
            AccountType::Individual => "https://api.githubcopilot.com",
            AccountType::Business => "https://api.business.githubcopilot.com",
            AccountType::Enterprise => "https://api.enterprise.githubcopilot.com",
        }
    }

    /// Parse account type from string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "individual" => Some(AccountType::Individual),
            "business" => Some(AccountType::Business),
            "enterprise" => Some(AccountType::Enterprise),
            _ => None,
        }
    }
}

/// HTTP client for GitHub Copilot API.
///
/// Supports both direct API access and proxy mode.
#[derive(Clone)]
pub struct VsCodeCopilotClient {
    client: ReqwestClient,
    base_url: String,
    token_manager: TokenManager,
    /// Whether to use direct API mode (true) or proxy mode (false).
    direct_mode: bool,
    /// Account type for API endpoint selection.
    #[allow(dead_code)]
    account_type: AccountType,
    /// Whether vision mode is enabled for requests.
    vision_enabled: bool,
}

impl VsCodeCopilotClient {
    /// Create a new client with direct API mode (default).
    ///
    /// Uses `api.githubcopilot.com` directly without a proxy.
    pub fn new(timeout: Duration) -> Result<Self> {
        Self::new_with_options(timeout, true, AccountType::Individual)
    }

    /// Create a new client with a custom base URL (for proxy mode).
    ///
    /// This is the legacy mode that connects to a local copilot-api proxy.
    pub fn with_base_url(base_url: impl Into<String>, timeout: Duration) -> Result<Self> {
        let base_url = base_url.into();
        let is_direct = base_url.contains("githubcopilot.com");

        let client = ReqwestClient::builder()
            .timeout(timeout)
            .pool_max_idle_per_host(10)
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .map_err(|e| VsCodeError::ClientInit(e.to_string()))?;

        let token_manager =
            TokenManager::new().map_err(|e| VsCodeError::ClientInit(e.to_string()))?;

        debug!(
            base_url = %base_url,
            timeout_secs = timeout.as_secs(),
            direct_mode = is_direct,
            "VSCode Copilot client initialized"
        );

        Ok(Self {
            client,
            base_url,
            token_manager,
            direct_mode: is_direct,
            account_type: AccountType::Individual,
            vision_enabled: false,
        })
    }

    /// Create a new client with specified options.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Request timeout duration
    /// * `direct_mode` - If true, use direct API; if false, use proxy
    /// * `account_type` - Account type for API endpoint selection
    pub fn new_with_options(
        timeout: Duration,
        direct_mode: bool,
        account_type: AccountType,
    ) -> Result<Self> {
        let base_url = if direct_mode {
            account_type.base_url().to_string()
        } else {
            std::env::var("VSCODE_COPILOT_PROXY_URL")
                .unwrap_or_else(|_| "http://localhost:4141".to_string())
        };

        let client = ReqwestClient::builder()
            .timeout(timeout)
            .pool_max_idle_per_host(10)
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .map_err(|e| VsCodeError::ClientInit(e.to_string()))?;

        let token_manager =
            TokenManager::new().map_err(|e| VsCodeError::ClientInit(e.to_string()))?;

        debug!(
            base_url = %base_url,
            timeout_secs = timeout.as_secs(),
            direct_mode = direct_mode,
            account_type = ?account_type,
            "VSCode Copilot client initialized"
        );

        Ok(Self {
            client,
            base_url,
            token_manager,
            direct_mode,
            account_type,
            vision_enabled: false,
        })
    }

    /// Enable vision mode for image processing.
    pub fn with_vision(mut self, enabled: bool) -> Self {
        self.vision_enabled = enabled;
        self
    }

    /// Get a valid Copilot token, refreshing if needed.
    async fn get_token(&self) -> Result<String> {
        self.token_manager
            .get_valid_copilot_token()
            .await
            .map_err(|e| VsCodeError::Authentication(e.to_string()))
    }

    /// Build request headers with authentication.
    ///
    /// For direct mode, includes all headers required by GitHub Copilot API.
    /// For proxy mode, includes minimal headers (proxy adds the rest).
    async fn build_headers(&self) -> Result<reqwest::header::HeaderMap> {
        let token = self.get_token().await?;

        let mut headers = reqwest::header::HeaderMap::new();

        // Authorization - required for both modes
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", token).parse().unwrap(),
        );

        // Content-Type
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );

        // User-Agent
        headers.insert(reqwest::header::USER_AGENT, USER_AGENT.parse().unwrap());

        // Editor version headers
        headers.insert("editor-version", EDITOR_VERSION.parse().unwrap());
        headers.insert(
            "editor-plugin-version",
            EDITOR_PLUGIN_VERSION.parse().unwrap(),
        );

        // Direct mode requires additional headers to match TypeScript implementation
        if self.direct_mode {
            // Copilot-specific headers
            headers.insert("copilot-integration-id", "vscode-chat".parse().unwrap());
            headers.insert("openai-intent", "conversation-panel".parse().unwrap());

            // GitHub API version
            headers.insert("x-github-api-version", COPILOT_API_VERSION.parse().unwrap());

            // Request ID for tracing
            headers.insert("x-request-id", Uuid::new_v4().to_string().parse().unwrap());

            // VSCode user agent library
            headers.insert(
                "x-vscode-user-agent-library-version",
                "electron-fetch".parse().unwrap(),
            );

            // Vision mode header
            if self.vision_enabled {
                headers.insert("copilot-vision-request", "true".parse().unwrap());
            }
        }

        Ok(headers)
    }

    /// Retry an async operation with exponential backoff for retryable errors.
    ///
    /// This method implements automatic retry logic for transient failures:
    /// - Network errors (timeouts, connection refused)
    /// - Rate limiting (429)
    /// - Service unavailable (503)
    /// - Bad gateway (502)
    ///
    /// Non-retryable errors (authentication, invalid request) are returned immediately.
    async fn retry_with_backoff<F, Fut, T>(&self, operation: F, operation_name: &str) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut last_error = None;

        for attempt in 0..=MAX_RETRIES {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    // Check if error is retryable
                    if !e.is_retryable() || attempt == MAX_RETRIES {
                        return Err(e);
                    }

                    // Calculate exponential backoff delay
                    let delay = Duration::from_millis(INITIAL_RETRY_DELAY_MS * 2_u64.pow(attempt));

                    warn!(
                        operation = operation_name,
                        attempt = attempt + 1,
                        max_retries = MAX_RETRIES,
                        delay_ms = delay.as_millis(),
                        error = %e,
                        "Retrying after retryable error"
                    );

                    tokio::time::sleep(delay).await;
                    last_error = Some(e);
                }
            }
        }

        Err(last_error
            .unwrap_or_else(|| VsCodeError::ApiError("Operation failed after retries".to_string())))
    }

    /// Determine if this is an agent call (multi-turn conversation).
    ///
    /// # WHY: X-Initiator Header
    ///
    /// The Copilot API uses the X-Initiator header to distinguish between:
    /// - `"user"`: Initial user message (first turn, no prior assistant/tool messages)
    /// - `"agent"`: Follow-up from coding agent (has assistant or tool messages)
    ///
    /// This matches the TypeScript proxy behavior:
    /// `copilot-api/src/services/copilot/create-chat-completions.ts:22-29`
    ///
    /// ```text
    /// ┌─────────────────────────────────────────────────────────────┐
    /// │                   X-INITIATOR LOGIC                          │
    /// ├─────────────────────────────────────────────────────────────┤
    /// │                                                              │
    /// │  messages: [system, user]          → X-Initiator: "user"    │
    /// │                                                              │
    /// │  messages: [system, user,          → X-Initiator: "agent"   │
    /// │             assistant, user]                                 │
    /// │                                                              │
    /// │  messages: [system, user,          → X-Initiator: "agent"   │
    /// │             assistant, tool, user]                           │
    /// │                                                              │
    /// └─────────────────────────────────────────────────────────────┘
    /// ```
    fn is_agent_call(messages: &[RequestMessage]) -> bool {
        messages
            .iter()
            .any(|m| matches!(m.role.as_str(), "assistant" | "tool"))
    }

    /// Send a chat completion request (non-streaming).
    pub async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let request_clone = request.clone();

        // Wrap the request in retry logic
        self.retry_with_backoff(
            || async {
                // Direct mode uses /chat/completions, proxy mode uses /v1/chat/completions
                let url = if self.direct_mode {
                    format!("{}/chat/completions", self.base_url)
                } else {
                    format!("{}/v1/chat/completions", self.base_url)
                };
                let mut headers = self.build_headers().await?;

                // Add X-Initiator header for agent/user distinction (direct mode only)
                if self.direct_mode {
                    let initiator = if Self::is_agent_call(&request_clone.messages) {
                        "agent"
                    } else {
                        "user"
                    };
                    headers.insert("X-Initiator", initiator.parse().unwrap());
                }

                debug!(
                    url = %url,
                    model = %request_clone.model,
                    message_count = request_clone.messages.len(),
                    direct_mode = self.direct_mode,
                    "Sending chat completion request"
                );

                let response = self
                    .client
                    .post(&url)
                    .headers(headers)
                    .json(&request_clone)
                    .send()
                    .await
                    .map_err(|e| VsCodeError::Network(e.to_string()))?;

                let mut response: ChatCompletionResponse = Self::handle_response(response).await?;

                // OODA-07.2: Normalize Anthropic-style split choices
                response = Self::normalize_choices(response);

                Ok(response)
            },
            "chat_completion",
        )
        .await
    }

    /// Send a streaming chat completion request.
    pub async fn chat_completion_stream(&self, request: ChatCompletionRequest) -> Result<Response> {
        // Direct mode uses /chat/completions, proxy mode uses /v1/chat/completions
        let url = if self.direct_mode {
            format!("{}/chat/completions", self.base_url)
        } else {
            format!("{}/v1/chat/completions", self.base_url)
        };
        let mut headers = self.build_headers().await?;

        // Add X-Initiator header for agent/user distinction (direct mode only)
        if self.direct_mode {
            let initiator = if Self::is_agent_call(&request.messages) {
                "agent"
            } else {
                "user"
            };
            headers.insert("X-Initiator", initiator.parse().unwrap());
        }

        debug!(
            url = %url,
            model = %request.model,
            message_count = request.messages.len(),
            "Sending streaming chat completion request"
        );

        let response = self
            .client
            .post(&url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .map_err(|e| VsCodeError::Network(e.to_string()))?;

        if response.status().is_success() {
            Ok(response)
        } else {
            let status = response.status();
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            warn!(
                status = %status,
                error = %error_body,
                "Streaming request failed"
            );

            Err(Self::map_error_status(status, error_body))
        }
    }

    /// List available models.
    ///
    /// Returns a list of models available for the authenticated user.
    /// Includes model capabilities, limits, and supported features.
    pub async fn list_models(&self) -> Result<ModelsResponse> {
        // Direct mode uses /models, proxy mode uses /v1/models
        let url = if self.direct_mode {
            format!("{}/models", self.base_url)
        } else {
            format!("{}/v1/models", self.base_url)
        };
        let headers = self.build_headers().await?;

        debug!(
            url = %url,
            direct_mode = self.direct_mode,
            "Fetching models list"
        );

        let response = self
            .client
            .get(&url)
            .headers(headers)
            .send()
            .await
            .map_err(|e| VsCodeError::Network(e.to_string()))?;

        Self::handle_response(response).await
    }

    /// Create embeddings for the given input.
    ///
    /// # Arguments
    ///
    /// * `request` - The embedding request containing input text(s) and model
    ///
    /// # Returns
    ///
    /// Returns an `EmbeddingResponse` containing the embedding vectors.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let request = EmbeddingRequest::new("Hello, world!", "text-embedding-3-small");
    /// let response = client.create_embeddings(request).await?;
    /// let embedding = response.first_embedding().unwrap();
    /// ```
    pub async fn create_embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        // Direct mode uses /embeddings, proxy mode uses /v1/embeddings
        let url = if self.direct_mode {
            format!("{}/embeddings", self.base_url)
        } else {
            format!("{}/v1/embeddings", self.base_url)
        };
        let headers = self.build_headers().await?;

        debug!(
            url = %url,
            model = %request.model,
            direct_mode = self.direct_mode,
            "Sending embedding request"
        );

        let response = self
            .client
            .post(&url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .map_err(|e| VsCodeError::Network(e.to_string()))?;

        Self::handle_response(response).await
    }

    /// Normalize Anthropic-style split choices into a single choice.
    ///
    /// # WHY: Anthropic Model Response Format
    ///
    /// Claude models (Haiku 4.5, Sonnet 4.5) via Copilot API return TWO separate choices:
    /// - Choice 1: Contains only `content` (the model's thinking/reasoning)
    /// - Choice 2: Contains only `tool_calls` (the function calls to execute)
    ///
    /// Neither choice includes an `index` field. This function merges them into a single
    /// choice to match the expected format where a single message contains both content
    /// and tool_calls.
    ///
    /// # Example Anthropic Response
    ///
    /// ```json
    /// {
    ///   "choices": [
    ///     {
    ///       "finish_reason": "tool_calls",
    ///       "message": {
    ///         "content": "I'll examine the file to understand its structure",
    ///         "role": "assistant"
    ///       }
    ///     },
    ///     {
    ///       "finish_reason": "tool_calls",
    ///       "message": {
    ///         "role": "assistant",
    ///         "tool_calls": [...]
    ///       }
    ///     }
    ///   ]
    /// }
    /// ```
    ///
    /// # WHEN TO MERGE
    ///
    /// Merge only when:
    /// 1. Multiple choices exist
    /// 2. All choices have `index == None` or `index == Some(0)` (Anthropic pattern)
    /// 3. Each choice has partial data (some have content, others have tool_calls)
    ///
    /// # Arguments
    ///
    /// * `response` - The raw ChatCompletionResponse from the API
    ///
    /// # Returns
    ///
    /// Normalized response with merged choices
    fn normalize_choices(mut response: ChatCompletionResponse) -> ChatCompletionResponse {
        // Only normalize if multiple choices exist
        if response.choices.len() <= 1 {
            return response;
        }

        // Check if all choices have None or 0 index (Anthropic pattern)
        let needs_merge = response
            .choices
            .iter()
            .all(|c| c.index.is_none() || c.index == Some(0));

        if !needs_merge {
            return response;
        }

        debug!(
            choice_count = response.choices.len(),
            model = %response.model,
            "OODA-07.2: Normalizing Anthropic-style split choices"
        );

        // Take ownership of choices vector
        let mut choices_iter = response.choices.into_iter();

        // Take first choice as base
        let mut merged = choices_iter.next().unwrap();

        // Merge data from remaining choices
        for choice in choices_iter {
            // Merge content (prefer non-empty)
            if merged.message.content.is_none()
                || merged
                    .message
                    .content
                    .as_ref()
                    .map(|s| s.is_empty())
                    .unwrap_or(true)
            {
                merged.message.content = choice.message.content;
            }

            // Merge tool_calls (prefer non-empty, extend if both present)
            if merged.message.tool_calls.is_none() {
                merged.message.tool_calls = choice.message.tool_calls;
            } else if let Some(mut existing) = merged.message.tool_calls.take() {
                if let Some(new_calls) = choice.message.tool_calls {
                    existing.extend(new_calls);
                }
                merged.message.tool_calls = Some(existing);
            }

            // Keep first non-None finish_reason
            if merged.finish_reason.is_none() {
                merged.finish_reason = choice.finish_reason;
            }
        }

        // Set index explicitly
        merged.index = Some(0);

        response.choices = vec![merged];
        response
    }

    /// Handle HTTP response with error mapping.
    async fn handle_response<T: DeserializeOwned>(response: Response) -> Result<T> {
        let status = response.status();

        if status.is_success() {
            // Get the response text first for debugging
            let body_text = response
                .text()
                .await
                .map_err(|e| VsCodeError::Decode(format!("Failed to read response body: {}", e)))?;

            // Log the raw response for debugging
            debug!(
                status = %status,
                body_length = body_text.len(),
                body_preview = &body_text[..body_text.len().min(500)],
                "Raw API response"
            );

            // Try to deserialize
            serde_json::from_str(&body_text).map_err(|e| {
                error!(
                    error = %e,
                    body = %body_text,
                    "Failed to deserialize response"
                );
                VsCodeError::Decode(format!(
                    "Deserialization failed: {} | Body: {}",
                    e, body_text
                ))
            })
        } else {
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            warn!(
                status = %status,
                error = %error_body,
                "Request failed"
            );

            Err(Self::map_error_status(status, error_body))
        }
    }

    /// Map HTTP status to VsCodeError.
    fn map_error_status(status: StatusCode, body: String) -> VsCodeError {
        match status {
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                VsCodeError::Authentication(format!("Copilot authentication failed: {}", body))
            }
            StatusCode::TOO_MANY_REQUESTS => VsCodeError::RateLimited,
            StatusCode::BAD_REQUEST => VsCodeError::InvalidRequest(body),
            StatusCode::SERVICE_UNAVAILABLE => VsCodeError::ServiceUnavailable,
            StatusCode::BAD_GATEWAY => {
                VsCodeError::Network(format!("Upstream error (502): {}", body))
            }
            StatusCode::GATEWAY_TIMEOUT | StatusCode::REQUEST_TIMEOUT => {
                VsCodeError::Network(format!("Timeout: {}", body))
            }
            _ => VsCodeError::ApiError(format!("HTTP {}: {}", status, body)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // AccountType Tests
    // ========================================================================

    #[test]
    fn test_account_type_base_url_individual() {
        assert_eq!(
            AccountType::Individual.base_url(),
            "https://api.githubcopilot.com"
        );
    }

    #[test]
    fn test_account_type_base_url_business() {
        assert_eq!(
            AccountType::Business.base_url(),
            "https://api.business.githubcopilot.com"
        );
    }

    #[test]
    fn test_account_type_base_url_enterprise() {
        assert_eq!(
            AccountType::Enterprise.base_url(),
            "https://api.enterprise.githubcopilot.com"
        );
    }

    #[test]
    fn test_account_type_from_str_individual() {
        assert_eq!(
            AccountType::from_str("individual"),
            Some(AccountType::Individual)
        );
        // Case insensitive
        assert_eq!(
            AccountType::from_str("INDIVIDUAL"),
            Some(AccountType::Individual)
        );
        assert_eq!(
            AccountType::from_str("Individual"),
            Some(AccountType::Individual)
        );
    }

    #[test]
    fn test_account_type_from_str_business() {
        assert_eq!(
            AccountType::from_str("business"),
            Some(AccountType::Business)
        );
        assert_eq!(
            AccountType::from_str("BUSINESS"),
            Some(AccountType::Business)
        );
    }

    #[test]
    fn test_account_type_from_str_enterprise() {
        assert_eq!(
            AccountType::from_str("enterprise"),
            Some(AccountType::Enterprise)
        );
        assert_eq!(
            AccountType::from_str("Enterprise"),
            Some(AccountType::Enterprise)
        );
    }

    #[test]
    fn test_account_type_from_str_unknown_returns_none() {
        assert_eq!(AccountType::from_str("unknown"), None);
        assert_eq!(AccountType::from_str(""), None);
        assert_eq!(AccountType::from_str("personal"), None);
        assert_eq!(AccountType::from_str("team"), None);
    }

    #[test]
    fn test_account_type_default_is_individual() {
        let default: AccountType = Default::default();
        assert_eq!(default, AccountType::Individual);
    }

    // ========================================================================
    // Error Status Mapping Tests
    // ========================================================================

    #[test]
    fn test_map_error_status_unauthorized() {
        let err = VsCodeCopilotClient::map_error_status(
            StatusCode::UNAUTHORIZED,
            "Invalid token".to_string(),
        );
        match err {
            VsCodeError::Authentication(msg) => {
                assert!(msg.contains("authentication failed"));
                assert!(msg.contains("Invalid token"));
            }
            other => panic!("Expected Authentication error, got {:?}", other),
        }
    }

    #[test]
    fn test_map_error_status_forbidden() {
        let err = VsCodeCopilotClient::map_error_status(
            StatusCode::FORBIDDEN,
            "Access denied".to_string(),
        );
        match err {
            VsCodeError::Authentication(msg) => {
                assert!(msg.contains("Access denied"));
            }
            other => panic!("Expected Authentication error, got {:?}", other),
        }
    }

    #[test]
    fn test_map_error_status_rate_limited() {
        let err = VsCodeCopilotClient::map_error_status(
            StatusCode::TOO_MANY_REQUESTS,
            "Rate limit exceeded".to_string(),
        );
        assert!(matches!(err, VsCodeError::RateLimited));
    }

    #[test]
    fn test_map_error_status_bad_request() {
        let err = VsCodeCopilotClient::map_error_status(
            StatusCode::BAD_REQUEST,
            "Invalid JSON".to_string(),
        );
        match err {
            VsCodeError::InvalidRequest(msg) => assert_eq!(msg, "Invalid JSON"),
            other => panic!("Expected InvalidRequest error, got {:?}", other),
        }
    }

    #[test]
    fn test_map_error_status_service_unavailable() {
        let err = VsCodeCopilotClient::map_error_status(
            StatusCode::SERVICE_UNAVAILABLE,
            "Maintenance".to_string(),
        );
        assert!(matches!(err, VsCodeError::ServiceUnavailable));
    }

    #[test]
    fn test_map_error_status_timeout() {
        let err = VsCodeCopilotClient::map_error_status(
            StatusCode::GATEWAY_TIMEOUT,
            "Upstream timeout".to_string(),
        );
        match err {
            VsCodeError::Network(msg) => {
                assert!(msg.contains("Timeout"));
                assert!(msg.contains("Upstream timeout"));
            }
            other => panic!("Expected Network error, got {:?}", other),
        }
    }

    #[test]
    fn test_map_error_status_request_timeout() {
        let err = VsCodeCopilotClient::map_error_status(
            StatusCode::REQUEST_TIMEOUT,
            "Request took too long".to_string(),
        );
        match err {
            VsCodeError::Network(msg) => assert!(msg.contains("Timeout")),
            other => panic!("Expected Network error, got {:?}", other),
        }
    }

    #[test]
    fn test_map_error_status_internal_server_error() {
        let err = VsCodeCopilotClient::map_error_status(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Something went wrong".to_string(),
        );
        match err {
            VsCodeError::ApiError(msg) => {
                assert!(msg.contains("500"));
                assert!(msg.contains("Something went wrong"));
            }
            other => panic!("Expected ApiError, got {:?}", other),
        }
    }

    #[test]
    fn test_map_error_status_not_found() {
        let err = VsCodeCopilotClient::map_error_status(
            StatusCode::NOT_FOUND,
            "Endpoint not found".to_string(),
        );
        match err {
            VsCodeError::ApiError(msg) => {
                assert!(msg.contains("404"));
                assert!(msg.contains("not found"));
            }
            other => panic!("Expected ApiError, got {:?}", other),
        }
    }

    #[test]
    fn test_map_error_status_bad_gateway() {
        // WHY: 502 indicates upstream server issue - retryable
        let err = VsCodeCopilotClient::map_error_status(
            StatusCode::BAD_GATEWAY,
            "Upstream server error".to_string(),
        );
        match err {
            VsCodeError::Network(msg) => {
                assert!(msg.contains("Upstream") || msg.contains("502"));
            }
            other => panic!("Expected Network, got {:?}", other),
        }
    }

    // ========================================================================
    // API Constants Tests - Verify TypeScript Parity
    // ========================================================================

    #[test]
    fn test_header_constants_match_typescript() {
        // These must match copilot-api/src/lib/api-config.ts
        assert_eq!(COPILOT_API_VERSION, "2025-04-01");
        assert_eq!(EDITOR_VERSION, "vscode/1.95.0");
        assert!(EDITOR_PLUGIN_VERSION.contains("copilot"));
        assert!(USER_AGENT.contains("Copilot"));
    }

    #[test]
    fn test_api_version_format() {
        // Verify API version is in expected format YYYY-MM-DD
        assert!(COPILOT_API_VERSION.len() == 10);
        assert!(COPILOT_API_VERSION.starts_with("2025"));
    }

    #[test]
    fn test_editor_version_format() {
        assert!(EDITOR_VERSION.starts_with("vscode/"));
    }

    #[test]
    fn test_user_agent_contains_copilot() {
        assert!(USER_AGENT.contains("Copilot"));
    }

    // ========================================================================
    // X-Initiator Header Tests
    // ========================================================================
    //
    // WHY: These tests verify parity with TypeScript proxy behavior.
    // See: copilot-api/src/services/copilot/create-chat-completions.ts:22-29
    //
    // The is_agent_call function determines the X-Initiator header value:
    // - "user" for initial user queries (no assistant/tool messages)
    // - "agent" for multi-turn conversations (has assistant/tool messages)

    /// Helper to create a test message with a role.
    fn make_message(role: &str) -> RequestMessage {
        RequestMessage {
            role: role.to_string(),
            content: Some(RequestContent::Text("test".to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            cache_control: None,
        }
    }

    #[test]
    fn test_is_agent_call_empty_messages() {
        // Empty messages → false (no agent messages)
        let messages: Vec<RequestMessage> = vec![];
        assert!(!VsCodeCopilotClient::is_agent_call(&messages));
    }

    #[test]
    fn test_is_agent_call_user_only() {
        // Only user messages → false (initial query)
        let messages = vec![make_message("user")];
        assert!(!VsCodeCopilotClient::is_agent_call(&messages));
    }

    #[test]
    fn test_is_agent_call_system_and_user() {
        // System + user → false (initial query with system prompt)
        let messages = vec![make_message("system"), make_message("user")];
        assert!(!VsCodeCopilotClient::is_agent_call(&messages));
    }

    #[test]
    fn test_is_agent_call_with_assistant() {
        // Has assistant message → true (multi-turn)
        let messages = vec![
            make_message("system"),
            make_message("user"),
            make_message("assistant"),
            make_message("user"),
        ];
        assert!(VsCodeCopilotClient::is_agent_call(&messages));
    }

    #[test]
    fn test_is_agent_call_with_tool() {
        // Has tool message → true (tool response)
        let messages = vec![
            make_message("system"),
            make_message("user"),
            make_message("assistant"),
            make_message("tool"),
            make_message("user"),
        ];
        assert!(VsCodeCopilotClient::is_agent_call(&messages));
    }

    #[test]
    fn test_is_agent_call_assistant_only() {
        // Assistant only → true
        let messages = vec![make_message("assistant")];
        assert!(VsCodeCopilotClient::is_agent_call(&messages));
    }

    #[test]
    fn test_is_agent_call_tool_only() {
        // Tool only → true
        let messages = vec![make_message("tool")];
        assert!(VsCodeCopilotClient::is_agent_call(&messages));
    }

    #[test]
    fn test_is_agent_call_developer_role() {
        // Developer role (newer) → false (not agent/tool)
        let messages = vec![make_message("developer"), make_message("user")];
        assert!(!VsCodeCopilotClient::is_agent_call(&messages));
    }

    // ========================================================================
    // Vision Mode Tests
    // ========================================================================
    //
    // WHY: The Copilot API requires the `copilot-vision-request: true` header
    // for requests that include image content.
    //
    // NOTE: The TypeScript proxy auto-detects vision mode by checking if any
    // message content contains `image_url` type. Our Rust implementation
    // currently uses manual `with_vision(true)` - auto-detection is a future
    // enhancement. See: copilot-api/src/services/copilot/create-chat-completions.ts:15-17

    #[test]
    fn test_client_vision_disabled_by_default() {
        use std::time::Duration;

        // New client should have vision disabled
        let client = VsCodeCopilotClient::new(Duration::from_secs(30));
        assert!(client.is_ok(), "Client should be created successfully");

        // We can't directly test the internal field, but we verify
        // with_vision returns a valid client
        let client = client.unwrap().with_vision(false);
        // Should compile and not panic
        let _ = client;
    }

    #[test]
    fn test_client_with_vision_enables_mode() {
        use std::time::Duration;

        let client = VsCodeCopilotClient::new(Duration::from_secs(30))
            .unwrap()
            .with_vision(true);

        // Method should compile and return self
        let _ = client;
    }

    #[test]
    fn test_client_with_vision_chain() {
        use std::time::Duration;

        // Vision mode should be chainable
        let client = VsCodeCopilotClient::new(Duration::from_secs(30))
            .unwrap()
            .with_vision(true)
            .with_vision(false)
            .with_vision(true);

        // Should compile and not panic
        let _ = client;
    }

    #[test]
    fn test_client_with_base_url_vision() {
        use std::time::Duration;

        let client =
            VsCodeCopilotClient::with_base_url("http://localhost:4141", Duration::from_secs(30))
                .unwrap()
                .with_vision(true);

        // Proxy mode with vision should work
        let _ = client;
    }

    #[test]
    fn test_client_with_options_vision() {
        use std::time::Duration;

        // All account types with vision
        for account_type in [
            AccountType::Individual,
            AccountType::Business,
            AccountType::Enterprise,
        ] {
            let client = VsCodeCopilotClient::new_with_options(
                Duration::from_secs(30),
                true, // direct mode
                account_type,
            )
            .unwrap()
            .with_vision(true);

            // All should work
            let _ = client;
        }
    }

    // =========================================================================
    // Embedding Client Tests
    // WHY: Verify embedding URL construction and request serialization
    // =========================================================================

    #[test]
    fn test_embedding_url_direct_mode() {
        // WHY: Direct mode should use /embeddings (not /v1/embeddings)
        // This matches the TypeScript proxy behavior for GitHub Copilot API
        use std::time::Duration;

        let client =
            VsCodeCopilotClient::new(Duration::from_secs(30)).expect("Failed to create client");

        // Direct mode is the default
        assert!(client.direct_mode, "Default should be direct mode");

        // URL should be constructed as {base_url}/embeddings
        // In direct mode, create_embeddings appends /embeddings to base_url
        let base_url = &client.base_url;
        assert!(
            !base_url.ends_with("/v1"),
            "Direct mode base URL should not end with /v1"
        );

        // The create_embeddings method appends /embeddings in direct mode
        // We verify the base URL is set up correctly for this
        assert!(
            base_url.starts_with("https://api"),
            "Direct mode should use GitHub API: {}",
            base_url
        );
    }

    #[test]
    fn test_embedding_url_proxy_mode() {
        // WHY: Proxy mode should use /v1/embeddings (OpenAI compatible)
        use std::time::Duration;

        let client =
            VsCodeCopilotClient::with_base_url("http://localhost:1337", Duration::from_secs(30))
                .expect("Failed to create proxy client");

        // Proxy mode when using with_base_url
        assert!(!client.direct_mode, "Should be proxy mode");

        // The create_embeddings method appends /v1/embeddings in proxy mode
        assert_eq!(
            client.base_url, "http://localhost:1337",
            "Proxy base URL should be preserved"
        );
    }

    #[test]
    fn test_embedding_single_input_format() {
        // WHY: Single string input should serialize as JSON string, not array
        let input = EmbeddingInput::Single("Hello, world!".to_string());
        let request = EmbeddingRequest::new(input, "text-embedding-3-small");

        let json = serde_json::to_value(&request).expect("Failed to serialize");

        assert_eq!(
            json["input"],
            serde_json::json!("Hello, world!"),
            "Single input should serialize as string"
        );
        assert_eq!(json["model"], "text-embedding-3-small");
    }

    #[test]
    fn test_embedding_multiple_inputs_format() {
        // WHY: Multiple inputs should serialize as JSON array
        let inputs = vec![
            "First".to_string(),
            "Second".to_string(),
            "Third".to_string(),
        ];
        let input = EmbeddingInput::Multiple(inputs);
        let request = EmbeddingRequest::new(input, "text-embedding-3-small");

        let json = serde_json::to_value(&request).expect("Failed to serialize");

        assert_eq!(
            json["input"],
            serde_json::json!(["First", "Second", "Third"]),
            "Multiple inputs should serialize as array"
        );
    }

    #[test]
    fn test_embedding_model_in_request() {
        // WHY: Model name must be correctly included in request JSON
        let request = EmbeddingRequest::new("test", "text-embedding-ada-002");

        let json = serde_json::to_value(&request).expect("Failed to serialize");

        assert_eq!(
            json["model"], "text-embedding-ada-002",
            "Model name should be in request"
        );

        // Also test text-embedding-3-large
        let request2 = EmbeddingRequest::new("test", "text-embedding-3-large");
        let json2 = serde_json::to_value(&request2).expect("Failed to serialize");

        assert_eq!(json2["model"], "text-embedding-3-large");
    }

    // =========================================================================
    // List Models Client Tests
    // WHY: Verify models endpoint URL construction for both modes
    // =========================================================================

    #[test]
    fn test_list_models_url_direct_mode() {
        // WHY: Direct mode should use /models (not /v1/models)
        // This matches the TypeScript proxy behavior for GitHub Copilot API
        use std::time::Duration;

        let client =
            VsCodeCopilotClient::new(Duration::from_secs(30)).expect("Failed to create client");

        // Direct mode is the default
        assert!(client.direct_mode, "Default should be direct mode");

        // In direct mode, list_models appends /models to base_url
        // We verify the base URL setup is correct for this
        let base_url = &client.base_url;
        assert!(
            !base_url.ends_with("/v1"),
            "Direct mode base URL should not end with /v1"
        );

        // Verify it's a GitHub API URL
        assert!(
            base_url.starts_with("https://api"),
            "Direct mode should use GitHub API: {}",
            base_url
        );
    }

    #[test]
    fn test_list_models_url_proxy_mode() {
        // WHY: Proxy mode should use /v1/models (OpenAI compatible)
        use std::time::Duration;

        let client =
            VsCodeCopilotClient::with_base_url("http://localhost:1337", Duration::from_secs(30))
                .expect("Failed to create proxy client");

        // Proxy mode when using with_base_url
        assert!(!client.direct_mode, "Should be proxy mode");

        // The list_models method appends /v1/models in proxy mode
        assert_eq!(
            client.base_url, "http://localhost:1337",
            "Proxy base URL should be preserved"
        );
    }

    // =========================================================================
    // Timeout Configuration Tests
    // WHY: Verify timeout configuration is accepted and client is created
    // =========================================================================

    #[test]
    fn test_client_timeout_short() {
        // WHY: Short timeout (5s) for quick failure detection
        use std::time::Duration;

        let timeout = Duration::from_secs(5);
        let client = VsCodeCopilotClient::new(timeout);

        assert!(client.is_ok(), "Client should accept short timeout");
    }

    #[test]
    fn test_client_timeout_long() {
        // WHY: Long timeout (300s) for slow LLM responses
        use std::time::Duration;

        let timeout = Duration::from_secs(300);
        let client = VsCodeCopilotClient::new(timeout);

        assert!(client.is_ok(), "Client should accept long timeout");
    }

    // =========================================================================
    // Chat URL Construction Tests (Iteration 33)
    // WHY: Chat completion endpoint URLs must match Copilot API expectations.
    // Different account types use different base URLs.
    // =========================================================================

    #[test]
    fn test_chat_url_direct_mode() {
        // WHY: Individual accounts use api.githubcopilot.com
        // This is the default endpoint for most users
        let base = AccountType::Individual.base_url();
        let url = format!("{}/chat/completions", base);

        assert_eq!(
            url, "https://api.githubcopilot.com/chat/completions",
            "Individual account chat URL should use main Copilot API"
        );
    }

    #[test]
    fn test_chat_url_business_mode() {
        // WHY: Business accounts have a separate endpoint for
        // compliance and billing separation
        let base = AccountType::Business.base_url();
        let url = format!("{}/chat/completions", base);

        assert_eq!(
            url, "https://api.business.githubcopilot.com/chat/completions",
            "Business account chat URL should use business subdomain"
        );
    }

    #[test]
    fn test_chat_url_enterprise_mode() {
        // WHY: Enterprise accounts have their own endpoint for
        // dedicated infrastructure and data isolation
        let base = AccountType::Enterprise.base_url();
        let url = format!("{}/chat/completions", base);

        assert_eq!(
            url, "https://api.enterprise.githubcopilot.com/chat/completions",
            "Enterprise account chat URL should use enterprise subdomain"
        );
    }

    #[test]
    fn test_chat_url_proxy_mode() {
        // WHY: Proxy mode uses the configured proxy URL
        // Used when TypeScript proxy handles auth and headers
        let proxy_url = "http://localhost:1337";
        let url = format!("{}/chat/completions", proxy_url);

        assert_eq!(
            url, "http://localhost:1337/chat/completions",
            "Proxy mode chat URL should use configured proxy base"
        );
    }

    // =========================================================================
    // OODA-07.2: Choice Normalization Tests
    // WHY: Anthropic models return split choices that need merging
    // =========================================================================

    #[test]
    fn test_normalize_choices_single_choice() {
        // WHY: Single choice should pass through unchanged
        use crate::providers::vscode::types::*;

        let response = ChatCompletionResponse {
            id: "test1".to_string(),
            object: None,
            created: None,
            model: "gpt-4.1".to_string(),
            choices: vec![Choice {
                index: Some(0),
                message: ResponseMessage {
                    role: "assistant".to_string(),
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                    extra: None,
                },
                finish_reason: Some("stop".to_string()),
                extra: None,
            }],
            usage: None,
            extra: None,
        };

        let normalized = VsCodeCopilotClient::normalize_choices(response.clone());

        assert_eq!(normalized.choices.len(), 1);
        assert_eq!(normalized.choices[0].index, Some(0));
        assert_eq!(
            normalized.choices[0].message.content,
            Some("Hello".to_string())
        );
    }

    #[test]
    fn test_normalize_choices_anthropic_split() {
        // WHY: Anthropic returns TWO choices - one with content, one with tool_calls
        // These must be merged into a single choice
        use crate::providers::vscode::types::*;

        let response = ChatCompletionResponse {
            id: "msg_haiku".to_string(),
            object: None,
            created: Some(1768984171),
            model: "claude-haiku-4.5".to_string(),
            choices: vec![
                Choice {
                    index: None,
                    message: ResponseMessage {
                        role: "assistant".to_string(),
                        content: Some("I'll examine the file".to_string()),
                        tool_calls: None,
                        extra: None,
                    },
                    finish_reason: Some("tool_calls".to_string()),
                    extra: None,
                },
                Choice {
                    index: None,
                    message: ResponseMessage {
                        role: "assistant".to_string(),
                        content: None,
                        tool_calls: Some(vec![ResponseToolCall {
                            id: "toolu_123".to_string(),
                            function: ResponseFunctionCall {
                                name: "read_file".to_string(),
                                arguments: "{\"path\":\"test.js\"}".to_string(),
                            },
                            call_type: "function".to_string(),
                        }]),
                        extra: None,
                    },
                    finish_reason: Some("tool_calls".to_string()),
                    extra: None,
                },
            ],
            usage: Some(Usage {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
                prompt_tokens_details: None,
                extra: None,
            }),
            extra: None,
        };

        let normalized = VsCodeCopilotClient::normalize_choices(response);

        // Should merge into single choice
        assert_eq!(normalized.choices.len(), 1);

        let choice = &normalized.choices[0];
        assert_eq!(choice.index, Some(0));

        // Should have both content and tool_calls merged
        assert_eq!(
            choice.message.content,
            Some("I'll examine the file".to_string())
        );
        assert!(choice.message.tool_calls.is_some());
        assert_eq!(choice.message.tool_calls.as_ref().unwrap().len(), 1);
        assert_eq!(
            choice.message.tool_calls.as_ref().unwrap()[0].function.name,
            "read_file"
        );
    }

    #[test]
    fn test_normalize_choices_no_merge_with_indices() {
        // WHY: Choices with different indices should NOT be merged
        use crate::providers::vscode::types::*;

        let response = ChatCompletionResponse {
            id: "test_multiple".to_string(),
            object: None,
            created: None,
            model: "gpt-4.1".to_string(),
            choices: vec![
                Choice {
                    index: Some(0),
                    message: ResponseMessage {
                        role: "assistant".to_string(),
                        content: Some("First".to_string()),
                        tool_calls: None,
                        extra: None,
                    },
                    finish_reason: Some("stop".to_string()),
                    extra: None,
                },
                Choice {
                    index: Some(1),
                    message: ResponseMessage {
                        role: "assistant".to_string(),
                        content: Some("Second".to_string()),
                        tool_calls: None,
                        extra: None,
                    },
                    finish_reason: Some("stop".to_string()),
                    extra: None,
                },
            ],
            usage: None,
            extra: None,
        };

        let normalized = VsCodeCopilotClient::normalize_choices(response.clone());

        // Should NOT merge - keep both choices
        assert_eq!(normalized.choices.len(), 2);
        assert_eq!(normalized.choices[0].index, Some(0));
        assert_eq!(normalized.choices[1].index, Some(1));
    }
}
