//! Google Cloud credentials for Vertex AI (SPEC: Gemini + Vertex fully working).
//!
//! # Why this module exists
//!
//! Vertex AI authenticates with an OAuth2 **access token**, never an API key. Before this
//! module, `GeminiProvider` could only read a token two ways: `GOOGLE_ACCESS_TOKEN` (a token
//! the caller had to mint) or the `gcloud` CLI. Both fail in the places EdgeQuake actually
//! runs — a container has no gcloud, and a token minted by hand expires in an hour.
//!
//! Application Default Credentials (ADC) is the mechanism Google defines for exactly this
//! case. It has two file shapes, and both are implemented here:
//!
//! | ADC type | Where it comes from | How a token is minted |
//! |----------|---------------------|-----------------------|
//! | `service_account` | `GOOGLE_APPLICATION_CREDENTIALS`, or a key file on disk | Sign a JWT, exchange it at `token_uri` |
//! | `authorized_user` | `gcloud auth application-default login` | Exchange the refresh token |
//!
//! Both are a `POST` to Google's token endpoint returning `{ access_token, expires_in }`, so
//! both are modelled as one [`TokenSource`] whose [`TokenSource::refresh`] does that POST.
//!
//! # Resolution order
//!
//! [`resolve`] mirrors the `google-auth` libraries so a machine that works with `gcloud`
//! works with this crate:
//!
//! 1. `GOOGLE_APPLICATION_CREDENTIALS` (explicit service-account key or authorized-user JSON)
//! 2. The well-known ADC file (`~/.config/gcloud/application_default_credentials.json`)
//! 3. The GCE/GKE/Cloud Run metadata server
//!
//! # References
//!
//! - <https://cloud.google.com/docs/authentication/application-default-credentials>
//! - <https://cloud.google.com/iam/docs/reference/credentials/rest/v1/projects.serviceAccounts/generateAccessToken>

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::error::{LlmError, Result};

/// Scope that grants access to Vertex AI and the rest of Google Cloud.
pub const CLOUD_PLATFORM_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

/// Google's OAuth2 token endpoint. A `token_uri` in the credentials file overrides it.
const DEFAULT_TOKEN_URI: &str = "https://oauth2.googleapis.com/token";

/// JWT assertion grant type (RFC 7523). Service accounts use it to trade a signed JWT for a token.
const JWT_BEARER_GRANT: &str = "urn:ietf:params:oauth:grant-type:jwt-bearer";

/// Refresh this long before the token actually expires, so a request never races the deadline.
const EXPIRY_MARGIN: Duration = Duration::from_secs(60);

/// The `type` field of a Google credentials JSON file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CredentialKind {
    /// A key file downloaded from a service account (`private_key`, `client_email`).
    ServiceAccount,
    /// What `gcloud auth application-default login` writes (`refresh_token`, `client_id`).
    AuthorizedUser,
}

/// A credentials file, parsed but not yet exercised.
///
/// Kept as an enum rather than a struct with optionals: each variant carries exactly the
/// fields its flow needs, so a missing field is a parse error instead of a runtime surprise.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GoogleCredentials {
    /// Key-file auth: sign a JWT with `private_key`, exchange it for a token.
    ServiceAccount {
        client_email: String,
        private_key: String,
        #[serde(default)]
        project_id: Option<String>,
        #[serde(default = "default_token_uri")]
        token_uri: String,
    },
    /// User auth: trade the stored `refresh_token` for a token.
    AuthorizedUser {
        client_id: String,
        client_secret: String,
        refresh_token: String,
        #[serde(default)]
        quota_project_id: Option<String>,
        #[serde(default = "default_token_uri")]
        token_uri: String,
    },
}

fn default_token_uri() -> String {
    DEFAULT_TOKEN_URI.to_string()
}

impl GoogleCredentials {
    /// Read and parse a credentials file. A file that parses but is neither shape is an error
    /// that names the file, because "which key is being used" is the first question when auth fails.
    pub fn from_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path).map_err(|err| {
            LlmError::ConfigError(format!(
                "Could not read Google credentials at {}: {err}",
                path.display()
            ))
        })?;
        serde_json::from_str(&raw).map_err(|err| {
            LlmError::ConfigError(format!(
                "Google credentials at {} are not a service_account or authorized_user file: {err}",
                path.display()
            ))
        })
    }

    /// The project the credentials belong to, when the file states one.
    pub fn project_id(&self) -> Option<&str> {
        match self {
            Self::ServiceAccount {
                project_id,
                token_uri,
                ..
            } => project_id.as_deref().filter(|_| !token_uri.is_empty()),
            Self::AuthorizedUser {
                quota_project_id, ..
            } => quota_project_id.as_deref(),
        }
    }

    /// The principal that will be billed for the call, for logs and error messages.
    pub fn principal(&self) -> &str {
        match self {
            Self::ServiceAccount { client_email, .. } => client_email,
            Self::AuthorizedUser {
                quota_project_id, ..
            } => quota_project_id.as_deref().unwrap_or("user credentials"),
        }
    }

    /// A token source for these credentials.
    pub fn into_source(self) -> TokenSource {
        match self {
            Self::ServiceAccount {
                client_email,
                private_key,
                token_uri,
                ..
            } => TokenSource::ServiceAccount {
                client_email,
                private_key,
                token_uri,
            },
            Self::AuthorizedUser {
                client_id,
                client_secret,
                refresh_token,
                token_uri,
                ..
            } => TokenSource::AuthorizedUser {
                client_id,
                client_secret,
                refresh_token,
                token_uri,
            },
        }
    }
}

/// Anything that can mint a Google access token.
///
/// One trait, three shapes: a service-account key, a user refresh token, and the metadata
/// server. The provider only ever calls `refresh` at a deadline, so it does not care which.
#[derive(Debug, Clone)]
pub enum TokenSource {
    ServiceAccount {
        client_email: String,
        private_key: String,
        token_uri: String,
    },
    AuthorizedUser {
        client_id: String,
        client_secret: String,
        refresh_token: String,
        token_uri: String,
    },
    /// The GCE/GKE/Cloud Run metadata server; mints a token for the attached service account.
    MetadataServer,
}

/// A minted token and the instant it should be replaced.
#[derive(Debug, Clone)]
pub struct AccessToken {
    pub token: String,
    /// When to ask for a new token. Already reduced by [`EXPIRY_MARGIN`].
    pub refresh_at: Instant,
}

impl AccessToken {
    /// Treat a token minted with `expires_in` seconds as expiring `EXPIRY_MARGIN` earlier.
    fn new(token: String, expires_in: u64) -> Self {
        let usable = expires_in.saturating_sub(EXPIRY_MARGIN.as_secs());
        Self {
            token,
            refresh_at: Instant::now() + Duration::from_secs(usable.max(1)),
        }
    }

    /// A token whose expiry is unknown (metadata server, or a caller-supplied string).
    /// Treated as short-lived so the next call refreshes it rather than trusting it forever.
    pub fn fixed(token: impl Into<String>) -> Self {
        Self::new(token.into(), 300)
    }

    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.refresh_at
    }
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    expires_in: Option<u64>,
}

impl TokenSource {
    /// Mint a fresh access token.
    pub async fn refresh(&self) -> Result<AccessToken> {
        match self {
            Self::ServiceAccount {
                client_email,
                private_key,
                token_uri,
            } => {
                let assertion = sign_service_account_jwt(client_email, private_key, token_uri)?;
                post_form(
                    token_uri,
                    &[
                        ("grant_type", JWT_BEARER_GRANT),
                        ("assertion", assertion.as_str()),
                    ],
                )
                .await
            }
            Self::AuthorizedUser {
                client_id,
                client_secret,
                refresh_token,
                token_uri,
            } => {
                post_form(
                    token_uri,
                    &[
                        ("grant_type", "refresh_token"),
                        ("client_id", client_id.as_str()),
                        ("client_secret", client_secret.as_str()),
                        ("refresh_token", refresh_token.as_str()),
                    ],
                )
                .await
            }
            Self::MetadataServer => fetch_metadata_token().await,
        }
    }
}

/// Exchange form fields for an access token.
async fn post_form(token_uri: &str, fields: &[(&str, &str)]) -> Result<AccessToken> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|err| LlmError::ConfigError(format!("Could not build OAuth client: {err}")))?;
    let form: Vec<(&str, &str)> = fields.to_vec();
    let response = client
        .post(token_uri)
        .form(&form)
        .send()
        .await
        .map_err(|err| {
            LlmError::ConfigError(format!("Token request to {token_uri} failed: {err}"))
        })?;

    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        // Google's error body says *why* (invalid_grant, invalid_rapt, …). Passing it through
        // is the difference between "auth failed" and "run gcloud auth application-default login".
        return Err(LlmError::ConfigError(format!(
            "Google token request was rejected ({status}): {}",
            token_error_hint(&body)
        )));
    }
    let parsed: TokenResponse = serde_json::from_str(&body).map_err(|err| {
        LlmError::ConfigError(format!("Could not read the token response: {err}"))
    })?;
    Ok(AccessToken::new(
        parsed.access_token,
        parsed.expires_in.unwrap_or(3600),
    ))
}

/// Turn a Google OAuth error body into one actionable sentence.
fn token_error_hint(body: &str) -> String {
    let parsed: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let description = parsed
        .get("error_description")
        .or_else(|| parsed.get("error"))
        .and_then(|value| value.as_str())
        .unwrap_or(body)
        .to_string();
    if description.contains("invalid_rapt") || description.contains("reauth") {
        return format!(
            "{description}. The user credentials need re-authentication: run \
             `gcloud auth application-default login`."
        );
    }
    if description.contains("account not found") || description.contains("invalid_grant") {
        return format!(
            "{description}. The service account may have been deleted, or its key revoked. \
             Create a new key or use a different GOOGLE_APPLICATION_CREDENTIALS file."
        );
    }
    description
}

/// Sign a service-account JWT assertion for the JWT-bearer grant.
///
/// The claims are fixed by Google's spec: `iss` is the service-account email, `aud` is the
/// token endpoint, `scope` is what we want, and `exp` is at most one hour out.
fn sign_service_account_jwt(
    client_email: &str,
    private_key_pem: &str,
    token_uri: &str,
) -> Result<String> {
    use jsonwebtoken::{Algorithm, EncodingKey, Header};

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    #[derive(serde::Serialize)]
    struct Claims<'a> {
        iss: &'a str,
        scope: &'a str,
        aud: &'a str,
        iat: u64,
        exp: u64,
    }

    let claims = Claims {
        iss: client_email,
        scope: CLOUD_PLATFORM_SCOPE,
        aud: token_uri,
        iat: now,
        // Google rejects anything beyond an hour; an hour is also the longest useful window.
        exp: now + 3600,
    };

    let key = EncodingKey::from_rsa_pem(private_key_pem.as_bytes()).map_err(|err| {
        LlmError::ConfigError(format!(
            "The service-account private_key is not a usable RSA PEM block: {err}"
        ))
    })?;

    jsonwebtoken::encode(&Header::new(Algorithm::RS256), &claims, &key).map_err(|err| {
        LlmError::ConfigError(format!(
            "Could not sign the service-account assertion: {err}"
        ))
    })
}

/// Token from the GCE/GKE/Cloud Run metadata server.
///
/// Only reachable when a Google identity is attached to the workload; every other
/// environment fails fast here and falls back to the next source in [`resolve`].
pub async fn fetch_metadata_token() -> Result<AccessToken> {
    fetch_metadata_token_from(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
    )
    .await
}

async fn fetch_metadata_token_from(url: &str) -> Result<AccessToken> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|err| LlmError::ConfigError(format!("Could not build HTTP client: {err}")))?;
    let response = client
        .get(url)
        .header("Metadata-Flavor", "Google")
        .send()
        .await
        .map_err(|err| LlmError::ConfigError(format!("Metadata server request failed: {err}")))?;
    if !response.status().is_success() {
        return Err(LlmError::ConfigError(format!(
            "Metadata server returned status {}",
            response.status()
        )));
    }
    let token: TokenResponse = response
        .json()
        .await
        .map_err(|err| LlmError::ConfigError(format!("Could not read metadata token: {err}")))?;
    Ok(AccessToken::new(
        token.access_token,
        token.expires_in.unwrap_or(300),
    ))
}

/// The well-known ADC path: `$GOOGLE_APPLICATION_CREDENTIALS`, else the gcloud file.
///
/// Returns `None` when neither exists, so the caller can move on to the metadata server.
pub fn adc_path() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
        let trimmed = explicit.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed));
        }
    }
    well_known_adc_path()
}

/// `~/.config/gcloud/application_default_credentials.json` (or `%APPDATA%\gcloud\...`).
fn well_known_adc_path() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        let appdata = std::env::var("APPDATA").ok()?;
        let path = PathBuf::from(appdata)
            .join("gcloud")
            .join("application_default_credentials.json");
        return path.is_file().then_some(path);
    }
    #[cfg(not(windows))]
    {
        let home = std::env::var("HOME").ok()?;
        let path = PathBuf::from(home)
            .join(".config")
            .join("gcloud")
            .join("application_default_credentials.json");
        path.is_file().then_some(path)
    }
}

/// Resolve credentials the way Google's own libraries do.
///
/// A credentials file wins over the metadata server: it is the explicit choice, and a
/// developer machine that happens to be on GCP should still use the key they pointed at.
pub fn resolve() -> Result<(TokenSource, Option<PathBuf>)> {
    if let Some(path) = adc_path() {
        if path.is_file() {
            let credentials = GoogleCredentials::from_file(&path)?;
            return Ok((credentials.into_source(), Some(path)));
        }
        // Explicit path set but missing is an error, not a silent fall-through: the operator
        // named a file, and using a different identity would hide their mistake.
        if std::env::var("GOOGLE_APPLICATION_CREDENTIALS").is_ok() {
            return Err(LlmError::ConfigError(format!(
                "GOOGLE_APPLICATION_CREDENTIALS points at {}, which does not exist",
                path.display()
            )));
        }
    }
    Ok((TokenSource::MetadataServer, None))
}

/// The project to bill, from `GOOGLE_CLOUD_PROJECT` or the credentials file.
pub fn resolve_project_id(credentials_path: Option<&Path>) -> Option<String> {
    if let Ok(project) = std::env::var("GOOGLE_CLOUD_PROJECT") {
        let trimmed = project.trim().to_string();
        if !trimmed.is_empty() {
            return Some(trimmed);
        }
    }
    if let Ok(project) = std::env::var("GCLOUD_PROJECT") {
        let trimmed = project.trim().to_string();
        if !trimmed.is_empty() {
            return Some(trimmed);
        }
    }
    let path = credentials_path.map(Path::to_path_buf).or_else(adc_path)?;
    let credentials = GoogleCredentials::from_file(&path).ok()?;
    credentials.project_id().map(str::to_string)
}

/// Encode helper shared with tests.
#[cfg(test)]
fn b64url(bytes: &[u8]) -> String {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    URL_SAFE_NO_PAD.encode(bytes)
}

#[cfg(all(test, not(windows)))]
mod tests {
    use super::*;

    #[test]
    fn parses_a_service_account_file() {
        let json = serde_json::json!({
            "type": "service_account",
            "project_id": "demo-project",
            "private_key": "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n",
            "client_email": "sa@demo-project.iam.gserviceaccount.com",
            "token_uri": "https://oauth2.googleapis.com/token"
        });
        let credentials: GoogleCredentials = serde_json::from_value(json).unwrap();
        assert_eq!(credentials.project_id(), Some("demo-project"));
        assert_eq!(
            credentials.principal(),
            "sa@demo-project.iam.gserviceaccount.com"
        );
        assert!(matches!(
            credentials.into_source(),
            TokenSource::ServiceAccount { .. }
        ));
    }

    #[test]
    fn parses_an_authorized_user_file() {
        let json = serde_json::json!({
            "type": "authorized_user",
            "client_id": "cid.apps.googleusercontent.com",
            "client_secret": "secret",
            "refresh_token": "refresh",
            "quota_project_id": "quota-project"
        });
        let credentials: GoogleCredentials = serde_json::from_value(json).unwrap();
        assert_eq!(credentials.project_id(), Some("quota-project"));
        assert!(matches!(
            credentials.into_source(),
            TokenSource::AuthorizedUser { .. }
        ));
    }

    #[test]
    fn token_uri_defaults_when_the_file_omits_it() {
        let json = serde_json::json!({
            "type": "authorized_user",
            "client_id": "cid",
            "client_secret": "secret",
            "refresh_token": "refresh"
        });
        let credentials: GoogleCredentials = serde_json::from_value(json).unwrap();
        match credentials.into_source() {
            TokenSource::AuthorizedUser { token_uri, .. } => {
                assert_eq!(token_uri, DEFAULT_TOKEN_URI);
            }
            other => panic!("expected authorized_user, got {other:?}"),
        }
    }

    #[test]
    fn an_unknown_type_is_a_named_error_not_a_silent_success() {
        let json = r#"{"type":"external_account","audience":"x"}"#;
        let parsed: std::result::Result<GoogleCredentials, _> = serde_json::from_str(json);
        assert!(parsed.is_err(), "external_account must not parse as ADC");
    }

    #[test]
    fn a_missing_file_names_the_path() {
        let err = GoogleCredentials::from_file(Path::new("/nonexistent/adc.json")).unwrap_err();
        assert!(err.to_string().contains("/nonexistent/adc.json"), "{err}");
    }

    #[test]
    fn invalid_rapt_error_says_to_reauthenticate() {
        let body = r#"{"error":"invalid_grant","error_description":"reauth related error (invalid_rapt)"}"#;
        let hint = token_error_hint(body);
        assert!(
            hint.contains("gcloud auth application-default login"),
            "{hint}"
        );
    }

    #[test]
    fn deleted_service_account_error_says_the_key_is_dead() {
        let body =
            r#"{"error":"invalid_grant","error_description":"Invalid grant: account not found"}"#;
        let hint = token_error_hint(body);
        assert!(
            hint.contains("deleted") || hint.contains("revoked"),
            "{hint}"
        );
    }

    #[test]
    fn expires_early_so_a_request_never_races_the_deadline() {
        let token = AccessToken::new("t".into(), 3600);
        assert!(!token.is_expired());
        // A token that is already inside the margin is refreshed straight away.
        let nearly_done = AccessToken::new("t".into(), 30);
        assert!(
            nearly_done.is_expired()
                || nearly_done.refresh_at <= Instant::now() + Duration::from_secs(1)
        );
    }

    #[test]
    fn explicit_credentials_path_wins_over_the_well_known_file() {
        let dir = std::env::temp_dir().join(format!("eq-adc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("key.json");
        std::fs::write(
            &path,
            serde_json::json!({
                "type": "service_account",
                "project_id": "from-file",
                "private_key": "x",
                "client_email": "a@b.iam.gserviceaccount.com"
            })
            .to_string(),
        )
        .unwrap();

        let previous = std::env::var("GOOGLE_APPLICATION_CREDENTIALS").ok();
        let previous_project = std::env::var("GOOGLE_CLOUD_PROJECT").ok();
        std::env::set_var("GOOGLE_APPLICATION_CREDENTIALS", &path);
        std::env::remove_var("GOOGLE_CLOUD_PROJECT");

        assert_eq!(adc_path(), Some(path.clone()));
        // The project comes from the file when the env var is unset.
        assert_eq!(
            resolve_project_id(Some(&path)).as_deref(),
            Some("from-file")
        );

        match previous {
            Some(value) => std::env::set_var("GOOGLE_APPLICATION_CREDENTIALS", value),
            None => std::env::remove_var("GOOGLE_APPLICATION_CREDENTIALS"),
        }
        match previous_project {
            Some(value) => std::env::set_var("GOOGLE_CLOUD_PROJECT", value),
            None => std::env::remove_var("GOOGLE_CLOUD_PROJECT"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn b64url_has_no_padding() {
        assert!(!b64url(b"any carnal pleasure").contains('='));
    }
}
