use std::process::Command;

use crate::imagegen::error::{ImageGenError, Result};

pub fn env_region() -> String {
    std::env::var("GOOGLE_CLOUD_REGION")
        .or_else(|_| std::env::var("GOOGLE_CLOUD_LOCATION"))
        .unwrap_or_else(|_| "us-central1".to_string())
}

pub fn access_token_from_env_or_gcloud() -> Result<String> {
    if let Ok(token) = std::env::var("GOOGLE_ACCESS_TOKEN") {
        if !token.trim().is_empty() {
            return Ok(token);
        }
    }

    if let Ok(token) = run_gcloud_token_cmd(&["auth", "print-access-token"]) {
        return Ok(token);
    }

    if let Ok(token) = run_gcloud_token_cmd(&["auth", "application-default", "print-access-token"])
    {
        return Ok(token);
    }

    Err(ImageGenError::ConfigError(
        "could not obtain a Google Cloud access token; set GOOGLE_ACCESS_TOKEN or run `gcloud auth login` / `gcloud auth application-default login`".to_string(),
    ))
}

fn run_gcloud_token_cmd(args: &[&str]) -> Result<String> {
    let output = Command::new("gcloud")
        .args(args)
        .output()
        .map_err(|err| ImageGenError::ConfigError(format!("failed to run gcloud: {err}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(ImageGenError::ConfigError(stderr));
    }

    let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if token.is_empty() {
        return Err(ImageGenError::ConfigError(
            "gcloud returned an empty access token".to_string(),
        ));
    }

    Ok(token)
}
