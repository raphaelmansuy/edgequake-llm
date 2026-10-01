//! SPEC-131 — Chat Completions ↔ Responses API mapper (SRP / DRY).
//!
//! Shared by OpenAIProvider and OpenAICompatibleProvider. Always sets
//! `store: false` (LAW-131-7).

use serde_json::{json, Value};

use crate::omit_env::{reasoning_effort_for_wire, temperature_for_wire};
use crate::traits::{ChatMessage, ChatRole, CompletionOptions, LLMResponse, StreamChunk};

/// Join `{base}/responses` (base already includes `/v1` or `/openai/v1`).
pub fn responses_url(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    if base.is_empty() {
        "https://api.openai.com/v1/responses".to_string()
    } else {
        format!("{base}/responses")
    }
}

/// Build a Responses API request body from product messages + options.
pub fn build_responses_request(
    model: &str,
    messages: &[ChatMessage],
    options: &CompletionOptions,
    stream: bool,
) -> Value {
    let mut instructions: Option<String> = None;
    let mut input: Vec<Value> = Vec::new();

    for msg in messages {
        match msg.role {
            ChatRole::System => {
                // Prefer `instructions` for system (OpenAI Responses idiom).
                match &mut instructions {
                    Some(existing) => {
                        existing.push('\n');
                        existing.push_str(&msg.content);
                    }
                    None => instructions = Some(msg.content.clone()),
                }
            }
            ChatRole::User => {
                input.push(user_input_item(msg));
            }
            ChatRole::Assistant => {
                input.push(json!({
                    "role": "assistant",
                    "content": msg.content,
                }));
            }
            ChatRole::Tool | ChatRole::Function => {
                // v1: flatten tool results as user text (no hosted tool loop).
                input.push(json!({
                    "role": "user",
                    "content": format!("[tool {}] {}", msg.name.as_deref().unwrap_or("result"), msg.content),
                }));
            }
        }
    }

    if let Some(sys) = options.system_prompt.as_ref() {
        match &mut instructions {
            Some(existing) => {
                existing.push('\n');
                existing.push_str(sys);
            }
            None => instructions = Some(sys.clone()),
        }
    }

    let mut body = json!({
        "model": model,
        "input": input,
        "store": false,
    });

    if let Some(instr) = instructions {
        body["instructions"] = json!(instr);
    }
    if stream {
        body["stream"] = json!(true);
    }
    if let Some(max) = options.max_tokens {
        body["max_output_tokens"] = json!(max);
    }
    if let Some(temp) = temperature_for_wire(options.temperature) {
        body["temperature"] = json!(temp);
    }
    if let Some(effort) = reasoning_effort_for_wire(options.reasoning_effort.clone()) {
        body["reasoning"] = json!({ "effort": effort });
    }
    if let Some(key) = options.prompt_cache_key.as_ref() {
        body["prompt_cache_key"] = json!(key);
    }

    match options.response_format.as_deref() {
        Some("json_schema") => {
            if let Some(schema) = options.response_schema.as_ref() {
                body["text"] = json!({
                    "format": {
                        "type": "json_schema",
                        "name": "response",
                        "schema": schema,
                        "strict": true
                    }
                });
            }
        }
        Some("json_object") | Some("json") => {
            body["text"] = json!({
                "format": { "type": "json_object" }
            });
        }
        _ => {}
    }

    body
}

fn user_input_item(msg: &ChatMessage) -> Value {
    if let Some(images) = msg.images.as_ref() {
        if !images.is_empty() {
            let mut parts: Vec<Value> = Vec::new();
            if !msg.content.is_empty() {
                parts.push(json!({ "type": "input_text", "text": msg.content }));
            }
            for img in images {
                let url = if img.data.starts_with("http://")
                    || img.data.starts_with("https://")
                    || img.data.starts_with("data:")
                {
                    img.data.clone()
                } else {
                    format!("data:{};base64,{}", img.mime_type.as_str(), img.data)
                };
                parts.push(json!({
                    "type": "input_image",
                    "image_url": url
                }));
            }
            return json!({ "role": "user", "content": parts });
        }
    }
    json!({ "role": "user", "content": msg.content })
}

/// Map a Responses API JSON body to [`LLMResponse`].
pub fn parse_responses_body(body: &Value, fallback_model: &str) -> LLMResponse {
    let content = extract_output_text(body);
    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or(fallback_model)
        .to_string();

    let mut resp = LLMResponse::new(content, &model);
    if let Some(usage) = body.get("usage") {
        let input = usage
            .get("input_tokens")
            .or_else(|| usage.get("prompt_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let output = usage
            .get("output_tokens")
            .or_else(|| usage.get("completion_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        resp = resp.with_usage(input, output);
    }
    if let Some(status) = body.get("status").and_then(|s| s.as_str()) {
        resp = resp.with_finish_reason(status);
    }
    resp
}

/// Prefer top-level `output_text`, else concatenate message `output_text` parts
/// (skipping reasoning items — E2E-131-08).
pub fn extract_output_text(body: &Value) -> String {
    if let Some(t) = body.get("output_text").and_then(|v| v.as_str()) {
        if !t.is_empty() {
            return t.to_string();
        }
    }
    let mut out = String::new();
    if let Some(items) = body.get("output").and_then(|v| v.as_array()) {
        for item in items {
            let ty = item.get("type").and_then(|t| t.as_str()).unwrap_or("");
            if ty == "reasoning" {
                continue;
            }
            if ty == "message" || item.get("role").and_then(|r| r.as_str()) == Some("assistant") {
                if let Some(parts) = item.get("content").and_then(|c| c.as_array()) {
                    for part in parts {
                        let ptype = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        if ptype == "output_text" || ptype == "text" {
                            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                out.push_str(text);
                            }
                        }
                    }
                } else if let Some(text) = item.get("content").and_then(|c| c.as_str()) {
                    out.push_str(text);
                }
            }
        }
    }
    out
}

/// Parse one SSE `data:` JSON payload into optional content delta.
pub fn parse_sse_data_line(data: &str) -> Option<StreamChunk> {
    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        return None;
    }
    let v: Value = serde_json::from_str(data).ok()?;
    let event = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
    match event {
        "response.output_text.delta" | "response.content_part.delta" => {
            let delta = v
                .get("delta")
                .and_then(|d| d.as_str())
                .or_else(|| v.pointer("/delta/text").and_then(|t| t.as_str()))
                .unwrap_or("");
            if delta.is_empty() {
                None
            } else {
                Some(StreamChunk::Content(delta.to_string()))
            }
        }
        "response.completed" => Some(StreamChunk::Finished {
            reason: "stop".into(),
            ttft_ms: None,
            usage: None,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::omit_env::ENV_OMIT_TEMPERATURE;

    #[test]
    fn store_false_always() {
        std::env::remove_var(ENV_OMIT_TEMPERATURE);
        let body = build_responses_request(
            "gpt-test",
            &[ChatMessage::user("hi")],
            &CompletionOptions::default(),
            false,
        );
        assert_eq!(body["store"], false);
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn json_schema_maps_to_text_format() {
        let opts = CompletionOptions {
            response_format: Some("json_schema".into()),
            response_schema: Some(json!({"type":"object"})),
            ..Default::default()
        };
        let body = build_responses_request("m", &[ChatMessage::user("x")], &opts, false);
        assert_eq!(body["text"]["format"]["type"], "json_schema");
        assert_eq!(body["store"], false);
    }

    #[test]
    fn skips_reasoning_items_in_output() {
        let body = json!({
            "output": [
                { "type": "reasoning", "summary": [] },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": "hello" }]
                }
            ]
        });
        assert_eq!(extract_output_text(&body), "hello");
    }

    #[test]
    fn sse_delta_maps_to_content() {
        let line = r#"{"type":"response.output_text.delta","delta":"Hi"}"#;
        match parse_sse_data_line(line) {
            Some(StreamChunk::Content(s)) => assert_eq!(s, "Hi"),
            other => panic!("unexpected {other:?}"),
        }
    }
}
