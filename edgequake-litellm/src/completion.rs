//! Synchronous and asynchronous completion functions exposed to Python.
//!
//! # Functions
//!
//! * `completion(provider, model, messages_json, options_json)` → `ModelResponse`
//! * `acompletion(provider, model, messages_json, options_json)` → `Awaitable[ModelResponse]`
//! * `stream_completion(provider, model, messages_json, options_json)` → `Awaitable[List[StreamChunk]]`

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use edgequake_llm::application_context::{ApplicationContext, ApplicationContextBuilder};
use edgequake_llm::error::LlmError;
use edgequake_llm::factory::{ProviderFactory, ProviderType};
use edgequake_llm::traits::{
    ChatMessage, CompletionOptions, StreamChunk, StreamUsage, ToolChoice, ToolDefinition,
};

use futures::StreamExt;

use crate::attribution::PyApplicationContext;
use crate::bridge::runtime;
use crate::types::{llm_response_to_py, stream_chunk_to_py, to_py_err, PyModelResponse};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_provider(s: &str) -> PyResult<ProviderType> {
    ProviderType::from_str(s)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown provider '{}'. Valid providers: openai, anthropic, gemini, vertexai, mistral, openrouter, xai, azure, bedrock, ollama, lmstudio, huggingface, vscode-copilot, openai-compatible, mock", s)))
}

fn parse_messages(json: &str) -> PyResult<Vec<ChatMessage>> {
    serde_json::from_str(json)
        .map_err(|e| PyValueError::new_err(format!("Invalid messages JSON: {}", e)))
}

fn parse_options(json: Option<&str>) -> PyResult<Option<CompletionOptions>> {
    match json {
        None | Some("") => Ok(None),
        Some(s) => serde_json::from_str(s)
            .map_err(|e| PyValueError::new_err(format!("Invalid options JSON: {}", e)))
            .map(Some),
    }
}

fn parse_tools(json: Option<&str>) -> PyResult<Vec<ToolDefinition>> {
    match json {
        None | Some("") => Ok(vec![]),
        Some(s) => serde_json::from_str(s)
            .map_err(|e| PyValueError::new_err(format!("Invalid tools JSON: {}", e))),
    }
}

fn parse_tool_choice(json: Option<&str>) -> PyResult<Option<ToolChoice>> {
    match json {
        None | Some("") => Ok(None),
        Some(s) => serde_json::from_str(s)
            .map_err(|e| PyValueError::new_err(format!("Invalid tool_choice JSON: {}", e)))
            .map(Some),
    }
}

fn resolve_application_context(
    application_context: Option<PyApplicationContext>,
    application_id: Option<&str>,
    application_name: Option<&str>,
    application_url: Option<&str>,
    request_id: Option<&str>,
    end_user_id: Option<&str>,
) -> PyResult<Option<ApplicationContext>> {
    if let Some(ctx) = application_context {
        return Ok(Some(ctx.into_inner()));
    }
    if application_id.is_none()
        && application_name.is_none()
        && application_url.is_none()
        && request_id.is_none()
        && end_user_id.is_none()
    {
        return Ok(None);
    }
    let mut builder = ApplicationContextBuilder::new();
    if let Some(id) = application_id {
        builder = builder.app_id(id);
    }
    if let Some(name) = application_name {
        builder = builder.app_name(name);
    }
    if let Some(url) = application_url {
        builder = builder.app_url(url);
    }
    if let Some(rid) = request_id {
        builder = builder.request_id(rid);
    }
    if let Some(uid) = end_user_id {
        builder = builder.end_user_id(uid);
    }
    builder.build().map(Some).map_err(to_py_err)
}

fn create_llm_provider(
    provider_type: ProviderType,
    model: &str,
    ctx: Option<ApplicationContext>,
) -> PyResult<std::sync::Arc<dyn edgequake_llm::traits::LLMProvider>> {
    match ctx {
        Some(ctx) => {
            ProviderFactory::create_with_context(provider_type, Some(model), ctx).map_err(to_py_err)
        }
        None => ProviderFactory::create_with_model(provider_type, Some(model))
            .map(|(llm, _)| llm)
            .map_err(to_py_err),
    }
}

// ---------------------------------------------------------------------------
// Synchronous completion — releases GIL while waiting
// ---------------------------------------------------------------------------

/// Call an LLM provider synchronously. Releases the GIL while doing I/O.
///
/// Args:
///     provider:      Provider name (e.g. "openai", "anthropic", "ollama").
///     model:         Model name (e.g. "gpt-4o", "claude-3-5-sonnet-20241022").
///     messages_json: JSON array of {role, content} objects.
///     options_json:  Optional JSON object with completion options.
///     tools_json:    Optional JSON array of tool definitions.
///     tool_choice_json: Optional JSON string for tool choice.
///
/// Returns:
///     ModelResponse on success.
///
/// Raises:
///     PermissionError:     Authentication failure.
///     RuntimeError:        Rate limit or other runtime error.
///     ValueError:          Invalid request parameters.
///     TimeoutError:        Request timed out.
///     ConnectionError:     Network error.
///     NotImplementedError: Feature not supported by this provider.
#[pyfunction]
#[pyo3(signature = (provider, model, messages_json, options_json=None, tools_json=None, tool_choice_json=None, application_context=None, application_id=None, application_name=None, application_url=None, request_id=None, end_user_id=None))]
#[allow(clippy::too_many_arguments)]
pub fn completion(
    py: Python<'_>,
    provider: &str,
    model: &str,
    messages_json: &str,
    options_json: Option<&str>,
    tools_json: Option<&str>,
    tool_choice_json: Option<&str>,
    application_context: Option<PyApplicationContext>,
    application_id: Option<&str>,
    application_name: Option<&str>,
    application_url: Option<&str>,
    request_id: Option<&str>,
    end_user_id: Option<&str>,
) -> PyResult<PyModelResponse> {
    let provider_type = parse_provider(provider)?;
    let messages = parse_messages(messages_json)?;
    let options = parse_options(options_json)?;
    let tools = parse_tools(tools_json)?;
    let tool_choice = parse_tool_choice(tool_choice_json)?;
    let model_owned = model.to_string();
    let ctx = resolve_application_context(
        application_context,
        application_id,
        application_name,
        application_url,
        request_id,
        end_user_id,
    )?;

    let _ = py; // GIL held during blocking I/O (correct for single-threaded callers)
    runtime().block_on(async move {
        let llm = create_llm_provider(provider_type, &model_owned, ctx)?;

        let resp = if tools.is_empty() {
            llm.chat(&messages, options.as_ref())
                .await
                .map_err(to_py_err)?
        } else {
            llm.chat_with_tools(&messages, &tools, tool_choice, options.as_ref())
                .await
                .map_err(to_py_err)?
        };

        Ok(llm_response_to_py(resp))
    })
}

// ---------------------------------------------------------------------------
// Asynchronous completion — returns Python coroutine
// ---------------------------------------------------------------------------

/// Call an LLM provider asynchronously.
///
/// Returns a Python awaitable (coroutine). Must be awaited inside an async context.
///
/// Args: same as `completion`.
///
/// Returns:
///     Awaitable[ModelResponse]
#[pyfunction]
#[pyo3(signature = (provider, model, messages_json, options_json=None, tools_json=None, tool_choice_json=None, application_context=None, application_id=None, application_name=None, application_url=None, request_id=None, end_user_id=None))]
#[allow(clippy::too_many_arguments)]
pub fn acompletion<'py>(
    py: Python<'py>,
    provider: &str,
    model: &str,
    messages_json: &str,
    options_json: Option<&str>,
    tools_json: Option<&str>,
    tool_choice_json: Option<&str>,
    application_context: Option<PyApplicationContext>,
    application_id: Option<&str>,
    application_name: Option<&str>,
    application_url: Option<&str>,
    request_id: Option<&str>,
    end_user_id: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let provider_type = parse_provider(provider)?;
    let messages = parse_messages(messages_json)?;
    let options = parse_options(options_json)?;
    let tools = parse_tools(tools_json)?;
    let tool_choice = parse_tool_choice(tool_choice_json)?;
    let model_owned = model.to_string();
    let ctx = resolve_application_context(
        application_context,
        application_id,
        application_name,
        application_url,
        request_id,
        end_user_id,
    )?;

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let llm = create_llm_provider(provider_type, &model_owned, ctx)?;

        let resp = if tools.is_empty() {
            llm.chat(&messages, options.as_ref())
                .await
                .map_err(to_py_err)?
        } else {
            llm.chat_with_tools(&messages, &tools, tool_choice, options.as_ref())
                .await
                .map_err(to_py_err)?
        };

        Ok(llm_response_to_py(resp))
    })
}

// ---------------------------------------------------------------------------
// Streaming completion
// ---------------------------------------------------------------------------

/// Stream a completion, returning a list of all chunks after the stream finishes.
///
/// The Python layer wraps this into an async generator.
///
/// Returns:
///     Awaitable[List[StreamChunk]]
#[pyfunction]
#[pyo3(signature = (provider, model, messages_json, options_json=None, tools_json=None, tool_choice_json=None, application_context=None, application_id=None, application_name=None, application_url=None, request_id=None, end_user_id=None))]
#[allow(clippy::too_many_arguments)]
pub fn stream_completion<'py>(
    py: Python<'py>,
    provider: &str,
    model: &str,
    messages_json: &str,
    options_json: Option<&str>,
    tools_json: Option<&str>,
    tool_choice_json: Option<&str>,
    application_context: Option<PyApplicationContext>,
    application_id: Option<&str>,
    application_name: Option<&str>,
    application_url: Option<&str>,
    request_id: Option<&str>,
    end_user_id: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let provider_type = parse_provider(provider)?;
    let messages = parse_messages(messages_json)?;
    let options = parse_options(options_json)?;
    let tools = parse_tools(tools_json)?;
    let tool_choice = parse_tool_choice(tool_choice_json)?;
    let model_owned = model.to_string();
    let ctx = resolve_application_context(
        application_context,
        application_id,
        application_name,
        application_url,
        request_id,
        end_user_id,
    )?;

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let llm = create_llm_provider(provider_type, &model_owned, ctx)?;

        // Clone tool_choice before passing ownership into the stream call so we can
        // reuse it in the fallback branch.
        let tool_choice_fb = tool_choice.clone();

        // Collect raw chunks, with a graceful fallback for providers that don't support
        // streaming tool calls: call chat() and synthesise Content + Finished chunks.
        let raw_chunks: Vec<std::result::Result<StreamChunk, LlmError>> = match llm
            .chat_with_tools_stream(&messages, &tools, tool_choice, options.as_ref())
            .await
        {
            Ok(mut stream) => {
                let mut chunks = Vec::new();
                while let Some(chunk) = stream.next().await {
                    chunks.push(chunk);
                }
                chunks
            }
            Err(LlmError::NotSupported(_)) => {
                // Provider doesn't stream — fall back to regular (possibly tool) chat.
                let resp = if tools.is_empty() {
                    llm.chat(&messages, options.as_ref())
                        .await
                        .map_err(to_py_err)?
                } else {
                    llm.chat_with_tools(&messages, &tools, tool_choice_fb, options.as_ref())
                        .await
                        .map_err(to_py_err)?
                };
                let reason = resp
                    .finish_reason
                    .clone()
                    .unwrap_or_else(|| "stop".to_string());
                let mut usage = StreamUsage::new(resp.prompt_tokens, resp.completion_tokens);
                if let Some(cache_hit_tokens) = resp.cache_hit_tokens {
                    usage = usage.with_cache_hit_tokens(cache_hit_tokens);
                }
                if let Some(thinking_tokens) = resp.thinking_tokens {
                    usage = usage.with_thinking_tokens(thinking_tokens);
                }
                vec![
                    Ok(StreamChunk::Content(resp.content)),
                    Ok(StreamChunk::Finished {
                        reason,
                        ttft_ms: None,
                        usage: Some(usage),
                    }),
                ]
            }
            Err(e) => return Err(to_py_err(e)),
        };

        // Convert to PyStreamChunk inside a brief GIL re-acquisition.
        Python::try_attach(|py| -> PyResult<Py<PyAny>> {
            let py_chunks: Vec<Py<PyAny>> = raw_chunks
                .into_iter()
                .map(|c| {
                    let chunk = c.map_err(to_py_err)?;
                    let py_chunk = stream_chunk_to_py(chunk);
                    Ok(Py::new(py, py_chunk)?.into_any())
                })
                .collect::<PyResult<Vec<_>>>()?;
            let list = PyList::new(py, py_chunks)?;
            Ok(list.into_any().unbind())
        })
        .ok_or_else(|| PyValueError::new_err("Python interpreter not available"))?
    })
}

// ---------------------------------------------------------------------------
// Provider info helpers
// ---------------------------------------------------------------------------

/// Return the list of provider names that edgequake-python supports.
#[pyfunction]
pub fn list_providers() -> Vec<String> {
    edgequake_llm::ProviderFactory::list_providers()
        .into_iter()
        .map(String::from)
        .collect()
}

/// Detect the best available provider from environment variables.
///
/// Returns the provider name string or None if no credentials found.
#[pyfunction]
pub fn detect_provider() -> PyResult<Option<String>> {
    match ProviderFactory::from_env() {
        Ok((llm, _)) => Ok(Some(llm.name().to_string())),
        Err(_) => Ok(None),
    }
}
