//! Python-exposed types: PyLLMResponse, PyStreamChunk, PyUsage, etc.

use edgequake_llm::error::LlmError;
use edgequake_llm::traits::{LLMResponse, StreamChunk, ToolCall};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// ===========================================================================
// Error conversion
// ===========================================================================

/// Map an LlmError to an appropriate Python exception.
pub fn to_py_err(err: LlmError) -> PyErr {
    use pyo3::exceptions::*;
    match err {
        LlmError::AuthError(msg) => {
            PyPermissionError::new_err(format!("Authentication error: {}", msg))
        }
        LlmError::RateLimited(msg) => {
            PyRuntimeError::new_err(format!("Rate limit exceeded: {}", msg))
        }
        LlmError::TokenLimitExceeded { max, got } => {
            PyValueError::new_err(format!("Token limit exceeded: max={}, got={}", max, got))
        }
        LlmError::ModelNotFound(msg) => PyValueError::new_err(format!("Model not found: {}", msg)),
        LlmError::Timeout => PyTimeoutError::new_err("Request timed out"),
        LlmError::NetworkError(msg) => {
            PyConnectionError::new_err(format!("Network error: {}", msg))
        }
        LlmError::NotSupported(msg) => {
            PyNotImplementedError::new_err(format!("Not supported: {}", msg))
        }
        LlmError::ConfigError(msg) => {
            PyValueError::new_err(format!("Configuration error: {}", msg))
        }
        LlmError::InvalidRequest(msg) => PyValueError::new_err(format!("Invalid request: {}", msg)),
        other => PyRuntimeError::new_err(format!("LLM error: {}", other)),
    }
}

// ===========================================================================
// Usage stats
// ===========================================================================

/// Token usage statistics, matches litellm's Usage schema.
#[pyclass(name = "Usage", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyUsage {
    #[pyo3(get)]
    pub prompt_tokens: usize,
    #[pyo3(get)]
    pub completion_tokens: usize,
    #[pyo3(get)]
    pub total_tokens: usize,
    /// Tokens served from provider KV-cache (None if not reported).
    #[pyo3(get)]
    pub cache_read_input_tokens: Option<usize>,
    /// Reasoning/thinking tokens (o1, Claude, etc.).
    #[pyo3(get)]
    pub reasoning_tokens: Option<usize>,
}

#[pymethods]
impl PyUsage {
    fn __repr__(&self) -> String {
        format!(
            "Usage(prompt={}, completion={}, total={})",
            self.prompt_tokens, self.completion_tokens, self.total_tokens
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("prompt_tokens", self.prompt_tokens)?;
        dict.set_item("completion_tokens", self.completion_tokens)?;
        dict.set_item("total_tokens", self.total_tokens)?;
        match self.cache_read_input_tokens {
            Some(v) => dict.set_item("cache_read_input_tokens", v)?,
            None => dict.set_item("cache_read_input_tokens", py.None())?,
        }
        match self.reasoning_tokens {
            Some(v) => dict.set_item("reasoning_tokens", v)?,
            None => dict.set_item("reasoning_tokens", py.None())?,
        }
        Ok(dict)
    }
}

// ===========================================================================
// Tool call response
// ===========================================================================

/// A single tool/function call requested by the model.
#[pyclass(name = "ToolCall", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyToolCall {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub function_name: String,
    #[pyo3(get)]
    pub function_arguments: String,
}

#[pymethods]
impl PyToolCall {
    fn __repr__(&self) -> String {
        format!(
            "ToolCall(id='{}', function='{}')",
            self.id, self.function_name
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("id", &self.id)?;
        dict.set_item("type", "function")?;
        let func = PyDict::new(py);
        func.set_item("name", &self.function_name)?;
        func.set_item("arguments", &self.function_arguments)?;
        dict.set_item("function", func)?;
        Ok(dict)
    }
}

fn convert_tool_calls(calls: &[ToolCall]) -> Vec<PyToolCall> {
    calls
        .iter()
        .map(|tc| PyToolCall {
            id: tc.id.clone(),
            function_name: tc.function.name.clone(),
            function_arguments: tc.function.arguments.clone(),
        })
        .collect()
}

// ===========================================================================
// ModelResponse — LiteLLM compatible response object
// ===========================================================================

/// Response from a completion call, mirroring litellm's ModelResponse.
#[pyclass(name = "ModelResponse", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyModelResponse {
    /// Primary text content.
    #[pyo3(get)]
    pub content: String,
    /// Model that produced this response.
    #[pyo3(get)]
    pub model: String,
    /// Finish reason (stop, length, tool_calls, content_filter, …).
    #[pyo3(get)]
    pub finish_reason: Option<String>,
    /// Token usage statistics.
    #[pyo3(get)]
    pub usage: PyUsage,
    /// Tool calls requested by the model.
    #[pyo3(get)]
    pub tool_calls: Vec<PyToolCall>,
    /// Extended thinking / reasoning text (Claude, DeepSeek, Gemini Flash Thinking).
    #[pyo3(get)]
    pub thinking_content: Option<String>,
}

#[pymethods]
impl PyModelResponse {
    fn __repr__(&self) -> String {
        let preview = if self.content.len() > 80 {
            format!("{}…", &self.content[..80])
        } else {
            self.content.clone()
        };
        format!(
            "ModelResponse(model='{}', content='{}', total_tokens={})",
            self.model, preview, self.usage.total_tokens
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("content", &self.content)?;
        dict.set_item("model", &self.model)?;
        match &self.finish_reason {
            Some(s) => dict.set_item("finish_reason", s.as_str())?,
            None => dict.set_item("finish_reason", py.None())?,
        }
        dict.set_item("usage", self.usage.to_dict(py)?)?;
        // Convert tool_calls to list of dicts
        let calls: Vec<Py<PyAny>> = self
            .tool_calls
            .iter()
            .map(|tc| tc.to_dict(py).map(|d| d.into_any().unbind()))
            .collect::<PyResult<_>>()?;
        dict.set_item("tool_calls", PyList::new(py, calls)?)?;
        match &self.thinking_content {
            Some(s) => dict.set_item("thinking_content", s.as_str())?,
            None => dict.set_item("thinking_content", py.None())?,
        }
        Ok(dict)
    }

    /// True when the model requested at least one tool call.
    fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

/// Convert a Rust `LLMResponse` to a Python `PyModelResponse`.
pub fn llm_response_to_py(resp: LLMResponse) -> PyModelResponse {
    let usage = PyUsage {
        prompt_tokens: resp.prompt_tokens,
        completion_tokens: resp.completion_tokens,
        total_tokens: resp.total_tokens,
        cache_read_input_tokens: resp.cache_hit_tokens,
        reasoning_tokens: resp.thinking_tokens,
    };
    PyModelResponse {
        content: resp.content,
        model: resp.model,
        finish_reason: resp.finish_reason,
        usage,
        tool_calls: convert_tool_calls(&resp.tool_calls),
        thinking_content: resp.thinking_content,
    }
}

// ===========================================================================
// StreamChunk — single chunk from a streaming response
// ===========================================================================

/// Tool call delta inside a streaming chunk.
#[pyclass(name = "ToolCallDelta", skip_from_py_object)]
#[derive(Clone, Debug, Default)]
pub struct PyToolCallDelta {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub id: Option<String>,
    #[pyo3(get)]
    pub function_name: Option<String>,
    #[pyo3(get)]
    pub function_arguments: Option<String>,
}

/// A chunk of a streaming completion response.
#[pyclass(name = "StreamChunk", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyStreamChunk {
    /// Incremental content text (may be empty for tool/thinking chunks).
    #[pyo3(get)]
    pub content: Option<String>,
    /// Incremental thinking/reasoning text.
    #[pyo3(get)]
    pub thinking: Option<String>,
    /// Whether the stream is finished.
    #[pyo3(get)]
    pub is_finished: bool,
    /// Finish reason (present when is_finished=True).
    #[pyo3(get)]
    pub finish_reason: Option<String>,
    /// Partial tool call delta (None for content/thinking chunks).
    #[pyo3(get)]
    pub tool_call_delta: Option<PyToolCallDelta>,
}

#[pymethods]
impl PyStreamChunk {
    fn __repr__(&self) -> String {
        if self.is_finished {
            format!("StreamChunk(finished, reason={:?})", self.finish_reason)
        } else {
            format!(
                "StreamChunk(content={:?}, thinking={:?})",
                self.content, self.thinking
            )
        }
    }
}

/// Convert a Rust `StreamChunk` to a `PyStreamChunk`.
/// Does NOT need the Python GIL — all fields are plain Rust types.
pub fn stream_chunk_to_py(chunk: StreamChunk) -> PyStreamChunk {
    match chunk {
        StreamChunk::Content(text) => PyStreamChunk {
            content: Some(text),
            thinking: None,
            is_finished: false,
            finish_reason: None,
            tool_call_delta: None,
        },
        StreamChunk::ThinkingContent { text, .. } => PyStreamChunk {
            content: None,
            thinking: Some(text),
            is_finished: false,
            finish_reason: None,
            tool_call_delta: None,
        },
        StreamChunk::Finished { reason, .. } => PyStreamChunk {
            content: None,
            thinking: None,
            is_finished: true,
            finish_reason: Some(reason),
            tool_call_delta: None,
        },
        StreamChunk::ToolCallDelta {
            index,
            id,
            function_name,
            function_arguments,
        } => PyStreamChunk {
            content: None,
            thinking: None,
            is_finished: false,
            finish_reason: None,
            tool_call_delta: Some(PyToolCallDelta {
                index,
                id,
                function_name,
                function_arguments,
            }),
        },
    }
}
