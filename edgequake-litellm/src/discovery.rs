//! PyO3 bindings for the model discovery system.
//!
//! Exposes `ModelDiscoveryService` functionality to Python:
//! - `discover_all()` / `adiscover_all()` — find all models across providers
//! - `find_models()` / `afind_models()` — filter by capabilities
//! - `get_model()` / `aget_model()` — look up a specific model
//! - `list_discovery_providers()` — list registered discovery provider IDs

use edgequake_llm::discovery::{
    CapabilityFilter, DiscoveredModel, DiscoverySource, ModelDiscoveryService,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::bridge;

// ===========================================================================
// Python type: PyDiscoveredModel
// ===========================================================================

/// A discovered model with normalized capabilities — Python representation.
#[pyclass(name = "DiscoveredModel", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyDiscoveredModel {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub provider: String,
    #[pyo3(get)]
    pub context_length: usize,
    #[pyo3(get)]
    pub max_output_tokens: usize,
    #[pyo3(get)]
    pub supports_vision: bool,
    #[pyo3(get)]
    pub supports_tools: bool,
    #[pyo3(get)]
    pub supports_thinking: bool,
    #[pyo3(get)]
    pub supports_streaming: bool,
    #[pyo3(get)]
    pub supports_json_mode: bool,
    #[pyo3(get)]
    pub source: String,
    #[pyo3(get)]
    pub available: bool,
    #[pyo3(get)]
    pub cost_per_m_input: Option<f64>,
    #[pyo3(get)]
    pub cost_per_m_output: Option<f64>,
    #[pyo3(get)]
    pub model_type: String,
    #[pyo3(get)]
    pub tags: Vec<String>,
    #[pyo3(get)]
    pub deprecated: bool,
}

#[pymethods]
impl PyDiscoveredModel {
    fn __repr__(&self) -> String {
        format!(
            "DiscoveredModel(id='{}', provider='{}', ctx={}, vision={}, tools={}, thinking={})",
            self.id,
            self.provider,
            self.context_length,
            self.supports_vision,
            self.supports_tools,
            self.supports_thinking
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("id", &self.id)?;
        dict.set_item("name", &self.name)?;
        dict.set_item("provider", &self.provider)?;
        dict.set_item("context_length", self.context_length)?;
        dict.set_item("max_output_tokens", self.max_output_tokens)?;
        dict.set_item("supports_vision", self.supports_vision)?;
        dict.set_item("supports_tools", self.supports_tools)?;
        dict.set_item("supports_thinking", self.supports_thinking)?;
        dict.set_item("supports_streaming", self.supports_streaming)?;
        dict.set_item("supports_json_mode", self.supports_json_mode)?;
        dict.set_item("source", &self.source)?;
        dict.set_item("available", self.available)?;
        dict.set_item("cost_per_m_input", self.cost_per_m_input)?;
        dict.set_item("cost_per_m_output", self.cost_per_m_output)?;
        dict.set_item("model_type", &self.model_type)?;
        dict.set_item("tags", PyList::new(py, &self.tags)?)?;
        dict.set_item("deprecated", self.deprecated)?;
        Ok(dict)
    }
}

fn source_to_str(source: &DiscoverySource) -> &'static str {
    match source {
        DiscoverySource::DynamicApi => "dynamic_api",
        DiscoverySource::StaticRegistry => "static_registry",
        DiscoverySource::Hybrid => "hybrid",
        DiscoverySource::UserConfig => "user_config",
        DiscoverySource::Unknown => "unknown",
    }
}

fn model_type_to_str(mt: &edgequake_llm::model_config::ModelType) -> &'static str {
    use edgequake_llm::model_config::ModelType;
    match mt {
        ModelType::Llm => "llm",
        ModelType::Embedding => "embedding",
        ModelType::Multimodal => "multimodal",
    }
}

fn to_py_model(m: DiscoveredModel) -> PyDiscoveredModel {
    PyDiscoveredModel {
        id: m.id,
        name: m.name,
        provider: m.provider,
        context_length: m.context_length,
        max_output_tokens: m.max_output_tokens,
        supports_vision: m.capabilities.supports_vision,
        supports_tools: m.capabilities.supports_function_calling,
        supports_thinking: m.capabilities.supports_thinking,
        supports_streaming: m.capabilities.supports_streaming,
        supports_json_mode: m.capabilities.supports_json_mode,
        source: source_to_str(&m.source).to_string(),
        available: m.available,
        cost_per_m_input: m.cost_per_m_input,
        cost_per_m_output: m.cost_per_m_output,
        model_type: model_type_to_str(&m.model_type).to_string(),
        tags: m.tags,
        deprecated: m.deprecated,
    }
}

fn build_filter(filter_json: Option<&str>) -> CapabilityFilter {
    let Some(json_str) = filter_json else {
        return CapabilityFilter::default();
    };
    let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) else {
        return CapabilityFilter::default();
    };

    CapabilityFilter {
        min_context_length: val["min_context_length"].as_u64().map(|v| v as usize),
        min_output_tokens: val["min_output_tokens"].as_u64().map(|v| v as usize),
        requires_vision: val["requires_vision"].as_bool(),
        requires_tools: val["requires_tools"].as_bool(),
        requires_thinking: val["requires_thinking"].as_bool(),
        requires_streaming: val["requires_streaming"].as_bool(),
        requires_json_mode: val["requires_json_mode"].as_bool(),
        model_type: None,
        provider: val["provider"].as_str().map(String::from),
        tags: val["tags"].as_array().map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        }),
        max_cost_per_m_input: val["max_cost_per_m_input"].as_f64(),
        exclude_deprecated: val["exclude_deprecated"].as_bool(),
    }
}

// ===========================================================================
// Sync functions (use bridge runtime)
// ===========================================================================

/// Discover all models across all registered providers (blocking).
#[pyfunction]
pub fn discover_all() -> PyResult<Vec<PyDiscoveredModel>> {
    let rt = bridge::runtime();
    let service = ModelDiscoveryService::new();
    let models = rt
        .block_on(service.discover_all())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(models.into_iter().map(to_py_model).collect())
}

/// Find models matching a capability filter (blocking).
///
/// `filter_json` is a JSON string with optional fields:
/// `min_context_length`, `requires_vision`, `requires_tools`,
/// `requires_thinking`, `provider`, `max_cost_per_m_input`, etc.
#[pyfunction]
#[pyo3(signature = (filter_json = None))]
pub fn find_models(filter_json: Option<&str>) -> PyResult<Vec<PyDiscoveredModel>> {
    let rt = bridge::runtime();
    let service = ModelDiscoveryService::new();
    let filter = build_filter(filter_json);
    let models = rt
        .block_on(service.find_models(&filter))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(models.into_iter().map(to_py_model).collect())
}

/// Get a specific model by provider and model ID (blocking).
#[pyfunction]
pub fn get_model(provider: &str, model_id: &str) -> PyResult<Option<PyDiscoveredModel>> {
    let rt = bridge::runtime();
    let service = ModelDiscoveryService::new();
    let model = rt
        .block_on(service.get_model(provider, model_id))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(model.map(to_py_model))
}

/// Discover models from a specific provider only (blocking).
#[pyfunction]
pub fn discover_provider(provider_id: &str) -> PyResult<Vec<PyDiscoveredModel>> {
    let rt = bridge::runtime();
    let service = ModelDiscoveryService::new();
    let models = rt
        .block_on(service.discover_for_provider(provider_id))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(models.into_iter().map(to_py_model).collect())
}

/// List all registered discovery provider IDs.
#[pyfunction]
pub fn list_discovery_providers() -> Vec<String> {
    let service = ModelDiscoveryService::new();
    service
        .provider_ids()
        .into_iter()
        .map(String::from)
        .collect()
}

// ===========================================================================
// Async functions (use pyo3-async-runtimes)
// ===========================================================================

/// Discover all models across all registered providers (async).
#[pyfunction]
pub fn adiscover_all(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let service = ModelDiscoveryService::new();
        let models = service
            .discover_all()
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(models.into_iter().map(to_py_model).collect::<Vec<_>>())
    })
}

/// Find models matching a capability filter (async).
#[pyfunction]
#[pyo3(signature = (filter_json = None))]
pub fn afind_models(py: Python<'_>, filter_json: Option<String>) -> PyResult<Bound<'_, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let service = ModelDiscoveryService::new();
        let filter = build_filter(filter_json.as_deref());
        let models = service
            .find_models(&filter)
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(models.into_iter().map(to_py_model).collect::<Vec<_>>())
    })
}

/// Get a specific model by provider and model ID (async).
#[pyfunction]
pub fn aget_model(
    py: Python<'_>,
    provider: String,
    model_id: String,
) -> PyResult<Bound<'_, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let service = ModelDiscoveryService::new();
        let model = service
            .get_model(&provider, &model_id)
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(model.map(to_py_model))
    })
}

/// Get information about the discovery strategies per provider.
#[pyfunction]
pub fn discovery_info(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    let service = ModelDiscoveryService::new();
    let mut infos = Vec::new();

    for pid in service.provider_ids() {
        let dict = PyDict::new(py);
        dict.set_item("provider_id", pid)?;

        // Determine strategy from the providers registered
        let strategy = match pid {
            "ollama" | "lmstudio" | "anthropic" | "gemini" | "openrouter" | "mistral" => "dynamic",
            "openai" | "nvidia" | "bedrock" => "hybrid",
            "xai" => "static",
            _ => "unknown",
        };
        dict.set_item("strategy", strategy)?;
        infos.push(dict.into_any().unbind());
    }
    PyList::new(py, infos)
}
