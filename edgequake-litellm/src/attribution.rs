//! Application attribution types and catalog helpers for Python.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use edgequake_llm::application_context::{ApplicationContext, ApplicationContextBuilder};
use edgequake_llm::provider_catalog::{AttributionSupport, ProviderCatalog};

use crate::types::to_py_err;

fn attribution_support_label(support: AttributionSupport) -> &'static str {
    match support {
        AttributionSupport::Full => "full",
        AttributionSupport::Passthrough => "passthrough",
        AttributionSupport::ObservabilityOnly => "observability_only",
        AttributionSupport::None => "none",
    }
}

/// Application identity for provider attribution propagation.
#[pyclass(name = "ApplicationContext", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyApplicationContext {
    inner: ApplicationContext,
}

#[pymethods]
impl PyApplicationContext {
    #[new]
    #[pyo3(signature = (application_id=None, application_name=None, application_url=None, tenant_id=None, request_id=None, end_user_id=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        application_id: Option<String>,
        application_name: Option<String>,
        application_url: Option<String>,
        tenant_id: Option<String>,
        request_id: Option<String>,
        end_user_id: Option<String>,
    ) -> PyResult<Self> {
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
        if let Some(tid) = tenant_id {
            builder = builder.tenant_id(tid);
        }
        if let Some(rid) = request_id {
            builder = builder.request_id(rid);
        }
        if let Some(uid) = end_user_id {
            builder = builder.end_user_id(uid);
        }
        Ok(Self {
            inner: builder.build().map_err(to_py_err)?,
        })
    }

    /// Load defaults from ``EDGEQUAKE_APP_*`` environment variables.
    #[staticmethod]
    fn from_env() -> Self {
        Self {
            inner: ApplicationContext::from_env(),
        }
    }

    /// Parse canonical ingress headers (``X-EdgeQuake-App-Id``, etc.).
    #[staticmethod]
    fn from_headers(headers: Bound<'_, PyDict>) -> PyResult<Self> {
        let mut map = std::collections::HashMap::new();
        for (key, value) in headers.iter() {
            let k = key.extract::<String>()?;
            let v = value.extract::<String>()?;
            map.insert(k, v);
        }
        Ok(Self {
            inner: ApplicationContext::from_ingress_headers(&map).map_err(to_py_err)?,
        })
    }

    #[getter]
    fn application_id(&self) -> Option<String> {
        self.inner.app_id.clone()
    }

    #[getter]
    fn application_name(&self) -> Option<String> {
        self.inner.app_name.clone()
    }

    #[getter]
    fn application_url(&self) -> Option<String> {
        self.inner.app_url.clone()
    }

    #[getter]
    fn tenant_id(&self) -> Option<String> {
        self.inner.tenant_id.clone()
    }

    #[getter]
    fn request_id(&self) -> Option<String> {
        self.inner.request_id.clone()
    }

    #[getter]
    fn end_user_id(&self) -> Option<String> {
        self.inner.end_user_id.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ApplicationContext(application_id={:?}, application_name={:?})",
            self.inner.app_id, self.inner.app_name
        )
    }
}

impl PyApplicationContext {
    pub fn into_inner(self) -> ApplicationContext {
        self.inner
    }
}

/// Return attribution support level for one provider ID or alias.
#[pyfunction]
#[pyo3(signature = (provider))]
pub fn get_provider_attribution(provider: &str) -> PyResult<String> {
    let id = ProviderCatalog::resolve_id(provider)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown provider: {provider}")))?;
    let descriptor = ProviderCatalog::get(id)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown provider: {provider}")))?;
    Ok(attribution_support_label(descriptor.attribution_support()).to_string())
}

/// Return a mapping of canonical provider ID → attribution support level.
#[pyfunction]
pub fn list_provider_attribution<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for descriptor in ProviderCatalog::all() {
        if descriptor.features.chat {
            dict.set_item(
                descriptor.id,
                attribution_support_label(descriptor.attribution_support()),
            )?;
        }
    }
    Ok(dict)
}
