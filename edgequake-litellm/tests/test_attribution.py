"""
test_attribution.py — Application attribution (Phase 4 / spec 003).

Unit tests for ApplicationContext, catalog helpers, and completion kwargs.
Uses provider="mock" — no API keys required.
"""
from __future__ import annotations

import pytest

from edgequake_litellm import (
    ApplicationContext,
    completion,
    get_provider_attribution,
    list_provider_attribution,
)
from edgequake_litellm._types import ModelResponse
from edgequake_litellm._compat import ModelResponseCompat

_ModelResponseTypes = (ModelResponse, ModelResponseCompat)
MESSAGES = [{"role": "user", "content": "Hello"}]


class TestApplicationContext:
    def test_construct_with_fields(self):
        ctx = ApplicationContext(
            application_id="my-backend",
            application_name="My Service",
            application_url="https://app.example.com",
            tenant_id="tenant-1",
            request_id="req-abc",
            end_user_id="user-42",
        )
        assert ctx.application_id == "my-backend"
        assert ctx.application_name == "My Service"
        assert ctx.application_url == "https://app.example.com"
        assert ctx.tenant_id == "tenant-1"
        assert ctx.request_id == "req-abc"
        assert ctx.end_user_id == "user-42"

    def test_from_headers_parses_ingress(self):
        ctx = ApplicationContext.from_headers(
            {
                "x-edgequake-app-id": "workspace-api",
                "x-edgequake-request-id": "req-1",
                "traceparent": "00-abc-def-01",
            }
        )
        assert ctx.application_id == "workspace-api"
        assert ctx.request_id == "req-1"

    def test_rejects_api_key_like_app_id(self):
        with pytest.raises(Exception):
            ApplicationContext(application_id="sk-secret123")

    def test_repr_contains_id(self):
        ctx = ApplicationContext(application_id="demo")
        assert "demo" in repr(ctx)


class TestProviderAttributionCatalog:
    def test_get_openai_full(self):
        assert get_provider_attribution("openai") == "full"

    def test_get_ollama_passthrough(self):
        assert get_provider_attribution("ollama") == "passthrough"

    def test_get_vscode_copilot_observability_only(self):
        assert get_provider_attribution("vscode-copilot") == "observability_only"

    def test_resolve_alias(self):
        assert get_provider_attribution("claude") == "full"

    def test_unknown_provider_raises(self):
        with pytest.raises(Exception):
            get_provider_attribution("not-a-real-provider-xyz")

    def test_list_includes_major_providers(self):
        levels = list_provider_attribution()
        assert isinstance(levels, dict)
        assert levels.get("openai") == "full"
        assert levels.get("ollama") == "passthrough"
        assert "anthropic" in levels


class TestCompletionAttributionKwargs:
    def test_completion_with_application_id(self):
        resp = completion(
            "mock/test-model",
            MESSAGES,
            application_id="my-backend",
            request_id="req-py-1",
        )
        assert isinstance(resp, _ModelResponseTypes)

    def test_completion_with_application_context(self):
        ctx = ApplicationContext(
            application_id="ctx-app",
            application_name="Ctx App",
        )
        resp = completion(
            "mock/test-model",
            MESSAGES,
            application_context=ctx,
        )
        assert isinstance(resp, _ModelResponseTypes)

    def test_completion_context_overrides_fields(self):
        ctx = ApplicationContext(application_id="from-ctx")
        resp = completion(
            "mock/test-model",
            MESSAGES,
            application_context=ctx,
            application_id="ignored-when-ctx-set",
        )
        assert isinstance(resp, _ModelResponseTypes)
