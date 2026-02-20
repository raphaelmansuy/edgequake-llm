"""
conftest.py — Shared pytest fixtures for edgequake-litellm tests.
"""
from __future__ import annotations

import os
import pytest


# ---------------------------------------------------------------------------
# Message fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_messages():
    """Simple single-turn messages for testing."""
    return [{"role": "user", "content": "Hello, world!"}]


@pytest.fixture
def multi_turn_messages():
    """Multi-turn conversation messages."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "user", "content": "What is 4+4?"},
    ]


@pytest.fixture
def ping_messages():
    """Ultra-short messages to minimise token cost in E2E tests."""
    return [{"role": "user", "content": "Reply with exactly one word: pong"}]


# ---------------------------------------------------------------------------
# Provider availability fixtures — skip when credentials are absent
# ---------------------------------------------------------------------------

@pytest.fixture
def openai_available():
    """Skip if OPENAI_API_KEY is not set."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture
def anthropic_available():
    """Skip if ANTHROPIC_API_KEY is not set."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


@pytest.fixture
def gemini_available():
    """Skip if GEMINI_API_KEY is not set."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY not set")
    return key


@pytest.fixture
def mistral_available():
    """Skip if MISTRAL_API_KEY is not set."""
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        pytest.skip("MISTRAL_API_KEY not set")
    return key


@pytest.fixture
def xai_available():
    """Skip if XAI_API_KEY is not set."""
    key = os.environ.get("XAI_API_KEY")
    if not key:
        pytest.skip("XAI_API_KEY not set")
    return key


@pytest.fixture
def openrouter_available():
    """Skip if OPENROUTER_API_KEY is not set."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set")
    return key


@pytest.fixture
def ollama_available():
    """Skip if Ollama is not reachable at OLLAMA_HOST (default localhost:11434)."""
    import urllib.request
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        urllib.request.urlopen(host, timeout=2)
        return host
    except Exception:
        pytest.skip(f"Ollama not reachable at {host}")
