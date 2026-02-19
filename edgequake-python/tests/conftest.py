"""
conftest.py — Shared pytest fixtures.
"""
from __future__ import annotations

import os
import pytest


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
def ollama_available():
    """Skip if OLLAMA_HOST is not set (or localhost unreachable)."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    import urllib.request
    try:
        urllib.request.urlopen(host, timeout=2)
    except Exception:
        pytest.skip(f"Ollama not reachable at {host}")
    return host
