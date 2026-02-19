"""
test_e2e_ollama.py — End-to-end tests against a local Ollama server.

These tests are automatically skipped when Ollama is not reachable.
Set OLLAMA_HOST to override the default (http://localhost:11434).

Required models:
    - ollama/llama3.2  (or any chat model)
    - ollama/nomic-embed-text  (for embedding tests)
"""
from __future__ import annotations

import os
import pytest

from edgequake_litellm import completion, acompletion, stream, embedding, aembedding


CHAT_MODEL = "ollama/llama3.2"
EMBED_MODEL = "ollama/nomic-embed-text"
MESSAGES = [{"role": "user", "content": "Reply with exactly one word: pong"}]


@pytest.mark.usefixtures("ollama_available")
class TestOllamaCompletion:
    def test_sync_completion(self, ollama_available):
        resp = completion(CHAT_MODEL, MESSAGES, api_base=ollama_available)
        assert isinstance(resp.content, str)
        assert len(resp.content) > 0

    def test_sync_completion_model_name(self, ollama_available):
        resp = completion(CHAT_MODEL, MESSAGES, api_base=ollama_available)
        assert resp.model != ""

    def test_completion_with_system(self, ollama_available):
        msgs = [{"role": "user", "content": "Say hello."}]
        resp = completion(
            CHAT_MODEL, msgs,
            system="You are a helpful assistant.",
            api_base=ollama_available,
        )
        assert isinstance(resp.content, str)
        assert len(resp.content) > 0

    def test_completion_with_max_tokens(self, ollama_available):
        resp = completion(CHAT_MODEL, MESSAGES, max_tokens=32, api_base=ollama_available)
        assert isinstance(resp.content, str)

    def test_completion_with_temperature(self, ollama_available):
        resp = completion(CHAT_MODEL, MESSAGES, temperature=0.1, api_base=ollama_available)
        assert isinstance(resp.content, str)

    async def test_async_completion(self, ollama_available):
        resp = await acompletion(CHAT_MODEL, MESSAGES, api_base=ollama_available)
        assert isinstance(resp.content, str)
        assert len(resp.content) > 0

    async def test_streaming(self, ollama_available):
        chunks = []
        async for chunk in stream(CHAT_MODEL, MESSAGES, api_base=ollama_available):
            chunks.append(chunk)
        assert len(chunks) >= 1
        text = "".join(c.content or "" for c in chunks)
        assert isinstance(text, str)

    async def test_streaming_accumulates_content(self, ollama_available):
        accumulated = ""
        async for chunk in stream(CHAT_MODEL, MESSAGES, api_base=ollama_available):
            if chunk.content:
                accumulated += chunk.content
        assert len(accumulated) >= 1

    def test_multi_turn(self, ollama_available):
        msgs = [
            {"role": "user", "content": "My name is Bob."},
            {"role": "assistant", "content": "Nice to meet you, Bob!"},
            {"role": "user", "content": "What is my name?"},
        ]
        resp = completion(CHAT_MODEL, msgs, api_base=ollama_available)
        assert isinstance(resp.content, str)


@pytest.mark.usefixtures("ollama_available")
class TestOllamaEmbedding:
    def test_embed_returns_vectors(self, ollama_available):
        vecs = embedding(EMBED_MODEL, ["Hello world"], api_base=ollama_available)
        assert len(vecs) == 1
        assert len(vecs[0]) > 0

    def test_embed_multiple_texts(self, ollama_available):
        texts = ["foo", "bar", "baz"]
        vecs = embedding(EMBED_MODEL, texts, api_base=ollama_available)
        assert len(vecs) == 3
        # All vectors have the same dimension
        dims = {len(v) for v in vecs}
        assert len(dims) == 1

    def test_embed_values_are_floats(self, ollama_available):
        vecs = embedding(EMBED_MODEL, ["test"], api_base=ollama_available)
        for v in vecs[0]:
            assert isinstance(v, float)

    async def test_async_embedding(self, ollama_available):
        vecs = await aembedding(EMBED_MODEL, ["hello", "world"], api_base=ollama_available)
        assert len(vecs) == 2
        assert all(len(v) > 0 for v in vecs)
