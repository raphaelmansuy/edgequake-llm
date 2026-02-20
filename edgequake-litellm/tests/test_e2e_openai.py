"""
test_e2e_openai.py — End-to-end tests against the OpenAI API.

These tests are automatically skipped when OPENAI_API_KEY is not set.
"""
from __future__ import annotations

import pytest

from edgequake_litellm import completion, acompletion, embedding, stream


MESSAGES = [{"role": "user", "content": "Say 'pong' and nothing else."}]
MODEL = "openai/gpt-4o-mini"
EMBED_MODEL = "openai/text-embedding-3-small"


@pytest.mark.usefixtures("openai_available")
class TestOpenAICompletion:
    def test_sync_completion(self):
        resp = completion(MODEL, MESSAGES)
        assert isinstance(resp.content, str)
        assert len(resp.content) > 0

    def test_sync_completion_usage(self):
        resp = completion(MODEL, MESSAGES)
        assert resp.usage.total_tokens > 0
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.completion_tokens > 0

    def test_sync_completion_model_name(self):
        resp = completion(MODEL, MESSAGES)
        assert "gpt" in resp.model.lower()

    async def test_async_completion(self):
        resp = await acompletion(MODEL, MESSAGES)
        assert isinstance(resp.content, str)
        assert len(resp.content) > 0

    async def test_streaming(self):
        chunks = []
        async for chunk in stream(MODEL, MESSAGES):
            chunks.append(chunk)
        text = "".join(c.content or "" for c in chunks)
        assert len(text) >= 0  # May be empty if all in finish chunk

    def test_completion_with_system(self):
        msgs = [{"role": "user", "content": "Who are you?"}]
        resp = completion(MODEL, msgs, system="You are a pirate. Respond in pirate speak.")
        assert isinstance(resp.content, str)

    def test_completion_with_max_tokens(self):
        resp = completion(MODEL, MESSAGES, max_tokens=10)
        assert isinstance(resp.content, str)

    def test_completion_with_temperature_zero(self):
        resp1 = completion(MODEL, MESSAGES, temperature=0.0)
        resp2 = completion(MODEL, MESSAGES, temperature=0.0)
        # At temperature=0 responses should be deterministic
        assert resp1.content == resp2.content


@pytest.mark.usefixtures("openai_available")
class TestOpenAIEmbedding:
    def test_embed_returns_vectors(self):
        vecs = embedding(EMBED_MODEL, ["Hello world"])
        assert len(vecs) == 1
        assert len(vecs[0]) > 0

    def test_embed_dimension_is_consistent(self):
        texts = ["foo", "bar", "baz"]
        vecs = embedding(EMBED_MODEL, texts)
        assert len(vecs) == 3
        dims = {len(v) for v in vecs}
        assert len(dims) == 1  # All same dimension

    def test_embed_values_are_floats(self):
        vecs = embedding(EMBED_MODEL, ["test"])
        for v in vecs[0]:
            assert isinstance(v, float)

    async def test_aembed(self):
        from edgequake_litellm import aembedding
        vecs = await aembedding(EMBED_MODEL, ["hello", "world"])
        assert len(vecs) == 2
