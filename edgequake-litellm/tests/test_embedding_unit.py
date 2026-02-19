"""
test_embedding_unit.py — Unit tests for embedding using the mock provider.
"""
from __future__ import annotations

import pytest

from edgequake_litellm import embedding, aembedding


TEXTS = ["Hello world", "Rust is fast", "edgequake-litellm rocks"]


class TestEmbedUnit:
    def test_embed_returns_list_of_lists(self):
        result = embedding("mock/test-embedding", TEXTS)
        assert isinstance(result, list)
        assert len(result) == len(TEXTS)
        for vec in result:
            assert isinstance(vec, list)

    def test_embed_vectors_are_floats(self):
        result = embedding("mock/test-embedding", TEXTS)
        for vec in result:
            for v in vec:
                assert isinstance(v, float)

    def test_embed_single_text(self):
        result = embedding("mock/test-embedding", ["single text"])
        assert len(result) == 1
        assert isinstance(result[0], list)

    def test_embed_consistent_dimension(self):
        """All vectors in a batch should have the same dimension."""
        result = embedding("mock/test-embedding", TEXTS)
        dims = {len(v) for v in result}
        assert len(dims) == 1  # All same dimension

    def test_embed_unknown_provider_raises(self):
        with pytest.raises(Exception):
            embedding("bad_provider/model", TEXTS)


class TestAembedUnit:
    async def test_aembed_returns_list_of_lists(self):
        result = await aembedding("mock/test-embedding", TEXTS)
        assert isinstance(result, list)
        assert len(result) == len(TEXTS)

    async def test_aembed_vectors_are_floats(self):
        result = await aembedding("mock/test-embedding", TEXTS)
        for vec in result:
            for v in vec:
                assert isinstance(v, float)

    async def test_aembed_unknown_provider_raises(self):
        with pytest.raises(Exception):
            await aembedding("bad_provider/model", TEXTS)
