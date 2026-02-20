"""
test_embedding_unit.py — Unit tests for embedding using the mock provider.
"""
from __future__ import annotations

import pytest

from edgequake_litellm import embedding, aembedding
from edgequake_litellm._compat import EmbeddingResponseCompat


TEXTS = ["Hello world", "Rust is fast", "edgequake-litellm rocks"]


class TestEmbedUnit:
    def test_embed_returns_list_of_lists(self):
        result = embedding("mock/test-embedding", TEXTS)
        # Result is EmbeddingResponseCompat — supports both list-style and .data access
        assert isinstance(result, (list, EmbeddingResponseCompat))
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

    def test_embed_compat_data_access(self):
        """EmbeddingResponseCompat supports litellm-style .data[i].embedding access."""
        result = embedding("mock/test-embedding", TEXTS)
        assert isinstance(result, EmbeddingResponseCompat)
        assert len(result.data) == len(TEXTS)
        assert isinstance(result.data[0].embedding, list)
        assert isinstance(result.model, str)

    def test_embed_compat_legacy_list_access(self):
        """EmbeddingResponseCompat still behaves like List[List[float]] for backwards compat."""
        result = embedding("mock/test-embedding", TEXTS)
        # Index access
        assert isinstance(result[0], list)
        # Iteration
        vecs = list(result)
        assert len(vecs) == len(TEXTS)
        assert all(isinstance(v, list) for v in vecs)


class TestAembedUnit:
    async def test_aembed_returns_list_of_lists(self):
        result = await aembedding("mock/test-embedding", TEXTS)
        assert isinstance(result, (list, EmbeddingResponseCompat))
        assert len(result) == len(TEXTS)

    async def test_aembed_vectors_are_floats(self):
        result = await aembedding("mock/test-embedding", TEXTS)
        for vec in result:
            for v in vec:
                assert isinstance(v, float)

    async def test_aembed_unknown_provider_raises(self):
        with pytest.raises(Exception):
            await aembedding("bad_provider/model", TEXTS)
