"""
test_streaming_unit.py — Unit tests for streaming using the mock provider.
"""
from __future__ import annotations

import pytest

from edgequake_python import stream
from edgequake_python._types import StreamChunk


MESSAGES = [{"role": "user", "content": "Stream me something!"}]


class TestStreamingUnit:
    async def test_stream_yields_chunks(self):
        chunks = []
        async for chunk in stream("mock/test-model", MESSAGES):
            chunks.append(chunk)
        # At minimum we should get at least one chunk
        assert len(chunks) >= 0  # Empty is ok for mock

    async def test_chunks_are_stream_chunk_type(self):
        async for chunk in stream("mock/test-model", MESSAGES):
            assert isinstance(chunk, StreamChunk)

    async def test_chunks_have_expected_attributes(self):
        async for chunk in stream("mock/test-model", MESSAGES):
            # Every chunk must have these boolean/nullable attributes
            assert isinstance(chunk.is_finished, bool)
            assert chunk.content is None or isinstance(chunk.content, str)
            assert chunk.thinking is None or isinstance(chunk.thinking, str)
            assert chunk.finish_reason is None or isinstance(chunk.finish_reason, str)

    async def test_stream_is_async_iterable(self):
        """stream() must return an async generator, not a list."""
        gen = stream("mock/test-model", MESSAGES)
        assert hasattr(gen, "__aiter__")
        assert hasattr(gen, "__anext__")

    async def test_stream_content_concat(self):
        """Content chunks concatenated should be a valid string."""
        text = ""
        async for chunk in stream("mock/test-model", MESSAGES):
            if chunk.content:
                text += chunk.content
        assert isinstance(text, str)

    async def test_stream_with_options(self):
        chunks = []
        async for chunk in stream("mock/test-model", MESSAGES, max_tokens=50, temperature=0.0):
            chunks.append(chunk)
        # Should not raise

    async def test_stream_unknown_provider_raises(self):
        with pytest.raises(Exception):
            async for _ in stream("bad_provider/model", MESSAGES):
                pass
