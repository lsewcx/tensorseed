"""Tests for Python Buffer Protocol and zero-copy interoperability."""

from __future__ import annotations

import pytest
import tensorseed as ts


class TestTensorBufferProtocol:
    """Test suite for Python Buffer Protocol support."""

    def test_memoryview_read(self) -> None:
        data = [10.0, 20.0, 30.0, 40.0]
        t = ts.tensor(data)

        # Inspect via standard Python memoryview
        mv = memoryview(t)
        assert len(mv) == 4
        assert mv.tolist() == data

    def test_buffer_attributes(self) -> None:
        t = ts.empty([2, 3])
        mv = memoryview(t)

        assert mv.ndim == 2
        assert mv.shape == (2, 3)
        # In bytes: float32 = 4 bytes, strides [3, 1] -> [12, 4] bytes
        assert mv.strides == (12, 4)
        assert mv.itemsize == 4
