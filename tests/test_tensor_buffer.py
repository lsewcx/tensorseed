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

    def test_buffer_multi_dtype(self) -> None:
        t_f64 = ts.empty([2, 2], dtype=ts.float64)
        assert memoryview(t_f64).itemsize == 8
        assert memoryview(t_f64).format == "d"

        t_i32 = ts.empty([2, 2], dtype=ts.int32)
        assert memoryview(t_i32).itemsize == 4
        assert memoryview(t_i32).format == "i"

        t_i64 = ts.empty([2, 2], dtype=ts.int64)
        assert memoryview(t_i64).itemsize == 8
        assert memoryview(t_i64).format == "q"

        t_u8 = ts.empty([2, 2], dtype=ts.uint8)
        assert memoryview(t_u8).itemsize == 1
        assert memoryview(t_u8).format == "B"
