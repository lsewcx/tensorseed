"""Tests for Tensor creation, attributes, and basic conversions."""

from __future__ import annotations

import pytest
import tensorseed as ts


class TestTensorCreation:
    """Test suite for Tensor initialization and metadata."""

    def test_from_list_1d(self) -> None:
        data = [1.5, 2.0, 3.5, 4.0]
        t = ts.tensor(data)

        assert t.shape == [4]
        assert t.strides == [1]
        assert t.ndim == 1
        assert len(t) == 4
        assert t.is_contiguous is True
        assert t.tolist() == data

    def test_empty_factory(self) -> None:
        shape = [2, 3, 4]
        t = ts.empty(shape, ts.dtype.float32)

        assert t.shape == shape
        assert t.strides == [12, 4, 1]  # Standard row-major contiguous strides
        assert t.ndim == 3
        assert len(t) == 24
        assert t.is_contiguous is True

    def test_dtype_variants(self) -> None:
        t_f32 = ts.empty([2, 2], ts.dtype.float32)
        assert "dtype=float32" in repr(t_f32)

        t_i32 = ts.empty([2, 2], ts.dtype.int32)
        assert "dtype=int32" in repr(t_i32)

        t_i64 = ts.empty([2, 2], ts.dtype.int64)
        assert "dtype=int64" in repr(t_i64)

    def test_repr_formatting(self) -> None:
        t = ts.tensor([1.0, 2.0])
        repr_str = repr(t)
        assert "Tensor(shape=[2]" in repr_str
        assert "dtype=float32" in repr_str
        assert "contiguous=True" in repr_str
