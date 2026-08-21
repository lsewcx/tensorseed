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
        shape = (2, 3, 4)
        t = ts.empty(shape, dtype=ts.float32)

        assert t.shape == [2, 3, 4]
        assert t.strides == [12, 4, 1]  # Standard row-major contiguous strides
        assert t.ndim == 3
        assert len(t) == 24
        assert t.is_contiguous is True

    def test_zeros_factory(self) -> None:
        # Default float32 zeros
        t_f32 = ts.zeros((2, 3))
        assert t_f32.shape == [2, 3]
        assert t_f32.strides == [3, 1]
        assert t_f32.is_contiguous is True
        assert t_f32.tolist() == [0.0] * 6

        # int32 zeros
        t_i32 = ts.zeros([4], dtype=ts.int32)
        assert t_i32.tolist() == [0, 0, 0, 0]

        # float64 zeros
        t_f64 = ts.zeros([2, 2], dtype=ts.float64)
        assert t_f64.tolist() == [0.0, 0.0, 0.0, 0.0]

        # uint8 zeros
        t_u8 = ts.zeros([3], dtype=ts.uint8)
        assert t_u8.tolist() == [0, 0, 0]

    def test_ones_factory(self) -> None:
        # Default float32 ones
        t_f32 = ts.ones((2, 3))
        assert t_f32.shape == [2, 3]
        assert t_f32.strides == [3, 1]
        assert t_f32.is_contiguous is True
        assert t_f32.tolist() == [1.0] * 6

        # int32 ones (must be exactly 1, not 0x01010101)
        t_i32 = ts.ones([4], dtype=ts.int32)
        assert t_i32.tolist() == [1, 1, 1, 1]

        # int64 ones
        t_i64 = ts.ones([2], dtype=ts.int64)
        assert t_i64.tolist() == [1, 1]

        # float64 ones
        t_f64 = ts.ones([2, 2], dtype=ts.float64)
        assert t_f64.tolist() == [1.0, 1.0, 1.0, 1.0]

        # uint8 ones
        t_u8 = ts.ones([3], dtype=ts.uint8)
        assert t_u8.tolist() == [1, 1, 1]

    def test_dtype_variants(self) -> None:
        t_f32 = ts.empty([2, 2], dtype=ts.float32)
        assert "dtype=float32" in repr(t_f32)

        t_f64 = ts.empty([2, 2], dtype=ts.float64)
        assert "dtype=float64" in repr(t_f64)

        t_i32 = ts.empty([2, 2], dtype=ts.int32)
        assert "dtype=int32" in repr(t_i32)

        t_i64 = ts.empty([2, 2], dtype=ts.int64)
        assert "dtype=int64" in repr(t_i64)

        t_u8 = ts.empty([2, 2], dtype=ts.uint8)
        assert "dtype=uint8" in repr(t_u8)

    def test_repr_formatting(self) -> None:
        t = ts.tensor([1.0, 2.0])
        repr_str = repr(t)
        assert "Tensor(shape=[2]" in repr_str
        assert "dtype=float32" in repr_str
        assert "contiguous=True" in repr_str
