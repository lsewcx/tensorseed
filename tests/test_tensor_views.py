"""Tests for Tensor view operations, reshaping, and transposition."""

from __future__ import annotations

import pytest
import tensorseed as ts


class TestTensorViewOperations:
    """Test suite for tensor reshaping and view projections."""

    def test_view_reshape(self) -> None:
        t = ts.empty([2, 3])
        t_v = t.view([3, 2])

        assert t_v.shape == [3, 2]
        assert t_v.strides == [2, 1]
        assert len(t_v) == 6
        assert t_v.is_contiguous is True

    def test_view_infer_dimension(self) -> None:
        t = ts.empty([2, 3, 4])  # total = 24

        # Infer first dimension: [-1, 4] -> [6, 4]
        t1 = t.view([-1, 4])
        assert t1.shape == [6, 4]

        # Infer middle dimension: [2, -1, 2] -> [2, 6, 2]
        t2 = t.view([2, -1, 2])
        assert t2.shape == [2, 6, 2]

    def test_view_invalid_infer_raises(self) -> None:
        t = ts.empty([2, 3])  # total = 6

        # Multiple -1 dimensions should fail
        with pytest.raises(ValueError, match="Only one dimension can be -1"):
            t.view([-1, -1])

        # Indivisible dimension should fail
        with pytest.raises(ValueError, match="Shape mismatch"):
            t.view([-1, 4])

    def test_view_mismatch_without_infer_raises(self) -> None:
        t = ts.empty([2, 3])  # total = 6
        # Shape product != total_elements should fail
        with pytest.raises(ValueError, match="total elements do not match"):
            t.view([2, 4])

    def test_transpose_2d(self) -> None:
        t = ts.empty([2, 3])
        t_t = t.t()

        assert t_t.shape == [3, 2]
        assert t_t.strides == [1, 3]  # Swapped strides, zero-copy
        assert t_t.is_contiguous is False

    def test_double_transpose_is_identity(self) -> None:
        t = ts.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).view([2, 3])
        t_double = t.t().t()
        assert t_double.shape == [2, 3]
        assert t_double.strides == [3, 1]
        assert t_double.is_contiguous is True
        assert t_double.tolist() == t.tolist()

    def test_3d_transpose_and_negative_dims(self) -> None:
        t = ts.empty([2, 3, 4])
        # Transpose dim 0 and 2
        t_swapped = t.transpose(0, 2)
        assert t_swapped.shape == [4, 3, 2]
        assert t_swapped.strides == [1, 4, 12]

        # Negative dimension transpose
        t_neg = t.transpose(-3, -1)
        assert t_neg.shape == [4, 3, 2]

    def test_transpose_invalid_dimensions(self) -> None:
        t = ts.empty([2, 3])

        # Out of bounds dimension
        with pytest.raises(Exception):
            t.transpose(0, 5)

        # .t() on 1D tensor should raise
        t_1d = ts.tensor([1.0, 2.0])
        with pytest.raises(RuntimeError, match="2D tensor"):
            t_1d.t()

    def test_view_on_non_contiguous_raises(self) -> None:
        t = ts.empty([2, 3])
        t_non_contig = t.t()  # Transposed tensor is non-contiguous

        with pytest.raises(RuntimeError, match="contiguous"):
            t_non_contig.view([6])
