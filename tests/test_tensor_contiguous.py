"""Tests for memory continuity and .contiguous() copy operations."""

from __future__ import annotations

import pytest
import tensorseed as ts


class TestTensorContiguity:
    """Test suite for memory continuity validation and contiguous reordering."""

    def test_contiguous_on_already_contiguous(self) -> None:
        t = ts.tensor([1.0, 2.0, 3.0])
        assert t.is_contiguous is True

        # Calling .contiguous() on contiguous tensor should preserve continuity
        t_c = t.contiguous()
        assert t_c.is_contiguous is True
        assert t_c.tolist() == [1.0, 2.0, 3.0]
        assert t_c.shape == [3]

    def test_contiguous_on_transposed_matrix(self) -> None:
        # Create a 1D tensor [1, 2, 3, 4, 5, 6] and view as 2x3 matrix:
        # [[1, 2, 3],
        #  [4, 5, 6]]
        t_1d = ts.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        t_2d = t_1d.view([2, 3])

        # Transpose to 3x2:
        # [[1, 4],
        #  [2, 5],
        #  [3, 6]]
        t_transposed = t_2d.t()
        assert t_transposed.shape == [3, 2]
        assert t_transposed.is_contiguous is False

        # .contiguous() should produce a contiguous copy with row-major layout:
        # [1, 4, 2, 5, 3, 6]
        t_contiguous = t_transposed.contiguous()
        assert t_contiguous.is_contiguous is True
        assert t_contiguous.shape == [3, 2]
        assert t_contiguous.strides == [2, 1]
        assert t_contiguous.tolist() == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
