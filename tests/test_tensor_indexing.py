"""Tests for Tensor indexing, slicing, and __getitem__/__setitem__ support."""

from __future__ import annotations

import pytest
import tensorseed as ts


class TestTensorIndexing:
    """Test suite for tensor subscripting and view slicing."""

    def test_1d_indexing_read(self) -> None:
        t = ts.tensor([10.0, 20.0, 30.0, 40.0])

        assert t[0] == 10.0
        assert t[1] == 20.0
        assert t[3] == 40.0
        # Negative indexing
        assert t[-1] == 40.0
        assert t[-2] == 30.0

    def test_1d_indexing_out_of_bounds(self) -> None:
        t = ts.tensor([1.0, 2.0])
        with pytest.raises(IndexError):
            _ = t[5]
        with pytest.raises(IndexError):
            _ = t[-3]

    def test_2d_row_slice_view(self) -> None:
        # [[1.0, 2.0, 3.0],
        #  [4.0, 5.0, 6.0]]
        t = ts.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).view([2, 3])

        row0 = t[0]
        assert isinstance(row0, ts.Tensor)
        assert row0.shape == [3]
        assert row0.tolist() == [1.0, 2.0, 3.0]

        row1 = t[1]
        assert row1.tolist() == [4.0, 5.0, 6.0]

        # Negative indexing for rows
        assert t[-1].tolist() == [4.0, 5.0, 6.0]

    def test_chain_indexing_2d(self) -> None:
        t = ts.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).view([2, 3])

        # Chained subscripting t[row][col]
        assert t[0][0] == 1.0
        assert t[0][1] == 2.0
        assert t[0][2] == 3.0
        assert t[1][0] == 4.0
        assert t[1][1] == 5.0
        assert t[1][2] == 6.0

    def test_tuple_indexing_2d(self) -> None:
        t = ts.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).view([2, 3])

        # Tuple indexing t[row, col]
        assert t[0, 0] == 1.0
        assert t[0, 1] == 2.0
        assert t[1, 2] == 6.0
        assert t[-1, -1] == 6.0

    def test_slice_operations(self) -> None:
        # [0, 1, 2, 3, 4, 5]
        t = ts.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        sub = t[1:4]
        assert sub.shape == [3]
        assert sub.tolist() == [1.0, 2.0, 3.0]

        sub_step = t[0:6:2]
        assert sub_step.shape == [3]
        assert sub_step.tolist() == [0.0, 2.0, 4.0]

    def test_multidim_slice(self) -> None:
        # [[1, 2, 3],
        #  [4, 5, 6],
        #  [7, 8, 9]]
        t = ts.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).view([3, 3])

        sub = t[0:2, 1:3]
        assert sub.shape == [2, 2]
        assert sub.tolist() == [2.0, 3.0, 5.0, 6.0]

    def test_1d_setitem_mutation(self) -> None:
        t = ts.tensor([1.0, 2.0, 3.0])
        t[1] = 99.0
        assert t[1] == 99.0
        assert t.tolist() == [1.0, 99.0, 3.0]

    def test_2d_setitem_mutation(self) -> None:
        t = ts.tensor([1.0, 2.0, 3.0, 4.0]).view([2, 2])
        t[0, 1] = 88.0
        assert t[0, 1] == 88.0
        assert t.tolist() == [1.0, 88.0, 3.0, 4.0]

    def test_zero_copy_view_mutation_reflects_in_parent(self) -> None:
        # Modifying a row view should mutate the underlying storage
        t = ts.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).view([2, 3])
        row0 = t[0]
        row0[1] = 777.0

        # Parent tensor must reflect the change!
        assert t[0, 1] == 777.0
        assert t.tolist() == [1.0, 777.0, 3.0, 4.0, 5.0, 6.0]

    def test_item_method(self) -> None:
        t = ts.tensor([42.5])
        assert t.item() == 42.5

        t_multi = ts.tensor([1.0, 2.0])
        with pytest.raises(RuntimeError, match="1 element"):
            t_multi.item()
