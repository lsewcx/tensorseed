"""Tests for Tensor indexing, slicing, and __getitem__/__setitem__ support."""

from __future__ import annotations

import pytest
import tensorseed as ts


class TestTensorIndexing:
    """Comprehensive test suite for tensor subscripting, view slicing, and mutations."""

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

    def test_3d_tensor_indexing_and_slicing(self) -> None:
        # 3D Tensor of shape [2, 3, 4] with elements 0..23
        data = [float(i) for i in range(24)]
        t = ts.tensor(data).view([2, 3, 4])

        assert t.shape == [2, 3, 4]
        # Rank reduction from 3D to 2D
        t0 = t[0]
        assert t0.shape == [3, 4]
        assert t0[0, 0] == 0.0
        assert t0[2, 3] == 11.0

        # Rank reduction from 3D to 1D via tuple
        t0_1 = t[0, 1]
        assert t0_1.shape == [4]
        assert t0_1.tolist() == [4.0, 5.0, 6.0, 7.0]

        # Scalar access via 3D tuple
        assert t[1, 2, 3] == 23.0
        assert t[0, 0, 0] == 0.0
        assert t[-1, -1, -1] == 23.0

        # Chained 3D access
        assert t[1][2][3] == 23.0

    def test_slice_operations_1d(self) -> None:
        t = ts.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        sub = t[1:4]
        assert sub.shape == [3]
        assert sub.tolist() == [1.0, 2.0, 3.0]

        sub_step = t[0:6:2]
        assert sub_step.shape == [3]
        assert sub_step.tolist() == [0.0, 2.0, 4.0]

    def test_slice_omitted_bounds(self) -> None:
        t = ts.tensor([10.0, 20.0, 30.0, 40.0, 50.0])

        # t[:]
        assert t[:].tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]
        # t[2:]
        assert t[2:].tolist() == [30.0, 40.0, 50.0]
        # t[:3]
        assert t[:3].tolist() == [10.0, 20.0, 30.0]
        # t[::2]
        assert t[::2].tolist() == [10.0, 30.0, 50.0]

    def test_multidim_slice(self) -> None:
        # [[1, 2, 3],
        #  [4, 5, 6],
        #  [7, 8, 9]]
        t = ts.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).view([3, 3])

        sub = t[0:2, 1:3]
        assert sub.shape == [2, 2]
        assert sub.tolist() == [2.0, 3.0, 5.0, 6.0]

    def test_indexing_on_non_contiguous_transposed_tensor(self) -> None:
        # [[1, 2, 3],
        #  [4, 5, 6]]
        t = ts.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).view([2, 3])
        # Transposed shape: [3, 2]
        # [[1, 4],
        #  [2, 5],
        #  [3, 6]]
        t_t = t.t()
        assert not t_t.is_contiguous

        assert t_t[0, 0] == 1.0
        assert t_t[0, 1] == 4.0
        assert t_t[1, 0] == 2.0
        assert t_t[1, 1] == 5.0
        assert t_t[2, 0] == 3.0
        assert t_t[2, 1] == 6.0

        # Row slice on transposed
        row1 = t_t[1]
        assert row1.shape == [2]
        assert row1.tolist() == [2.0, 5.0]

    def test_mutation_on_non_contiguous_transposed_tensor(self) -> None:
        t = ts.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).view([2, 3])
        t_t = t.t()

        # Modify element (1, 0) in transposed view (corresponds to (0, 1) in parent)
        t_t[1, 0] = 999.0
        assert t_t[1, 0] == 999.0
        assert t[0, 1] == 999.0

    def test_multi_dtype_indexing_and_mutation(self) -> None:
        # float64
        t_f64 = ts.zeros([2, 2], dtype=ts.float64)
        t_f64[0, 1] = 3.141592653589793
        assert abs(t_f64[0, 1] - 3.141592653589793) < 1e-12

        # int32
        t_i32 = ts.zeros([2, 2], dtype=ts.int32)
        t_i32[1, 0] = 123456
        assert t_i32[1, 0] == 123456

        # int64
        t_i64 = ts.zeros([2, 2], dtype=ts.int64)
        t_i64[1, 1] = 9876543210123456
        assert t_i64[1, 1] == 9876543210123456

        # uint8
        t_u8 = ts.zeros([2, 2], dtype=ts.uint8)
        t_u8[0, 0] = 255
        assert t_u8[0, 0] == 255

    def test_too_many_indices_raises_index_error(self) -> None:
        t = ts.zeros([2, 3])
        with pytest.raises(IndexError, match="Too many indices"):
            _ = t[0, 0, 0]

    def test_invalid_index_types_raise_type_error(self) -> None:
        t = ts.zeros([2, 3])
        with pytest.raises(TypeError):
            _ = t["invalid_string"]  # type: ignore[index]
        with pytest.raises(TypeError):
            _ = t[1.5]  # type: ignore[index]

    def test_select_method_directly(self) -> None:
        t = ts.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).view([2, 3])
        # select along dim 0
        s0 = t.select(0, 1)
        assert s0.shape == [3]
        assert s0.tolist() == [4.0, 5.0, 6.0]

        # select along dim 1
        s1 = t.select(1, 2)
        assert s1.shape == [2]
        assert s1.tolist() == [3.0, 6.0]

        # out of range dim
        with pytest.raises(IndexError):
            _ = t.select(5, 0)
        # out of range index
        with pytest.raises(IndexError):
            _ = t.select(0, 10)

    def test_slice_method_directly(self) -> None:
        t = ts.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).view([2, 3])
        sl = t.slice(1, 0, 2, 1)
        assert sl.shape == [2, 2]
        assert sl.tolist() == [1.0, 2.0, 4.0, 5.0]

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
