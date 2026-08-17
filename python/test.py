import sys
from pathlib import Path

module_path: Path = (
    r"C:\code\c\TensorSeed\cpp\build\build\Release"
)

sys.path.insert(0, str(module_path))

import _core


def main() -> None:
    """验证 Python 能够创建、读取并操作 C++ Tensor。"""
    print("====== 1. 基础 1D Tensor 测试 ======")
    data: list[float] = [1.5, 2.0, 3.0]
    tensor: _core.Tensor = _core.Tensor(data)

    assert tensor.tolist() == data, f"Expected {data}, got {tensor.tolist()}"
    assert len(tensor) == 3
    assert tensor.shape == [3]
    assert tensor.strides == [1]
    assert tensor.ndim == 1
    assert tensor.is_contiguous is True
    print(f"1D Tensor: {tensor}")
    print(f"tolist: {tensor.tolist()}, size: {len(tensor)}")

    print("\n====== 2. Buffer Protocol 测试 ======")
    mv = memoryview(tensor)
    assert mv.tolist() == data
    print(f"Memoryview 正常工作: {list(mv)}")

    print("\n====== 3. 多维 Tensor 与 View 变换测试 ======")
    t_2d: _core.Tensor = _core.Tensor.empty([2, 3], _core.dtype.float32)
    assert t_2d.shape == [2, 3]
    assert t_2d.strides == [3, 1]
    assert t_2d.ndim == 2
    assert len(t_2d) == 6
    print(f"2D Empty Tensor: {t_2d}")

    # View 变换
    t_view = t_2d.view([3, 2])
    assert t_view.shape == [3, 2]
    assert t_view.strides == [2, 1]
    print(f"View [3, 2]: {t_view}")

    # 支持 -1 自动推导
    t_infer = t_2d.view([-1, 1])
    assert t_infer.shape == [6, 1]
    print(f"View [-1, 1]: {t_infer}")

    print("\n====== 4. 零拷贝转置与连续化测试 ======")
    t_transposed = t_2d.t()
    assert t_transposed.shape == [3, 2]
    assert t_transposed.strides == [1, 3]  # stride 交换，零拷贝
    assert t_transposed.is_contiguous is False
    print(f"Transposed (非连续): {t_transposed}")

    # 连续化
    t_contig = t_transposed.contiguous()
    assert t_contig.shape == [3, 2]
    assert t_contig.strides == [2, 1]  # 连续内存 stride 重新对齐
    assert t_contig.is_contiguous is True
    print(f"Contiguous 副本: {t_contig}")

    print("\n[OK] All Tensor unit tests passed successfully!")


if __name__ == "__main__":
    main()
