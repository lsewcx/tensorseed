"""Quick start and functional verification script for TensorSeed."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure `python/` is in sys.path
_PYTHON_ROOT = Path(__file__).resolve().parent
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

import tensorseed as ts


def main() -> None:
    """Run interactive quick-verification suite."""
    print("========================================")
    print("       TensorSeed Quick Demo            ")
    print("========================================")

    # 1. 基础 Tensor 创建与元数据
    t1 = ts.tensor([1.5, 2.0, 3.0])
    print(f"\n[1] 1D Tensor: {t1}")
    print(f"    shape: {t1.shape}, strides: {t1.strides}, ndim: {t1.ndim}, numel: {len(t1)}")
    print(f"    tolist: {t1.tolist()}")
    assert t1.tolist() == [1.5, 2.0, 3.0]

    # 2. Buffer Protocol (memoryview)
    mv = memoryview(t1)
    print(f"\n[2] Buffer Protocol (memoryview): {list(mv)}")
    assert mv.tolist() == [1.5, 2.0, 3.0]

    # 3. 多维 Tensor 与 View 变换
    t2 = ts.empty([2, 3])
    print(f"\n[3] 2D Empty Tensor: {t2}")
    t2_view = t2.view([3, 2])
    print(f"    View (3x2): {t2_view}")
    t2_infer = t2.view([-1, 1])
    print(f"    Inferred View (-1, 1 -> 6x1): {t2_infer}")

    # 4. 零拷贝转置与连续化
    t2_t = t2.t()
    print(f"\n[4] Transposed (Zero-copy, non-contiguous): {t2_t}")
    assert t2_t.is_contiguous is False
    t2_c = t2_t.contiguous()
    print(f"    Contiguous copy: {t2_c}")
    assert t2_c.is_contiguous is True

    print("\n========================================")
    print("[SUCCESS] Quick verification completed!")
    print("========================================")


if __name__ == "__main__":
    main()
