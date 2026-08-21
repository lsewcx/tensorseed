"""TensorSeed: A PyTorch-style Tensor Engine in C++ and Python."""

from __future__ import annotations

import sys
from pathlib import Path

# 自动探测并加载 C++ 编译生成的 _core 动态库
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SEARCH_PATHS = [
    Path(__file__).resolve().parent,
    _PROJECT_ROOT / "cpp" / "build" / "build" / "Release",
    _PROJECT_ROOT / "cpp" / "build" / "Release",
    _PROJECT_ROOT / "cpp" / "build" / "build" / "Debug",
    _PROJECT_ROOT / "cpp" / "build" / "Debug",
    _PROJECT_ROOT / "cpp" / "build",
]

for _path in _SEARCH_PATHS:
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    import _core  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        "Could not load the compiled C++ `_core` module. "
        "Please compile the C++ extension first by running `.\\scripts\\cpp.ps1`."
    ) from exc

from typing import Sequence

Tensor = _core.Tensor
dtype = _core.dtype

# 顶级导出常用 ScalarType 常量（对齐 PyTorch 用法）
float32: _core.dtype = _core.dtype.float32
float64: _core.dtype = _core.dtype.float64
int32: _core.dtype = _core.dtype.int32
int64: _core.dtype = _core.dtype.int64
uint8: _core.dtype = _core.dtype.uint8


def empty(shape: Sequence[int], dtype: _core.dtype = _core.dtype.float32) -> Tensor:
    """Create an uninitialized Tensor with the specified shape and dtype."""
    return Tensor.empty(shape, dtype)


def tensor(data: Sequence[float]) -> Tensor:
    """Construct a 1D Tensor from a Python float sequence."""
    return Tensor(data)


def zeros(shape: Sequence[int], dtype: _core.dtype = _core.dtype.float32) -> Tensor:
    """Create a zero-initialized Tensor with the specified shape and dtype."""
    return Tensor.zeros(shape, dtype)


def ones(shape: Sequence[int], dtype: _core.dtype = _core.dtype.float32) -> Tensor:
    """Create a ones-initialized Tensor with the specified shape and dtype."""
    return Tensor.ones(shape, dtype)


__all__ = [
    "Tensor",
    "dtype",
    "empty",
    "zeros",
    "ones",
    "tensor",
    "float32",
    "float64",
    "int32",
    "int64",
    "uint8",
]
