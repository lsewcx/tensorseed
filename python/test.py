import sys
from pathlib import Path

module_path: Path = (
    r"C:\code\c\TensorSeed\cpp\build\build\Release"
)

sys.path.insert(0, str(module_path))

import _core


def main() -> None:
    """验证Python能够创建并读取C++ Tensor。

    Returns:
        None。
    """
    data: list[float] = [1.5, 2.0, 3.0]
    tensor: _core.Tensor = _core.Tensor(data)

    assert tensor.data == data
    assert len(tensor) == 3
    print(f"Tensor测试成功：data={tensor.data}, size={len(tensor)}")


if __name__ == "__main__":
    main()
