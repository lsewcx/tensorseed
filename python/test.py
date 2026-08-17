import _core


def main() -> None:
    """验证Python能够调用C++函数。

    Returns:
        None。
    """
    result: int = _core.add(2, 3)

    assert result == 5
    print(f"pybind11测试成功：2 + 3 = {result}")


if __name__ == "__main__":
    main()