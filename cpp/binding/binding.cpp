#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * 计算两个整数的和。
 *
 * Args:
 *     left: 左侧整数。
 *     right: 右侧整数。
 *
 * Returns:
 *     两个整数的和。
 */
int add(const int left, const int right) {
    return left + right;
}

/**
 * 注册TensorSeed的Python扩展模块。
 *
 * Args:
 *     module: Python模块对象。
 */
PYBIND11_MODULE(_core, module) {
    module.doc() = "TensorSeed的C++核心模块";

    module.def(
        "add",
        &add,
        py::arg("left"),
        py::arg("right"),
        "计算两个整数的和"
    );
}