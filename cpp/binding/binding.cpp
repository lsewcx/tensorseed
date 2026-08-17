#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensorseed/tensor.hpp"

namespace py = pybind11;

/**
 * 注册TensorSeed的Python扩展模块。
 *
 * Args:
 *     module: Python模块对象。
 */
PYBIND11_MODULE(_core, module) {
    module.doc() = "TensorSeed的C++核心模块";

    py::class_<tensorseed::Tensor>(module, "Tensor")
        .def(
            py::init<std::vector<float>>(),
            py::arg("data")
        )
        .def_property_readonly(
            "data",
            &tensorseed::Tensor::data
        )
        .def(
            "__len__",
            &tensorseed::Tensor::size
        );
}