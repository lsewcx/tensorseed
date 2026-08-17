#include "../include/tensorseed/tensor.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace tensorseed;

PYBIND11_MODULE(_core, m) {
  m.doc() = "TensorSeed PyTorch-style Tensor Engine";

  py::enum_<ScalarType>(m, "dtype")
      .value("float32", ScalarType::Float32)
      .value("float64", ScalarType::Float64)
      .value("int32", ScalarType::Int32)
      .value("int64", ScalarType::Int64)
      .export_values();

  py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
      // 从 1D 列表构造
      .def(py::init([](const std::vector<float> &data) {
             return Tensor::from_vector(data);
           }),
           py::arg("data"))

      // empty 工厂函数
      .def_static("empty", &Tensor::empty, py::arg("shape"),
                  py::arg("dtype") = ScalarType::Float32)

      // 属性
      .def_property_readonly("shape", &Tensor::shape)
      .def_property_readonly("strides", &Tensor::strides)
      .def_property_readonly("ndim", &Tensor::ndim)
      .def_property_readonly("is_contiguous", &Tensor::is_contiguous)
      .def("__len__", &Tensor::numel)
      .def("__repr__", &Tensor::to_string)

      // 视图变换算子 (零拷贝)
      .def("transpose", &Tensor::transpose, py::arg("dim0"), py::arg("dim1"))
      .def("t", &Tensor::t)
      .def("view", &Tensor::view, py::arg("shape"))
      .def("contiguous", &Tensor::contiguous)

      // Buffer protocol 支持 (支持 memoryview / numpy.asarray 等)
      .def_buffer([](Tensor &t) -> py::buffer_info {
        std::vector<py::ssize_t> shape(t.shape().begin(), t.shape().end());
        std::vector<py::ssize_t> strides;
        for (auto s : t.strides()) {
          strides.push_back(s * static_cast<py::ssize_t>(element_size(t.dtype())));
        }
        return py::buffer_info(
            t.data_ptr<float>(),
            sizeof(float),
            py::format_descriptor<float>::format(),
            static_cast<py::ssize_t>(t.ndim()),
            shape,
            strides
        );
      })

      // 导出数据为 Python 列表
      .def("tolist", [](const Tensor &t) {
        Tensor c = t.is_contiguous() ? t : t.contiguous();
        const float *ptr = c.data_ptr<float>();
        return std::vector<float>(ptr, ptr + c.numel());
      });
}
