#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tensorseed/tensor.hpp>

namespace py = pybind11;
using namespace tensorseed;

PYBIND11_MODULE(_core, m)
{
  m.doc() = "TensorSeed: High-performance PyTorch-style Tensor Engine in C++";

  py::enum_<ScalarType>(m, "dtype", "Data types supported by TensorSeed")
      .value("float32", ScalarType::Float32, "32-bit floating point")
      .value("float64", ScalarType::Float64, "64-bit floating point")
      .value("int32", ScalarType::Int32, "32-bit signed integer")
      .value("int64", ScalarType::Int64, "64-bit signed integer")
      .value("uint8", ScalarType::UInt8, "8-bit unsigned integer")
      .export_values();

  py::class_<Tensor>(m, "Tensor", py::buffer_protocol(),
                     "A multi-dimensional array with strided view capabilities")
      // 从 1D 列表构造
      .def(py::init([](const std::vector<float> &data)
                    { return Tensor::from_vector(data); }),
           py::arg("data"), "Construct a 1D Tensor from a float list")

      .def_static("zeros", &Tensor::zeros, py::arg("shape"),
                  py::arg("dtype") = ScalarType::Float32,
                  "Create a zero-initialized Tensor with given shape and dtype")

      // empty 工厂函数
      .def_static("empty", &Tensor::empty, py::arg("shape"),
                  py::arg("dtype") = ScalarType::Float32,
                  "Create an uninitialized Tensor with given shape and dtype")

      .def_static("ones", &Tensor::ones, py::arg("shape"),
                  py::arg("dtype") = ScalarType::Float32,
                  "Create a ones-initialized Tensor with given shape and dtype")
      .def_static("randn", &Tensor::randn, py::arg("shape"),
                  py::arg("dtype") = ScalarType::Float32,
                  "Create a Tensor filled with random numbers from a standard normal distribution")
      // 核心属性
      .def_property_readonly("shape", &Tensor::shape, "List of dimension sizes")
      .def_property_readonly("strides", &Tensor::strides,
                             "List of strides for each dimension")
      .def_property_readonly("ndim", &Tensor::ndim,
                             "Number of dimensions (rank)")
      .def_property_readonly(
          "is_contiguous", &Tensor::is_contiguous,
          "True if the memory is contiguous in standard C-order")
      .def("__len__", &Tensor::numel, "Total number of elements in the tensor")
      .def("__repr__", &Tensor::to_string,
           "String representation of the tensor")

      // 视图变换算子 (零拷贝 Zero-copy)
      .def("transpose", &Tensor::transpose, py::arg("dim0"), py::arg("dim1"),
           "Return a new tensor with dimensions dim0 and dim1 swapped "
           "(zero-copy)")
      .def("t", &Tensor::t, "Short-hand 2D matrix transpose (zero-copy)")
      .def("view", &Tensor::view, py::arg("shape"),
           "Return a new tensor with the specified shape (zero-copy, requires "
           "contiguous memory)")
      .def("contiguous", &Tensor::contiguous,
           "Return a contiguous in memory copy of tensor if not contiguous; "
           "self otherwise")

      // Buffer protocol 支持 (支持 memoryview / numpy.asarray 等)
      .def_buffer([](Tensor &t) -> py::buffer_info
                  {
        std::vector<py::ssize_t> shape(t.shape().begin(), t.shape().end());
        std::vector<py::ssize_t> strides;
        for (auto s : t.strides()) {
          strides.push_back(s *
                            static_cast<py::ssize_t>(element_size(t.dtype())));
        }
        std::string format;
        size_t itemsize = element_size(t.dtype());
        void *ptr = nullptr;
        switch (t.dtype()) {
        case ScalarType::Float32:
          format = py::format_descriptor<float>::format();
          ptr = t.data_ptr<float>();
          break;
        case ScalarType::Float64:
          format = py::format_descriptor<double>::format();
          ptr = t.data_ptr<double>();
          break;
        case ScalarType::Int32:
          format = py::format_descriptor<int32_t>::format();
          ptr = t.data_ptr<int32_t>();
          break;
        case ScalarType::Int64:
          format = py::format_descriptor<int64_t>::format();
          ptr = t.data_ptr<int64_t>();
          break;
        case ScalarType::UInt8:
          format = py::format_descriptor<uint8_t>::format();
          ptr = t.data_ptr<uint8_t>();
          break;
        default:
          throw std::runtime_error("Unsupported dtype for buffer protocol");
        }
        return py::buffer_info(ptr, itemsize, format,
                               static_cast<py::ssize_t>(t.ndim()), shape,
                               strides); })

      // 导出数据为 Python 列表
      .def(
          "tolist",
          [](const Tensor &t) -> py::object
          {
            Tensor c = t.is_contiguous() ? t : t.contiguous();
            switch (c.dtype())
            {
            case ScalarType::Float32:
            {
              const float *ptr = c.data_ptr<float>();
              return py::cast(std::vector<float>(ptr, ptr + c.numel()));
            }
            case ScalarType::Float64:
            {
              const double *ptr = c.data_ptr<double>();
              return py::cast(std::vector<double>(ptr, ptr + c.numel()));
            }
            case ScalarType::Int32:
            {
              const int32_t *ptr = c.data_ptr<int32_t>();
              return py::cast(std::vector<int32_t>(ptr, ptr + c.numel()));
            }
            case ScalarType::Int64:
            {
              const int64_t *ptr = c.data_ptr<int64_t>();
              return py::cast(std::vector<int64_t>(ptr, ptr + c.numel()));
            }
            case ScalarType::UInt8:
            {
              const uint8_t *ptr = c.data_ptr<uint8_t>();
              return py::cast(std::vector<uint8_t>(ptr, ptr + c.numel()));
            }
            default:
              throw std::runtime_error("Unsupported dtype for tolist");
            }
          },
          "Return the tensor data as a flat Python list");
}
