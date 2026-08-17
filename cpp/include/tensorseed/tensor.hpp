#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tensorseed {
// ==========================================
// 1. 数据类型 (DType) 定义
// ==========================================
enum class ScalarType : int8_t {
  Float32 = 0,
  Float64 = 1,
  Int32 = 2,
  Int64 = 3,
  UInt8 = 4,
  Undefined = -1
};

inline size_t element_size(ScalarType dtype) {
  switch (dtype) {
  case ScalarType::Float32:
    return sizeof(float);
  case ScalarType::Float64:
    return sizeof(double);
  case ScalarType::Int32:
    return sizeof(int32_t);
  case ScalarType::Int64:
    return sizeof(int64_t);
  case ScalarType::UInt8:
    return sizeof(uint8_t);
  default:
    throw std::invalid_argument("Unsupported ScalarType");
  }
}
inline std::string dtype_to_string(ScalarType dtype) {
  switch (dtype) {
  case ScalarType::Float32:
    return "float32";
  case ScalarType::Float64:
    return "float64";
  case ScalarType::Int32:
    return "int32";
  case ScalarType::Int64:
    return "int64";
  case ScalarType::UInt8:
    return "uint8";
  default:
    return "undefined";
  }
}
// 类型萃取模板
template <typename T> struct TypeToScalarType;
template <> struct TypeToScalarType<float> {
  static constexpr ScalarType value = ScalarType::Float32;
};
template <> struct TypeToScalarType<double> {
  static constexpr ScalarType value = ScalarType::Float64;
};
template <> struct TypeToScalarType<int32_t> {
  static constexpr ScalarType value = ScalarType::Int32;
};
template <> struct TypeToScalarType<int64_t> {
  static constexpr ScalarType value = ScalarType::Int64;
};
template <> struct TypeToScalarType<uint8_t> {
  static constexpr ScalarType value = ScalarType::UInt8;
};

class Tensor final {
public:
  /**
   * 使用一维浮点数据创建Tensor。
   *
   * Args:
   *     data: 需要存储的浮点数据。
   *     explicit 禁止隐式转换数据类型
   */
  explicit Tensor(std::vector<float> data) : data_(std::move(data)) {}

  /**
   * 获取Tensor中的数据。
   *
   * Returns:
   *     Tensor数据的只读引用。
   */
  const std::vector<float> &data() const { return data_; }

  /**
   * 获取Tensor中的元素数量。
   *
   * Returns:
   *     Tensor中的元素数量。
   */
  std::size_t size() const { return data_.size(); }

private:
  std::vector<float> data_;
};

}; // namespace tensorseed