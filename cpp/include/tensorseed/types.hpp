#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace tensorseed {

// 数据类型枚举
enum class ScalarType : int8_t {
  Float32 = 0,
  Float64 = 1,
  Int32 = 2,
  Int64 = 3,
  UInt8 = 4,
  Undefined = -1
};

// 获取数据类型的字节大小
size_t element_size(ScalarType dtype);

// 获取数据类型的字符串表示
std::string dtype_to_string(ScalarType dtype);

// 类型萃取模板 (将 C++ 原生类型映射到 ScalarType)
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

} // namespace tensorseed
