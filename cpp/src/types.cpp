#include <stdexcept>
#include <tensorseed/types.hpp>

namespace tensorseed {

size_t element_size(ScalarType dtype) {
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

std::string dtype_to_string(ScalarType dtype) {
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

} // namespace tensorseed
