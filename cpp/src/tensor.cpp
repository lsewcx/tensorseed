#include <algorithm>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <tensorseed/tensor.hpp>

namespace tensorseed {

Tensor Tensor::empty(const std::vector<int64_t> &shape, ScalarType dtype) {
  return Tensor(std::make_shared<TensorImpl>(shape, dtype));
}

Tensor Tensor::zeros(const std::vector<int64_t> &shape, ScalarType dtype) {
  Tensor t = Tensor::empty(shape, dtype);
  if (t.numel() > 0 && t.data_ptr<uint8_t>()) {
    std::memset(t.data_ptr<uint8_t>(), 0, t.numel() * element_size(dtype));
  }
  return t;
}

Tensor Tensor::ones(const std::vector<int64_t> &shape, ScalarType dtype) {
  Tensor t = Tensor::empty(shape, dtype);
  int64_t total = t.numel();
  if (total == 0)
    return t;

  switch (dtype) {
  case ScalarType::Float32:
    std::fill_n(t.data_ptr<float>(), total, 1.0f);
    break;
  case ScalarType::Float64:
    std::fill_n(t.data_ptr<double>(), total, 1.0);
    break;
  case ScalarType::Int32:
    std::fill_n(t.data_ptr<int32_t>(), total, 1);
    break;
  case ScalarType::Int64:
    std::fill_n(t.data_ptr<int64_t>(), total, 1LL);
    break;
  case ScalarType::UInt8:
    std::fill_n(t.data_ptr<uint8_t>(), total, static_cast<uint8_t>(1));
    break;
  default:
    throw std::invalid_argument("Unsupported dtype in ones()");
  }
  return t;
}

Tensor Tensor::randn(const std::vector<int64_t> &shape, ScalarType dtype) {
  if (dtype != ScalarType::Float32 && dtype != ScalarType::Float64) {
    throw std::invalid_argument(
        "randn only supports floating-point dtypes (float32, float64)");
  }

  Tensor t = Tensor::empty(shape, dtype);
  int64_t total = t.numel();
  if (total == 0)
    return t;

  thread_local std::mt19937 gen(std::random_device{}());

  if (dtype == ScalarType::Float32) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    float *ptr = t.data_ptr<float>();
    for (int64_t i = 0; i < total; ++i) {
      ptr[i] = dist(gen);
    }
  } else {
    std::normal_distribution<double> dist(0.0, 1.0);
    double *ptr = t.data_ptr<double>();
    for (int64_t i = 0; i < total; ++i) {
      ptr[i] = dist(gen);
    }
  }

  return t;
}

const std::vector<int64_t> &Tensor::shape() const { return impl_->sizes(); }
const std::vector<int64_t> &Tensor::strides() const { return impl_->strides(); }
int64_t Tensor::ndim() const { return impl_->ndim(); }
int64_t Tensor::numel() const { return impl_->numel(); }
int64_t Tensor::storage_offset() const { return impl_->storage_offset(); }
ScalarType Tensor::dtype() const { return impl_->dtype(); }
bool Tensor::is_contiguous() const { return impl_->is_contiguous(); }

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
  int64_t rank = ndim();
  if (dim0 < 0)
    dim0 += rank;
  if (dim1 < 0)
    dim1 += rank;
  if (dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank) {
    throw std::out_of_range("Dimension out of range for transpose");
  }
  auto new_sizes = impl_->sizes();
  auto new_strides = impl_->strides();
  std::swap(new_sizes[dim0], new_sizes[dim1]);
  std::swap(new_strides[dim0], new_strides[dim1]);
  auto new_impl =
      std::make_shared<TensorImpl>(impl_->storage(), impl_->storage_offset(),
                                   new_sizes, new_strides, impl_->dtype());
  return Tensor(new_impl);
}

Tensor Tensor::t() const {
  if (ndim() != 2) {
    throw std::runtime_error("t() expects a 2D tensor");
  }
  return transpose(0, 1);
}

Tensor Tensor::view(const std::vector<int64_t> &new_shape) const {
  if (!is_contiguous()) {
    throw std::runtime_error(
        "view size is not compatible with non-contiguous tensor, use "
        ".contiguous() first");
  }
  int64_t total_elements = numel();
  int64_t infer_idx = -1;
  int64_t product = 1;
  std::vector<int64_t> resolved_shape = new_shape;
  for (size_t i = 0; i < new_shape.size(); ++i) {
    if (new_shape[i] == -1) {
      if (infer_idx != -1)
        throw std::invalid_argument("Only one dimension can be -1 in view");
      infer_idx = static_cast<int64_t>(i);
    } else {
      product *= new_shape[i];
    }
  }
  if (infer_idx != -1) {
    if (product == 0 || total_elements % product != 0) {
      throw std::invalid_argument("Shape mismatch for inferring dimension -1");
    }
    resolved_shape[infer_idx] = total_elements / product;
  } else {
    if (product != total_elements) {
      throw std::invalid_argument(
          "Shape mismatch in view: total elements do not match");
    }
  }
  auto new_strides = TensorImpl::compute_contiguous_strides(resolved_shape);
  auto new_impl = std::make_shared<TensorImpl>(
      impl_->storage(), impl_->storage_offset(), resolved_shape, new_strides,
      impl_->dtype());
  return Tensor(new_impl);
}

Tensor Tensor::contiguous() const {
  if (is_contiguous()) {
    return *this;
  }
  Tensor result = Tensor::empty(shape(), dtype());
  copy_to_contiguous(result);
  return result;
}

Tensor Tensor::select(int64_t dim, int64_t index) const {
  int64_t rank = ndim();
  if (dim < 0)
    dim += rank;
  if (dim < 0 || dim >= rank) {
    throw std::out_of_range("Dimension out of range in select()");
  }

  int64_t dim_size = shape()[dim];
  if (index < 0)
    index += dim_size;
  if (index < 0 || index >= dim_size) {
    throw std::out_of_range("Index out of range in select()");
  }

  int64_t new_offset = storage_offset() + index * strides()[dim];
  std::vector<int64_t> new_shape, new_strides;
  for (int64_t d = 0; d < rank; ++d) {
    if (d != dim) {
      new_shape.push_back(shape()[d]);
      new_strides.push_back(strides()[d]);
    }
  }

  auto new_impl = std::make_shared<TensorImpl>(
      impl_->storage(), new_offset, new_shape, new_strides, dtype());
  return Tensor(new_impl);
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
  int64_t rank = ndim();
  if (dim < 0)
    dim += rank;
  if (dim < 0 || dim >= rank) {
    throw std::out_of_range("Dimension out of range in slice()");
  }
  if (step <= 0) {
    throw std::invalid_argument("slice step must be positive");
  }

  int64_t dim_size = shape()[dim];
  if (start < 0)
    start += dim_size;
  if (end < 0)
    end += dim_size;
  start = std::max<int64_t>(0, std::min(start, dim_size));
  end = std::max<int64_t>(0, std::min(end, dim_size));

  int64_t slice_len = (end > start) ? ((end - start + step - 1) / step) : 0;

  int64_t new_offset = storage_offset() + start * strides()[dim];
  auto new_shape = shape();
  auto new_strides = strides();
  new_shape[dim] = slice_len;
  new_strides[dim] = strides()[dim] * step;

  auto new_impl = std::make_shared<TensorImpl>(
      impl_->storage(), new_offset, new_shape, new_strides, dtype());
  return Tensor(new_impl);
}

double Tensor::item() const {
  if (numel() != 1) {
    throw std::runtime_error(
        "item() only supports tensors with exactly 1 element, but got " +
        std::to_string(numel()) + " elements");
  }
  switch (dtype()) {
  case ScalarType::Float32:
    return static_cast<double>(*data_ptr<float>());
  case ScalarType::Float64:
    return *data_ptr<double>();
  case ScalarType::Int32:
    return static_cast<double>(*data_ptr<int32_t>());
  case ScalarType::Int64:
    return static_cast<double>(*data_ptr<int64_t>());
  case ScalarType::UInt8:
    return static_cast<double>(*data_ptr<uint8_t>());
  default:
    throw std::runtime_error("Unsupported dtype in item()");
  }
}

std::string Tensor::to_string() const {
  std::ostringstream oss;
  oss << "Tensor(shape=[";
  for (size_t i = 0; i < shape().size(); ++i) {
    oss << shape()[i] << (i + 1 < shape().size() ? ", " : "");
  }
  oss << "], dtype=" << dtype_to_string(dtype())
      << ", contiguous=" << (is_contiguous() ? "True" : "False") << ")";
  return oss.str();
}

template <typename T>
void Tensor::copy_to_contiguous_impl(Tensor &dst) const {
  int64_t total = numel();
  int64_t rank = ndim();
  const auto &s = shape();
  const auto &st = strides();

  T *dst_ptr = dst.data_ptr<T>();
  const T *src_base = impl_->data_ptr<T>();
  std::vector<int64_t> coords(rank, 0);
  for (int64_t i = 0; i < total; ++i) {
    int64_t src_offset = 0;
    for (int64_t d = 0; d < rank; ++d) {
      src_offset += coords[d] * st[d];
    }
    dst_ptr[i] = src_base[src_offset];
    for (int64_t d = rank - 1; d >= 0; --d) {
      coords[d]++;
      if (coords[d] < s[d] || d == 0)
        break;
      coords[d] = 0;
    }
  }
}

void Tensor::copy_to_contiguous(Tensor &dst) const {
  switch (dtype()) {
  case ScalarType::Float32:
    copy_to_contiguous_impl<float>(dst);
    break;
  case ScalarType::Float64:
    copy_to_contiguous_impl<double>(dst);
    break;
  case ScalarType::Int32:
    copy_to_contiguous_impl<int32_t>(dst);
    break;
  case ScalarType::Int64:
    copy_to_contiguous_impl<int64_t>(dst);
    break;
  case ScalarType::UInt8:
    copy_to_contiguous_impl<uint8_t>(dst);
    break;
  default:
    throw std::runtime_error("Unsupported dtype in copy_to_contiguous");
  }
}

} // namespace tensorseed
