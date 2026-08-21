#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <tensorseed/storage.hpp>
#include <tensorseed/tensor_impl.hpp>
#include <tensorseed/types.hpp>

namespace tensorseed {

class Tensor {
public:
  Tensor() = default;
  explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

  // 基础工厂方法
  static Tensor empty(const std::vector<int64_t> &shape,
                      ScalarType dtype = ScalarType::Float32);

  static Tensor zeros(const std::vector<int64_t> &shape,
                      ScalarType dtype = ScalarType::Float32);

  static Tensor ones(const std::vector<int64_t> &shape,
                     ScalarType dtype = ScalarType::Float32);

  static Tensor randn(const std::vector<int64_t> &shape,
                      ScalarType dtype = ScalarType::Float32);

  // 从 std::vector 构造 1D Tensor (深拷贝数据)
  template <typename T> static Tensor from_vector(const std::vector<T> &data) {
    ScalarType dtype = TypeToScalarType<T>::value;
    Tensor t = Tensor::empty({static_cast<int64_t>(data.size())}, dtype);
    std::copy(data.begin(), data.end(), t.data_ptr<T>());
    return t;
  }

  // 从已有内存构造 Tensor (零拷贝外部引用)
  template <typename T>
  static Tensor
  from_blob(T *data, const std::vector<int64_t> &shape,
            const std::function<void(void *)> &deleter = nullptr) {
    ScalarType dtype = TypeToScalarType<T>::value;
    size_t total_elements = 1;
    for (auto s : shape)
      total_elements *= s;
    auto storage =
        std::make_shared<Storage>(data, total_elements * sizeof(T), deleter);
    auto strides = TensorImpl::compute_contiguous_strides(shape);
    auto impl = std::make_shared<TensorImpl>(storage, 0, shape, strides, dtype);
    return Tensor(impl);
  }

  // 基础属性
  const std::vector<int64_t> &shape() const;
  const std::vector<int64_t> &strides() const;
  int64_t ndim() const;
  int64_t numel() const;
  int64_t storage_offset() const;
  ScalarType dtype() const;
  bool is_contiguous() const;

  template <typename T> T *data_ptr() const {
    return impl_ ? impl_->data_ptr<T>() : nullptr;
  }

  // 视图操作 (全部零拷贝)
  Tensor transpose(int64_t dim0, int64_t dim1) const;
  Tensor t() const;
  Tensor view(const std::vector<int64_t> &new_shape) const;
  Tensor contiguous() const;

  std::string to_string() const;

private:
  template <typename T> void copy_to_contiguous_impl(Tensor &dst) const;
  void copy_to_contiguous(Tensor &dst) const;

  std::shared_ptr<TensorImpl> impl_{nullptr};
};

} // namespace tensorseed
