#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <tensorseed/storage.hpp>
#include <tensorseed/types.hpp>

namespace tensorseed {

class TensorImpl {
public:
  using SizeVector = std::vector<int64_t>;
  using StrideVector = std::vector<int64_t>;

  TensorImpl(SizeVector sizes, ScalarType dtype);

  TensorImpl(std::shared_ptr<Storage> storage, int64_t storage_offset,
             SizeVector sizes, StrideVector strides, ScalarType dtype);

  static StrideVector compute_contiguous_strides(const SizeVector &sizes);

  int64_t numel() const;
  int64_t ndim() const;
  const SizeVector &sizes() const { return sizes_; }
  const StrideVector &strides() const { return strides_; }
  int64_t storage_offset() const { return storage_offset_; }
  ScalarType dtype() const { return dtype_; }
  std::shared_ptr<Storage> storage() const { return storage_; }

  template <typename T> T *data_ptr() const {
    if (!storage_)
      return nullptr;
    uint8_t *raw_ptr = static_cast<uint8_t *>(storage_->data());
    return reinterpret_cast<T *>(raw_ptr) + storage_offset_;
  }

  bool is_contiguous() const;

private:
  std::shared_ptr<Storage> storage_{nullptr};
  int64_t storage_offset_{0};
  SizeVector sizes_;
  StrideVector strides_;
  ScalarType dtype_{ScalarType::Float32};
};

} // namespace tensorseed
