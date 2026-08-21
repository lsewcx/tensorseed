#include <numeric>
#include <utility>
#include <tensorseed/tensor_impl.hpp>

namespace tensorseed {

TensorImpl::TensorImpl(SizeVector sizes, ScalarType dtype)
    : sizes_(std::move(sizes)), dtype_(dtype), storage_offset_(0) {
  strides_ = compute_contiguous_strides(sizes_);
  size_t total_elements = numel();
  storage_ = std::make_shared<Storage>(total_elements * element_size(dtype_));
}

TensorImpl::TensorImpl(std::shared_ptr<Storage> storage, int64_t storage_offset,
                       SizeVector sizes, StrideVector strides, ScalarType dtype)
    : storage_(std::move(storage)), storage_offset_(storage_offset),
      sizes_(std::move(sizes)), strides_(std::move(strides)), dtype_(dtype) {}

TensorImpl::StrideVector
TensorImpl::compute_contiguous_strides(const SizeVector &sizes) {
  StrideVector strides(sizes.size());
  if (sizes.empty())
    return strides;
  int64_t current_stride = 1;
  for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
    strides[i] = current_stride;
    current_stride *= sizes[i];
  }
  return strides;
}

int64_t TensorImpl::numel() const {
  return std::accumulate(sizes_.begin(), sizes_.end(),
                         static_cast<int64_t>(1), std::multiplies<int64_t>());
}

int64_t TensorImpl::ndim() const {
  return static_cast<int64_t>(sizes_.size());
}

bool TensorImpl::is_contiguous() const {
  StrideVector expected = compute_contiguous_strides(sizes_);
  return strides_ == expected;
}

} // namespace tensorseed
