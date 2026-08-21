#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace tensorseed {

class Storage {
public:
  explicit Storage(size_t size_in_bytes);

  Storage(void *external_ptr, size_t size_in_bytes,
          const std::function<void(void *)> &deleter = nullptr);

  void *data() const { return data_.get(); }
  size_t nbytes() const { return size_bytes_; }

private:
  std::shared_ptr<uint8_t[]> data_{nullptr};
  size_t size_bytes_{0};
};

} // namespace tensorseed
