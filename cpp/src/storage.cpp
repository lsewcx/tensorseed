#include <tensorseed/storage.hpp>

namespace tensorseed {

Storage::Storage(size_t size_in_bytes) : size_bytes_(size_in_bytes) {
  if (size_in_bytes > 0) {
    data_ = std::shared_ptr<uint8_t[]>(new uint8_t[size_in_bytes]());
  }
}

Storage::Storage(void *external_ptr, size_t size_in_bytes,
                 const std::function<void(void *)> &deleter)
    : size_bytes_(size_in_bytes) {
  if (deleter) {
    data_ = std::shared_ptr<uint8_t[]>(static_cast<uint8_t *>(external_ptr),
                                       deleter);
  } else {
    data_ = std::shared_ptr<uint8_t[]>(static_cast<uint8_t *>(external_ptr),
                                       [](uint8_t *) {});
  }
}

} // namespace tensorseed
