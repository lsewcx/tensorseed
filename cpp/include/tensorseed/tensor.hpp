#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tensorseed
{
  // ==========================================
  // 1. 数据类型 (DType) 定义
  // ==========================================
  enum class ScalarType : int8_t
  {
    Float32 = 0,
    Float64 = 1,
    Int32 = 2,
    Int64 = 3,
    UInt8 = 4,
    Undefined = -1
  };

  inline size_t element_size(ScalarType dtype)
  {
    switch (dtype)
    {
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
  inline std::string dtype_to_string(ScalarType dtype)
  {
    switch (dtype)
    {
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
  // 编译器自动转换数据类型 传入不同的数据类型会自动匹配
  template <typename T>
  struct TypeToScalarType;
  template <>
  struct TypeToScalarType<float>
  {
    static constexpr ScalarType value = ScalarType::Float32;
  };
  template <>
  struct TypeToScalarType<double>
  {
    static constexpr ScalarType value = ScalarType::Float64;
  };
  template <>
  struct TypeToScalarType<int32_t>
  {
    static constexpr ScalarType value = ScalarType::Int32;
  };
  template <>
  struct TypeToScalarType<int64_t>
  {
    static constexpr ScalarType value = ScalarType::Int64;
  };
  template <>
  struct TypeToScalarType<uint8_t>
  {
    static constexpr ScalarType value = ScalarType::UInt8;
  };

  // ==========================================
  // 2. Storage：管理原始字节内存
  // ==========================================
  class Storage
  {
  public:
    // explicit：
    // 防止编译器把一个 size_t/int 自动隐式转换成 Storage。
    //
    // 例如：
    // Storage s = 1024;   // explicit 存在时不允许
    // Storage s(1024);    // 正确
    //
    // size_in_bytes 表示需要申请多少“字节”的内存。
    explicit Storage(size_t size_in_bytes) : size_bytes_(size_in_bytes)
    {
      if (size_in_bytes > 0)
      {
        // 使用标准对齐分配内存 (可替换为 CUDA 显存分配器)
        data_ = std::shared_ptr<uint8_t[]>(new uint8_t[size_in_bytes]());
      }
    }
    // 支持从外部已有裸指针包装（例如从 Python buffer 或 mmap 引入）
    // Storage 自己不申请内存，而是接收“别人已经申请好的内存”。
    //
    // 典型用途：
    // 1. Python buffer
    // 2. mmap
    // 3. CUDA 内存
    // 4. malloc 得到的内存
    // 5. 第三方库提供的内存
    //
    // external_ptr:
    //     外部内存的首地址
    //
    // size_in_bytes:
    //     这块内存有多少字节
    //
    // deleter:
    //     当 Storage 不再使用这块内存时，
    //     应该如何释放它。
    Storage(void *external_ptr, size_t size_in_bytes,
            const std::function<void(void *)> &deleter = nullptr)
        : size_bytes_(size_in_bytes)
    {
      if (deleter)
      {
        data_ = std::shared_ptr<uint8_t[]>(static_cast<uint8_t *>(external_ptr),
                                           deleter);
      }
      else
      {
        // 默认无所有权 deleter (外部内存借用)
        data_ = std::shared_ptr<uint8_t[]>(static_cast<uint8_t *>(external_ptr),
                                           [](uint8_t *) {});
      }
    }
    void *data() const { return data_.get(); }
    size_t nbytes() const { return size_bytes_; }

  private:
    std::shared_ptr<uint8_t[]> data_{nullptr};
    size_t size_bytes_{0};
  };

  // ==========================================
  // 3. TensorImpl：核心元数据与跨步逻辑
  // ==========================================
  class TensorImpl
  {
  public:
    using SizeVector = std::vector<int64_t>;
    using StrideVector = std::vector<int64_t>;
    TensorImpl(SizeVector sizes, ScalarType dtype)
        : sizes_(std::move(sizes)), dtype_(dtype), storage_offset_(0)
    {
      strides_ = compute_contiguous_strides(sizes_);
      size_t total_elements = numel();
      storage_ = std::make_shared<Storage>(total_elements * element_size(dtype_));
    }
    TensorImpl(std::shared_ptr<Storage> storage, int64_t storage_offset,
               SizeVector sizes, StrideVector strides, ScalarType dtype)
        : storage_(std::move(storage)), storage_offset_(storage_offset),
          sizes_(std::move(sizes)), strides_(std::move(strides)), dtype_(dtype) {}
    // 计算连续内存下的标准 Strides (C-contiguous / Row-major)
    static StrideVector compute_contiguous_strides(const SizeVector &sizes)
    {
      StrideVector strides(sizes.size());
      if (sizes.empty())
        return strides;
      int64_t current_stride = 1;
      for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i)
      {
        strides[i] = current_stride;
        current_stride *= sizes[i];
      }
      return strides;
    }
    // 元素总数
    int64_t numel() const
    {
      if (sizes_.empty())
        return 0;
      return std::accumulate(sizes_.begin(), sizes_.end(),
                             static_cast<int64_t>(1), std::multiplies<int64_t>());
    }
    // 维度数量 (Rank)
    int64_t ndim() const { return static_cast<int64_t>(sizes_.size()); }
    const SizeVector &sizes() const { return sizes_; }
    const StrideVector &strides() const { return strides_; }
    int64_t storage_offset() const { return storage_offset_; }
    ScalarType dtype() const { return dtype_; }
    std::shared_ptr<Storage> storage() const { return storage_; }
    // 获取特定类型的首地址指针（考虑 storage_offset）
    template <typename T>
    T *data_ptr() const
    {
      if (!storage_)
        return nullptr;
      uint8_t *raw_ptr = static_cast<uint8_t *>(storage_->data());
      return reinterpret_cast<T *>(raw_ptr) + storage_offset_;
    }
    // 检查是否连续存储
    bool is_contiguous() const
    {
      StrideVector expected = compute_contiguous_strides(sizes_);
      return strides_ == expected;
    }

  private:
    std::shared_ptr<Storage> storage_{nullptr};
    int64_t storage_offset_{0};
    SizeVector sizes_;
    StrideVector strides_;
    ScalarType dtype_{ScalarType::Float32};
  };

  // ==========================================
  // 4. Tensor：上层轻量句柄与算子操作
  // ==========================================
  class Tensor
  {
  public:
    Tensor() = default;
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}
    // 构造空 Tensor
    static Tensor empty(const std::vector<int64_t> &shape,
                        ScalarType dtype = ScalarType::Float32)
    {
      return Tensor(std::make_shared<TensorImpl>(shape, dtype));
    }

    // 构造全 0 Tensor
    static Tensor zeros(const std::vector<int64_t> &shape,
                        ScalarType dtype = ScalarType::Float32)
    {
      Tensor t = Tensor::empty(shape, dtype);
      if (t.numel() > 0 && t.data_ptr<uint8_t>())
      {
        std::memset(t.data_ptr<uint8_t>(), 0, t.numel() * element_size(dtype));
      }
      return t;
    }

    // 构造全 1 Tensor
    static Tensor ones(const std::vector<int64_t> &shape,
                       ScalarType dtype = ScalarType::Float32)
    {
      Tensor t = Tensor::empty(shape, dtype);
      int64_t total = t.numel();
      if (total == 0)
        return t;

      switch (dtype)
      {
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

    // 从 std::vector 构造 1D Tensor (深拷贝数据)
    template <typename T>
    static Tensor from_vector(const std::vector<T> &data)
    {
      ScalarType dtype = TypeToScalarType<T>::value;
      Tensor t = Tensor::empty({static_cast<int64_t>(data.size())}, dtype);
      std::copy(data.begin(), data.end(), t.data_ptr<T>());
      return t;
    }
    // 从已有内存构造 Tensor (零拷贝外部引用)
    template <typename T>
    static Tensor
    from_blob(T *data, const std::vector<int64_t> &shape,
              const std::function<void(void *)> &deleter = nullptr)
    {
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
    const std::vector<int64_t> &shape() const { return impl_->sizes(); }
    const std::vector<int64_t> &strides() const { return impl_->strides(); }
    int64_t ndim() const { return impl_->ndim(); }
    int64_t numel() const { return impl_->numel(); }
    int64_t storage_offset() const { return impl_->storage_offset(); }
    ScalarType dtype() const { return impl_->dtype(); }
    bool is_contiguous() const { return impl_->is_contiguous(); }
    template <typename T>
    T *data_ptr() const { return impl_->data_ptr<T>(); }
    // -------------------------------------------------------------
    // 视图操作 (View Operations) - 全部零拷贝 (Zero-Copy)
    // -------------------------------------------------------------
    // 转置两个维度 (零拷贝修改 stride)
    Tensor transpose(int64_t dim0, int64_t dim1) const
    {
      int64_t rank = ndim();
      if (dim0 < 0)
        dim0 += rank;
      if (dim1 < 0)
        dim1 += rank;
      if (dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank)
      {
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
    // 2D 矩阵快捷转置
    Tensor t() const
    {
      if (ndim() != 2)
      {
        throw std::runtime_error("t() expects a 2D tensor");
      }
      return transpose(0, 1);
    }
    // View 变换形状 (要求 Tensor 必须是连续的)
    Tensor view(const std::vector<int64_t> &new_shape) const
    {
      if (!is_contiguous())
      {
        throw std::runtime_error(
            "view size is not compatible with non-contiguous tensor, use "
            ".contiguous() first");
      }
      // 支持推导 -1 维度
      int64_t total_elements = numel();
      int64_t infer_idx = -1;
      int64_t product = 1;
      std::vector<int64_t> resolved_shape = new_shape;
      for (size_t i = 0; i < new_shape.size(); ++i)
      {
        if (new_shape[i] == -1)
        {
          if (infer_idx != -1)
            throw std::invalid_argument("Only one dimension can be -1 in view");
          infer_idx = static_cast<int64_t>(i);
        }
        else
        {
          product *= new_shape[i];
        }
      }
      if (infer_idx != -1)
      {
        if (product == 0 || total_elements % product != 0)
        {
          throw std::invalid_argument(
              "Shape mismatch for inferring dimension -1");
        }
        resolved_shape[infer_idx] = total_elements / product;
      }
      else
      {
        if (product != total_elements)
        {
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
    // 返回连续内存的副本（如果已连续则直接返回自身）
    Tensor contiguous() const
    {
      if (is_contiguous())
      {
        return *this;
      }
      Tensor result = Tensor::empty(shape(), dtype());
      // 将非连续数据按 strides 线性拷贝到连续内存
      copy_to_contiguous(result);
      return result;
    }
    // 打印信息 (类似 PyTorch repr)
    std::string to_string() const
    {
      std::ostringstream oss;
      oss << "Tensor(shape=[";
      for (size_t i = 0; i < shape().size(); ++i)
      {
        oss << shape()[i] << (i + 1 < shape().size() ? ", " : "");
      }
      oss << "], dtype=" << dtype_to_string(dtype())
          << ", contiguous=" << (is_contiguous() ? "True" : "False") << ")";
      return oss.str();
    }

  private:
    template <typename T>
    void copy_to_contiguous_impl(Tensor &dst) const
    {
      int64_t total = numel();
      int64_t rank = ndim();
      const auto &s = shape();
      const auto &st = strides();

      T *dst_ptr = dst.data_ptr<T>();
      const T *src_base = impl_->data_ptr<T>();
      std::vector<int64_t> coords(rank, 0);
      for (int64_t i = 0; i < total; ++i)
      {
        // 计算当前坐标对应非连续源数据的物理偏移量
        int64_t src_offset = 0;
        for (int64_t d = 0; d < rank; ++d)
        {
          src_offset += coords[d] * st[d];
        }
        dst_ptr[i] = src_base[src_offset];
        // 递增高维坐标 (Row-major 递增)
        for (int64_t d = rank - 1; d >= 0; --d)
        {
          coords[d]++;
          if (coords[d] < s[d] || d == 0)
            break;
          coords[d] = 0;
        }
      }
    }

    void copy_to_contiguous(Tensor &dst) const
    {
      switch (dtype())
      {
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
    std::shared_ptr<TensorImpl> impl_{nullptr};
  };

} // namespace tensorseed