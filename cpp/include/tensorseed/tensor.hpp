#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace tensorseed{

    class Tensor final {
        public:
            /**
             * 使用一维浮点数据创建Tensor。
             *
             * Args:
             *     data: 需要存储的浮点数据。
             *     explicit 禁止隐式转换数据类型
             */
            explicit Tensor(std::vector<float> data)
                : data_(std::move(data)) {}
        
            /**
             * 获取Tensor中的数据。
             *
             * Returns:
             *     Tensor数据的只读引用。
             */
            const std::vector<float>& data() const {
                return data_;
            }
        
            /**
             * 获取Tensor中的元素数量。
             *
             * Returns:
             *     Tensor中的元素数量。
             */
            std::size_t size() const {
                return data_.size();
            }
        
        private:
            std::vector<float> data_;
        };
        
};