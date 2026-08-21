cmake_minimum_required(VERSION 3.18)

project(tensorseed LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 1. 核心 C++ 静态库
add_library(
    tensorseed STATIC
    src/types.cpp
    src/storage.cpp
    src/tensor_impl.cpp
    src/tensor.cpp
)

target_include_directories(
    tensorseed PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(
        tensorseed
        PRIVATE
        /utf-8
    )
endif()

# 2. Python pybind11 扩展模块
find_package(pybind11 REQUIRED)

pybind11_add_module(
    _core
    binding/binding.cpp
)

target_link_libraries(
    _core
    PRIVATE
    tensorseed
)

if(MSVC)
    target_compile_options(
        _core
        PRIVATE
        /utf-8
    )
endif()