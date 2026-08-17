cmake_minimum_required(VERSION 3.18)

project(tensorseed LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(
    _core
    binding/binding.cpp
)