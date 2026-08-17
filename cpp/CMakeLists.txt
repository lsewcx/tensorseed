cmake_minimum_required(VERSION 3.18)

project(tensorseed LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

target_link_libraries(tensorseed pybind11_all_do_not_use)