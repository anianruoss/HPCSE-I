cmake_minimum_required(VERSION 3.10)

project(HPCSE-I)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -O3 -march=native")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})


add_subdirectory(utils)
include_directories(utils)

add_subdirectory(exercise01)
