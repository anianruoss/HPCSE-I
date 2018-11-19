cmake_minimum_required(VERSION 3.5.2)

project(HPCSE-I)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -O3 -march=native")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()
find_package(BLAS REQUIRED)
find_package(MKL REQUIRED)
include_directories(${MKL_INCLUDE_DIRS})
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_subdirectory(utils)
include_directories(utils)

add_subdirectory(exercise01)
add_subdirectory(exercise02)
add_subdirectory(exercise03)
add_subdirectory(exercise04)
add_subdirectory(exercise05)
add_subdirectory(exercise06)
add_subdirectory(exercise07)
add_subdirectory(exercise08)
