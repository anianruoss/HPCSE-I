include_directories(${PROJECT_SOURCE_DIR}})

add_executable(ispc_gemm ispc_gemm/gemm.cpp)
add_executable(vec_red vectorized_reduction/vectorized_reduction.cpp)
