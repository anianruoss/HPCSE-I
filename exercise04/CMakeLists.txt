include_directories(${PROJECT_SOURCE_DIR}})


add_custom_command(
        OUTPUT gemm_sse2.o
        COMMAND ispc --target=sse2 --arch=x86-64 ispc_gemm/gemm.ispc -o
        gemm_sse2.o
        DEPENDS ispc_gemm/gemm.ispc
)
add_custom_command(
        OUTPUT gemm_avx2.o
        COMMAND ispc --target=avx2 --arch=x86-64 ispc_gemm/gemm.ispc -o
        gemm_avx2.o
        DEPENDS ispc_gemm/gemm.ispc
)
add_executable(gemm ispc_gemm/gemm.cpp gemm_sse2.o gemm_avx2.o)
add_executable(vec_red vectorized_reduction/vectorized_reduction.cpp)
