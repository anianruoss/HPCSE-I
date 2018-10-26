// File       : gemm.ispc
// Created    : Wed Oct 17 2018 09:43:19 AM (+0200)
// Description: ISPC GEMM body
// Copyright 2018 ETH Zurich. All Rights Reserved.
#include "common.h"

/**
 * @brief General matrix-matrix multiplication kernel (GEMM). Computes C = AB.
 * ISPC implementation used for SSE2 and AVX2 extended ISA
 *
 * @param A Matrix dimension p x r
 * @param B Matrix dimension r x q
 * @param C Matrix dimension p x q
 * @param p Dimensional parameter
 * @param r Dimensional parameter
 * @param q Dimensional parameter
 */
export
#ifdef _ISPC_SSE2_
void gemm_sse2(
#else
void gemm_avx2(
#endif /* _ISPC_SSE2_ */
    const uniform Real * const uniform A,
    const uniform Real * const uniform B,
    uniform Real * const uniform C,
    const uniform int p,
    const uniform int r,
    const uniform int q
) {
    const uniform int tile_size = (1 << 14) / sizeof(Real);

    for (uniform int i = 0; i < p; ++i) {
        for (uniform int k = 0; k < r; ++k) {
            for (uniform int tile = 0; tile < (q / tile_size); ++tile) {
                foreach (t = 0...tile_size) {
                    C[i*q + t + (tile*tile_size)] +=
                    A[i*r + k] * B[k*q + t + (tile*tile_size)];
                }

            }
        }
    }
}
