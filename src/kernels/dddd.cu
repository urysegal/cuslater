#include "kernels.cuh"

namespace cuslater {
    __global__ void evalIntegrand_3DBloackReduce_DDDD_81(int n, real_t hxyz, real_t rlx, real_t rly,
                                                         real_t rlz, real_t r, int pitchX,
                                                         const real2_t* __restrict__ distance_pair_grid,
                                                         real_t* __restrict__ d_out /* length 81 */
    ) {
        int tid     = blockIdx.x * blockDim.x + threadIdx.x;
        int totalXY = n * n;

        real_t acc[81] = {real_t(0.0)};

        if (tid < totalXY) {
            int y = tid / n;
            int x = tid - y * n;

            int pitchXY = pitchX * n;
            int base    = y * pitchX + x;

            const real3_t c1 = reinterpret_cast<const real3_t*>(d_c + 0)[0];
            const real3_t c2 = reinterpret_cast<const real3_t*>(d_c + 3)[0];
            const real3_t c3 = reinterpret_cast<const real3_t*>(d_c + 6)[0];
            const real3_t c4 = reinterpret_cast<const real3_t*>(d_c + 9)[0];

            const real2_t alpha12 = reinterpret_cast<const real2_t*>(d_alpha + 0)[0];
            const real2_t alpha34 = reinterpret_cast<const real2_t*>(d_alpha + 2)[0];

            real_t w = hxyz;
            if (x == 0 || x == n - 1) w *= real_t(0.5);
            if (y == 0 || y == n - 1) w *= real_t(0.5);

            const real_t X = __ldg(&d_x_grid[x]);
            const real_t Y = __ldg(&d_y_grid[y]);

            // Left-side coordinates, unshifted
            const real_t x1 = X - c1.x;
            const real_t y1 = Y - c1.y;

            const real_t x2 = X - c2.x;
            const real_t y2 = Y - c2.y;

            // Right-side coordinates, shifted by r * omega
            const real_t x3 = X - c3.x + rlx;
            const real_t y3 = Y - c3.y + rly;

            const real_t x4 = X - c4.x + rlx;
            const real_t y4 = Y - c4.y + rly;

            for (int k = 0; k < n; ++k) {
                const real_t Z = __ldg(&d_z_grid[k]);

                const real2_t d12 = __ldg(distance_pair_grid + (k * pitchXY + base));
                const real_t  d1  = d12.x;
                const real_t  d2  = d12.y;
                const real_t  a12 = alpha12.x * d1 + alpha12.y * d2;

                const real_t z1 = Z - c1.z;
                const real_t z2 = Z - c2.z;
                const real_t z3 = Z - c3.z + rlz;
                const real_t z4 = Z - c4.z + rlz;

                const real_t d3 = norm(x3, y3, z3);
                const real_t d4 = norm(x4, y4, z4);

                const real_t expo = r - (a12 + alpha34.x * d3 + alpha34.y * d4);
                const real_t wz   = (k == 0 || k == n - 1) ? real_t(0.5) : real_t(1.0);
                const real_t T    = exp_fn(expo) * wz;

                // D components used:
                // 0 = xy
                // 1 = xz
                // 2 = yz

                // Center 1 D values
                const real_t D1_0 = x1 * y1;
                const real_t D1_1 = x1 * z1;
                const real_t D1_2 = y1 * z1;

                // Center 2 D values
                const real_t D2_0 = x2 * y2;
                const real_t D2_1 = x2 * z2;
                const real_t D2_2 = y2 * z2;

                // Center 3 D values
                const real_t D3_0 = x3 * y3;
                const real_t D3_1 = x3 * z3;
                const real_t D3_2 = y3 * z3;

                // Center 4 D values
                const real_t D4_0 = x4 * y4;
                const real_t D4_1 = x4 * z4;
                const real_t D4_2 = y4 * z4;

                // Left DD pair vector, size 9:
                // L index = 3*a + b, where a is center 1 D component, b is center 2 D component.
                const real_t L0 = D1_0 * D2_0;
                const real_t L1 = D1_0 * D2_1;
                const real_t L2 = D1_0 * D2_2;

                const real_t L3 = D1_1 * D2_0;
                const real_t L4 = D1_1 * D2_1;
                const real_t L5 = D1_1 * D2_2;

                const real_t L6 = D1_2 * D2_0;
                const real_t L7 = D1_2 * D2_1;
                const real_t L8 = D1_2 * D2_2;

                // Right DD pair vector, size 9:
                // R index = 3*c + d, where c is center 3 D component, d is center 4 D component.
                const real_t R0 = D3_0 * D4_0;
                const real_t R1 = D3_0 * D4_1;
                const real_t R2 = D3_0 * D4_2;

                const real_t R3 = D3_1 * D4_0;
                const real_t R4 = D3_1 * D4_1;
                const real_t R5 = D3_1 * D4_2;

                const real_t R6 = D3_2 * D4_0;
                const real_t R7 = D3_2 * D4_1;
                const real_t R8 = D3_2 * D4_2;

                const real_t TL0 = T * L0;
                const real_t TL1 = T * L1;
                const real_t TL2 = T * L2;
                const real_t TL3 = T * L3;
                const real_t TL4 = T * L4;
                const real_t TL5 = T * L5;
                const real_t TL6 = T * L6;
                const real_t TL7 = T * L7;
                const real_t TL8 = T * L8;

                // 9 x 9 outer product: acc[9*li + ri] += T * L[li] * R[ri]

                acc[0] += TL0 * R0;
                acc[1] += TL0 * R1;
                acc[2] += TL0 * R2;
                acc[3] += TL0 * R3;
                acc[4] += TL0 * R4;
                acc[5] += TL0 * R5;
                acc[6] += TL0 * R6;
                acc[7] += TL0 * R7;
                acc[8] += TL0 * R8;

                acc[9] += TL1 * R0;
                acc[10] += TL1 * R1;
                acc[11] += TL1 * R2;
                acc[12] += TL1 * R3;
                acc[13] += TL1 * R4;
                acc[14] += TL1 * R5;
                acc[15] += TL1 * R6;
                acc[16] += TL1 * R7;
                acc[17] += TL1 * R8;

                acc[18] += TL2 * R0;
                acc[19] += TL2 * R1;
                acc[20] += TL2 * R2;
                acc[21] += TL2 * R3;
                acc[22] += TL2 * R4;
                acc[23] += TL2 * R5;
                acc[24] += TL2 * R6;
                acc[25] += TL2 * R7;
                acc[26] += TL2 * R8;

                acc[27] += TL3 * R0;
                acc[28] += TL3 * R1;
                acc[29] += TL3 * R2;
                acc[30] += TL3 * R3;
                acc[31] += TL3 * R4;
                acc[32] += TL3 * R5;
                acc[33] += TL3 * R6;
                acc[34] += TL3 * R7;
                acc[35] += TL3 * R8;

                acc[36] += TL4 * R0;
                acc[37] += TL4 * R1;
                acc[38] += TL4 * R2;
                acc[39] += TL4 * R3;
                acc[40] += TL4 * R4;
                acc[41] += TL4 * R5;
                acc[42] += TL4 * R6;
                acc[43] += TL4 * R7;
                acc[44] += TL4 * R8;

                acc[45] += TL5 * R0;
                acc[46] += TL5 * R1;
                acc[47] += TL5 * R2;
                acc[48] += TL5 * R3;
                acc[49] += TL5 * R4;
                acc[50] += TL5 * R5;
                acc[51] += TL5 * R6;
                acc[52] += TL5 * R7;
                acc[53] += TL5 * R8;

                acc[54] += TL6 * R0;
                acc[55] += TL6 * R1;
                acc[56] += TL6 * R2;
                acc[57] += TL6 * R3;
                acc[58] += TL6 * R4;
                acc[59] += TL6 * R5;
                acc[60] += TL6 * R6;
                acc[61] += TL6 * R7;
                acc[62] += TL6 * R8;

                acc[63] += TL7 * R0;
                acc[64] += TL7 * R1;
                acc[65] += TL7 * R2;
                acc[66] += TL7 * R3;
                acc[67] += TL7 * R4;
                acc[68] += TL7 * R5;
                acc[69] += TL7 * R6;
                acc[70] += TL7 * R7;
                acc[71] += TL7 * R8;

                acc[72] += TL8 * R0;
                acc[73] += TL8 * R1;
                acc[74] += TL8 * R2;
                acc[75] += TL8 * R3;
                acc[76] += TL8 * R4;
                acc[77] += TL8 * R5;
                acc[78] += TL8 * R6;
                acc[79] += TL8 * R7;
                acc[80] += TL8 * R8;
            }

#pragma unroll
            for (int i = 0; i < 81; ++i) {
                acc[i] *= w;
            }
        }

        using BlockReduce = cub::BlockReduce<real_t, THREADS_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;

        real_t block_sum[81];

#pragma unroll
        for (int i = 0; i < 81; ++i) {
            block_sum[i] = BlockReduce(temp).Sum(acc[i]);
            if (i != 80) __syncthreads();
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int i = 0; i < 81; ++i) {
                atomicAdd(d_out + i, block_sum[i]);
            }
        }
    }


} // namespace cuslater