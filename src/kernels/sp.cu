#include "kernels.cuh"
namespace cuslater {
    __global__ void evalIntegrand_3DBloackReduce_SMulti(int n, real_t hxyz, real_t rlx, real_t rly,
                                                        real_t rlz, real_t r, int pitchX,
                                                        const real2_t* __restrict__ distance_pair_grid,
                                                        real_t* __restrict__ d_out /* length 16 */) {

        int tid     = blockIdx.x * blockDim.x + threadIdx.x;
        int totalXY = n * n;

        real_t acc[16] = {0.0};

        if (tid < totalXY) {
            int y = tid / n;
            int x = tid - y * n;

            int pitchXY = pitchX * n;
            int base    = y * pitchX + x;

            const real3_t c3 = reinterpret_cast<const real3_t*>(d_c + 6)[0];
            const real3_t c4 = reinterpret_cast<const real3_t*>(d_c + 9)[0];

            const real2_t alpha12 = reinterpret_cast<const real2_t*>(d_alpha + 0)[0];
            const real2_t alpha34 = reinterpret_cast<const real2_t*>(d_alpha + 2)[0];

            // trapezoid weights in x/y
            real_t w = hxyz;
            if (x == 0 || x == n - 1) w *= real_t(0.5);
            if (y == 0 || y == n - 1) w *= real_t(0.5);

            const real_t X = __ldg(&d_x_grid[x]);
            const real_t Y = __ldg(&d_y_grid[y]);

            // r2 = r1 + r*omega offsets
            const real_t x3 = X - c3.x + rlx;
            const real_t y3 = Y - c3.y + rly;
            const real_t x4 = X - c4.x + rlx;
            const real_t y4 = Y - c4.y + rly;

            const real2_t* d12_grid = distance_pair_grid;

            for (int k = 0; k < n; ++k) {
                const real_t Z = __ldg(&d_z_grid[k]);

                const real2_t d12 = __ldg(d12_grid + (k * pitchXY + base));
                const real_t  d1  = d12.x;
                const real_t  d2  = d12.y;
                const real_t  a12 = alpha12.x * d1 + alpha12.y * d2;

                // distances on r2 side
                const real_t d3 = norm(x3, y3, Z - c3.z + rlz);
                const real_t d4 = norm(x4, y4, Z - c4.z + rlz);

                const real_t expo = r - (a12 + alpha34.x * d3 + alpha34.y * d4);
                const real_t wz   = (k == 0 || k == n - 1) ? real_t(0.5) : real_t(1.0);
                const real_t T    = exp_fn(expo) * wz;

                // Left factors:  (orb1,orb2) = {(1,1), (2,1), (1,2), (2,2)}
                const real_t L0 = real_t(1.0);
                const real_t L1 = d1;
                const real_t L2 = d2;
                const real_t L3 = d1 * d2;

                // Right factors: (orb3,orb4) = {(1,1), (2,1), (1,2), (2,2)}
                const real_t R0 = real_t(1.0);
                const real_t R1 = d3;
                const real_t R2 = d4;
                const real_t R3 = d3 * d4;

                // Precompute T * L_i
                const real_t TL0 = T * L0;
                const real_t TL1 = T * L1;
                const real_t TL2 = T * L2;
                const real_t TL3 = T * L3;

                // Outer-product accumulation: acc[4*li + ri] += (T*L[li]) * R[ri]
                acc[0] += TL0 * R0;
                acc[1] += TL0 * R1;
                acc[2] += TL0 * R2;
                acc[3] += TL0 * R3;

                acc[4] += TL1 * R0;
                acc[5] += TL1 * R1;
                acc[6] += TL1 * R2;
                acc[7] += TL1 * R3;

                acc[8] += TL2 * R0;
                acc[9] += TL2 * R1;
                acc[10] += TL2 * R2;
                acc[11] += TL2 * R3;

                acc[12] += TL3 * R0;
                acc[13] += TL3 * R1;
                acc[14] += TL3 * R2;
                acc[15] += TL3 * R3;
            }

            // Apply x/y weight once after z accumulation
#pragma unroll
            for (int i = 0; i < 16; ++i)
                acc[i] *= w;
        }

        using BlockReduce = cub::BlockReduce<real_t, THREADS_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;

        real_t block_sum[16];

#pragma unroll
        for (int i = 0; i < 16; ++i) {
            block_sum[i] = BlockReduce(temp).Sum(acc[i]);
            if (i != 15) __syncthreads();
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                atomicAdd(d_out + i, block_sum[i]);
            }
        }
    }

    __global__ void evalIntegrand_3DBloackReduce_SSSS_SSSP(
        int n, real_t hxyz, real_t rlx, real_t rly, real_t rlz, real_t r, int pitchX,
        const real2_t* __restrict__ distance_pair_grid, real_t* __restrict__ d_out /* length 40 */) {

        int tid     = blockIdx.x * blockDim.x + threadIdx.x;
        int totalXY = n * n;

        real_t acc[40] = {real_t(0.0)};

        if (tid < totalXY) {
            int y = tid / n;
            int x = tid - y * n;

            int pitchXY = pitchX * n;
            int base    = y * pitchX + x;

            const real3_t c4 = reinterpret_cast<const real3_t*>(d_c + 9)[0];
            const real3_t c3 = reinterpret_cast<const real3_t*>(d_c + 6)[0];

            const real2_t alpha12 = reinterpret_cast<const real2_t*>(d_alpha + 0)[0];
            const real2_t alpha34 = reinterpret_cast<const real2_t*>(d_alpha + 2)[0];

            real_t w = hxyz;
            if (x == 0 || x == n - 1) w *= real_t(0.5);
            if (y == 0 || y == n - 1) w *= real_t(0.5);

            const real_t X = __ldg(&d_x_grid[x]);
            const real_t Y = __ldg(&d_y_grid[y]);

            // r2 = r1 + r*omega
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

                const real_t z4 = Z - c4.z + rlz;
                const real_t z3 = Z - c3.z + rlz;

                const real_t d3 = norm(x3, y3, z3);
                const real_t d4 = norm(x4, y4, z4);

                const real_t expo = r - (a12 + alpha34.x * d3 + alpha34.y * d4);
                const real_t wz   = (k == 0 || k == n - 1) ? real_t(0.5) : real_t(1.0);
                const real_t T    = exp_fn(expo) * wz;

                // left SS vector: (1, d1, d2, d1*d2)
                const real_t L0 = real_t(1.0);
                const real_t L1 = d1;
                const real_t L2 = d2;
                const real_t L3 = d1 * d2;

                const real_t RSS0 = real_t(1.0);
                const real_t RSS1 = d3;
                const real_t RSS2 = d4;
                const real_t RSS3 = d3 * d4;

                // right SP vector: (x4, y4, z4, d3x4, d3y4, d3z4)
                // center 3 S: [1, d3]
                // center 4 P: [x4, y4, z4]
                const real_t RSP0 = x4;
                const real_t RSP1 = y4;
                const real_t RSP2 = z4;
                const real_t RSP3 = d3 * x4;
                const real_t RSP4 = d3 * y4;
                const real_t RSP5 = d3 * z4;

                const real_t TL0 = T * L0;
                const real_t TL1 = T * L1;
                const real_t TL2 = T * L2;
                const real_t TL3 = T * L3;

                // 4 by 6 outer product
                acc[0] += TL0 * RSS0;
                acc[1] += TL0 * RSS1;
                acc[2] += TL0 * RSS2;
                acc[3] += TL0 * RSS3;
                acc[4] += TL1 * RSS0;
                acc[5] += TL1 * RSS1;
                acc[6] += TL1 * RSS2;
                acc[7] += TL1 * RSS3;
                acc[8] += TL2 * RSS0;
                acc[9] += TL2 * RSS1;
                acc[10] += TL2 * RSS2;
                acc[11] += TL2 * RSS3;
                acc[12] += TL3 * RSS0;
                acc[13] += TL3 * RSS1;
                acc[14] += TL3 * RSS2;
                acc[15] += TL3 * RSS3;

                acc[16] += TL0 * RSP0;
                acc[17] += TL0 * RSP1;
                acc[18] += TL0 * RSP2;
                acc[19] += TL0 * RSP3;
                acc[20] += TL0 * RSP4;
                acc[21] += TL0 * RSP5;

                acc[22] += TL1 * RSP0;
                acc[23] += TL1 * RSP1;
                acc[24] += TL1 * RSP2;
                acc[25] += TL1 * RSP3;
                acc[26] += TL1 * RSP4;
                acc[27] += TL1 * RSP5;

                acc[28] += TL2 * RSP0;
                acc[29] += TL2 * RSP1;
                acc[30] += TL2 * RSP2;
                acc[31] += TL2 * RSP3;
                acc[32] += TL2 * RSP4;
                acc[33] += TL2 * RSP5;

                acc[34] += TL3 * RSP0;
                acc[35] += TL3 * RSP1;
                acc[36] += TL3 * RSP2;
                acc[37] += TL3 * RSP3;
                acc[38] += TL3 * RSP4;
                acc[39] += TL3 * RSP5;
            }

#pragma unroll
            for (int i = 0; i < 40; ++i)
                acc[i] *= w;
        }

        using BlockReduce = cub::BlockReduce<real_t, THREADS_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;

        real_t block_sum[40];

#pragma unroll
        for (int i = 0; i < 40; ++i) {
            block_sum[i] = BlockReduce(temp).Sum(acc[i]);
            if (i != 39) __syncthreads();
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int i = 0; i < 40; ++i) {
                atomicAdd(d_out + i, block_sum[i]);
            }
        }
    }

    __global__ void evalIntegrand_3DBloackReduce_SSPS(int n, real_t hxyz, real_t rlx, real_t rly,
                                                      real_t rlz, real_t r, int pitchX,
                                                      const real2_t* __restrict__ distance_pair_grid,
                                                      real_t* __restrict__ d_out /* length 24 */) {

        int tid     = blockIdx.x * blockDim.x + threadIdx.x;
        int totalXY = n * n;

        real_t acc[24] = {real_t(0.0)};

        if (tid < totalXY) {
            int y = tid / n;
            int x = tid - y * n;

            int pitchXY = pitchX * n;
            int base    = y * pitchX + x;

            const real3_t c3 = reinterpret_cast<const real3_t*>(d_c + 6)[0];
            const real3_t c4 = reinterpret_cast<const real3_t*>(d_c + 9)[0];

            const real2_t alpha12 = reinterpret_cast<const real2_t*>(d_alpha + 0)[0];
            const real2_t alpha34 = reinterpret_cast<const real2_t*>(d_alpha + 2)[0];

            real_t w = hxyz;
            if (x == 0 || x == n - 1) w *= real_t(0.5);
            if (y == 0 || y == n - 1) w *= real_t(0.5);

            const real_t X = __ldg(&d_x_grid[x]);
            const real_t Y = __ldg(&d_y_grid[y]);

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

                const real_t z3 = Z - c3.z + rlz;
                const real_t z4 = Z - c4.z + rlz;

                const real_t d3 = norm(x3, y3, z3);
                const real_t d4 = norm(x4, y4, z4);

                const real_t expo = r - (a12 + alpha34.x * d3 + alpha34.y * d4);
                const real_t wz   = (k == 0 || k == n - 1) ? real_t(0.5) : real_t(1.0);
                const real_t T    = exp_fn(expo) * wz;

                // Left SS vector
                const real_t L0 = real_t(1.0);
                const real_t L1 = d1;
                const real_t L2 = d2;
                const real_t L3 = d1 * d2;

                // Right PS vector: center 3 P, center 4 S
                const real_t R0 = x3;
                const real_t R1 = y3;
                const real_t R2 = z3;
                const real_t R3 = x3 * d4;
                const real_t R4 = y3 * d4;
                const real_t R5 = z3 * d4;

                const real_t TL0 = T * L0;
                const real_t TL1 = T * L1;
                const real_t TL2 = T * L2;
                const real_t TL3 = T * L3;

                acc[0] += TL0 * R0;
                acc[1] += TL0 * R1;
                acc[2] += TL0 * R2;
                acc[3] += TL0 * R3;
                acc[4] += TL0 * R4;
                acc[5] += TL0 * R5;

                acc[6] += TL1 * R0;
                acc[7] += TL1 * R1;
                acc[8] += TL1 * R2;
                acc[9] += TL1 * R3;
                acc[10] += TL1 * R4;
                acc[11] += TL1 * R5;

                acc[12] += TL2 * R0;
                acc[13] += TL2 * R1;
                acc[14] += TL2 * R2;
                acc[15] += TL2 * R3;
                acc[16] += TL2 * R4;
                acc[17] += TL2 * R5;

                acc[18] += TL3 * R0;
                acc[19] += TL3 * R1;
                acc[20] += TL3 * R2;
                acc[21] += TL3 * R3;
                acc[22] += TL3 * R4;
                acc[23] += TL3 * R5;
            }

#pragma unroll
            for (int i = 0; i < 24; ++i)
                acc[i] *= w;
        }

        using BlockReduce = cub::BlockReduce<real_t, THREADS_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;

        real_t block_sum[24];

#pragma unroll
        for (int i = 0; i < 24; ++i) {
            block_sum[i] = BlockReduce(temp).Sum(acc[i]);
            if (i != 23) __syncthreads();
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int i = 0; i < 24; ++i) {
                atomicAdd(d_out + i, block_sum[i]);
            }
        }
    }

    __global__ void evalIntegrand_3DBloackReduce_SSSS_SSSP_SSPS(
        int n, real_t hxyz, real_t rlx, real_t rly, real_t rlz, real_t r, int pitchX,
        const real2_t* __restrict__ distance_pair_grid, real_t* __restrict__ d_out /* length 64 */) {

        int tid     = blockIdx.x * blockDim.x + threadIdx.x;
        int totalXY = n * n;

        real_t acc[64] = {real_t(0.0)};

        if (tid < totalXY) {
            int y = tid / n;
            int x = tid - y * n;

            int pitchXY = pitchX * n;
            int base    = y * pitchX + x;

            const real3_t c3 = reinterpret_cast<const real3_t*>(d_c + 6)[0];
            const real3_t c4 = reinterpret_cast<const real3_t*>(d_c + 9)[0];

            const real2_t alpha12 = reinterpret_cast<const real2_t*>(d_alpha + 0)[0];
            const real2_t alpha34 = reinterpret_cast<const real2_t*>(d_alpha + 2)[0];

            real_t w = hxyz;
            if (x == 0 || x == n - 1) w *= real_t(0.5);
            if (y == 0 || y == n - 1) w *= real_t(0.5);

            const real_t X = __ldg(&d_x_grid[x]);
            const real_t Y = __ldg(&d_y_grid[y]);

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

                const real_t z3 = Z - c3.z + rlz;
                const real_t z4 = Z - c4.z + rlz;

                const real_t d3 = norm(x3, y3, z3);
                const real_t d4 = norm(x4, y4, z4);

                const real_t expo = r - (a12 + alpha34.x * d3 + alpha34.y * d4);
                const real_t wz   = (k == 0 || k == n - 1) ? real_t(0.5) : real_t(1.0);
                const real_t T    = exp_fn(expo) * wz;

                // Left SS
                const real_t L0 = real_t(1.0);
                const real_t L1 = d1;
                const real_t L2 = d2;
                const real_t L3 = d1 * d2;

                // Right SS
                const real_t RSS0 = real_t(1.0);
                const real_t RSS1 = d3;
                const real_t RSS2 = d4;
                const real_t RSS3 = d3 * d4;

                // Right SP
                const real_t RSP0 = x4;
                const real_t RSP1 = y4;
                const real_t RSP2 = z4;
                const real_t RSP3 = d3 * x4;
                const real_t RSP4 = d3 * y4;
                const real_t RSP5 = d3 * z4;

                // Right PS
                const real_t RPS0 = x3;
                const real_t RPS1 = y3;
                const real_t RPS2 = z3;
                const real_t RPS3 = x3 * d4;
                const real_t RPS4 = y3 * d4;
                const real_t RPS5 = z3 * d4;

                const real_t TL0 = T * L0;
                const real_t TL1 = T * L1;
                const real_t TL2 = T * L2;
                const real_t TL3 = T * L3;

                // SS|SS
                acc[0] += TL0 * RSS0;
                acc[1] += TL0 * RSS1;
                acc[2] += TL0 * RSS2;
                acc[3] += TL0 * RSS3;
                acc[4] += TL1 * RSS0;
                acc[5] += TL1 * RSS1;
                acc[6] += TL1 * RSS2;
                acc[7] += TL1 * RSS3;
                acc[8] += TL2 * RSS0;
                acc[9] += TL2 * RSS1;
                acc[10] += TL2 * RSS2;
                acc[11] += TL2 * RSS3;
                acc[12] += TL3 * RSS0;
                acc[13] += TL3 * RSS1;
                acc[14] += TL3 * RSS2;
                acc[15] += TL3 * RSS3;

                // SS|SP
                acc[16] += TL0 * RSP0;
                acc[17] += TL0 * RSP1;
                acc[18] += TL0 * RSP2;
                acc[19] += TL0 * RSP3;
                acc[20] += TL0 * RSP4;
                acc[21] += TL0 * RSP5;

                acc[22] += TL1 * RSP0;
                acc[23] += TL1 * RSP1;
                acc[24] += TL1 * RSP2;
                acc[25] += TL1 * RSP3;
                acc[26] += TL1 * RSP4;
                acc[27] += TL1 * RSP5;

                acc[28] += TL2 * RSP0;
                acc[29] += TL2 * RSP1;
                acc[30] += TL2 * RSP2;
                acc[31] += TL2 * RSP3;
                acc[32] += TL2 * RSP4;
                acc[33] += TL2 * RSP5;

                acc[34] += TL3 * RSP0;
                acc[35] += TL3 * RSP1;
                acc[36] += TL3 * RSP2;
                acc[37] += TL3 * RSP3;
                acc[38] += TL3 * RSP4;
                acc[39] += TL3 * RSP5;

                // SS|PS
                acc[40] += TL0 * RPS0;
                acc[41] += TL0 * RPS1;
                acc[42] += TL0 * RPS2;
                acc[43] += TL0 * RPS3;
                acc[44] += TL0 * RPS4;
                acc[45] += TL0 * RPS5;

                acc[46] += TL1 * RPS0;
                acc[47] += TL1 * RPS1;
                acc[48] += TL1 * RPS2;
                acc[49] += TL1 * RPS3;
                acc[50] += TL1 * RPS4;
                acc[51] += TL1 * RPS5;

                acc[52] += TL2 * RPS0;
                acc[53] += TL2 * RPS1;
                acc[54] += TL2 * RPS2;
                acc[55] += TL2 * RPS3;
                acc[56] += TL2 * RPS4;
                acc[57] += TL2 * RPS5;

                acc[58] += TL3 * RPS0;
                acc[59] += TL3 * RPS1;
                acc[60] += TL3 * RPS2;
                acc[61] += TL3 * RPS3;
                acc[62] += TL3 * RPS4;
                acc[63] += TL3 * RPS5;
            }

#pragma unroll
            for (int i = 0; i < 64; ++i)
                acc[i] *= w;
        }

        using BlockReduce = cub::BlockReduce<real_t, THREADS_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;

        real_t block_sum[64];

#pragma unroll
        for (int i = 0; i < 64; ++i) {
            block_sum[i] = BlockReduce(temp).Sum(acc[i]);
            if (i != 63) __syncthreads();
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int i = 0; i < 64; ++i) {
                atomicAdd(d_out + i, block_sum[i]);
            }
        }
    }

    __global__ void evalIntegrand_3DBloackReduce_SSSS_SSSP_SSPS_SPSS(
        int n, real_t hxyz, real_t rlx, real_t rly, real_t rlz, real_t r, int pitchX,
        const real2_t* __restrict__ distance_pair_grid, real_t* __restrict__ d_out /* length 88 */) {

        int tid     = blockIdx.x * blockDim.x + threadIdx.x;
        int totalXY = n * n;

        real_t acc[88] = {real_t(0.0)};

        if (tid < totalXY) {
            int y = tid / n;
            int x = tid - y * n;

            int pitchXY = pitchX * n;
            int base    = y * pitchX + x;

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

            // center-2 relative coords for SP|SS
            const real_t x2 = X - c2.x;
            const real_t y2 = Y - c2.y;

            // shifted coords for centers 3 and 4
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

                const real_t z2 = Z - c2.z;
                const real_t z3 = Z - c3.z + rlz;
                const real_t z4 = Z - c4.z + rlz;

                const real_t d3 = norm(x3, y3, z3);
                const real_t d4 = norm(x4, y4, z4);

                const real_t expo = r - (a12 + alpha34.x * d3 + alpha34.y * d4);
                const real_t wz   = (k == 0 || k == n - 1) ? real_t(0.5) : real_t(1.0);
                const real_t T    = exp_fn(expo) * wz;

                // Left SS
                const real_t L0 = real_t(1.0);
                const real_t L1 = d1;
                const real_t L2 = d2;
                const real_t L3 = d1 * d2;

                // Right SS
                const real_t RSS0 = real_t(1.0);
                const real_t RSS1 = d3;
                const real_t RSS2 = d4;
                const real_t RSS3 = d3 * d4;

                // Right SP
                const real_t RSP0 = x4;
                const real_t RSP1 = y4;
                const real_t RSP2 = z4;
                const real_t RSP3 = d3 * x4;
                const real_t RSP4 = d3 * y4;
                const real_t RSP5 = d3 * z4;

                // Right PS
                const real_t RPS0 = x3;
                const real_t RPS1 = y3;
                const real_t RPS2 = z3;
                const real_t RPS3 = x3 * d4;
                const real_t RPS4 = y3 * d4;
                const real_t RPS5 = z3 * d4;

                // Left SP
                const real_t LSP0 = x2;
                const real_t LSP1 = y2;
                const real_t LSP2 = z2;
                const real_t LSP3 = d1 * x2;
                const real_t LSP4 = d1 * y2;
                const real_t LSP5 = d1 * z2;

                const real_t TLSS0 = T * L0;
                const real_t TLSS1 = T * L1;
                const real_t TLSS2 = T * L2;
                const real_t TLSS3 = T * L3;
                // SP|SS
                const real_t TLSP0 = T * LSP0;
                const real_t TLSP1 = T * LSP1;
                const real_t TLSP2 = T * LSP2;
                const real_t TLSP3 = T * LSP3;
                const real_t TLSP4 = T * LSP4;
                const real_t TLSP5 = T * LSP5;

                // SS|SS
                acc[0] += TLSS0 * RSS0;
                acc[1] += TLSS0 * RSS1;
                acc[2] += TLSS0 * RSS2;
                acc[3] += TLSS0 * RSS3;
                acc[4] += TLSS1 * RSS0;
                acc[5] += TLSS1 * RSS1;
                acc[6] += TLSS1 * RSS2;
                acc[7] += TLSS1 * RSS3;
                acc[8] += TLSS2 * RSS0;
                acc[9] += TLSS2 * RSS1;
                acc[10] += TLSS2 * RSS2;
                acc[11] += TLSS2 * RSS3;
                acc[12] += TLSS3 * RSS0;
                acc[13] += TLSS3 * RSS1;
                acc[14] += TLSS3 * RSS2;
                acc[15] += TLSS3 * RSS3;

                // SS|SP
                acc[16] += TLSS0 * RSP0;
                acc[17] += TLSS0 * RSP1;
                acc[18] += TLSS0 * RSP2;
                acc[19] += TLSS0 * RSP3;
                acc[20] += TLSS0 * RSP4;
                acc[21] += TLSS0 * RSP5;

                acc[22] += TLSS1 * RSP0;
                acc[23] += TLSS1 * RSP1;
                acc[24] += TLSS1 * RSP2;
                acc[25] += TLSS1 * RSP3;
                acc[26] += TLSS1 * RSP4;
                acc[27] += TLSS1 * RSP5;

                acc[28] += TLSS2 * RSP0;
                acc[29] += TLSS2 * RSP1;
                acc[30] += TLSS2 * RSP2;
                acc[31] += TLSS2 * RSP3;
                acc[32] += TLSS2 * RSP4;
                acc[33] += TLSS2 * RSP5;

                acc[34] += TLSS3 * RSP0;
                acc[35] += TLSS3 * RSP1;
                acc[36] += TLSS3 * RSP2;
                acc[37] += TLSS3 * RSP3;
                acc[38] += TLSS3 * RSP4;
                acc[39] += TLSS3 * RSP5;

                // SS|PS
                acc[40] += TLSS0 * RPS0;
                acc[41] += TLSS0 * RPS1;
                acc[42] += TLSS0 * RPS2;
                acc[43] += TLSS0 * RPS3;
                acc[44] += TLSS0 * RPS4;
                acc[45] += TLSS0 * RPS5;

                acc[46] += TLSS1 * RPS0;
                acc[47] += TLSS1 * RPS1;
                acc[48] += TLSS1 * RPS2;
                acc[49] += TLSS1 * RPS3;
                acc[50] += TLSS1 * RPS4;
                acc[51] += TLSS1 * RPS5;

                acc[52] += TLSS2 * RPS0;
                acc[53] += TLSS2 * RPS1;
                acc[54] += TLSS2 * RPS2;
                acc[55] += TLSS2 * RPS3;
                acc[56] += TLSS2 * RPS4;
                acc[57] += TLSS2 * RPS5;

                acc[58] += TLSS3 * RPS0;
                acc[59] += TLSS3 * RPS1;
                acc[60] += TLSS3 * RPS2;
                acc[61] += TLSS3 * RPS3;
                acc[62] += TLSS3 * RPS4;
                acc[63] += TLSS3 * RPS5;

                // SP|SS
                acc[64] += TLSP0 * RSS0;
                acc[65] += TLSP0 * RSS1;
                acc[66] += TLSP0 * RSS2;
                acc[67] += TLSP0 * RSS3;
                acc[68] += TLSP1 * RSS0;
                acc[69] += TLSP1 * RSS1;
                acc[70] += TLSP1 * RSS2;
                acc[71] += TLSP1 * RSS3;
                acc[72] += TLSP2 * RSS0;
                acc[73] += TLSP2 * RSS1;
                acc[74] += TLSP2 * RSS2;
                acc[75] += TLSP2 * RSS3;
                acc[76] += TLSP3 * RSS0;
                acc[77] += TLSP3 * RSS1;
                acc[78] += TLSP3 * RSS2;
                acc[79] += TLSP3 * RSS3;
                acc[80] += TLSP4 * RSS0;
                acc[81] += TLSP4 * RSS1;
                acc[82] += TLSP4 * RSS2;
                acc[83] += TLSP4 * RSS3;
                acc[84] += TLSP5 * RSS0;
                acc[85] += TLSP5 * RSS1;
                acc[86] += TLSP5 * RSS2;
                acc[87] += TLSP5 * RSS3;
            }

#pragma unroll
            for (int i = 0; i < 88; ++i)
                acc[i] *= w;
        }

        using BlockReduce = cub::BlockReduce<real_t, THREADS_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;

        real_t block_sum[88];

#pragma unroll
        for (int i = 0; i < 88; ++i) {
            block_sum[i] = BlockReduce(temp).Sum(acc[i]);
            if (i != 87) __syncthreads();
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int i = 0; i < 88; ++i) {
                atomicAdd(d_out + i, block_sum[i]);
            }
        }
    }

    __global__ void evalIntegrand_3DBloackReduce_SSSS_SSSP_SSPS_SPSS_PSSS(
        int n, real_t hxyz, real_t rlx, real_t rly, real_t rlz, real_t r, int pitchX,
        const real2_t* __restrict__ distance_pair_grid,
        real_t* __restrict__ d_out /* length 112 */) {

        int tid     = blockIdx.x * blockDim.x + threadIdx.x;
        int totalXY = n * n;

        real_t acc[112] = {real_t(0.0)};

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

            // static left coords
            const real_t x1 = X - c1.x;
            const real_t y1 = Y - c1.y;
            const real_t x2 = X - c2.x;
            const real_t y2 = Y - c2.y;

            // shifted right coords
            const real_t x3 = X - c3.x + rlx;
            const real_t y3 = Y - c3.y + rly;
            const real_t x4 = X - c4.x + rlx;
            const real_t y4 = Y - c4.y + rly;

            {
                const real_t Z = __ldg(&d_z_grid[0]);

                const real2_t d12 = __ldg(distance_pair_grid + (base));
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
                const real_t T    = exp_fn(expo) * 0.5;

                // Left SS
                const real_t L0 = real_t(1.0);
                const real_t L1 = d1;
                const real_t L2 = d2;
                const real_t L3 = d1 * d2;

                // Right SS
                const real_t RSS0 = real_t(1.0);
                const real_t RSS1 = d3;
                const real_t RSS2 = d4;
                const real_t RSS3 = d3 * d4;

                // Right SP
                const real_t RSP0 = x4;
                const real_t RSP1 = y4;
                const real_t RSP2 = z4;
                const real_t RSP3 = d3 * x4;
                const real_t RSP4 = d3 * y4;
                const real_t RSP5 = d3 * z4;

                // Right PS
                const real_t RPS0 = x3;
                const real_t RPS1 = y3;
                const real_t RPS2 = z3;
                const real_t RPS3 = x3 * d4;
                const real_t RPS4 = y3 * d4;
                const real_t RPS5 = z3 * d4;

                // Left SP
                const real_t LSP0 = x2;
                const real_t LSP1 = y2;
                const real_t LSP2 = z2;
                const real_t LSP3 = d1 * x2;
                const real_t LSP4 = d1 * y2;
                const real_t LSP5 = d1 * z2;

                // Left PS
                const real_t LPS0 = x1;
                const real_t LPS1 = y1;
                const real_t LPS2 = z1;
                const real_t LPS3 = x1 * d2;
                const real_t LPS4 = y1 * d2;
                const real_t LPS5 = z1 * d2;

                const real_t TL0 = T * L0;
                const real_t TL1 = T * L1;
                const real_t TL2 = T * L2;
                const real_t TL3 = T * L3;
                // SP|SS
                const real_t TSP0 = T * LSP0;
                const real_t TSP1 = T * LSP1;
                const real_t TSP2 = T * LSP2;
                const real_t TSP3 = T * LSP3;
                const real_t TSP4 = T * LSP4;
                const real_t TSP5 = T * LSP5;
                // PS|SS
                const real_t TPS0 = T * LPS0;
                const real_t TPS1 = T * LPS1;
                const real_t TPS2 = T * LPS2;
                const real_t TPS3 = T * LPS3;
                const real_t TPS4 = T * LPS4;
                const real_t TPS5 = T * LPS5;

                // SS|SS
                acc[0] += TL0 * RSS0;
                acc[1] += TL0 * RSS1;
                acc[2] += TL0 * RSS2;
                acc[3] += TL0 * RSS3;
                acc[4] += TL1 * RSS0;
                acc[5] += TL1 * RSS1;
                acc[6] += TL1 * RSS2;
                acc[7] += TL1 * RSS3;
                acc[8] += TL2 * RSS0;
                acc[9] += TL2 * RSS1;
                acc[10] += TL2 * RSS2;
                acc[11] += TL2 * RSS3;
                acc[12] += TL3 * RSS0;
                acc[13] += TL3 * RSS1;
                acc[14] += TL3 * RSS2;
                acc[15] += TL3 * RSS3;

                // SS|SP
                acc[16] += TL0 * RSP0;
                acc[17] += TL0 * RSP1;
                acc[18] += TL0 * RSP2;
                acc[19] += TL0 * RSP3;
                acc[20] += TL0 * RSP4;
                acc[21] += TL0 * RSP5;
                acc[22] += TL1 * RSP0;
                acc[23] += TL1 * RSP1;
                acc[24] += TL1 * RSP2;
                acc[25] += TL1 * RSP3;
                acc[26] += TL1 * RSP4;
                acc[27] += TL1 * RSP5;
                acc[28] += TL2 * RSP0;
                acc[29] += TL2 * RSP1;
                acc[30] += TL2 * RSP2;
                acc[31] += TL2 * RSP3;
                acc[32] += TL2 * RSP4;
                acc[33] += TL2 * RSP5;
                acc[34] += TL3 * RSP0;
                acc[35] += TL3 * RSP1;
                acc[36] += TL3 * RSP2;
                acc[37] += TL3 * RSP3;
                acc[38] += TL3 * RSP4;
                acc[39] += TL3 * RSP5;

                // SS|PS
                acc[40] += TL0 * RPS0;
                acc[41] += TL0 * RPS1;
                acc[42] += TL0 * RPS2;
                acc[43] += TL0 * RPS3;
                acc[44] += TL0 * RPS4;
                acc[45] += TL0 * RPS5;
                acc[46] += TL1 * RPS0;
                acc[47] += TL1 * RPS1;
                acc[48] += TL1 * RPS2;
                acc[49] += TL1 * RPS3;
                acc[50] += TL1 * RPS4;
                acc[51] += TL1 * RPS5;
                acc[52] += TL2 * RPS0;
                acc[53] += TL2 * RPS1;
                acc[54] += TL2 * RPS2;
                acc[55] += TL2 * RPS3;
                acc[56] += TL2 * RPS4;
                acc[57] += TL2 * RPS5;
                acc[58] += TL3 * RPS0;
                acc[59] += TL3 * RPS1;
                acc[60] += TL3 * RPS2;
                acc[61] += TL3 * RPS3;
                acc[62] += TL3 * RPS4;
                acc[63] += TL3 * RPS5;

                acc[64] += TSP0 * RSS0;
                acc[65] += TSP0 * RSS1;
                acc[66] += TSP0 * RSS2;
                acc[67] += TSP0 * RSS3;
                acc[68] += TSP1 * RSS0;
                acc[69] += TSP1 * RSS1;
                acc[70] += TSP1 * RSS2;
                acc[71] += TSP1 * RSS3;
                acc[72] += TSP2 * RSS0;
                acc[73] += TSP2 * RSS1;
                acc[74] += TSP2 * RSS2;
                acc[75] += TSP2 * RSS3;
                acc[76] += TSP3 * RSS0;
                acc[77] += TSP3 * RSS1;
                acc[78] += TSP3 * RSS2;
                acc[79] += TSP3 * RSS3;
                acc[80] += TSP4 * RSS0;
                acc[81] += TSP4 * RSS1;
                acc[82] += TSP4 * RSS2;
                acc[83] += TSP4 * RSS3;
                acc[84] += TSP5 * RSS0;
                acc[85] += TSP5 * RSS1;
                acc[86] += TSP5 * RSS2;
                acc[87] += TSP5 * RSS3;

                acc[88] += TPS0 * RSS0;
                acc[89] += TPS0 * RSS1;
                acc[90] += TPS0 * RSS2;
                acc[91] += TPS0 * RSS3;
                acc[92] += TPS1 * RSS0;
                acc[93] += TPS1 * RSS1;
                acc[94] += TPS1 * RSS2;
                acc[95] += TPS1 * RSS3;
                acc[96] += TPS2 * RSS0;
                acc[97] += TPS2 * RSS1;
                acc[98] += TPS2 * RSS2;
                acc[99] += TPS2 * RSS3;
                acc[100] += TPS3 * RSS0;
                acc[101] += TPS3 * RSS1;
                acc[102] += TPS3 * RSS2;
                acc[103] += TPS3 * RSS3;
                acc[104] += TPS4 * RSS0;
                acc[105] += TPS4 * RSS1;
                acc[106] += TPS4 * RSS2;
                acc[107] += TPS4 * RSS3;
                acc[108] += TPS5 * RSS0;
                acc[109] += TPS5 * RSS1;
                acc[110] += TPS5 * RSS2;
                acc[111] += TPS5 * RSS3;
            }

            for (int k = 1; k < n - 1; ++k) {
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
                const real_t T    = exp_fn(expo);

                // Left SS
                const real_t L0 = real_t(1.0);
                const real_t L1 = d1;
                const real_t L2 = d2;
                const real_t L3 = d1 * d2;

                // Right SS
                const real_t RSS0 = real_t(1.0);
                const real_t RSS1 = d3;
                const real_t RSS2 = d4;
                const real_t RSS3 = d3 * d4;

                // Right SP
                const real_t RSP0 = x4;
                const real_t RSP1 = y4;
                const real_t RSP2 = z4;
                const real_t RSP3 = d3 * x4;
                const real_t RSP4 = d3 * y4;
                const real_t RSP5 = d3 * z4;

                // Right PS
                const real_t RPS0 = x3;
                const real_t RPS1 = y3;
                const real_t RPS2 = z3;
                const real_t RPS3 = x3 * d4;
                const real_t RPS4 = y3 * d4;
                const real_t RPS5 = z3 * d4;

                // Left SP
                const real_t LSP0 = x2;
                const real_t LSP1 = y2;
                const real_t LSP2 = z2;
                const real_t LSP3 = d1 * x2;
                const real_t LSP4 = d1 * y2;
                const real_t LSP5 = d1 * z2;

                // Left PS
                const real_t LPS0 = x1;
                const real_t LPS1 = y1;
                const real_t LPS2 = z1;
                const real_t LPS3 = x1 * d2;
                const real_t LPS4 = y1 * d2;
                const real_t LPS5 = z1 * d2;

                const real_t TL0 = T * L0;
                const real_t TL1 = T * L1;
                const real_t TL2 = T * L2;
                const real_t TL3 = T * L3;
                // SP|SS
                const real_t TSP0 = T * LSP0;
                const real_t TSP1 = T * LSP1;
                const real_t TSP2 = T * LSP2;
                const real_t TSP3 = T * LSP3;
                const real_t TSP4 = T * LSP4;
                const real_t TSP5 = T * LSP5;
                // PS|SS
                const real_t TPS0 = T * LPS0;
                const real_t TPS1 = T * LPS1;
                const real_t TPS2 = T * LPS2;
                const real_t TPS3 = T * LPS3;
                const real_t TPS4 = T * LPS4;
                const real_t TPS5 = T * LPS5;

                // SS|SS
                acc[0] += TL0 * RSS0;
                acc[1] += TL0 * RSS1;
                acc[2] += TL0 * RSS2;
                acc[3] += TL0 * RSS3;
                acc[4] += TL1 * RSS0;
                acc[5] += TL1 * RSS1;
                acc[6] += TL1 * RSS2;
                acc[7] += TL1 * RSS3;
                acc[8] += TL2 * RSS0;
                acc[9] += TL2 * RSS1;
                acc[10] += TL2 * RSS2;
                acc[11] += TL2 * RSS3;
                acc[12] += TL3 * RSS0;
                acc[13] += TL3 * RSS1;
                acc[14] += TL3 * RSS2;
                acc[15] += TL3 * RSS3;

                // SS|SP
                acc[16] += TL0 * RSP0;
                acc[17] += TL0 * RSP1;
                acc[18] += TL0 * RSP2;
                acc[19] += TL0 * RSP3;
                acc[20] += TL0 * RSP4;
                acc[21] += TL0 * RSP5;
                acc[22] += TL1 * RSP0;
                acc[23] += TL1 * RSP1;
                acc[24] += TL1 * RSP2;
                acc[25] += TL1 * RSP3;
                acc[26] += TL1 * RSP4;
                acc[27] += TL1 * RSP5;
                acc[28] += TL2 * RSP0;
                acc[29] += TL2 * RSP1;
                acc[30] += TL2 * RSP2;
                acc[31] += TL2 * RSP3;
                acc[32] += TL2 * RSP4;
                acc[33] += TL2 * RSP5;
                acc[34] += TL3 * RSP0;
                acc[35] += TL3 * RSP1;
                acc[36] += TL3 * RSP2;
                acc[37] += TL3 * RSP3;
                acc[38] += TL3 * RSP4;
                acc[39] += TL3 * RSP5;

                // SS|PS
                acc[40] += TL0 * RPS0;
                acc[41] += TL0 * RPS1;
                acc[42] += TL0 * RPS2;
                acc[43] += TL0 * RPS3;
                acc[44] += TL0 * RPS4;
                acc[45] += TL0 * RPS5;
                acc[46] += TL1 * RPS0;
                acc[47] += TL1 * RPS1;
                acc[48] += TL1 * RPS2;
                acc[49] += TL1 * RPS3;
                acc[50] += TL1 * RPS4;
                acc[51] += TL1 * RPS5;
                acc[52] += TL2 * RPS0;
                acc[53] += TL2 * RPS1;
                acc[54] += TL2 * RPS2;
                acc[55] += TL2 * RPS3;
                acc[56] += TL2 * RPS4;
                acc[57] += TL2 * RPS5;
                acc[58] += TL3 * RPS0;
                acc[59] += TL3 * RPS1;
                acc[60] += TL3 * RPS2;
                acc[61] += TL3 * RPS3;
                acc[62] += TL3 * RPS4;
                acc[63] += TL3 * RPS5;

                acc[64] += TSP0 * RSS0;
                acc[65] += TSP0 * RSS1;
                acc[66] += TSP0 * RSS2;
                acc[67] += TSP0 * RSS3;
                acc[68] += TSP1 * RSS0;
                acc[69] += TSP1 * RSS1;
                acc[70] += TSP1 * RSS2;
                acc[71] += TSP1 * RSS3;
                acc[72] += TSP2 * RSS0;
                acc[73] += TSP2 * RSS1;
                acc[74] += TSP2 * RSS2;
                acc[75] += TSP2 * RSS3;
                acc[76] += TSP3 * RSS0;
                acc[77] += TSP3 * RSS1;
                acc[78] += TSP3 * RSS2;
                acc[79] += TSP3 * RSS3;
                acc[80] += TSP4 * RSS0;
                acc[81] += TSP4 * RSS1;
                acc[82] += TSP4 * RSS2;
                acc[83] += TSP4 * RSS3;
                acc[84] += TSP5 * RSS0;
                acc[85] += TSP5 * RSS1;
                acc[86] += TSP5 * RSS2;
                acc[87] += TSP5 * RSS3;

                acc[88] += TPS0 * RSS0;
                acc[89] += TPS0 * RSS1;
                acc[90] += TPS0 * RSS2;
                acc[91] += TPS0 * RSS3;
                acc[92] += TPS1 * RSS0;
                acc[93] += TPS1 * RSS1;
                acc[94] += TPS1 * RSS2;
                acc[95] += TPS1 * RSS3;
                acc[96] += TPS2 * RSS0;
                acc[97] += TPS2 * RSS1;
                acc[98] += TPS2 * RSS2;
                acc[99] += TPS2 * RSS3;
                acc[100] += TPS3 * RSS0;
                acc[101] += TPS3 * RSS1;
                acc[102] += TPS3 * RSS2;
                acc[103] += TPS3 * RSS3;
                acc[104] += TPS4 * RSS0;
                acc[105] += TPS4 * RSS1;
                acc[106] += TPS4 * RSS2;
                acc[107] += TPS4 * RSS3;
                acc[108] += TPS5 * RSS0;
                acc[109] += TPS5 * RSS1;
                acc[110] += TPS5 * RSS2;
                acc[111] += TPS5 * RSS3;
            }

            {

                const real_t Z = __ldg(&d_z_grid[n - 1]);

                const real2_t d12 = __ldg(distance_pair_grid + ((n - 1) * pitchXY + base));
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
                const real_t T    = exp_fn(expo) * 0.5;

                // Left SS
                const real_t L0 = real_t(1.0);
                const real_t L1 = d1;
                const real_t L2 = d2;
                const real_t L3 = d1 * d2;

                // Right SS
                const real_t RSS0 = real_t(1.0);
                const real_t RSS1 = d3;
                const real_t RSS2 = d4;
                const real_t RSS3 = d3 * d4;

                // Right SP
                const real_t RSP0 = x4;
                const real_t RSP1 = y4;
                const real_t RSP2 = z4;
                const real_t RSP3 = d3 * x4;
                const real_t RSP4 = d3 * y4;
                const real_t RSP5 = d3 * z4;

                // Right PS
                const real_t RPS0 = x3;
                const real_t RPS1 = y3;
                const real_t RPS2 = z3;
                const real_t RPS3 = x3 * d4;
                const real_t RPS4 = y3 * d4;
                const real_t RPS5 = z3 * d4;

                // Left SP
                const real_t LSP0 = x2;
                const real_t LSP1 = y2;
                const real_t LSP2 = z2;
                const real_t LSP3 = d1 * x2;
                const real_t LSP4 = d1 * y2;
                const real_t LSP5 = d1 * z2;

                // Left PS
                const real_t LPS0 = x1;
                const real_t LPS1 = y1;
                const real_t LPS2 = z1;
                const real_t LPS3 = x1 * d2;
                const real_t LPS4 = y1 * d2;
                const real_t LPS5 = z1 * d2;

                const real_t TL0 = T * L0;
                const real_t TL1 = T * L1;
                const real_t TL2 = T * L2;
                const real_t TL3 = T * L3;
                // SP|SS
                const real_t TSP0 = T * LSP0;
                const real_t TSP1 = T * LSP1;
                const real_t TSP2 = T * LSP2;
                const real_t TSP3 = T * LSP3;
                const real_t TSP4 = T * LSP4;
                const real_t TSP5 = T * LSP5;
                // PS|SS
                const real_t TPS0 = T * LPS0;
                const real_t TPS1 = T * LPS1;
                const real_t TPS2 = T * LPS2;
                const real_t TPS3 = T * LPS3;
                const real_t TPS4 = T * LPS4;
                const real_t TPS5 = T * LPS5;

                // SS|SS
                acc[0] += TL0 * RSS0;
                acc[1] += TL0 * RSS1;
                acc[2] += TL0 * RSS2;
                acc[3] += TL0 * RSS3;
                acc[4] += TL1 * RSS0;
                acc[5] += TL1 * RSS1;
                acc[6] += TL1 * RSS2;
                acc[7] += TL1 * RSS3;
                acc[8] += TL2 * RSS0;
                acc[9] += TL2 * RSS1;
                acc[10] += TL2 * RSS2;
                acc[11] += TL2 * RSS3;
                acc[12] += TL3 * RSS0;
                acc[13] += TL3 * RSS1;
                acc[14] += TL3 * RSS2;
                acc[15] += TL3 * RSS3;

                // SS|SP
                acc[16] += TL0 * RSP0;
                acc[17] += TL0 * RSP1;
                acc[18] += TL0 * RSP2;
                acc[19] += TL0 * RSP3;
                acc[20] += TL0 * RSP4;
                acc[21] += TL0 * RSP5;
                acc[22] += TL1 * RSP0;
                acc[23] += TL1 * RSP1;
                acc[24] += TL1 * RSP2;
                acc[25] += TL1 * RSP3;
                acc[26] += TL1 * RSP4;
                acc[27] += TL1 * RSP5;
                acc[28] += TL2 * RSP0;
                acc[29] += TL2 * RSP1;
                acc[30] += TL2 * RSP2;
                acc[31] += TL2 * RSP3;
                acc[32] += TL2 * RSP4;
                acc[33] += TL2 * RSP5;
                acc[34] += TL3 * RSP0;
                acc[35] += TL3 * RSP1;
                acc[36] += TL3 * RSP2;
                acc[37] += TL3 * RSP3;
                acc[38] += TL3 * RSP4;
                acc[39] += TL3 * RSP5;

                // SS|PS
                acc[40] += TL0 * RPS0;
                acc[41] += TL0 * RPS1;
                acc[42] += TL0 * RPS2;
                acc[43] += TL0 * RPS3;
                acc[44] += TL0 * RPS4;
                acc[45] += TL0 * RPS5;
                acc[46] += TL1 * RPS0;
                acc[47] += TL1 * RPS1;
                acc[48] += TL1 * RPS2;
                acc[49] += TL1 * RPS3;
                acc[50] += TL1 * RPS4;
                acc[51] += TL1 * RPS5;
                acc[52] += TL2 * RPS0;
                acc[53] += TL2 * RPS1;
                acc[54] += TL2 * RPS2;
                acc[55] += TL2 * RPS3;
                acc[56] += TL2 * RPS4;
                acc[57] += TL2 * RPS5;
                acc[58] += TL3 * RPS0;
                acc[59] += TL3 * RPS1;
                acc[60] += TL3 * RPS2;
                acc[61] += TL3 * RPS3;
                acc[62] += TL3 * RPS4;
                acc[63] += TL3 * RPS5;

                acc[64] += TSP0 * RSS0;
                acc[65] += TSP0 * RSS1;
                acc[66] += TSP0 * RSS2;
                acc[67] += TSP0 * RSS3;
                acc[68] += TSP1 * RSS0;
                acc[69] += TSP1 * RSS1;
                acc[70] += TSP1 * RSS2;
                acc[71] += TSP1 * RSS3;
                acc[72] += TSP2 * RSS0;
                acc[73] += TSP2 * RSS1;
                acc[74] += TSP2 * RSS2;
                acc[75] += TSP2 * RSS3;
                acc[76] += TSP3 * RSS0;
                acc[77] += TSP3 * RSS1;
                acc[78] += TSP3 * RSS2;
                acc[79] += TSP3 * RSS3;
                acc[80] += TSP4 * RSS0;
                acc[81] += TSP4 * RSS1;
                acc[82] += TSP4 * RSS2;
                acc[83] += TSP4 * RSS3;
                acc[84] += TSP5 * RSS0;
                acc[85] += TSP5 * RSS1;
                acc[86] += TSP5 * RSS2;
                acc[87] += TSP5 * RSS3;

                acc[88] += TPS0 * RSS0;
                acc[89] += TPS0 * RSS1;
                acc[90] += TPS0 * RSS2;
                acc[91] += TPS0 * RSS3;
                acc[92] += TPS1 * RSS0;
                acc[93] += TPS1 * RSS1;
                acc[94] += TPS1 * RSS2;
                acc[95] += TPS1 * RSS3;
                acc[96] += TPS2 * RSS0;
                acc[97] += TPS2 * RSS1;
                acc[98] += TPS2 * RSS2;
                acc[99] += TPS2 * RSS3;
                acc[100] += TPS3 * RSS0;
                acc[101] += TPS3 * RSS1;
                acc[102] += TPS3 * RSS2;
                acc[103] += TPS3 * RSS3;
                acc[104] += TPS4 * RSS0;
                acc[105] += TPS4 * RSS1;
                acc[106] += TPS4 * RSS2;
                acc[107] += TPS4 * RSS3;
                acc[108] += TPS5 * RSS0;
                acc[109] += TPS5 * RSS1;
                acc[110] += TPS5 * RSS2;
                acc[111] += TPS5 * RSS3;
            }

#pragma unroll
            for (int i = 0; i < 112; ++i)
                acc[i] *= w;
        }

        using BlockReduce = cub::BlockReduce<real_t, THREADS_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;

        real_t block_sum[112];

#pragma unroll
        for (int i = 0; i < 112; ++i) {
            block_sum[i] = BlockReduce(temp).Sum(acc[i]);
            if (i != 111) __syncthreads();
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int i = 0; i < 112; ++i) {
                atomicAdd(d_out + i, block_sum[i]);
            }
        }
    }
} // namespace cuslater