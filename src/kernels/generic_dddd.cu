#include "kernels.cuh"

namespace cuslater {

    template<int C>
    __device__ __forceinline__ real_t d_component(real_t x, real_t y, real_t z) {
        if constexpr (C == 0) {
            return x * y; // xy
        } else if constexpr (C == 1) {
            return x * z; // xz
        } else if constexpr (C == 2) {
            return y * z; // yz
        } else if constexpr (C == 3) {
            return x * x - y * y;                       // x2-y2
        } else {                                        // C == 4
            return real_t(2.0) * z * z - x * x - y * y; // z2
        }
    }

    template<int C0, int C1, int C2>
    __global__ void evalIntegrand_3DBloackReduce_DDDD_81_components_all_samples(
        int n, real_t hxyz, int pitchX, int blocksXY, int nr, int nl,
        const real2_t* __restrict__ r_grid_dev, const real4_t* __restrict__ l_grid_dev,
        const real2_t* __restrict__ distance_pair_grid, real_t* __restrict__ d_out /* length 81 */
    ) {
        const int global_block = blockIdx.x;

        const int sample = global_block / blocksXY;
        const int bxy    = global_block - sample * blocksXY;

        if (sample >= nr * nl) return;

        const int i = sample % nr;
        const int j = sample / nr;

        const real2_t rg = __ldg(r_grid_dev + i);
        const real4_t lg = l_grid_dev[j];

        const real_t r  = rg.x;
        const real_t rw = rg.y;
        const real_t lw = lg.w;

        const real_t rlx = r * lg.x;
        const real_t rly = r * lg.y;
        const real_t rlz = r * lg.z;

        const real_t sample_weight = rw * lw;

        const int tid     = bxy * blockDim.x + threadIdx.x;
        const int totalXY = n * n;

        real_t acc[81] = {real_t(0.0)};

        if (tid < totalXY) {
            const int y = tid / n;
            const int x = tid - y * n;

            const int pitchXY = pitchX * n;
            const int base    = y * pitchX + x;

            const real3_t c1 = reinterpret_cast<const real3_t*>(d_c + 0)[0];
            const real3_t c2 = reinterpret_cast<const real3_t*>(d_c + 3)[0];
            const real3_t c3 = reinterpret_cast<const real3_t*>(d_c + 6)[0];
            const real3_t c4 = reinterpret_cast<const real3_t*>(d_c + 9)[0];

            const real2_t alpha12 = reinterpret_cast<const real2_t*>(d_alpha + 0)[0];
            const real2_t alpha34 = reinterpret_cast<const real2_t*>(d_alpha + 2)[0];

            real_t w = hxyz;
            if (x == 0 || x == n - 1) w *= real_t(0.5);
            if (y == 0 || y == n - 1) w *= real_t(0.5);

            w *= sample_weight;

            const real_t X = __ldg(&d_x_grid[x]);
            const real_t Y = __ldg(&d_y_grid[y]);

            const real_t x1 = X - c1.x;
            const real_t y1 = Y - c1.y;

            const real_t x2 = X - c2.x;
            const real_t y2 = Y - c2.y;

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

                const real_t D1_0 = d_component<C0>(x1, y1, z1);
                const real_t D1_1 = d_component<C1>(x1, y1, z1);
                const real_t D1_2 = d_component<C2>(x1, y1, z1);

                const real_t D2_0 = d_component<C0>(x2, y2, z2);
                const real_t D2_1 = d_component<C1>(x2, y2, z2);
                const real_t D2_2 = d_component<C2>(x2, y2, z2);

                const real_t D3_0 = d_component<C0>(x3, y3, z3);
                const real_t D3_1 = d_component<C1>(x3, y3, z3);
                const real_t D3_2 = d_component<C2>(x3, y3, z3);

                const real_t D4_0 = d_component<C0>(x4, y4, z4);
                const real_t D4_1 = d_component<C1>(x4, y4, z4);
                const real_t D4_2 = d_component<C2>(x4, y4, z4);

                const real_t L0 = D1_0 * D2_0;
                const real_t L1 = D1_0 * D2_1;
                const real_t L2 = D1_0 * D2_2;

                const real_t L3 = D1_1 * D2_0;
                const real_t L4 = D1_1 * D2_1;
                const real_t L5 = D1_1 * D2_2;

                const real_t L6 = D1_2 * D2_0;
                const real_t L7 = D1_2 * D2_1;
                const real_t L8 = D1_2 * D2_2;

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
            for (int q = 0; q < 81; ++q) {
                acc[q] *= w;
            }
        }

        using BlockReduce = cub::BlockReduce<real_t, THREADS_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;

        real_t block_sum[81];

#pragma unroll
        for (int q = 0; q < 81; ++q) {
            block_sum[q] = BlockReduce(temp).Sum(acc[q]);
            if (q != 80) __syncthreads();
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int q = 0; q < 81; ++q) {
                atomicAdd(d_out + q, block_sum[q]);
            }
        }
    }

    template __global__ void evalIntegrand_3DBloackReduce_DDDD_81_components_all_samples<0, 1, 2>(
        int n, real_t hxyz, int pitchX, int blocksXY, int nr, int nl,
        const real2_t* __restrict__ r_grid_dev, const real4_t* __restrict__ l_grid_dev,
        const real2_t* __restrict__ distance_pair_grid, real_t* __restrict__ d_out);

    template __global__ void evalIntegrand_3DBloackReduce_DDDD_81_components_all_samples<0, 1, 3>(
        int n, real_t hxyz, int pitchX, int blocksXY, int nr, int nl,
        const real2_t* __restrict__ r_grid_dev, const real4_t* __restrict__ l_grid_dev,
        const real2_t* __restrict__ distance_pair_grid, real_t* __restrict__ d_out);

    template __global__ void evalIntegrand_3DBloackReduce_DDDD_81_components_all_samples<1, 2, 3>(
        int n, real_t hxyz, int pitchX, int blocksXY, int nr, int nl,
        const real2_t* __restrict__ r_grid_dev, const real4_t* __restrict__ l_grid_dev,
        const real2_t* __restrict__ distance_pair_grid, real_t* __restrict__ d_out);

    template __global__ void evalIntegrand_3DBloackReduce_DDDD_81_components_all_samples<0, 3, 4>(
        int n, real_t hxyz, int pitchX, int blocksXY, int nr, int nl,
        const real2_t* __restrict__ r_grid_dev, const real4_t* __restrict__ l_grid_dev,
        const real2_t* __restrict__ distance_pair_grid, real_t* __restrict__ d_out);

    template __global__ void evalIntegrand_3DBloackReduce_DDDD_81_components_all_samples<1, 3, 4>(
        int n, real_t hxyz, int pitchX, int blocksXY, int nr, int nl,
        const real2_t* __restrict__ r_grid_dev, const real4_t* __restrict__ l_grid_dev,
        const real2_t* __restrict__ distance_pair_grid, real_t* __restrict__ d_out);

    // compile time helper to get the value of a specific D pair component product
    template<int PAIR>
    __device__ __forceinline__ real_t d_pair_value(real_t xa, real_t ya, real_t za, real_t xb,
                                                   real_t yb, real_t zb) {
        static_assert(PAIR >= 0 && PAIR < 25);

        constexpr int ca = PAIR / 5;
        constexpr int cb = PAIR % 5;

        return d_component<ca>(xa, ya, za) * d_component<cb>(xb, yb, zb);
    }

    template<int BEGIN, int COUNT, int POS = 0>
    struct FillPairs {
        __device__ __forceinline__ static void left(real_t* out, real_t x1, real_t y1, real_t z1,
                                                    real_t x2, real_t y2, real_t z2) {
            out[POS] = d_pair_value<BEGIN + POS>(x1, y1, z1, x2, y2, z2);
            FillPairs<BEGIN, COUNT, POS + 1>::left(out, x1, y1, z1, x2, y2, z2);
        }

        __device__ __forceinline__ static void right(real_t* out, real_t x3, real_t y3, real_t z3,
                                                     real_t x4, real_t y4, real_t z4) {
            out[POS] = d_pair_value<BEGIN + POS>(x3, y3, z3, x4, y4, z4);
            FillPairs<BEGIN, COUNT, POS + 1>::right(out, x3, y3, z3, x4, y4, z4);
        }
    };

    template<int BEGIN, int COUNT>
    struct FillPairs<BEGIN, COUNT, COUNT> {
        __device__ __forceinline__ static void left(real_t*, real_t, real_t, real_t, real_t, real_t,
                                                    real_t) {}

        __device__ __forceinline__ static void right(real_t*, real_t, real_t, real_t, real_t,
                                                     real_t, real_t) {}
    };

    template<int LEFT_BEGIN, int LEFT_COUNT, int RIGHT_BEGIN, int RIGHT_COUNT>
    __global__ void evalIntegrand_3DBloackReduce_DDDD_tile_all_samples(
        int n, real_t hxyz, int pitchX, int blocksXY, int nr, int nl,
        const real2_t* __restrict__ r_grid_dev, const real4_t* __restrict__ l_grid_dev,
        const real2_t* __restrict__ distance_pair_grid,
        real_t* __restrict__ d_out // length LEFT_COUNT * RIGHT_COUNT
    ) {
        static_assert(LEFT_COUNT <= 9);
        static_assert(RIGHT_COUNT <= 9);
        static_assert(LEFT_BEGIN >= 0 && LEFT_BEGIN + LEFT_COUNT <= 25);
        static_assert(RIGHT_BEGIN >= 0 && RIGHT_BEGIN + RIGHT_COUNT <= 25);

        constexpr int OUT_COUNT = LEFT_COUNT * RIGHT_COUNT;

        const int global_block = blockIdx.x;

        const int sample = global_block / blocksXY;
        const int bxy    = global_block - sample * blocksXY;

        if (sample >= nr * nl) return;

        const int i = sample % nr;
        const int j = sample / nr;

        const real2_t rg = __ldg(r_grid_dev + i);
        const real4_t lg = l_grid_dev[j];

        const real_t r  = rg.x;
        const real_t rw = rg.y;
        const real_t lw = lg.w;

        const real_t rlx = r * lg.x;
        const real_t rly = r * lg.y;
        const real_t rlz = r * lg.z;

        const real_t sample_weight = rw * lw;

        const int tid     = bxy * blockDim.x + threadIdx.x;
        const int totalXY = n * n;

        real_t acc[OUT_COUNT] = {real_t(0.0)};

        if (tid < totalXY) {
            const int y = tid / n;
            const int x = tid - y * n;

            const int pitchXY = pitchX * n;
            const int base    = y * pitchX + x;

            const real3_t c1 = reinterpret_cast<const real3_t*>(d_c + 0)[0];
            const real3_t c2 = reinterpret_cast<const real3_t*>(d_c + 3)[0];
            const real3_t c3 = reinterpret_cast<const real3_t*>(d_c + 6)[0];
            const real3_t c4 = reinterpret_cast<const real3_t*>(d_c + 9)[0];

            const real2_t alpha12 = reinterpret_cast<const real2_t*>(d_alpha + 0)[0];
            const real2_t alpha34 = reinterpret_cast<const real2_t*>(d_alpha + 2)[0];

            real_t w = hxyz;
            if (x == 0 || x == n - 1) w *= real_t(0.5);
            if (y == 0 || y == n - 1) w *= real_t(0.5);
            w *= sample_weight;

            const real_t X = __ldg(&d_x_grid[x]);
            const real_t Y = __ldg(&d_y_grid[y]);

            const real_t x1 = X - c1.x;
            const real_t y1 = Y - c1.y;

            const real_t x2 = X - c2.x;
            const real_t y2 = Y - c2.y;

            const real_t x3 = X - c3.x + rlx;
            const real_t y3 = Y - c3.y + rly;

            const real_t x4 = X - c4.x + rlx;
            const real_t y4 = Y - c4.y + rly;

            for (int k = 0; k < n; ++k) {
                const real_t Z = __ldg(&d_z_grid[k]);

                const real2_t d12       = __ldg(distance_pair_grid + (k * pitchXY + base));
                const real_t  d1_cached = d12.x;
                const real_t  d2_cached = d12.y;
                const real_t  a12       = alpha12.x * d1_cached + alpha12.y * d2_cached;

                const real_t z1 = Z - c1.z;
                const real_t z2 = Z - c2.z;
                const real_t z3 = Z - c3.z + rlz;
                const real_t z4 = Z - c4.z + rlz;

                const real_t d3_cached = norm(x3, y3, z3);
                const real_t d4_cached = norm(x4, y4, z4);

                const real_t expo = r - (a12 + alpha34.x * d3_cached + alpha34.y * d4_cached);
                const real_t wz   = (k == 0 || k == n - 1) ? real_t(0.5) : real_t(1.0);
                const real_t T    = exp_fn(expo) * wz;

                // Full 5 D components on each center:
                // 0 = xy
                // 1 = xz
                // 2 = yz
                // 3 = x2-y2
                // 4 = z2 = 2z^2 - x^2 - y^2

                real_t L[LEFT_COUNT];
                real_t R[RIGHT_COUNT];

                FillPairs<LEFT_BEGIN, LEFT_COUNT>::left(L, x1, y1, z1, x2, y2, z2);
                FillPairs<RIGHT_BEGIN, RIGHT_COUNT>::right(R, x3, y3, z3, x4, y4, z4);

#pragma unroll
                for (int li = 0; li < LEFT_COUNT; ++li) {
                    const real_t TL = T * L[li];
#pragma unroll
                    for (int ri = 0; ri < RIGHT_COUNT; ++ri) {
                        acc[li * RIGHT_COUNT + ri] += TL * R[ri];
                    }
                }
            }

#pragma unroll
            for (int q = 0; q < OUT_COUNT; ++q) {
                acc[q] *= w;
            }
        }

        using BlockReduce = cub::BlockReduce<real_t, THREADS_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;

        real_t block_sum[OUT_COUNT];

#pragma unroll
        for (int q = 0; q < OUT_COUNT; ++q) {
            block_sum[q] = BlockReduce(temp).Sum(acc[q]);
            if (q != OUT_COUNT - 1) __syncthreads();
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int q = 0; q < OUT_COUNT; ++q) {
                atomicAdd(d_out + q, block_sum[q]);
            }
        }
    }

    template __global__ void evalIntegrand_3DBloackReduce_DDDD_tile_all_samples<0, 9, 0, 9>(
        int, real_t, int, int, int, int, const real2_t*, const real4_t*, const real2_t*, real_t*);

    template __global__ void evalIntegrand_3DBloackReduce_DDDD_tile_all_samples<0, 9, 9, 9>(
        int, real_t, int, int, int, int, const real2_t*, const real4_t*, const real2_t*, real_t*);

    template __global__ void evalIntegrand_3DBloackReduce_DDDD_tile_all_samples<0, 9, 18, 7>(
        int, real_t, int, int, int, int, const real2_t*, const real4_t*, const real2_t*, real_t*);

    template __global__ void evalIntegrand_3DBloackReduce_DDDD_tile_all_samples<9, 9, 0, 9>(
        int, real_t, int, int, int, int, const real2_t*, const real4_t*, const real2_t*, real_t*);

    template __global__ void evalIntegrand_3DBloackReduce_DDDD_tile_all_samples<9, 9, 9, 9>(
        int, real_t, int, int, int, int, const real2_t*, const real4_t*, const real2_t*, real_t*);

    template __global__ void evalIntegrand_3DBloackReduce_DDDD_tile_all_samples<9, 9, 18, 7>(
        int, real_t, int, int, int, int, const real2_t*, const real4_t*, const real2_t*, real_t*);

    template __global__ void evalIntegrand_3DBloackReduce_DDDD_tile_all_samples<18, 7, 0, 9>(
        int, real_t, int, int, int, int, const real2_t*, const real4_t*, const real2_t*, real_t*);

    template __global__ void evalIntegrand_3DBloackReduce_DDDD_tile_all_samples<18, 7, 9, 9>(
        int, real_t, int, int, int, int, const real2_t*, const real4_t*, const real2_t*, real_t*);

    template __global__ void evalIntegrand_3DBloackReduce_DDDD_tile_all_samples<18, 7, 18, 7>(
        int, real_t, int, int, int, int, const real2_t*, const real4_t*, const real2_t*, real_t*);
} // namespace cuslater