#include "kernels.cuh"

namespace cuslater {
    __global__ void evalIntegrand_3DBloackReduce_SSSS_SSSP_SSPS_SPSS_all_samples_simpson(
        int n, real_t hxyz, int pitchX, int blocksXY, int nr, int nl,
        const real2_t* __restrict__ r_grid_dev, const real4_t* __restrict__ l_grid_dev,
        const real2_t* __restrict__ distance_pair_grid,
        real_t* __restrict__ d_out // length 88, final weighted sums
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

            // Simpson x/y coefficient.
            const real_t wx = simpson_coeff(x, n);
            const real_t wy = simpson_coeff(y, n);

            // Since hxyz = hx * hy * hz, Simpson contributes /3 in each dimension.
            real_t w = hxyz * sample_weight * wx * wy / real_t(27.0);

            const real_t X = __ldg(&d_x_grid[x]);
            const real_t Y = __ldg(&d_y_grid[y]);

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

                const real_t z2 = Z - c2.z;
                const real_t z3 = Z - c3.z + rlz;
                const real_t z4 = Z - c4.z + rlz;

                const real_t d3 = norm(x3, y3, z3);
                const real_t d4 = norm(x4, y4, z4);

                const real_t expo = r - (a12 + alpha34.x * d3 + alpha34.y * d4);

                // Simpson z coefficient.
                const real_t wz = simpson_coeff(k, n);
                const real_t T  = exp_fn(expo) * wz;

                const real_t L0 = real_t(1.0);
                const real_t L1 = d1;
                const real_t L2 = d2;
                const real_t L3 = d1 * d2;

                const real_t RSS0 = real_t(1.0);
                const real_t RSS1 = d3;
                const real_t RSS2 = d4;
                const real_t RSS3 = d3 * d4;

                const real_t RSP0 = x4;
                const real_t RSP1 = y4;
                const real_t RSP2 = z4;
                const real_t RSP3 = d3 * x4;
                const real_t RSP4 = d3 * y4;
                const real_t RSP5 = d3 * z4;

                const real_t RPS0 = x3;
                const real_t RPS1 = y3;
                const real_t RPS2 = z3;
                const real_t RPS3 = x3 * d4;
                const real_t RPS4 = y3 * d4;
                const real_t RPS5 = z3 * d4;

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
            for (int q = 0; q < 88; ++q) {
                acc[q] *= w;
            }
        }

        using BlockReduce = cub::BlockReduce<real_t, THREADS_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;

        real_t block_sum[88];

#pragma unroll
        for (int q = 0; q < 88; ++q) {
            block_sum[q] = BlockReduce(temp).Sum(acc[q]);
            if (q != 87) __syncthreads();
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int q = 0; q < 88; ++q) {
                atomicAdd(d_out + q, block_sum[q]);
            }
        }
    }

} // namespace cuslater