#include "kernels.cuh"
#include <cub/cub.cuh>
namespace cuslater {
        // placeholder method for error analysis for trapezoidal rule
    __global__ void eval_1111_trap_all_samples(int n, real_t hxyz, int pitchX, int blocksXY, int nr,
                                               int nl, const real2_t* __restrict__ r_grid_dev,
                                               const real4_t* __restrict__ l_grid_dev,
                                               const real2_t* __restrict__ distance_pair_grid,
                                               real_t* __restrict__ d_out) {
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

        real_t local = real_t(0.0);

        if (tid < totalXY) {
            const int y = tid / n;
            const int x = tid - y * n;

            const int pitchXY = pitchX * n;
            const int base    = y * pitchX + x;

            const real3_t c3 = reinterpret_cast<const real3_t*>(d_c + 6)[0];
            const real3_t c4 = reinterpret_cast<const real3_t*>(d_c + 9)[0];

            const real2_t alpha12 = reinterpret_cast<const real2_t*>(d_alpha + 0)[0];
            const real2_t alpha34 = reinterpret_cast<const real2_t*>(d_alpha + 2)[0];

            real_t w = hxyz * sample_weight;

            if (x == 0 || x == n - 1) w *= real_t(0.5);
            if (y == 0 || y == n - 1) w *= real_t(0.5);

            const real_t X = __ldg(&d_x_grid[x]);
            const real_t Y = __ldg(&d_y_grid[y]);

            const real_t x3 = X - c3.x + rlx;
            const real_t y3 = Y - c3.y + rly;
            const real_t x4 = X - c4.x + rlx;
            const real_t y4 = Y - c4.y + rly;

            real_t sum_z = real_t(0.0);

            for (int k = 0; k < n; ++k) {
                const real_t Z = __ldg(&d_z_grid[k]);

                const real2_t d12 = __ldg(distance_pair_grid + (k * pitchXY + base));
                const real_t  d1  = d12.x;
                const real_t  d2  = d12.y;

                const real_t a12 = alpha12.x * d1 + alpha12.y * d2;

                const real_t z3 = Z - c3.z + rlz;
                const real_t z4 = Z - c4.z + rlz;

                const real_t d3 = norm(x3, y3, z3);
                const real_t d4 = norm(x4, y4, z4);

                const real_t expo = r - (a12 + alpha34.x * d3 + alpha34.y * d4);

                const real_t wz = (k == 0 || k == n - 1) ? real_t(0.5) : real_t(1.0);

                sum_z += exp_fn(expo) * wz;
            }

            local = sum_z * w;
        }

        using BlockReduce = cub::BlockReduce<real_t, THREADS_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;

        const real_t block_sum = BlockReduce(temp).Sum(local);

        if (threadIdx.x == 0) {
            atomicAdd(d_out, block_sum);
        }
    }

    // placeholder method for error analysis for simpson's rule
    __global__ void eval_1111_simpson_all_samples(int n, real_t hxyz, int pitchX, int blocksXY, int nr,
                                                  int nl, const real2_t* __restrict__ r_grid_dev,
                                                  const real4_t* __restrict__ l_grid_dev,
                                                  const real2_t* __restrict__ distance_pair_grid,
                                                  real_t* __restrict__ d_out) {
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

        real_t local = real_t(0.0);

        if (tid < totalXY) {
            const int y = tid / n;
            const int x = tid - y * n;

            const int pitchXY = pitchX * n;
            const int base    = y * pitchX + x;

            const real3_t c3 = reinterpret_cast<const real3_t*>(d_c + 6)[0];
            const real3_t c4 = reinterpret_cast<const real3_t*>(d_c + 9)[0];

            const real2_t alpha12 = reinterpret_cast<const real2_t*>(d_alpha + 0)[0];
            const real2_t alpha34 = reinterpret_cast<const real2_t*>(d_alpha + 2)[0];

            const real_t wx = simpson_coeff(x, n);
            const real_t wy = simpson_coeff(y, n);

            real_t w = hxyz * sample_weight * wx * wy / real_t(27.0);

            const real_t X = __ldg(&d_x_grid[x]);
            const real_t Y = __ldg(&d_y_grid[y]);

            const real_t x3 = X - c3.x + rlx;
            const real_t y3 = Y - c3.y + rly;
            const real_t x4 = X - c4.x + rlx;
            const real_t y4 = Y - c4.y + rly;

            real_t sum_z = real_t(0.0);

            for (int k = 0; k < n; ++k) {
                const real_t Z = __ldg(&d_z_grid[k]);

                const real2_t d12 = __ldg(distance_pair_grid + (k * pitchXY + base));
                const real_t  d1  = d12.x;
                const real_t  d2  = d12.y;

                const real_t a12 = alpha12.x * d1 + alpha12.y * d2;

                const real_t z3 = Z - c3.z + rlz;
                const real_t z4 = Z - c4.z + rlz;

                const real_t d3 = norm(x3, y3, z3);
                const real_t d4 = norm(x4, y4, z4);

                const real_t expo = r - (a12 + alpha34.x * d3 + alpha34.y * d4);

                const real_t wz = simpson_coeff(k, n);

                sum_z += exp_fn(expo) * wz;
            }

            local = sum_z * w;
        }

        using BlockReduce = cub::BlockReduce<real_t, THREADS_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;

        const real_t block_sum = BlockReduce(temp).Sum(local);

        if (threadIdx.x == 0) {
            atomicAdd(d_out, block_sum);
        }
    }
}