// updated april 2025
//  Created by gkluhana on 26/03/24.
//
#include "evalIntegral.h"
#include <algorithm>
#include <cub/cub.cuh>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
const double pi = 3.14159265358979323846;
#define THREADS_PER_BLOCK 256

__constant__ real_t d_c[12];
__constant__ real_t d_alpha[4];

__constant__ real_t d_x_grid[500];
__constant__ real_t d_y_grid[500];
__constant__ real_t d_z_grid[500];

namespace cuslater {

    __global__ void evalIntegrand_3DBloackReduce(int n, real_t hxyz, real_t rlx,
                                                 real_t rly, real_t rlz, real_t r,
                                                 real_t* __restrict__ block_sums) {
        int idx_flat = (blockIdx.x * blockDim.x + threadIdx.x);
        int totalXY  = n * n;
        int y        = idx_flat / n;
        int x        = idx_flat % n;

        // load to registers, or at least to shared memory
        // to avoid global memory access
        real3_t c1    = reinterpret_cast<real3_t*>(d_c)[0];
        real3_t c2    = reinterpret_cast<real3_t*>(d_c + 3)[0];
        real3_t c3    = reinterpret_cast<real3_t*>(d_c + 6)[0];
        real3_t c4    = reinterpret_cast<real3_t*>(d_c + 9)[0];
        real4_t alpha = reinterpret_cast<real4_t*>(d_alpha)[0];

        // inform the compiler this is unlikely (__builtin_expect(cond, 0))
        if (__builtin_expect(x == 0 || x == n - 1, 0)) [[unlikely]] {
            hxyz *= 0.5; // half weight at endpoints
        }
        if (__builtin_expect(y == 0 || y == n - 1, 0)) [[unlikely]] {
            hxyz *= 0.5; // half weight at endpoints
        }

        real_t local = 0.f;
        if (__builtin_expect(idx_flat < totalXY, 1)) [[likely]] {
            real_t xvalue = __ldg(&d_x_grid[x]);
            real_t yvalue = __ldg(&d_y_grid[y]);

            real_t xdiffc_1 = xvalue - c1.x;
            real_t xdiffc_2 = xvalue - c2.x;

            real_t ydiffc_1 = yvalue - c1.y;
            real_t ydiffc_2 = yvalue - c2.y;

            real_t xdiffc_3 = xvalue - c3.x + rlx;
            real_t xdiffc_4 = xvalue - c4.x + rlx;

            real_t ydiffc_3 = yvalue - c3.y + rly;
            real_t ydiffc_4 = yvalue - c4.y + rly;

            real_t v = 0.f;

            {
                real_t zvalue   = __ldg(&d_z_grid[0]);
                real_t zdiffc_1 = zvalue - c1.z;
                real_t zdiffc_2 = zvalue - c2.z;
                real_t zdiffc_3 = zvalue - c3.z + rlz;
                real_t zdiffc_4 = zvalue - c4.z + rlz;

                real_t term1    = alpha.x * norm(xdiffc_1, ydiffc_1, zdiffc_1);
                real_t term2    = alpha.y * norm(xdiffc_2, ydiffc_2, zdiffc_2);
                real_t term3    = alpha.z * norm(xdiffc_3, ydiffc_3, zdiffc_3);
                real_t term4    = alpha.w * norm(xdiffc_4, ydiffc_4, zdiffc_4);
                real_t exponent = -term1 - term2 - term3 - term4 + r;
                v += __expf(exponent) * 0.5f;
            } // first run
            int k = 1; // for k = 1 to n - 1
            for (; k < n - 4; k += 4) {
                real4_t zvalue = reinterpret_cast<real4_t*>(&d_z_grid[k])[0];

                real_t zdiffc_1 = zvalue.x - c1.z;
                real_t zdiffc_2 = zvalue.x - c2.z;
                real_t zdiffc_3 = zvalue.x - c3.z + rlz;
                real_t zdiffc_4 = zvalue.x - c4.z + rlz;
                real_t term1    = alpha.x * norm(xdiffc_1, ydiffc_1, zdiffc_1);
                real_t term2    = alpha.y * norm(xdiffc_2, ydiffc_2, zdiffc_2);
                real_t term3    = alpha.z * norm(xdiffc_3, ydiffc_3, zdiffc_3);
                real_t term4    = alpha.w * norm(xdiffc_4, ydiffc_4, zdiffc_4);
                real_t exponent = -term1 - term2 - term3 - term4 + r;
                v += __expf(exponent);

                zdiffc_1 = zvalue.y - c1.z;
                zdiffc_2 = zvalue.y - c2.z;
                zdiffc_3 = zvalue.y - c3.z + rlz;
                zdiffc_4 = zvalue.y - c4.z + rlz;
                term1    = alpha.x * norm(xdiffc_1, ydiffc_1, zdiffc_1);
                term2    = alpha.y * norm(xdiffc_2, ydiffc_2, zdiffc_2);
                term3    = alpha.z * norm(xdiffc_3, ydiffc_3, zdiffc_3);
                term4    = alpha.w * norm(xdiffc_4, ydiffc_4, zdiffc_4);
                exponent = -term1 - term2 - term3 - term4 + r;
                v += __expf(exponent);

                zdiffc_1 = zvalue.z - c1.z;
                zdiffc_2 = zvalue.z - c2.z;
                zdiffc_3 = zvalue.z - c3.z + rlz;
                zdiffc_4 = zvalue.z - c4.z + rlz;
                term1    = alpha.x * norm(xdiffc_1, ydiffc_1, zdiffc_1);
                term2    = alpha.y * norm(xdiffc_2, ydiffc_2, zdiffc_2);
                term3    = alpha.z * norm(xdiffc_3, ydiffc_3, zdiffc_3);
                term4    = alpha.w * norm(xdiffc_4, ydiffc_4, zdiffc_4);
                exponent = -term1 - term2 - term3 - term4 + r;
                v += __expf(exponent);

                zdiffc_1 = zvalue.w - c1.z;
                zdiffc_2 = zvalue.w - c2.z;
                zdiffc_3 = zvalue.w - c3.z + rlz;
                zdiffc_4 = zvalue.w - c4.z + rlz;
                term1    = alpha.x * norm(xdiffc_1, ydiffc_1, zdiffc_1);
                term2    = alpha.y * norm(xdiffc_2, ydiffc_2, zdiffc_2);
                term3    = alpha.z * norm(xdiffc_3, ydiffc_3, zdiffc_3);
                term4    = alpha.w * norm(xdiffc_4, ydiffc_4, zdiffc_4);
                exponent = -term1 - term2 - term3 - term4 + r;
                v += __expf(exponent);
            }
            for (; k < n - 1; ++k) { // clear up the remainder
                real_t zvalue   = __ldg(&d_z_grid[k]);
                real_t zdiffc_1 = zvalue - c1.z;
                real_t zdiffc_2 = zvalue - c2.z;
                real_t zdiffc_3 = zvalue - c3.z + rlz;
                real_t zdiffc_4 = zvalue - c4.z + rlz;

                real_t term1    = alpha.x * norm(xdiffc_1, ydiffc_1, zdiffc_1);
                real_t term2    = alpha.y * norm(xdiffc_2, ydiffc_2, zdiffc_2);
                real_t term3    = alpha.z * norm(xdiffc_3, ydiffc_3, zdiffc_3);
                real_t term4    = alpha.w * norm(xdiffc_4, ydiffc_4, zdiffc_4);
                real_t exponent = -term1 - term2 - term3 - term4 + r;
                v += __expf(exponent);
            }
            { // handle last
                real_t zvalue   = __ldg(&d_z_grid[k]);
                real_t zdiffc_1 = zvalue - c1.z;
                real_t zdiffc_2 = zvalue - c2.z;
                real_t zdiffc_3 = zvalue - c3.z + rlz;
                real_t zdiffc_4 = zvalue - c4.z + rlz;

                real_t term1    = alpha.x * norm(xdiffc_1, ydiffc_1, zdiffc_1);
                real_t term2    = alpha.y * norm(xdiffc_2, ydiffc_2, zdiffc_2);
                real_t term3    = alpha.z * norm(xdiffc_3, ydiffc_3, zdiffc_3);
                real_t term4    = alpha.w * norm(xdiffc_4, ydiffc_4, zdiffc_4);
                real_t exponent = -term1 - term2 - term3 - term4 + r;
                v += __expf(exponent) * 0.5f;
            }
            local += v * hxyz; // multiply by the volume element
        }

        using BlockReduce = cub::BlockReduce<real_t, THREADS_PER_BLOCK>;

        __shared__ typename BlockReduce::TempStorage temp;

        real_t block_sum = BlockReduce(temp).Sum(local);
        if (__builtin_expect(threadIdx.x == 0, 0)) [[unlikely]]
            block_sums[blockIdx.x] = block_sum;
    }

    bool checkZero(real_t* c, real_t* alpha) {

        real_t normdiff13 = sqrt((c[0] - c[6]) * (c[0] - c[6]) + (c[1] - c[7]) * (c[1] - c[7])
                                 + (c[2] - c[8]) * (c[2] - c[8]));
        real_t normdiff24 = sqrt((c[3] - c[9]) * (c[3] - c[9])
                                 + (c[4] - c[10]) * (c[4] - c[10])
                                 + (c[5] - c[11]) * (c[5] - c[11]));

        real_t cond = std::min(alpha[0], alpha[2]) * normdiff13
                    + std::min(alpha[1], alpha[3]) * normdiff24;
        real_t r0              = 1;
        int    inv_machine_eps = 1e8;
        if (cond > log(r0 * inv_machine_eps)) {
            return true;
        }
        return false;
    }

    double evaluateFourCenterIntegral(real_t* c, real_t* alpha, vector<real2_t>& r_grid,
                                      vector<real4_t>& l_grid, int n, double tol,
                                      bool check_zero_cond, Metric* metric) {
        if (check_zero_cond && checkZero(c, alpha)) {
            return 0.0;
        }

        const int nr = r_grid.size();
        const int nl = l_grid.size();

        // Now define physical domain boundaries
        real_t dx = std::abs(c[0] - c[3]);
        real_t dy = std::abs(c[1] - c[4]);
        real_t dz = std::abs(c[2] - c[5]);
        real_t lx = 18.0 + dx;
        real_t ly = 18.0 + dy;
        real_t lz = 18.0 + dz;
        real_t mx = (c[0] + c[3]) / 2.0;
        real_t my = (c[1] + c[4]) / 2.0;
        real_t mz = (c[2] + c[5]) / 2.0;
        real_t ax = mx - (lx / 2.0);
        real_t bx = mx + (lx / 2.0);
        real_t ay = my - (ly / 2.0);
        real_t by = my + (ly / 2.0);
        real_t az = mz - (lz / 2.0);
        real_t bz = mz + (lz / 2.0);

        // Generate trapezoidal nodes and weights for x1, y1, z1
        std::vector<real_t> x1_nodes(n);
        std::vector<real_t> x1_weights(n);
        real_t              hx = (bx - ax) / (n - 1);

        std::vector<real_t> y1_nodes(n);
        std::vector<real_t> y1_weights(n);
        real_t              hy = (by - ay) / (n - 1);

        std::vector<real_t> z1_nodes(n);
        std::vector<real_t> z1_weights(n);
        real_t              hz = (bz - az) / (n - 1);

        for (int i = 0; i < n; ++i) {
            x1_nodes[i] = ax + i * hx;
        }

        for (int i = 0; i < n; ++i) {
            y1_nodes[i] = ay + i * hy;
        }

        for (int i = 0; i < n; ++i) {
            z1_nodes[i] = az + i * hz;
        }

        constexpr int threads = THREADS_PER_BLOCK;               // Max threads per block
        int           blocks  = (n * n + threads - 1) / threads; // Max blocks, better if
                                                                 // multiple of SM = 80
        cudaMemcpyToSymbol(d_c, c, 12 * sizeof(real_t));
        cudaMemcpyToSymbol(d_alpha, alpha, 4 * sizeof(real_t));
        cudaMemcpyToSymbol(d_x_grid, x1_nodes.data(), n * sizeof(real_t));
        cudaMemcpyToSymbol(d_y_grid, y1_nodes.data(), n * sizeof(real_t));
        cudaMemcpyToSymbol(d_z_grid, z1_nodes.data(), n * sizeof(real_t));

        thrust::device_vector<real_t> d_block_sums(blocks);
        thrust::host_vector<real_t>   block_sums(blocks);

        double                    sum       = 0.0f;
        double                    delta_sum = 0.0f;
        int                       r_skipped = 0;
        std::chrono::microseconds duration(0);

        real_t hxyz = hx * hy * hz;

        auto grand_start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < nl; ++j) {
            for (int i = 0; i < nr; ++i) {
                auto   start = std::chrono::high_resolution_clock::now();
                real_t r     = r_grid[i].x;
                evalIntegrand_3DBloackReduce<<<blocks, THREADS_PER_BLOCK>>>(
                    n, hxyz, r * l_grid[j].x, r * l_grid[j].y, r * l_grid[j].z, r,
                    thrust::raw_pointer_cast(d_block_sums.data()));
                auto end = std::chrono::high_resolution_clock::now();
                duration += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                block_sums = d_block_sums;
                delta_sum  = std::accumulate(block_sums.begin(), block_sums.end(), 0.0);

                sum += delta_sum * r_grid[i].y * l_grid[j].w;
                if (delta_sum < tol) [[unlikely]] {
                    r_skipped += nr - i;
                    break;
                }
            }
        }
        auto grand_end = std::chrono::high_resolution_clock::now();
        auto grand_dur =
            std::chrono::duration_cast<std::chrono::microseconds>(grand_end - grand_start);

        sum *= (4.0 / pi) * std::pow(alpha[0] * alpha[1] * alpha[2] * alpha[3], 1.5);
        if (metric) {
            metric->totalTime          = grand_dur;
            if (nr * nl - r_skipped != 0) {
                metric->avgKernelTime    = duration / (nr * nl - r_skipped);
            } else {
                metric->avgKernelTime    = duration;
            }
            metric->totalKernelTime    = duration;
            metric->totalKernelCalls   = nr * nl - r_skipped;
            metric->skippedLebdevNodes = r_skipped;
            metric->effectiveBandwidth = (202.0 * 4 * n * n + blocks * 4)
                                       / metric->avgKernelTime.count() / 1e3; // GB/s
            metric->totalBlocks     = blocks;
            metric->totalThreads    = blocks * threads;
            metric->totalGridPoints = n * n * n;
            metric->a               = make_real3(ax, ay, az);
            metric->b               = make_real3(bx, by, bz);
        }
        return sum;
    }
} // namespace cuslater
