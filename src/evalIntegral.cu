// updated april 2025
//  Created by gkluhana on 26/03/24.
//
#include "evalIntegral.h"
#include "number.h"
#include <algorithm>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
const double pi = 3.14159265358979323846;
#define THREADS_PER_BLOCK 128

__constant__ real_t d_c[12];
__constant__ real_t d_alpha[4];

__constant__ real_t d_x_grid[500];
__constant__ real_t d_y_grid[500];
__constant__ real_t d_z_grid[500];

namespace cuslater {

    // result layout: idx = z * pitchXY + y * pitchX + x
    // where pitchX pads each row to a multiple of warp-aligned elements.
    __global__ void compute_distance_grid_zmajor(int n, int pitchX, real_t* __restrict__ result) {
        const int idx_xy  = blockIdx.x * blockDim.x + threadIdx.x;
        const int totalXY = n * n;
        if (idx_xy >= totalXY) return;

        const int y = idx_xy / n;
        const int x = idx_xy - y * n;

        const real3_t c1 = reinterpret_cast<const real3_t*>(d_c)[0];
        const real3_t c2 = reinterpret_cast<const real3_t*>(d_c + 3)[0];
        const real2_t a  = reinterpret_cast<const real2_t*>(d_alpha)[0];

        const real_t X = __ldg(&d_x_grid[x]);
        const real_t Y = __ldg(&d_y_grid[y]);

        const int row     = y * pitchX;
        const int pitchXY = pitchX * n;

        for (int z = 0; z < n; ++z) {
            const real_t Z  = __ldg(&d_z_grid[z]);
            const real_t d1 = norm(X - c1.x, Y - c1.y, Z - c1.z);
            const real_t d2 = norm(X - c2.x, Y - c2.y, Z - c2.z);
            const real_t v  = a.x * d1 + a.y * d2;

            result[z * pitchXY + row + x] = v;
        }
    }

    __launch_bounds__(THREADS_PER_BLOCK, 3) __global__
        void evalIntegrand_3DBloackReduce(int n, real_t hxyz, real_t rlx, real_t rly, real_t rlz,
                                          real_t r, int pitchX, real_t* __restrict__ alpha12,
                                          real_t* __restrict__ d_result) {
        int tid     = (blockIdx.x * blockDim.x + threadIdx.x);
        int totalXY = n * n;
        int y       = tid / n;
        int x       = tid - y * n;
        int pitchXY = pitchX * n;
        int rowBase = y * pitchX;
        int base    = rowBase + x;

        real_t local = real_t(0.0);
        if (__builtin_expect(tid < totalXY, 1)) [[likely]] {

            // load to registers, or at least to shared memory
            // to avoid global memory access
            real3_t c3    = reinterpret_cast<real3_t*>(d_c + 6)[0];
            real3_t c4    = reinterpret_cast<real3_t*>(d_c + 9)[0];
            real2_t alpha = reinterpret_cast<real2_t*>(d_alpha + 2)[0];

            // inform the compiler this is unlikely (__builtin_expect(cond, 0))
            if (__builtin_expect(x == 0 || x == n - 1, 0)) [[unlikely]] {
                hxyz *= 0.5; // half weight at endpoints
            }
            if (__builtin_expect(y == 0 || y == n - 1, 0)) [[unlikely]] {
                hxyz *= 0.5; // half weight at endpoints
            }
            // NOTE: BELOW z loop does not have half weights at the endpoints (to be done later)

            real_t xvalue = __ldg(&d_x_grid[x]);
            real_t yvalue = __ldg(&d_y_grid[y]);

            real_t xdiffc_3 = xvalue - c3.x + rlx;
            real_t xdiffc_4 = xvalue - c4.x + rlx;

            real_t ydiffc_3 = yvalue - c3.y + rly;
            real_t ydiffc_4 = yvalue - c4.y + rly;

            real_t v0 = 0, v1 = 0, v2 = 0, v3 = 0;

            constexpr int UNROLL = 8; // maybe 4 for double?
            int           k      = 0;

            // prefetch first UNROLL alpha12
            real_t a[UNROLL];
#pragma unroll
            for (int t = 0; t < UNROLL && t < n; ++t) {
                a[t] = __ldg(&alpha12[(k + t) * pitchXY + base]);
            }

            for (; k + UNROLL - 1 < n; k += UNROLL) {
                // compute on the prefetched
                real_t z0 = d_z_grid[k + 0], z1 = d_z_grid[k + 1], z2 = d_z_grid[k + 2],
                       z3 = d_z_grid[k + 3];
                real_t z4 = d_z_grid[k + 4], z5 = d_z_grid[k + 5], z6 = d_z_grid[k + 6],
                       z7 = d_z_grid[k + 7];

                v0 += exp_fn(r
                             - (a[0] + alpha.x * norm(xdiffc_3, ydiffc_3, z0 - c3.z + rlz)
                                + alpha.y * norm(xdiffc_4, ydiffc_4, z0 - c4.z + rlz)));
                v1 += exp_fn(r
                             - (a[1] + alpha.x * norm(xdiffc_3, ydiffc_3, z1 - c3.z + rlz)
                                + alpha.y * norm(xdiffc_4, ydiffc_4, z1 - c4.z + rlz)));
                v2 += exp_fn(r
                             - (a[2] + alpha.x * norm(xdiffc_3, ydiffc_3, z2 - c3.z + rlz)
                                + alpha.y * norm(xdiffc_4, ydiffc_4, z2 - c4.z + rlz)));
                v3 += exp_fn(r
                             - (a[3] + alpha.x * norm(xdiffc_3, ydiffc_3, z3 - c3.z + rlz)
                                + alpha.y * norm(xdiffc_4, ydiffc_4, z3 - c4.z + rlz)));

                v0 += exp_fn(r
                             - (a[4] + alpha.x * norm(xdiffc_3, ydiffc_3, z4 - c3.z + rlz)
                                + alpha.y * norm(xdiffc_4, ydiffc_4, z4 - c4.z + rlz)));
                v1 += exp_fn(r
                             - (a[5] + alpha.x * norm(xdiffc_3, ydiffc_3, z5 - c3.z + rlz)
                                + alpha.y * norm(xdiffc_4, ydiffc_4, z5 - c4.z + rlz)));
                v2 += exp_fn(r
                             - (a[6] + alpha.x * norm(xdiffc_3, ydiffc_3, z6 - c3.z + rlz)
                                + alpha.y * norm(xdiffc_4, ydiffc_4, z6 - c4.z + rlz)));
                v3 += exp_fn(r
                             - (a[7] + alpha.x * norm(xdiffc_3, ydiffc_3, z7 - c3.z + rlz)
                                + alpha.y * norm(xdiffc_4, ydiffc_4, z7 - c4.z + rlz)));

                // prefetch the NEXT UNROLL alpha12 early
                if (k + 2 * UNROLL - 1 < n) {
#pragma unroll
                    for (int t = 0; t < UNROLL; ++t) {
                        a[t] = __ldg(&alpha12[(k + UNROLL + t) * pitchXY + base]);
                    }
                }
            }

            // tail
            for (; k < n; ++k) {
                int    idx = k * pitchXY + base;
                real_t z   = d_z_grid[k];
                real_t a12 = __ldg(&alpha12[idx]);
                real_t t3  = alpha.x * norm(xdiffc_3, ydiffc_3, z - c3.z + rlz);
                real_t t4  = alpha.y * norm(xdiffc_4, ydiffc_4, z - c4.z + rlz);
                v0 += exp_fn(r - (a12 + t3 + t4));
            }

            real_t v = ((v0 + v1) + (v2 + v3));
            local += v * hxyz;
        }

        using BlockReduce = cub::BlockReduce<real_t, THREADS_PER_BLOCK>;

        __shared__ typename BlockReduce::TempStorage temp;

        real_t block_sum = BlockReduce(temp).Sum(local);
        if (__builtin_expect(threadIdx.x == 0, 0)) [[unlikely]]
            atomicAdd(d_result, block_sum);
    }

    bool checkZero(real_t* c, real_t* alpha) {

        real_t normdiff13 = sqrt((c[0] - c[6]) * (c[0] - c[6]) + (c[1] - c[7]) * (c[1] - c[7])
                                 + (c[2] - c[8]) * (c[2] - c[8]));
        real_t normdiff24 = sqrt((c[3] - c[9]) * (c[3] - c[9]) + (c[4] - c[10]) * (c[4] - c[10])
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

        // Build cache once per grid
        int warpBytes  = 128;
        int alignElems = warpBytes / sizeof(real_t); // 32 for float, 16 for double
        int pitchX     = ((n + alignElems - 1) / alignElems) * alignElems; // padded row length in
                                                                           // elements

        size_t pitchXY = size_t(pitchX) * n;
        size_t total   = size_t(n) * pitchXY; // z * (y * pitchX)

        real_t* d_distance_grid;
        cudaMalloc(&d_distance_grid, total * sizeof(real_t));
        compute_distance_grid_zmajor<<<(n * n + threads - 1) / threads, threads>>>(n, pitchX,
                                                                                   d_distance_grid);
        cudaFuncSetCacheConfig(evalIntegrand_3DBloackReduce, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(compute_distance_grid_zmajor, cudaFuncCachePreferL1);

        double                    sum       = 0.0f;
        int                       r_skipped = 0;
        std::chrono::microseconds duration(0);

        real_t  hxyz     = hx * hy * hz;
        real_t* d_result = nullptr; // NOTE: this approach does not show a big advantage over using
                                    // thrust::device_vector, and it shows slightly different result
                                    // (todo: investigate why)
        cudaMalloc(&d_result, sizeof(real_t));

        auto grand_start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < nl; ++j) {
            for (int i = 0; i < nr; ++i) {
                auto   start = std::chrono::high_resolution_clock::now();
                real_t r     = r_grid[i].x;
                cudaMemsetAsync(d_result, 0, sizeof(real_t));
                evalIntegrand_3DBloackReduce<<<blocks, THREADS_PER_BLOCK>>>(
                    n, hxyz, r * l_grid[j].x, r * l_grid[j].y, r * l_grid[j].z, r, pitchX,
                    d_distance_grid, d_result);
                real_t delta_sum;
                cudaMemcpyAsync(&delta_sum, d_result, sizeof(real_t), cudaMemcpyDeviceToHost);
                cudaStreamSynchronize(0); // default stream 0
                auto end = std::chrono::high_resolution_clock::now();
                duration += std::chrono::duration_cast<std::chrono::microseconds>(end - start);

                sum += delta_sum * r_grid[i].y * l_grid[j].w;
                if (delta_sum < tol) [[unlikely]] {
                    r_skipped += nr - i;
                    break;
                }
            }
            if (j % 100 == 0) {
                std::cout << "Progress: " << j << "/" << nl << std::endl;
            }
        }
        auto grand_end = std::chrono::high_resolution_clock::now();
        auto grand_dur = std::chrono::duration_cast<std::chrono::microseconds>(grand_end - grand_start);

        sum *= (4.0 / pi) * std::pow(alpha[0] * alpha[1] * alpha[2] * alpha[3], 1.5);
        if (metric) {
            metric->totalTime = grand_dur;
            if (nr * nl - r_skipped != 0) {
                metric->avgKernelTime = duration / (nr * nl - r_skipped);
            } else {
                metric->avgKernelTime = duration;
            }
            metric->totalKernelTime    = duration;
            metric->totalKernelCalls   = nr * nl - r_skipped;
            metric->skippedLebdevNodes = r_skipped;
            // this EBW is outdated for the current kernel (todo: update it)
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
