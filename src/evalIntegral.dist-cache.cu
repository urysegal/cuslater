// updated april 2025
//  Created by gkluhana on 26/03/24.
//
#include "evalIntegral.h"
#include "number.h"
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

constexpr int PACK_ELEMS = std::is_same<real_t, float>::value ? 4 : 2;

namespace cuslater {

    // result layout: idx = z * pitchXY + y * pitchX + x
    // where pitchX pads each row to a multiple of warp-aligned elements.
    __global__ void compute_distance_grid_zmajor(int n,      // n == nx == ny == nz
                                                 int pitchX, // padded row length in elements
                                                 real_t* __restrict__ result) // size: n * (pitchX *
                                                                              // n)
    {
        const int idx_xy  = blockIdx.x * blockDim.x + threadIdx.x;
        const int totalXY = n * n;
        if (idx_xy >= totalXY) return;

        // Map 1D -> (y,x) with x fastest (avoids modulo where possible)
        const int y = idx_xy / n;
        const int x = idx_xy - y * n;

        // Broadcast-ish constants (kept in registers)
        const real3_t c1 = reinterpret_cast<const real3_t*>(d_c)[0];
        const real3_t c2 = reinterpret_cast<const real3_t*>(d_c + 3)[0];
        const real2_t a  = reinterpret_cast<const real2_t*>(d_alpha)[0];

        // Per-thread coordinates (cached via read-only path)
        const real_t X = __ldg(&d_x_grid[x]);
        const real_t Y = __ldg(&d_y_grid[y]);

        const int row     = y * pitchX;
        const int pitchXY = pitchX * n;

        for (int z = 0; z < n; ++z) {
            const real_t Z  = __ldg(&d_z_grid[z]); // all threads use same z -> great cache locality
            const real_t d1 = norm(X - c1.x, Y - c1.y, Z - c1.z);
            const real_t d2 = norm(X - c2.x, Y - c2.y, Z - c2.z);
            const real_t v  = a.x * d1 + a.y * d2;

            // For fixed z, threads with consecutive x write consecutive elements -> coalesced
            result[z * pitchXY + row + x] = v;
        }
    }

    __global__ void compute_distance_grid(int n, real_t* __restrict__ result) {
        const int idx_xy  = blockIdx.x * blockDim.x + threadIdx.x;
        const int totalXY = n * n;
        if (idx_xy < totalXY) {
            const int y = idx_xy / n;
            const int x = idx_xy % n;

            const real3_t c1 = reinterpret_cast<const real3_t*>(d_c)[0];
            const real3_t c2 = reinterpret_cast<const real3_t*>(d_c + 3)[0];
            const real2_t a  = reinterpret_cast<const real2_t*>(d_alpha)[0];

            const real_t X = __ldg(&d_x_grid[x]);
            const real_t Y = __ldg(&d_y_grid[y]);

            int base = (y * n + x) * n;

            for (int z = 0; z < n; ++z) {
                const real_t Z   = __ldg(&d_z_grid[z]);
                const real_t d1  = norm(X - c1.x, Y - c1.y, Z - c1.z);
                const real_t d2  = norm(X - c2.x, Y - c2.y, Z - c2.z);
                result[base + z] = a.x * d1 + a.y * d2;
            }
        }
    }

    __global__ void evalIntegrand_3DBloackReduce(int n, real_t hxyz, real_t rlx, real_t rly,
                                                 real_t rlz, real_t r, real_t* __restrict__ alpha12,
                                                 real_t* __restrict__ block_sums) {
        int idx_flat = (blockIdx.x * blockDim.x + threadIdx.x);
        int totalXY  = n * n;
        int y        = idx_flat / n;
        int x        = idx_flat % n;

        // load to registers, or at least to shared memory
        // to avoid global memory access
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

        real_t local = real_t(0.0);

        if (__builtin_expect(idx_flat < totalXY, 1)) [[likely]] {
            real_t xvalue = __ldg(&d_x_grid[x]);
            real_t yvalue = __ldg(&d_y_grid[y]);

            real_t xdiffc_3 = xvalue - c3.x + rlx;
            real_t xdiffc_4 = xvalue - c4.x + rlx;

            real_t ydiffc_3 = yvalue - c3.y + rly;
            real_t ydiffc_4 = yvalue - c4.y + rly;

            real_t v = real_t(0.0);

            const int base = (y * n + x) * n;

            {
                real_t zvalue = __ldg(&d_z_grid[0]);
                real_t s12    = alpha12[base];

                real_t zdiffc_3 = zvalue - c3.z + rlz;
                real_t zdiffc_4 = zvalue - c4.z + rlz;

                real_t term3    = alpha.z * norm(xdiffc_3, ydiffc_3, zdiffc_3);
                real_t term4    = alpha.w * norm(xdiffc_4, ydiffc_4, zdiffc_4);
                real_t exponent = -s12 - term3 - term4 + r;
                v += exp_fn(exponent) * real_t(0.5);
            } // first run
            int k = 1; // for k = 1 to n - 1
            // scalar loads until aligned
            for (; k < n - 1 && ((base + k) % PACK_ELEMS) != 0; ++k) {
                const real_t z = __ldg(&d_z_grid[k]);
                const real_t s = __ldg(&alpha12[base + k]);

                const real_t t3 = alpha.z * norm(xdiffc_3, ydiffc_3, z - c3.z + rlz);
                const real_t t4 = alpha.w * norm(xdiffc_4, ydiffc_4, z - c4.z + rlz);
                v += exp_fn(-s - t3 - t4 + r);
            }
            for (; k < n - 4; k += 4) {
                real4_t zvalue = reinterpret_cast<real4_t*>(&d_z_grid[k])[0];
                real4_t d12val = make_real4(alpha12[base + k], alpha12[base + k + 1],
                                            alpha12[base + k + 2], alpha12[base + k + 3]);

                real_t zdiffc_3 = zvalue.x - c3.z + rlz;
                real_t zdiffc_4 = zvalue.x - c4.z + rlz;
                real_t term3    = alpha.z * norm(xdiffc_3, ydiffc_3, zdiffc_3);
                real_t term4    = alpha.w * norm(xdiffc_4, ydiffc_4, zdiffc_4);
                real_t exponent = -d12val.x - term3 - term4 + r;
                v += exp_fn(exponent);

                zdiffc_3 = zvalue.y - c3.z + rlz;
                zdiffc_4 = zvalue.y - c4.z + rlz;
                term3    = alpha.z * norm(xdiffc_3, ydiffc_3, zdiffc_3);
                term4    = alpha.w * norm(xdiffc_4, ydiffc_4, zdiffc_4);
                exponent = -d12val.y - term3 - term4 + r;
                v += exp_fn(exponent);

                zdiffc_3 = zvalue.z - c3.z + rlz;
                zdiffc_4 = zvalue.z - c4.z + rlz;
                term3    = alpha.z * norm(xdiffc_3, ydiffc_3, zdiffc_3);
                term4    = alpha.w * norm(xdiffc_4, ydiffc_4, zdiffc_4);
                exponent = -d12val.z - term3 - term4 + r;
                v += exp_fn(exponent);

                zdiffc_3 = zvalue.w - c3.z + rlz;
                zdiffc_4 = zvalue.w - c4.z + rlz;
                term3    = alpha.z * norm(xdiffc_3, ydiffc_3, zdiffc_3);
                term4    = alpha.w * norm(xdiffc_4, ydiffc_4, zdiffc_4);
                exponent = -d12val.w - term3 - term4 + r;
                v += exp_fn(exponent);
            }
            for (; k < n - 1; ++k) { // clear up the remainder
                real_t s12      = alpha12[base + k];
                real_t zvalue   = __ldg(&d_z_grid[k]);
                real_t zdiffc_3 = zvalue - c3.z + rlz;
                real_t zdiffc_4 = zvalue - c4.z + rlz;

                real_t term3    = alpha.z * norm(xdiffc_3, ydiffc_3, zdiffc_3);
                real_t term4    = alpha.w * norm(xdiffc_4, ydiffc_4, zdiffc_4);
                real_t exponent = -s12 - term3 - term4 + r;
                v += exp_fn(exponent);
            }
            { // handle last
                real_t s12      = alpha12[base + k];
                real_t zvalue   = __ldg(&d_z_grid[k]);
                real_t zdiffc_3 = zvalue - c3.z + rlz;
                real_t zdiffc_4 = zvalue - c4.z + rlz;

                real_t term3    = alpha.z * norm(xdiffc_3, ydiffc_3, zdiffc_3);
                real_t term4    = alpha.w * norm(xdiffc_4, ydiffc_4, zdiffc_4);
                real_t exponent = -s12 - term3 - term4 + r;
                v += exp_fn(exponent) * real_t(0.5);
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
        int pitchX     = ((n + alignElems - 1) / alignElems) * alignElems;

        size_t pitchXY = size_t(pitchX) * n;
        size_t total   = size_t(n) * pitchXY; // z * (y * pitchX)

        real_t* d_distance_grid;
        cudaMalloc(&d_distance_grid, total * sizeof(real_t));
        compute_distance_grid_zmajor<<<(n * n + threads - 1) / threads, threads>>>(n, pitchX, d_distance_grid);
        // cudaMalloc(&d_distance_grid, n * n * n * sizeof(real_t));
        // compute_distance_grid<<<blocks, threads>>>(n, d_distance_grid);

        cudaFuncSetCacheConfig(evalIntegrand_3DBloackReduce, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(compute_distance_grid, cudaFuncCachePreferL1);

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
                    n, hxyz, r * l_grid[j].x, r * l_grid[j].y, r * l_grid[j].z, r, d_distance_grid,
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
