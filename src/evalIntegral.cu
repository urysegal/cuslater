// updated april 2025
//  Created by gkluhana on 26/03/24.
//
#include "../include/evalIntegral.h"
#include "grids.h"
#include "utilities.h"
#include <algorithm>
#include <cub/cub.cuh>
#include <numeric>
#include <thrust/device_vector.h>
const double pi = 3.14159265358979323846;
#define THREADS_PER_BLOCK 256

__constant__ real_t d_c[12];
__constant__ real_t d_alpha[4];

__constant__ real_t d_x_grid[500];
__constant__ real_t d_y_grid[500];
__constant__ real_t d_z_grid[500];

namespace cuslater {
    __global__ void evalIntegrand_3DBloackReduce(int n, real_t hx, real_t hy, real_t hz, real_t r,
                                                 real_t lx, real_t ly, real_t lz,
                                                 real_t* __restrict__ block_sums) {
        int idx_flat = blockIdx.x * blockDim.x + threadIdx.x;
        int totalXY  = n * n;
        int y        = idx_flat / n;
        int x        = idx_flat % n;

        // load to registers, or at least to shared memory
        // to avoid global memory access
        real_t c0  = d_c[0];
        real_t c1  = d_c[1];
        real_t c2  = d_c[2];
        real_t c3  = d_c[3];
        real_t c4  = d_c[4];
        real_t c5  = d_c[5];
        real_t c6  = d_c[6];
        real_t c7  = d_c[7];
        real_t c8  = d_c[8];
        real_t c9  = d_c[9];
        real_t c10 = d_c[10];
        real_t c11 = d_c[11];
        real_t a0  = d_alpha[0];
        real_t a1  = d_alpha[1];
        real_t a2  = d_alpha[2];
        real_t a3  = d_alpha[3];

        real_t rlx  = r * lx;
        real_t rly  = r * ly;
        real_t rlz  = r * lz;
        real_t hxyz = hx * hy * hz;

        // inform the compiler this is unlikely (__builtin_expect(cond, 0))
        if (__builtin_expect(x == 0 || x == n - 1, 0)) [[unlikely]] {
            hx *= 0.5; // half weight at endpoints
        }
        if (__builtin_expect(y == 0 || y == n - 1, 0)) [[unlikely]] {
            hy *= 0.5; // half weight at endpoints
        }

        real_t local = 0;
        if (__builtin_expect(idx_flat < totalXY, 1)) [[likely]] {
            real_t xvalue = __ldg(&d_x_grid[x]);
            real_t yvalue = __ldg(&d_y_grid[y]);
            real_t zvalue;

            real_t xdiffc_1 = xvalue - c0;
            real_t xdiffc_2 = xvalue - c3;

            real_t ydiffc_1 = yvalue - c1;
            real_t ydiffc_2 = yvalue - c4;

            real_t xdiffc_3 = xvalue - c6 + rlx;
            real_t xdiffc_4 = xvalue - c9 + rlx;

            real_t ydiffc_3 = yvalue - c7 + rly;
            real_t ydiffc_4 = yvalue - c10 + rly;

            real_t v = 0.f;

            {
                zvalue          = __ldg(&d_z_grid[0]);
                real_t zdiffc_1 = zvalue - c2;
                real_t zdiffc_2 = zvalue - c5;
                real_t zdiffc_3 = zvalue - c8 + rlz;
                real_t zdiffc_4 = zvalue - c11 + rlz;

                real_t term1    = a0 * norm3df(xdiffc_1, ydiffc_1, zdiffc_1);
                real_t term2    = a1 * norm3df(xdiffc_2, ydiffc_2, zdiffc_2);
                real_t term3    = a2 * norm3df(xdiffc_3, ydiffc_3, zdiffc_3);
                real_t term4    = a3 * norm3df(xdiffc_4, ydiffc_4, zdiffc_4);
                real_t exponent = -term1 - term2 - term3 - term4 + r;
                v += __expf(exponent) * 0.5f;
            } // first run
            {
                zvalue = __ldg(&d_z_grid[n - 1]);

                real_t zdiffc_1 = zvalue - c2;
                real_t zdiffc_2 = zvalue - c5;
                real_t zdiffc_3 = zvalue - c8 + rlz;
                real_t zdiffc_4 = zvalue - c11 + rlz;

                real_t term1    = a0 * norm3df(xdiffc_1, ydiffc_1, zdiffc_1);
                real_t term2    = a1 * norm3df(xdiffc_2, ydiffc_2, zdiffc_2);
                real_t term3    = a2 * norm3df(xdiffc_3, ydiffc_3, zdiffc_3);
                real_t term4    = a3 * norm3df(xdiffc_4, ydiffc_4, zdiffc_4);
                real_t exponent = -term1 - term2 - term3 - term4 + r;
                v += __expf(exponent) * 0.5f;
            } // second run at the end point
            {
                float2 zvalue = reinterpret_cast<float2*>(&d_z_grid[n - 3])[0];

                real_t zdiffc_1 = zvalue.x - c2;
                real_t zdiffc_2 = zvalue.x - c5;
                real_t zdiffc_3 = zvalue.x - c8 + rlz;
                real_t zdiffc_4 = zvalue.x - c11 + rlz;

                real_t term1    = a0 * norm3df(xdiffc_1, ydiffc_1, zdiffc_1);
                real_t term2    = a1 * norm3df(xdiffc_2, ydiffc_2, zdiffc_2);
                real_t term3    = a2 * norm3df(xdiffc_3, ydiffc_3, zdiffc_3);
                real_t term4    = a3 * norm3df(xdiffc_4, ydiffc_4, zdiffc_4);
                real_t exponent = -term1 - term2 - term3 - term4 + r;
                v += __expf(exponent);

                zdiffc_1 = zvalue.y - c2;
                zdiffc_2 = zvalue.y - c5;
                zdiffc_3 = zvalue.y - c8 + rlz;
                zdiffc_4 = zvalue.y - c11 + rlz;

                term1    = a0 * norm3df(xdiffc_1, ydiffc_1, zdiffc_1);
                term2    = a1 * norm3df(xdiffc_2, ydiffc_2, zdiffc_2);
                term3    = a2 * norm3df(xdiffc_3, ydiffc_3, zdiffc_3);
                term4    = a3 * norm3df(xdiffc_4, ydiffc_4, zdiffc_4);
                exponent = -term1 - term2 - term3 - term4 + r;
                v += __expf(exponent);
            } // 2 more to cover the the remainder

            // each iteration we compute 4 z-nodes to reduce memory reads
            for (int k = 1; k < n - 3; k += 4) {
                // for (int k = 1; k < n - 1; ++k) {
                //     float  zvalue   = d_z_grid[k];
                //     real_t zdiffc_1 = zvalue - c2;
                //     real_t zdiffc_2 = zvalue - c5;
                //     real_t zdiffc_3 = zvalue - c8 + rlz;
                //     real_t zdiffc_4 = zvalue - c11 + rlz;
                //     real_t term1    = a0 * norm3df(xdiffc_1, ydiffc_1, zdiffc_1);
                //     real_t term2    = a1 * norm3df(xdiffc_2, ydiffc_2, zdiffc_2);
                //     real_t term3    = a2 * norm3df(xdiffc_3, ydiffc_3, zdiffc_3);
                //     real_t term4    = a3 * norm3df(xdiffc_4, ydiffc_4, zdiffc_4);
                //     real_t exponent = -term1 - term2 - term3 - term4 + r;
                //     v += __expf(exponent);
                float4 zvalue = reinterpret_cast<float4*>(&d_z_grid[k])[0];

                real_t zdiffc_1 = zvalue.x - c2;
                real_t zdiffc_2 = zvalue.x - c5;
                real_t zdiffc_3 = zvalue.x - c8 + rlz;
                real_t zdiffc_4 = zvalue.x - c11 + rlz;
                real_t term1    = a0 * norm3df(xdiffc_1, ydiffc_1, zdiffc_1);
                real_t term2    = a1 * norm3df(xdiffc_2, ydiffc_2, zdiffc_2);
                real_t term3    = a2 * norm3df(xdiffc_3, ydiffc_3, zdiffc_3);
                real_t term4    = a3 * norm3df(xdiffc_4, ydiffc_4, zdiffc_4);
                real_t exponent = -term1 - term2 - term3 - term4 + r;
                v += __expf(exponent);

                zdiffc_1 = zvalue.y - c2;
                zdiffc_2 = zvalue.y - c5;
                zdiffc_3 = zvalue.y - c8 + rlz;
                zdiffc_4 = zvalue.y - c11 + rlz;
                term1    = a0 * norm3df(xdiffc_1, ydiffc_1, zdiffc_1);
                term2    = a1 * norm3df(xdiffc_2, ydiffc_2, zdiffc_2);
                term3    = a2 * norm3df(xdiffc_3, ydiffc_3, zdiffc_3);
                term4    = a3 * norm3df(xdiffc_4, ydiffc_4, zdiffc_4);
                exponent = -term1 - term2 - term3 - term4 + r;
                v += __expf(exponent);

                zdiffc_1 = zvalue.z - c2;
                zdiffc_2 = zvalue.z - c5;
                zdiffc_3 = zvalue.z - c8 + rlz;
                zdiffc_4 = zvalue.z - c11 + rlz;
                term1    = a0 * norm3df(xdiffc_1, ydiffc_1, zdiffc_1);
                term2    = a1 * norm3df(xdiffc_2, ydiffc_2, zdiffc_2);
                term3    = a2 * norm3df(xdiffc_3, ydiffc_3, zdiffc_3);
                term4    = a3 * norm3df(xdiffc_4, ydiffc_4, zdiffc_4);
                exponent = -term1 - term2 - term3 - term4 + r;
                v += __expf(exponent);

                zdiffc_1 = zvalue.w - c2;
                zdiffc_2 = zvalue.w - c5;
                zdiffc_3 = zvalue.w - c8 + rlz;
                zdiffc_4 = zvalue.w - c11 + rlz;
                term1    = a0 * norm3df(xdiffc_1, ydiffc_1, zdiffc_1);
                term2    = a1 * norm3df(xdiffc_2, ydiffc_2, zdiffc_2);
                term3    = a2 * norm3df(xdiffc_3, ydiffc_3, zdiffc_3);
                term4    = a3 * norm3df(xdiffc_4, ydiffc_4, zdiffc_4);
                exponent = -term1 - term2 - term3 - term4 + r;
                v += __expf(exponent);
            }
            local += v * hxyz;
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
            // printf("zero condition: check wheter %f > %f, \n", cond, log(r0 * inv_machine_eps));
            // std::cout << "Zero condition met" << std::endl;
            return true;
        }
        return false;
    }

    double evaluateFourCenterIntegral(real_t* c, real_t* alpha, int nr, int nl, int n,
                                       double tol, bool check_zero_cond) {
        if (check_zero_cond && checkZero(c, alpha)) {
            return 0.0;
        }

        // read r grid
        std::cout << "Reading r Grid Files" << std::endl;
        const std::string   r_filepath = "grid_files/r_" + std::to_string(nr) + ".grid";
        std::vector<real_t> r_nodes;
        std::vector<real_t> r_weights;
        read_r_grid_from_file(r_filepath, r_nodes, r_weights);

        // read l grid
        std::cout << "Reading l Grid Files" << std::endl;
        const std::string   l_filepath = "grid_files/l_" + std::to_string(nl) + ".grid";
        std::vector<real_t> l_nodes_x;
        std::vector<real_t> l_nodes_y;
        std::vector<real_t> l_nodes_z;
        std::vector<real_t> l_weights;
        read_l_grid_from_file(l_filepath, l_nodes_x, l_nodes_y, l_nodes_z, l_weights);

        std::cout << "Generating x1 Grid using Trapezoidal Rule" << std::endl;

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

        std::cout << "Initializing Device Variables" << std::endl;

        constexpr int threads = THREADS_PER_BLOCK;               // Max threads per block
        int           blocks  = (n * n + threads - 1) / threads; // Max blocks, better if
                                                                 // multiple of SM = 80
        std::cout << "Total Blocks: " << blocks << std::endl;
        std::cout << "Total Threads: " << blocks * threads << std::endl;
        std::cout << "Total Grid Points: " << n * n * n << std::endl;

        cudaMemcpyToSymbol(d_c, c, 12 * sizeof(real_t));
        cudaMemcpyToSymbol(d_alpha, alpha, 4 * sizeof(real_t));
        cudaMemcpyToSymbol(d_x_grid, x1_nodes.data(), n * sizeof(real_t));
        cudaMemcpyToSymbol(d_y_grid, y1_nodes.data(), n * sizeof(real_t));
        cudaMemcpyToSymbol(d_z_grid, z1_nodes.data(), n * sizeof(real_t));

        std::cout << "Evaluating Integral for all values of r and l with\n";
        std::cout << "  a1=" << alpha[0] << ", a2=" << alpha[1] << ", a3=" << alpha[2]
                  << ", a4=" << alpha[3] << "\n";
        std::cout << "  c1 = (" << c[0] << ", " << c[1] << ", " << c[2] << ")\n";
        std::cout << "  c2 = (" << c[3] << ", " << c[4] << ", " << c[5] << ")\n";
        std::cout << "  c3 = (" << c[6] << ", " << c[7] << ", " << c[8] << ")\n";
        std::cout << "  c4 = (" << c[9] << ", " << c[10] << ", " << c[11] << ")\n";
        std::cout << "  Tolerance = " << tol << std::endl;
        std::cout << " Legendre Grid Parameters: " << std::endl;
        std::cout << " xgrid (ax , bx) : (" << ax << " , " << bx << ")" << std::endl;
        std::cout << " ygrid (ay , by) : (" << ay << " , " << by << ")" << std::endl;
        std::cout << " zgrid (az , bz) : (" << az << " , " << bz << ")" << std::endl;

        thrust::device_vector<real_t> d_block_sums(blocks);
        thrust::host_vector<real_t>   block_sums(blocks);

        double                    sum       = 0.0f;
        double                    delta_sum = 0.0f;
        int                       r_skipped = 0;
        std::chrono::microseconds duration(0);

        auto grand_start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < nl; ++j) {
            for (int i = 0; i < nr; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                evalIntegrand_3DBloackReduce<<<blocks, THREADS_PER_BLOCK>>>(
                    n, hx, hy, hz, r_nodes[i], l_nodes_x[j], l_nodes_y[j], l_nodes_z[j],
                    thrust::raw_pointer_cast(d_block_sums.data()));
                // cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                duration += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                block_sums = d_block_sums;
                delta_sum  = std::accumulate(block_sums.begin(), block_sums.end(), 0.0);

                sum += delta_sum * r_weights[i] * l_weights[j];
                if (delta_sum < tol) [[unlikely]] {
                    r_skipped += nr - i;
                    break;
                }
            }
            if (j % 100 == 0) {
                std::cout << "computed for l_j:" << j << "/" << nl << std::endl;
            }
        }
        auto grand_end = std::chrono::high_resolution_clock::now();
        auto grand_dur = std::chrono::duration_cast<std::chrono::microseconds>(grand_end - grand_start);

        // sum up result, multiply with constant and return
        std::cout << "sum before multiplication " << sum << std::endl;
        sum *= (4.0 / pi) * std::pow(alpha[0] * alpha[1] * alpha[2] * alpha[3], 1.5);

        std::cout << "Tolerance: " << tol << std::endl;
        std::cout << "Total values of r skipped for different l's: " << r_skipped << "/" << nr * nl
                  << std::endl;
        auto avgTime = duration.count() / (nr * nl - r_skipped);
        std::cout << "Total Kernel Time: " << duration.count() << " microseconds" << std::endl;
        std::cout << "Total Time: " << grand_dur.count() << " microseconds" << std::endl;
        std::cout << "Total Kernel Calls: " << nr * nl - r_skipped << std::endl;
        std::cout << "Avg Per Kernel Time: " << avgTime << " microseconds" << std::endl;
        std::cout << "Effective Bandwidth: " << (202.0 * 4 * n * n + blocks * 4) / avgTime / 1e3
                  << " GB/s" << std::endl;
        return sum;
    }
} // namespace cuslater
