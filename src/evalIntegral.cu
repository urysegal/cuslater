// updated april 2025
//  Created by gkluhana on 26/03/24.
//
#include "../include/evalIntegral.h"
#include "cuslater.cuh"
#include "grids.h"
#include "utilities.h"
#include <algorithm>
#include <cub/cub.cuh>
#include <numeric>
#include <thrust/device_vector.h>
const double pi = 3.14159265358979323846;
#define THREADS_PER_BLOCK 128

__constant__ real_t d_c[12];
__constant__ real_t d_alpha[4];

__constant__ real_t d_x_grid[600];
__constant__ real_t d_y_grid[600];
__constant__ real_t d_z_grid[600];

namespace cuslater {
    /*
     * 1. Figure out how to take advantage of equidistant grid points
     * 2. Use shared memory to reduce global memory access
     * 3. Use block reduction to reduce the number of global memory accesses
     * 4. Use cub::BlockReduce to reduce the number of global memory accesses
     * 5. Use __ldg() for reading constant memory
     * 6. Coalesce (l, r) calls into one kernel call (This is tricky)
     */

    // this will now reduce the sum over the 3Dblock
    // so the host only needs to sum over per block sum
    __global__ void evalIntegrand_3DBloackReduce(int n, real_t hx, real_t hy, real_t hz, real_t r,
                                                 real_t lx, real_t ly, real_t lz, double* block_sums) {
        int idx_flat = blockIdx.x * blockDim.x + threadIdx.x;
        int totalXY  = n * n;
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

        real_t rlx = r * lx;
        real_t rly = r * ly;
        real_t rlz = r * lz;

        real_t dxy = hx * hy;
        real_t dz  = hz;

        real_t local = 0;
        for (; idx_flat < totalXY; idx_flat += gridDim.x * blockDim.x) {
            int    y = idx_flat / n, x = idx_flat % n;
            real_t xvalue = __ldg(&d_x_grid[x]);
            real_t yvalue = __ldg(&d_y_grid[y]);

            // inform the compiler this is unlikely (__builtin_expect(cond, 0))
            if (__builtin_expect(x == 0 || x == n - 1, 0)) {
                dxy *= 0.5; // half weight at endpoints
            }
            if (__builtin_expect(y == 0 || y == n - 1, 0)) {
                dxy *= 0.5; // half weight at endpoints
            }

            real_t xdiffc_1 = xvalue - c0;
            real_t ydiffc_1 = yvalue - c1;
            real_t xdiffc_2 = xvalue - c3;
            real_t ydiffc_2 = yvalue - c4;
            real_t xdiffc_3 = xvalue - c6 + rlx;
            real_t ydiffc_3 = yvalue - c7 + rly;
            real_t xdiffc_4 = xvalue - c9 + rlx;
            real_t ydiffc_4 = yvalue - c10 + rly;

            real_t xysq1 = xdiffc_1 * xdiffc_1 + ydiffc_1 * ydiffc_1;
            real_t xysq2 = xdiffc_2 * xdiffc_2 + ydiffc_2 * ydiffc_2;
            real_t xysq3 = xdiffc_3 * xdiffc_3 + ydiffc_3 * ydiffc_3;
            real_t xysq4 = xdiffc_4 * xdiffc_4 + ydiffc_4 * ydiffc_4;

            double v = 0;

            {
                real_t zvalue = __ldg(&d_z_grid[0]);
                real_t dz     = hz * 0.5;

                real_t zdiffc_1 = zvalue - c2;
                real_t zdiffc_2 = zvalue - c5;
                real_t zdiffc_3 = zvalue - c8 + rlz;
                real_t zdiffc_4 = zvalue - c11 + rlz;
                real_t term1    = a0 * __fsqrt_rn(xysq1 + zdiffc_1 * zdiffc_1);
                real_t term2    = a1 * __fsqrt_rn(xysq2 + zdiffc_2 * zdiffc_2);
                real_t term3    = a2 * __fsqrt_rn(xysq3 + zdiffc_3 * zdiffc_3);
                real_t term4    = a3 * __fsqrt_rn(xysq4 + zdiffc_4 * zdiffc_4);
                real_t exponent = -term1 - term2 - term3 - term4 + r;
                v += __expf(exponent) * dz;
            } // first run
            {
                real_t zvalue = __ldg(&d_z_grid[n - 1]);
                real_t dz     = hz * 0.5;

                real_t zdiffc_1 = zvalue - c2;
                real_t zdiffc_2 = zvalue - c5;
                real_t zdiffc_3 = zvalue - c8 + rlz;
                real_t zdiffc_4 = zvalue - c11 + rlz;
                real_t term1    = a0 * __fsqrt_rn(xysq1 + zdiffc_1 * zdiffc_1);
                real_t term2    = a1 * __fsqrt_rn(xysq2 + zdiffc_2 * zdiffc_2);
                real_t term3    = a2 * __fsqrt_rn(xysq3 + zdiffc_3 * zdiffc_3);
                real_t term4    = a3 * __fsqrt_rn(xysq4 + zdiffc_4 * zdiffc_4);
                real_t exponent = -term1 - term2 - term3 - term4 + r;
                v += __expf(exponent) * dz;
            } // second run at the end point

            for (int k = 1; k < n - 1; ++k) {
                real_t zvalue = __ldg(&d_z_grid[k]);

                real_t zdiffc_1 = zvalue - c2;
                real_t zdiffc_2 = zvalue - c5;
                real_t zdiffc_3 = zvalue - c8 + rlz;
                real_t zdiffc_4 = zvalue - c11 + rlz;
                real_t term1    = a0 * __fsqrt_rn(xysq1 + zdiffc_1 * zdiffc_1);
                real_t term2    = a1 * __fsqrt_rn(xysq2 + zdiffc_2 * zdiffc_2);
                real_t term3    = a2 * __fsqrt_rn(xysq3 + zdiffc_3 * zdiffc_3);
                real_t term4    = a3 * __fsqrt_rn(xysq4 + zdiffc_4 * zdiffc_4);
                real_t exponent = -term1 - term2 - term3 - term4 + r;
                v += __expf(exponent) * dz;
            }
            local += v * dxy;
        }
        using BlockReduce = cub::BlockReduce<real_t, THREADS_PER_BLOCK>;

        __shared__ typename BlockReduce::TempStorage temp;

        double block_sum = BlockReduce(temp).Sum(local);
        if (threadIdx.x == 0) block_sums[blockIdx.x] = block_sum;
    }

    template<int BLOCKSIZE>
    double evaluateInnerSum(int n, real_t hx, real_t hy, real_t hz, real_t r, real_t l_x,
                            real_t l_y, real_t                                            l_z,
                            thrust::device_vector<double>& __restrict__ d_block_sums, int blocks) {
        int shared_size = blocks * sizeof(double);
        evalIntegrand_3DBloackReduce<<<blocks, BLOCKSIZE, shared_size>>>(
            n, hx, hy, hz, r, l_x, l_y, l_z, thrust::raw_pointer_cast(d_block_sums.data()));

        thrust::host_vector<double> v = d_block_sums;
        return std::accumulate(v.begin(), v.end(), 0.0);
    } // evaluateInner
      //
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
            printf("zero condition: check wheter %f > %f, \n", cond, log(r0 * inv_machine_eps));
            std::cout << "Zero condition met" << std::endl;
            return true;
        }
        return false;
    }

    double evaluateFourCenterIntegral(real_t* c, real_t* alpha, int nr, int nl, int nx, int ny, int nz,
                                      const std::string x1_type, double tol, bool check_zero_cond) {
        if (check_zero_cond && checkZero(c, alpha)) {
            return 0.0;
        }

        HANDLE_CUDA_ERROR(cudaSetDevice(0));

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
        std::vector<real_t> x1_nodes(nx);
        std::vector<real_t> x1_weights(nx);
        real_t              hx = (bx - ax) / (nx - 1);

        std::vector<real_t> y1_nodes(nx);
        std::vector<real_t> y1_weights(nx);
        real_t              hy = (by - ay) / (nx - 1);

        std::vector<real_t> z1_nodes(nx);
        std::vector<real_t> z1_weights(nx);
        real_t              hz = (bz - az) / (nx - 1);

        for (int i = 0; i < nx; ++i) {
            x1_nodes[i] = ax + i * hx;
        }

        for (int i = 0; i < nx; ++i) {
            y1_nodes[i] = ay + i * hy;
        }

        for (int i = 0; i < nx; ++i) {
            z1_nodes[i] = az + i * hz;
        }

        std::cout << "Initializing Device Variables" << std::endl;
        unsigned int PX = x1_nodes.size();
        unsigned int PY = y1_nodes.size();
        unsigned int PZ = z1_nodes.size();

        int threads = THREADS_PER_BLOCK;                 // Max threads per block
        int blocks  = (PX * PY + threads - 1) / threads; // Max blocks, better if
                                                         // multiple of SM = 80
        std::cout << "Total Threads: " << blocks * threads << std::endl;
        std::cout << "Total Grid Points: " << nx * ny * nz << std::endl;

        cudaMemcpyToSymbol(d_c, c, 12 * sizeof(real_t));
        cudaMemcpyToSymbol(d_alpha, alpha, 4 * sizeof(real_t));
        cudaMemcpyToSymbol(d_x_grid, x1_nodes.data(), PX * sizeof(real_t));
        cudaMemcpyToSymbol(d_y_grid, y1_nodes.data(), PY * sizeof(real_t));
        cudaMemcpyToSymbol(d_z_grid, z1_nodes.data(), PZ * sizeof(real_t));


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

        static thrust::device_vector<double> d_block_sums(blocks);

        double sum       = 0.0;
        double delta_sum = 0.0;
        int    r_skipped = 0;
        int    n         = nx; // number of grid points in x1, y1, z1

        for (int j = 0; j < nl; ++j) {
            for (int i = 0; i < nr; ++i) {
                delta_sum = evaluateInnerSum<THREADS_PER_BLOCK>(n, hx, hy, hz, r_nodes[i],
                                                                l_nodes_x[j], l_nodes_y[j],
                                                                l_nodes_z[j], d_block_sums, blocks);

                sum += delta_sum * r_weights[i] * l_weights[j];
                if (delta_sum < tol) {
                    r_skipped += nr - i;
                    break;
                }
            }
            if (j % 100 == 0) {
                std::cout << "computed for l_j:" << j << "/" << nl << std::endl;
            }
        }
        std::cout << "sum before multiplication " << sum << std::endl;
        sum *= (4.0 / pi) * std::pow(alpha[0] * alpha[1] * alpha[2] * alpha[3], 1.5);

        // sum up result, multiply with constant and return
        std::cout << "Tolerance: " << tol << std::endl;
        std::cout << "Total values of r skipped for different l's: " << r_skipped << "/" << nr * nl
                  << std::endl;
        return sum;
    }
} // namespace cuslater
