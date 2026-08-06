// updated april 2025
//  Created by gkluhana on 26/03/24.
//
#include "evalIntegral.cuh"
#include "kernels/kernels.cuh"
#include "number.cuh"
#include "slot.cuh"
#include <algorithm>
#include <cub/cub.cuh>
#include <span>
#include <tuple>
#include <vector>

namespace cuslater {
    using namespace cudapp;

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

    using std::tuple;
    using std::vector;
    using arr = vector<real_t>;

    tuple<arr, arr, arr> build_nodes(Domain& domain, const int n) {

        // Generate trapezoidal nodes and weights for x1, y1, z1
        auto [ax, bx, ay, by, az, bz, hx, hy, hz] = domain;
        std::vector<real_t> x1_nodes(n);
        std::vector<real_t> x1_weights(n);

        std::vector<real_t> y1_nodes(n);
        std::vector<real_t> y1_weights(n);

        std::vector<real_t> z1_nodes(n);
        std::vector<real_t> z1_weights(n);
        for (int i = 0; i < n; ++i) {
            x1_nodes[i] = ax + i * hx;
        }

        for (int i = 0; i < n; ++i) {
            y1_nodes[i] = ay + i * hy;
        }

        for (int i = 0; i < n; ++i) {
            z1_nodes[i] = az + i * hz;
        }
        return {x1_nodes, y1_nodes, z1_nodes};
    }

    void copy_grid_to_device(real_t* c, real_t* alpha, arr& x_nodes, arr& y_nodes, arr& z_nodes) {
        const int n = x_nodes.size();
        cudaMemcpyToSymbol(d_c, c, 12 * sizeof(real_t));
        cudaMemcpyToSymbol(d_alpha, alpha, 4 * sizeof(real_t));
        cudaMemcpyToSymbol(d_x_grid, x_nodes.data(), n * sizeof(real_t));
        cudaMemcpyToSymbol(d_y_grid, y_nodes.data(), n * sizeof(real_t));
        cudaMemcpyToSymbol(d_z_grid, z_nodes.data(), n * sizeof(real_t));
    }

    double evaluateFourCenterIntegral(real_t* c, real_t* alpha, vector<real2_t>& r_grid,
                                      vector<real4_t>& l_grid, int n, double tol,
                                      bool check_zero_cond, Metric* metric) {
        if (check_zero_cond && checkZero(c, alpha)) {
            return 0.0;
        }

        const int nr = r_grid.size();
        const int nl = l_grid.size();

        Domain domain                       = Domain(std::span<real_t, 12>(c, 12), n);
        auto [x1_nodes, y1_nodes, z1_nodes] = build_nodes(domain, n);

        copy_grid_to_device(c, alpha, x1_nodes, y1_nodes, z1_nodes);
        auto [distance_pair_grid, pitchX] = build_distance_pair_grid(n);
        cudaFuncSetCacheConfig(evalIntegrand_3DBloackReduce, cudaFuncCachePreferL1);

        constexpr int threads = THREADS_PER_BLOCK; // Max threads per block
        int           blocks  = (n * n + threads - 1) / threads;

        int                       r_skipped = 0;
        std::chrono::microseconds duration(0);

        real_t hxyz = domain.delta_volume();

        constexpr int                     NUM_INTEGRALS = 81;
        std::vector<double>               sums(NUM_INTEGRALS, 0.0);
        CudaArray                         results(NUM_INTEGRALS);
        std::array<real_t, NUM_INTEGRALS> h_results = {};

        CudaEvent start, stop;
        CudaArray d_r_grid(2 * nr);
        CudaArray d_l_grid(4 * nl);

        cudaMemcpy(d_r_grid.data(), r_grid.data(), nr * sizeof(real2_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_l_grid.data(), l_grid.data(), nl * sizeof(real4_t), cudaMemcpyHostToDevice);

        const int blocksXY    = (n * n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const int samples     = nr * nl;
        const int totalBlocks = blocksXY * samples;

        // auto grand_start = std::chrono::high_resolution_clock::now();
        start.record();

        // eval_1111_simpson_all_samples<<<totalBlocks, THREADS_PER_BLOCK>>>(
        //     n,
        //     hxyz,
        //     pitchX,
        //     blocksXY,
        //     nr,
        //     nl,
        //     reinterpret_cast<const real2_t*>(d_r_grid.data()),
        //     reinterpret_cast<const real4_t*>(d_l_grid.data()),
        //     reinterpret_cast<const real2_t*>(distance_pair_grid.data()),
        //     results.data()
        // );

        // eval_1111_trap_all_samples<<<totalBlocks, THREADS_PER_BLOCK>>>(
        //     n,
        //     hxyz,
        //     pitchX,
        //     blocksXY,
        //     nr,
        //     nl,
        //     reinterpret_cast<const real2_t*>(d_r_grid.data()),
        //     reinterpret_cast<const real4_t*>(d_l_grid.data()),
        //     reinterpret_cast<const real2_t*>(distance_pair_grid.data()),
        //     results.data()
        // );

        // evalIntegrand_3DBloackReduce_SSSS_SSSP_SSPS_SPSS_all_samples<<<totalBlocks,
        // THREADS_PER_BLOCK>>>(
        //     n, hxyz, pitchX, blocksXY, nr, nl, reinterpret_cast<const real2_t*>(d_r_grid.data()),
        //     reinterpret_cast<const real4_t*>(d_l_grid.data()),
        //     reinterpret_cast<const real2_t*>(distance_pair_grid.data()), results.data());

        evalIntegrand_3DBloackReduce_DDDD_tile_all_samples<0, 9, 0,9>
            <<<totalBlocks, THREADS_PER_BLOCK>>>(
                n, hxyz, pitchX, blocksXY, nr, nl, reinterpret_cast<const real2_t*>(d_r_grid.data()),
                reinterpret_cast<const real4_t*>(d_l_grid.data()),
                reinterpret_cast<const real2_t*>(distance_pair_grid.data()), results.data());

        // evalIntegrand_3DBloackReduce_SSSS_SSSP_SSPS_SPSS_all_samples_simpson<<<totalBlocks,
        // THREADS_PER_BLOCK>>>(
        //     n, hxyz, pitchX, blocksXY, nr, nl, reinterpret_cast<const real2_t*>(d_r_grid.data()),
        //     reinterpret_cast<const real4_t*>(d_l_grid.data()),
        //     reinterpret_cast<const real2_t*>(distance_pair_grid.data()), results.data());

        stop.record();
        cudaMemcpy(sums.data(), results.data(), NUM_INTEGRALS * sizeof(real_t), cudaMemcpyDeviceToHost);
        // auto grand_end = std::chrono::high_resolution_clock::now();

        // unfused:
        // auto grand_start = std::chrono::high_resolution_clock::now();
        // for (int j = 0; j < nl; ++j) {
        //     for (int i = 0; i < nr; ++i) {
        //         results.setZero();
        //         real_t r = r_grid[i].x;

        //         auto start = std::chrono::high_resolution_clock::now();
        //         // evalIntegrand_3DBloackReduce_SSSS_SSSP_SSPS_SPSS<<<blocks,
        //         THREADS_PER_BLOCK>>>(
        //         //     n, hxyz, r * l_grid[j].x, r * l_grid[j].y, r * l_grid[j].z, r, pitchX,
        //         //     reinterpret_cast<const real2_t*>(distance_pair_grid.data()),
        //         results.data());

        //         // evalIntegrand_3DBloackReduce_SSSS_SSSP_SSPS<<<blocks, THREADS_PER_BLOCK>>>(
        //         //     n, hxyz, r * l_grid[j].x, r * l_grid[j].y, r * l_grid[j].z, r, pitchX,
        //         //     reinterpret_cast<const real2_t*>(distance_pair_grid.data()),
        //         // results.data());

        //         evalIntegrand_3DBloackReduce_DDDD_81<<<blocks, THREADS_PER_BLOCK>>>(
        //             n, hxyz, r * l_grid[j].x, r * l_grid[j].y, r * l_grid[j].z, r, pitchX,
        //             reinterpret_cast<const real2_t*>(distance_pair_grid.data()),
        //         results.data());

        //         // evalIntegrand_3DBloackReduce_SSPS<<<blocks, THREADS_PER_BLOCK, 0, s1>>>(
        //         //     n, hxyz, r * l_grid[j].x, r * l_grid[j].y, r * l_grid[j].z, r, pitchX,
        //         //     reinterpret_cast<const real2_t*>(distance_pair_grid.data()),
        //         // results.data() + //     40);

        //         auto end = std::chrono::high_resolution_clock::now();

        //         // real_t delta_sum = result; // async read from device
        //         cudaMemcpy(h_results.data(), results.d_array, NUM_INTEGRALS * sizeof(real_t),
        //                         cudaMemcpyDeviceToHost);
        //         duration += std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        //         const real_t weight = r_grid[i].y * l_grid[j].w;
        //         for (int idx = 0; idx < NUM_INTEGRALS; ++idx) {
        //             sums[idx] += h_results[idx] * weight;
        //         }
        //     }
        // }
        // auto grand_end = std::chrono::high_resolution_clock::now();

        // auto grand_dur = std::chrono::duration_cast<std::chrono::microseconds>(grand_end -
        // grand_start);

        auto grand_dur = stop.since(start);

        const std::array<double, 4> a = {static_cast<double>(alpha[0]), static_cast<double>(alpha[1]),
                                         static_cast<double>(alpha[2]), static_cast<double>(alpha[3])};

        normalize_sums_d_tile(sums, a, 0, 9, 0, 9);
        print_results_d_tile(sums, 0, 9, 0, 9);
        // normalize_sums_d_subset81(sums, a, {0, 1, 2});
        // print_results_d_subset81(sums, {0, 1, 2});
        // normalize_sums(sums, a);
        // print_results(sums);

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
            metric->a               = make_real3(domain.ax, domain.ay, domain.az);
            metric->b               = make_real3(domain.bx, domain.by, domain.bz);
        }
        return sums[0];
    }
} // namespace cuslater
