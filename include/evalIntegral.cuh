//
// Created by gkluhana on 26/03/24.
//
// editted by MarkEwert03 on 13/05/24

#pragma once
#include "number.cuh"
#include <chrono>
#include <ostream>
#include <span>
#include <vector>
namespace cuslater {
    using namespace std;

    struct Domain {
        real_t ax, bx;
        real_t ay, by;
        real_t az, bz;
        real_t hx, hy, hz;

        Domain(std::span<real_t, 12> c, int n);

        real_t volume() const {
            return (bx - ax) * (by - ay) * (bz - az);
        }

        real_t delta_volume() const {
            return hx * hy * hz;
        }

        real2_t x_bounds() const {
            return {ax, bx};
        }

        real2_t y_bounds() const {
            return {ay, by};
        }

        real2_t z_bounds() const {
            return {az, bz};
        }

        real3_t h() const {
            return {hx, hy, hz};
        }

        real3_t left() const {
            return {ax, ay, az};
        }

        real3_t right() const {
            return {bx, by, bz};
        }
    };
    std::ostream& operator<<(ostream& os, const Domain& m);
    struct Metric {
        chrono::microseconds totalTime;
        chrono::microseconds avgKernelTime;
        chrono::microseconds totalKernelTime;
        int                  totalKernelCalls;
        int                  skippedLebdevNodes;
        double               effectiveBandwidth;
        int                  totalThreads;
        int                  totalBlocks;
        int                  totalGridPoints;
        int                  N, L, R;
        double               tol;
        real3_t              a, b;
    };
    std::ostream& operator<<(ostream& os, const Metric& m);

    void print_results(const std::vector<double>& sums);
    void normalize_sums(std::vector<double>& sums, const std::array<double, 4>& a);
    void print_results_d_subset81(const std::vector<double>& sums, const std::array<int, 3>& comps);
    void normalize_sums_d_subset81(std::vector<double>& sums, const std::array<double, 4>& a,
                                   const std::array<int, 3>& comps);
    void print_results_d_tile(const std::vector<double>& tile_vals, int left_begin, int left_count,
                              int right_begin, int right_count);
    void normalize_sums_d_tile(std::vector<double>& tile_vals, const std::array<double, 4>& a,
                               int left_begin, int left_count, int right_begin, int right_count);

    /**
     * @brief Evaluates the four-center integral using the Lebedev grid.
     *
     * @param c The coordinates of the centers.
     * @param alphas The alpha values for the centers.
     * @param r_nodes The radial nodes for the Lebedev grid.
     * @param l_nodes The angular nodes for the Lebedev grid.
     * @param n The number of nodes in each dimension.
     * @param tol The tolerance for convergence.
     * @param check_zero_cond Whether to check for zero condition.
     * @param metric Optional pointer to a Metric object to store performance metrics.
     * @return The result of the integral evaluation.
     */
    double evaluateFourCenterIntegral(real_t* c, real_t* alphas, vector<real2_t>& r_nodes,
                                      vector<real4_t>& l_nodes, int n, double tol,
                                      bool check_zero_cond, Metric* metric = nullptr);

} // namespace cuslater
