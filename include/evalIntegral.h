//
// Created by gkluhana on 26/03/24.
//
// editted by MarkEwert03 on 13/05/24

#pragma once
#include "number.h"
#include <chrono>
#include <ostream>
#include <thrust/device_vector.h>
#include <vector>
namespace cuslater {
    using namespace std;
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
        real3_t              a, b;
    };
    std::ostream& operator<<(ostream& os, const Metric& m);

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
