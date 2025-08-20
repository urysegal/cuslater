//
// Created by gkluhana on 26/03/24.
//

//
// Created by gkluhana on 04/03/24.
//
#pragma once
#include "number.h"
#include <vector>

namespace cuslater {
    /**
     * @brief Reads the radial nodes from a file.
     *
     * @param nr The number of radial nodes to read.
     * @return A vector of float2 containing the radial nodes and their weights.
     *         r.x will be the node and r.y will be the weight.
     */
    std::vector<real2_t> read_r_grid(int nr);

    /**
     * @brief Reads the angular nodes from a file.
     *
     * @param nl The number of angular nodes to read.
     * @return A vector of float4 containing the angular nodes and weights.
     *         Each float4 represents a point in 3D space (x, y, z) and the weight (w).
     */
    std::vector<real4_t> read_l_grid(int nl);

} // namespace cuslater
