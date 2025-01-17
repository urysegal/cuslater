//
// Created by gkluhana on 26/03/24.
//

//
// Created by gkluhana on 04/03/24.
//
#include "utilities.h"
#include <cassert>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace cuslater {

void make_1d_grid_legendre(double start, double stop, unsigned int N, std::vector<double> *grid,
                           std::vector<double> *weights);

void make_1d_grid_simpson(double start, double stop, unsigned int N, std::vector<double> *grid,
                          std::vector<double> *weights);

    void read_l_grid_from_file(const std::string& filepath,
                               std::vector<real_t>& l_nodes_x, std::vector<real_t>& l_nodes_y,std::vector<real_t>& l_nodes_z,
                               std::vector<real_t>& l_weights);
    void generate_x1_from_std(real_t a, real_t b, 
			      const std::vector<real_t>& x1_standard_nodes, const std::vector<real_t>& x1_standard_weights, 
			      std::vector<real_t>& x1_nodes, std::vector<real_t>& x1_weights);

    void read_x1_1d_grid_from_file(const std::string& filepath,
                                std::vector<real_t>& x1_nodes,
                                std::vector<real_t>& x1_weights);

void read_r_grid_from_file(const std::string &filepath, std::vector<real_t> &r_nodes,
                           std::vector<real_t> &r_weights);

void read_l_grid_from_file(const std::string &filepath, std::vector<real_t> &l_nodes_x,
                           std::vector<real_t> &l_nodes_y, std::vector<real_t> &l_nodes_z,
                           std::vector<real_t> &l_weights);
void read_x1_1d_grid_from_file(const std::string &filepath, real_t &a, real_t &b,
                               std::vector<real_t> &x1_nodes, std::vector<real_t> &x1_weights);

} // namespace cuslater
