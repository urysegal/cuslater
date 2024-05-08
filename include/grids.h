//
// Created by gkluhana on 26/03/24.
//

//
// Created by gkluhana on 04/03/24.
//
#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda_runtime_api.h>
#include "utilities.h"

namespace cuslater{

    void make_1d_grid_legendre(double start, double stop, unsigned int N,
                               std::vector<double>* grid, std::vector<double>*weights);

    void make_1d_grid_simpson(double start, double stop, unsigned int N,
                              std::vector<double>* grid, std::vector<double>* weights);

    double make_1d_grid_uniform(double start, double stop, unsigned int N,
                        std::vector<double>* grid);

    void read_r_grid_from_file(const std::string& filepath,
                                 std::vector<float>& r_nodes,
                                 std::vector<float>& r_weights);

    void read_l_grid_from_file(const std::string& filepath,
                               std::vector<float>& l_nodes_x, std::vector<float>& l_nodes_y,std::vector<float>& l_nodes_z,
                               std::vector<float>& l_weights);
    void read_x1_1d_grid_from_file(const std::string& filepath,
                                std::vector<float>& x1_nodes,
                                std::vector<float>& x1_weights);



}
