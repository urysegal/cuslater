
#include "../include/grids.h"
#include <iostream>
using namespace std;

int
main(int argc, const char *argv[])
{
        const std::string filepath = "grid_files/r_2.grid";

        // Vectors to store grid_files
        std::vector<double> r_nodes;
        std::vector<double> r_weights;
        // Read grid_files from file
        cuslater::read_r_grid_from_file(filepath,r_nodes,r_weights);
        // Output grid_files
        std::cout << "r_nodes:" << std::endl;
        for (const auto& node : r_nodes) {
                std::cout << node << std::endl;
        }

        std::cout << "r_weights:" << std::endl;
        for (const auto& weight : r_weights) {
                std::cout << weight << std::endl;
        }

        const std::string l_filepath = "grid_files/l_6.grid";
        // Vectors to store grid_files
        std::vector<double> l_nodes_x;
        std::vector<double> l_nodes_y;
        std::vector<double> l_nodes_z;
        std::vector<double> l_weights;
        // Read grid_files from file
        cuslater::read_l_grid_from_file(l_filepath,
                                        l_nodes_x,l_nodes_y, l_nodes_z,
                                        l_weights);
        // Output grid_files
        std::cout << "l_nodes_x:" << std::endl;
        for (const auto& node : l_nodes_x) {
                std::cout << node << std::endl;
        }
        std::cout << "l_nodes_y:" << std::endl;
        for (const auto& node : l_nodes_y) {
                std::cout << node << std::endl;
        }\
        std::cout << "l_nodes_z:" << std::endl;
        for (const auto& node : l_nodes_z) {
                std::cout << node << std::endl;
        }
        std::cout << "l_weights:" << std::endl;
        for (const auto& weight : l_weights) {
                std::cout << weight << std::endl;
        }

        const std::string x1_filepath = "grid_files/x1_legendre_1d_21.grid";

        // Vectors to store grid_files
        std::vector<double> x1_nodes;
        std::vector<double> x1_weights;
        double a;
        double b;
        // Read grid_files from file
        cuslater::read_x1_1d_grid_from_file(x1_filepath,a,b,x1_nodes,x1_weights);
        // Output grid_files
        std::cout << "a:" << a<< std::endl;
        std::cout <<"b:" << b << std::endl;
        std::cout << "x1_nodes:" << std::endl;
        for (const auto& node : x1_nodes) {
                std::cout << node << std::endl;
        }

        std::cout << "x1_weights:" << std::endl;
        for (const auto& weight : x1_weights) {
                std::cout << weight << std::endl;
        }

        return 0;

}
