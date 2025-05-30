//
// Created by gkluhana on 26/03/24.
//
#include "../include/grids.h"
#include <fstream>
#include <iostream>

namespace cuslater {
    // delete this // clean this .grid files leg32 and leg64 from grid_adap

    void read_r_grid_from_file(const std::string& filepath, std::vector<real_t>& r_nodes,
                               std::vector<real_t>& r_weights) {
        // Open the file
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filepath << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Clear vectors to ensure they are empty
        r_nodes.clear();
        r_weights.clear();

        // Read data from file
        real_t node, weight;
        while (file >> node >> weight) {
            // Store data in vectors
            r_nodes.push_back(node);
            r_weights.push_back(weight);
        }

        // Close the file
        file.close();
    }
    void read_l_grid_from_file(const std::string& filepath, std::vector<real_t>& l_nodes_x,
                               std::vector<real_t>& l_nodes_y, std::vector<real_t>& l_nodes_z,
                               std::vector<real_t>& l_weights) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filepath << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Clear vectors to ensure they are empty
        l_nodes_x.clear();
        l_nodes_y.clear();
        l_nodes_z.clear();
        l_weights.clear();

        // Read data from file
        double nodex, nodey, nodez, weight;
        while (file >> nodex >> nodey >> nodez >> weight) {
            // Store data in vectors
            l_nodes_x.push_back(nodex);
            l_nodes_y.push_back(nodey);
            l_nodes_z.push_back(nodez);
            l_weights.push_back(weight);
        }
        // Close the file
        file.close();
    }

    void generate_x1_from_std(real_t a, real_t b, const std::vector<real_t>& x1_standard_nodes,
                              const std::vector<real_t>& x1_standard_weights,
                              std::vector<real_t>& x1_nodes, std::vector<real_t>& x1_weights) {
        real_t shift  = (a + b) / 2.0;
        real_t factor = (b - a) / 2.0;
        real_t node;
        real_t weight;
        for (std::vector<real_t>::size_type i = 0; i < x1_standard_nodes.size(); ++i) {
            node = x1_standard_nodes[i] * factor + shift;
            x1_nodes.push_back(node);
            weight = x1_standard_weights[i] * factor;
            x1_weights.push_back(weight);
        }
    }

    void read_x1_1d_grid_from_file(const std::string& filepath, std::vector<real_t>& x1_nodes,
                                   std::vector<real_t>& x1_weights) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filepath << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Clear vectors to ensure they are empty
        x1_nodes.clear();
        x1_weights.clear();

        // Read data from file
        real_t node, weight;
        while (file >> node >> weight) {
            // Store data in vectors
            x1_nodes.push_back(node);
            x1_weights.push_back(weight);
        }

        // Close the file
        file.close();
    }
} // namespace cuslater
