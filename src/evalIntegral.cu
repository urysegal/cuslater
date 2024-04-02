//
// Created by gkluhana on 26/03/24.
//
#include "../include/evalIntegral.h"
#include <thrust/device_vector.h>
const double pi = 3.14159265358979323846;
#define THREADS_PER_BLOCK 256
#include <thread>
namespace cuslater{

    double evaluateTotalIntegral( double* c,
                                int nr,  int nl,  int nx,
                               const std::string x1_type) {

            // read r grid
            std::cout << "Reading r Grid Files" << std::endl;
            const std::string r_filepath = "grid_files/r_" + std::to_string(nr) + ".grid";
            std::vector<double> r_nodes;
            std::vector<double> r_weights;
            read_r_grid_from_file(r_filepath, r_nodes, r_weights);

            //read l grid
            std::cout << "Reading l Grid Files" << std::endl;
            const std::string l_filepath = "grid_files/l_" + std::to_string(nl) + ".grid";
            std::vector<double> l_nodes_x;
            std::vector<double> l_nodes_y;
            std::vector<double> l_nodes_z;
            std::vector<double> l_weights;
            read_l_grid_from_file(l_filepath,
                                  l_nodes_x, l_nodes_y, l_nodes_z,
                                  l_weights);
            // Read x1 grid
            std::cout << "Reading x1 Grid Files" << std::endl;
            const std::string x1_filepath = "grid_files/x1_"+x1_type+"_1d_" + std::to_string(nx) + ".grid";
            std::vector<double> x1_nodes;
            std::vector<double> x1_weights;
            double a;
            double b;
            read_x1_1d_grid_from_file(x1_filepath, a, b, x1_nodes, x1_weights);

            std::cout << "Initializing Device Variables"<< std::endl;
            double* w = new double[3];
            unsigned int PX = x1_nodes.size();
            int threads = THREADS_PER_BLOCK; // Max threads per block
            int blocks = 160; // Max blocks, better if multiple of SM = 80
	    std::cout << "Total Threads: " << blocks*threads << std::endl;
	    std::cout << "Total Grid Points: " << nx*nx*nx << std::endl; 
	    thrust::device_vector<double> d_x_grid(PX);
	    thrust::device_vector<double> d_x_weights(PX);
	    thrust::device_vector<double> d_r_weights(nr);
	    thrust::device_vector<double> d_l_weights(nl);
	    thrust::device_vector<double> d_c1234(12);
	    thrust::device_vector<double> d_result(blocks);
	    thrust::device_vector<double> d_term12r(1);
	    thrust::copy(x1_nodes.begin(), x1_nodes.end(), d_x_grid.begin());
	    thrust::copy(x1_weights.begin(), x1_weights.end(), d_x_weights.begin());
	    thrust::copy(r_weights.begin(), r_weights.end(), d_r_weights.begin());
	    thrust::copy(l_weights.begin(), l_weights.end(), d_l_weights.begin());
	    thrust::copy(c, c + 12, d_c1234.begin());
	    double* d_sum;
            HANDLE_CUDA_ERROR(cudaMalloc(&d_sum, sizeof(double)));
	    HANDLE_CUDA_ERROR(cudaMemset(d_sum, 0, sizeof(double)));
            double sum = 0.0;
            std::cout << "Evaluating Integral for all values of r and l" << std::endl;
	    cudaProfilerStart();
            for (int i=0; i < nr; ++i) {
                    for (int j = 0; j < nl; ++j) {
                            sum = evaluateInnerPreProcessed(d_c1234,
                                                             r_nodes[i],
                                                             l_nodes_x[j], l_nodes_y[j], l_nodes_z[j],
                                                             d_x_grid, d_x_weights, nx,
							     d_r_weights, i,
							     d_l_weights, j,
							     d_term12r,
                                                             d_result, 
							     d_sum, blocks, threads, 0);
                    }
                    if (i % 10 == 0) {
                    std::cout << "computed for r_i:" << i <<"/" << nr << std::endl;
                        }
            }
            HANDLE_CUDA_ERROR(cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
	    cudaProfilerStop();
            sum *= (4.0/pi);
	    //
            // sum up result, multiply with constant and return
            return sum;
    }

}
