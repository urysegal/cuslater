//
// Created by gkluhana on 26/03/24.
//
#include "../include/evalIntegral.h"
const double pi = 3.14159265358979323846;
#define THREADS_PER_BLOCK 128

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
            double* d_x_grid = nullptr;
            double* d_x_weights = nullptr;
            double* d_r_weights = nullptr;
            double* d_l_weights = nullptr;
            double* d_c1234 = nullptr;
            double* d_result = nullptr;
            double* d_term12r = nullptr;
	    double* d_sum;

            int threads = THREADS_PER_BLOCK; // Max threads per block
            int blocks = (nx*nx*nx + threads -1)/threads; // Max blocks, better if multiple of SM = 80

            HANDLE_CUDA_ERROR(cudaMalloc(&d_result, blocks*sizeof(double)));
            HANDLE_CUDA_ERROR(cudaMalloc(&d_term12r, PX*PX*PX*sizeof(double)));
            HANDLE_CUDA_ERROR(cudaMalloc(&d_x_grid, PX*sizeof(double)));
            HANDLE_CUDA_ERROR(cudaMalloc(&d_x_weights, PX*sizeof(double)));
            HANDLE_CUDA_ERROR(cudaMalloc(&d_r_weights, nr*sizeof(double)));
            HANDLE_CUDA_ERROR(cudaMalloc(&d_l_weights, nl*sizeof(double)));
            HANDLE_CUDA_ERROR(cudaMalloc(&d_c1234, 12*sizeof(double)));
            HANDLE_CUDA_ERROR(cudaMalloc(&d_sum, sizeof(double)));

            HANDLE_CUDA_ERROR(cudaMemcpy(d_x_grid, x1_nodes.data(), PX*sizeof(double), cudaMemcpyHostToDevice));
            HANDLE_CUDA_ERROR(cudaMemcpy(d_x_weights, x1_weights.data(), PX*sizeof(double), cudaMemcpyHostToDevice));
            HANDLE_CUDA_ERROR(cudaMemcpy(d_r_weights, r_weights.data(), nr*sizeof(double), cudaMemcpyHostToDevice));
            HANDLE_CUDA_ERROR(cudaMemcpy(d_l_weights, l_weights.data(), nl*sizeof(double), cudaMemcpyHostToDevice));
            HANDLE_CUDA_ERROR(cudaMemcpy(d_c1234, c, 12*sizeof(double), cudaMemcpyHostToDevice));
	    HANDLE_CUDA_ERROR(cudaMemset(d_sum, 0, sizeof(double)));
            // Check Number of points are correct for x1 grid
            // assert(PX== (x_axis_points));
            // assert(PY== (y_axis_points));
            // assert(PZ== (z_axis_points));
            double sum = 0.0;
            std::cout << "Evaluating Integral for all values of r and l" << std::endl;
	    cudaProfilerStart();
            for (int i=0; i < nr; ++i) {
			
		    evaluateConstantTerm<<<blocks,threads>>>(d_c1234, 
						d_x_grid, d_x_grid, d_x_grid,
						nx, 
						r_nodes[i], d_term12r);
                    for (int j = 0; j < nl; ++j) {
                            w[0] = l_nodes_x[j];
                            w[1] = l_nodes_y[j];
                            w[2] = l_nodes_z[j];
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
            HANDLE_CUDA_ERROR(cudaFree(d_result));
            HANDLE_CUDA_ERROR(cudaFree(d_x_grid));
            HANDLE_CUDA_ERROR(cudaFree(d_x_weights));
            HANDLE_CUDA_ERROR(cudaFree(d_c1234));
	    HANDLE_CUDA_ERROR(cudaFree(d_sum));
            // sum up result, multiply with constant and return
            return sum;
    }

}
