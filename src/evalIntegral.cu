//
// Created by gkluhana on 26/03/24.
//
#include "../include/evalIntegral.h"
#include <thrust/device_vector.h>
const double pi = 3.14159265358979323846;
#include <thread>
#define THREADS_PER_BLOCK 256
__constant__ double d_c[12];
__constant__ double d_x_grid[600];
__constant__ double d_x_weights[600];
namespace cuslater{
	__device__ double fake_sqrt(double x) {return x;}
	__device__ double fake_exp(double x) {return x;}
	__global__ void accumulateSum(double result, 
						double r_weight, 
						double l_weight, 
						double* d_sum)
	{
	    double sum = *d_sum;
	    sum+= result*r_weight*l_weight;
	    *d_sum = sum;
	   	
	}
	
	__global__
	void evaluateIntegrandReduceZ(int x_dim,
	                                      double r, double l_x, double l_y, double l_z,
	                                      double *res)
	    {
	            int idx = blockIdx.x * blockDim.x + threadIdx.x;
	            if ( idx < x_dim*x_dim ) {

			    int y_idx = idx / x_dim;
			    int x_idx = idx % x_dim;
			    double xvalue = d_x_grid[x_idx];
	                    double yvalue = d_x_grid[y_idx];
	
	                    double dx = d_x_weights[x_idx];
	                    double dy = d_x_weights[y_idx];
	                    double xdiffc_1 = xvalue-d_c[0];
	                    double ydiffc_1 = yvalue-d_c[1];
	                    double xdiffc_2 = xvalue-d_c[3];
	                    double ydiffc_2 = yvalue-d_c[4];
	                    double xdiffc_3 = xvalue-d_c[6]+    r*l_x;
	              	    double ydiffc_3 = yvalue-d_c[7]+    r*l_y;
	                    double xdiffc_4 = xvalue-d_c[9]+    r*l_x;
	                    double ydiffc_4 = yvalue-d_c[10]+   r*l_y;

			    double xysq1 = xdiffc_1*xdiffc_1+ydiffc_1*ydiffc_1;
			    double xysq2 = xdiffc_2*xdiffc_2+ydiffc_2*ydiffc_2;
			    double xysq3 = xdiffc_3*xdiffc_3+ydiffc_3*ydiffc_3;
			    double xysq4 = xdiffc_4*xdiffc_4+ydiffc_4*ydiffc_4;
			    res[idx] = 0.0;
			    double dxy = dx*dy;
			    
			    for (int z_idx =0; z_idx <x_dim; ++z_idx){
	                         double zvalue = d_x_grid[z_idx];
	                   	 double dz = d_x_weights[z_idx];
	                   	 // compute function value
	                   	 // exp(-|x1-c1| - |x1-c2| -|x1+r*w_hat - c3| - |x1 + rw_hat -c4|
	                   	 // constants needed: r, c1,c2,c3,c4
	                   	 // First six are precomputed
	                         double zdiffc_1 = zvalue-d_c[2];
	                         double zdiffc_2 = zvalue-d_c[5];
	                   	 double zdiffc_3 = zvalue-d_c[8]+    r*l_z;
	                   	 double zdiffc_4 = zvalue-d_c[11]+   r*l_z;
	               	 	 double term1 = sqrt(float(xysq1 + zdiffc_1 * zdiffc_1));
	                	 double term2 = sqrt(float(xysq2 + zdiffc_2 * zdiffc_2));
	                   	 double term3 = sqrt(float(xysq3 + zdiffc_3*zdiffc_3 ));
	                   	 double term4 = sqrt(float(xysq4 + zdiffc_4*zdiffc_4));
	                   	 double exponent =  -term1 - term2- term3 -term4 + r ;
	                   	 res[idx] += exp(float(exponent))*dxy*dz;
	            	    }
		}
	
	 } //evaluateReduceInnerIntegrandz

	double evaluateInnerSumX1_rl_preAllocated(unsigned int x_axis_points,
	                                 double r, double l_x, double l_y, double l_z,
					 double r_weight, double l_weight,
	                                 thrust::device_vector<double>& d_term12r, 
	                                 thrust::device_vector<double>& d_result, 
	                                 double* d_sum, 
	                                 int blocks, 
	                                 int threads, 
	                                 int gpu_num) 
	    {
		    HANDLE_CUDA_ERROR(cudaSetDevice(gpu_num));
	
		     evaluateIntegrandReduceZ<<<blocks,threads>>>(  
	                                   x_axis_points,r, l_x, l_y, l_z,                
		                           raw_pointer_cast(d_result.data()));
		        //for(long unsigned int i = 0; i < d_result.size(); i++)
	        //std::cout << "d_result[" << i << "] = " << d_result[i] << std::endl;
	            //Reduce vector on GPU within each block
		    double sum = thrust::reduce(d_result.begin(),d_result.end(),(double) 0.0, thrust::plus<double>());
		    // Accumulate result on device
	    	        accumulateSum<<<1, 1>>>(sum,
		        			r_weight, 
		        			l_weight, 
		        			d_sum);
	            return 0.0;
	    }//evaluateInner


    double evaluateFourCenterIntegral( double* c,
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
            int blocks = (PX*PX+threads-1)/threads; // Max blocks, better if multiple of SM = 80
	    std::cout << "Total Threads: " << blocks*threads << std::endl;
	    std::cout << "Total Grid Points: " << nx*nx*nx << std::endl; 
	    
	    cudaMemcpyToSymbol(d_c, c, 12*sizeof(double));
	    //std::cout << x1_nodes[3] << std::endl;
            cudaMemcpyToSymbol(d_x_grid, x1_nodes.data(), PX*sizeof(double));
	    cudaMemcpyToSymbol(d_x_weights, x1_weights.data(), PX*sizeof(double));
	    double* d_sum;
            thrust::device_vector<double> d_r_weights(nr);
            thrust::device_vector<double> d_l_weights(nl);
	    thrust::device_vector<double> d_result(PX*PX);
            thrust::device_vector<double> d_term12r(1);

            HANDLE_CUDA_ERROR(cudaMalloc(&d_sum, sizeof(double)));
	    HANDLE_CUDA_ERROR(cudaMemset(d_sum, 0, sizeof(double)));
            double sum = 0.0;
            std::cout << "Evaluating Integral for all values of r and l" << std::endl;
            for (int i=0; i < nr; ++i) {
                    for (int j = 0; j < nl; ++j) {
                            sum = evaluateInnerSumX1_rl_preAllocated(nx,
							     r_nodes[i], l_nodes_x[j],l_nodes_y[j],l_nodes_z[j],
							     r_weights[i], l_weights[j],
							     d_term12r,
                                                             d_result, 
							     d_sum, blocks, threads, 0);
                    }
                    if (i % 10 == 0) {
                    std::cout << "computed for r_i:" << i <<"/" << nr << std::endl;
                        }
            }
            HANDLE_CUDA_ERROR(cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
            sum *= (4.0/pi);
	    //
            // sum up result, multiply with constant and return
            return sum;
    }

}
