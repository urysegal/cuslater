//
// Created by gkluhana on 26/03/24.
//
#include "../include/evalIntegral.h"
#include <thrust/device_vector.h>
const double pi = 3.14159265358979323846;
#include <thread>
#include <cmath> //for abs function
#define THREADS_PER_BLOCK 128 
__constant__ float d_c[12];
__constant__ float d_x_grid[600];
__constant__ float d_x_weights[600];
__constant__ float d_y_grid[600];
__constant__ float d_y_weights[600];
__constant__ float d_z_grid[600];
__constant__ float d_z_weights[600];

namespace cuslater{
	__global__ void accumulateSum(double result, 
						float r_weight, 
						float l_weight, 
						double* __restrict__ d_sum)
	{
	    double sum = *d_sum;
	    sum+= result*r_weight*l_weight;
	    *d_sum = sum;
	   	
	}
	
	__global__
	void evaluateIntegrandReduceZ(int x_dim,
	                                      float r, float l_x, float l_y, float l_z,
	                                      double * __restrict__ res)
	    {
	            int idx = blockIdx.x * blockDim.x + threadIdx.x;
		    if (idx < x_dim*x_dim) {
			    int y_idx = idx / x_dim;
			    int x_idx = idx % x_dim;
			    float xvalue = d_x_grid[x_idx];
	                    float yvalue = d_y_grid[y_idx];
	
	                    float dx = d_x_weights[x_idx];
	                    float dy = d_y_weights[y_idx];
	                    float xdiffc_1 = xvalue-d_c[0];
	                    float ydiffc_1 = yvalue-d_c[1];
	                    float xdiffc_2 = xvalue-d_c[3];
	                    float ydiffc_2 = yvalue-d_c[4];
	                    float xdiffc_3 = xvalue-d_c[6]+    r*l_x;
	              	    float ydiffc_3 = yvalue-d_c[7]+    r*l_y;
	                    float xdiffc_4 = xvalue-d_c[9]+    r*l_x;
	                    float ydiffc_4 = yvalue-d_c[10]+   r*l_y;

			    float xysq1 = xdiffc_1*xdiffc_1+ydiffc_1*ydiffc_1;
			    float xysq2 = xdiffc_2*xdiffc_2+ydiffc_2*ydiffc_2;
			    float xysq3 = xdiffc_3*xdiffc_3+ydiffc_3*ydiffc_3;
			    float xysq4 = xdiffc_4*xdiffc_4+ydiffc_4*ydiffc_4;
			    double dxy = dx*dy;
			    double v = 0.0; 
			    for (int z_idx =0; z_idx <x_dim; ++z_idx){
	                         float zvalue = d_z_grid[z_idx];
	                   	 float dz = d_z_weights[z_idx];
	                   	 // compute function value
	                   	 // exp(-|x1-c1| - |x1-c2| -|x1+r*w_hat - c3| - |x1 + rw_hat -c4|
	                   	 // constants needed: r, c1,c2,c3,c4
	                   	 // First six are precomputed
	                         float zdiffc_1 = zvalue-d_c[2];
	                         float zdiffc_2 = zvalue-d_c[5];
	                   	 float zdiffc_3 = zvalue-d_c[8]+    r*l_z;
	                   	 float zdiffc_4 = zvalue-d_c[11]+   r*l_z;
	               	 	 float term1 = sqrt(xysq1 + zdiffc_1 * zdiffc_1);
	                	 float term2 = sqrt(xysq2 + zdiffc_2 * zdiffc_2);
	                   	 float term3 = sqrt(xysq3 + zdiffc_3*zdiffc_3 );
	                   	 float term4 = sqrt(xysq4 + zdiffc_4*zdiffc_4);
	                   	 float exponent =  -term1 - term2- term3 -term4 + r ;
	                   	 v += exp(exponent)*dxy*dz;
	            	    }
			   res[idx] = v;
		}
	 } //evaluateReduceInnerIntegrandz

	double evaluateInnerSumX1_rl_preAllocated(unsigned int x_axis_points,
	                                 float r, float l_x, float l_y, float l_z,
					 float r_weight, float l_weight,
	                                 thrust::device_vector<double>& __restrict__ d_result, 
	                                 double* __restrict__ d_sum, 
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
		    double delta_sum = thrust::reduce(d_result.begin(),d_result.end(),(double) 0.0, thrust::plus<double>());
		    // Accumulate result on device
	    	        accumulateSum<<<1, 1>>>(delta_sum,
		        			r_weight, 
		        			l_weight, 
		        			d_sum);
	            return delta_sum;
	    }//evaluateInner

    void generate_x1_from_std(float a, float b, const std::vector<float>& x1_standard_nodes, const std::vector<float>& x1_standard_weights, std::vector<float>& x1_nodes, std::vector<float>& x1_weights) {
		float shift = (a + b) / 2.0;
		float factor = (a - b) / 2.0;
		float node;
		float weight;
		for (std::vector<float>::size_type i = 0; i < x1_standard_nodes.size(); ++i) {
			node = x1_standard_nodes[i] * factor + shift;
			x1_nodes.push_back(node);
			weight = x1_standard_weights[i] * factor;
			x1_weights.push_back(weight);
		}
	}




    double evaluateFourCenterIntegral( float* c,
                                int nr,  int nl,  int nx,
                               const std::string x1_type, double tol) {

            // read r grid
            std::cout << "Reading r Grid Files" << std::endl;
            const std::string r_filepath = "grid_files/r_" + std::to_string(nr) + ".grid";
            std::vector<float> r_nodes;
            std::vector<float> r_weights;
            read_r_grid_from_file(r_filepath, r_nodes, r_weights);

            //read l grid
            std::cout << "Reading l Grid Files" << std::endl;
            const std::string l_filepath = "grid_files/l_" + std::to_string(nl) + ".grid";
            std::vector<float> l_nodes_x;
            std::vector<float> l_nodes_y;
            std::vector<float> l_nodes_z;
            std::vector<float> l_weights;
            read_l_grid_from_file(l_filepath,
                                  l_nodes_x, l_nodes_y, l_nodes_z,
                                  l_weights);
            // Read x1 grid
            // Avleen: You can call the funciton three times for initializing different grids for each dimension 

	        std::cout << "Reading x1 Grid Files" << std::endl;
            const std::string x1_filepath = "grid_files_adap/leg64/x1_"+ x1_type +"_1d_" + std::to_string(nx) + ".grid";
            std::vector<float> x1_standard_nodes;
            std::vector<float> x1_standard_weights;
            read_x1_1d_grid_from_file(x1_filepath, x1_standard_nodes, x1_standard_weights);

            std::vector<float> x1_nodes;
            std::vector<float> x1_weights;
            std::vector<float> y1_nodes;
            std::vector<float> y1_weights;
            std::vector<float> z1_nodes;
            std::vector<float> z1_weights;
	    // Avleen : You can change these to the min max functions with the centers
	    // the centers are stored in vector c as [c1x,c1y,c1z, and so on till c4z] 
			float delta = 1;
			float ax = std::min(c[0], c[3]) - (10.0 / (std::abs(c[0] - c[3])+delta));
			float bx = std::max(c[0], c[3]) + (10.0 / (std::abs(c[0] - c[3])+delta));
			float ay = std::min(c[1], c[4]) - (10.0 / (std::abs(c[1] - c[4])+delta));
			float by = std::max(c[1], c[4]) + (10.0 / (std::abs(c[1] - c[4])+delta));
			float az = std::min(c[2], c[5]) - (10.0 / (std::abs(c[2] - c[5])+delta));
			float bz = std::max(c[2], c[5]) + (10.0 / (std::abs(c[2] - c[5])+delta));
            
	    generate_x1_from_std(ax,bx, x1_standard_nodes, x1_standard_weights, x1_nodes, x1_weights); 
	    generate_x1_from_std(ay,by, x1_standard_nodes, x1_standard_weights, y1_nodes, y1_weights); 
	    generate_x1_from_std(az,bz, x1_standard_nodes, x1_standard_weights, z1_nodes, z1_weights); 

	    std::cout << "Initializing Device Variables"<< std::endl;
            double* w = new double[3];
            unsigned int PX = x1_nodes.size();
            unsigned int PY = y1_nodes.size();
            unsigned int PZ = z1_nodes.size();

            int threads = THREADS_PER_BLOCK; // Max threads per block
            int blocks = (PX*PX+threads-1)/threads; // Max blocks, better if multiple of SM = 80
	    std::cout << "Total Threads: " << blocks*threads << std::endl;
	    std::cout << "Total Grid Points: " << nx*nx*nx << std::endl; 
	    
	    cudaMemcpyToSymbol(d_c, c, 12*sizeof(float));
	    //std::cout << x1_nodes[3] << std::endl;
            cudaMemcpyToSymbol(d_x_grid, x1_nodes.data(), PX*sizeof(float));
	    cudaMemcpyToSymbol(d_x_weights, x1_weights.data(), PX*sizeof(float));
            cudaMemcpyToSymbol(d_y_grid, y1_nodes.data(), PY*sizeof(float));
	    cudaMemcpyToSymbol(d_y_weights, y1_weights.data(), PY*sizeof(float));
            cudaMemcpyToSymbol(d_z_grid, z1_nodes.data(), PZ*sizeof(float));
	    cudaMemcpyToSymbol(d_z_weights, z1_weights.data(), PZ*sizeof(float));
	    
	    double* d_sum;
            thrust::device_vector<float> d_r_weights(nr);
            thrust::device_vector<float> d_l_weights(nl);
	    thrust::device_vector<double> d_result(PX*PX);

            HANDLE_CUDA_ERROR(cudaMalloc(&d_sum, sizeof(double)));
	    HANDLE_CUDA_ERROR(cudaMemset(d_sum, 0, sizeof(double)));
            double sum = 0.0;
            double delta_sum=0.0;
	    int r_skipped = 0; //Number of r values skipped for all Lebedev points
            std::cout << "Evaluating Integral for all values of r and l" << std::endl;
                    for (int j = 0; j < nl; ++j) { //loop over each Lebedev point
            		for (int i=0; i < nr; ++i) { //loop over each r-Laguerre point
					//calculate the sum over entire x_1 grid with (r_i,l_j)
                            delta_sum = evaluateInnerSumX1_rl_preAllocated(nx,
							     r_nodes[i], l_nodes_x[j],l_nodes_y[j],l_nodes_z[j],
							     r_weights[i], l_weights[j],
                                                             d_result, 
							     d_sum, blocks, threads, 0); 
			if (delta_sum < tol )  { 
			//if the sum over x_1 is smaller than tol for given l_j, skip the next r_i
		//		std::cout << "delta_sum smaller than: "<< tol<< " l_j,r= " 
		//					<< j << " , " << r_nodes[i] << std::endl;
				std::cout << "delta_sum smaller than: "<< tol<< " l_j,r= " << j << " , " << r_nodes[i] << std::endl;
				r_skipped += nr - i; // the number of skipped itertaions, as i iterations have been evaluated
				break;//breaks loop over r_i
			} 
                    }
                    if (j % 100 == 0) {
                    	std::cout << "computed for l_j:" << j <<"/" << nl << std::endl;
                    }
            }
            HANDLE_CUDA_ERROR(cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
            sum *= (4.0/pi);
	    //
            // sum up result, multiply with constant and return
            std::cout << "Tolerance: " << tol << std::endl;
            std::cout << "Total values of r skipped for different l's: " << r_skipped << "/" << nr*nl << std::endl;
            return sum;
    }

}
