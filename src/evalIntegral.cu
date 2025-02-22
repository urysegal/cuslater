//
// Created by gkluhana on 26/03/24.
//
#include "../include/evalIntegral.h"
#include <thrust/device_vector.h>
const double pi = 3.14159265358979323846;
#include <thread>
#define THREADS_PER_BLOCK 128 
__constant__ real_t d_c[12];
__constant__ real_t d_alpha[4];
__constant__ real_t d_x_grid[600];
__constant__ real_t d_x_weights[600];
__constant__ real_t d_y_grid[600];
__constant__ real_t d_y_weights[600];
__constant__ real_t d_z_grid[600];
__constant__ real_t d_z_weights[600];
namespace cuslater{
	__global__ void accumulateSum(double result, 
						real_t r_weight, 
						real_t l_weight, 
						double* __restrict__ d_sum)
	{
	    double sum = *d_sum;
	    sum+= result*r_weight*l_weight;
	    *d_sum = sum;
	   	
	}
	
	__global__
	void evaluateIntegrandReduceZ(int x_dim, int y_dim, int z_dim,
	                                      real_t r, real_t l_x, real_t l_y, real_t l_z,
	                                      double * __restrict__ res)
	    {
	        int idx = blockIdx.x * blockDim.x + threadIdx.x;
		    if (idx < x_dim*x_dim) {
			    int y_idx = idx / x_dim;
			    int x_idx = idx % x_dim;
			    real_t xvalue = d_x_grid[x_idx];
	            real_t yvalue = d_x_grid[y_idx];
	
	            real_t dx = d_x_weights[x_idx];
	            real_t dy = d_x_weights[y_idx];
	            real_t xdiffc_1 = xvalue-d_c[0];
	            real_t ydiffc_1 = yvalue-d_c[1];
	            real_t xdiffc_2 = xvalue-d_c[3];
	            real_t ydiffc_2 = yvalue-d_c[4];
	            real_t xdiffc_3 = xvalue-d_c[6]+    r*l_x;
	            real_t ydiffc_3 = yvalue-d_c[7]+    r*l_y;
	            real_t xdiffc_4 = xvalue-d_c[9]+    r*l_x;
	            real_t ydiffc_4 = yvalue-d_c[10]+   r*l_y;

			    real_t xysq1 = xdiffc_1*xdiffc_1+ydiffc_1*ydiffc_1;
			    real_t xysq2 = xdiffc_2*xdiffc_2+ydiffc_2*ydiffc_2;
			    real_t xysq3 = xdiffc_3*xdiffc_3+ydiffc_3*ydiffc_3;
			    real_t xysq4 = xdiffc_4*xdiffc_4+ydiffc_4*ydiffc_4;
			    double dxy = dx*dy;
			    double v = 0.0; 
			    for (int z_idx =0; z_idx <x_dim; ++z_idx){
	                         real_t zvalue = d_x_grid[z_idx];
	                   	 real_t dz = d_x_weights[z_idx];
	                   	 // compute function value
	                   	 // exp(-|x1-c1| - |x1-c2| -|x1+r*w_hat - c3| - |x1 + rw_hat -c4|
	                   	 // constants needed: r, c1,c2,c3,c4
	                   	 // First six are precomputed
	                     real_t zdiffc_1 = zvalue-d_c[2];
	                     real_t zdiffc_2 = zvalue-d_c[5];
	                   	 real_t zdiffc_3 = zvalue-d_c[8]+    r*l_z;
	                   	 real_t zdiffc_4 = zvalue-d_c[11]+   r*l_z;
	               	 	 real_t term1 = d_alpha[0] * sqrt(xysq1 + zdiffc_1 * zdiffc_1);
	                	 real_t term2 = d_alpha[1] * sqrt(xysq2 + zdiffc_2 * zdiffc_2);
	                   	 real_t term3 = d_alpha[2] * sqrt(xysq3 + zdiffc_3*zdiffc_3 );
	                   	 real_t term4 = d_alpha[3] * sqrt(xysq4 + zdiffc_4*zdiffc_4);
	                   	 real_t exponent =  -term1 - term2- term3 -term4 + r ;
	                   	 v += exp(exponent)*dxy*dz;
	            	    }
			   res[idx] = v;
		}
	} //evaluateReduceInnerIntegrandz

	double evaluateInnerSum(unsigned int x_axis_points, unsigned int y_axis_points, unsigned int z_axis_points,
	                                 real_t r, real_t l_x, real_t l_y, real_t l_z,
					 real_t r_weight, real_t l_weight,
	                                 thrust::device_vector<double>& __restrict__ d_result, 
					 thrust::device_vector<double>& __restrict__ d_sorted,
	                                 int i, int j, int nl,
					 double* __restrict__ d_sum, 
	                                 int blocks, 
	                                 int threads, 
	                                 int gpu_num) 
	{
         HANDLE_CUDA_ERROR(cudaSetDevice(gpu_num));
	
         evaluateIntegrandReduceZ<<<blocks,threads>>>(
	                               x_axis_points, y_axis_points, z_axis_points ,r, l_x, l_y, l_z,
	                           raw_pointer_cast(d_result.data()));
        //for(long unsigned int i = 0; i < d_result.size(); i++)
	    //std::cout << "d_result[" << i << "] = " << d_result[i] << std::endl;
        //Reduce vector on GPU within each block
	    //thrust::sort(d_result.begin(), d_result.end());
	    double delta_sum = thrust::reduce(d_result.begin(),d_result.end(),(double) 0.0, thrust::plus<double>());
	    
	    d_sorted[ (i)*nl + (j) ] = delta_sum*r_weight*l_weight;
	    return delta_sum;
	}//evaluateInner


    double evaluateFourCenterIntegral( real_t* c, real_t* alpha,
                                int nr,  int nl,  int nx, int ny, int nz,
                               const std::string x1_type, double tol, bool check_zero_cond) {

	real_t normdiff13 = sqrt((c[0] - c[6]) * (c[0] - c[6]) +
                            (c[1] - c[7]) * (c[1] - c[7]) +
                            (c[2] - c[8]) * (c[2] - c[8]));
    	real_t normdiff24 = sqrt((c[3] - c[9]) * (c[3] - c[9]) +
                            (c[4] - c[10]) * (c[4] - c[10]) +
                            (c[5] - c[11]) * (c[5] - c[11]));
    	real_t cond = min(alpha[0], alpha[2]) * normdiff13 + min(alpha[1], alpha[3]) * normdiff24;
	real_t r0 = 1;
    	int inv_machine_eps = 1e8;
    	if (check_zero_cond && cond > log(r0 * inv_machine_eps)){
    		printf("zero condition: check wheter %f > %f, \n", cond, log(r0 * inv_machine_eps));
        	std::cout << "Zero condition met" << std::endl;
		return 0;
	} else{
	
        // read r grid
        std::cout << "Reading r Grid Files" << std::endl;
        const std::string r_filepath = "grid_files/r_" + std::to_string(nr) + ".grid";
        std::vector<real_t> r_nodes;
        std::vector<real_t> r_weights;
        read_r_grid_from_file(r_filepath, r_nodes, r_weights);

        //read l grid
        std::cout << "Reading l Grid Files" << std::endl;
        const std::string l_filepath = "grid_files/l_" + std::to_string(nl) + ".grid";
        std::vector<real_t> l_nodes_x;
        std::vector<real_t> l_nodes_y;
        std::vector<real_t> l_nodes_z;
        std::vector<real_t> l_weights;
        read_l_grid_from_file(l_filepath,
                              l_nodes_x, l_nodes_y, l_nodes_z,
                              l_weights);
        // Read x1 grid
        std::cout << "Reading x1 Grid Files" << std::endl;
        const std::string x1_filepath = "grid_files_adap/leg64/x1_"+x1_type+"_1d_" + std::to_string(nx) + ".grid";
        std::vector<real_t> x1_standard_nodes;
        std::vector<real_t> x1_standard_weights;
        read_x1_1d_grid_from_file(x1_filepath, x1_standard_nodes, x1_standard_weights);
        std::vector<real_t> x1_nodes;
        std::vector<real_t> x1_weights;
	std::vector<real_t> y1_nodes;
	std::vector<real_t> y1_weights;
	std::vector<real_t> z1_nodes;
	std::vector<real_t> z1_weights;
	//real_t ax = -10;
	//real_t bx = 11;
	//real_t ay = -10;
	//real_t by = 11;
	//real_t az= -10;
	//real_t bz=11;
	real_t dx = std::abs(c[0]-c[3]);
	real_t dy = std::abs(c[1]-c[4]);
	real_t dz = std::abs(c[2]-c[5]);
	real_t lx = 18.0 + dx; // why 18?
	real_t ly = 18.0 + dy;
	real_t lz = 18.0 + dz;
	real_t mx = (c[0] + c[3])/2;
	real_t my = (c[1] + c[4])/2;
	real_t mz = (c[2] + c[5])/2;
	real_t ax = mx - (lx/2);
	real_t bx = mx + (lx/2);
	real_t ay = my - (ly/2);
	real_t by = my + (ly/2);
	real_t az = mz - (lz/2);
	real_t bz = mz + (lz/2);
	generate_x1_from_std(ax,bx, x1_standard_nodes, x1_standard_weights, x1_nodes, x1_weights); 
	generate_x1_from_std(ay,by, x1_standard_nodes, x1_standard_weights, y1_nodes, y1_weights); 
	generate_x1_from_std(az,bz, x1_standard_nodes, x1_standard_weights, z1_nodes, z1_weights); 

        //read_x1_1d_grid_from_file(x1_filepath, a, b, x1_nodes, x1_weights);
        //read_x1_1d_grid_from_file(x1_filepath, a, b, y1_nodes, y1_weights);
        //read_x1_1d_grid_from_file(x1_filepath, a, b, z1_nodes, z1_weights);

        std::cout << "Initializing Device Variables"<< std::endl;
        double* w = new double[3];
        unsigned int PX = x1_nodes.size();
        unsigned int PY = y1_nodes.size();
        unsigned int PZ = z1_nodes.size();

        int threads = THREADS_PER_BLOCK; // Max threads per block
        int blocks = (PX*PY+threads-1)/threads; // Max blocks, better if multiple of SM = 80
	    std::cout << "Total Threads: " << blocks*threads << std::endl;
	    std::cout << "Total Grid Points: " << nx*ny*nz << std::endl; 
	    
	cudaMemcpyToSymbol(d_c, c, 12*sizeof(real_t));
	cudaMemcpyToSymbol(d_alpha, alpha, 4*sizeof(real_t));
        cudaMemcpyToSymbol(d_x_grid, x1_nodes.data(), PX*sizeof(real_t));
	cudaMemcpyToSymbol(d_x_weights, x1_weights.data(), PX*sizeof(real_t));
        cudaMemcpyToSymbol(d_y_grid, y1_nodes.data(), PY*sizeof(real_t));
	cudaMemcpyToSymbol(d_y_weights, y1_weights.data(), PY*sizeof(real_t));
        cudaMemcpyToSymbol(d_z_grid, z1_nodes.data(), PZ*sizeof(real_t));
	cudaMemcpyToSymbol(d_z_weights, z1_weights.data(), PZ*sizeof(real_t));

	double* d_sum;
        thrust::device_vector<real_t> d_r_weights(nr);
        thrust::device_vector<real_t> d_l_weights(nl);
	thrust::device_vector<double> d_result(PX*PX);
	thrust::device_vector<double> d_sorted(nr*nl,0);

        HANDLE_CUDA_ERROR(cudaMalloc(&d_sum, sizeof(double)));
	HANDLE_CUDA_ERROR(cudaMemset(d_sum, 0, sizeof(double)));
        double sum = 0.0;
        double delta_sum=0.0;
	int r_skipped = 0;

	std::cout << "Evaluating Integral for all values of r and l with\n";
        std::cout << "  a1=" << alpha[0] << ", a2=" << alpha[1] << ", a3=" << alpha[2]
                  << ", a4=" << alpha[3] << "\n";
        std::cout << "  c1 = (" << c[0] << ", " << c[1] << ", " << c[2] << ")\n";
        std::cout << "  c2 = (" << c[3] << ", " << c[4] << ", " << c[5] << ")\n";
        std::cout << "  c3 = (" << c[6] << ", " << c[7] << ", " << c[8] << ")\n";
        std::cout << "  c4 = (" << c[9] << ", " << c[10] << ", " << c[11] << ")\n";
        std::cout << "  Tolerance = " << tol << std::endl;
	std::cout << " Legendre Grid Parameters: " << std::endl;
	std::cout << " xgrid (ax , bx) : (" << ax << " , " << bx << ")" << std::endl;
	std::cout << " ygrid (ay , by) : (" << ay << " , " << by << ")" << std::endl;
	std::cout << " zgrid (az , bz) : (" << az << " , " << bz << ")" << std::endl;

                for (int j = 0; j < nl; ++j) {
        		    for (int i=0; i < nr; ++i) {
                        delta_sum = evaluateInnerSum(nx, ny, nz,
						     r_nodes[i], l_nodes_x[j],l_nodes_y[j],l_nodes_z[j],
						     r_weights[i], l_weights[j],
                                                         d_result, d_sorted ,i,j,nl,
						     d_sum, blocks, threads, 0);
	            		if (delta_sum < tol )  {
	            			r_skipped += nr - i;
	            			break;
	            		}
                    }
                    if (j % 100 == 0) {
                    	std::cout << "computed for l_j:" << j <<"/" << nl << std::endl;
                    }
        }
        //HANDLE_CUDA_ERROR(cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
	//thrust::sort(d_sorted.begin(), d_sorted.end());
	//thrust:: host_vector<double> h_vector = d_sorted;
	//for (long unsigned int i = 0; i < h_vector.size(); ++i) {
        //	std::cout << h_vector[i] << " ";
    	//}
	sum  = thrust::reduce(d_sorted.begin(), d_sorted.end(), 0.0);
	std::cout << "sum before multiplication " << sum << std::endl;
        sum *= (4.0/pi) * std::pow(alpha[0] * alpha[1] * alpha[2] * alpha[3], 1.5);

        // sum up result, multiply with constant and return
        std::cout << "Tolerance: " << tol << std::endl;
        std::cout << "Total values of r skipped for different l's: " << r_skipped << "/" << nr*nl << std::endl;
        return sum;
    }
  }
}
