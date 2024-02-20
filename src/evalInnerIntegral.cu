#include "cudaProfiler.h"
#include "../include/evalInnerIntegral.h"
#include <chrono>
<<<<<<< HEAD
#include "cooperative_groups.h"
namespace cg = cooperative_groups;

namespace cuslater {
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
=======
#include <cuda_runtime.h>
#include <cutensor.h>
#include "../include/evalInnerIntegral.h"
namespace cuslater {
__constant__ double c1[3];
__constant__ double c2[3];
__constant__ double c3[3];
__constant__ double c4[3];
__constant__ double w[3];


__global__
void evalInnerIntegral(	double* d_x_grid_points,
			double* d_y_grid_points,
			double* d_z_grid_points,  
			int x_dim,
                          double r, double *res)
{
	int bx = blockIdx.x;    
	int by = blockIdx.y;
    	int bz = blockIdx.z;
    	int tx = threadIdx.x;
    	int ty = threadIdx.y;
    	int tz = threadIdx.z;
	int h = gridDim.z*blockDim.z;
	int d = gridDim.y*blockDim.y;
        int xpos = 256*blockIdx.x;
	xpos += threadIdx.x;
	int idx = h*d*(blockDim.x*bx + tx)+ d*(blockDim.y*by + ty)+ (blockDim.z*bz + tz);
        if ( idx < x_dim*x_dim*x_dim ) {
        double xvalue = d_x_grid_points[xpos];
        double yvalue = d_y_grid_points[blockIdx.y];
        double zvalue = d_z_grid_points[blockIdx.z];
>>>>>>> f54066d (fix and test file)

    return __longlong_as_double(old);
 	// compute function value
	// exp(-|x1-c1| - |x1-c2| -|x1+r*w_hat - c3| - |x1 + rw_hat -c4|
	// constants needed: r, c1,c2,c3,c4
	double term1 = sqrt( (xvalue-c1[0])*(xvalue-c1[0]) + (yvalue-c1[1])*(yvalue-c1[1]) + (zvalue - c1[2])*(zvalue-c1[2]));
	double term2 = sqrt( (xvalue-c2[0])*(xvalue-c2[0]) + (yvalue-c2[1])*(yvalue-c2[1]) + (zvalue - c2[2])*(zvalue-c2[2]));
	double term3 = sqrt( (xvalue-c3[0]+r*w[0])*(xvalue-c3[0]+r*w[0]) + (yvalue-c3[1]+r*w[1])*(yvalue-c3[1]+r*w[1]) + (zvalue - c3[2]+r*w[2])*(zvalue-c3[2]+r*w[2]));
	double term4 = sqrt( (xvalue-c4[0]+r*w[0])*(xvalue-c4[0]+r*w[0]) + (yvalue-c4[1]+r*w[1])*(yvalue-c4[1]+r*w[1]) + (zvalue - c4[2]+r*w[2])*(zvalue-c4[2]+r*w[2]));

	double exponent = -term1 - term2 - term3 -term4;
	res[idx] = exp(exponent);	
	}
}

#define THREADS_PER_BLOCK 128
__global__
void evaluateReduceInnerIntegrand(double* d_c,	double* d_x_grid_points,
                              double* d_y_grid_points,
                              double* d_z_grid_points,
                              double* d_x_weights,
                              double* d_y_weights,
                              double* d_z_weights,
                              int x_dim,
                              double r,double *w, double *res)
{
	 int b_idx = blockIdx.x;
   	 int t_idx = threadIdx.x;
   	 int grid_t_idx = b_idx * blockDim.x + t_idx;
   	 int z_size = x_dim * x_dim;
   	 int z_idx = grid_t_idx/z_size;
   	 int y_size = x_dim;
   	 int y_idx = (grid_t_idx - z_idx*z_size)/y_size;
   	 int x_idx = grid_t_idx - z_idx*z_size - y_idx*y_size;
   	 __shared__ volatile double local_sum[THREADS_PER_BLOCK];
   	 if ( grid_t_idx < x_dim*x_dim*x_dim ) {
                double xvalue = d_x_grid_points[x_idx];
                double yvalue = d_y_grid_points[y_idx];
                double zvalue = d_z_grid_points[z_idx];

                double dx = d_x_weights[x_idx];
                double dy = d_y_weights[y_idx];
                double dz = d_z_weights[z_idx];
                // compute function value
                // exp(-|x1-c1| - |x1-c2| -|x1+r*w_hat - c3| - |x1 + rw_hat -c4|
                // constants needed: r, c1,c2,c3,c4
                //TODO : First six can be precomputed
                double xdiffc_1 = xvalue-d_c[0];
                double ydiffc_1 = yvalue-d_c[1];
                double zdiffc_1 = zvalue-d_c[2];

                double xdiffc_2 = xvalue-d_c[3];
                double ydiffc_2 = yvalue-d_c[4];
                double zdiffc_2 = zvalue-d_c[5];

                double xdiffc_3 = xvalue-d_c[6]+    r*w[0];
                double ydiffc_3 = yvalue-d_c[7]+    r*w[1];
                double zdiffc_3 = zvalue-d_c[8]+    r*w[2];

                double xdiffc_4 = xvalue-d_c[9]+    r*w[0];
                double ydiffc_4 = yvalue-d_c[10]+   r*w[1];
                double zdiffc_4 = zvalue-d_c[11]+   r*w[2];

                double term1 = sqrt( xdiffc_1*xdiffc_1 + ydiffc_1*ydiffc_1 + zdiffc_1*zdiffc_1);
                double term2 = sqrt( xdiffc_2*xdiffc_2 + ydiffc_2*ydiffc_2 + zdiffc_2*zdiffc_2 );
                double term3 = sqrt(xdiffc_3*xdiffc_3 + ydiffc_3*ydiffc_3 + zdiffc_3*zdiffc_3 );
                double term4 = sqrt( xdiffc_4*xdiffc_4 + ydiffc_4*ydiffc_4 + zdiffc_4*zdiffc_4);

                double exponent = -term1 - term2 - term3 -term4 + r ;
                local_sum[t_idx] = exp(exponent)*dx*dy*dz;
   	 } else {
                local_sum[t_idx] = 0.0f;
   	 }
   	 __syncthreads();
   	//Parallel Reduction over thread block 
	for (int stride = blockDim.x / 2; stride >0 ; stride >>=1) {
   	 	if(t_idx < stride) {
   	 		local_sum[t_idx] += local_sum[t_idx + stride];
   	 	} 	
   	 	__syncthreads();
   	 }
   	 if (t_idx == 0) {
   	     res[blockIdx.x] = local_sum[0];
//   		res[blockIdx.x] = w[0]; 
	}
} //evaluateReduceInnerIntegrand
__global__
void evaluateConstantTerm(double* d_c,	double* d_x_grid_points,
                                       double* d_y_grid_points,
                                       double* d_z_grid_points,
                                       int x_dim,
                                       double r, double *res) {
        int b_idx = blockIdx.x;
        int t_idx = threadIdx.x;
        int grid_t_idx = b_idx * blockDim.x + t_idx;
        int z_size = x_dim * x_dim;
        int z_idx = grid_t_idx / z_size;
        int y_size = x_dim;
        int y_idx = (grid_t_idx - z_idx * z_size) / y_size;
        int x_idx = grid_t_idx - z_idx * z_size - y_idx * y_size;
        if (grid_t_idx < x_dim * x_dim * x_dim) {
                double xvalue = d_x_grid_points[x_idx];
                double yvalue = d_y_grid_points[y_idx];
                double zvalue = d_z_grid_points[z_idx];

                // precompute function constant values
                // |x1-c1| - |x1-c2|
                // constants needed: c1,c2,c3,c4
                double xdiffc_1 = xvalue - d_c[0];
                double ydiffc_1 = yvalue - d_c[1];
                double zdiffc_1 = zvalue - d_c[2];

<<<<<<< HEAD
                double xdiffc_2 = xvalue - d_c[3];
                double ydiffc_2 = yvalue - d_c[4];
                double zdiffc_2 = zvalue - d_c[5];
                double term1 = sqrt(xdiffc_1 * xdiffc_1 + ydiffc_1 * ydiffc_1 + zdiffc_1 * zdiffc_1);
                double term2 = sqrt(xdiffc_2 * xdiffc_2 + ydiffc_2 * ydiffc_2 + zdiffc_2 * zdiffc_2);
                res[grid_t_idx] = -term1 - term2 + r;
=======
double make_1d_grid(double start, double stop, unsigned int N, std::vector<double>* grid){
	auto val = start;
	auto step = (stop - start) / N; 	
   	for (unsigned int i=0;i < N; ++i){
   	    grid->push_back(val);
//	    std::cout << "pushed value to grid: " << val << std::endl;
	    val += step;	
   	}
	return step;
}



    void launch_reduceSum(double *input, double *output, int size, int block_size, int num_blocks)
    { 
    	double *d_input = nullptr;
    	double *d_output = nullptr;
    	cudaMalloc(&d_input, size*sizeof(double));
	cudaMalloc(&d_output, num_blocks*sizeof(double));
	
    	cudaMemcpy(d_input, input, size*sizeof(double), cudaMemcpyHostToDevice);
	reduceSum<<<num_blocks, block_size, block_size*sizeof(double)>>>(d_input,d_output,size);   
	cudaError_t error = cudaGetLastError();
    	cudaMemcpy(output,d_output, num_blocks*sizeof(double), cudaMemcpyDeviceToHost);
	std::cout << "cuda error after reduceSum function:" << error << std::endl;
	std::cout << "output vector after calling reduceSum" << std::endl;
    	for (int i=0; i < num_blocks; i++){
		std::cout << output[i] << std::endl;	
>>>>>>> f54066d (fix and test file)
        }
}
__global__
void evaluateReduceInnerIntegrand2(double* d_c,	double* d_x_grid_points,
                                      double* d_y_grid_points,
                                      double* d_z_grid_points,
                                      double* d_x_weights,
                                      double* d_y_weights,
                                      double* d_z_weights,
                                      int x_dim,
                                      double r,double l_x, double l_y, double l_z,
                                      double *term12r_arr,
                                      double *res)
    {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int t_idx = threadIdx.x;
            __shared__ volatile double local_sum[THREADS_PER_BLOCK];
            if ( idx < x_dim*x_dim*x_dim ) {
                    int z_idx = idx / (x_dim * x_dim);
		    int y_idx = (idx / x_dim) % x_dim;
		    int x_idx = idx % x_dim;
		    double xvalue = d_x_grid_points[x_idx];
                    double yvalue = d_y_grid_points[y_idx];
                    double zvalue = d_z_grid_points[z_idx];

                    double dx = d_x_weights[x_idx];
                    double dy = d_y_weights[y_idx];
                    double dz = d_z_weights[z_idx];
                    // compute function value
                    // exp(-|x1-c1| - |x1-c2| -|x1+r*w_hat - c3| - |x1 + rw_hat -c4|
                    // constants needed: r, c1,c2,c3,c4
                    // First six are precomputed
                    double xdiffc_3 = xvalue-d_c[6]+    r*l_x;
                    double ydiffc_3 = yvalue-d_c[7]+    r*l_y;
                    double zdiffc_3 = zvalue-d_c[8]+    r*l_z;

                    double xdiffc_4 = xvalue-d_c[9]+    r*l_x;
                    double ydiffc_4 = yvalue-d_c[10]+   r*l_y;
                    double zdiffc_4 = zvalue-d_c[11]+   r*l_z;
                    double term12r = term12r_arr[idx];
                    double term3 = sqrt(xdiffc_3*xdiffc_3 + ydiffc_3*ydiffc_3 + zdiffc_3*zdiffc_3 );
                    double term4 = sqrt( xdiffc_4*xdiffc_4 + ydiffc_4*ydiffc_4 + zdiffc_4*zdiffc_4);
                    double exponent =  term12r - term3 -term4 ;
                    local_sum[t_idx] = exp(exponent)*dx*dy*dz;
            } 
	    else {
                    local_sum[t_idx] = 0.0f;
            }
            __syncthreads();
            //Parallel Reduction over thread block
            for (int stride = blockDim.x / 2; stride >0 ; stride >>=1) {
                    if(t_idx < stride) {
                            local_sum[t_idx] += local_sum[t_idx + stride];
                    }
                    __syncthreads();
            }
            if (t_idx == 0) {
                    res[blockIdx.x] = local_sum[0];
//   	 	res[blockIdx.x] = d_x_grid_points[x_idx];
            }
    } //evaluateReduceInnerIntegrand2
__global__
void IntegrateReduce(double* d_c,	double* d_x_grid_points,
                                      double* d_y_grid_points,
                                      double* d_z_grid_points,
                                      double* d_x_weights,
                                      double* d_y_weights,
                                      double* d_z_weights,
                                      int x_dim,
                                      double r,double l_x, double l_y, double l_z,
                                      double *res){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<32>(block);
	
	double v = 0.0;
	auto N = x_dim*x_dim*x_dim;
	if (threadIdx.x == 0) res[blockIdx.x] = 0;
	for (int tid = grid.thread_rank(); tid< N; tid+=grid.size())
	{
		auto z_idx = tid / (x_dim*x_dim);
		auto y_idx = (tid / x_dim) % x_dim;
		auto x_idx = tid % x_dim;
		
                double xvalue = d_x_grid_points[x_idx];
                double yvalue = d_y_grid_points[y_idx];
                double zvalue = d_z_grid_points[z_idx];
		
		
                double dx = d_x_weights[x_idx];
                double dy = d_y_weights[y_idx];
                double dz = d_z_weights[z_idx];

                double xdiffc_1 = xvalue-d_c[0];
                double ydiffc_1 = yvalue-d_c[1];
                double zdiffc_1 = zvalue-d_c[2];

                double xdiffc_2 = xvalue-d_c[3];
                double ydiffc_2 = yvalue-d_c[4];
                double zdiffc_2 = zvalue-d_c[5];

                double xdiffc_3 = xvalue-d_c[6]+    r*l_x;
                double ydiffc_3 = yvalue-d_c[7]+    r*l_y;
                double zdiffc_3 = zvalue-d_c[8]+    r*l_z;

                double xdiffc_4 = xvalue-d_c[9]+    r*l_x;
                double ydiffc_4 = yvalue-d_c[10]+   r*l_y;
                double zdiffc_4 = zvalue-d_c[11]+   r*l_z;

                double term1 = sqrt( xdiffc_1*xdiffc_1 + ydiffc_1*ydiffc_1 + zdiffc_1*zdiffc_1 );
                double term2 = sqrt( xdiffc_2*xdiffc_2 + ydiffc_2*ydiffc_2 + zdiffc_2*zdiffc_2 );
                double term3 = sqrt( xdiffc_3*xdiffc_3 + ydiffc_3*ydiffc_3 + zdiffc_3*zdiffc_3 );
                double term4 = sqrt( xdiffc_4*xdiffc_4 + ydiffc_4*ydiffc_4 + zdiffc_4*zdiffc_4 );

                double exponent = -term1 - term2 - term3 -term4 + r ;
		v += exp(exponent)*dx*dy*dz;
		//v +=1 ;

	}
	
	warp.sync();
	
	v += warp.shfl_down( v , 16); 
	v += warp.shfl_down( v , 8); 
	v += warp.shfl_down( v , 4); 
	v += warp.shfl_down( v , 2); 
	v += warp.shfl_down( v , 1);

	if (warp.thread_rank() == 0 )
		atomicAddDouble(&res[block.group_index().x], v); 
}

__global__
void evaluateIntegrandReduceZ(double* d_c,	double* d_x_grid_points,
                                      double* d_y_grid_points,
                                      double* d_z_grid_points,
                                      double* d_x_weights,
                                      double* d_y_weights,
                                      double* d_z_weights,
                                      int x_dim,
                                      double r,double l_x, double l_y, double l_z,
                                      double *res)
    {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if ( idx < x_dim*x_dim ) {
		    int y_idx = idx / x_dim;
		    int x_idx = idx % x_dim;
		    double xvalue = d_x_grid_points[x_idx];
                    double yvalue = d_y_grid_points[y_idx];

                    double dx = d_x_weights[x_idx];
                    double dy = d_y_weights[y_idx];
                    double xdiffc_1 = xvalue-d_c[0];
                    double ydiffc_1 = yvalue-d_c[1];

                    double xdiffc_2 = xvalue-d_c[3];
                    double ydiffc_2 = yvalue-d_c[4];

                    double xdiffc_3 = xvalue-d_c[6]+    r*l_x;
              	    double ydiffc_3 = yvalue-d_c[7]+    r*l_y;
                    
                    double xdiffc_4 = xvalue-d_c[9]+    r*l_x;
                    double ydiffc_4 = yvalue-d_c[10]+   r*l_y;
		    res[idx] = 0.0;
		    for (int z_idx =0; z_idx <x_dim; ++z_idx){
                         double zvalue = d_z_grid_points[z_idx];
                   	 double dz = d_z_weights[z_idx];
                   	 // compute function value
                   	 // exp(-|x1-c1| - |x1-c2| -|x1+r*w_hat - c3| - |x1 + rw_hat -c4|
                   	 // constants needed: r, c1,c2,c3,c4
                   	 // First six are precomputed
                         double zdiffc_1 = zvalue-d_c[2];
                         double zdiffc_2 = zvalue-d_c[5];
                   	 double zdiffc_3 = zvalue-d_c[8]+    r*l_z;
                   	 double zdiffc_4 = zvalue-d_c[11]+   r*l_z;
               	 	 double term1 = sqrt(xdiffc_1 * xdiffc_1 + ydiffc_1 * ydiffc_1 + zdiffc_1 * zdiffc_1);
                	 double term2 = sqrt(xdiffc_2 * xdiffc_2 + ydiffc_2 * ydiffc_2 + zdiffc_2 * zdiffc_2);
                   	 double term3 = sqrt(xdiffc_3*xdiffc_3 + ydiffc_3*ydiffc_3 + zdiffc_3*zdiffc_3 );
                   	 double term4 = sqrt( xdiffc_4*xdiffc_4 + ydiffc_4*ydiffc_4 + zdiffc_4*zdiffc_4);
                   	 double exponent =  -term1 - term2- term3 -term4 + r ;
                   	 res[idx] += exp(exponent)*dx*dy*dz;
            	    }
	}

 } //evaluateReduceInnerIntegrandz

__global__
void evaluateInnerIntegrand(double* d_c,	double* d_x_grid_points,
                              double* d_y_grid_points,
                              double* d_z_grid_points,
                              double* d_x_weights,
                              double* d_y_weights,
                              double* d_z_weights,
                              int x_dim,
                              double r,double *w, double *res
)
{
	 int b_idx = blockIdx.x;
   	 int t_idx = threadIdx.x;
   	 int grid_t_idx = b_idx * blockDim.x + t_idx;
   	 int z_size = x_dim * x_dim;
   	 int z_idx = grid_t_idx/z_size;
   	 int y_size = x_dim;
   	 int y_idx = (grid_t_idx - z_idx*z_size)/y_size;
   	 int x_idx = grid_t_idx - z_idx*z_size - y_idx*y_size;
   	 if ( grid_t_idx < x_dim*x_dim*x_dim ) {
                double xvalue = d_x_grid_points[x_idx];
                double yvalue = d_y_grid_points[y_idx];
                double zvalue = d_z_grid_points[z_idx];

                double dx = d_x_weights[x_idx];
                double dy = d_y_weights[y_idx];
                double dz = d_z_weights[z_idx];
                // compute function value
                // exp(-|x1-c1| - |x1-c2| -|x1+r*w_hat - c3| - |x1 + rw_hat -c4|
                // constants needed: r, c1,c2,c3,c4
                //TODO : First six can be precomputed
                double xdiffc_1 = xvalue-d_c[0];
                double ydiffc_1 = yvalue-d_c[1];
                double zdiffc_1 = zvalue-d_c[2];

                double xdiffc_2 = xvalue-d_c[3];
                double ydiffc_2 = yvalue-d_c[4];
                double zdiffc_2 = zvalue-d_c[5];

                double xdiffc_3 = xvalue-d_c[6]+    r*w[0];
                double ydiffc_3 = yvalue-d_c[7]+    r*w[1];
                double zdiffc_3 = zvalue-d_c[8]+    r*w[2];

                double xdiffc_4 = xvalue-d_c[9]+    r*w[0];
                double ydiffc_4 = yvalue-d_c[10]+   r*w[1];
                double zdiffc_4 = zvalue-d_c[11]+   r*w[2];

           double term1 = sqrt( xdiffc_1*xdiffc_1 + ydiffc_1*ydiffc_1 + zdiffc_1*zdiffc_1);
                double term2 = sqrt( xdiffc_2*xdiffc_2 + ydiffc_2*ydiffc_2 + zdiffc_2*zdiffc_2 );
                double term3 = sqrt(xdiffc_3*xdiffc_3 + ydiffc_3*ydiffc_3 + zdiffc_3*zdiffc_3 );
                double term4 = sqrt( xdiffc_4*xdiffc_4 + ydiffc_4*ydiffc_4 + zdiffc_4*zdiffc_4);

                double exponent = -term1 - term2 - term3 -term4 + r ;
                res[grid_t_idx] = exp(exponent)*dx*dy*dz;
	}
}//evalInnerIntegrand

double** preProcessIntegral(int total_grid_points, int &num_grids, int &max_grids)
    {
            // Copy Constants to GPU  memory
//        double* d_c1234;
//        cudaMalloc(&d_c1234, 12*sizeof(double));
//        cudaMemcpy(d_c1234,c1234_input,12*sizeof(double),cudaMemcpyHostToDevice);
            size_t availableMemory;
            getAvailableMemory(availableMemory);

            // Decide how many grid_files should be solved concurrently in this gpu
            // Maximum vector size of one_grid_memory is total number of blocks in first pass
            //auto one_grid_points = (total_grid_points + THREADS_PER_BLOCK -1 ) / THREADS_PER_BLOCK;
            auto one_grid_points = total_grid_points ;
            auto total_possible_grids = availableMemory/(one_grid_points*8);
            auto memory_factor = 0.75;
            std::cout << "Points for One Grid  = " << one_grid_points << std::endl;
            std::cout << "Memory for One Grid  = " << one_grid_points*8 << std::endl;
            std::cout << "Available Memory     = " << availableMemory <<std::endl;
            std::cout << "Total Possible Grids = " << total_possible_grids << std::endl;

            num_grids = std::floor(memory_factor*total_possible_grids);
            if (max_grids < num_grids){
                    num_grids = max_grids;
            }
            //       std::cout << "Number of grid_files on this device: " <<num_grids << std::endl;
            double** d_results = new double*[num_grids];

            for (int i=0; i < num_grids; i++) {
                    cudaMalloc(&d_results[i], one_grid_points * sizeof(double));
            }
//        std::cout << "Allocated Memory on Device" << std::endl;
            getAvailableMemory(availableMemory);
            std::cout << "Available Memory after Allocation     = " << availableMemory <<std::endl;
            return d_results;
    }//preProcessIntegral

double evaluateInner(double* c1234_input,
                         double r,
                         double* w_input,
                         double* xrange, double* yrange, double* zrange,
                         unsigned int x_axis_points, unsigned int y_axis_points, unsigned int z_axis_points,
                         double** d_results_ptr, int gpu_num)
{
        HANDLE_CUDA_ERROR(cudaSetDevice(gpu_num));
        assert(x_axis_points==y_axis_points);
        assert(x_axis_points==z_axis_points);

        int threads = THREADS_PER_BLOCK; // Max threads per block
        int blocks = (x_axis_points*y_axis_points*z_axis_points + threads -1)/threads; // Max blocks, better if multiple of SM = 80
//        double* d_result = *d_results_ptr;
//
//   	if (d_result == nullptr){
////		std::cout<< "null ptr received, initializing d_results"	<< std::endl;
//                d_result = nullptr;
//                checkCudaError(cudaMalloc(d_results_ptr, blocks*sizeof(double)));
//                d_result = *d_results_ptr;
//        }

        double* d_result = nullptr;
        HANDLE_CUDA_ERROR(cudaMalloc(&d_result, blocks*sizeof(double)));
//        double* d_term12r_arr  = nullptr;
//        checkCudaError(cudaMalloc(&d_term12r_arr, x_axis_points*y_axis_points*z_axis_points));

        double *d_x_grid = nullptr;
    double evaluateInner(double* c1_input, double* c2_input, double* c3_input, double* c4_input, double r, double* w_input, double* xrange, double* yrange, double* zrange, unsigned int x_axis_points, unsigned int y_axis_points, unsigned int z_axis_points, double *result_array)
    { 
   	if (result_array == nullptr){
		std::cout<< "null ptr received"	<< std::endl;
	}
	double *d_result = nullptr;
	double *d_x_grid = nullptr;
	double *d_y_grid = nullptr;
	double *d_z_grid = nullptr;
<<<<<<< HEAD
	double *d_x_weights = nullptr;
	double *d_y_weights = nullptr;
	double *d_z_weights = nullptr;
	
	double *d_w = nullptr;
        double *d_c1234 = nullptr;
=======
	double *result = new double[x_axis_points*y_axis_points*z_axis_points](); 

    	//Initialize Timer Variables for GPU computations
   	cudaEvent_t startGPU,stopGPU, startTransfer,stopTransfer, startReduce, stopReduce;
   	cudaEventCreate(&startGPU);
   	cudaEventCreate(&stopGPU);
   	cudaEventCreate(&startTransfer);
   	cudaEventCreate(&stopTransfer);
   	cudaEventCreate(&startReduce);
   	cudaEventCreate(&stopReduce);

   	// dummy uniform values for x, y, and z grid, keeping them between 0 and 1
   	// not strictly within sphere
>>>>>>> f54066d (fix and test file)

   	std::vector<double> x_grid;
   	std::vector<double> y_grid;
   	std::vector<double> z_grid;
   	
	std::vector<double> x_weights;
   	std::vector<double> y_weights;
   	std::vector<double> z_weights;

	make_1d_grid_simpson(xrange[0],xrange[1],x_axis_points, &x_grid, &x_weights);
	make_1d_grid_simpson(yrange[0],yrange[1],y_axis_points, &y_grid, &y_weights);
	make_1d_grid_simpson(zrange[0],zrange[1],z_axis_points, &z_grid, &z_weights);
       // make_1d_grid_legendre(xrange[0],xrange[1],x_axis_points, &x_grid, &x_weights);
       // make_1d_grid_legendre(yrange[0],yrange[1],y_axis_points, &y_grid, &y_weights);
       // make_1d_grid_legendre(zrange[0],zrange[1],z_axis_points, &z_grid, &z_weights);

   	unsigned int PX = x_grid.size();
   	unsigned int PY = y_grid.size();
   	unsigned int PZ = z_grid.size();

    // Allocate memory on GPU
    // TODO: write a preprocess function that generates and saves grid_files and weights on GPU
        HANDLE_CUDA_ERROR(cudaMalloc(&d_x_grid, PX*sizeof(double)));
        HANDLE_CUDA_ERROR(cudaMalloc(&d_x_weights, PX*sizeof(double)));
        HANDLE_CUDA_ERROR(cudaMalloc(&d_w, 3*sizeof(double)));
        HANDLE_CUDA_ERROR(cudaMalloc(&d_c1234, 12*sizeof(double)));

   	// Evaluate Funciton on GPU
       // assert(PX== (x_axis_points));
       // assert(PY== (y_axis_points));
       // assert(PZ== (z_axis_points));
 	assert(PX== (x_axis_points+1));
 	assert(PY== (y_axis_points+1));
 	assert(PZ== (z_axis_points+1));
        HANDLE_CUDA_ERROR(cudaMemcpy(d_x_grid, x_grid.data(), PX*sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_CUDA_ERROR(cudaMemcpy(d_x_weights, x_weights.data(), PX*sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_CUDA_ERROR(cudaMemcpy(d_w, w_input, 3*sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_CUDA_ERROR(cudaMemcpy(d_c1234, c1234_input, 12*sizeof(double), cudaMemcpyHostToDevice));

        evaluateReduceInnerIntegrand<<<blocks,threads>>>(d_c1234,
                                                      d_x_grid, d_x_grid, d_x_grid,
                                                      d_x_weights, d_x_weights, d_x_weights,
                                                      x_grid.size(),r, d_w, d_result);

//        evaluateReduceInnerIntegrand2<<<blocks,threads>>>(d_c1234,
//                                                         d_x_grid, d_x_grid, d_x_grid,
//                                                         d_x_weights,d_x_weights, d_x_weights,
//                                                         x_grid.size(),r, d_w, d_term12r_arr, d_result);
	// Copy Constants to GPU const memory
   	cudaMemcpyToSymbol(c1, c1_input, sizeof(double) * 3);
   	cudaMemcpyToSymbol(c2, c2_input, sizeof(double) * 3);
   	cudaMemcpyToSymbol(c3, c3_input, sizeof(double) * 3);
   	cudaMemcpyToSymbol(c4, c4_input, sizeof(double) * 3);
   	cudaMemcpyToSymbol(w, w_input, sizeof(double) * 3);
//	std::cout << "Allocating Memory on GPU" << std::endl;
   	// Evaluate Funciton on GPU and Time it
   	cudaMalloc(&d_result, PX*PY*PZ*sizeof(double));
   	cudaMalloc(&d_x_grid, PX*sizeof(double));
   	cudaMalloc(&d_y_grid, PY*sizeof(double));
   	cudaMalloc(&d_z_grid, PZ*sizeof(double));

//	std::cout << "Copying data on GPU" << std::endl;
   	cudaMemcpy(d_x_grid, x_grid.data(), PX*sizeof(double), cudaMemcpyHostToDevice);
   	cudaMemcpy(d_y_grid, y_grid.data(), PY*sizeof(double), cudaMemcpyHostToDevice);
   	cudaMemcpy(d_z_grid, z_grid.data(), PZ*sizeof(double), cudaMemcpyHostToDevice);
   	
	dim3 block3d((PX+255)/256, PY, PZ);
   	dim3 threads3d(256, 1, 1);
   	cudaEventRecord(startGPU);
   	evalInnerIntegral<<<block3d,threads3d>>>( d_x_grid, d_y_grid, d_z_grid, x_grid.size(),r, d_result);
   	cudaDeviceSynchronize();
   	cudaEventRecord(stopGPU);
//	cudaError_t err = cudaGetLastError();
//	std::cout << "Error: "<< err << std::endl;
   	//Transfer Vector back to CPU and time transfer
   	cudaEventRecord(startTransfer);
   	cudaMemcpy(result, d_result, PX*PY*PZ*sizeof(double), cudaMemcpyDeviceToHost);
   	cudaEventRecord(stopTransfer);
   	cudaDeviceSynchronize();


   	//Reduce vector on GPU within each block
   	int numBlocksReduced = (blocks+threads-1)/threads;
        while (blocks > threads)
   	    {
   		reduceSum<<<numBlocksReduced, threads, threads* sizeof(double)>>>(d_result, d_result, blocks);
            	blocks = numBlocksReduced;
   	        numBlocksReduced = (blocks+threads-1)/threads;
   	    }
    	reduceSum<<<1, blocks, blocks* sizeof(double)>>>(d_result, d_result, blocks);

   	double sumGPU=0.0;
        HANDLE_CUDA_ERROR(cudaMemcpy(&sumGPU, d_result, sizeof(double), cudaMemcpyDeviceToHost));

   	//std::cout << "Sum on GPU: " << sumGPU << std::endl;

        HANDLE_CUDA_ERROR(cudaFree(d_result));
        HANDLE_CUDA_ERROR(cudaFree(d_x_grid));
        HANDLE_CUDA_ERROR(cudaFree(d_y_grid));
        HANDLE_CUDA_ERROR(cudaFree(d_z_grid));
        HANDLE_CUDA_ERROR(cudaFree(d_x_weights));
        HANDLE_CUDA_ERROR(cudaFree(d_y_weights));
        HANDLE_CUDA_ERROR(cudaFree(d_z_weights));
        HANDLE_CUDA_ERROR(cudaFree(d_c1234));

	return sumGPU;
}//evaluateInner

__global__ void accumulateSum(double result, 
					double* d_r_weights,int r_i, 
					double* d_l_weights,int l_i, 
					double* d_sum)
{
    double sum = *d_sum;
    sum+= result*d_r_weights[r_i]*d_l_weights[l_i];
    *d_sum = sum;
}
double evaluateInnerPreProcessed(thrust::device_vector<double>& d_c1234, 
                                 double r,
                                 double l_x, double l_y, double l_z,
                                 thrust::device_vector<double>& d_x_grid, 
                                 thrust::device_vector<double>& d_x_weights, 
                                 unsigned int x_axis_points,
                                 thrust::device_vector<double>& d_r_weights, 
                                 int r_i,  
                                 thrust::device_vector<double>& d_l_weights, 
                                 int l_i,
                                 thrust::device_vector<double>& d_term12r, 
                                 thrust::device_vector<double>& d_result, 
                                 double* d_sum, 
                                 int blocks, 
                                 int threads, 
                                 int gpu_num) 
    {
            cuProfilerStart();
	    HANDLE_CUDA_ERROR(cudaSetDevice(gpu_num));

<<<<<<< HEAD
	     IntegrateReduce<<<blocks,threads>>>(thrust::raw_pointer_cast(d_c1234.data()),  
                                   thrust::raw_pointer_cast(d_x_grid.data()),     
                                   thrust::raw_pointer_cast(d_x_grid.data()),       
                                   thrust::raw_pointer_cast(d_x_grid.data()),       
                                   thrust::raw_pointer_cast(d_x_weights.data()),  
                                   thrust::raw_pointer_cast(d_x_weights.data()),    
                                   thrust::raw_pointer_cast(d_x_weights.data()),    
                                   x_axis_points,r, l_x, l_y, l_z,                
	                           thrust::raw_pointer_cast(d_result.data()));
	        for(long unsigned int i = 0; i < d_result.size(); i++)
        std::cout << "d_result[" << i << "] = " << d_result[i] << std::endl;
            //Reduce vector on GPU within each block
	    double sum = thrust::reduce(d_result.begin(),d_result.end(),(double) 0.0, thrust::plus<double>());
	    // Accumulate result on device
    	        accumulateSum<<<1, 1>>>(sum,
	        			thrust::raw_pointer_cast(d_r_weights.data()),
	        			r_i, 
	        			thrust::raw_pointer_cast(d_l_weights.data()),l_i, 
	        			d_sum);
            return 0.0;
	    cuProfilerStop();
    }//evaluateInner

void postProcessIntegral(double** d_results, int nl)
    {
            for (int i =0; i< nl; ++i){
                    cudaFree(d_results[i]);
            }
    }//postProcessIntegral


}
=======
}

>>>>>>> f54066d (fix and test file)
