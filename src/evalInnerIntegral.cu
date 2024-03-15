#include <iostream>
#include <assert.h>
#include <array>
#include "../include/cuslater.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <exception>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cutensor.h>
#include "../include/evalInnerIntegral.h"

namespace cuslater {
//__constant__ double c1[3];
//__constant__ double c2[3];
//__constant__ double c3[3];
//__constant__ double c4[3];
#define THREADS_PER_BLOCK 1024

void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void getAvailableMemory(size_t& availableMemory) {
    size_t free_bytes, total_bytes;
    cudaError_t error = cudaMemGetInfo(&free_bytes, &total_bytes);
    checkCudaError(error);

    availableMemory = free_bytes;
}
__global__
void evalInnerIntegral(	double* d_c, double* d_x_grid_points,
			double* d_y_grid_points,
			double* d_z_grid_points,  
			int x_dim,
                          double r,double* w, double *res
			)
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

        	// compute function value
        	// exp(-|x1-c1| - |x1-c2| -|x1+r*w_hat - c3| - |x1 + rw_hat -c4|
        	// constants needed: r, c1,c2,c3,c4
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
        	double exponent = -term1 - term2 - term3 -term4 ;
        	res[idx] = exp(exponent);
	}
}
__global__
void evalInnerIntegralSimpson(double* d_c,	double* d_x_grid_points,
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
	//Store result
   	 if (t_idx == 0) {
   	     res[blockIdx.x] = local_sum[0];
   	 }
}
	double* preProcessIntegral(double *c1234_input, int total_grid_points)
	{
    		// Copy Constants to GPU  memory
		size_t availableMemory;
		getAvailableMemory(availableMemory);
		// Decide how many grids should be solved concurrently in this gpu
		// Maximum vector size is total number of blocks in first pass
		one_grid_memory = (total_grid_points + THREADS_PER_BLOCK -1 ) / THREADS_PER_BLOCK * 8;
		std::cout << one_grid_memory << std::endl;
		std::cout << availableMemory << std::endl;
	   	cudaMalloc(&d_c1234, 12*sizeof(double));
	}
    double evaluateInner(double* c1234_input,
                         double r,
                         double* w_input,
                         double* xrange, double* yrange, double* zrange,
                         unsigned int x_axis_points, unsigned int y_axis_points, unsigned int z_axis_points,
                         double *result_array, int gpu_num)
    { 
//   	if (result_array == nullptr){
//		std::cout<< "null ptr received"	<< std::endl;
//	}

	cudaSetDevice(gpu_num);

	double *d_result = nullptr;
	double *d_x_grid = nullptr;
	double *d_y_grid = nullptr;
	double *d_z_grid = nullptr;
	double *d_x_weights = nullptr;
	double *d_y_weights = nullptr;
	double *d_z_weights = nullptr;
	
	double *d_w = nullptr;
        double *d_c1234 = nullptr;

   	std::vector<double> x_grid;
   	std::vector<double> y_grid;
   	std::vector<double> z_grid;
   	
	std::vector<double> x_weights;
   	std::vector<double> y_weights;
   	std::vector<double> z_weights;

	make_1d_grid_simpson(xrange[0],xrange[1],x_axis_points, &x_grid, &x_weights);
	make_1d_grid_simpson(yrange[0],yrange[1],y_axis_points, &y_grid, &y_weights);
	make_1d_grid_simpson(zrange[0],zrange[1],z_axis_points, &z_grid, &z_weights);

   	unsigned int PX = x_grid.size();
   	unsigned int PY = y_grid.size();
   	unsigned int PZ = z_grid.size();

    // Allocate memory on GPU
    // TODO: write a preprocess function that generates and saves grids and weights on GPU
    cudaMalloc(&d_x_grid, PX*sizeof(double));
//    cudaMalloc(&d_y_grid, PY*sizeof(double));
//    cudaMalloc(&d_z_grid, PZ*sizeof(double));
    cudaMalloc(&d_x_weights, PX*sizeof(double));
//    cudaMalloc(&d_y_weights, PY*sizeof(double));
//    cudaMalloc(&d_z_weights, PZ*sizeof(double));
    cudaMalloc(&d_w, 3*sizeof(double));
    cudaMalloc(&d_c1234, 12*sizeof(double));

   	// Evaluate Funciton on GPU
    assert(PX== (x_axis_points+1));
    assert(PY== (y_axis_points+1));
    assert(PZ== (z_axis_points+1));
    //TODO: this array is not needed if not saving data, remove allocation


   	cudaMemcpy(d_x_grid, x_grid.data(), PX*sizeof(double), cudaMemcpyHostToDevice);
//   	cudaMemcpy(d_y_grid, y_grid.data(), PY*sizeof(double), cudaMemcpyHostToDevice);
//   	cudaMemcpy(d_z_grid, z_grid.data(), PZ*sizeof(double), cudaMemcpyHostToDevice);
   	cudaMemcpy(d_x_weights, x_weights.data(), PX*sizeof(double), cudaMemcpyHostToDevice);
//   	cudaMemcpy(d_y_weights, y_weights.data(), PY*sizeof(double), cudaMemcpyHostToDevice);
 //  	cudaMemcpy(d_z_weights, z_weights.data(), PZ*sizeof(double), cudaMemcpyHostToDevice);
   	cudaMemcpy(d_w, w_input, 3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c1234, c1234_input, 12*sizeof(double), cudaMemcpyHostToDevice);


//    double *result = new double[(PX)*(PY)*(PZ)]();
        int threads = THREADS_PER_BLOCK; // Max threads per block
        int blocks = (PX*PY*PZ + threads -1)/threads; // Max blocks, better if multiple of SM = 80
    	cudaMalloc(&d_result, blocks*sizeof(double));
        evalInnerIntegralSimpson<<<blocks,threads>>>(d_c1234,
                                                      d_x_grid, d_x_grid, d_x_grid,
                                                      d_x_weights, d_x_weights, d_x_weights,
                                                      x_grid.size(),r, d_w, d_result);


   	//Transfer Vector back to CPU and time transfer
//   	cudaMemcpy(result, d_result, PX*PY*PZ*sizeof(double), cudaMemcpyDeviceToHost);

// //	Debug values
// 	     std::cout << "Result vector values:" << std::endl;
// 	   for (unsigned int i = 0; i < PX*PY*PZ; ++i) {
// 	       std::cout << result[i] << " ";
// 	   }
   	

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
   	cudaMemcpy(&sumGPU, d_result, sizeof(double), cudaMemcpyDeviceToHost);

   	std::cout << "Sum on GPU: " << sumGPU << std::endl;

//	delete[] result;
   	cudaFree(d_result);
   	cudaFree(d_x_grid);
   	cudaFree(d_y_grid);
   	cudaFree(d_z_grid);
   	cudaFree(d_x_weights);
   	cudaFree(d_y_weights);
   	cudaFree(d_z_weights);
        cudaFree(d_c1234);

	return sumGPU;
}
    double evaluateInnerStreams(double* c1234_input,
                         double r,
                         double* w_input, double* w_weights, int Nl,
                         double* xrange, double* yrange, double* zrange,
                         unsigned int x_axis_points, unsigned int y_axis_points, unsigned int z_axis_points,
                         double *result_array, int gpu_num)
    { 
//   	if (result_array == nullptr){
//		std::cout<< "null ptr received"	<< std::endl;
//	}

	cudaSetDevice(gpu_num);
	double *d_result = nullptr;
	double *d_x_grid = nullptr;
	double *d_y_grid = nullptr;
	double *d_z_grid = nullptr;
	double *d_x_weights = nullptr;
	double *d_y_weights = nullptr;
	double *d_z_weights = nullptr;
	
	double *d_w = nullptr;
        double *d_c1234 = nullptr;

   	std::vector<double> x_grid;
   	std::vector<double> y_grid;
   	std::vector<double> z_grid;
   	
	std::vector<double> x_weights;
   	std::vector<double> y_weights;
   	std::vector<double> z_weights;

	make_1d_grid_simpson(xrange[0],xrange[1],x_axis_points, &x_grid, &x_weights);
	make_1d_grid_simpson(yrange[0],yrange[1],y_axis_points, &y_grid, &y_weights);
	make_1d_grid_simpson(zrange[0],zrange[1],z_axis_points, &z_grid, &z_weights);

   	unsigned int PX = x_grid.size();
   	unsigned int PY = y_grid.size();
   	unsigned int PZ = z_grid.size();

    // Allocate memory on GPU
    // TODO: write a preprocess function that generates and saves grids and weights on GPU
    cudaMalloc(&d_x_grid, PX*sizeof(double));
    cudaMalloc(&d_x_weights, PX*sizeof(double));

    cudaMalloc(&d_c1234, 12*sizeof(double));

   	// Evaluate Funciton on GPU
    assert(PX== (x_axis_points+1));
    assert(PY== (y_axis_points+1));
    assert(PZ== (z_axis_points+1));


   	cudaMemcpy(d_x_grid, x_grid.data(), PX*sizeof(double), cudaMemcpyHostToDevice);
   	cudaMemcpy(d_x_weights, x_weights.data(), PX*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c1234, c1234_input, 12*sizeof(double), cudaMemcpyHostToDevice);


//    double *result = new double[(PX)*(PY)*(PZ)]();
        int threads = 1024; // Max threads per block
        int blocks = (PX*PY*PZ + threads -1)/threads; // Max blocks, better if multiple of SM = 80
	const int num_streams = Nl;
	cudaStream_t streams[num_streams];
    	double* d_results[num_streams];
	double* d_ws[num_streams];
	double* w_values[num_streams];
	for (int i =0; i< num_streams; ++i) {
    		cudaMalloc(&d_ws[i], 3*sizeof(double));
		cudaStreamCreate(&streams[i]);
		cudaMalloc(&d_results[i],blocks*sizeof(double));
   		cudaMemcpy(d_ws[i], w_values[i], 3*sizeof(double), cudaMemcpyHostToDevice);
	}	
        for (int i=0; i<num_streams; ++i) {
	evalInnerIntegralSimpson<<<blocks,threads>>>(d_c1234,
                                                      d_x_grid, d_x_grid, d_x_grid,
                                                      d_x_weights, d_x_weights, d_x_weights,
                                                      x_grid.size(),r, d_ws[i], d_results[i]);
	}
	// Synchronize streams
	for (int i = 0; i < num_streams; ++i) {
    		cudaStreamSynchronize(streams[i]);
	}
	
// Destroy CUDA streams
   	//Transfer Vector back to CPU and time transfer
//   	cudaMemcpy(result, d_result, PX*PY*PZ*sizeof(double), cudaMemcpyDeviceToHost);

// //	Debug values
// 	     std::cout << "Result vector values:" << std::endl;
// 	   for (unsigned int i = 0; i < PX*PY*PZ; ++i) {
// 	       std::cout << result[i] << " ";
// 	   }
   	

   	//Reduce vector on GPU within each block
	threads = 1024; // Max threads per block
	int rblocks = 2048; // Max blocks, multiple of SM = 80

   	reduceSum<<<rblocks, threads, threads* sizeof(double)>>>(d_result, d_result, blocks);
        int    numBlocksReduced = rblocks;
       while (rblocks > threads)
   	    {
   	        numBlocksReduced = (rblocks+threads-1)/threads;
   		    reduceSum<<<numBlocksReduced, threads, threads* sizeof(double)>>>(d_result, d_result, rblocks);
            rblocks = numBlocksReduced;
   	    }
    reduceSum<<<1, rblocks, rblocks* sizeof(double)>>>(d_result, d_result, rblocks);

   	double sumGPU=0.0;
   	cudaMemcpy(&sumGPU, d_result, sizeof(double), cudaMemcpyDeviceToHost);

   	std::cout << "Sum on GPU: " << sumGPU << std::endl;
	for (int i = 0; i < num_streams; ++i) {
    		cudaStreamDestroy(streams[i]);
	}

//	delete[] result;
   	for (int i = 0; i < num_streams; ++i) {
	    cudaFree(d_results[i]);
	}
   	
	cudaFree(d_x_grid);
   	cudaFree(d_y_grid);
   	cudaFree(d_z_grid);
   	cudaFree(d_x_weights);
   	cudaFree(d_y_weights);
   	cudaFree(d_z_weights);
        cudaFree(d_c1234);

	return sumGPU;
}

}

