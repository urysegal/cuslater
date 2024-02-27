#include <iostream>
#include <assert.h>
#include <array>
#include "../include/cuslater.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
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

 	// compute function value
	// exp(-|x1-c1| - |x1-c2| -|x1+r*w_hat - c3| - |x1 + rw_hat -c4|
	// constants needed: r, c1,c2,c3,c4
	double term1 = sqrt( (xvalue-c1[0])*(xvalue-c1[0]) + (yvalue-c1[1])*(yvalue-c1[1]) + (zvalue - c1[2])*(zvalue-c1[2]));
	double term2 = sqrt( (xvalue-c2[0])*(xvalue-c2[0]) + (yvalue-c2[1])*(yvalue-c2[1]) + (zvalue - c2[2])*(zvalue-c2[2]));
	double term3 = sqrt( (xvalue-c3[0]+r*w[0])*(xvalue-c3[0]+r*w[0]) + (yvalue-c3[1]+r*w[1])*(yvalue-c3[1]+r*w[1]) + (zvalue - c3[2]+r*w[2])*(zvalue-c3[2]+r*w[2]));
	double term4 = sqrt( (xvalue-c4[0]+r*w[0])*(xvalue-c4[0]+r*w[0]) + (yvalue-c4[1]+r*w[1])*(yvalue-c4[1]+r*w[1]) + (zvalue - c4[2]+r*w[2])*(zvalue-c4[2]+r*w[2]));
	double exponent = -term1 - term2 - term3 -term4 + r;
	res[idx] = exp(exponent);	
	}
}

__global__
void multiplyVolumeElement(int x_dim,
			double dxdydz,
			double *res)
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
		res[idx] = res[idx]*dxdydz;	
    	}
}

__global__ 
void reduceSum(double *input, double *output,  int size)
{
	extern __shared__ double tsum[];
	int id = threadIdx.x;
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = gridDim.x*blockDim.x;
	tsum[id] = 0.0;
	for(int k = tid; k<size; k+=stride) tsum[id] += input[k];
	__syncthreads();
	
	for(int k = blockDim.x /2; k>0; k/=2) {
		if(id<k) tsum[id] += tsum[id+k];
	__syncthreads();
	}
	if (id==0) output[blockIdx.x] = tsum[0];

}

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
        }
    }

    double evaluateInner(double* c1_input, double* c2_input, double* c3_input, double* c4_input, double r, double* w_input, double* xrange, double* yrange, double* zrange, unsigned int x_axis_points, unsigned int y_axis_points, unsigned int z_axis_points, double *result_array)
    { 
//   	if (result_array == nullptr){
//		std::cout<< "null ptr received"	<< std::endl;
//	}
	double *d_result = nullptr;
	double *d_x_grid = nullptr;
	double *d_y_grid = nullptr;
	double *d_z_grid = nullptr;
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

   	std::vector<double> x_grid;
   	std::vector<double> y_grid;
   	std::vector<double> z_grid;

	double dx= make_1d_grid(xrange[0],xrange[1],x_axis_points, &x_grid);
	double dy= make_1d_grid(yrange[0],yrange[1],y_axis_points, &y_grid);
	double dz= make_1d_grid(zrange[0],zrange[1],z_axis_points, &z_grid);
	
	double dxdydz = dx*dy*dz;

   	unsigned int PX = x_grid.size();
   	unsigned int PY = y_grid.size();
   	unsigned int PZ = z_grid.size();

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

 //	Debug values
// 	     std::cout << "Result vector values:" << std::endl;
// 	   for (unsigned int i = 0; i < PX*PY*PZ; ++i) {
// 	       std::cout << result[i] << " ";
// 	   }
   	
   	multiplyVolumeElement<<<block3d,threads3d>>>( x_grid.size(), dxdydz, d_result);
   	 
   	//Reduce vector on GPU within each block and time it
   	cudaEventRecord(startReduce);
   	int blockSize = 256;
   	int numBlocks = (PX*PY*PZ+255)/256;
   	reduceSum<<<numBlocks, blockSize, blockSize* sizeof(double)>>>(d_result, d_result, PX* PY* PZ);
   	int numBlocksReduced=numBlocks;
   	
   	while (numBlocks > blockSize)
   	    {
   	        numBlocksReduced = (numBlocks+255)/256;
   	        numBlocksReduced = (numBlocks+255)/256;
   		    reduceSum<<<numBlocksReduced, blockSize, blockSize* sizeof(double)>>>(d_result, d_result, numBlocks);
   	        numBlocks = numBlocksReduced;
   	    }
   	cudaEventRecord(stopReduce);
   	cudaDeviceSynchronize();
   	
   	// copy reduced vector of size < 256 to cpu and sum remaining vector
   	double sumGPU=0.0;
   	double *cpu_sum_array =(double*)malloc(numBlocksReduced*sizeof(double));    
   	cudaMemcpy(cpu_sum_array, d_result, numBlocksReduced*sizeof(double), cudaMemcpyDeviceToHost);
  	auto start_time_cpu_remaining = std::chrono::high_resolution_clock::now();
   	for (int i = 0; i < numBlocksReduced; ++i) {
   		sumGPU += cpu_sum_array[i];
   	}
   	auto end_time_cpu_remaining = std::chrono::high_resolution_clock::now();
   	auto elapsed_time_cpu_remaining = std::chrono::duration_cast<std::chrono::microseconds>(end_time_cpu_remaining - start_time_cpu_remaining).count();
 
   	
   	// Calculate elapsed time
   	float millisecondsGPU = 0.0f;
   	cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);
   	float millisecondsReduce = 0.0f;
   	cudaEventElapsedTime(&millisecondsReduce, startReduce, stopReduce);
   	float millisecondsTransfer = 0.0f;
   	cudaEventElapsedTime(&millisecondsTransfer, startTransfer, stopTransfer);

   	// Report Values and Times
//   	std::cout << "Sum on GPU: " << sumGPU << std::endl;
//   	std::cout << "GPU Calculation time: " << millisecondsGPU << " ms" << std::endl;
//   	std::cout << "Calculations performed: "<< PX*PY*PZ << std::endl;
//   	std::cout << "Time for Reduction on GPU: " << millisecondsReduce << " ms" << std::endl; 	   
//   	std::cout << "Time taken for remaining summation: " << elapsed_time_cpu_remaining << " microseconds" << std::endl;
//   	std::cout << "Time for data transfer: " << millisecondsTransfer << " ms" << std::endl; 	   
//   	std::cout << "Bytes Transferred: " << PX*PY*PZ*sizeof(double) <<std::endl;
//   	std::cout << "Sum on CPU: " << sumCPU << std::endl;
//   	std::cout << "Time taken for sequential summation: " << elapsed_time_cpu << " microseconds" << std::endl;
   	
   	cudaFree(d_result);
	return sumGPU; 
}

}

