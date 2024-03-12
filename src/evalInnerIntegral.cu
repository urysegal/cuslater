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
//__constant__ double c1[3];
//__constant__ double c2[3];
//__constant__ double c3[3];
//__constant__ double c4[3];


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

        double dx = d_x_weights[xpos];
        double dy = d_y_weights[blockIdx.y];
        double dz = d_z_weights[blockIdx.z];

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
        res[idx] = exp(exponent)*dx*dy*dz;
        //	res[idx] = dx;
    }
}
    __global__
    void evalInnerIntegralSimpson2(double* d_c,	double* d_x_grid_points,
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
//            	res[grid_t_idx] =x_dim*x_dim*x_dim;
        }
    }
	double* preProcessIntegral(double *c1234_input)
	{
    // Copy Constants to GPU  memory
	double* d_c1234;
    cudaMalloc(&d_c1234, 12*sizeof(double));
   	cudaMemcpy(d_c1234, c1234_input, 12*sizeof(double), cudaMemcpyHostToDevice);
	return d_c1234;
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
    	//Initialize Timer Variables for GPU computations
//   	cudaEvent_t startGPU,stopGPU, startTransfer,stopTransfer, startReduce, stopReduce;
//   	cudaEventCreate(&startGPU);
//   	cudaEventCreate(&stopGPU);
//   	cudaEventCreate(&startTransfer);
//   	cudaEventCreate(&stopTransfer);
//   	cudaEventCreate(&startReduce);
//   	cudaEventCreate(&stopReduce);

   	// dummy uniform values for x, y, and z grid, keeping them between 0 and 1
   	// not strictly within sphere

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
    cudaMalloc(&d_y_grid, PY*sizeof(double));
    cudaMalloc(&d_z_grid, PZ*sizeof(double));
    cudaMalloc(&d_x_weights, PX*sizeof(double));
    cudaMalloc(&d_y_weights, PY*sizeof(double));
    cudaMalloc(&d_z_weights, PZ*sizeof(double));
    cudaMalloc(&d_w, 3*sizeof(double));
    cudaMalloc(&d_c1234, 12*sizeof(double));

   	// Evaluate Funciton on GPU
    assert(PX== (x_axis_points+1));
    assert(PY== (y_axis_points+1));
    assert(PZ== (z_axis_points+1));
    //TODO: this array is not needed if not saving data, remove allocation


   	cudaMemcpy(d_x_grid, x_grid.data(), PX*sizeof(double), cudaMemcpyHostToDevice);
   	cudaMemcpy(d_y_grid, y_grid.data(), PY*sizeof(double), cudaMemcpyHostToDevice);
   	cudaMemcpy(d_z_grid, z_grid.data(), PZ*sizeof(double), cudaMemcpyHostToDevice);
   	cudaMemcpy(d_x_weights, x_weights.data(), PX*sizeof(double), cudaMemcpyHostToDevice);
   	cudaMemcpy(d_y_weights, y_weights.data(), PY*sizeof(double), cudaMemcpyHostToDevice);
   	cudaMemcpy(d_z_weights, z_weights.data(), PZ*sizeof(double), cudaMemcpyHostToDevice);
   	cudaMemcpy(d_w, w_input, 3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c1234, c1234_input, 12*sizeof(double), cudaMemcpyHostToDevice);

//    dim3 block3d((PX+255)/256, PY, PZ);
//    dim3 threads3d(256, 1, 1);//   	cudaEventRecord(startGPU);

    cudaMalloc(&d_result, PX *PY*PZ*sizeof(double));
//    double *result = new double[(PX)*(PY)*(PZ)]();
        int threads = 256; // Max threads per block
        int blocks = (PX*PY*PZ + threads -1)/threads; // Max blocks, multiple of SM = 80
//        evalInnerIntegralSimpson<<<block3d,threads3d>>>(d_c1234, d_x_grid, d_y_grid, d_z_grid,d_x_weights, d_y_weights, d_z_weights, x_grid.size(),r, d_w, d_result);
        evalInnerIntegralSimpson2<<<blocks,threads>>>(d_c1234,
                                                      d_x_grid, d_y_grid, d_z_grid,
                                                      d_x_weights, d_y_weights, d_z_weights,
                                                      x_grid.size(),r, d_w, d_result);

   	//cudaDeviceSynchronize();
//   	cudaEventRecord(stopGPU);
//	cudaError_t err = cudaGetLastError();

   	//Transfer Vector back to CPU and time transfer
//   	cudaEventRecord(startTransfer);
//   	cudaMemcpy(result, d_result, PX*PY*PZ*sizeof(double), cudaMemcpyDeviceToHost);
//   	cudaDeviceSynchronize();
//   	cudaEventRecord(stopTransfer);

// //	Debug values
// 	     std::cout << "Result vector values:" << std::endl;
// 	   for (unsigned int i = 0; i < PX*PY*PZ; ++i) {
// 	       std::cout << result[i] << " ";
// 	   }
   	

   	//Reduce vector on GPU within each block
//   	cudaEventRecord(startReduce);
threads = 1024; // Max threads per block
blocks = 2048; // Max blocks, multiple of SM = 80

   	reduceSum<<<blocks, threads, threads* sizeof(double)>>>(d_result, d_result, PX* PY* PZ);
        int    numBlocksReduced = blocks;
       while (blocks > threads)
   	    {
   	        numBlocksReduced = (blocks+threads-1)/threads;
   		    reduceSum<<<numBlocksReduced, threads, threads* sizeof(double)>>>(d_result, d_result, blocks);
            blocks = numBlocksReduced;
   	    }
  // 	cudaEventRecord(stopReduce);
//   	cudaDeviceSynchronize();
    reduceSum<<<1, blocks, blocks* sizeof(double)>>>(d_result, d_result, blocks);

        // copy reduced vector of size < 256 to cpu and sum remaining vector
       // TODO fix summation for array of length <256 in reduceSum funciton
   	double sumGPU=0.0;
   	cudaMemcpy(&sumGPU, d_result, sizeof(double), cudaMemcpyDeviceToHost);
//  	auto start_time_cpu_remaining = std::chrono::high_resolution_clock::now();

//   	auto end_time_cpu_remaining = std::chrono::high_resolution_clock::now();
//   	auto elapsed_time_cpu_remaining = std::chrono::duration_cast<std::chrono::microseconds>(end_time_cpu_remaining - start_time_cpu_remaining).count();


   	// Calculate elapsed time
//   	float millisecondsGPU = 0.0f;
//   	cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);
//   	float millisecondsReduce = 0.0f;
//   	cudaEventElapsedTime(&millisecondsReduce, startReduce, stopReduce);
//   	float millisecondsTransfer = 0.0f;
//   	cudaEventElapsedTime(&millisecondsTransfer, startTransfer, stopTransfer);

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

}

