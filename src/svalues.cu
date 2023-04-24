#include <assert.h>
#include <array>
#include "../include/grids.h"
#include "../include/sto.h"
#include "../include/tensors.cuh"
#include "../stocalculator.h"
#include <stdio.h>
#include <stdlib.h>

#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include <cutensor.h>
#define MAX_S_POINTS 512 // You can go to 4K, but cache size is just 8K.

__constant__ double inv_sqrt_pi;
__constant__ double s_points[MAX_S_POINTS];


__global__
void calculate_s_value( int N, double *res)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if ( i < N ) {
        double s_value = s_points[i];
        res[i] = inv_sqrt_pi * (1.0 / sqrt(s_value));
    }
}

namespace cuslater {

void gpu_calculate_s_values(const std::vector<real_t> &points, real_t *result)
{
    const double pi = 3.14159265358979311599796346854;
    double _inv_sqrt_pi = 1.0 / (sqrt(pi)) ;
    double *d_result = nullptr;
    int N = points.size();

    if ( N > MAX_S_POINTS ) {
        throw std::exception();
    }

   cudaMemcpyToSymbol(inv_sqrt_pi, &_inv_sqrt_pi,  sizeof(double));
   cudaMemcpy(s_points, points.data(), sizeof(double) *N, cudaMemcpyHostToDevice);
   cudaMalloc(&d_result, N*sizeof(double));
   calculate_s_value<<<(N+255)/256, 256>>>(N, d_result);
   cudaMemcpy(result, d_result, N*sizeof(double ), cudaMemcpyDeviceToHost);
   cudaFree(d_result);
}

}
