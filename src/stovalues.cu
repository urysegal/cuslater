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
void calculate_sto_value( int N, double *res)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if ( i < N ) {
        double s_value = s_points[i];
        res[i] = inv_sqrt_pi * (1.0 / sqrt(s_value));
    }
}

namespace cuslater {

void
gpu_calculate_sto_function_values(
    sto_exponent_t exponent,
    double x, double y, double z,
    principal_quantum_number_t n ,angular_quantum_number_t l, magnetic_quantum_number_t m,
    const std::vector<double> &x_grid, const std::vector<double> &y_grid, const std::vector<double> &z_grid,
    real_t *result
)
{
    double *d_result = nullptr;
    int N = points.size();

    if ( N > MAX_S_POINTS ) {
        throw std::exception();
    }

    cudaMemcpyToSymbol(inv_sqrt_pi, &_inv_sqrt_pi,  sizeof(double));
    cudaMemcpyToSymbol(s_points, points.data(), sizeof(double) *N);
    cudaMalloc(&d_result, N*sizeof(double));
    calculate_s_value<<<(N+255)/256, 256>>>(N, d_result);
    cudaMemcpy(result, d_result, N*sizeof(double ), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

}

