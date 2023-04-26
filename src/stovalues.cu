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

#define MAX_CONST_MEMORY (64*1024)
#define MAX_CONST_DOUBLES (MAX_CONST_MEMORY/sizeof(double))
#define MAX_AXIS_POINTS (MAX_CONST_DOUBLES/4)

__constant__ double x_grid_points[MAX_AXIS_POINTS];
__constant__ double y_grid_points[MAX_AXIS_POINTS];
__constant__ double z_grid_points[MAX_AXIS_POINTS];


__global__
void calculate_sto_value_S_orbital( cuslater::sto_exponent_t exponent,
                          double x, double y, double z, int x_dim,
                          double *res)
{
    const double coeff = .28209479177387814897 ; // 0.5*(1/sqrt(pi))

    int i = (blockIdx.z* (blockDim.x *blockDim.y)) ;
    i+= blockIdx.y * blockDim.x ;

    int xpos = 256*blockIdx.x;
    xpos += threadIdx.x;
    i+= xpos;

    if ( xpos < x_dim ) {
        double x_value = x_grid_points[xpos];
        double y_value = x_grid_points[blockIdx.y];
        double z_value = x_grid_points[blockIdx.z];

        x_value -= x;
        y_value -= y;
        z_value -= z;
        double r = sqrt(x_value*x_value + y_value * y_value + z_value *z_value);
        res[i] = coeff * exp(-exponent*r);
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
    unsigned int PX = x_grid.size();
    unsigned int PY = y_grid.size();
    unsigned int PZ = z_grid.size();

    if ( PX > MAX_AXIS_POINTS or PY > MAX_AXIS_POINTS or PZ > MAX_AXIS_POINTS) {
        throw std::exception();
    }

    cudaMemcpyToSymbol(x_grid_points, x_grid.data(), sizeof(double) * PX);
    cudaMemcpyToSymbol(y_grid_points, y_grid.data(), sizeof(double) * PY);
    cudaMemcpyToSymbol(z_grid_points, z_grid.data(), sizeof(double) * PZ);

    cudaMalloc(&d_result, PX*PY*PZ*sizeof(double));
    dim3 block3d((PX+255)/256, PY, PZ);

    calculate_sto_value_S_orbital<<<block3d, 256>>>(exponent, x, y, z, x_grid.size(), d_result);

    cudaMemcpy(result, d_result, PX*PY*PZ*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

}

