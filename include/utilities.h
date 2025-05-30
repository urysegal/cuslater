//
// Created by gkluhana on 04/03/24.
//
#ifndef UTILITIES_H
#define UTILITIES_H
#include <assert.h>
#include <cassert>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef PRECISION_DOUBLE
typedef double  real_t;
typedef double3 real3_t;
    #pragma message("real_t is set to double.")
#else
typedef float  real_t;
typedef float3 real3_t;
    #pragma message("real_t is set to float.")
#endif

// helper to construct real3_t from three real_t’s
static inline __host__ __device__ real3_t make_real3(real_t x, real_t y, real_t z) {
#ifdef PRECISION_DOUBLE
    return make_double3(x, y, z);
#else
    return make_float3(x, y, z);
#endif
}

static inline __device__ real_t norm3(real3_t x) {
#ifdef PRECISION_DOUBLE
    return norm3d(x.x, x.y, x.z);
#else
    return norm3df(x.x, x.y, x.z);
#endif
}

namespace cuslater {

    struct ProgramParameters {
        int    nr       = 97;
        int    nl       = 590;
        int    nx       = 200;
        int    ny       = 200;
        int    nz       = 200;
        double tol      = 1e-10;
        float  alpha[4] = {1, 1, 1, 1};

        float c[12]           = {0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0};
        bool  check_zero_cond = false;
    };
    void handleArguments(int argc, const char* argv[], ProgramParameters& params);

    void getAvailableMemory(size_t& availableMemory);

    __global__ void reduceSum(double* output, double* input, int size);

    __global__ void reduceSumWrapper(double* d_results_w_i, int blocks, int threads);

    __global__ void reduceSumWithWeights(double* input, double* output, double* weights, int size);

    __device__ unsigned long upper_power_of_two(unsigned long v);

    __global__ void reduceSumFast(const float* __restrict data, float* __restrict sums, int n);

    __global__ void multiplyVolumeElement(int x_dim, double dxdydz, double* res);

} // namespace cuslater
#endif // UTILITIES_H
