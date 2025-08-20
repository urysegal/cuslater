//
// Created by gkluhana on 04/03/24.
//
#pragma once
#include <cuda_runtime_api.h>

namespace cuslater {

    struct ProgramParameters {
        int    nr       = 97;
        int    nl       = 590;
        int    nx       = 200;
        int    ny       = 200;
        int    nz       = 200;
        double tol      = 1e-8;
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
