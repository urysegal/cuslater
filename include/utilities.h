//
// Created by gkluhana on 04/03/24.
//
#include "cuslater.cuh"


#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <assert.h>
#include <array>
#include <stdio.h>
#include <stdlib.h>
#include <exception>
#include <chrono>
#include <unordered_map>

#include "cooperative_groups.h"
#include "cooperative_groups/reduce.h"
#include <cuda_runtime_api.h>


namespace cg = cooperative_groups;
namespace cuslater{

    void getAvailableMemory(size_t& availableMemory);

    __global__
    void reduceSum(double *output,double *input,   int size);

    __global__
    void reduceSumWrapper(double* d_results_w_i, int blocks, int threads);

    __global__
    void reduceSumWithWeights(double *input, double *output, double *weights,  int size);


    __device__
    unsigned long upper_power_of_two(unsigned long v);

    __global__
    void reduceSumFast(const float* __restrict data, float* __restrict sums,int n);

    __global__
    void multiplyVolumeElement(int x_dim, double dxdydz,
                               double *res);

}
