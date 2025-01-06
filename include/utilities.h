//
// Created by gkluhana on 04/03/24.
//
#ifndef UTILITIES_H
#define UTILITIES_H
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

    struct ProgramParameters {
        int nr = 97;
        int nl = 590;
        int nx = 200;
        int ny = 200;
        int nz = 200;
        double tol = 1e-10;	
	float alpha[4] = {1,1,1,1};

	float c[12] = {0,0,0,
                      1,0,0,
                      2,0,0,
                      3,0,0};
	bool check_zero_cond  = false;
    };
    void handleArguments(int argc,const char* argv[], ProgramParameters& params);

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
#endif // UTILITIES_H
