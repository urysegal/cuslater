//
// Created by gkluhana on 26/03/24.
//
// editted by MarkEwert03 on 13/05/24

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <vector>

#include "cuslater.cuh"
#include "grids.h"
#include "utilities.h"
// #include "evalInnerIntegral.h"
namespace cuslater {
double evaluateFourCenterIntegral(float* c, float* alpha, int nr, int nl,
                                  int nx, int ny, int nz,
                                  const std::string x1_type, double tol);

double evaluateInnerSum(
    unsigned int nx, unsigned int ny, unsigned int nz, float r, float l_x,
    float l_y, float l_z, float r_weight, float l_weight,
    thrust::device_vector<double>& __restrict__ d_result,
    double* __restrict__ d_sum, int blocks, int threads, int gpu_num);

__global__ void evaluateIntegrandReduceZ(int nx, int ny, int nz, float r,
                                         float l_x, float l_y, float l_z,
                                         double* __restrict__ res);

__global__ void accumulateSum(double result, float r_weight, float l_weight,
                              double* __restrict__ d_sum);

}  // namespace cuslater
