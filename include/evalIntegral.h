//
// Created by gkluhana on 26/03/24.
//
#include <vector>
#include "cuslater.cuh"
#include "utilities.h"
#include "grids.h"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
//#include "evalInnerIntegral.h"
namespace cuslater{
	__global__ void accumulateSum(double result, 
						float r_weight, 
						float l_weight, 
						double* __restrict__ d_sum);
	__global__
	void evaluateIntegrandReduceZ(int x_dim,
	                                      float r, float l_x, float l_y, float l_z,
	                                      double * __restrict__ res);

	double evaluateInnerSumX1_rl_preAllocated(unsigned int x_axis_points,
	                                 float r,float l_x,float l_y,float l_z,
					 float r_weight,float  l_weight,
	                                 thrust::device_vector<double>& __restrict__ d_result, 
	                                 double* __restrict__ d_sum, 
	                                 int blocks, 
	                                 int threads, 
	                                 int gpu_num); 
	    

	double evaluateFourCenterIntegral( float* c,
	                            int nr,  int nl,  int nx,
	                            const std::string x1_type, double tol);
	double evaluateFourCenterIntegral( float* c,
                                  int nr,  int nl,  int nx,
                                  const std::string x1_type,
                                  int num_gpus);

}
