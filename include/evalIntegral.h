//
// Created by gkluhana on 26/03/24.
//
// editted by MarkEwert03 on 13/05/24

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <vector>

#include "cuda_profiler_api.h"
#include "cuslater.cuh"
#include "grids.h"
#include "evalInnerIntegral.h"
namespace cuslater{
	__global__ void accumulateSum(double result, 
						float r_weight, 
						float l_weight, 
						double* __restrict__ d_sum);
	__global__
	void evaluateIntegrandReduceZ(int x_dim, int y_dim, int z_dim,
	                                      float r, float l_x, float l_y, float l_z,
	                                      double * __restrict__ res);

	double evaluateInnerSum(unsigned int x_axis_points, unsigned int y_axis_points, unsigned int z_axis_points,
	                                 float r,float l_x,float l_y,float l_z,
					 float r_weight,float  l_weight,
	                                 thrust::device_vector<double>& __restrict__ d_result, 
	                                 thrust::device_vector<double>& __restrict__ d_sorted, 
					 int i, int j, int nl,
	                                 double* __restrict__ d_sum, 
	                                 int blocks, 
	                                 int threads, 
	                                 int gpu_num); 
	    

	double evaluateFourCenterIntegral( float* c, float* alphas,
                                    int nr,  int nl,  int nx, int ny, int nz,
	                            const std::string x1_type, double tol, bool check_zero_cond);
	double evaluateFourCenterIntegral( float* c, float* alphas,
                                  int nr,  int nl,  int nx, int ny, int nz,
                                  const std::string x1_type,
                                  int num_gpus);


} // namespace cuslater
