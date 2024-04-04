//
// Created by gkluhana on 26/03/24.
//
#include <vector>
#include "cuslater.cuh"
#include "utilities.h"
#include "grids.h"
#include "evalInnerIntegral.h"
namespace cuslater{
	__global__ 
	void accumulateSum(double* result,				 
				double* d_r_weights,int r_i, 
				double* d_l_weights,int l_i, 
				 double* d_sum);
	__global__
	void evaluateIntegrandReduceZ(double* d_c,	double* d_x_grid_points,
	                                      double* d_y_grid_points,
	                                      double* d_z_grid_points,
	                                      double* d_x_weights,
	                                      double* d_y_weights,
	                                      double* d_z_weights,
	                                      int x_dim,
	                                      double r,double l_x, double l_y, double l_z,
	                                      double *res);

	double evaluateInnerSumX1_rl_preAllocated(
                                 unsigned int x_axis_points,
                                 int r_i,  
                                 int l_i,
                                 thrust::device_vector<double>& d_term12r, 
                                 thrust::device_vector<double>& d_result, 
                                 double* d_sum, 
                                 int blocks, 
                                 int threads, 
                                 int gpu_num); 

	double evaluateFourCenterIntegral( double* c,
	                            int nr,  int nl,  int nx,
	                            const std::string x1_type);
	double evaluateFourCenterIntegral( double* c,
                                  int nr,  int nl,  int nx,
                                  const std::string x1_type,
                                  int num_gpus);

}
