#pragma once
#include "../include/utilities.h"
#include <cutensor.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include "../include/grids.h"


namespace cuslater{

        __global__
        void evaluateReduceInnerIntegrand(double* d_c,	double* d_x_grid_points,
                                      double* d_y_grid_points,
                                      double* d_z_grid_points,
                                      double* d_x_weights,
                                      double* d_y_weights,
                                      double* d_z_weights,
                                      int x_dim,
                                      double r,double *w, double *res
        );
	__global__
	void evaluateConstantTerm(double* d_c,	double* d_x_grid_points,
                                       double* d_y_grid_points,
                                       double* d_z_grid_points,
                                       int x_dim,
                                       double r, double *res);

        __global__
        void evaluateReduceInnerIntegrand2(double* d_c,	double* d_x_grid_points,
                               double* d_y_grid_points,
                               double* d_z_grid_points,
                               double* d_x_weights,
                               double* d_y_weights,
                               double* d_z_weights,
                               int x_dim,
                               double r,double l_x, double l_y, double l_z,
                               double *term12r_arr,
                               double *res);

        __global__
        void evaluateInnerIntegrand(double* d_c,	double* d_x_grid_points,
                        double* d_y_grid_points,
                        double* d_z_grid_points,
                        double* d_x_weights,
                        double* d_y_weights,
                        double* d_z_weights,
                        int x_dim,
                        double r,double *w, double *res
        );
	__global__ 
	void accumulateSum(double* result,				 
				double* d_r_weights,int r_i, 
				double* d_l_weights,int l_i, 
				 double* d_sum);


                extern "C" {
                double** preProcessIntegral(int total_grid_points, int &num_grids, int &max_grids );

                double evaluateInner(double* d_c1234, double r,
                                     double* w_input,
                                     double* xrange, double* yrange, double* zrange,
                                     unsigned int x_axis_points, unsigned int y_axis_points, unsigned int z_axis_points,
                                     double **d_results_ptr, int gpu_num);

		double evaluateInnerPreProcessed(thrust::device_vector<double>& d_c1234, 
                                 double r,
                                 double l_x, double l_y, double l_z,
                                 thrust::device_vector<double>& d_x_grid, 
                                 thrust::device_vector<double>& d_x_weights, 
                                 unsigned int x_axis_points,
                                 thrust::device_vector<double>& d_r_weights, 
                                 int r_i,  
                                 thrust::device_vector<double>& d_l_weights, 
                                 int l_i,
                                 thrust::device_vector<double>& d_term12r, 
                                 thrust::device_vector<double>& d_result, 
                                 double* d_sum, 
                                 int blocks, 
                                 int threads, 
                                 int gpu_num); 

                void postProcessIntegral(double** d_results, int nl);

    }
}
