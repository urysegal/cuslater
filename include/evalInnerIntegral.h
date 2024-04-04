#pragma once
#include "../include/utilities.h"
#include <cutensor.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include "../include/grids.h"

#include <vector>
#include <unordered_map>

namespace cuslater{

        __global__
        void evaluateIntegrandX1ReduceBlocks(double* d_c,	double* d_x_grid_points,
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
        void evaluateIntegrandX1ReduceBlocks(double* d_c,	double* d_x_grid_points,
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
        void evaluateIntegrandX1(double* d_c,	double* d_x_grid_points,
                        double* d_y_grid_points,
                        double* d_z_grid_points,
                        double* d_x_weights,
                        double* d_y_weights,
                        double* d_z_weights,
                        int x_dim,
                        double r,double *w, double *res
        );
	
	
	
	
	extern "C" {

                double** allocateGridMemory(int total_grid_points, int &num_grids, int &max_grids );

                double evaluateInnerSumX1_rl(double* c1234, double r,
                                     double* w_input,
                                     double* xrange, double* yrange, double* zrange,
                                     unsigned int x_axis_points, unsigned int y_axis_points, unsigned int z_axis_points,
                                     double **d_results_ptr, int gpu_num);


                void deallocateGridMemory(double** d_results, int nl);

    }

}
