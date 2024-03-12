#pragma once
#include <vector>
#include <unordered_map>
#include "../include/utilities.h"

namespace cuslater{
__global__
void evalInnerIntegral(	double* d_x_grid_points,
			double* d_y_grid_points,
			double* d_z_grid_points,  
			int x_dim,
                          double r, double *res);
__global__
void multiplyVolumeElement(int x_dim,
			double dxdydz,
			double *res);

__global__
void evalInnerIntegralSimpson(double* d_c,	double* d_x_grid_points,
                              double* d_y_grid_points,
                              double* d_z_grid_points,
                              double* d_x_weights,
                              double* d_y_weights,
                              double* d_z_weights,
                              int x_dim,
                              double r,double *w, double *res
);


double make_1d_grid(double start, double stop, unsigned int N, std::vector<double>* grid);

extern "C" {
void launch_reduceSum(double *input, double *output, int size, int block_size, int num_blocks);
	double* preProcessIntegral(double *c1234_input);
    double evaluateInner(double* d_c1234, double r, double* w_input, double* xrange, double* yrange, double* zrange, unsigned int x_axis_points, unsigned int y_axis_points, unsigned int z_axis_points, double *result_array, int gpu_num);
}
}
