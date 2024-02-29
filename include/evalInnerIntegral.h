#pragma once
#include <vector>
#include <unordered_map>

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
void reduceSum(double *input, double *output,  int size);


double make_1d_grid(double start, double stop, unsigned int N, std::vector<double>* grid);

extern "C" {
void launch_reduceSum(double *input, double *output, int size, int block_size, int num_blocks);
    double evaluateInner(double* c1_input, double* c2_input, double* c3_input, double* c4_input, double r, double* w_input, double* xrange, double* yrange, double* zrange, unsigned int x_axis_points, unsigned int y_axis_points, unsigned int z_axis_points, double *result_array);
}
}
