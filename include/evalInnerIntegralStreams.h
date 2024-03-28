//
// Created by gkluhana on 26/03/24.
//
#pragma once
#include "../include/utilities.h"
#include <cutensor.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include "../include/grids.h"


namespace cuslater{

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


    extern "C" {
    double** preProcessIntegral(int total_grid_points, int &num_grids, int &max_grids );

    double evaluateInnerWithStreams(double* c1234_input,
                                    double r,
                                    double* w_input, double* w_weights, int Nl,
                                    double* xrange, double* yrange, double* zrange,
                                    unsigned int x_axis_points, unsigned int y_axis_points, unsigned int z_axis_points,
                                    int gpu_num);

    void postProcessIntegral(double** d_results, int nl);


    }
}
