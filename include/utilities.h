//
// Created by gkluhana on 04/03/24.
//
#include <vector>
#include "cooperative_groups.h"
#include "cooperative_groups/reduce.h"
namespace cg = cooperative_groups;
namespace cuslater{
    void make_1d_grid_simpson(double start, double stop, unsigned int N,
                              std::vector<double>* grid, std::vector<double>* weights);

    double make_1d_grid(double start, double stop, unsigned int N,
                        std::vector<double>* grid);

    __global__
    void reduceSum(double *output,double *input,   int size);
    __device__
    unsigned long upper_power_of_two(unsigned long v);

    __global__
    void reduceSumFast(const float* __restrict data, float* __restrict sums,int n);

    __global__
    void multiplyVolumeElement(int x_dim, double dxdydz,
                               double *res);

}