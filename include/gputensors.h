#pragma once
#include <vector>
#include <unordered_map>


namespace cuslater {
int hadamar(std::vector<int> &modes, std::unordered_map<int, int64_t> &extent, const double *A, const double *C,
            double *D);
void gpu_calculate_s_values( const std::vector<double> &points, double *result);
void
gpu_calculate_sto_function_values(
    double exponent,
    double x, double y, double z,
    int n ,int l, int m,
    const std::vector<double> &x_grid, const std::vector<double> &y_grid, const std::vector<double> &z_grid,
    double *result_data
) ;

}