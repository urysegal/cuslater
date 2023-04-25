#pragma once
#include <vector>
#include <unordered_map>

#include "tensors.cuh"

namespace cuslater {
int hadamar(std::vector<int> &modes, std::unordered_map<int, int64_t> &extent, const real_t *A, const real_t *C,
            real_t *D);
void gpu_calculate_s_values( const std::vector<real_t> &points, real_t *result);
void
gpu_calculate_sto_function_values(
    sto_exponent_t exponent,
    double x, double y, double z,
    principal_quantum_number_t n ,angular_quantum_number_t l, magnetic_quantum_number_t m,
    const std::vector<double> &x_grid, const std::vector<double> &y_grid, const std::vector<double> &z_grid,
    real_t *result_data
) ;

}