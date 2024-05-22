//
// Created by gkluhana on 26/03/24.
//
// editted by MarkEwert03 on 13/05/24

#include <thrust/device_vector.h>

#include "../include/evalIntegral.h"

const double pi = 3.14159265358979323846;
#include <thread>
#define THREADS_PER_BLOCK 128
__constant__ float d_c[12];
__constant__ float d_alphas[4];
__constant__ float d_x_grid[600];
__constant__ float d_x_weights[600];

namespace cuslater {
__global__ void accumulateSum(double result, float r_weight, float l_weight,
                              double* __restrict__ d_sum) {
    double sum = *d_sum;
    sum += result * r_weight * l_weight;
    *d_sum = sum;
}

__global__ void evaluateIntegrandReduceZ(int x_dim, int y_dim, int z_dim,
                                         float r, float l_x, float l_y,
                                         float l_z, double* __restrict__ res) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < x_dim * x_dim) {
        int y_idx = idx / x_dim;
        int x_idx = idx % x_dim;
        float xvalue = d_x_grid[x_idx];
        float yvalue = d_x_grid[y_idx];

        float dx = d_x_weights[x_idx];
        float dy = d_x_weights[y_idx];
        float xdiffc_1 = xvalue - d_c[0];
        float ydiffc_1 = yvalue - d_c[1];
        float xdiffc_2 = xvalue - d_c[3];
        float ydiffc_2 = yvalue - d_c[4];
        float xdiffc_3 = xvalue - d_c[6] + r * l_x;
        float ydiffc_3 = yvalue - d_c[7] + r * l_y;
        float xdiffc_4 = xvalue - d_c[9] + r * l_x;
        float ydiffc_4 = yvalue - d_c[10] + r * l_y;

        float xysq1 = xdiffc_1 * xdiffc_1 + ydiffc_1 * ydiffc_1;
        float xysq2 = xdiffc_2 * xdiffc_2 + ydiffc_2 * ydiffc_2;
        float xysq3 = xdiffc_3 * xdiffc_3 + ydiffc_3 * ydiffc_3;
        float xysq4 = xdiffc_4 * xdiffc_4 + ydiffc_4 * ydiffc_4;
        double dxy = dx * dy;
        double v = 0.0;
        for (int z_idx = 0; z_idx < x_dim; ++z_idx) {
            float zvalue = d_x_grid[z_idx];
            float dz = d_x_weights[z_idx];
            // compute function value
            // exp(-|x1-c1| - |x1-c2| -|x1+r*w_hat - c3| - |x1 + rw_hat -c4|
            // constants needed: r, c1,c2,c3,c4
            // First six are precomputed
            float zdiffc_1 = zvalue - d_c[2];
            float zdiffc_2 = zvalue - d_c[5];
            float zdiffc_3 = zvalue - d_c[8] + r * l_z;
            float zdiffc_4 = zvalue - d_c[11] + r * l_z;
            float term1 = d_alphas[0] * sqrt(xysq1 + zdiffc_1 * zdiffc_1);
            float term2 = d_alphas[1] * sqrt(xysq2 + zdiffc_2 * zdiffc_2);
            float term3 = d_alphas[2] * sqrt(xysq3 + zdiffc_3 * zdiffc_3);
            float term4 = d_alphas[3] * sqrt(xysq4 + zdiffc_4 * zdiffc_4);
            float exponent = -term1 - term2 - term3 - term4 + r;
            v += exp(exponent) * dxy * dz;
        }
        res[idx] = v;
    }
}  // evaluateReduceInnerIntegrandz

double evaluateInnerSumX1_rl_preAllocated(
    unsigned int x_axis_points, unsigned int y_axis_points,
    unsigned int z_axis_points, float r, float l_x, float l_y, float l_z,
    float r_weight, float l_weight,
    thrust::device_vector<double>& __restrict__ d_result,
    double* __restrict__ d_sum, int blocks, int threads, int gpu_num) {
    HANDLE_CUDA_ERROR(cudaSetDevice(gpu_num));

    evaluateIntegrandReduceZ<<<blocks, threads>>>(
        x_axis_points, y_axis_points, z_axis_points, r, l_x, l_y, l_z,
        raw_pointer_cast(d_result.data()));
    // for(long unsigned int i = 0; i < d_result.size(); i++)
    // std::cout << "d_result[" << i << "] = " << d_result[i] << std::endl;
    // Reduce vector on GPU within each block
    double delta_sum = thrust::reduce(d_result.begin(), d_result.end(),
                                      (double)0.0, thrust::plus<double>());
    // Accumulate result on device
    accumulateSum<<<1, 1>>>(delta_sum, r_weight, l_weight, d_sum);
    return delta_sum;
}  // evaluateInner

double evaluateFourCenterIntegral(float* c, float* alphas, int nr, int nl,
                                  int nx, int ny, int nz,
                                  const std::string x1_type, double tol) {
    // read r grid
    std::cout << "Reading r Grid Files" << std::endl;
    const std::string r_filepath =
        "grid_files/r_" + std::to_string(nr) + ".grid";
    std::vector<float> r_nodes;
    std::vector<float> r_weights;
    read_r_grid_from_file(r_filepath, r_nodes, r_weights);

    // read l grid
    std::cout << "Reading l Grid Files" << std::endl;
    const std::string l_filepath =
        "grid_files/l_" + std::to_string(nl) + ".grid";
    std::vector<float> l_nodes_x;
    std::vector<float> l_nodes_y;
    std::vector<float> l_nodes_z;
    std::vector<float> l_weights;
    read_l_grid_from_file(l_filepath, l_nodes_x, l_nodes_y, l_nodes_z,
                          l_weights);
    // Read x1 grid
    std::cout << "Reading x1 Grid Files" << std::endl;
    const std::string x1_filepath =
        "grid_files/x1_" + x1_type + "_1d_" + std::to_string(nx) + ".grid";
    std::vector<float> x1_nodes;
    std::vector<float> x1_weights;
    float a;
    float b;
    read_x1_1d_grid_from_file(x1_filepath, a, b, x1_nodes, x1_weights);

    std::cout << "Initializing Device Variables" << std::endl;
    double* w = new double[3];
    unsigned int PX = x1_nodes.size();
    int threads = THREADS_PER_BLOCK;  // Max threads per block
    int blocks = (PX * PX + threads - 1) /
                 threads;  // Max blocks, better if multiple of SM = 80
    std::cout << "Total Threads: " << blocks * threads << std::endl;
    std::cout << "Total Grid Points: " << nx * ny * nz << std::endl;

    cudaMemcpyToSymbol(d_c, c, 12 * sizeof(float));
    cudaMemcpyToSymbol(d_alphas, alphas, 4 * sizeof(float));
    cudaMemcpyToSymbol(d_x_grid, x1_nodes.data(), PX * sizeof(float));
    cudaMemcpyToSymbol(d_x_weights, x1_weights.data(), PX * sizeof(float));
    double* d_sum;
    thrust::device_vector<float> d_r_weights(nr);
    thrust::device_vector<float> d_l_weights(nl);
    thrust::device_vector<double> d_result(PX * PX);

    HANDLE_CUDA_ERROR(cudaMalloc(&d_sum, sizeof(double)));
    HANDLE_CUDA_ERROR(cudaMemset(d_sum, 0, sizeof(double)));
    double sum = 0.0;
    double delta_sum = 0.0;
    int r_skipped = 0;
    std::cout << "Evaluating Integral for all values of r and l" << std::endl;
    for (int j = 0; j < nl; ++j) {
        for (int i = 0; i < nr; ++i) {
            delta_sum = evaluateInnerSumX1_rl_preAllocated(
                nx, ny, nz, r_nodes[i], l_nodes_x[j], l_nodes_y[j],
                l_nodes_z[j], r_weights[i], l_weights[j], d_result, d_sum,
                blocks, threads, 0);
            if (delta_sum < tol) {
                r_skipped += nr - i;
                break;
            }
        }
        if (j % 100 == 0) {
            std::cout << "computed for l_j:" << j << "/" << nl << std::endl;
        }
    }
    HANDLE_CUDA_ERROR(
        cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    sum = sum * (4.0 / pi) *
          std::pow(alphas[0] * alphas[1] * alphas[2] * alphas[3], 1.5);

    // sum up result, multiply with constant and return
    std::cout << "Tolerance: " << tol << std::endl;
    std::cout << "Total values of r skipped for different l's: " << r_skipped
              << "/" << nr * nl << std::endl;
    return sum;
}

}  // namespace cuslater
