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
__constant__ float d_alpha[4];
__constant__ float d_x_grid[600];
__constant__ float d_x_weights[600];

namespace cuslater {
double evaluateFourCenterIntegral(float* c, float* alpha, int nr, int nl,
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
    cudaMemcpyToSymbol(d_alpha, alpha, 4 * sizeof(float));
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
    std::cout << "Evaluating Integral for all values of r and l with a1="
              << alpha[0] << ", a2=" << alpha[1] << ", a3=" << alpha[2]
              << ", a4=" << alpha[3] << std::endl;
    for (int j = 0; j < nl; ++j) {
        for (int i = 0; i < nr; ++i) {
            delta_sum = evaluateInnerSum(nx, ny, nz, r_nodes[i], l_nodes_x[j],
                                         l_nodes_y[j], l_nodes_z[j],
                                         r_weights[i], l_weights[j], d_result,
                                         d_sum, blocks, threads, 0);
            if (delta_sum < tol) {
                r_skipped += nr - i;
                break;
            }
        }
        if (j % 50 == 0) {
            std::cout << "computed for l_j:" << j << "/" << nl << std::endl;
        }
    }
    HANDLE_CUDA_ERROR(
        cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    sum = sum * (4.0 / pi) *
          std::pow(alpha[0] * alpha[1] * alpha[2] * alpha[3], 1.5);

    // sum up result, multiply with constant and return
    std::cout << "Tolerance: " << tol << std::endl;
    std::cout << "Total values of r skipped for different l's: " << r_skipped
              << "/" << nr * nl << std::endl;
    return sum;
}

double evaluateInnerSum(unsigned int nx, unsigned int ny, unsigned int nz,
                        float r, float l_x, float l_y, float l_z,
                        float r_weight, float l_weight,
                        thrust::device_vector<double>& __restrict__ d_result,
                        double* __restrict__ d_sum, int blocks, int threads,
                        int gpu_num) {
    HANDLE_CUDA_ERROR(cudaSetDevice(gpu_num));

    evaluateIntegrandReduceZ<<<blocks, threads>>>(
        nx, ny, nz, r, l_x, l_y, l_z, raw_pointer_cast(d_result.data()));
    // for(long unsigned int i = 0; i < d_result.size(); i++)
    // std::cout << "d_result[" << i << "] = " << d_result[i] << std::endl;
    // Reduce vector on GPU within each block
    double delta_sum = thrust::reduce(d_result.begin(), d_result.end(),
                                      (double)0.0, thrust::plus<double>());
    // Accumulate result on device
    accumulateSum<<<1, 1>>>(delta_sum, r_weight, l_weight, d_sum);
    return delta_sum;
}  // evaluateInner

__global__ void evaluateIntegrandReduceZ(int nx, int ny, int nz, float r,
                                         float l_x, float l_y, float l_z,
                                         double* __restrict__ res) {
    // gets index for current thread and block
    // printf("threadIdx = %d, blockIdx = %d, blockDim = %d\n", threadIdx.x,
    // blockIdx.x, blockDim.x);
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < nx * ny) {
        int y_idx = idx / nx;
        int x_idx = idx % nx;
        float xvalue = d_x_grid[x_idx];
        float yvalue = d_x_grid[y_idx];
        float wx = d_x_weights[x_idx];
        float wy = d_x_weights[y_idx];
        // compute function value
        // exp(-α1|x1-c1| - α2|x1-c2| - α3|x1-c3+r*l| - α4|x1-c4+r*l|)
        // note |a-b| = sqrt( (a.x-b.x)^2 + (a.y-b.y)^2 + (a.z-b.z)^2 )

        float xdiffc_1 = xvalue - d_c[0];             // x1.x - c1.x
        float ydiffc_1 = yvalue - d_c[1];             // x1.y - c1.y
        float xdiffc_2 = xvalue - d_c[3];             // x1.x - c2.x
        float ydiffc_2 = yvalue - d_c[4];             // x1.y - c2.y
        float xdiffc_3 = xvalue - d_c[6] + r * l_x;   // x1.x - c3.x + lx
        float ydiffc_3 = yvalue - d_c[7] + r * l_y;   // x1.y - c3.y + ly
        float xdiffc_4 = xvalue - d_c[9] + r * l_x;   // x1.x - c4.x + lx
        float ydiffc_4 = yvalue - d_c[10] + r * l_y;  // x1.y - c4.y + ly

        // (x1.x - c1.x)^2 + (x1.y - c1.y)^2
        float xysq1 = xdiffc_1 * xdiffc_1 + ydiffc_1 * ydiffc_1;
        // (x1.x - c2.x)^2 + (x1.y - c2.y)^2
        float xysq2 = xdiffc_2 * xdiffc_2 + ydiffc_2 * ydiffc_2;
        // (x1.x - c3.x + lx)^2 + (x1.y - c3.y + ly)^2
        float xysq3 = xdiffc_3 * xdiffc_3 + ydiffc_3 * ydiffc_3;
        // (x1.x - c4.x + lx)^2 + (x1.y - c4.y + ly)^2
        float xysq4 = xdiffc_4 * xdiffc_4 + ydiffc_4 * ydiffc_4;

        double wxy = wx * wy;
        double v = 0.0;
        for (int z_idx = 0; z_idx < nz; ++z_idx) {
            float zvalue = d_x_grid[z_idx];
            float wz = d_x_weights[z_idx];
            float zdiffc_1 = zvalue - d_c[2];             // x1.z - c1.z
            float zdiffc_2 = zvalue - d_c[5];             // x1.z - c2.z
            float zdiffc_3 = zvalue - d_c[8] + r * l_z;   // x1.z - c3.z + lz
            float zdiffc_4 = zvalue - d_c[11] + r * l_z;  // x1.z - c4.z + lz
            // α1 * ✓|x - c1|
            float term1 = d_alpha[0] * sqrt(xysq1 + zdiffc_1 * zdiffc_1);
            // α2 * ✓|x - c2|
            float term2 = d_alpha[1] * sqrt(xysq2 + zdiffc_2 * zdiffc_2);
            // α3 * ✓|x - c3 + r*l|
            float term3 = d_alpha[2] * sqrt(xysq3 + zdiffc_3 * zdiffc_3);
            // α4 * ✓|x - c4 + r*l|
            float term4 = d_alpha[3] * sqrt(xysq4 + zdiffc_4 * zdiffc_4);
            float exponent = -term1 - term2 - term3 - term4 + r;
            v += exp(exponent) * wxy * wz;
        }
        res[idx] = v;
    }
}

__global__ void evaluateIntegrandReduceXY(int nx, int ny, int nz, float r,
                                          float l_x, float l_y, float l_z,
                                          double* __restrict__ res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < nz) {
        int z_idx = idx;
        float zvalue = d_x_grid[z_idx];
        float wz = d_x_weights[z_idx];
        // compute function value
        // exp(-α1|x1-c1| - α2|x1-c2| - α3|x1-c3+r*l| - α4|x1-c4+r*l|)
        // note |a-b| = sqrt( (a.x-b.x)^2 + (a.y-b.y)^2 + (a.z-b.z)^2 )
        float zdiffc_1 = zvalue - d_c[2];             // x1.z - c1.z
        float zdiffc_2 = zvalue - d_c[5];             // x1.z - c2.z
        float zdiffc_3 = zvalue - d_c[8] + r * l_z;   // x1.z - c3.z + lz
        float zdiffc_4 = zvalue - d_c[11] + r * l_z;  // x1.z - c4.z + lz

        float zsq1 = zdiffc_1 * zdiffc_1;
        float zsq2 = zdiffc_2 * zdiffc_2;
        float zsq3 = zdiffc_3 * zdiffc_3;
        float zsq4 = zdiffc_4 * zdiffc_4;

        double v = 0.0;
        for (int x_idx = 0; x_idx < nx; ++x_idx) {
            float xvalue = d_x_grid[x_idx];
            float wxz = d_x_weights[x_idx] * wz;

            float xdiffc_1 = xvalue - d_c[0];            // x1.x - c1.x
            float xdiffc_2 = xvalue - d_c[3];            // x1.x - c2.x
            float xdiffc_3 = xvalue - d_c[6] + r * l_x;  // x1.x - c3.x + lx
            float xdiffc_4 = xvalue - d_c[9] + r * l_x;  // x1.x - c4.x + lx

            float xsq1 = xdiffc_1 * xdiffc_1;
            float xsq2 = xdiffc_2 * xdiffc_2;
            float xsq3 = xdiffc_3 * xdiffc_3;
            float xsq4 = xdiffc_4 * xdiffc_4;

            for (int y_idx = 0; y_idx < ny; ++y_idx) {
                float yvalue = d_x_grid[y_idx];
                float wxyz = d_x_weights[y_idx] * wxz;

                float ydiffc_1 = yvalue - d_c[1];            // x1.y - c1.y
                float ydiffc_2 = yvalue - d_c[4];            // x1.y - c2.y
                float ydiffc_3 = yvalue - d_c[7] + r * l_y;  // x1.y - c3.y + ly
                float ydiffc_4 = yvalue - d_c[10] + r * l_y;  // x1.y - c4.y + ly

                // (x1.x - c1.x)^2 + (x1.y - c1.y)^2 + (x1.z - c1.z)^2
                float xyzsq1 = xsq1 + ydiffc_1 * ydiffc_1 + zsq1;
                // (x1.x - c2.x)^2 + (x1.y - c2.y)^2 + (x1.z - c2.z)^2
                float xyzsq2 = xsq2 + ydiffc_2 * ydiffc_2 + zsq2;
                // (x1.x - c3.x + lx)^2 + (x1.y - c3.y + ly)^2 + (x1.z - c3.z + lz)^2
                float xyzsq3 = xsq3 + ydiffc_3 * ydiffc_3 + zsq3;
                // (x1.x - c4.x + lx)^2 + (x1.y - c4.y + ly)^2 + (x1.z - c4.z + lz)^2
                float xyzsq4 = xsq4 + ydiffc_4 * ydiffc_4 + zsq4;

                // α1 * ✓|x - c1|
                float term1 = d_alpha[0] * sqrt(xyzsq1);
                // α2 * ✓|x - c2|
                float term2 = d_alpha[1] * sqrt(xyzsq2);
                // α3 * ✓|x - c3 + r*l|
                float term3 = d_alpha[2] * sqrt(xyzsq3);
                // α4 * ✓|x - c4 + r*l|
                float term4 = d_alpha[3] * sqrt(xyzsq4);

                float exponent = -term1 - term2 - term3 - term4 + r;
                v += exp(exponent) * wxyz;
            }
        }
        res[idx] = v;
    }
}

__global__ void accumulateSum(double result, float r_weight, float l_weight,
                              double* __restrict__ d_sum) {
    double sum = *d_sum;
    sum += result * r_weight * l_weight;
    *d_sum = sum;
}

}  // namespace cuslater
