// Created by gkluhana on 26/03/24.
//
// editted by MarkEwert03 on 13/05/24

#include "cuslater.cuh"
#include "evalIntegral.cuh"
#include "grids.h"
#include "utilities.h"
#include <chrono>
#include <iomanip> // for std::setprecision
#include <iostream>
using namespace std;

int main(int argc, const char* argv[]) {
    HANDLE_CUDA_ERROR(cudaSetDevice(0));
    // Default Parameter Values
    cuslater::ProgramParameters sys;
    // Process Input Parameters
    cuslater::handleArguments(argc, argv, sys);

    int nr = sys.nr;
    int nl = sys.nl;
    int nx = sys.nx;
    int ny = sys.ny;
    int nz = sys.nz;
    if (nx != ny || nx != nz || ny != nz) {
        std::cerr << "nx, ny, and nz must be equal for this example." << std::endl;
        return 1;
    }
    real_t alpha[4];
    real_t c[12];

    for (int i = 0; i < 4; ++i) {
        alpha[i] = sys.alpha[i];
    }
    for (int i = 0; i < 12; ++i) {
        c[i] = sys.c[i];
    }

    const int n = nx;

    int gpu = 0;
    cudaGetDeviceCount(&gpu);

    for (int i = 0; i < gpu; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("GPU %d: %s\n", i, prop.name);
        printf("  Total Global Memory: %zu bytes\n", prop.totalGlobalMem);
        printf("  Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Memory Clock Rate: %d kHz\n", prop.memoryClockRate);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth: %.2f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0) / 1e6);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Multiprocessor Count: %d\n", prop.multiProcessorCount);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads Dimension: (%d, %d, %d)\n", prop.maxThreadsDim[0],
               prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    }

    double tol             = sys.tol;
    int    check_zero_cond = sys.check_zero_cond;

    std::cout << "Evaluating Integral for all values of r and l with\n";
    std::cout << "  a1=" << alpha[0] << ", a2=" << alpha[1] << ", a3=" << alpha[2]
              << ", a4=" << alpha[3] << "\n";
    std::cout << "  c1 = (" << c[0] << ", " << c[1] << ", " << c[2] << ")\n";
    std::cout << "  c2 = (" << c[3] << ", " << c[4] << ", " << c[5] << ")\n";
    std::cout << "  c3 = (" << c[6] << ", " << c[7] << ", " << c[8] << ")\n";
    std::cout << "  c4 = (" << c[9] << ", " << c[10] << ", " << c[11] << ")\n";

    std::cout << "nr: " << nr << " nl: " << nl << " nx: " << nx << " ny: " << ny
              << " nz: " << nz << std::endl;

    vector<real2_t> r = cuslater::read_r_grid(nr);
    vector<real4_t> l = cuslater::read_l_grid(nl);

    cuslater::Metric metric;

    auto start = std::chrono::high_resolution_clock::now();

    double sum = cuslater::evaluateFourCenterIntegral(c, alpha, r, l, n, tol,
                                                      check_zero_cond, &metric);

    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << metric << std::endl;
    std::cout << "Tolerance: " << tol << std::endl;
    std::cout << "result: " << std::fixed
              << std::setprecision(std::numeric_limits<double>::max_digits10) << sum
              << std::endl;
    std::cout << "Time Elapsed: " << duration.count() / 1e6 << " seconds" << std::endl;
}
