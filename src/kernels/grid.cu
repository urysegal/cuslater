#include "kernels.cuh"
#include <tuple>
namespace cuslater {
    using namespace std;
    // result layout: idx = z * pitchXY + y * pitchX + x
    // where pitchX pads each row to a multiple of warp-aligned elements.
    __global__ void compute_distance_grid_zmajor(int n, int pitchX, real_t* __restrict__ result) {
        const int idx_xy  = blockIdx.x * blockDim.x + threadIdx.x;
        const int totalXY = n * n;
        if (idx_xy >= totalXY) return;

        const int y = idx_xy / n;
        const int x = idx_xy - y * n;

        const real3_t c1 = reinterpret_cast<const real3_t*>(d_c)[0];
        const real3_t c2 = reinterpret_cast<const real3_t*>(d_c + 3)[0];
        const real2_t a  = reinterpret_cast<const real2_t*>(d_alpha)[0];

        const real_t X = __ldg(&d_x_grid[x]);
        const real_t Y = __ldg(&d_y_grid[y]);

        const int row     = y * pitchX;
        const int pitchXY = pitchX * n;

        for (int z = 0; z < n; ++z) {
            const real_t Z  = __ldg(&d_z_grid[z]);
            const real_t d1 = norm(X - c1.x, Y - c1.y, Z - c1.z);
            const real_t d2 = norm(X - c2.x, Y - c2.y, Z - c2.z);
            const real_t v  = a.x * d1 + a.y * d2;

            result[z * pitchXY + row + x] = v;
        }
    }

    __global__ void compute_distance_pair_grid_zmajor(int n,
                                                      int pitchX, // pitch in real2_t elements
                                                      real2_t* __restrict__ result // stores {d1, d2}
    ) {
        const int idx_xy  = blockIdx.x * blockDim.x + threadIdx.x;
        const int totalXY = n * n;
        if (idx_xy >= totalXY) return;

        const int y = idx_xy / n;
        const int x = idx_xy - y * n;

        const real3_t c1 = reinterpret_cast<const real3_t*>(d_c + 0)[0];
        const real3_t c2 = reinterpret_cast<const real3_t*>(d_c + 3)[0];

        const real_t X = __ldg(&d_x_grid[x]);
        const real_t Y = __ldg(&d_y_grid[y]);

        const int row     = y * pitchX;
        const int pitchXY = pitchX * n;

        for (int z = 0; z < n; ++z) {
            const real_t Z  = __ldg(&d_z_grid[z]);
            const real_t d1 = norm(X - c1.x, Y - c1.y, Z - c1.z);
            const real_t d2 = norm(X - c2.x, Y - c2.y, Z - c2.z);

            result[z * pitchXY + row + x] = real2_t{d1, d2};
        }
    }

    tuple<CudaArray, int> build_distance_pair_grid(const int n) {
        constexpr int threads = THREADS_PER_BLOCK;

        // Align rows in units of real2_t
        const int warpBytes  = 128;
        const int alignElems = warpBytes / sizeof(real2_t); // float2: 16, double2: 8
        const int pitchX     = ((n + alignElems - 1) / alignElems) * alignElems;

        const size_t pitchXY    = size_t(pitchX) * n;  // number of real2_t elements per z-slab
        const size_t totalPairs = size_t(n) * pitchXY; // total number of real2_t entries

        CudaArray distance_pair_grid(2 * totalPairs);

        compute_distance_pair_grid_zmajor<<<(n * n + threads - 1) / threads, threads>>>(
            n, pitchX, reinterpret_cast<real2_t*>(distance_pair_grid.data()));

        cudaFuncSetCacheConfig(compute_distance_pair_grid_zmajor, cudaFuncCachePreferL1);
        cudaStreamSynchronize(0);

        return {std::move(distance_pair_grid), pitchX};
    }

    tuple<CudaArray, int> build_distance_grid(const int n) {
        constexpr int threads = THREADS_PER_BLOCK;
        // Build cache once per grid
        int warpBytes  = 128;
        int alignElems = warpBytes / sizeof(real_t); // 32 for float, 16 for double
        int pitchX     = ((n + alignElems - 1) / alignElems) * alignElems; // padded row length in
                                                                           // elements

        size_t pitchXY = size_t(pitchX) * n;
        size_t total   = size_t(n) * pitchXY; // z * (y * pitchX)

        CudaArray distance_grid(total);
        compute_distance_grid_zmajor<<<(n * n + threads - 1) / threads, threads>>>(n, pitchX,
                                                                                   distance_grid);
        cudaFuncSetCacheConfig(compute_distance_grid_zmajor, cudaFuncCachePreferL1);
        cudaStreamSynchronize(0);
        return {std::move(distance_grid), pitchX};
    }
} // namespace cuslater