#pragma once
#include "number.cuh"
#include <cub/cub.cuh>
#include <tuple>

#define THREADS_PER_BLOCK 128
namespace cuslater {

    extern __constant__ real_t d_c[12];
    extern __constant__ real_t d_alpha[4];

    extern __constant__ real_t d_x_grid[500];
    extern __constant__ real_t d_y_grid[500];
    extern __constant__ real_t d_z_grid[500];

    __device__ __forceinline__ real_t simpson_coeff(int idx, int n) {
        if (idx == 0 || idx == n - 1) return real_t(1.0);
        return (idx & 1) ? real_t(4.0) : real_t(2.0);
    }

    __global__ void compute_distance_pair_grid_zmajor(int n,
                                                      int pitchX, // pitch in real2_t elements
                                                      real2_t* __restrict__ result // stores {d1, d2}
    );

    std::tuple<CudaArray, int> build_distance_pair_grid(const int n);

    __global__ void eval_1111_trap_all_samples(int n, real_t hxyz, int pitchX, int blocksXY, int nr,
                                               int nl, const real2_t* __restrict__ r_grid_dev,
                                               const real4_t* __restrict__ l_grid_dev,
                                               const real2_t* __restrict__ distance_pair_grid,
                                               real_t* __restrict__ d_out);

    __global__ void eval_1111_simpson_all_samples(int n, real_t hxyz, int pitchX, int blocksXY, int nr,
                                                  int nl, const real2_t* __restrict__ r_grid_dev,
                                                  const real4_t* __restrict__ l_grid_dev,
                                                  const real2_t* __restrict__ distance_pair_grid,
                                                  real_t* __restrict__ d_out);

    /*============================= SP kernels =========================*/

    // initial kernel that does 1111 integral fast.
    __launch_bounds__(THREADS_PER_BLOCK, 8) __global__
        void evalIntegrand_3DBloackReduce(int n, real_t hxyz, real_t rlx, real_t rly, real_t rlz,
                                          real_t r, int pitchX, real_t* __restrict__ alpha12,
                                          real_t* __restrict__ d_result);

    // experimental testing for multi-integrals
    __global__ void evalIntegrand_3DBloackReduce_SMulti(int n, real_t hxyz, real_t rlx, real_t rly,
                                                        real_t rlz, real_t r, int pitchX,
                                                        const real2_t* __restrict__ distance_pair_grid,
                                                        real_t* __restrict__ d_out /* length 16 */);

    __global__ void evalIntegrand_3DBloackReduce_SSPS(int n, real_t hxyz, real_t rlx, real_t rly,
                                                      real_t rlz, real_t r, int pitchX,
                                                      const real2_t* __restrict__ distance_pair_grid,
                                                      real_t* __restrict__ d_out /* length 24 */);

    __global__ void evalIntegrand_3DBloackReduce_SSSS_SSSP(
        int n, real_t hxyz, real_t rlx, real_t rly, real_t rlz, real_t r, int pitchX,
        const real2_t* __restrict__ distance_pair_grid, real_t* __restrict__ d_out /* length 40 */);

    __global__ void evalIntegrand_3DBloackReduce_SSSS_SSSP_SSPS(
        int n, real_t hxyz, real_t rlx, real_t rly, real_t rlz, real_t r, int pitchX,
        const real2_t* __restrict__ distance_pair_grid, real_t* __restrict__ d_out /* length 64 */);

    __global__ void evalIntegrand_3DBloackReduce_SSSS_SSSP_SSPS_SPSS(
        int n, real_t hxyz, real_t rlx, real_t rly, real_t rlz, real_t r, int pitchX,
        const real2_t* __restrict__ distance_pair_grid, real_t* __restrict__ d_out /* length 88 */);

    __global__ void evalIntegrand_3DBloackReduce_SSSS_SSSP_SSPS_SPSS_PSSS(
        int n, real_t hxyz, real_t rlx, real_t rly, real_t rlz, real_t r, int pitchX,
        const real2_t* __restrict__ distance_pair_grid, real_t* __restrict__ d_out /* length 112 */);

    /*============================= DDDD kernels =========================*/
    __global__ void evalIntegrand_3DBloackReduce_DDDD_81(int n, real_t hxyz, real_t rlx, real_t rly,
                                                         real_t rlz, real_t r, int pitchX,
                                                         const real2_t* __restrict__ distance_pair_grid,
                                                         real_t* __restrict__ d_out /* length 81
                                                                                     */
    );

    /*============================= fused kernels =========================*/

    __global__ void evalIntegrand_3DBloackReduce_SSSS_SSSP_SSPS_SPSS_all_samples(
        int n, real_t hxyz, int pitchX, int blocksXY, int nr, int nl,
        const real2_t* __restrict__ r_grid_dev, const real4_t* __restrict__ l_grid_dev,
        const real2_t* __restrict__ distance_pair_grid,
        real_t* __restrict__ d_out // length 88, final weighted sums
    );

    __global__ void evalIntegrand_3DBloackReduce_DDDD_81_all_samples(
        int n, real_t hxyz, int pitchX, int blocksXY, int nr, int nl,
        const real2_t* __restrict__ r_grid_dev, const real4_t* __restrict__ l_grid_dev,
        const real2_t* __restrict__ distance_pair_grid, real_t* __restrict__ d_out /* length 81
                                                                                    */
    );

    /**
     *  global D component id
        0 -> xy
        1 -> xz
        2 -> yz
        3 -> x2-y2
        4 -> z2 = 2*z*z - x*x - y*y

        this kernel computes 3^4 = 81 components. Choose C0, C1, C2 in {0, 1, 2, 3, 4} to select
        which component to compute for each of the 4 electrons.
     */
    template<int C0, int C1, int C2>
    __global__ void evalIntegrand_3DBloackReduce_DDDD_81_components_all_samples(
        int n, real_t hxyz, int pitchX, int blocksXY, int nr, int nl,
        const real2_t* __restrict__ r_grid_dev, const real4_t* __restrict__ l_grid_dev,
        const real2_t* __restrict__ distance_pair_grid, real_t* __restrict__ d_out /* length 81 */
    );

    /**
     * Full generic 625-components DDDD kernel. Ordered as follows: (25 by 25)
     * 25 for the left 2 electrons, 25 for the right 2 electrons.
     * Need to supply:
     *      LEFT_BEGIN:
     *      LEFT_COUNT:
     *      RIGHT_BEGIN:
     *      RIGHT_COUNT:
     *
     * 25 by 25 components:
     * 9x9, 9x9, 9x7
     * 9x9, 9x9, 9x7
     * 7x9, 7x9, 7x7
     */
    template<int LEFT_BEGIN, int LEFT_COUNT, int RIGHT_BEGIN, int RIGHT_COUNT>
    __global__ void evalIntegrand_3DBloackReduce_DDDD_tile_all_samples(
        int n, real_t hxyz, int pitchX, int blocksXY, int nr, int nl,
        const real2_t* __restrict__ r_grid_dev, const real4_t* __restrict__ l_grid_dev,
        const real2_t* __restrict__ distance_pair_grid,
        real_t* __restrict__ d_out // length LEFT_COUNT * RIGHT_COUNT
    );

} // namespace cuslater