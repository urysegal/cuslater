#include "kernels.cuh"

namespace cuslater {
    __launch_bounds__(THREADS_PER_BLOCK, 8) __global__
        void evalIntegrand_3DBloackReduce(int n, real_t hxyz, real_t rlx, real_t rly, real_t rlz,
                                          real_t r, int pitchX, real_t* __restrict__ alpha12,
                                          real_t* __restrict__ d_result) {
        int tid     = (blockIdx.x * blockDim.x + threadIdx.x);
        int totalXY = n * n;
        int y       = tid / n;
        int x       = tid - y * n;
        int pitchXY = pitchX * n;
        int rowBase = y * pitchX;
        int base    = rowBase + x;

        real_t local = real_t(0.0);
        if (__builtin_expect(tid < totalXY, 1)) [[likely]] {

            // load to registers, or at least to shared memory
            // to avoid global memory access
            real3_t c3    = reinterpret_cast<real3_t*>(d_c + 6)[0];
            real3_t c4    = reinterpret_cast<real3_t*>(d_c + 9)[0];
            real2_t alpha = reinterpret_cast<real2_t*>(d_alpha + 2)[0];

            // inform the compiler this is unlikely (__builtin_expect(cond, 0))
            if (__builtin_expect(x == 0 || x == n - 1, 0)) [[unlikely]] {
                hxyz *= 0.5; // half weight at endpoints
            }
            if (__builtin_expect(y == 0 || y == n - 1, 0)) [[unlikely]] {
                hxyz *= 0.5; // half weight at endpoints
            }
            // NOTE: BELOW z loop does not have half weights at the endpoints (to be done later)

            real_t xvalue = __ldg(&d_x_grid[x]);
            real_t yvalue = __ldg(&d_y_grid[y]);

            real_t xdiffc_3 = xvalue - c3.x + rlx;
            real_t xdiffc_4 = xvalue - c4.x + rlx;

            real_t ydiffc_3 = yvalue - c3.y + rly;
            real_t ydiffc_4 = yvalue - c4.y + rly;

            real_t v0 = 0, v1 = 0, v2 = 0, v3 = 0;

            constexpr int UNROLL = 8; // maybe 4 for double?
            int           k      = 0;

            // prefetch first UNROLL alpha12
            real_t a[UNROLL];
#pragma unroll
            for (int t = 0; t < UNROLL && t < n; ++t) {
                a[t] = __ldg(&alpha12[(k + t) * pitchXY + base]);
            }

            for (; k + UNROLL - 1 < n; k += UNROLL) {
                // compute on the prefetched
                real_t z0 = d_z_grid[k + 0], z1 = d_z_grid[k + 1], z2 = d_z_grid[k + 2],
                       z3 = d_z_grid[k + 3];
                real_t z4 = d_z_grid[k + 4], z5 = d_z_grid[k + 5], z6 = d_z_grid[k + 6],
                       z7 = d_z_grid[k + 7];

                v0 += exp_fn(r
                             - (a[0] + alpha.x * norm(xdiffc_3, ydiffc_3, z0 - c3.z + rlz)
                                + alpha.y * norm(xdiffc_4, ydiffc_4, z0 - c4.z + rlz)));
                v1 += exp_fn(r
                             - (a[1] + alpha.x * norm(xdiffc_3, ydiffc_3, z1 - c3.z + rlz)
                                + alpha.y * norm(xdiffc_4, ydiffc_4, z1 - c4.z + rlz)));
                v2 += exp_fn(r
                             - (a[2] + alpha.x * norm(xdiffc_3, ydiffc_3, z2 - c3.z + rlz)
                                + alpha.y * norm(xdiffc_4, ydiffc_4, z2 - c4.z + rlz)));
                v3 += exp_fn(r
                             - (a[3] + alpha.x * norm(xdiffc_3, ydiffc_3, z3 - c3.z + rlz)
                                + alpha.y * norm(xdiffc_4, ydiffc_4, z3 - c4.z + rlz)));

                v0 += exp_fn(r
                             - (a[4] + alpha.x * norm(xdiffc_3, ydiffc_3, z4 - c3.z + rlz)
                                + alpha.y * norm(xdiffc_4, ydiffc_4, z4 - c4.z + rlz)));
                v1 += exp_fn(r
                             - (a[5] + alpha.x * norm(xdiffc_3, ydiffc_3, z5 - c3.z + rlz)
                                + alpha.y * norm(xdiffc_4, ydiffc_4, z5 - c4.z + rlz)));
                v2 += exp_fn(r
                             - (a[6] + alpha.x * norm(xdiffc_3, ydiffc_3, z6 - c3.z + rlz)
                                + alpha.y * norm(xdiffc_4, ydiffc_4, z6 - c4.z + rlz)));
                v3 += exp_fn(r
                             - (a[7] + alpha.x * norm(xdiffc_3, ydiffc_3, z7 - c3.z + rlz)
                                + alpha.y * norm(xdiffc_4, ydiffc_4, z7 - c4.z + rlz)));

                // prefetch the NEXT UNROLL alpha12 early
                if (k + 2 * UNROLL - 1 < n) {
#pragma unroll
                    for (int t = 0; t < UNROLL; ++t) {
                        a[t] = __ldg(&alpha12[(k + UNROLL + t) * pitchXY + base]);
                    }
                }
            }

            // tail
            for (; k < n; ++k) {
                int    idx = k * pitchXY + base;
                real_t z   = d_z_grid[k];
                real_t a12 = __ldg(&alpha12[idx]);
                real_t t3  = alpha.x * norm(xdiffc_3, ydiffc_3, z - c3.z + rlz);
                real_t t4  = alpha.y * norm(xdiffc_4, ydiffc_4, z - c4.z + rlz);
                v0 += exp_fn(r - (a12 + t3 + t4));
            }

            real_t v = ((v0 + v1) + (v2 + v3));
            local += v * hxyz;
        }

        using BlockReduce = cub::BlockReduce<real_t, THREADS_PER_BLOCK>;

        __shared__ typename BlockReduce::TempStorage temp;

        real_t block_sum = BlockReduce(temp).Sum(local);
        if (__builtin_expect(threadIdx.x == 0, 0)) [[unlikely]]
            atomicAdd(d_result, block_sum);
    }
} // namespace cuslater