#pragma once

/**
 * @brief Type definitions for different precisions.
 * real_t is the floating-point type used for calculations.
 * real2_t, real3_t, and real4_t are vector types for 2D, 3D, and 4D respectively.
 */
#ifdef PRECISION_DOUBLE
using real_t  = double;
using real2_t = double2;
using real3_t = double3;
using real4_t = double4;
#else
using real_t  = float;
using real2_t = float2;
using real3_t = float3;
using real4_t = float4;
#endif

/**
 * @brief CUDA device function. Compute the norm of a 3D vector.
 *
 * @param v The 3D vector.
 * @return The norm of the vector.
 */
__device__ __forceinline__ real_t norm(real3_t v) {
    if constexpr (std::is_same<real_t, float>::value) {
        return norm3df(v.x, v.y, v.z);
    } else {
        return norm3d(v.x, v.y, v.z);
    }
}

/**
 * @brief CUDA device function. Compute the norm of a 3D vector.
 *
 * @param x The x component.
 * @param y The y component.
 * @param z The z component.
 * @return The norm of the vector.
 */
__device__ __forceinline__ real_t norm(real_t x, real_t y, real_t z) {
    if constexpr (std::is_same<real_t, float>::value) {
        return norm3df(x, y, z);
    } else {
        return norm3d(x, y, z);
    }
}

__device__ __forceinline__ real_t exp_fn(real_t x) {
    if constexpr (std::is_same<real_t, float>::value) {
        return __expf(x);
    } else {
        return exp(x);
    }
}

__host__ __device__ __forceinline__ real2_t make_real2(real_t x, real_t y) {
    return real2_t{x, y};
}

__host__ __device__ __forceinline__ real3_t make_real3(real_t x, real_t y, real_t z) {
    return real3_t{x, y, z};
}

__host__ __device__ __forceinline__ real4_t make_real4(real_t x, real_t y, real_t z, real_t w) {
    return real4_t{x, y, z, w};
}