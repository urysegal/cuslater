#pragma once
#include "cudapp.cuh"
#include <chrono>
#include <ratio>

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

#if defined(PRECISION_DOUBLE)
using vec_t              = double2; // 16B
static constexpr int VEC = 2;
#else
using vec_t              = float4; // 16B
static constexpr int VEC = 4;
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

__device__ __forceinline__ double approx_sqrt_from_float(double x) {

    // double y = static_cast<double>(__fsqrt_rn(static_cast<float>(x)));

    // return 0.5 * (y + x / y);
    float xf = static_cast<float>(x);

    float rf = rsqrtf(xf); // approx 1/sqrt(x), float

    double y = static_cast<double>(rf);

    // Newton refinement for reciprocal sqrt:

    // y <- y * (1.5 - 0.5*x*y*y)

    y = y * (1.5 - 0.5 * x * y * y);

    // sqrt(x) = x * rsqrt(x)

    return x * y;
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
        // sqrt
        // return norm3d(x, y, z);

        // fsqrt + Newton
        // real_t a = norm3df(x, y, z);
        // real_t a  = (x * x + y * y + z * z);
        // real_t xk = __fsqrt_rn(a);
        // return (a / xk + xk) * real_t(0.5);

        // frqrt + newton approach
        // double r2 = fma(x, x, fma(y, y, z * z));
        // float r2f = static_cast<float>(r2);
        // float invf = __frsqrt_rn(r2f);
        // double inv = static_cast<double>(invf);
        // // double inv = rsqrt(r2);
        // inv = 0.5 * inv * (3 - r2 * inv * inv);
        // return r2 * inv;

        // fast asm frqrt + newton approach
        double r2  = fma(x, x, fma(y, y, z * z));
        float  r2f = static_cast<float>(r2);
        float  invf;
        asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(invf) : "f"(r2f));
        double inv = static_cast<double>(invf);
        double s   = r2 * inv;
        return s * (1.5 - 0.5 * s * inv);

        // fast asm frsqrt + second order Newton
        // double r2 = fma(x, x, fma(y, y, z * z));
        // float  af = static_cast<float>(r2);
        // float  y0f;
        // asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y0f) : "f"(af));
        // double y = static_cast<double>(y0f);
        // double r = 1 - r2 * y * y;
        // double c = r * (0.375 * r + 0.5);
        // return r2 * y * (1 + c);
    }
}

__device__ __forceinline__ real_t exp_fn(real_t x) {
    if constexpr (std::is_same<real_t, float>::value) {
        return __expf(x);
    } else {
        // return exp(x);
        // __device__ __forceinline__ double exp_njuffa_core(double a) {
        const double ln2_hi = 6.9314718055829871e-01;
        const double ln2_lo = 1.6465949582897082e-12;
        const double l2e    = 1.4426950408889634;
        const double cvt    = 6755399441055744.0; // 3 * 2^51

        double f, j, p;
        int    i;

        // i = rint(a / log(2)), using magic-number rounding
        j = fma(l2e, x, cvt);
        i = __double2loint(j);
        j = j - cvt;

        // f = a - i * ln2
        f = fma(j, -ln2_hi, x);
        f = fma(j, -ln2_lo, f);

        // tuned polynomial for exp(f)
        p = 2.5022018235176802e-8;
        p = fma(p, f, 2.7630903481118922e-7);
        p = fma(p, f, 2.7557514543922205e-6);
        p = fma(p, f, 2.4801491039429033e-5);
        p = fma(p, f, 1.9841269589083001e-4);
        p = fma(p, f, 1.3888888945916664e-3);
        p = fma(p, f, 8.3333333334557492e-3);
        p = fma(p, f, 4.1666666666519782e-2);
        p = fma(p, f, 1.6666666666666477e-1);
        p = fma(p, f, 5.0000000000000122e-1);
        p = fma(p, f, 1.0000000000000000e+0);
        p = fma(p, f, 1.0000000000000000e+0);

        // scale by 2^i by adding i to the exponent bits of p
        int rlo = __double2loint(p);
        int rhi = (i << 20) + __double2hiint(p);

        return __hiloint2double(rhi, rlo);
        // }
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

namespace cuslater {
    struct CudaArray {
        real_t*     d_array;
        std::size_t n;
        explicit inline CudaArray(int size) {
            cudaMalloc(&this->d_array, sizeof(real_t) * size);
            this->n = size;
        }
        ~CudaArray() {
            cudaFree(this->d_array);
            this->n = 0;
        }
        // copy
        CudaArray(const CudaArray& rhs) = delete;
        // move
        CudaArray(CudaArray&& rhs) {
            this->d_array = rhs.d_array;
            this->n       = rhs.n;
            rhs.d_array   = nullptr;
            rhs.n         = 0;
        }

        // copy assign
        CudaArray& operator=(const CudaArray& rhs) = delete;
        // move assign
        CudaArray& operator=(CudaArray&& rhs) {
            if (this != &rhs) {
                cudaFree(this->d_array);
                this->d_array = rhs.d_array;
                rhs.d_array   = nullptr;
                rhs.n         = 0;
            }
            return *this;
        }

        void setZero() {
            cudaMemset(this->d_array, 0, sizeof(real_t) * n);
        }

        // to real_t
        operator real_t() const  = delete;
        operator real_t&() const = delete;
        operator real_t*() const {
            return this->d_array;
        }

        real_t* data() noexcept {
            return d_array;
        }
        const real_t* data() const noexcept {
            return d_array;
        }
        std::size_t size() const noexcept {
            return n;
        }

        // span interface
        real_t* begin() noexcept {
            return d_array;
        }
        real_t* end() noexcept {
            return d_array + n;
        }
        const real_t* begin() const noexcept {
            return d_array;
        }
        const real_t* end() const noexcept {
            return d_array + n;
        }
    };

    struct CudaNumber {
        real_t* d_number;

        inline CudaNumber(real_t value = real_t(0)) {
            cudaMalloc(&this->d_number, sizeof(real_t));
            cudaMemcpy(this->d_number, &value, sizeof(real_t), cudaMemcpyHostToDevice);
        }
        ~CudaNumber() {
            cudaFree(this->d_number);
        }
        CudaNumber(const CudaNumber&)            = delete;
        CudaNumber(const CudaNumber&&)           = delete;
        CudaNumber& operator=(const CudaNumber&) = delete;

        void setZero(cudapp::CudaStream& stream) {
            real_t zero = real_t(0);
            stream.copyH2D(d_number, &zero, sizeof(real_t));
        }

        void set(real_t value, cudapp::CudaStream& stream) {
            stream.copyH2D(d_number, &value, sizeof(real_t));
        }

        void get(real_t& value, cudapp::CudaStream& stream) const {
            stream.copyD2H(&value, d_number, sizeof(real_t));
        }

        void operator=(const real_t val) {
            cudaMemcpyAsync(this->d_number, &val, sizeof(real_t), cudaMemcpyHostToDevice);
        }

        operator real_t() const {
            real_t h_value;
            cudaMemcpyAsync(&h_value, this->d_number, sizeof(real_t), cudaMemcpyDeviceToHost);
            return h_value;
        }

        // operator real_t&() const = delete;
        operator real_t*() const {
            return this->d_number;
        }
    };

    struct CudaEvent {
      private:
        cudaEvent_t event;

      public:
        CudaEvent() {
            cudaEventCreate(&event);
        }
        ~CudaEvent() {
            cudaEventDestroy(event);
        }
        CudaEvent(const CudaEvent&)            = delete;
        CudaEvent(const CudaEvent&&)           = delete;
        CudaEvent& operator=(const CudaEvent&) = delete;

        void record() {
            cudaEventRecord(event, 0);
        }

        std::chrono::microseconds since(const CudaEvent& other) const {
            float ms;
            cudaEventElapsedTime(&ms, other.event, event);
            // return ms;
            return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::duration<float, std::milli>(ms));
        }
    };
} // namespace cuslater