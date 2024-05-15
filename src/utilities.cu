//
// Created by gkluhana on 04/03/24.
//

#include "../include/utilities.h"
namespace cuslater {

void getAvailableMemory(size_t &availableMemory) {
    size_t free_bytes, total_bytes;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    availableMemory = free_bytes;
}

__device__ unsigned long upper_power_of_two(unsigned long v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

__global__ void reduceSum(double *input, double *output, int size) {
    extern __shared__ double tsum[];
    int id = threadIdx.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    tsum[id] = 0.0;
    for (int k = tid; k < size; k += stride) tsum[id] += input[k];
    __syncthreads();
    int block2 = upper_power_of_two(static_cast<unsigned long>(blockDim.x));
    for (int k = block2 / 2; k > 0; k >>= 1) {
        if (id < k && id + k < blockDim.x) tsum[id] += tsum[id + k];
        __syncthreads();
    }
    if (id == 0) output[blockIdx.x] = tsum[0];
}

__global__ void reduceSumWrapper(double *d_results_i, int blocks, int threads) {
    // Reduce vector on GPU within each block
    int blocks_evaluated = blocks;
    int numBlocksReduced = (blocks + threads - 1) / threads;
    // Reduce vector down to < threads_per_block
    while (blocks > threads) {
        reduceSum<<<numBlocksReduced, threads, threads * sizeof(double)>>>(
            d_results_i, d_results_i, blocks);
        blocks = numBlocksReduced;
        numBlocksReduced = (blocks + threads - 1) / threads;
    }

    // Reduce vector down to 1 value
    reduceSum<<<1, blocks, blocks * sizeof(double)>>>(d_results_i, d_results_i,
                                                      blocks);
    blocks = blocks_evaluated;
}

__global__ void reduceSumWithWeights(double *input, double *output,
                                     double *weights, int size) {
    extern __shared__ double tsum[];
    int id = threadIdx.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    tsum[id] = 0.0;
    for (int k = tid; k < size; k += stride) tsum[id] += input[k] * weights[k];
    __syncthreads();
    int block2 = upper_power_of_two(static_cast<unsigned long>(blockDim.x));
    for (int k = block2 / 2; k > 0; k >>= 1) {
        if (id < k && id + k < blockDim.x) tsum[id] += tsum[id + k];
        __syncthreads();
    }
    if (id == 0) output[blockIdx.x] = tsum[0];
}

__global__ void reduceSumFast(const float *__restrict data,
                              float *__restrict sums, int n) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    float v = 0.0f;

    for (int tid = grid.thread_rank(); tid < n; tid += grid.size())
        v += data[tid];
    warp.sync();
    v = cg::reduce(warp, v, cg::plus<float>());

    if (warp.thread_rank() == 0) atomicAdd(&sums[block.group_index().x], v);
}

__global__ void multiplyVolumeElement(int x_dim, double dxdydz, double *res) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int h = gridDim.z * blockDim.z;
    int d = gridDim.y * blockDim.y;
    int xpos = 256 * blockIdx.x;
    xpos += threadIdx.x;
    int idx = h * d * (blockDim.x * bx + tx) + d * (blockDim.y * by + ty) +
              (blockDim.z * bz + tz);

    if (idx < x_dim * x_dim * x_dim) {
        res[idx] = res[idx] * dxdydz;
    }
}

}  // namespace cuslater
