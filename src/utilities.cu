//
// Created by gkluhana on 04/03/24.
//
#include "../include/utilities.h"
#include <cassert>

namespace cuslater{
    void make_1d_grid_simpson(double start, double stop, unsigned int N, std::vector<double>* grid, std::vector<double>* weights){
        // N must be multiple of 3
        assert( N % 3 == 0);

        auto node = start;
        auto h = (stop - start) / N;
        auto weight_factor = h * (3.0/8.0);
        for (unsigned int i=0;i < N+1; ++i){
            grid->push_back(node);
            if (((i ) % 3 == 0 && i > 0) && (i < (N))){
                weights -> push_back(weight_factor * 2);
            }
            else if (( i > 0) && (i < (N))){
                weights -> push_back(weight_factor * 3);
            }
            else{
                weights -> push_back(weight_factor);
            }
            node += h;

        }
    }

    double make_1d_grid(double start, double stop, unsigned int N, std::vector<double>* grid){
        auto val = start;
        auto step = (stop - start) / N;
        for (unsigned int i=0;i < N; ++i){
            grid->push_back(val);
//	    std::cout << "pushed value to grid: " << val << std::endl;
            val += step;
        }
        return step;
    }
    __device__
    unsigned long upper_power_of_two(unsigned long v)
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;

    }

    __global__
    void reduceSum(double *input, double *output,  int size)
    {
        extern __shared__ double tsum[];
        int id = threadIdx.x;
        int tid = blockDim.x*blockIdx.x + threadIdx.x;
        int stride = gridDim.x*blockDim.x;
        tsum[id] = 0.0;
        for(int k = tid; k<size; k+=stride) tsum[id] += input[k];
        __syncthreads();
        int block2 = upper_power_of_two(static_cast<unsigned long>(blockDim.x));
        for(int k = block2 /2; k>0; k >>=1) {
            if(id<k && id+k<blockDim.x) tsum[id] += tsum[id+k];
            __syncthreads();
        }
        if (id==0) output[blockIdx.x] = tsum[0];
    }


    __global__
    void reduceSumFast(const float* __restrict data,float* __restrict sums, int n)
    {
        auto grid = cg::this_grid();
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<32>(block);

        float v = 0.0f;

        for (int tid = grid.thread_rank(); tid<n; tid += grid.size())
            v+= data[tid];
        warp.sync();
        v = cg::reduce(warp, v, cg::plus<float> () );

        if(warp.thread_rank()==0)
            atomicAdd(&sums[block.group_index() .x],v);
    }

    __global__
    void multiplyVolumeElement(int x_dim,
                               double dxdydz,
                               double *res)
    {
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int bz = blockIdx.z;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int tz = threadIdx.z;
        int h = gridDim.z*blockDim.z;
        int d = gridDim.y*blockDim.y;
        int xpos = 256*blockIdx.x;
        xpos += threadIdx.x;
        int idx = h*d*(blockDim.x*bx + tx)+ d*(blockDim.y*by + ty)+ (blockDim.z*bz + tz);

        if ( idx < x_dim*x_dim*x_dim ) {
            res[idx] = res[idx]*dxdydz;
        }
    }



}