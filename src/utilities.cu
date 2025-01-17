//
// Created by gkluhana on 04/03/24.
//

#include "../include/utilities.h"
#include <cstring>

namespace cuslater{

    void handleArguments(int argc, const char* argv[], ProgramParameters& params) {

   	int nr=params.nr, nl=params.nl, nx=params.nx, ny=params.ny, nz=params.nz;
	double tol=params.tol;
	bool check_zero_cond=params.check_zero_cond;

	real_t alpha[4] ;
	real_t c[12];
	for (int i = 0; i < 4; ++i) {
    		alpha[i] = params.alpha[i];
	}	
	for (int i = 0; i < 12; ++i) {
    		c[i] = params.c[i];
	}	

	for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) {
            // Print help message
            std::cout << "Usage: ./simple [OPTION] [PARAMETERS ...]\n";
            std::cout << "Options:\n";
            std::cout << "  --help\t\tDisplay this help message and exit\n";
            std::cout << "  -a a1 a2 a3 a4\tSet alpha values\n";
            std::cout << "  -c1 x1 y1 z1\t\tSets coordinates for c1 (c1.x, c1.y, c1.z)\n";
            std::cout << "  -c2 x2 y2 z2\t\tSets coordinates for c2 (c2.x, c2.y, c2.z)\n";
            std::cout << "  -c3 x3 y3 z3\t\tSets coordinates for c3 (c3.x, c3.y, c3.z)\n";
            std::cout << "  -c4 x4 y4 z4\t\tSets coordinates for c4 (c4.x, c4.y, c4.z)\n";
            std::cout << "  -t tol\t\tSet tolerance value\n";
            std::cout << "  -l nl\t\tSet nl\n";
            std::cout << "  -r nr\t\tSet nr\n";
            std::cout << "  -n nx\t\tSet nx, ny, nz\n";
            std::cout << "  -z \t\tChecks Zero Condition\n";
            exit(EXIT_SUCCESS); // Exit after printing help message

        } else if (std::strcmp(argv[i], "-a") == 0) {
            // Read next 4 values as alphas
            if (i + 4 >= argc) {
                std::cerr << "Error: Fewer than 4 alpha parameters provided.\n";
                exit(EXIT_FAILURE);
            }
            for (int j = 1; j <= 4; ++j) {
                // a_j = i+1, i+2, i+3, i+4
                try {
                    alpha[j - 1] = std::atof(argv[i + j]);
                } catch (...) {
                    std::cerr << "Error: Insufficient numerical values provided for -a option.\n";
                    exit(EXIT_FAILURE);
                }
            }
            i += 4; // Skip over processed alpha values

        } else if (argv[i][0] == '-' && argv[i][1] == 'c') {
            int centNum = 0;
            std::string argi = argv[i];
            if (argi.length() != 3) {
                std::cerr << "Error: Please provide 1 digit for the center number.\n";
                exit(EXIT_FAILURE);
            } else if (argi.substr(2, 1) != "1" && argi.substr(2, 1) != "2" &&
                       argi.substr(2, 1) != "3" && argi.substr(2, 1) != "4") {
                std::cerr << "Error: Please provide `-ci` where i = 1,2,3,4.\n";
                exit(EXIT_FAILURE);
            }
            // shift by '0' to convert char->int
            centNum = argi[2] - '0';
            // Read next 3 values coordinates for ci
            if (i + 3 >= argc) {
                std::cerr << "Error: Fewer than 3 coordinates provided for c" << centNum << ".\n";
                exit(EXIT_FAILURE);
            }
            for (int j = 0; j < 3; ++j) {
                // ci_j = i+1, i+2, i+3
                try {
                    c[(centNum - 1) * 3 + j] = std::atof(argv[i + 1 + j]);
                } catch (...) {
                    std::cerr << "Error: Insufficient numerical values provided for -c" << centNum
                              << " option.\n";
                    exit(EXIT_FAILURE);
                }
            }
            i += 3; // Skip over processed ci values

        } else if (std::strcmp(argv[i], "-t") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: No tolerance parameter provided.\n";
                exit(EXIT_FAILURE);
            } else {
                tol = std::atof(argv[i + 1]);
                ++i; // Skip over tol value
            }
        } else if (std::strcmp(argv[i], "-z") == 0) {
		check_zero_cond= true ;
        } else if (std::strcmp(argv[i], "-l") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: No nl parameter provided.\n";
                exit(EXIT_FAILURE);
            } else {
                nl = std::atof(argv[i + 1]);
                ++i; // Skip over tol value
            }
        } else if (std::strcmp(argv[i], "-r") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: No nr parameter provided.\n";
                exit(EXIT_FAILURE);
            } else {
                nr = std::atof(argv[i + 1]);
                ++i; // Skip over tol value
            }
        } else if (std::strcmp(argv[i], "-n") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: No nx parameter provided.\n";
                exit(EXIT_FAILURE);
            } else {
                nx = std::atof(argv[i + 1]);
                ny = nx;
                nz = nx;
                ++i; // Skip over tol value
            }
        } else {
            std::cerr << "Error: Invalid command line parameter. Use ./simple --help for more "
                         "information.\n";
            exit(EXIT_FAILURE);
        }
    }
	//Update Program Parameters
	params.nr = nr;
	params.nl = nl;
	params.nx = nx;
	params.ny = ny;
	params.nz = nz;
	params.tol = tol;
	for (int i = 0; i < 4; ++i) {
    		params.alpha[i] = alpha[i];
	}	
	for (int i = 0; i < 12; ++i) {
    		params.c[i] = c[i];
	}	
	params.check_zero_cond = check_zero_cond;
    }
    void getAvailableMemory(size_t& availableMemory) {
        size_t free_bytes, total_bytes;
        cudaError_t  err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err != cudaSuccess){
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

__global__ void reduceSumFast(const real_t *__restrict data,
                              real_t *__restrict sums, int n) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    real_t v = 0.0f;

    for (int tid = grid.thread_rank(); tid < n; tid += grid.size())
        v += data[tid];
    warp.sync();
    v = cg::reduce(warp, v, cg::plus<real_t>());

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
