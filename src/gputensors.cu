/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name(s) of the copyright holder(s) nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <array>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>
#include "../include/cuslater.cuh"
#include <cuda.h>

namespace cuslater {

void
init_cuslater()
{
    CUresult res;
    if ( (res=cuInit(0)) != CUDA_SUCCESS ) {
        fprintf(stderr, "Cannot init CUDA: %d\n",res);
        exit(1);
    }
}


int hadamard(std::vector<int> &modes, std::unordered_map<int, int64_t> &extent, const double *A, const double *C,
             double *D)
{
    typedef float floatTypeA;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    cutensorDataType_t          const typeA       = CUTENSOR_R_32F;
    cutensorDataType_t          const typeC       = CUTENSOR_R_32F;
    cutensorDataType_t          const typeD       = CUTENSOR_R_32F;
    cutensorComputeDescriptor_t const descCompute = CUTENSOR_COMPUTE_DESC_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.0f;
    floatTypeCompute gamma = (floatTypeCompute)1.0f;

    /**********************
     * Computing: D_{a,b,c} = alpha * A_{b,a,c}  + gamma * C_{a,b,c}
     **********************/

//    std::vector<int> modeC{'a','b','c'};
//    std::vector<int> modeA{'c','b','a'};
    int nmodes = modes.size();
//    int nmodeC = modeC.size();

//    std::unordered_map<int, int64_t> extent;
//    extent['a'] = 400;
//    extent['b'] = 200;
//    extent['c'] = 300;

    std::vector<int64_t> extentA;
    for (auto mode : modes)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentC;
    for (auto mode : modes)
        extentC.push_back(extent[mode]);
    std::vector<int64_t> extentD;
    for (auto mode : modes)
        extentD.push_back(extent[mode]);

    /**********************
     * Allocating data
     **********************/

    size_t elements = 1;
    for (auto mode : modes)
        elements *= extent[mode];

    size_t sizeM = sizeof(floatTypeA) * elements;
    printf("Total memory: %.2f GiB\n",(sizeM + sizeM)/1024./1024./1024);

    void *A_d, *C_d, *D_d;
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &A_d, sizeM));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &C_d, sizeM));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &D_d, sizeM));

    uint32_t const kAlignment = 128;  // Alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(A_d) % kAlignment == 0);
    assert(uintptr_t(C_d) % kAlignment == 0);
    assert(uintptr_t(D_d) % kAlignment == 0);


    if (A == nullptr || C == nullptr)
    {
        printf("Error: Host allocation of A or C.\n");
        return -1;
    }

    /*******************
     * Initialize data
     *******************/


    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(C_d, sizeM, C, sizeM, sizeM, 1, cudaMemcpyDefault, nullptr));
    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(D_d, sizeM, D, sizeM, sizeM, 1, cudaMemcpyDefault, nullptr));
    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(A_d, sizeM, A, sizeM, sizeM, 1, cudaMemcpyDefault, nullptr));

    /*************************
     * Memcpy perf 
     *************************/

    double minTimeMEMCPY = 1e100;
    cudaDeviceSynchronize();
    GPUTimer timer;
    timer.start();
    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(D_d, sizeM, C_d, sizeM, sizeM, 1, cudaMemcpyDefault, nullptr));
    cudaDeviceSynchronize();
    minTimeMEMCPY = timer.seconds();

    /*************************
     * cuTENSOR
     *************************/

    cutensorHandle_t handle;
    HANDLE_TENSOR_ERROR(cutensorCreate(&handle));

    /**********************
     * Create Tensor Descriptors
     **********************/

    cutensorTensorDescriptor_t  descA;
    HANDLE_TENSOR_ERROR(cutensorCreateTensorDescriptor(handle,
                                                &descA, nmodes, extentA.data(),
                                                nullptr /* stride */,
                                                typeA,
                                                kAlignment));

    cutensorTensorDescriptor_t  descC;
    HANDLE_TENSOR_ERROR(cutensorCreateTensorDescriptor(handle,
                                                &descC, nmodes, extentC.data(),
                                                nullptr /* stride */,
                                                typeC,
                                                kAlignment));

    cutensorTensorDescriptor_t  descD;
    HANDLE_TENSOR_ERROR(cutensorCreateTensorDescriptor(handle,
                                                &descD, nmodes, extentD.data(),
                                                nullptr /* stride */,
                                                typeD,
                                                kAlignment));
    /*******************************
     * Create Elementwise Binary Descriptor
     *******************************/

    cutensorOperationDescriptor_t  desc;
    HANDLE_TENSOR_ERROR(cutensorCreateElementwiseBinary(handle, &desc,
                                                 descA, modes.data(), /* unary operator A  */ CUTENSOR_OP_IDENTITY,
                                                 descC, modes.data(), /* unary operator C  */ CUTENSOR_OP_IDENTITY,
                                                 descD, modes.data(), /* unary operator AC */ CUTENSOR_OP_MUL,
                                                 descCompute));

    /*****************************
     * Optional (but recommended): ensure that the scalar type is correct.
     *****************************/

    cutensorDataType_t scalarType;
    HANDLE_TENSOR_ERROR(cutensorOperationDescriptorGetAttribute(handle, desc,
                                                         CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                                         (void*)&scalarType,
                                                         sizeof(scalarType)));

    assert(scalarType == CUTENSOR_R_32F);

    /**************************
    * Set the algorithm to use
    ***************************/

    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t  planPref;
    HANDLE_TENSOR_ERROR(cutensorCreatePlanPreference(handle,
                                              &planPref,
                                              algo,
                                              CUTENSOR_JIT_MODE_NONE));

    /**************************
     * Create Plan
     **************************/

    cutensorPlan_t  plan;
    HANDLE_TENSOR_ERROR(cutensorCreatePlan(handle,
                                    &plan,
                                    desc,
                                    planPref,
                                    0 /* workspaceSizeLimit */));

    /**********************
     * Run
     **********************/

    double minTimeCUTENSOR = 1e100;
    for (int i = 0; i < 3; i++)
    {
        HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(C_d, sizeM, C, sizeM, sizeM, 1, cudaMemcpyDefault, nullptr));
        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
        timer.start();
        HANDLE_TENSOR_ERROR(cutensorElementwiseBinaryExecute(handle, plan,
                                               (void*)&alpha, A_d,
                                               (void*)&gamma, C_d,
                                                              D_d, nullptr /* stream */));
        auto time = timer.seconds();
        minTimeCUTENSOR = (minTimeCUTENSOR < time)? minTimeCUTENSOR : time;
    }

    /*************************/

    double transferedBytes = sizeM;
    transferedBytes += ((float)alpha != 0.f) ? sizeM : 0;
    transferedBytes += ((float)gamma != 0.f) ? sizeM : 0;
    transferedBytes /= 1e9;
    printf("cuTensor: %.2f GB/s\n", transferedBytes / minTimeCUTENSOR);
    printf("memcpy: %.2f GB/s\n", 2 * sizeM / minTimeMEMCPY / 1e9 );

    HANDLE_TENSOR_ERROR(cutensorDestroy(handle));

    if (A_d) cudaFree(A_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);

    return 0;
}




}
