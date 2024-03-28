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

    cudaDataType_t typeA = CUDA_R_64F;
    cudaDataType_t typeC = CUDA_R_64F;
    cudaDataType_t typeCompute = CUDA_R_64F;

    double alpha = 1;
    double gamma = 1;

    /**********************
     * Computing: D_{a,b,c} =  A_{a,b,c}  *  C_{a,b,c}
     **********************/

    //std::vector<int> modeC{'a','b','c'};
    //std::vector<int> modeA{'c','b','a'};
    int nmodes = modes.size();
//    int nmodeC = modeC.size();


    //extent['a'] = 400;
    //extent['b'] = 200;
    //extent['c'] = 300;

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

    size_t sizeM = sizeof(double) * elements;
    printf("Total memory: %.2f GiB\n",(sizeM + sizeM)/1024./1024./1024);

    void *A_d, *C_d, *D_d;
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &A_d, sizeM));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &C_d, sizeM));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &D_d, sizeM));

    /*******************
     * Initialize data
     *******************/

    cudaDeviceSynchronize();
    GPUTimer timer;
    timer.start();

    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(C_d, sizeM, C, sizeM, sizeM, 1, cudaMemcpyDefault, 0));
    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(A_d, sizeM, A, sizeM, sizeM, 1, cudaMemcpyDefault, 0));

    /*************************
     * Memcpy perf
     *************************/

    double minTimeMEMCPY = 1e100;
    cudaDeviceSynchronize();
    minTimeMEMCPY = timer.seconds();

    /*************************
     * cuTENSOR
     *************************/
    cutensorStatus_t err;
    cutensorHandle_t handle;
    HANDLE_TENSOR_ERROR(cutensorInit(&handle));

    /**********************
     * Create Tensor Descriptors
     **********************/
    cutensorTensorDescriptor_t descA;
    HANDLE_TENSOR_ERROR(cutensorInitTensorDescriptor( &handle,
                                               &descA,
                                               nmodes,
                                               extentA.data(),
                                               NULL /* stride */,
                                               typeA, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descC;
    HANDLE_TENSOR_ERROR(cutensorInitTensorDescriptor( &handle,
                                               &descC,
                                               nmodes,
                                               extentC.data(),
                                               NULL /* stride */,
                                               typeC, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descD;
    HANDLE_TENSOR_ERROR(cutensorInitTensorDescriptor( &handle,
                                               &descD,
                                               nmodes,
                                               extentD.data(),
                                               NULL /* stride */,
                                               typeC, CUTENSOR_OP_IDENTITY));


        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
        timer.start();
        err = cutensorElementwiseBinary(&handle,
                                        (void*)&alpha, A_d, &descA, modes.data(),
                                        (void*)&gamma, C_d, &descC, modes.data(),
                                        D_d, &descD, modes.data(),
                                        CUTENSOR_OP_MUL, typeCompute, 0 /* stream */);
        auto time = timer.seconds();
        if (err != CUTENSOR_STATUS_SUCCESS)
        {
            printf("ERROR: %s\n", cutensorGetErrorString(err) );
        }

    HANDLE_CUDA_ERROR(cudaMemcpy2D(D_d, sizeM, D_d, sizeM, sizeM, 1, cudaMemcpyDefault));

    /*************************/


    double transferedBytes = sizeM;
    transferedBytes += ((float)alpha != 0.f) ? sizeM : 0;
    transferedBytes += ((float)gamma != 0.f) ? sizeM : 0;
    transferedBytes /= 1e9;
    printf("cuTensor: %.2f GB/s\n", transferedBytes / time);
    printf("memcpy: %.2f GB/s\n", 2 * sizeM / minTimeMEMCPY / 1e9 );

    if (A_d) cudaFree(A_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);

    return 0;
}




}
