#!/bin/bash
module load cuda
module load gcc
module load cmake
module load apptainer

export CUDA_ROOT=/arc/software/spack-2024/opt/spack/linux-rocky9-skylake_avx512/gcc-9.4.0/cuda-11.3.1-pwzx2bw72sresgc76i7fv54qvt3xwrxf
export LD_LIBRARY_PATH=/arc/software/spack-2024/opt/spack/linux-rocky9-skylake_avx512/gcc-9.4.0/cuda-11.3.1-pwzx2bw72sresgc76i7fv54qvt3xwrxf/lib64/:${LD_LIBRARY_PATH}
