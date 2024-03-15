#!/bin/bash
export LD_LIBRARY_PATH=/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/lib64/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/scratch/st-greif-1/gkluhana/lib64/:$LD_LIBRARY_PATH

nvidia-smi
julia lior_sum.jl


