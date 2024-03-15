#!/bin/bash
#SBATCH --job-name=fourc-enter           
#SBATCH --account=st-greif-1-gpu    
#SBATCH --nodes=1                  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                           
#SBATCH --mem=64G                  
#SBATCH --time=1:00:00             
#SBATCH --gpus-per-node=1
#SBATCH --output=job_outputs/output-%j.txt         
#SBATCH --error=job_outputs/error-%j.txt          
#SBATCH --mail-user=gkluhana@cs.ubc.ca
#SBATCH --mail-type=ALL     

module load gcc
module load openmpi
module load cuda
module load intel-mkl
module load apptainer
module load git
module load cmake

export CUDA_ROOT=/arc/software/spack-2021/spack/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-56n7klho37iu7zbc7aslccxr5akscmrf
export LD_LIBRARY_PATH=/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/lib64/:${LD_LIBRARY_PATH}
apptainer exec --nv julia_latest.sif  ./run_julia.sh
