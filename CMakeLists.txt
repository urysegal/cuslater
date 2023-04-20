cmake_minimum_required(VERSION 3.21)
project(cuSlater CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_library(cuSlater cuslater.cu src/gputensors.cu src/tensors.cu)

set_target_properties(cuSlater PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
