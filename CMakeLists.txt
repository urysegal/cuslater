cmake_minimum_required(VERSION 3.21)
project(cuSlater CUDA)

set(CMAKE_CUDA_STANDARD 11)
include_directories(${cuSlater_SOURCE_DIR} ${cuSlater_SOURCE_DIR}/include)
add_library(cuSlater cuslater.cu src/gputensors.cu src/tensors.cu include/grids.h)

set_target_properties(cuSlater PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
