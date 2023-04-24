cmake_minimum_required(VERSION 3.21)
project(cuSlater CUDA)


set(CMAKE_CUDA_STANDARD 14)

set(COMPILE_FLAGS -g -Wall -Xcompiler -Werror)

include_directories(${cuSlater_SOURCE_DIR} ${cuSlater_SOURCE_DIR}/include)
add_library(cuSlater stocalculator.cu src/gputensors.cu src/tensors.cu include/grids.h include/sto.h include/tensors.cuh
        include/gputensors.h stocalculator.h src/svalues.cu)
target_link_libraries(cuSlater cutensor cublasLt)
target_compile_options(cuSlater PUBLIC ${COMPILE_FLAGS})
target_include_directories(cuSlater PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(cuSlater PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(test examples/test.cu stocalculator.h)
add_dependencies(test cuSlater)
target_include_directories(test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test cuSlater cublasLt)
target_compile_options(test PUBLIC ${COMPILE_FLAGS})
