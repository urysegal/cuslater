cmake_minimum_required(VERSION 3.21)
project(cuSlater CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo --tool memcheck -gencode=arch=compute_70,code=sm_70")
set(COMPILE_FLAGS -g -Wall -Xcompiler -Werror)

# Set the desired CUDA version
set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -lineinfo -gencode arch=compute_70,code=sm_70 -G -O0")


set(CMAKE_CUDA_COMPILER_VERSION 12.0)

# Specify the location of the CUDA toolkit
find_package(CUDAToolkit 12.0 REQUIRED)


include_directories(${cuSlater_SOURCE_DIR} ${cuSlater_SOURCE_DIR}/include )

add_library(cuSlater SHARED 
                src/gputensors.cu include/gputensors.h
                src/svalues.cu src/stovalues.cu
                include/utilities.h src/utilities.cu
                include/grids.h         src/grids.cu
                include/evalInnerIntegral.h src/evalInnerIntegral.cu
                include/evalInnerIntegralStreams.h src/evalInnerIntegralStreams.cu
                include/evalIntegral.h src/evalIntegral.cu
		)
target_link_libraries(cuSlater  cutensor cuda)
target_compile_options(cuSlater PUBLIC ${COMPILE_FLAGS})
target_include_directories(cuSlater PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(cuSlater PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(hadamard examples/test-hadamard.cu)
add_executable(evaluate examples/test-evalInner.cu)
add_executable(legendre examples/test-legendre.cu)
add_executable(testgrids examples/test-grids.cu)
add_executable(simple examples/simple.cu)

add_dependencies(hadamard cuSlater)
add_dependencies(evaluate cuSlater)
add_dependencies(legendre cuSlater)
add_dependencies(testgrids cuSlater)
add_dependencies(simple cuSlater)

target_link_libraries(hadamard cuSlater)
target_link_libraries(evaluate cuSlater)
target_link_libraries(legendre cuSlater)
target_link_libraries(testgrids cuSlater)
target_link_libraries(simple cuSlater)

