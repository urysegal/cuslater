cmake_minimum_required(VERSION 3.21)
project(cuSlater CUDA)

set(CMAKE_CUDA_STANDARD 14)

set(COMPILE_FLAGS -g -Wall -Xcompiler -Werror)

# Set compiler flags for debug mode
set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -G -O0")

set(CUDA_LIB_DIR /arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/lib64/)
find_library(CUBLASLT_LIBRARY
	NAMES cublasLt
	HINTS ${CUDA_LIB_DIR}
)

set(CUTENSOR_LIB_DIR /arc/project/st-greif-1/sfw/libcutensor-linux-x86_64-1.6.2.3-archive/)
find_library(CUTENSOR_LIBRARY
	NAMES cutensor
	HINTS ${CUTENSOR_LIB_DIR}
	PATH_SUFFIXES lib/11.0
)
include_directories(${cuSlater_SOURCE_DIR} ${cuSlater_SOURCE_DIR}/include ${CUTENSOR_LIB_DIR}/include 
)
add_library(cuSlater SHARED src/gputensors.cu
        include/gputensors.h src/svalues.cu src/stovalues.cu
		src/evalInnerIntegral.cu)


target_link_libraries(cuSlater ${CUBLASLT_LIBRARY} ${CUTENSOR_LIBRARY} cuda)
target_compile_options(cuSlater PUBLIC ${COMPILE_FLAGS})
target_include_directories(cuSlater PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(cuSlater PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(hadamard examples/test-hadamard.cu)
add_executable(evaluate examples/evalFunction.cu)
add_dependencies(hadamard cuSlater)
add_dependencies(evaluate cuSlater)
target_link_libraries(hadamard cuSlater)
target_link_libraries(evaluate cuSlater)
