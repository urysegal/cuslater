cmake_minimum_required(VERSION 3.21)
project(cuSlater CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=SM_70")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo  --tool memcheck -arch=SM_70")
set(COMPILE_FLAGS -g -O4 -Wall -Xcompiler -Werror)

#Precision Flag
set(PRECISION PRECISION_DOUBLE)

# Set compiler flags for debug mode
set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -lineinfo -G -O0 -arch=SM_70")

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
add_library(cuSlater SHARED
		include/utilities.h src/utilities.cu
		include/grids.h		src/grids.cu
		include/evalIntegral.h src/evalIntegral.cu
)

set_property(TARGET cuSlater PROPERTY CUDA_ARCHITECTURES 70)

target_link_libraries(cuSlater ${CUBLASLT_LIBRARY} ${CUTENSOR_LIBRARY} cuda)
target_compile_options(cuSlater PUBLIC ${COMPILE_FLAGS} $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_70,code=sm_70>)
target_compile_definitions(cuSlater PRIVATE ${PRECISION})
target_include_directories(cuSlater PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(cuSlater PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuSlater PROPERTIES CUDA_ARCHITECTURE 70-real)

add_executable(simple examples/simple.cu)

add_dependencies(simple cuSlater)

target_compile_definitions(simple PRIVATE ${PRECISION})
target_link_libraries(simple cuSlater)

