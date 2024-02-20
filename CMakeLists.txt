cmake_minimum_required(VERSION 3.21)
project(cuSlater CUDA)

<<<<<<< HEAD
set(CMAKE_CUDA_STANDARD 14)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo --tool memcheck")
=======
set(CMAKE_CUDA_STANDARD 11)

>>>>>>> 0a886f4 (gputesnors compiles correctly)
set(COMPILE_FLAGS -g -Wall -Xcompiler -Werror)
# Set the desired CUDA version
set(CMAKE_CUDA_COMPILER_VERSION 12.0)

<<<<<<< HEAD
# Set compiler flags for debug mode
set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -lineinfo -G -O0")

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
		src/gputensors.cu include/gputensors.h
		src/svalues.cu src/stovalues.cu
		include/utilities.h src/utilities.cu
		include/grids.h		src/grids.cu
		include/evalInnerIntegral.h src/evalInnerIntegral.cu
		include/evalInnerIntegralStreams.h src/evalInnerIntegralStreams.cu
		include/evalIntegral.h src/evalIntegral.cu
)


target_link_libraries(cuSlater ${CUBLASLT_LIBRARY} ${CUTENSOR_LIBRARY} cuda)
include_directories(${cuSlater_SOURCE_DIR} ${cuSlater_SOURCE_DIR}/include)
add_library(cuSlater SHARED src/gputensors.cu
        include/gputensors.h src/svalues.cu src/stovalues.cu
		src/evalInnerIntegral.cu)
target_link_libraries(cuSlater cutensor cublasLt)
=======
# Specify the location of the CUDA toolkit
find_package(CUDAToolkit 12.0 REQUIRED)


include_directories(${cuSlater_SOURCE_DIR} ${cuSlater_SOURCE_DIR}/include )

add_library(cuSlater SHARED src/gputensors.cu src/svalues.cu src/stovalues.cu src/evalInnerIntegral.cu)

target_link_libraries(cuSlater  cutensor cuda)
>>>>>>> 0a886f4 (gputesnors compiles correctly)
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

