cmake_minimum_required(VERSION 3.21)
project(cuSlater CUDA)

set(CMAKE_CUDA_STANDARD 11)

set(COMPILE_FLAGS -g -Wall -Xcompiler -Werror)
# Set the desired CUDA version
set(CMAKE_CUDA_COMPILER_VERSION 12.0)

# Specify the location of the CUDA toolkit
find_package(CUDAToolkit 12.0 REQUIRED)


include_directories(${cuSlater_SOURCE_DIR} ${cuSlater_SOURCE_DIR}/include )

add_library(cuSlater SHARED src/gputensors.cu src/svalues.cu src/stovalues.cu src/evalInnerIntegral.cu)

target_link_libraries(cuSlater  cutensor cuda)
target_compile_options(cuSlater PUBLIC ${COMPILE_FLAGS})
target_include_directories(cuSlater PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(cuSlater PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(hadamard examples/test-hadamard.cu)
add_dependencies(hadamard cuSlater)
target_link_libraries(hadamard cuSlater)
