cmake_minimum_required(VERSION 3.21)
project(cuSlater CUDA)


set(CMAKE_CUDA_STANDARD 14)

set(COMPILE_FLAGS -g -Wall -Xcompiler -Werror)

include_directories(${cuSlater_SOURCE_DIR} ${cuSlater_SOURCE_DIR}/include)
add_library(cuSlater src/gputensors.cu
        include/gputensors.h src/svalues.cu src/stovalues.cu)
target_link_libraries(cuSlater cutensor cublasLt)
target_compile_options(cuSlater PUBLIC ${COMPILE_FLAGS})
target_include_directories(cuSlater PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(cuSlater PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(hadamar examples/test-hadamar.cu)
add_dependencies(hadamar cuSlater)
target_link_libraries(hadamar cuSlater)