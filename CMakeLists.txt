cmake_minimum_required(VERSION 3.21)
project(cuSlater CUDA CXX)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)      # <---- Changing to 17 does not give an error
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

#Precision Flag
set(PRECISION PRECISION_FLOAT)

# Set compiler flags for debug mode
include_directories(${cuSlater_SOURCE_DIR} ${cuSlater_SOURCE_DIR}/include)
add_library(cuSlater SHARED
		include/utilities.h src/utilities.cu
		include/grids.h		src/grids.cu
		include/evalIntegral.h src/evalIntegral.cu
)

set_property(TARGET cuSlater PROPERTY CUDA_ARCHITECTURES 70)

target_compile_options(cuSlater PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Werror -O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_70,code=sm_70>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math -prec-sqrt=false -ftz=true -prec-div=false -fmad=true>
)
target_compile_definitions(cuSlater PRIVATE ${PRECISION})
target_include_directories(cuSlater PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(cuSlater PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuSlater PROPERTIES CUDA_ARCHITECTURE 70-real)

add_executable(simple examples/simple.cu)

add_dependencies(simple cuSlater)

target_compile_definitions(simple PRIVATE ${PRECISION})
target_link_libraries(simple cuSlater)

