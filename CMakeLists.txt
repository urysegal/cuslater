cmake_minimum_required(VERSION 3.30)
project(cuSlater CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)


#Precision Flag
set(PRECISION PRECISION_DOUBLE)

# grab all kernel files
file(GLOB CUSLATER_KERNEL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cu")
# Set compiler flags for debug mode
# include_directories(${cuSlater_SOURCE_DIR} ${cuSlater_SOURCE_DIR}/include)
add_library(cuSlater SHARED
		include/utilities.h src/utilities.cu
		include/grids.h		src/grids.cu src/metric.cu src/normalizer.cu
		include/slot.cuh
		src/kernels/kernels.cuh
		${CUSLATER_KERNEL_SOURCES}
		include/evalIntegral.cuh src/evalIntegral.cu
		src/number.cu src/domain.cu
)
target_include_directories(cuSlater PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_compile_options(cuSlater PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Werror -O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_70,code=sm_70>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -v>
        $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets> # suppress warnings for older architectures
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math -prec-sqrt=false -ftz=true -prec-div=false -fmad=true>
        # $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -g>
)
target_compile_definitions(cuSlater PRIVATE ${PRECISION})
target_include_directories(cuSlater PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(cuSlater PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(simple examples/simple.cu)
add_executable(conver examples/trap.cu)
add_executable(sqrt_test sqrt.cu)
add_executable(expo_test exp.cu)

add_dependencies(simple cuSlater)
add_dependencies(conver cuSlater)

target_compile_definitions(simple PRIVATE ${PRECISION})
target_link_libraries(simple cuSlater)

target_compile_definitions(conver PRIVATE ${PRECISION})
target_link_libraries(conver cuSlater)

# say the architecture
get_property(ARCHITECTURES_VALUE TARGET cuSlater PROPERTY CUDA_ARCHITECTURES)
message(STATUS "CUDA Architectures: ${ARCHITECTURES_VALUE}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
if(PRECISION_DOUBLE)
  message(STATUS "real_t is set to double.")
else()
  message(STATUS "real_t is set to float.")
endif()

