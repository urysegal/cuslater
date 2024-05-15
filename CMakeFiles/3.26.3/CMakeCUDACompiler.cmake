set(CMAKE_CUDA_COMPILER "/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.2.0/gcc-9.4.0-7ymwiib6wuklec5ju7m3ktapgtbzhxjl/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "11.3.109")
set(CMAKE_CUDA_DEVICE_LINKER "/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "14")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "9.4")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop")
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "11.3.109")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop")

set(CMAKE_CUDA_ARCHITECTURES_ALL "35-real;37-real;50-real;52-real;53-real;60-real;61-real;62-real;70-real;72-real;75-real;80-real;86")
set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "35-real;50-real;60-real;70-real;80")
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/targets/x86_64-linux/lib/stubs;/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/intel-mkl-2020.4.304-got6o5exzqzqlteqiai6ykwg6vmadyhc/compilers_and_libraries_2020.4.304/linux/mkl/include;/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.2.0/gcc-9.4.0-7ymwiib6wuklec5ju7m3ktapgtbzhxjl/include/c++/9.4.0;/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.2.0/gcc-9.4.0-7ymwiib6wuklec5ju7m3ktapgtbzhxjl/include/c++/9.4.0/x86_64-pc-linux-gnu;/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.2.0/gcc-9.4.0-7ymwiib6wuklec5ju7m3ktapgtbzhxjl/include/c++/9.4.0/backward;/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.2.0/gcc-9.4.0-7ymwiib6wuklec5ju7m3ktapgtbzhxjl/lib/gcc/x86_64-pc-linux-gnu/9.4.0/include;/usr/local/include;/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.2.0/gcc-9.4.0-7ymwiib6wuklec5ju7m3ktapgtbzhxjl/include;/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.2.0/gcc-9.4.0-7ymwiib6wuklec5ju7m3ktapgtbzhxjl/lib/gcc/x86_64-pc-linux-gnu/9.4.0/include-fixed;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/targets/x86_64-linux/lib/stubs;/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/targets/x86_64-linux/lib;/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.2.0/gcc-9.4.0-7ymwiib6wuklec5ju7m3ktapgtbzhxjl/lib/gcc/x86_64-pc-linux-gnu/9.4.0;/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.2.0/gcc-9.4.0-7ymwiib6wuklec5ju7m3ktapgtbzhxjl/lib64;/lib64;/usr/lib64;/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/intel-mkl-2020.4.304-got6o5exzqzqlteqiai6ykwg6vmadyhc/compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin;/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/intel-mkl-2020.4.304-got6o5exzqzqlteqiai6ykwg6vmadyhc/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64_lin;/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.2.0/gcc-9.4.0-7ymwiib6wuklec5ju7m3ktapgtbzhxjl/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_MT "")
