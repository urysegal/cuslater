# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cmake-3.26.3-jymy3coxptnuxi2nos4tpenemitu7hfr/bin/cmake

# The command to remove a file.
RM = /arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cmake-3.26.3-jymy3coxptnuxi2nos4tpenemitu7hfr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /project/st-greif-1/mewert/cuslater

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /project/st-greif-1/mewert/cuslater/build

# Include any dependencies generated for this target.
include CMakeFiles/legendre.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/legendre.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/legendre.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/legendre.dir/flags.make

CMakeFiles/legendre.dir/examples/test-legendre.cu.o: CMakeFiles/legendre.dir/flags.make
CMakeFiles/legendre.dir/examples/test-legendre.cu.o: CMakeFiles/legendre.dir/includes_CUDA.rsp
CMakeFiles/legendre.dir/examples/test-legendre.cu.o: /project/st-greif-1/mewert/cuslater/examples/test-legendre.cu
CMakeFiles/legendre.dir/examples/test-legendre.cu.o: CMakeFiles/legendre.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/project/st-greif-1/mewert/cuslater/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/legendre.dir/examples/test-legendre.cu.o"
	/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/legendre.dir/examples/test-legendre.cu.o -MF CMakeFiles/legendre.dir/examples/test-legendre.cu.o.d -x cu -c /project/st-greif-1/mewert/cuslater/examples/test-legendre.cu -o CMakeFiles/legendre.dir/examples/test-legendre.cu.o

CMakeFiles/legendre.dir/examples/test-legendre.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/legendre.dir/examples/test-legendre.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/legendre.dir/examples/test-legendre.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/legendre.dir/examples/test-legendre.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target legendre
legendre_OBJECTS = \
"CMakeFiles/legendre.dir/examples/test-legendre.cu.o"

# External object files for target legendre
legendre_EXTERNAL_OBJECTS =

legendre: CMakeFiles/legendre.dir/examples/test-legendre.cu.o
legendre: CMakeFiles/legendre.dir/build.make
legendre: libcuSlater.so
legendre: /arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/lib64/libcublasLt.so
legendre: /arc/project/st-greif-1/sfw/libcutensor-linux-x86_64-1.6.2.3-archive/lib/11.0/libcutensor.so
legendre: CMakeFiles/legendre.dir/linkLibs.rsp
legendre: CMakeFiles/legendre.dir/objects1.rsp
legendre: CMakeFiles/legendre.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/project/st-greif-1/mewert/cuslater/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable legendre"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/legendre.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/legendre.dir/build: legendre
.PHONY : CMakeFiles/legendre.dir/build

CMakeFiles/legendre.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/legendre.dir/cmake_clean.cmake
.PHONY : CMakeFiles/legendre.dir/clean

CMakeFiles/legendre.dir/depend:
	cd /project/st-greif-1/mewert/cuslater/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /project/st-greif-1/mewert/cuslater /project/st-greif-1/mewert/cuslater /project/st-greif-1/mewert/cuslater/build /project/st-greif-1/mewert/cuslater/build /project/st-greif-1/mewert/cuslater/build/CMakeFiles/legendre.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/legendre.dir/depend

